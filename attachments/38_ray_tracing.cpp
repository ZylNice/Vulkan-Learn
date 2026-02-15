// #pragma warning(disable : 26813)  // 屏蔽 C26813 警告: "使用‘按位与’来检查标志是否设置"

#include <algorithm>
#include <chrono>        // 用于获取时间
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // 导入 glfwCreateWindowSurface 函数（条件编译 glfw3.h）
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS                  // 强制 GLM 使用弧度制（Vulkan 和 GLM 推荐）
#define GLM_FORCE_DEPTH_ZERO_TO_ONE        // 强制 GLM 生成的透视投影矩阵将深度值（Z轴）映射到 [0, 1] 范围
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>        //引入 GLM 的哈希扩展，用于计算 glm::vec3 等类型的哈希值

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#ifndef LAB_TASK_LEVEL
#	define LAB_TASK_LEVEL 11
#endif

#define LAB_TASK_AS_BUILD_AND_BIND 4        // 绑定和构建加速结构
#define LAB_TASK_AS_ANIMATION 6             // 加速结构动画更新
#define LAB_TASK_AS_OPAQUE_FLAG 7           // 处理不透明标记
#define LAB_TASK_INSTANCE_LUT 9             // 实例查找表
#define LAB_TASK_REFLECTIONS 11             // 反射效果

const uint32_t    WIDTH                = 800;
const uint32_t    HEIGHT               = 600;
const std::string MODEL_PATH           = "models/plant_on_table.obj";
constexpr int     MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;        // 发布时关闭验证层，保证性能
#else
constexpr bool enableValidationLayers = true;
#endif

// 顶点
struct Vertex
{
	glm::vec3                                pos;
	glm::vec3                                color;
	glm::vec2                                texCoord;
	glm::vec3                                normal;
	static vk::VertexInputBindingDescription getBindingDescription()        // 顶点输入绑定点
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};        // 绑定点，顶点步长（字节），更新频率（顶点/实例）
	}

	static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()        // 属性描述（如何读取一个顶点中的具体属性）
	{
		return {
		    vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),          // 位置（location，顶点输入绑定点，格式，大小）
		    vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),        // 颜色
		    vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)),        // UV
		    vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal))        // 法线
		};
	}
	bool operator==(const Vertex &other) const
	{
		return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
	}
};

// 顶点去重
template <>
struct std::hash<Vertex>
{
	size_t operator()(Vertex const &vertex) const noexcept
	{
		auto h = hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1);
		h      = (h >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		h      = (h >> 1) ^ (hash<glm::vec3>()(vertex.normal) << 1);
		return h;
	}
};

// UBO
struct UniformBufferObject
{
	alignas(16) glm::mat4 model;            // 模型矩阵
	alignas(16) glm::mat4 view;             // 视图矩阵
	alignas(16) glm::mat4 proj;             // 投影矩阵
	alignas(16) glm::vec3 cameraPos;        // 相机位置
};

// 推送常量
struct PushConstant
{
	uint32_t materialIndex;        // 当前绘制物体的材质索引
#if LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
	uint32_t reflective;        // 反射标记
#endif
};

class HelloTriangleApplication
{
  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	GLFWwindow *window = nullptr;

	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR             surface        = nullptr;        // 操作系统窗口在 Vulkan 中的抽象表示

	vk::raii::PhysicalDevice physicalDevice = nullptr;        // 物理设备（显卡）
	vk::raii::Device         device         = nullptr;        // 逻辑设备

	vk::raii::Queue graphicsQueue = nullptr;        // 图形队列（Vulkan 规定，图形/计算队列，必须支持传输（Transfer）操作）
	vk::raii::Queue presentQueue  = nullptr;        // 呈现队列

	// 交换链
	vk::raii::SwapchainKHR           swapChain = nullptr;
	std::vector<vk::Image>           swapChainImages;               // 图像
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;        // 图像格式
	vk::Extent2D                     swapChainExtent;               // 图像尺寸
	std::vector<vk::raii::ImageView> swapChainImageViews;           // 图像视图（图形管线通过它，访问交换链图像）

	// 管线
	vk::raii::DescriptorSetLayout descriptorSetLayoutGlobal   = nullptr;        // 通用接口，所有物体存入的相机数据相同
	vk::raii::DescriptorSetLayout descriptorSetLayoutMaterial = nullptr;        // 材质接口，不同物体存入的材质数据不同
	vk::raii::PipelineLayout      pipelineLayout              = nullptr;        // 管线布局
	vk::raii::Pipeline            graphicsPipeline            = nullptr;        // 图形管线

	// vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;        // 硬件支持的最大 MSAA 采样数
	// vk::raii::Image        colorImage       = nullptr;        // MSAA 使用的颜色缓冲区
	// vk::raii::DeviceMemory colorImageMemory = nullptr;
	// vk::raii::ImageView    colorImageView   = nullptr;
	//   uint32_t mipLevels = 0;        // 存储根据纹理尺寸计算出的 Mipmap 层级总数

	// 深度缓冲
	vk::raii::Image        depthImage       = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView    depthImageView   = nullptr;

	// 纹理资源
	std::vector<vk::raii::Image>        textureImages;                   // 纹理图像句柄（Vulkan 规定，对于 Image，GPU 一定可见，CPU 一般不可见）（仅当图像为线性平铺时，CPU 才可见）
	std::vector<vk::raii::DeviceMemory> textureImageMemories;            // 内存（图像）
	std::vector<vk::raii::ImageView>    textureImageViews;               // 图像视图
	vk::raii::Sampler                   textureSampler = nullptr;        // 采样器

	// 几何缓冲区（仅顶点阶段）
	std::vector<Vertex>    vertices;
	std::vector<uint32_t>  indices;
	vk::raii::Buffer       vertexBuffer       = nullptr;        // 顶点缓冲区句柄（Vulkan 规定，对于 Buffer，GPU 一定可见，CPU 不一定可见）
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;        // 内存（顶点缓冲区）
	vk::raii::Buffer       indexBuffer        = nullptr;        // 索引缓冲区句柄
	vk::raii::DeviceMemory indexBufferMemory  = nullptr;        // 内存（索引缓冲区）
	vk::raii::Buffer       uvBuffer           = nullptr;        // UV 缓冲区句柄
	vk::raii::DeviceMemory uvBufferMemory     = nullptr;        // 内存（UV 缓冲区）

	// BLAS
	std::vector<vk::raii::Buffer>                   blasBuffers;        // （底层加速结构）（每个模型一个，所以用 vector 存储）
	std::vector<vk::raii::DeviceMemory>             blasMemories;
	std::vector<vk::raii::AccelerationStructureKHR> blasHandles;        // 句柄，类似于图像视图

	// 实例数据
	std::vector<vk::AccelerationStructureInstanceKHR> instances;
	vk::raii::Buffer                                  instanceBuffer = nullptr;
	vk::raii::DeviceMemory                            instanceMemory = nullptr;

	// TLAS
	vk::raii::Buffer                   tlasBuffer        = nullptr;        // （顶层加速结构）（对应整个场景，所以只有一个）
	vk::raii::DeviceMemory             tlasMemory        = nullptr;
	vk::raii::Buffer                   tlasScratchBuffer = nullptr;        // TLAS 构建时的临时缓冲区
	vk::raii::DeviceMemory             tlasScratchMemory = nullptr;
	vk::raii::AccelerationStructureKHR tlas              = nullptr;        // TLAS 句柄

	// 实例查找表
	struct InstanceLUT        // （用于在 Shader 中根据实例 ID 找到材质 ID）
	{
		uint32_t materialID;
		uint32_t indexBufferOffset;
	};
	std::vector<InstanceLUT> instanceLUTs;
	vk::raii::Buffer         instanceLUTBuffer       = nullptr;
	vk::raii::DeviceMemory   instanceLUTBufferMemory = nullptr;

	// UBO
	UniformBufferObject                 ubo{};
	std::vector<vk::raii::Buffer>       uniformBuffers;              // 句柄
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;        // 内存
	std::vector<void *>                 uniformBuffersMapped;        // 持久映射指针

	// 子网格
	struct SubMesh        // （用于区分模型中的不同材质部分）
	{
		uint32_t indexOffset;        // 子网格在全局索引缓冲区的起始位置
		uint32_t indexCount;         // 索引总数
		int      materialID;         // 材质索引
		uint32_t firstVertex;        // 最小顶点索引值
		uint32_t maxVertex;          // 最大顶点索引值
		bool     alphaCut;           // 是否 alpha 裁剪
		bool     reflective;         // 是否反射
	};
	std::vector<SubMesh>             submeshes;        // 子网格列表
	std::vector<tinyobj::material_t> materials;        // 材质列表

	// 描述符池和集合
	vk::raii::DescriptorPool             descriptorPool = nullptr;        // 描述符池
	std::vector<vk::raii::DescriptorSet> globalDescriptorSets;            // 全局数据
	std::vector<vk::raii::DescriptorSet> materialDescriptorSets;          // 材质数据

	// 命令池
	vk::raii::CommandPool                commandPool = nullptr;        // 命令池（分配命令缓冲）
	std::vector<vk::raii::CommandBuffer> commandBuffers;               // 命令缓冲（录制命令）
	uint32_t                             graphicsIndex = ~0;           // 图形队列族索引

	// 同步对象
	std::vector<vk::raii::Semaphore> presentCompleteSemphores;        // 图像可用（信号）
	std::vector<vk::raii::Semaphore> renderFinishedSemphores;         // 渲染完成（信号）
	std::vector<vk::raii::Fence>     inFlightFences;                  // 上帧 GPU 工作完成（栅栏）
	uint32_t                         frameIndex = 0;                  // 当前帧索引（0 或 1）

	bool framebufferResized = false;        // 窗口大小是否改变

	// 设备扩展（一个拓展往往包含多个特性，两者粒度不同）
	std::vector<const char *> requiredDeviceExtension = {
	    vk::KHRSwapchainExtensionName,
	    vk::KHRSpirv14ExtensionName,
	    vk::KHRSynchronization2ExtensionName,
	    vk::KHRCreateRenderpass2ExtensionName,
	    vk::KHRAccelerationStructureExtensionName,        // 加速结构
	    vk::KHRBufferDeviceAddressExtensionName,          // 缓冲区设备地址（光追必需）
	    vk::KHRDeferredHostOperationsExtensionName,
	    vk::KHRRayQueryExtensionName        // Ray Query（内联光追）
	};

	void initWindow()
	{
		glfwInit();                                                                  // 初始化 glfw 库
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);                                // 禁用 OpenGL 上下文
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);        // 创建窗口 (宽, 高, 标题, 显示器, 共享资源)
		glfwSetWindowUserPointer(window, this);                                      // 将类实例指针传入 window，供回调函数使用
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// 窗口大小改变的回调
	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto app                = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));        // 从 window 中取出当前类对象指针
		app->framebufferResized = true;
	}

	// Vulkan 初始化
	void initVulkan()
	{
		// 基础环境
		createInstance();             // 实例
		setupDebugMessenger();        // 设置调试回调
		createSurface();              // 窗口（Vulkan 抽象）
		pickPhysicalDevice();         // 物理设备
		createLogicalDevice();        // 逻辑设备

		createSwapChain();         // 交换链
		createImageViews();        // 创建图像视图

		createCommandPool();        // 命令池
		loadModel();                // 加载模型

		createDescriptorSetLayout();        // 描述符集布局
		createGraphicsPipeline();           // 图形管线

		createDepthResources();        // 深度缓冲
		createTextureSampler();        // 纹理采样器

		// createTextureImage();
		// createTextureImageView();

		createVertexBuffer();        // 顶点缓冲
		createIndexBuffer();         // 索引缓冲
		createUVBuffer();            // UV 缓冲

		createAccelerationStructures();        // 构建加速结构（BLAS、TLAS）
		createInstanceLUTBuffer();             // 实例查找表

		createUniformBuffers();        // UBO 缓冲
		createDescriptorPool();        // 描述符池
		createDescriptorSets();        // 描述符集
		createCommandBuffer();         // 命令缓冲

		createSyncObjects();        // 同步对象
	}

	// 主循环
	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();        // 处理上一帧积压的输入（窗口回调在此触发）（操作系统用事件队列保存上一帧积压的输入事件）
			drawFrame();
		}
		device.waitIdle();        // 避免在 GPU 结束工作前关闭窗口，释放显存资源，导致 GPU 非法访问释放的资源，进而驱动崩溃
	}

	// 销毁交换链
	void cleanupSwapChain()
	{
		swapChainImageViews.clear();        // 清空旧的 imageView
		swapChain = nullptr;                // 通过 RAII 销毁旧的交换链
	}

	// 销毁窗口
	void cleanup()
	{
		glfwDestroyWindow(window);        // 销毁窗口
		glfwTerminate();                  // 清理 glfw 资源
	}

	// 重建交换链（窗口大小改变时调用）
	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)        // 如果是窗口最小化，则进入循环（创建交换链时，图片尺寸不允许为 0）
		{
			glfwWaitEvents();        // 线程休眠，等待事件触发
			glfwGetFramebufferSize(window, &width, &height);
		}

		device.waitIdle();        // 确保清除交换链前，GPU 已经不再使用交换链中的图片

		cleanupSwapChain();
		createSwapChain();
		createImageViews();
		createDepthResources();
	}

	// 创建 Vulkan 实例
	void createInstance()
	{
		constexpr vk::ApplicationInfo appInfo{
		    .pApplicationName   = "Hello Triangle",
		    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		    .pEngineName        = "No Engine",
		    .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		    .apiVersion         = vk::ApiVersion14};

		// 启用验证层
		std::vector<char const *> requiredLayers;
		if (enableValidationLayers)
		{
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		auto layerProperties = context.enumerateInstanceLayerProperties();        // 查询支持的 Vulkan 验证层（操作系统层面）（Vulkan Loader 通过注册表找）
		for (auto const &requiredLayer : requiredLayers)
		{
			if (std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
			{
				throw std::runtime_error("Required layer not supported:" + std::string(requiredLayer));
			}
		}

		auto requiredExtensions = getRequiredExtensions();

		auto extensionProperties = context.enumerateInstanceExtensionProperties();        // 查询支持的 Vulkan 实例拓展（操作系统层面）
		for (auto const &requiredExtension : requiredExtensions)
		{
			if (std::ranges::none_of(extensionProperties, [requiredExtension](auto const &extensionPropertie) { return strcmp(extensionPropertie.extensionName, requiredExtension) == 0; }))
			{
				throw std::runtime_error("Required extension not supported:" + std::string(requiredExtension));
			}
		}

		vk::InstanceCreateInfo createInfo{
		    .pApplicationInfo        = &appInfo,
		    .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
		    .ppEnabledLayerNames     = requiredLayers.data(),
		    .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
		    .ppEnabledExtensionNames = requiredExtensions.data()};
		instance = vk::raii::Instance(context, createInfo);
	}

	// 设置调试回调
	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;
		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		                                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		                                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);        // 哪些严重等级的消息是需要的
		vk::DebugUtilsMessageTypeFlagsEXT     messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);        // 哪些类型的消息是需要的
		vk::DebugUtilsMessengerCreateInfoEXT  debugUtilsMessengerCreateInfoEXT{
		     .messageSeverity = severityFlags,
		     .messageType     = messageTypeFlags,
		     .pfnUserCallback = &debugCallback};
		debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
	}

	// 创建窗口（Vulkan 抽象）
	void createSurface()
	{
		VkSurfaceKHR _surface;

		auto result = glfwCreateWindowSurface(*instance, window, nullptr, &_surface);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	// 选择物理设备
	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();        // 获取实例（操作系统）上的物理设备

		const auto devIter = std::ranges::find_if(
		    devices,
		    [&](auto const &device) {        // 用 auto 作为 lambda 参数类型时，相当于用模板实现一个泛型 lambda
			    // 是否支持 Vulkan 1.3
			    bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

			    // 是否支持图形队列
			    auto queueFamilies    = device.getQueueFamilyProperties();        // 获取物理设备上的队列族
			    bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp) {
				    return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
			    });

			    // 是否支持所需设备拓展
			    auto availExts                     = device.enumerateDeviceExtensionProperties();                                           // 获取显卡支持的所有设备拓展
			    bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availExts](auto const &reqExt) {        // 检查显卡是否支持所有需要的设备拓展
				    return std::ranges::any_of(availExts, [reqExt](auto const &availExt) {
					    return strcmp(availExt.extensionName, reqExt) == 0;
				    });
			    });

			    // 是否支持所需 Vulkan 特性
			    auto features = device.template getFeatures2<
			        vk::PhysicalDeviceFeatures2,                               // Vulkan 1.0（Vulkan 规定，必须先查询这个）
			        vk::PhysicalDeviceVulkan12Features,                        // Vulkan 1.2
			        vk::PhysicalDeviceVulkan13Features,                        // Vulkan 1.3
			        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,         // 动态状态（动态修改管线配置，无需重新编译管线）
			        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,        // 加速结构
			        vk::PhysicalDeviceRayQueryFeaturesKHR                      // 射线查询
			        >();
			    bool supporsRequiredFeatures =
			        features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&                        // 各向异性过滤
			        features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&                           // 动态渲染（淘汰 RenderPass 对象）
			        features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&        // 动态状态（复用 Pipeline 对象）
			        // 无绑定渲染（Bindless）
			        features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind &&        // 绑定后更新（允许描述符集在绑定到命令缓冲后，还能更新它引用的图片）
			        features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingPartiallyBound &&                     // 部分绑定（允许描述符集的某些槽位是空的，但需确保 Shader 不去读）
			        features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount &&            // 可变描述符数量（允许描述符集最后一个绑定的数组大小是动态的）
			        features.template get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray &&                              // 运行时描述符数量（允许着色器使用运行时计算的索引来访问描述符数组）
			        features.template get<vk::PhysicalDeviceVulkan12Features>().shaderSampledImageArrayNonUniformIndexing &&           // 非均匀索引（告诉驱动，同一个 Warp 中的不同线程可能会访问纹理数组中的不同元素）
			        // 光线追踪
			        features.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress &&                          // 缓冲区设备地址（允许直接访问 Buffer）
			        features.template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure &&        // 加速结构
			        features.template get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;                                    // 光线查询

			    // 结果汇总
			    return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supporsRequiredFeatures;
		    });

		if (devIter != devices.end())
		{
			physicalDevice = *devIter;
		}
		else
		{
			throw std::runtime_error("failed to find a suitable GPU");
		}
	}

	// 创建逻辑设备（当图形和呈现是不同队列时，有逻辑问题）
	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](auto const &qfp) {
			return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
		});

		graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));        // 图像队列族索引

		auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface) ? graphicsIndex : ~0;

		if (presentIndex == queueFamilyProperties.size())
		{
			// 查找支持图形和显示的队列族
			for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
			{
				if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
				    physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
				{
					graphicsIndex = static_cast<uint32_t>(i);
					presentIndex  = graphicsIndex;
					break;
				}
			}

			// 查找支持显示的队列族（备选）
			if (presentIndex == queueFamilyProperties.size())
			{
				for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
				{
					if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
					{
						presentIndex = static_cast<uint32_t>(i);
						break;
					}
				}
			}
		}
		if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size()))
		{
			throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
		}

		// 启用特性
		vk::StructureChain<
		    vk::PhysicalDeviceFeatures2,
		    vk::PhysicalDeviceVulkan12Features,
		    vk::PhysicalDeviceVulkan13Features,
		    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
		    vk::PhysicalDeviceRayQueryFeaturesKHR>
		    featureChain = {
		        {.features = {.samplerAnisotropy = true}},                    // 开启各项异性过滤
		        {.shaderSampledImageArrayNonUniformIndexing    = true,        // 开启无绑定特性
		         .descriptorBindingSampledImageUpdateAfterBind = true,
		         .descriptorBindingPartiallyBound              = true,
		         .descriptorBindingVariableDescriptorCount     = true,
		         .runtimeDescriptorArray                       = true,
		         .bufferDeviceAddress                          = true},
		        {.synchronization2 = true, .dynamicRendering = true},        // 开启同步、动态渲染
		        {.extendedDynamicState = true},                              // 开启动态状态
		        {.accelerationStructure = true},                             // 开启加速架构
		        {.rayQuery = true}                                           // 开启光线查询（Ray Query）
		    };

		// 队列创建信息
		float                     queuePriority = 0.5f;        // 队列优先级（0 ~ 1）
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
		    .queueFamilyIndex = graphicsIndex,
		    .queueCount       = 1,
		    .pQueuePriorities = &queuePriority};

		// 设备创建信息
		vk::DeviceCreateInfo deviceCreateInfo{
		    .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),        // 将 featureChain 挂载到 pNext
		    .queueCreateInfoCount    = 1,
		    .pQueueCreateInfos       = &deviceQueueCreateInfo,
		    .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
		    .ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device        = vk::raii::Device(physicalDevice, deviceCreateInfo);
		graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);        // 获取图形队列（由于仅申请了一个队列，所以下标是 0）
		presentQueue  = vk::raii::Queue(device, presentIndex, 0);         // 获取呈现队列
	}

	// 创建交换链
	void createSwapChain()
	{
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);        // 获取窗口表面能力

		swapChainExtent        = chooseSwapExtent(surfaceCapabilities);                                         // 选择格式
		swapChainSurfaceFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));        // 选择分辨率
		auto minImageCount     = std::max(3u, surfaceCapabilities.minImageCount);                               // 三缓冲
		minImageCount          = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{
		    .surface          = *surface,                                                                         // 抽象窗口
		    .minImageCount    = minImageCount,                                                                    // 图像数量（vulkan 规定，至少为 2（双缓冲））
		    .imageFormat      = swapChainSurfaceFormat.format,                                                    // 图像格式
		    .imageColorSpace  = swapChainSurfaceFormat.colorSpace,                                                // 色彩空间
		    .imageExtent      = swapChainExtent,                                                                  // 图像尺寸
		    .imageArrayLayers = 1,                                                                                // 数组层数（每图像）
		    .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,                                         // 颜色附件（用途，管线将其作为渲染目标）
		    .imageSharingMode = vk::SharingMode::eExclusive,                                                      // 独占（仅归属一个队列族，若图形和呈现队列不同，需显式转移所有权）
		    .preTransform     = surfaceCapabilities.currentTransform,                                             // 预变换标志（声明程序手动处理旋转，手机无需再纠正）（手机横置时，需对调宽高）
		    .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,                                           // 混合模式（与操作系统其他窗口的混合）
		    .presentMode      = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface)),        // 呈现模式（eFifo/eMailbox/eImmediate)
		    .clipped          = true                                                                              // 开启裁剪（本窗口被其他窗口遮挡、超出屏幕的部分被裁剪）
		};

		swapChain       = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	// 创建图像视图
	void createImageViews()
	{
		vk::ImageViewCreateInfo imageViewCreateInfo{
		    .viewType         = vk::ImageViewType::e2D,
		    .format           = swapChainSurfaceFormat.format,
		    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};        // 访问颜色层面，具体通道由 format 决定
		for (auto &image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	// 创建描述符布局
	void createDescriptorSetLayout()
	{
		// 描述符布局（全局）
		std::array global_bindings = {
		    // 绑定点（全局）
		    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),        // MVP + camerapos（绑定点，描述符类型，描述符数量，使用阶段，不可变采样器）
		    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eFragment, nullptr),                                // TLAS
		    vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),                                           // 索引数据 SSBO
		    vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),                                           // UV 数据 SSBO
		    vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),                                           // 实例表 SSBO
		};
		vk::DescriptorSetLayoutCreateInfo globalLayoutInfo{.bindingCount = static_cast<uint32_t>(global_bindings.size()), .pBindings = global_bindings.data()};
		descriptorSetLayoutGlobal = vk::raii::DescriptorSetLayout(device, globalLayoutInfo);        // 描述符布局->多个绑定点，绑定点->多个描述符

		// 描述符布局（材质）
		uint32_t   textureCount      = static_cast<uint32_t>(textureImageViews.size());
		std::array material_bindings = {
		    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr),                                               // 采样方式（nullptr 表示后续分配实际采样器对象）（eCombinedImageSampler = eSampledImage + eSampler 绑死，但不同图片的采样方式可能是相同的）
		    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eSampledImage, static_cast<uint32_t>(textureCount), vk::ShaderStageFlagBits::eFragment, nullptr),        // 采样图片
		};

		std::vector<vk::DescriptorBindingFlags> bingdingFlags = {
		    // 绑定标志（材质）
		    vk::DescriptorBindingFlagBits::eUpdateAfterBind,                                                                                                                  // 绑定后更新（绑定点 0）
		    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::eUpdateAfterBind        // 部分绑定 | 可变数量 | 绑定后更新（绑定点 1）（Vukan 规定，可变描述符数量的绑定点，必须是布局的最后一个绑定点）
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo{.bindingCount = static_cast<uint32_t>(bingdingFlags.size()), .pBindingFlags = bingdingFlags.data()};        // 扩展结构体

		vk::DescriptorSetLayoutCreateInfo materialLayoutInfo{
		    .pNext        = &flagsCreateInfo,
		    .flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
		    .bindingCount = static_cast<uint32_t>(material_bindings.size()),
		    .pBindings    = material_bindings.data()};
		descriptorSetLayoutMaterial = vk::raii::DescriptorSetLayout(device, materialLayoutInfo);
	}

	// 创建图形管线
	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		// 着色器阶段（顶点 + 片元）
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		// 顶点输入
		auto bindingDescription    = Vertex::getBindingDescription();           // 绑定描述（绑定点，顶点步长，更新频率）
		auto attributeDescriptions = Vertex::getAttributeDescriptions();        // 属性描述

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		    .vertexBindingDescriptionCount   = 1,
		    .pVertexBindingDescriptions      = &bindingDescription,        // 绑定描述（绑定点->缓冲区）
		    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		    .pVertexAttributeDescriptions    = attributeDescriptions.data()        // 属性描述
		};

		// 输入装配
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		    .topology               = vk::PrimitiveTopology::eTriangleList,        // 图元拓扑（三角形列表，每三个点画一个三角形）
		    .primitiveRestartEnable = vk::False                                    // 图元重启（关闭，仅 Strip 图元需要）
		};

		// 视口 + 裁剪
		vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};        // 先指定数量，后续指定内容（动态状态）

		// 光栅化器
		vk::PipelineRasterizationStateCreateInfo rasterizer{
		    .depthClampEnable        = vk::False,                               // 深度截断（关闭，则超出视锥体的片元直接丢弃）
		    .rasterizerDiscardEnable = vk::False,                               // 光栅化丢弃（开启，则不会输出任何片元）
		    .polygonMode             = vk::PolygonMode::eFill,                  // 多边形模式（填充三角形内部）（eFill / eLine）
		    .cullMode                = vk::CullModeFlagBits::eBack,             // 剔除模式（剔除背面）
		    .frontFace               = vk::FrontFace::eCounterClockwise,        // 正面定义（逆时针为正面）
		    .depthBiasEnable         = vk::False,                               // 关闭深度偏移
		    .lineWidth               = 1.0f                                     // 线宽（仅 eLine 下有效）
		};

		// 多重采样
		vk::PipelineMultisampleStateCreateInfo multisampling{
		    .rasterizationSamples = vk::SampleCountFlagBits::e1,        // 采样数（1，关闭 MSAA）
		    .sampleShadingEnable  = vk::False                           // 子样本着色（关闭）
		};

		// 深度 + 模板测试
		vk::PipelineDepthStencilStateCreateInfo depthStencil{
		    .depthTestEnable       = vk::True,                    // 深度测试（开启）
		    .depthWriteEnable      = vk::True,                    // 深度写入（开启）
		    .depthCompareOp        = vk::CompareOp::eLess,        // 比较（更小的值通过测试）
		    .depthBoundsTestEnable = vk::False,                   // 深度边界测试（开启，则丢弃不在指定 [min,max] 深度范围内的片元）（延迟贴花）
		    .stencilTestEnable     = vk::False                    // 关闭模板测试
		};

		// 颜色混合
		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		    .blendEnable    = vk::False,                                                                                                                               // 关闭混合
		    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA        // 写入通道（RGBA）
		};
		vk::PipelineColorBlendStateCreateInfo colorBlending{
		    .logicOpEnable   = vk::False,                 // 逻辑操作（过时，对源和目标颜色进行位运算，得到混合结果）
		    .logicOp         = vk::LogicOp::eCopy,        // 逻辑操作类型
		    .attachmentCount = 1,
		    .pAttachments    = &colorBlendAttachment        // 每渲染目标->颜色混合附件
		};

		// 动态状态（视口 + 裁剪）
		std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

		vk::PipelineDynamicStateCreateInfo dynamicState{
		    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		    .pDynamicStates    = dynamicStates.data()};

		// 管线布局
		vk::DescriptorSetLayout setLayouts[] = {*descriptorSetLayoutGlobal, *descriptorSetLayoutMaterial};

		vk::PushConstantRange pushConstantRange{
		    // 推送常量
		    .stageFlags = vk::ShaderStageFlagBits::eFragment,
		    .offset     = 0,
		    .size       = sizeof(PushConstant)};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		    .setLayoutCount         = 2,                 // 描述符布局数量（管线布局->多个描述符布局）
		    .pSetLayouts            = setLayouts,        // 描述符布局（数组）
		    .pushConstantRangeCount = 1,
		    .pPushConstantRanges    = &pushConstantRange};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		// 渲染管线
		vk::Format depthFormat = findDepthFormat();

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		    // 标准管线配置
		    {
		        .stageCount          = 2,
		        .pStages             = shaderStages,            // 着色器阶段
		        .pVertexInputState   = &vertexInputInfo,        // 顶点输入
		        .pInputAssemblyState = &inputAssembly,          // 输入装配
		        .pViewportState      = &viewportState,          // 视口 + 裁剪
		        .pRasterizationState = &rasterizer,             // 光栅化
		        .pMultisampleState   = &multisampling,          // 多重采样
		        .pDepthStencilState  = &depthStencil,           // 深度 + 模板测试
		        .pColorBlendState    = &colorBlending,          // 颜色混合
		        .pDynamicState       = &dynamicState,           // 动态状态（允许在 CommandBuffer 中动态修改）
		        .layout              = pipelineLayout,          // 管线布局
		        .renderPass          = nullptr                  // 渲染通道（动态渲染不需要）
		    },
		    // 动态渲染配置
		    {
		        .colorAttachmentCount    = 1,
		        .pColorAttachmentFormats = &swapChainSurfaceFormat.format,        // 颜色附件格式
		        .depthAttachmentFormat   = depthFormat                            // 深度附件格式
		    }};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// 创建命令池
	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
		    .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,        // 允许单独重置（允许命令池分配的命令缓冲，单独重置以复用）
		    .queueFamilyIndex = graphicsIndex                                              // 绑定图形队列族（命令池分配的命令缓冲，只能提交给此队列族执行）
		};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	// 创建深度资源
	void createDepthResources()
	{
		vk::Format depthFormat = findDepthFormat();        // 深度格式
		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
		            vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
		            vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);                  // 创建深度图像
		depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);        // 创建深度图像视图
	}

	// 辅助函数：选取深度格式（候选格式，平铺模式要求，特性要求）
	vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlagBits features) const
	{
		// 遍历候选格式
		for (auto format : candidates)
		{
			vk::FormatProperties props = physicalDevice.getFormatProperties(format);        // 像物理设备查询改格式的属性

			if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)        // 要求线性平铺，且该格式在线性平铺下支持要求的特性
				return format;
			if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)        // 要求优化平铺，且该格式在优化平铺下支持要求的特性
				return format;
		}
		throw std::runtime_error("failed to find supported format");
	}

	// 辅助函数：选取深度格式
	[[nodiscard]] vk::Format findDepthFormat() const
	{
		return findSupportedFormat(
		    {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},        // 候选格式列表
		    vk::ImageTiling::eOptimal,                                                                  // 平铺模式要求
		    vk::FormatFeatureFlagBits::eDepthStencilAttachment                                          // 特性要求
		);
	}

	// 辅助函数：检查模板分量
	static bool hasStencilComponent(vk::Format format)
	{
		return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
	}

	// 创建纹理
	std::pair<vk::raii::Image, vk::raii::DeviceMemory> createTextureImage(const std::string &path)
	{
		// 加载图像（stbi 库）
		int            texWidth, texHeight, texChannels;
		stbi_uc       *pixels    = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);        // STBI_rgb_alpha 表示强制加载 alpha 通道，即使原图没有
		vk::DeviceSize imageSize = texWidth * texHeight * 4;                                                            // 4 是每个像素的字节数

		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		// 暂存缓冲区
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		// 拷贝图像->暂存缓冲区
		void *data = stagingBufferMemory.mapMemory(0, imageSize);        // 暂存缓冲区指针（开启映射）（从 0 开始）
		memcpy(data, pixels, imageSize);                                 // 拷贝
		stagingBufferMemory.unmapMemory();                               // 解释映射
		stbi_image_free(pixels);                                         // 释放图像

		// 纹理图像（显存）
		vk::raii::Image        textureImage       = nullptr;
		vk::raii::DeviceMemory textureImageMemory = nullptr;
		createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
		            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		            vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

		// 图像布局转换 + 拷贝
		transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);                    // 适合 GPU 拷贝
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));         // 拷贝
		transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);        // 适合 Shader 采样
	}

	// 创建纹理视图
	vk::raii::ImageView createTextureImageView(vk::raii::Image &textureImage)
	{
		return createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);        // shader 通过 ImageView 访问 Image
	}

	// 创建纹理采样器
	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();        // 获取物理设备属性

		vk::SamplerCreateInfo samplerInfo{
		    .magFilter        = vk::Filter::eLinear,                           // 放大过滤器（纹理像素 > 屏幕像素）（线性插值）
		    .minFilter        = vk::Filter::eLinear,                           // 缩小过滤器（纹理像素 < 屏幕像素）（线性插值）
		    .mipmapMode       = vk::SamplerMipmapMode::eLinear,                // Mipmap 模式（Mipmap 之间，线性插值）
		    .addressModeU     = vk::SamplerAddressMode::eRepeat,               // U 轴（重复）（UV 超出 [0,1] 时，重复纹理）
		    .addressModeV     = vk::SamplerAddressMode::eRepeat,               // V 轴（重复）
		    .addressModeW     = vk::SamplerAddressMode::eRepeat,               // W 轴（重复）
		    .mipLodBias       = 0.0f,                                          // Mipmap 偏移量（正值->模糊，负值->清晰）
		    .anisotropyEnable = vk::True,                                      // 各向异性过滤（短轴决定 Mipmap，沿长轴采样多个正方形 Mipmap）（解决倾斜观察时的模糊问题）（若长轴决定 mipmap，则短轴使用的 mipmap 级别过高导致模糊）(若短轴决定 mipmap，则有严重闪烁）
		    .maxAnisotropy    = properties.limits.maxSamplerAnisotropy,        // 各向异性采样最大样本数
		    .compareEnable    = vk::False,                                     // 比较操作（禁用，仅阴影贴图需要)
		    .compareOp        = vk::CompareOp::eAlways,
		    .minLod           = 0.0f,                   // 最小 Mipmap 限制（最清晰 Mipmap，0）
		    .maxLod           = vk::LodClampNone        // 最大 Mipmap 限制（最模糊 Mipmap，无限制）
		};

		textureSampler = vk::raii::Sampler(device, samplerInfo);
	}

	// 辅助函数：创建图像视图
	vk::raii::ImageView createImageView(vk::raii::Image &image, vk::Format format, vk::ImageAspectFlagBits aspectFlags) const
	{
		vk::ImageViewCreateInfo viewInfo{.image = image, .viewType = vk::ImageViewType::e2D, .format = format,        // 图像，视图类型，解释格式（通常与 Image 一致，但深度模板纹理有两种格式，深度浮点，模板 UINT）
		                                 .subresourceRange = {aspectFlags, 0, 1, 0, 1}};                              // 访问层面，mipmap 起始+长度，数组起始+长度
		return vk::raii::ImageView(device, viewInfo);
	}

	// 辅助函数：创建图像
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling,                                    // 宽，高，Mipmap 数，格式，内存排列
	                 vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory)        // 用途，内存属性，传出参数（图像+内存）
	{
		vk::ImageCreateInfo imageInfo{
		    .imageType   = vk::ImageType::e2D,                 // 图像类型（1D/2D/3D）
		    .format      = format,                             // 像素格式（颜色通道的排列、大小、数据类型）
		    .extent      = {width, height, 1},                 // 图像尺寸（宽，高，深）（2D 图深度为 1）
		    .mipLevels   = mipLevels,                          // MIP 级数
		    .arrayLayers = 1,                                  // 数组层数
		    .samples     = vk::SampleCountFlagBits::e1,        // MSAA 采样数
		    .tiling      = tiling,                             // 内存排列（eLinear：行主序排列，CPU 可读写，GPU 性能差）（eOptimal：GPU 分块优化排列，CPU 无法读写，GPU 性能最佳）(图像创建后不可更改）
		    .usage       = usage,                              // 图像用途
		    .sharingMode = vk::SharingMode::eExclusive         // 队列族共享模式
		};

		image = vk::raii::Image(device, imageInfo);        // 创建图像句柄

		// 分配内存
		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{.allocationSize  = memRequirements.size,
		                                 .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);

		image.bindMemory(imageMemory, 0);        // 图像句柄绑定内存（从该内存的 0 字节开始绑定）
	}

	// 辅助函数：图像布局转换（预处理阶段）
	void transitionImageLayout(const vk::raii::Image &image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)        // 图像，原布局，目标布局（按可能布局中，内存最大的分配内存）(布局转换改变有效内存大小，物理内存不变）
	{
		auto commandBuffer = beginSingleTimeCommands();        // 一次性命令缓冲

		vk::ImageMemoryBarrier barrier{
		    .oldLayout = oldLayout, .newLayout = newLayout, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,        // 原布局，目标布局，原队列族，目标队列族，图像（这里无队列族所有权转移）
		    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};                        // 同步范围（层面，mipmap，数组）

		vk::PipelineStageFlags sourceStage;             // 屏障执行前，必须完成的阶段（生产者）(屏障之中，是图像布局转换操作）
		vk::PipelineStageFlags destinationStage;        // 屏障执行后，才能开始的阶段（消费者）

		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			// 内存依赖（挂载于执行依赖，可以没有）
			barrier.srcAccessMask = {};                                        // 屏障执行前，sourceStage 中的 srcAccessMask 操作必须对屏障可见（从 L1 刷入 L2）（eUndefined 无需同步旧数据）
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;        // destinationStage 中的 dstAccessMask 操作执行前，屏障执行结果必须可见（元数据缓存失效）

			// 执行依赖（必须有）
			sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;        // 管线顶端（不等待）
			destinationStage = vk::PipelineStageFlagBits::eTransfer;         // 传输阶段
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;        // 屏障执行前，传输写入操作必须可见
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;           // Shader 读取前，屏障执行结果必须可见

			sourceStage      = vk::PipelineStageFlagBits::eTransfer;              // 传输阶段
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;        // 片段着色器阶段
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		commandBuffer->pipelineBarrier(sourceStage, destinationStage,        // 源阶段，目标阶段
		                               {},                                   // 依赖标志（eByRegion（局部依赖）: 移动端分块渲染 + 片上缓存优化）（空（全局依赖）: 下一阶段必须等待上一阶段处理完所有像素）
		                               {},                                   // 全局内存屏障（屏障无执行操作，会同步所有满足要求的内存）
		                               nullptr, barrier);                    // 缓冲区内存屏障，图像内存屏障（屏障有执行操作，同步指定内存）

		endSingleTimeCommands(*commandBuffer);        // 结束录制，提交命令缓冲
	}

	// 辅助函数：缓冲区拷贝到图像
	void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height)        // 缓冲区，图像，宽，高
	{
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();        // 一次性命令缓冲

		vk::BufferImageCopy region{
		    // 源区域
		    .bufferOffset      = 0,        // 偏移量
		    .bufferRowLength   = 0,        // 行长（像素单位）（0 表示像素紧密排列，无空隙）
		    .bufferImageHeight = 0,        // 高度
		    // 目标区域
		    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},        // 层面，0 级 Mipmap，数组
		    .imageOffset      = {0, 0, 0},                                                                                                   // 起始坐标 (x, y, z)
		    .imageExtent      = {width, height, 1}                                                                                           // 图像尺寸 (宽, 高, 深)
		};
		commandBuffer->copyBufferToImage(*buffer, *image, vk::ImageLayout::eTransferDstOptimal, {region});        // 缓冲区，图像，图像当前布局，拷贝区域

		endSingleTimeCommands(*commandBuffer);
	}

	// 加载模型（.obj）
	void loadModel()
	{
		tinyobj::attrib_t                attrib;                // 顶点属性（OBJ 保证顶点属性去重）
		std::vector<tinyobj::shape_t>    shapes;                // 子网格
		std::vector<tinyobj::material_t> localMaterials;        // 材质
		std::string                      warn, err;             // 警告/错误信息

		if (!tinyobj::LoadObj(&attrib, &shapes, &localMaterials, &warn, &err, MODEL_PATH.c_str()))
		{
			throw std::runtime_error(warn + err);
		}

		size_t materialOffset  = materials.size();        // 全局材质偏移量
		size_t oldTextureCount = textureImageViews.size();

		materials.insert(materials.end(), localMaterials.begin(), localMaterials.end());        // 将新加载材质添加到全局材质列表

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};

		uint32_t indexOffset = 0;        // 已处理的索引总数

		// 遍历子网格
		for (const auto &shape : shapes)
		{
			std::cout << "Loading mesh: " << shape.name << ": " << shape.mesh.indices.size() / 3 << " triangles\n";

			uint32_t startOffset = indexOffset;        // 子网格在索引缓冲区的起始偏移量
			uint32_t localMaxV   = 0;                  // 子网格引用的最大顶点索引

			// 子网格顶点
			for (const auto &index : shape.mesh.indices)
			{
				Vertex vertex{};

				vertex.pos = {
				    attrib.vertices[3 * index.vertex_index + 0],
				    attrib.vertices[3 * index.vertex_index + 1],
				    attrib.vertices[3 * index.vertex_index + 2]};

				vertex.texCoord = {
				    attrib.texcoords[2 * index.texcoord_index + 0],
				    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};        // Vulkan UV 坐标原点在左上角，OBJ/OpenGL 在左下角

				vertex.color = {1.0f, 1.0f, 1.0f};

				if (index.normal_index >= 0)
				{
					vertex.normal = {
					    attrib.normals[3 * index.normal_index + 0],
					    attrib.normals[3 * index.normal_index + 1],
					    attrib.normals[3 * index.normal_index + 2]};
				}

				if (!uniqueVertices.contains(vertex))        // 顶点去重（共用顶点被去重)
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());        // 记录顶点索引
					vertices.push_back(vertex);
				}

				indices.push_back(uniqueVertices[vertex]);        // 将顶点索引，加入索引缓冲区
				indexOffset++;

				uint32_t vi = uniqueVertices[vertex];
				localMaxV   = std::max(localMaxV, vi);        // 更新子网格最大顶点索引
			}

			// 子网格材质
			int localMaterialID  = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[0];                      // OBJ 中的材质 ID（material_ids 长度 == 子网格面数）（这里假设，子网格与材质一比一）
			int globalMaterialID = (localMaterialID < 0) ? -1 : static_cast<int>(materialOffset + localMaterialID);        // 全局材质 ID

			uint32_t indexCount = indexOffset - startOffset;        // 当前子网格索引总数

			bool alphaCut   = (shape.name.find("nettle_plant") != std::string::npos);        // alpha 裁剪（开关）（子串匹配）
			bool reflective = (shape.name.find("table") != std::string::npos);               // 反射（开关）

			submeshes.push_back({
			    .indexOffset = startOffset,             // 起始索引
			    .indexCount  = indexCount,              // 索引总数
			    .materialID  = globalMaterialID,        // 全局材质 ID
			    .firstVertex = 0u,                      // 顶点起始偏移（无）
			    .maxVertex   = localMaxV + 1,
			    .alphaCut    = alphaCut,         // alpha 裁剪
			    .reflective  = reflective        // 反射
			});
		}

		// 遍历材质
		for (size_t i = 0; i < localMaterials.size(); i++)
		{
			const auto &material = localMaterials[i];

			// 加载漫反射纹理
			if (!material.diffuse_texname.empty())
			{
				std::string texturePath = MODEL_PATH.substr(0, MODEL_PATH.find_last_of("/\\")) + "/" + material.diffuse_texname;        // 拼接纹理路径
				auto [img, mem]         = createTextureImage(texturePath);                                                              // 创建纹理
				textureImages.push_back(std::move(img));
				textureImageMemories.push_back(std::move(mem));
				textureImageViews.emplace_back(createTextureImageView(textureImages.back()));        // 创建纹理视图
			}
			else
			{
				// 无漫反射纹理
				std::cout << "No Texture for material: " << material.name << std::endl;
			}
		}
	}

	// 创建顶点缓冲区
	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();        // 顶点总数

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		// 暂存缓冲区（CPU 可访问，GPU 通过 PCIe 慢速读取）
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);        // CPU 可访问，写入数据自动同步缓存，使 GPU 可见

		// 映射内存
		void *data = stagingBufferMemory.mapMemory(0, bufferSize);        // CPU 可访问的虚拟地址，映射到缓冲区内存（内存必须具备 eHostVisible 属性）（修改页表）
		memcpy(data, vertices.data(), bufferSize);                        // 拷贝（后续 GPU DMA 通过 PCIe 搬运数据到显存）
		stagingBufferMemory.unmapMemory();                                // 解除映射，虚拟地址不再可访问（恢复页表）

		// 顶点缓冲区（显存）
		createBuffer(bufferSize,
		             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer |
		                 vk::BufferUsageFlagBits::eShaderDeviceAddress |                              // 允许获取缓冲区的 GPU 虚拟地址（光线追踪必须）
		                 vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,        // 构建加速结构（AS）的输入源
		             vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

		// 拷贝缓冲区（阻塞等待拷贝完成，才能安全 RAII 释放暂存缓冲区内存）
		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	// 创建索引缓冲区
	void createIndexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();        // 索引总数

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		// 暂存缓冲区
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);        // CPU 可访问，写入数据自动同步缓存，使 GPU 可见

		// 映射内存
		void *data = stagingBufferMemory.mapMemory(0, bufferSize);        // 从第 0 字节开始映射，映射 bufferSize 长度
		memcpy(data, indices.data(), (size_t) bufferSize);
		stagingBufferMemory.unmapMemory();

		// 索引缓冲区（显存）
		createBuffer(bufferSize,
		             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer |
		                 vk::BufferUsageFlagBits::eShaderDeviceAddress |                               // GPU 地址（虚拟）
		                 vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |        // 构建加速结构（AS）的输入源
		                 vk::BufferUsageFlagBits::eStorageBuffer,                                      // SSBO
		             vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}

	// 创建 UV 缓冲区
	void createUVBuffer()
	{
		// 提取顶点 UV 坐标
		std::vector<glm::vec2> uvs;
		uvs.reserve(vertices.size());
		for (auto &v : vertices)
		{
			uvs.push_back(v.texCoord);
		}

		vk::DeviceSize         bufferSize = sizeof(uvs[0]) * uvs.size();
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		// 暂存缓冲区
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		// 映射内存
		void *data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, indices.data(), (size_t) bufferSize);
		stagingBufferMemory.unmapMemory();

		// UV 缓冲区（显存）
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, uvBuffer, uvBufferMemory);        // SSBO，允许被 Shader 随机访问

		copyBuffer(stagingBuffer, uvBuffer, bufferSize);
	}

	// 创建实例查找表缓冲区
	void createInstanceLUTBuffer()
	{
#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
		vk::DeviceSize         bufferSize = sizeof(InstanceLUT) * instanceLUTs.size();
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		// 暂存缓冲区
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		// 映射内存
		void *data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, indices.data(), (size_t) bufferSize);
		stagingBufferMemory.unmapMemory();

		// 实例查找表缓冲区（显存）
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, instanceLUTBuffer, instanceLUTBufferMemory);        // SSBO，允许被 Shader 随机访问

		copyBuffer(stagingBuffer, instanceLUTBuffer, bufferSize);
#endif
	}

	// 创建 UBO
	void createUniformBuffers()
	{
		// 清理旧数据（重建 SwapChain 需要）
		uniformBuffers.clear();
		uniformBuffersMemory.clear();
		uniformBuffersMapped.clear();

		// 每帧创建 UBO
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DeviceSize         bufferSize = sizeof(UniformBufferObject);
			vk::raii::Buffer       buffer({});
			vk::raii::DeviceMemory bufferMem({});

			// 创建 UBO（不用暂存缓冲区）
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);        // UBO 常驻于 CPU 内存，GPU 通过 PCIe 读取（UBO 频繁更新且数据量小）（频繁更新且数据量大，用 SSBO + ComputeShader）

			uniformBuffers.emplace_back(std::move(buffer));
			uniformBuffersMemory.emplace_back(std::move(bufferMem));
			uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));        // 持久映射
		}
	}

	// 创建加速结构（AS）
	void createAccelerationStructures()
	{
#if LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
		vk::BufferDeviceAddressInfo vai{.buffer = *vertexBuffer};
		vk::DeviceAddress           vertexAddr = device.getBufferAddressKHR(vai);        // 顶点缓冲区 GPU 地址（虚拟）
		vk::BufferDeviceAddressInfo iai{.buffer = *indexBuffer};
		vk::DeviceAddress           indexAddr = device.getBufferAddressKHR(iai);        // 索引缓冲区 GPU 地址（虚拟）

		instances.reserve(submeshes.size());
		blasBuffers.reserve(submeshes.size());
		blasMemories.reserve(submeshes.size());
		blasHandles.reserve(submeshes.size());

		// 仿射变换（3x4 矩阵，旋转、缩放、平移，不含投影变换）（省显存）
		vk::TransformMatrixKHR identity{};
		identity.matrix = std::array<std::array<float, 4>, 3>{{std::array<float, 4>{1.f, 0.f, 0.f, 0.f},
		                                                       std::array<float, 4>{0.f, 1.f, 0.f, 0.f},
		                                                       std::array<float, 4>{0.f, 0.f, 1.f, 0.f}}};

		// 遍历子网格（构建 BLAS）
		for (size_t i = 0; i < submeshes.size(); i++)
		{
			const auto &submesh = submeshes[i];

			// 几何体描述（三角形）
			auto trianglesData = vk::AccelerationStructureGeometryTrianglesDataKHR{
			    .vertexFormat = vk::Format::eR32G32B32Sfloat,                             // 顶点位置格式
			    .vertexData   = vertexAddr,                                               // 顶点缓冲区地址
			    .vertexStride = sizeof(Vertex),                                           // 顶点步长
			    .maxVertex    = submesh.maxVertex,                                        // 涉及的最大顶点索引（优化用，避免加载整个顶点缓冲区）
			    .indexType    = vk::IndexType::eUint32,                                   // 顶点索引格式
			    .indexData    = indexAddr + submesh.indexOffset * sizeof(uint32_t)        // 索引缓冲区地址
			};

			// 几何体描述（标准）（统一包装三角形、AABB、实例）
			vk::AccelerationStructureGeometryDataKHR geometryData(trianglesData);

			// 几何体描述（BLAS）
			vk::AccelerationStructureGeometryKHR blasGeometry{
			    .geometryType = vk::GeometryTypeKHR::eTriangles,        // 三角形几何体
			    .geometry     = geometryData,
			    .flags        = vk::GeometryFlagBitsKHR::eOpaque        // 不透明物体
			};

#	if LAB_TASK_LEVEL >= LAB_TASK_AS_OPAQUE_FLAG
			// 若是 AlphaCut 材质，则移除 Opaque 标志
			blasGeometry.flags = (submesh.alphaCut) ? vk::GeometryFlagsKHR(0) : vk::GeometryFlagBitsKHR::eOpaque;
#	endif

			// 构建信息（BLAS）
			vk::AccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo{
			    .type          = vk::AccelerationStructureTypeKHR::eBottomLevel,        // 类型（BLAS/TLAS）
			    .mode          = vk::BuildAccelerationStructureModeKHR::eBuild,         // 模式（新建/更新）
			    .geometryCount = 1,
			    .pGeometries   = &blasGeometry,        // 几何体
			};

			// 暂存缓冲区（大小 == 构建所需内存）（存放构建时的临时数据）
			auto primitiveCount = static_cast<uint32_t>(submesh.indexCount / 3);        // 三角形数量

			vk::AccelerationStructureBuildSizesInfoKHR blasBuildSizes =        // 构建所需内存（BLAS）
			    device.getAccelerationStructureBuildSizesKHR(
			        vk::AccelerationStructureBuildTypeKHR::eDevice,
			        blasBuildGeometryInfo,
			        {primitiveCount}        // 图元数量
			    );

			// 暂存缓冲区（创建）
			vk::raii::Buffer       scratchBuffer = nullptr;
			vk::raii::DeviceMemory scratchMemory = nullptr;
			createBuffer(blasBuildSizes.buildScratchSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,        // SSBO，GPU 地址（虚拟）
			             vk::MemoryPropertyFlagBits::eDeviceLocal, scratchBuffer, scratchMemory);

			// 暂存缓冲区（GPU 地址填入构建信息）
			vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *scratchBuffer};
			vk::DeviceAddress           scratchAddr         = device.getBufferAddressKHR(scratchAddressInfo);        // 暂存缓冲区 GPU 地址（虚拟）
			blasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;                                           // 将暂存缓冲区地址，填入构建信息

			// BLAS 缓冲区（存储构建好的 BLAS）
			vk::raii::Buffer       blasBuffer = nullptr;
			vk::raii::DeviceMemory blasMemory = nullptr;
			blasBuffers.emplace_back(std::move(blasBuffer));
			blasMemories.emplace_back(std::move(blasMemory));
			createBuffer(blasBuildSizes.accelerationStructureSize,
			             vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |                      // AS 专用存储
			                 vk::BufferUsageFlagBits::eShaderDeviceAddress |                              // GPU 地址（虚拟）
			                 vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,        // AS 构建时的输入
			             vk::MemoryPropertyFlagBits::eDeviceLocal,
			             blasBuffers[i], blasMemories[i]);

			// BLAS 句柄（填入构建信息）
			vk::AccelerationStructureCreateInfoKHR blasCreateInfo{
			    .buffer = blasBuffers[i],
			    .offset = 0,
			    .size   = blasBuildSizes.accelerationStructureSize,
			    .type   = vk::AccelerationStructureTypeKHR::eBottomLevel};
			blasHandles.emplace_back(device.createAccelerationStructureKHR(blasCreateInfo));
			blasBuildGeometryInfo.dstAccelerationStructure = blasHandles[i];        // 将 BLAS 句柄填入构建信息（存储构建结果）

			// 构建范围（BLAS）
			vk::AccelerationStructureBuildRangeInfoKHR blasRangeInfo{
			    .primitiveCount  = primitiveCount,             // 图元数量
			    .primitiveOffset = 0,                          // 图元偏移
			    .firstVertex     = submesh.firstVertex,        // 顶点偏移
			    .transformOffset = 0                           // 变换矩阵偏移
			};

			// 构建 BLAS
			auto cmd = beginSingleTimeCommands();
			cmd->buildAccelerationStructuresKHR({blasBuildGeometryInfo}, {&blasRangeInfo});        // GPU 构建加速结构（构建信息，构建范围）
			endSingleTimeCommands(*cmd);

			// BLAS 实例（实例化）
			vk::AccelerationStructureDeviceAddressInfoKHR addrInfo{.accelerationStructure = *blasHandles[i]};
			vk::DeviceAddress                             blasDeviceAddr = device.getAccelerationStructureAddressKHR(addrInfo);        // BLAS GPU 地址

			vk::AccelerationStructureInstanceKHR instance{
			    .transform                      = identity,             // 变换矩阵
			    .mask                           = 0xFF,                 // 可见性掩码
			    .accelerationStructureReference = blasDeviceAddr        // BLAS GPU 地址
			};
			instances.push_back(instance);

#	if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
			instances[i].instanceCustomIndex = static_cast<uint32_t>(i);                                     // 自定义索引（Shader 击中 Instance 时可获取此索引）
			instanceLUTs.push_back({static_cast<uint32_t>(submesh.materialID), submesh.indexOffset});        // 根据索引，去实例查找表获取材质
#	endif
		}

		// 实例缓冲区（TLAS）
		vk::DeviceSize instBufferSize = sizeof(instances[0]) * instances.size();
		createBuffer(instBufferSize, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
		             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, instanceBuffer, instanceMemory);

		void *ptr = instanceMemory.mapMemory(0, instBufferSize);
		memcpy(ptr, instances.data(), instBufferSize);
		instanceMemory.unmapMemory();

		vk::BufferDeviceAddressInfo instanceAddrInfo{.buffer = instanceBuffer};
		vk::DeviceAddress           instanceAddr = device.getBufferAddressKHR(instanceAddrInfo);

		// 几何体描述（实例）
		auto instanceData = vk::AccelerationStructureGeometryInstancesDataKHR{
		    .arrayOfPointers = vk::False,        // （True：指针数组；False：结构体数组）
		    .data            = instanceAddr};

		// 几何体描述（标准）
		vk::AccelerationStructureGeometryDataKHR geometryData(instanceData);

		// 几何体描述（TLAS）
		vk::AccelerationStructureGeometryKHR tlasGeometry{
		    .geometryType = vk::GeometryTypeKHR::eInstances,        // 实例数据
		    .geometry     = geometryData};

		// 构建信息（TLAS）
		vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{
		    .type          = vk::AccelerationStructureTypeKHR::eTopLevel,
		    .mode          = vk::BuildAccelerationStructureModeKHR::eBuild,
		    .geometryCount = 1,
		    .pGeometries   = &tlasGeometry};

#	if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
		tlasBuildGeometryInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;        // 允许更新（动画）
#	endif

		// 暂存缓冲区（大小 == 构建所需内存）
		auto primitiveCount = static_cast<uint32_t>(instances.size());        // 图元数量 = 实例数量

		vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizes =
		    device.getAccelerationStructureBuildSizesKHR(
		        vk::AccelerationStructureBuildTypeKHR::eDevice,
		        tlasBuildGeometryInfo,
		        {primitiveCount}        // 图元数量
		    );

		// 暂存缓冲区（创建）
		createBuffer(tlasBuildSizes.buildScratchSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		             vk::MemoryPropertyFlagBits::eDeviceLocal, tlasScratchBuffer, tlasScratchMemory);

		// 暂存缓冲区（GPU 地址填入构建信息）
		vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *tlasScratchBuffer};
		vk::DeviceAddress           scratchAddr         = device.getBufferAddressKHR(scratchAddressInfo);
		tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

		// TLAS 缓冲区
		createBuffer(tlasBuildSizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
		             vk::MemoryPropertyFlagBits::eDeviceLocal, tlasBuffer, tlasMemory);

		// TLAS 句柄
		vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{
		    .buffer = tlasBuffer,
		    .offset = 0,
		    .size   = tlasBuildSizes.accelerationStructureSize,
		    .type   = vk::AccelerationStructureTypeKHR::eTopLevel};
		tlas                                           = device.createAccelerationStructureKHR(tlasCreateInfo);
		tlasBuildGeometryInfo.dstAccelerationStructure = tlas;        // 将 TLAS 句柄填入构建信息（存储构建结果）

		// 构建范围（TLAS）
		vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
		    .primitiveCount  = primitiveCount,        // 实例数量
		    .primitiveOffset = 0,
		    .firstVertex     = 0,
		    .transformOffset = 0};

		// 构建 TLAS
		auto cmd = beginSingleTimeCommands();
		cmd->buildAccelerationStructuresKHR({tlasBuildGeometryInfo}, {&tlasRangeInfo});
		endSingleTimeCommands(*cmd);
#endif
	}

	// 创建描述符池
	void createDescriptorPool()
	{
		// 描述符总量
		std::array poolSize{
		    vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),                   // UBO（描述符类型，描述符数量）
		    vk::DescriptorPoolSize(vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT),        // TLAS/BLAS
		    vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 3),               // SSBO
		    vk::DescriptorPoolSize(vk::DescriptorType::eSampler, MAX_FRAMES_IN_FLIGHT),                         // Sampler
		    vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, (uint32_t) materials.size())              // Sampled Image
		};

		// 创建描述符池
		vk::DescriptorPoolCreateInfo poolInfo{
		    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |        // 允许单独释放描述符集
		             vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,           // 允许绑定后更新描述符
		    .maxSets       = MAX_FRAMES_IN_FLIGHT + 1,                             // 描述符集最大分配数量（+ 1 用于材质描述符集）
		    .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		    .pPoolSizes    = poolSize.data()};
		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);        // 描述符池(分配)->描述符集(容器)->描述符(指针)，均非实际资源
	}

	// 创建描述符集
	void createDescriptorSets()
	{
		// 全局描述符集（分配）
		std::vector<vk::DescriptorSetLayout> globalLayouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayoutGlobal);
		vk::DescriptorSetAllocateInfo        allocInfoGlobal{.descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(globalLayouts.size()), .pSetLayouts = globalLayouts.data()};        // 描述符池，描述符集(数组)，描述符集布局(数组，决定分配的描述符）
		globalDescriptorSets.clear();
		globalDescriptorSets = device.allocateDescriptorSets(allocInfoGlobal);

		// 全局描述符集（更新）
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			// 绑定点 0：UBO
			vk::DescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};                                                                                                           // UBO，偏移量，读取长度
			vk::WriteDescriptorSet   bufferWrite{.dstSet = globalDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo};        // 描述符集，绑定点，描述符(起始 + 长度 + 类型 + 数据源)

#if LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
			// 绑定点 1：TLAS
			vk::WriteDescriptorSetAccelerationStructureKHR asInfo{.accelerationStructureCount = 1, .pAccelerationStructures = {&*tlas}};                                                                                                                      // TLAS 句柄
			vk::WriteDescriptorSet                         asWrite{.pNext = &asInfo, .dstSet = globalDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eAccelerationStructureKHR};        // 扩展(数据源)，描述符集，绑定点，描述符(起始 + 长度 + 类型)
#endif

			// 绑定点 2：索引缓冲（SSBO）
			vk::DescriptorBufferInfo indexBufferInfo{.buffer = indexBuffer, .offset = 0, .range = sizeof(uint32_t) * indices.size()};                                                                                                                // 索引缓冲
			vk::WriteDescriptorSet   indexBufferWrite{.dstSet = globalDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indexBufferInfo};        // 描述符集，绑定点，描述符(起始 + 长度 + 类型 + 数据源)

			// 绑定点 3：UV 缓冲（SSBO）
			vk::DescriptorBufferInfo uvBufferInfo{.buffer = uvBuffer, .offset = 0, .range = sizeof(glm::vec2) * vertices.size()};                                                                                                              // uv 缓冲
			vk::WriteDescriptorSet   uvBufferWrite{.dstSet = globalDescriptorSets[i], .dstBinding = 3, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &uvBufferInfo};        // 描述符集，绑定点，描述符(起始 + 长度 + 类型 + 数据源)

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
			// 绑定点 4：实例查找表缓冲（SSBO）
			vk::DescriptorBufferInfo instanceLUTBufferInfo{.buffer = instanceLUTBuffer, .offset = 0, .range = sizeof(InstanceLUT) * instanceLUTs.size()};                                                                                                        // 实例查找表缓冲
			vk::WriteDescriptorSet   instanceLUTBufferWrite{.dstSet = globalDescriptorSets[i], .dstBinding = 4, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceLUTBufferInfo};        // 描述符集，绑定点，描述符(起始 + 长度 + 类型 + 数据源)
#endif

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
			// UBO，AS，Index，UV，LUT
			std::array<vk::WriteDescriptorSet, 5> descriptorWrites{bufferWrite, asWrite, indexBufferWrite, uvBufferWrite, instanceLUTBufferWrite};
#elif LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
			// UBO，AS，Index，UV
			std::array<vk::WriteDescriptorSet, 4> descriptorWrites{bufferWrite, asWrite, indexBufferWrite, uvBufferWrite};
#else
			// UBO，Index，UV
			std::array<vk::WriteDescriptorSet, 3> descriptorWrites{bufferWrite, indexBufferWrite, uvBufferWrite};
#endif
			device.updateDescriptorSets(descriptorWrites, {});        // 更新描述符集
		}

		// 材质描述符集（分配）
		std::vector<uint32_t>                                variableCounts = {static_cast<uint32_t>(textureImageViews.size())};
		vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{.descriptorSetCount = 1, .pDescriptorCounts = variableCounts.data()};        // 描述符数量（可变）
		std::vector<vk::DescriptorSetLayout>                 layouts{*descriptorSetLayoutMaterial};
		vk::DescriptorSetAllocateInfo                        allocInfo{.pNext = &variableCountInfo, .descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()};        // 扩展，描述符池，描述符集(数组)，描述符布局(数组)（Vukan 规定，可变描述符数量绑定点必须是布局最后一个绑定点）
		materialDescriptorSets = device.allocateDescriptorSets(allocInfo);

		// 绑定点 0：纹理采样器（Sampler）
		vk::DescriptorImageInfo samplerInfo{.sampler = textureSampler};
		vk::WriteDescriptorSet  samplerWrite{.dstSet = materialDescriptorSets[0], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &samplerInfo};        // 描述符集，绑定点，描述符(起始 + 长度 + 类型 + 数据源)
		device.updateDescriptorSets({samplerWrite}, {});

		// 绑定点 1：纹理数组（Sampled Images）
		std::vector<vk::DescriptorImageInfo> imageInfos;
		imageInfos.reserve(textureImageViews.size());

		// 遍历纹理
		for (auto &iv : textureImageViews)
		{
			vk::DescriptorImageInfo imageInfo{.imageView = iv, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};        // 数据源
			imageInfos.push_back(imageInfo);
		}
		vk::WriteDescriptorSet materialWrite{.dstSet = materialDescriptorSets[0], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = static_cast<uint32_t>(imageInfos.size()), .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = imageInfos.data()};        // 描述符集，绑定点，描述符(起始 + 长度 + 类型 + 数据源)
		device.updateDescriptorSets({materialWrite}, {});
	}

	// 辅助函数：创建缓冲区
	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)        // 大小，用途，内存属性，缓冲区(传出)，缓冲区内存(传出)
	{
		// 缓冲区（创建句柄）
		vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};        // 大小，用途，共享模式
		buffer = vk::raii::Buffer(device, bufferInfo);

		// 缓冲区（分配内存）
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();                                                                                        // 内存需求
		vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};        // 实际大小（内存对齐，填充尾部），内存类型索引（满足内存属性要求）

		vk::MemoryAllocateFlagsInfo allocFlagsInfo{};        // 处理扩展：缓冲区 GPU 地址
		if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)
		{
			allocFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
			allocInfo.pNext      = &allocFlagsInfo;
		}

		bufferMemory = vk::raii::DeviceMemory(device, allocInfo);        // 分配内存

		buffer.bindMemory(bufferMemory, 0);        // 句柄绑定内存（内存，内存偏移量）
	}

	// 辅助函数：一次性命令缓冲（开始）
	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
	{
		// 命令缓冲（分配）
		vk::CommandBufferAllocateInfo            allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};        // 命令池，主命令缓冲区(可直接提交给队列)
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

		// 开始录制
		vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};        // 仅提交一次（驱动基于此信息优化）
		commandBuffer->begin(beginInfo);
		return commandBuffer;
	}

	// 辅助函数：一次性命令缓冲（结束 + 提交 + 等待)
	void endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer)
	{
		// 结束录制
		commandBuffer.end();

		// 提交 + 等待
		vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
		graphicsQueue.submit(submitInfo, nullptr);        // 命令缓冲，栅栏
		graphicsQueue.waitIdle();                         // 阻塞 CPU 线程，直到图形队列所有命令执行完毕
	}

	// 辅助函数：拷贝缓冲区（缓冲区拷贝->无压缩传输，图形拷贝->压缩传输）
	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)        // 源缓冲，目标缓冲，拷贝大小(字节)
	{
		// 命令缓冲（分配）
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};        // 命令池，主命令缓冲
		vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());

		// 录制
		commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});        // 开始录制（eOneTimeSubmit：该缓冲仅提交一次，驱动可据此优化）
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));                                    // 拷贝命令（缓冲区内存对齐：内部数据紧密排列，尾部填充对齐）（拷贝有效区间，省略尾部即可）
		commandCopyBuffer.end();                                                                                             // 结束录制

		// 提交 + 等待
		graphicsQueue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);        // 命令缓冲，栅栏
		graphicsQueue.waitIdle();                                                                                              // 阻塞 CPU 线程，直到拷贝完成
	}

	// 辅助函数：查找内存类型索引
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)        // 内存类型（要求，位掩码），内存属性（要求，位掩码）
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();        // GPU 内存堆、内存类型（详细信息）

		// 遍历 GPU 内存类型
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)        // 理论上可用位运算优化，仅遍历要求的内存类型
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)        // 内存类型合规 && 内存属性合规
			{
				return i;        // 返回内存类型索引
			}
		}
		throw std::runtime_error("failed to find suitable memory type");
	};

	// 创建命令缓冲
	void createCommandBuffer()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};        // 命令池，主命令缓冲，两个缓冲
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);                                                                                                      // CommandBuffers() 返回命令缓冲数组
	}

	// 录制命令缓冲
	void recordCommandBuffer(uint32_t imageIndex)
	{
		// 开始录制
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});

		// 转换图像布局（交换链图：未定义->颜色附件）
		transition_image_layout(
		    swapChainImages[imageIndex],                               // 图像
		    vk::ImageLayout::eUndefined,                               // 原布局
		    vk::ImageLayout::eColorAttachmentOptimal,                  // 目标布局
		    {},                                                        // 访问标志（屏障执行前）（同步区域）（要同步缓存的操作，管线阶段->多个操作）
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // 访问标志（屏障执行后）
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // 管线阶段（屏障执行前）（同步时机）
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // 管线阶段（屏障执行后）
		    vk::ImageAspectFlagBits::eColor                            // 图像层面（颜色）
		);

		// 转换图像布局（深度图：未定义->深度附件）
		transition_image_layout(
		    *depthImage,
		    vk::ImageLayout::eUndefined,
		    vk::ImageLayout::eDepthAttachmentOptimal,
		    vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		    vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,        // 影响在屏障指令之前提交的 drawcall
		    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,        // 影响在屏障指令之后提交的 drawcall
		    vk::ImageAspectFlagBits::eDepth);

		// 清除值
		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);        // 黑色背景
		vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);                // 深度 1.0（最远）

		// 颜色附件
		vk::RenderingAttachmentInfo colorAttachmentInfo = {
		    .imageView   = swapChainImageViews[imageIndex],                 // 图像
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,        // 图像布局（图像必须为此布局）
		    .loadOp      = vk::AttachmentLoadOp::eClear,                    // 加载操作（清除）
		    .storeOp     = vk::AttachmentStoreOp::eStore,                   // 存储操作（保存）
		    .clearValue  = clearColor};

		// 深度附件
		vk::RenderingAttachmentInfo depthAttachmentInfo = {
		    .imageView   = depthImageView,
		    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,            // 加载操作（清除）
		    .storeOp     = vk::AttachmentStoreOp::eDontCare,        // 存储操作（不保存）
		    .clearValue  = clearDepth};

		// 渲染信息
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},        // 渲染区域（左上->右下）
		    .layerCount           = 1,                                                    // 视图层数（顶点->多个相机空间位置）
		    .colorAttachmentCount = 1,
		    .pColorAttachments    = &colorAttachmentInfo,        // 颜色附件
		    .pDepthAttachment     = &depthAttachmentInfo         // 深度附件
		};

		// 开始渲染（动态）
		commandBuffer.beginRendering(renderingInfo);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);                                                                                  // 绑定图形管线（着色器 + 各项固定配置）
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));        // 动态视口(0)（左上角，宽高，min，max）(深度映射：[0,1]->[min,max])（Vulkan/DirectX NDC 的 z 轴范围是 [0,1]）
		commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));                                                                                     // 动态裁剪(0)（左上角，宽高）

		commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});                        // 顶点缓冲（顶点输入绑定点 0，偏移量）(顶点输入绑定点(location)，描述符绑定点(set, binding)，不同）
		commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);        // 索引缓冲（偏移量，索引类型）（索引缓冲唯一，故无需绑定点）

		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *globalDescriptorSets[frameIndex], nullptr);          // 全局描述符集（管线，管线布局，描述符集索引(管线布局)，描述符集数据源，动态偏移量）（动态偏移量：仅影响描述符集中的动态 UBO，使不同模型切换时无需切换描述符集，更改偏移量即可）
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, *materialDescriptorSets[frameIndex], nullptr);        // 材质描述符集

		for (auto &sub : submeshes)
		{
			// 推送常量
			PushConstant pushConstant = {
			    .materialIndex = sub.materialID < 0 ? 0u : static_cast<uint32_t>(sub.materialID),        // 材质 ID
#if LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
			    .reflective = sub.reflective        // 反射标记
#endif
			};
			commandBuffer.pushConstants<PushConstant>(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pushConstant);        // 发送

			// 绘制
			commandBuffer.drawIndexed(sub.indexCount, 1, sub.indexOffset, 0, 0);        // 索引总数，实例总数，偏移量(索引)，偏移量(顶点)，偏移量(实例)
		}

		// 结束渲染（动态）
		commandBuffer.endRendering();

		// 转换图像布局（交换链图：颜色附件->呈现状态）（准备显示）
		transition_image_layout(
		    swapChainImages[imageIndex],
		    vk::ImageLayout::eColorAttachmentOptimal,
		    vk::ImageLayout::ePresentSrcKHR,
		    vk::AccessFlagBits2::eColorAttachmentWrite,
		    {},
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		    vk::PipelineStageFlagBits2::eBottomOfPipe,
		    vk::ImageAspectFlagBits::eColor);

		// 结束录制
		commandBuffer.end();
	}

	// 辅助函数：图像布局转换（主循环阶段）
	void transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags)        // 图像，原布局，目标布局，访问标志(前)，访问标志(后)，管线阶段(前），管线阶段(后)
	{
		// 图像内存屏障（屏障中，执行图像布局转换操作）
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,                 // 屏障执行前，必须完成的管线阶段
		    .srcAccessMask       = src_access_mask,                // 屏障执行前，管线阶段中必须同步缓存的操作（缓存刷出）
		    .dstStageMask        = dst_stage_mask,                 // 屏障执行后，才能开始的管线阶段
		    .dstAccessMask       = dst_access_mask,                // 屏障执行后，管线阶段中必须同步缓冲的操作（缓存失效）
		    .oldLayout           = old_layout,                     // 原布局
		    .newLayout           = new_layout,                     // 新布局
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,        // 原队列族（所有权转移，这里无）
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,        // 目标队列族
		    .image               = image,                          // 图像
		    .subresourceRange =
		        {.aspectMask = image_aspect_flags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}        // 子资源范围（图像层面，Mipmap，图像数组）
		};

		// 依赖信息
		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},        // 依赖标志（全局依赖(默认)：源阶段->目标阶段(全屏等待)；区域依赖：源阶段->目标阶段(图块等待)，片上缓存优化）
		    .imageMemoryBarrierCount = 1,
		    .pImageMemoryBarriers    = &barrier        // 图像内存屏障
		};

		// 录制屏障指令
		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
	}

	// 创建同步对象（栅栏->CPU/GPU 同步，信号量->提交同步(同/不同队列)，管线屏障->命令同步(同队列））
	void createSyncObjects()
	{
		assert(presentCompleteSemphores.empty() && renderFinishedSemphores.empty() && inFlightFences.empty());

		// 每图像
		for (size_t i = 0; i < swapChainImages.size(); i++)        // 图像渲染完成（信号量）
		{
			renderFinishedSemphores.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		// 每帧
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemphores.emplace_back(device, vk::SemaphoreCreateInfo());                                     // 图像获取完成（信号量）
			inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});        // 帧工作完成（栅栏）（初始为已触发，否则第一帧会死锁）
		}
	}

	// 更新 UBO
	void updateUniformBuffer(uint32_t currentImage)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();        // 静态变量记录开始时间

		auto  currentTime = std::chrono::high_resolution_clock::now();
		float time        = std::chrono::duration<float>(currentTime - startTime).count();        // 计算经过的时间

		UniformBufferObject ubo{};
		ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));                                                                  // 模型矩阵，随时间绕 Z 轴旋转
		ubo.view  = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));                                                     // 视图矩阵，摄像机位置 (2,2,2)，看向原点 (0,0,0)，上方向为 Z 轴
		ubo.proj  = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);        // 投影矩阵，fov，宽高比，近平面，远平面
		ubo.proj[1][1] *= -1;                                                                                                                                          // 翻转 Y 轴，GLM 是为 OpenGL 设计的（Y 轴向上），而 Vulkan 的 Clip Space Y 轴向下

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));        // 更新（GPU 可见）特殊 CPU 内存中的 UBO
	}

	// 绘制帧
	void drawFrame()
	{
		auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);        // 确保当前工作帧的上一帧的所有 GPU 工作已完成（不代表渲染结果已经被呈现）
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		device.resetFences(*inFlightFences[frameIndex]);        // 手动将栅栏重置为 Unsignaled 状态（表示当前帧工作处于未完成状态）

		auto [result, imageIndex] = swapChain.acquireNextImage(        // 向交换链请求一张空闲的画布（非空闲的画布可能被 GPU 或显示器占用，有可能正在绘制或显示）
		    UINT64_MAX,                                                // 等待时间（此处表示等待时间无限长）（三缓冲+邮箱：本质是非阻塞调用，传统垂直同步：画满了后会阻塞）
		    *presentCompleteSemphores[frameIndex],                     // 异步操作，返回后，当图片真正可用时触发信号量（返回时逻辑上交割完毕，还需等待硬件上的交割完毕）
		    nullptr                                                    // 可填写栅栏，让 CPU 也感知到图片准备好了
		);
		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		updateUniformBuffer(frameIndex);        // 更新 UBO 缓冲区

		commandBuffers[frameIndex].reset();
		recordCommandBuffer(imageIndex);        // 转为写入布局-绑定渲染目标--绘制图形-转为呈现布局

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);

		const vk::SubmitInfo submitInfo{
		    .waitSemaphoreCount   = 1,
		    .pWaitSemaphores      = &*presentCompleteSemphores[frameIndex],        // GPU 在等待哪个信号量被触发（此处为 presentCompleteSemphore）
		    .pWaitDstStageMask    = &waitDestinationStageMask,                     // GPU 在哪个流水线阶段等待（此处 GPU 可以执行顶点着色器、片元输出颜色计算，但颜色输出阶段必须停下来等待信号量被触发）
		    .commandBufferCount   = 1,
		    .pCommandBuffers      = &*commandBuffers[frameIndex],        // GPU 执行哪个命令缓冲区的命令
		    .signalSemaphoreCount = 1,
		    .pSignalSemaphores    = &*renderFinishedSemphores[imageIndex]};        // GPU 执行完命令后，触发哪个信号量（GPU）

		queue.submit(submitInfo, *inFlightFences[frameIndex]);        // 提交命令，GPU 执行完毕后触发信号量（CPU）

		try
		{
			const vk::PresentInfoKHR presentInfoKHR{
			    .waitSemaphoreCount = 1,
			    .pWaitSemaphores    = &*renderFinishedSemphores[imageIndex],        // 等待该信号量被触发（渲染完成）
			    .swapchainCount     = 1,
			    .pSwapchains        = &*swapChain,
			    .pImageIndices      = &imageIndex};        // 要展示的图片
			result = queue.presentKHR(presentInfoKHR);
			if (result == vk::Result::eSuboptimalKHR || framebufferResized)        // eSuboptimalKHR 表示交换链能用，但和当前窗口不完全匹配（如分辨率不同，但可拉伸），
			                                                                       // framebufferResized，在某些驱动上，改变窗口大小时可能仍返回 eSuccess，因为驱动通过自动缩放交换链图像以适应窗口尺寸
			{
				framebufferResized = false;
				recreateSwapChain();
			}
			else if (result != vk::Result::eSuccess)
			{
				throw std::runtime_error("failed to present swap chain image!");
			}
		}
		catch (const vk::SystemError &e)
		{
			if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR))        // 交换链彻底失效，需要重建
			{
				recreateSwapChain();
				return;
			}
			else
			{
				throw;
			}
		}
		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{
		    .codeSize = code.size() * sizeof(char),
		    .pCode    = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule shaderModule{device, createInfo};

		return shaderModule;
	}

	// 辅助函数，选择 Surface 格式
	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats)
	{
		assert(!availableFormats.empty());

		const auto formatIt = std::ranges::find_if(
		    availableFormats,
		    [](const auto &format) {
			    return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;        // eB8G8R8A8Srgb（GPU 内部） : Shader 输出颜色时 GPU 自动做 x^{1/2.2} 编码，Texture Sampler 采样时 GPU 自动做 x^{2.2} 解码
		    });                                                                                                                     // eSrgbNonlinear 显示器怎么解释这个显存数据（色域不同）

		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	// 辅助函数，选择呈现模式
	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));

		return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
		           vk::PresentModeKHR::eMailbox :        // 首选 Mailbox
		           vk::PresentModeKHR::eFifo;            // 垂直同步
	}

	// 辅助函数，选择 Swap Extent
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
	{
		if (capabilities.currentExtent.width != 0xFFFFFFFF)        // 驱动是否写死了窗口大小
		{
			return capabilities.currentExtent;        // 驱动写死了窗口大小
		}

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);        // 询问显示器要渲染的图像大小，由于显示器的缩放，glCreateWindow 的宽高参数会被缩放，缩放后才是真实要渲染的图像大小

		return {// 确保图像分辨率在显卡支持的范围内
		        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
	}

	std::vector<const char *> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		auto     glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);        // 允许注册验证层的回调函数
		}

		return extensions;
	}

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *)
	{
		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer: type" << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
		}

		return vk::False;
	}

	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);        // 从文件末尾开始，以二进制格式读取
		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file");
		}
		std::vector<char> buffer(file.tellg());                                       // 由于从文件末尾开始读，可以通过当前读指针确定缓冲区大小
		file.seekg(0, std::ios::beg);                                                 // 回到文件开头
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));        // 读文件所有数据
		file.close();
		return buffer;
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}