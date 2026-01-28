// #pragma warning(disable : 26813)  // 屏蔽 C26813 警告: "使用‘按位与’来检查标志是否设置"

#include <algorithm>
#include <chrono>        // 用于获取高精度时间，实现平滑旋转
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

#define GLM_FORCE_RADIANS        // 强制 GLM 使用弧度制（Vulkan 和 GLM 推荐）
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

const uint32_t WIDTH                = 800;
const uint32_t HEIGHT               = 600;
constexpr int  MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;        // 发布时关闭验证层，保证性能
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription()        // 绑定描述（如何读取一个顶点）
	{
		return {
		    0,                                  // 顶点输入绑定点（供渲染管线的输入装配器使用，只能绑定顶点缓冲区，一个绑定点对应一个 Buffer）
		    sizeof(Vertex),                     // 每个顶点数据的字节跨度
		    vk::VertexInputRate::eVertex        // 数据更新频率（逐顶点/逐实例（实例化））
		};
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()        // 属性描述（如何读取一个顶点中的具体属性）
	{
		return {
		    // 顶点属性的着色器位置及其数据来源
		    vk::VertexInputAttributeDescription(        // 位置属性
		        0,                                      // 该顶点属性在着色器中的 0 号位置（layout(location = 0))
		        0,                                      // 该顶点属性在 0 号顶点输入绑定点中，要从该绑定点对应的缓冲区中获取属性数据
		        vk::Format::eR32G32B32Sfloat,           // (对应 float3）
		        offsetof(Vertex, pos)                   // 自动计算 pos 成员在结构体中的偏移量
		        ),
		    vk::VertexInputAttributeDescription(        // 颜色属性
		        1,
		        0,
		        vk::Format::eR32G32B32Sfloat,        // （对应 float3）
		        offsetof(Vertex, color)),
		    vk::VertexInputAttributeDescription(
		        2,
		        0,
		        vk::Format::eR32G32Sfloat,
		        offsetof(Vertex, texCoord))};
	}
};

struct UniformBufferObject
{
	glm::mat4 model;        // 模型矩阵
	glm::mat4 view;         // 视图矩阵
	glm::mat4 proj;         // 投影矩阵
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4};

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
	GLFWwindow                      *window = nullptr;
	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR             surface        = nullptr;        // 窗口表面
	vk::raii::PhysicalDevice         physicalDevice = nullptr;        // 使用的显卡
	vk::raii::Device                 device         = nullptr;        // 逻辑设备
	uint32_t                         queueIndex     = ~0;             // 队列族索引，初始化为最大整数，作为无效值标记
	vk::raii::Queue                  queue          = nullptr;        // 队列（同时支持图形和显示）（Vulkan 规定，凡是支持图形/计算的队列族，默认强制支持传输（Transfer）操作）
	vk::raii::SwapchainKHR           swapChain      = nullptr;
	std::vector<vk::Image>           swapChainImages;               // 交换链中的图像
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;        // 交换链中图像格式
	vk::Extent2D                     swapChainExtent;               // 交换链中图像分辨率
	std::vector<vk::raii::ImageView> swapChainImageViews;           // 管线通过 imageview 接口，访问交换链中的图像

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      pipelineLayout      = nullptr;        // 管线布局
	vk::raii::Pipeline            graphicsPipeline    = nullptr;        // 图形管线对象

	vk::raii::Image        depthImage       = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView    depthImageView   = nullptr;

	vk::raii::Image        textureImage       = nullptr;        // 纹理图像句柄
	vk::raii::DeviceMemory textureImageMemory = nullptr;        // 分配给纹理图像的显存
	vk::raii::ImageView    textureImageView   = nullptr;
	vk::raii::Sampler      textureSampler     = nullptr;

	vk::raii::Buffer       vertexBuffer       = nullptr;        // 顶点缓冲区句柄（描述大小和用途）(Buffer 一定对 GPU 可见，不一定对 CPU 可见)
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;        // 顶点缓冲区内存对象（实际显存）
	vk::raii::Buffer       indexBuffer        = nullptr;        // 索引缓冲区句柄
	vk::raii::DeviceMemory indexBufferMemory  = nullptr;        // 索引缓冲区内存对象

	std::vector<vk::raii::Buffer>       uniformBuffers;              // 统一缓冲区句柄
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;        // 统一缓冲区内存对象
	std::vector<void *>                 uniformBuffersMapped;        // 持久映射指针（避免频繁调用 map/unmap）

	vk::raii::DescriptorPool             descriptorPool = nullptr;        // 描述符池
	std::vector<vk::raii::DescriptorSet> descriptorSets;                  // 描述符集

	vk::raii::CommandPool                commandPool = nullptr;        // 命令池，用于分配命令缓冲
	std::vector<vk::raii::CommandBuffer> commandBuffers;               // 命令缓冲，用于记录绘图指令

	std::vector<vk::raii::Semaphore> presentCompleteSemphores;        // 图像获取完成信号（GPU内）
	std::vector<vk::raii::Semaphore> renderFinishedSemphores;         // 渲染完成信号（GPU内）
	std::vector<vk::raii::Fence>     inFlightFences;                  // CPU 等待 GPU 完成的栅栏
	uint32_t                         frameIndex = 0;                  // 当前帧索引（0 或 1）

	bool framebufferResized = false;        // 窗口大小是否改变的标记

	std::vector<const char *> requiredDeviceExtension = {        // 需要的物理设备拓展
	    vk::KHRSwapchainExtensionName,
	    vk::KHRSpirv14ExtensionName,
	    vk::KHRSynchronization2ExtensionName,
	    vk::KHRCreateRenderpass2ExtensionName};

	void initWindow()
	{
		glfwInit();        // 初始化 glfw 库

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);        // 不要创建 OpenGL 上下文
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);           // 禁止窗口改变大小（暂时禁止，因为这处理起来有些复杂）

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);        // 创建窗口，返回窗口指针 (宽, 高, 标题, 显示器, 共享资源)
		glfwSetWindowUserPointer(window, this);                                      // 将当前类对象指针传入 window
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto app                = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));        // 从 window 中取出当前类对象指针
		app->framebufferResized = true;
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createDepthResources();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffer();
		createSyncObjects();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();        // 取出上一帧积压的输入（操作系统用事件队列保存上一帧积压的输入事件）
			drawFrame();
		}
		device.waitIdle();        // 避免在 GPU 结束工作前关闭窗口，释放显存资源，导致 GPU 非法访问释放的资源，进而驱动崩溃
	}

	void cleanup()
	{
		glfwDestroyWindow(window);        // 销毁窗口

		glfwTerminate();        // 清理 glfw 资源
	}

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

	void cleanupSwapChain()
	{
		swapChainImageViews.clear();        // 清空旧的 imageView
		swapChain = nullptr;                // 通过 RAII 销毁旧的交换链
	}

	void createInstance()
	{
		constexpr vk::ApplicationInfo appInfo{
		    .pApplicationName   = "Hello Triangle",
		    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		    .pEngineName        = "No Engine",
		    .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		    .apiVersion         = vk::ApiVersion14};

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

	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();        // 获取所有物理设备

		const auto devIter = std::ranges::find_if(
		    devices,
		    [&](auto const &device) {                                                                    // 用 auto 作为 lambda 参数类型时，相当于用模板实现一个泛型 lambda
			    bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;        // 检查是否支持 Vulkan 1.3

			    auto queueFamilies    = device.getQueueFamilyProperties();                              // 获取所有队列族
			    bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp) {        // 检查是否支持图形队列
				    return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
			    });

			    auto availExts                     = device.enumerateDeviceExtensionProperties();                                           // 获取显卡支持的所有设备拓展
			    bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availExts](auto const &reqExt) {        // 检查显卡是否支持所有需要的设备拓展
				    return std::ranges::any_of(availExts, [reqExt](auto const &availExt) {
					    return strcmp(availExt.extensionName, reqExt) == 0;
				    });
			    });

			    auto features = device.template getFeatures2<                    // 查询显卡支持的 Vulkan 特性
			        vk::PhysicalDeviceFeatures2,                                 // 查询支持的 Vulkan 1.0 基础特性（链表头，Vulkan 规定第一个必须查询这个）
			        vk::PhysicalDeviceVulkan13Features,                          // 查询支持的 Vulkan 1.3 新特性（看是否支持动态渲染）
			        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();        // 查询动态渲染状态特性（扩展特性）

			    bool supporsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&        // 是否支持各向异性过滤
			                                   features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
			                                   features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

			    // 汇总条件
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

	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)        // 遍历查找同时同时支持图形和显示的队列族
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
		{
			throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
		}

		// 配置特性链
		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
		    {.features = {.samplerAnisotropy = true}},                   // 请求开启各项异性过滤
		    {.shaderDrawParameters = true},                              // 获取 gl_BaseVertex 访问能力（shader 中的 SV_VertexID = gl_VertexIndex - gl_BaseVertex，仅 gl_VertexIndex 常驻）
		    {.synchronization2 = true, .dynamicRendering = true},        // 开启动态渲染
		    {.extendedDynamicState = true}};                             // 开启扩展动态状态

		float                     queuePriority = 0.5f;        // 队列优先级(0 ~ 1)
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{       // 队列创建信息
		                                                .queueFamilyIndex = queueIndex,
		                                                .queueCount       = 1,
		                                                .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo      deviceCreateInfo{
		         .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),        // 将特性链挂载到 pNext
		         .queueCreateInfoCount    = 1,
		         .pQueueCreateInfos       = &deviceQueueCreateInfo,
		         .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
		         .ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue  = vk::raii::Queue(device, queueIndex, 0);        // 获取图形队列族的 0 号队列
	}

	void createSwapChain()
	{
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);

		swapChainExtent        = chooseSwapExtent(surfaceCapabilities);
		swapChainSurfaceFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{
		    .surface          = *surface,                                                                         // 指定交换链连接的 SurfaceKHR 对象
		    .minImageCount    = chooseSwapMinImageCount(surfaceCapabilities),                                     // 交换链中包含的图像数量，vulkan 硬性规定最小为 2（双缓冲），一般选 3（三缓冲）
		    .imageFormat      = swapChainSurfaceFormat.format,                                                    // 图像格式
		    .imageColorSpace  = swapChainSurfaceFormat.colorSpace,                                                // 图像的色彩空间
		    .imageExtent      = swapChainExtent,                                                                  // 图像分辨率
		    .imageArrayLayers = 1,                                                                                // 每个图像包含的层数（VR 应用才设为 2，对应左右眼）
		    .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,                                         // 直接通过渲染管线将颜色画到这张图上（图像用途）（后处理的离屏渲染用 eTransferDst
		    .imageSharingMode = vk::SharingMode::eExclusive,                                                      // 独占模式（一张图像同一时间只能属于一个队列族）（图像在多个队列族之间如何共享）
		    .preTransform     = surfaceCapabilities.currentTransform,                                             // 在呈现之前对图像进行的变换（平板电脑旋转时可能用到）
		    .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,                                           // 窗口的 Alpha 通道如何与操作系统的其他窗口混合
		    .presentMode      = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface)),        // 呈现模式（eFifo/eMailbox/eImmediate)
		    .clipped          = true                                                                              // 开启裁剪（如果另一个窗口遮挡了本窗口，或本窗口有部分被移出了屏幕边缘，允许 Vulkan 丢弃那些看不见像素的渲染操作）
		};

		swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);

		swapChainImages = swapChain.getImages();
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());
		vk::ImageViewCreateInfo imageViewCreateInfo{
		    .viewType         = vk::ImageViewType::e2D,
		    .format           = swapChainSurfaceFormat.format,
		    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

		for (auto &image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	void createDescriptorSetLayout()
	{
		std::array bindings = {
		    vk::DescriptorSetLayoutBinding(
		        0,                                         // 描述符集布局绑定点（layout(binding = 0)）
		        vk::DescriptorType::eUniformBuffer,        // 描述符类型（统一缓冲区，UBO）
		        1,                                         // 描述符数量（有多个时，shader 需要以数组形式接收）（描述符相当于显存资源指针）
		        vk::ShaderStageFlagBits::eVertex,          // 仅在顶点着色器阶段使用
		        nullptr                                    // (图像采样器才需要，这里为空)
		        ),
		    vk::DescriptorSetLayoutBinding(
		        1,
		        vk::DescriptorType::eCombinedImageSampler,
		        1,
		        vk::ShaderStageFlagBits::eFragment,
		        nullptr        // 动态采样器（后续创建描述符集时，指定具体的采样器对象）
		        )};

		// 创建布局信息结构体
		vk::DescriptorSetLayoutCreateInfo layoutInfo{
		    .bindingCount = static_cast<uint32_t>(bindings.size()),
		    .pBindings    = bindings.data()};

		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);        // 创建描述符集布局（一个描述符集布局可以有多个绑定点，每个绑定点可以绑定多个同类型的描述符）
	}

	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto                                   bindingDescription    = Vertex::getBindingDescription();           // 绑定点，顶点步长，顶点更新频率
		auto                                   attributeDescriptions = Vertex::getAttributeDescriptions();        // 一个顶点中有多个属性（属性描述）
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		    .vertexBindingDescriptionCount   = 1,
		    .pVertexBindingDescriptions      = &bindingDescription,        // 绑定描述指针（一个绑定点对应一个缓冲区）
		    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		    .pVertexAttributeDescriptions    = attributeDescriptions.data()        // 属性描述指针
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};        // 输入装配
		vk::PipelineViewportStateCreateInfo      viewportState{.viewportCount = 1, .scissorCount = 1};                   // 仅指定数量，不指定内容（就算指定了内容也会被忽略，因为后续其被指定为动态状态）

		// 光栅化器
		vk::PipelineRasterizationStateCreateInfo rasterizer{
		    .depthClampEnable        = vk::False,
		    .rasterizerDiscardEnable = vk::False,
		    .polygonMode             = vk::PolygonMode::eFill,                  // 填充三角形内部（填充模式）
		    .cullMode                = vk::CullModeFlagBits::eBack,             // 剔除背面
		    .frontFace               = vk::FrontFace::eCounterClockwise,        // 逆时针为正面
		    .depthBiasEnable         = vk::False,
		    .lineWidth               = 1.0f};

		// 多重采样
		vk::PipelineMultisampleStateCreateInfo multisampling{
		    .rasterizationSamples = vk::SampleCountFlagBits::e1,        // 采样数为 1（关闭 MSAA）
		    .sampleShadingEnable  = vk::False};

		// 深度与模板状态配置
		vk::PipelineDepthStencilStateCreateInfo depthStencil{
		    .depthTestEnable       = vk::True,
		    .depthWriteEnable      = vk::True,
		    .depthCompareOp        = vk::CompareOp::eLess,
		    .depthBoundsTestEnable = vk::False,
		    .stencilTestEnable     = vk::False};

		// 颜色混合附件
		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		    .blendEnable    = vk::False,        // 关闭混合
		    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

		// 全局颜色混合设置
		vk::PipelineColorBlendStateCreateInfo colorBlending{
		    .logicOpEnable   = vk::False,
		    .attachmentCount = 1,
		    .pAttachments    = &colorBlendAttachment};

		// 动态渲染的状态
		std::vector                        dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{
		    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		    .pDynamicStates    = dynamicStates.data()};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		    .setLayoutCount         = 1,                            // 描述符集布局的数量（一个管线仅有一个管线布局，一个管线布局可以有多个描述符集布局）
		    .pSetLayouts            = &*descriptorSetLayout,        // 描述符集布局（数组指针）
		    .pushConstantRangeCount = 0                             // 推送常量数量为 0
		};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::Format depthFormat = findDepthFormat();

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		    // 标准管线配置
		    {
		        .stageCount          = 2,                       // 着色器阶段数量（顶点 + 片段 = 2）
		        .pStages             = shaderStages,            // 指向着色器阶段入口
		        .pVertexInputState   = &vertexInputInfo,        // 顶点输入状态
		        .pInputAssemblyState = &inputAssembly,          // 输入装配状态
		        .pViewportState      = &viewportState,          // 视口状态（视口+裁剪）
		        .pRasterizationState = &rasterizer,             // 光栅化状态
		        .pMultisampleState   = &multisampling,          // 多重采样状态
		        .pDepthStencilState  = &depthStencil,           // 讲深度状态绑定到管线
		        .pColorBlendState    = &colorBlending,          // 颜色混合状态
		        .pDynamicState       = &dynamicState,           // 动态状态（允许在 CommandBuffer 中动态修改）
		        .layout              = pipelineLayout,          // 管线布局
		        .renderPass          = nullptr                  // 动态渲染不需要 renderPass
		    },
		    // 动态渲染配置
		    {
		        .colorAttachmentCount    = 1,
		        .pColorAttachmentFormats = &swapChainSurfaceFormat.format,        // 颜色附件格式列表
				.depthAttachmentFormat = depthFormat
		    }};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
		    .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,        // 允许单独重置该命令池分配出的 CommandBuffer（复用）
		    .queueFamilyIndex = queueIndex                                                 // 该命令池分配的命令缓冲只能提交给序号为 queueIndex 的队列族
		};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	void createTextureImage()
	{
		// 使用 STB 库加载图像数据
		int            texWidth, texHeight, texChannels;
		stbi_uc       *pixels    = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);        // STBI_rgb_alpha 表示强制加载 alpha 通道，即使原图没有
		vk::DeviceSize imageSize = texWidth * texHeight * 4;
		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		// 创建暂存缓冲区
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		// 将纹理数据拷贝到暂存缓冲区
		void *data = stagingBufferMemory.mapMemory(0, imageSize);        // 从 0 开始
		memcpy(data, pixels, imageSize);
		stagingBufferMemory.unmapMemory();
		stbi_image_free(pixels);        // 释放原始纹理数组

		// 创建显存上的的纹理图像
		createImage(texWidth,
		            texHeight,
		            vk::Format::eR8G8B8A8Srgb,
		            vk::ImageTiling::eOptimal,
		            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		            vk::MemoryPropertyFlagBits::eDeviceLocal,
		            textureImage,
		            textureImageMemory);

		// 图像布局转换与复制
		transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);        // 将图像改为适合 GPU 拷贝引擎高效写入的格式
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);        // 将图像改为适合 Shader 高效读取的格式
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);        // 创建纹理图像的视图（着色器必须通过 ImageView 来访问 Image
	}

	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();        // 获取物理设备属性

		vk::SamplerCreateInfo samplerInfo{
		    .magFilter        = vk::Filter::eLinear,                           // 纹理放大过滤器，线性插值（一个纹理像素覆盖多个屏幕像素）
		    .minFilter        = vk::Filter::eLinear,                           // 纹理缩小过滤器，线性插值（一个屏幕像素覆盖多个纹理像素）
		    .mipmapMode       = vk::SamplerMipmapMode::eLinear,                // Mipmap 模式（如何在不同的 Mipmap 层级之间插值）
		    .addressModeU     = vk::SamplerAddressMode::eRepeat,               // U 轴超出范围时，重复纹理（平铺，从头开始重复，类似铺地砖）
		    .addressModeV     = vk::SamplerAddressMode::eRepeat,               // V 轴超出范围时，重复纹理
		    .addressModeW     = vk::SamplerAddressMode::eRepeat,               // W 轴超出范围时，重复纹理
		    .mipLodBias       = 0.0f,                                          // Mipmap 级别偏移量
		    .anisotropyEnable = vk::True,                                      // 启用各项异性过滤（解决倾斜观察时的模糊问题）（纹理根据长轴来决定 mipmap 层级，短轴使用的 mipmap 级别过高导致模糊）(如果用短轴来决定 mipmap 会有更严重的闪烁）
		    .maxAnisotropy    = properties.limits.maxSamplerAnisotropy,        // 使用设备支持的最大各项异性级别
		    .compareEnable    = vk::False,                                     // 禁用比较操作（说明这不是阴影贴图)
		    .compareOp        = vk::CompareOp::eAlways};

		textureSampler = vk::raii::Sampler(device, samplerInfo);
	}

	vk::raii::ImageView createImageView(vk::raii::Image &image, vk::Format format)
	{
		vk::ImageViewCreateInfo viewInfo{
		    .image            = image,                         // 要为哪个 Image 对象创建视图
		    .viewType         = vk::ImageViewType::e2D,        // 告诉 GPU 将此数据视为 2D 纹理
		    .format           = format,                        // 指定数据的解释方式（通常与 Image 格式一致）（自动 Gamma 校正）（D32_SFLOAT_S8_UINT 格式下，需要对深度与模板缓冲做格式分离，一个是浮点，一个是 UINT 格式）
		    .subresourceRange = {
		        // 视图可以看到图像的哪些部分
		        vk::ImageAspectFlagBits::eColor,        // 访问颜色分量
		        0,                                      // mipmap 层级，从 0 开始，共 1 层
		        1,
		        0,        // 数组层级， 从 0 开始，共 1 层
		        1,
		    }};
		return vk::raii::ImageView(device, viewInfo);
	}

	void createImage(uint32_t                width,              // 图像宽度
	                 uint32_t                height,             // 图像高度
	                 vk::Format              format,             // 图像格式
	                 vk::ImageTiling         tiling,             // 图像数据的内存排列模式（ImageTiling 在图像创建后不可更改，ImageLayout 在图像创建后可更改）
	                 vk::ImageUsageFlags     usage,              // 图像的用途标志位
	                 vk::MemoryPropertyFlags properties,         // 所需的内存属性
	                 vk::raii::Image        &image,              // 图像对象（传出参数）
	                 vk::raii::DeviceMemory &imageMemory)        // 图像显存对象（传出参数）
	{
		vk::ImageCreateInfo imageInfo{
		    .imageType   = vk::ImageType::e2D,                 // 图像类型，1D/2D/3D
		    .format      = format,                             // 像素格式，指定颜色通道的排列和大小
		    .extent      = {width, height, 1},                 // 图像范围（宽，高，深），2D 图像的深度是 1
		    .mipLevels   = 1,                                  // MIP 贴图级别数量，这里 1 表示没有生成 MIP 链
		    .arrayLayers = 1,                                  // 纹理数组层数，这里 1 表示不是纹理数组
		    .samples     = vk::SampleCountFlagBits::e1,        // 多重采样计数，e1 表示每像素 1 个样本（不启用 MSAA）
		    .tiling      = tiling,                             // 内存平铺模式，（eLinear：线性排列，CPU 可直接读取，但在 GPU 上性能差）（eOptimal：硬件特定的优化排列，GPU 性能最佳，但 CPU 无法直接读取）
		    .usage       = usage,                              // 纹理的用途标志
		    .sharingMode = vk::SharingMode::eExclusive         // 队列族共享模式
		};

		image = vk::raii::Image(device, imageInfo);        // 创建图像句柄

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
		    .allocationSize  = memRequirements.size,
		    .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);        // 分配内存

		image.bindMemory(imageMemory, 0);        // 绑定实际显存，偏移量为 0
	}

	void transitionImageLayout(const vk::raii::Image &image,            // 需要转换布局的图像
	                           vk::ImageLayout        oldLayout,        // 图像当前布局（图像创建时，按占用内存最大的无压缩布局分配内存，运行时通常采用压缩布局以节省显存带宽，图像布局的转换不会改变图像分配的实际内存大小，只会改变有效数据的大小）
	                           vk::ImageLayout        newLayout         // 图像将要转换的布局
	)
	{
		auto commandBuffer = beginSingleTimeCommands();        // 分配一个一次性的命令缓冲

		vk::ImageMemoryBarrier barrier{
		    .oldLayout           = oldLayout,
		    .newLayout           = newLayout,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,        // 表示不涉及跨队列族的所有权转移
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = image,        // 受影响的图像
		    .subresourceRange    = {
                // 指定屏障影响图像的哪些部分
		           .aspectMask     = vk::ImageAspectFlagBits::eColor,        // 仅影响颜色分量
		           .baseMipLevel   = 0,                                      // 从第 0 层 Mipmap 开始
		           .levelCount     = 1,                                      // 仅影响 1 层 Mipmap
		           .baseArrayLayer = 0,                                      // 从第 0 层数组层开始
		           .layerCount     = 1                                       // 仅影响 1 层数组层
            }};

		vk::PipelineStageFlags sourceStage;             // 屏障执行前，必须完成的阶段（生产者）(屏障之中，是图像布局转换操作）
		vk::PipelineStageFlags destinationStage;        // 屏障执行后，才能开始的阶段（消费者）

		// 执行依赖必须有，内存依赖可以没有，内存依赖挂载于执行依赖
		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			// 内存依赖
			barrier.srcAccessMask = {};                                        // 屏障执行前，sourceStage 中的 srcAccessMask 操作必须可见（因为此情况下屏障的执行不在乎旧数据，所以无需等待 sourceStage 的任何 L1 缓存刷入 L2）
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;        // destinationStage 中的 dstAccessMask 操作执行前，屏障执行结果必须可见（保证传输写入操作可以安全进行，通过元数据缓存失效实现）

			// 执行依赖
			sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;        // 管线起点就能立马执行屏障
			destinationStage = vk::PipelineStageFlagBits::eTransfer;         // 屏障执行后，才能开始传输拷贝阶段
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;        // 屏障执行前，传输写入操作必须可见
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;           // Shader 读取前，屏障执行结果必须可见

			sourceStage      = vk::PipelineStageFlagBits::eTransfer;              // 屏障执行前，传输阶段必须完成
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;        // 屏障执行后，片段着色器才能开始执行
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		commandBuffer->pipelineBarrier(
		    sourceStage,
		    destinationStage,
		    {},
		    {},
		    nullptr,
		    barrier);

		endSingleTimeCommands(*commandBuffer);        // 结束录制并提交命令缓冲
	}

	void copyBufferToImage(const vk::raii::Buffer &buffer,        // 源数据缓冲
	                       vk::raii::Image        &image,         // 目标图像对象
	                       uint32_t                width,         // 图像宽度
	                       uint32_t                height         // 图像高度
	)
	{
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();        // beginSingleTimeCommands 用于执行那些不需要每帧都重复的短指令

		vk::BufferImageCopy region{
		    .bufferOffset      = 0,        // 缓冲区的起始字节偏移量
		    .bufferRowLength   = 0,        // 缓冲区中数据的行长（以像素为单位）（设置为 0 表示数据紧密排列，即行长等于 imageExtent.width ，若因对齐内存而有空隙，则需显式指定真实值）
		    .bufferImageHeight = 0,        // 缓冲区中图像的高度（以像素为单位）（设置为 0 表示数据紧密排列，即高度等于 imageExtent.height，若因对齐内存而有空隙，则需显式指定真实值）
		    .imageSubresource  = {
                // 指定要拷贝到的图像子资源
		         .aspectMask     = vk::ImageAspectFlagBits::eColor,        // 拷贝到图像的哪些通道
		         .mipLevel       = 0,                                      // 拷贝到 0 级 Mipmap
		         .baseArrayLayer = 0,                                      // 纹理数组的起始索引（纹理数组）
		         .layerCount     = 1,                                      // 要拷贝的层数（纹理数组）
            },
		    .imageOffset = {0, 0, 0},                // 图像中的拷贝起始坐标 (x, y, z)
		    .imageExtent = {width, height, 1}        // 图像中的拷贝尺寸 (宽, 高, 深)
		};

		commandBuffer->copyBufferToImage(
		    *buffer,                                     // 源缓冲区
		    *image,                                      // 目标图像
		    vk::ImageLayout::eTransferDstOptimal,        // 图像当前的布局（必须匹配）
		    {region}                                     // 拷贝区域列表（可以一次拷贝多个区域）
		);
		endSingleTimeCommands(*commandBuffer);
	}

	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		// 在 CPU 上创建暂存缓冲区，分配由驱动管理的特殊 CPU 内存（GPU 可见，通过 PCIe 慢速读取）
		createBuffer(bufferSize,
		             vk::BufferUsageFlagBits::eTransferSrc,                                                       // 作为传输操作的源端
		             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,        // CPU 可见，CPU 写入后立即对 GPU 可见
		             stagingBuffer,
		             stagingBufferMemory);

		// 使用映射内存（不是统一内存）
		void *data = stagingBufferMemory.mapMemory(0, bufferSize);        // 建立映射（不是对显存的映射），data 指向 GPU 可见的特殊 CPU 内存（GPU 通过 PCIe 总线读取该 CPU 内存）（修改页表以建立虚拟地址与特殊 CPU 内存的映射关系）
		memcpy(data, vertices.data(), bufferSize);                        // 将顶点数据从普通 CPU 内存拷贝到特殊 CPU 内存（不是显存，后续 GPU 的 DMA 通过 PCI-E 总线读取内存上的顶点缓冲区数据，较慢）
		stagingBufferMemory.unmapMemory();                                // 解除映射（恢复页表，终止该虚拟地址(data)与特殊 CPU 内存的映射关系，不能再通过该指针访问特殊 CPU 内存）

		// 在 GPU 上创建顶点缓冲区
		createBuffer(bufferSize,
		             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		             vk::MemoryPropertyFlagBits::eDeviceLocal,        // 分配在显存中
		             vertexBuffer,
		             vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	void createIndexBuffer()
	{
		// 计算索引数据所需的总字节数
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		// 创建暂存缓冲区
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(bufferSize,
		             vk::BufferUsageFlagBits::eTransferSrc,                                                       // 这个缓冲区将作为传输数据的“源”
		             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,        // CPU 可见，且 CPU 写入自动同步缓存，使 GPU 立即看到更新
		             stagingBuffer,
		             stagingBufferMemory);

		// 将索引数据拷贝到暂存缓冲区
		void *data = stagingBufferMemory.mapMemory(0, bufferSize);        // 从这块内存的第 0 字节开始映射，映射 bufferSize 长度
		memcpy(data, indices.data(), (size_t) bufferSize);
		stagingBufferMemory.unmapMemory();

		// 创建 GPU 专用的索引缓冲区
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}

	void createUniformBuffers()
	{
		// 清理旧数据（重建 SwapChain 时调用）
		uniformBuffers.clear();
		uniformBuffersMemory.clear();
		uniformBuffersMapped.clear();

		// 为每一并行帧创建一个独立的 Buffer
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DeviceSize         bufferSize = sizeof(UniformBufferObject);
			vk::raii::Buffer       buffer({});
			vk::raii::DeviceMemory bufferMem({});

			// 创建 UBO
			createBuffer(bufferSize,
			             vk::BufferUsageFlagBits::eUniformBuffer,
			             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,        // UBO 常驻于 CPU 内存，GPU 通过 PCIe 读取（因为 UBO 通常很小、 PCIe 带宽够用加上缓存机制，所以够用）
			             buffer,
			             bufferMem);

			uniformBuffers.emplace_back(std::move(buffer));
			uniformBuffersMemory.emplace_back(std::move(bufferMem));
			uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
		}
	}

	void createDescriptorPool()
	{
		std::array poolSize{
		    vk::DescriptorPoolSize(                        // 描述符池的大小
		        vk::DescriptorType::eUniformBuffer,        // 描述符池存储的描述符的类型
		        MAX_FRAMES_IN_FLIGHT                       // 描述符池存储的描述符的数量
		        ),
		    vk::DescriptorPoolSize(
		        vk::DescriptorType::eCombinedImageSampler,
		        MAX_FRAMES_IN_FLIGHT)};

		vk::DescriptorPoolCreateInfo poolInfo{
		    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,        // 允许单独释放描述符池中的某一描述符集
		    .maxSets       = MAX_FRAMES_IN_FLIGHT,                                        // 描述符池能分配的描述符集的最大数量（因为描述符集这个容器本身也是要占显存的）
		    .poolSizeCount = static_cast<uint32_t>(poolSize.size()),                      // 描述符池的数量
		    .pPoolSizes    = poolSize.data()                                              // 每个描述符池的大小（数组指针）
		};

		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);        // 创建描述符池（描述符池不存放实际资源，描述符集相当于容器，描述符相当于指针，都不是实际资源）
	}

	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);        // 将 descriptorSetLayout 重复 MAX_FRAMES_IN_FLIGHT 填入数组
		vk::DescriptorSetAllocateInfo        allocInfo{
		           .descriptorPool     = descriptorPool,                               // 指定从哪个描述符池中分配内存
		           .descriptorSetCount = static_cast<uint32_t>(layouts.size()),        // 要分配多少个描述符集
		           .pSetLayouts        = layouts.data()                                // 指定每个集合使用什么布局
        };

		descriptorSets.clear();
		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)        // 遍历并行帧进行配置
		{
			// 获取描述符对应的 Buffer
			vk::DescriptorBufferInfo bufferInfo{
			    .buffer = uniformBuffers[i],                 // 帧对应的 UBO 缓冲区
			    .offset = 0,                                 // 从 Buffer 哪一位置开始读取
			    .range  = sizeof(UniformBufferObject)        // 读取多长的数据
			};

			vk::DescriptorImageInfo imageInfo{
			    .sampler     = textureSampler,                                // 指定采样器
			    .imageView   = textureImageView,                              // 指定图像视图
			    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal        // 指定图像布局
			};

			std::array descriptorWrites{
			    // 描述如何更新描述符（此结构一次只能更新一个绑定点）
			    vk::WriteDescriptorSet{
			        .dstSet          = descriptorSets[i],                         // 要更新哪一个描述符集
			        .dstBinding      = 0,                                         // 描述符集布局绑定点
			        .dstArrayElement = 0,                                         // 从第 0 个元素开始写
			        .descriptorCount = 1,                                         // 更新 1 个描述符
			        .descriptorType  = vk::DescriptorType::eUniformBuffer,        // 描述符类型
			        .pBufferInfo     = &bufferInfo                                // 数据来源
			    },
			    vk::WriteDescriptorSet{
			        .dstSet          = descriptorSets[i],
			        .dstBinding      = 1,
			        .dstArrayElement = 0,
			        .descriptorCount = 1,
			        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
			        .pImageInfo      = &imageInfo}};

			device.updateDescriptorSets(descriptorWrites, {});        // 更新描述符集
		}
	}

	void createBuffer(
	    vk::DeviceSize          size,               // 缓冲区大小
	    vk::BufferUsageFlags    usage,              // 缓冲区用途（驱动要求）
	    vk::MemoryPropertyFlags properties,         // 用户所需的内存属性（用户要求）
	    vk::raii::Buffer       &buffer,             // 创建好的 RAII 缓冲区对象引用（传出参数）
	    vk::raii::DeviceMemory &bufferMemory        // 分配好的 RAII 显存对象引用（传出参数）
	)
	{
		// 缓冲区创建信息
		vk::BufferCreateInfo bufferInfo{
		    .size        = size,
		    .usage       = usage,
		    .sharingMode = vk::SharingMode::eExclusive};

		buffer = vk::raii::Buffer(device, bufferInfo);        // 创建缓冲区句柄

		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

		// 内存分配信息
		vk::MemoryAllocateInfo allocInfo{
		    .allocationSize  = memRequirements.size,                                             // 驱动要求的实际缓冲区大小，可能比我们请求的 size 略大，用于对齐（[Buffer] + [空隙])
		    .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)        // 同时满足驱动要求（memoryTypeBits）和用户要求（properties）的内存类型
		};

		// 分配显存
		bufferMemory = vk::raii::DeviceMemory(device, allocInfo);

		// 将显存绑定到缓冲区句柄，从显存的第 0 个字节开始用
		buffer.bindMemory(bufferMemory, 0);
	}

	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = commandPool,                             // 从哪个命令池分配命令缓冲区
		    .level              = vk::CommandBufferLevel::ePrimary,        // 主命令缓冲，可以直接提交给队列执行
		    .commandBufferCount = 1                                        // 仅创建一个命令缓冲
		};
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

		vk::CommandBufferBeginInfo beginInfo{
		    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit        // 告诉驱动程序，这个命令缓冲区仅会提交一次，用完就抛弃（让驱动基于这个信息做优化）
		};
		commandBuffer->begin(beginInfo);
		return commandBuffer;
	}

	void endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer)
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo{
		    .commandBufferCount = 1,                      // 提交的命令缓冲区数量
		    .pCommandBuffers    = &*commandBuffer,        // 命令缓冲区句柄指针
		};
		queue.submit(submitInfo, nullptr);        // 提交命令缓冲区
		queue.waitIdle();                         // 阻塞 CPU 线程，等待 GPU 执行完队列中的所有任务
	}

	void copyBuffer(
	    vk::raii::Buffer &srcBuffer,        // 源缓冲区（缓冲区的拷贝不涉及压缩传输，图像的拷贝才涉及压缩传输）
	    vk::raii::Buffer &dstBuffer,        // 目标缓冲区
	    vk::DeviceSize    size              // 要拷贝的字节大小
	)
	{
		// 配置命令缓冲区分配信息
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = commandPool,                             // 从哪个命令池分配（必须支持传输操作）
		    .level              = vk::CommandBufferLevel::ePrimary,        // 主命令缓冲区，可以直接提交给队列
		    .commandBufferCount = 1                                        // 仅创建一个命令缓冲区
		};

		// 分配命令缓冲区
		vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());

		// eOneTimeSubmit 是一个性能提示，告诉驱动这个命令缓冲区只会被提交一次
		commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});        // 开始录制

		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));        // 录制拷贝命令，由于缓冲区的对齐是 [Buffer] + [空隙]，所以 memcpy 前半段（这里内部结构无需对齐），后半段本就是空隙

		commandCopyBuffer.end();        // 结束录制

		queue.submit(
		    vk::SubmitInfo{
		        .commandBufferCount = 1,
		        .pCommandBuffers    = &*commandCopyBuffer},
		    nullptr);        // 提交到命令队列

		// 阻塞 CPU，直到队列中所有操作完成（确保拷贝操作完成）
		queue.waitIdle();
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)        // 根据过滤器和属性查找适合的内存类型索引
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();        // 获取显卡所有内存堆和内存类型的信息

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)        // 遍历显卡支持的内存类型（理论上可以直接用极客写法，仅遍历 buffer 要求的内存类型，但可读性和兼容性都差）
		{
			if ((typeFilter & (1 << i)) &&                                                     // 检查 Buffer 是否支持第 i 种内存类型
			    (memProperties.memoryTypes[i].propertyFlags & properties) == properties        // 检查第 i 种内存类型是否包含了我们需要的所有属性
			)
			{
				return i;        // 找到了，返回索引
			}
		}

		throw std::runtime_error("failed to find suitable memory type");
	};

	void createCommandBuffer()
	{
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = commandPool,                             // 从哪个命令池分配命令缓冲
		    .level              = vk::CommandBufferLevel::ePrimary,        // 主要缓冲，可以直接提交给队列执行
		    .commandBufferCount = MAX_FRAMES_IN_FLIGHT                     // 分配(两个)命令缓冲
		};
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);        // CommandBuffers 函数返回的是命令缓冲数组
	}

	void createDepthResources()
	{
	}

	vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlagBits features)
	{
	}

	vk::Format findDepthFormat()
	{
	}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});        // 开始录制命令

		transition_image_layout(        // 设置管线屏障，这里是对图像内存布局转化做同步
		    imageIndex,
		    vk::ImageLayout::eUndefined,                               // 不关心图像的原布局（因为不保留原内容）
		    vk::ImageLayout::eColorAttachmentOptimal,                  // 将图像布局切换为颜色附件最优布局
		    {},                                                        // 无需对源阶段地输出结果做任何同步处理（从源阶段缓存写入内存）
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // 颜色写入操作（动作）（真正参与同步的操作）（一个流水线阶段有多个操作，不是每个都要参与同步）
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // 上一颜色写入阶段（时间点）（该阶段一定在屏障前结束）（确保颜色写入结束后，才做图像内存布局转换）
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput         // 下一颜色写入阶段（时间点）（该阶段一定在屏障后开始）（确保图像内存布局转换结束后，才执行颜色写入）
		);

		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);        // 定义清除颜色

		// 颜色附件信息
		vk::RenderingAttachmentInfo attachmentInfo = {
		    .imageView   = swapChainImageViews[imageIndex],
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,
		    .storeOp     = vk::AttachmentStoreOp::eStore,        // 渲染结果要从缓存写会显存（最后阶段的深度缓冲就可以选择不写入，优化性能）（多重抗锯齿的 MSAA 原图用完也不用写回显存）
		    .clearValue  = clearColor};

		// 渲染信息
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},        // 渲染区域，从左上角（0，0）向右下渲染 extent 宽高大小的图
		    .layerCount           = 1,                                                    // 纹理层数
		    .colorAttachmentCount = 1,                                                    // 颜色附件数量
		    .pColorAttachments    = &attachmentInfo                                       // 颜色附件信息结构体
		};

		commandBuffer.beginRendering(renderingInfo);        // 开始动态渲染

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);        // 绑定图形管线（告诉 GPU 使用那套着色器和装态配置）

		commandBuffer.setViewport(0,                                                     // 第 0 号视口（Vulkan 支持同时使用多个视口，分屏游戏）
		                          vk::Viewport(                                          // 设置动态视口
		                              0.0f, 0.0f,                                        // 视口矩形左上角坐标
		                              static_cast<float>(swapChainExtent.width),         // 视口宽度
		                              static_cast<float>(swapChainExtent.height),        // 视口高度
		                              0.0f,                                              // 最小深度（Vulkan 的 NDC 空间与 DirectX 保持一致，与 OpenGL 不同）(Vulkan 的 NDC 的 z 轴范围是 [0, 1]，不再是标准立方体的 [-1, 1]）
		                              1.0f                                               // 最大深度
		                              ));

		commandBuffer.setScissor(0,                             // 对应第 0 号视口的裁剪区域
		                         vk::Rect2D(                    // 设置动态裁剪
		                             vk::Offset2D(0, 0),        // 左上角起点
		                             swapChainExtent            // 裁剪矩形宽高
		                             ));

		commandBuffer.bindVertexBuffers(0,                    // 将 Buffer 绑定到管线的 0 号绑定点（管线创建时已经将 0 号绑定点解释为了顶点缓冲区）
		                                *vertexBuffer,        // 顶点缓冲区
		                                {0}                   // 从 buffer 的第 0 个字节开始读
		);

		commandBuffer.bindIndexBuffer(*indexBuffer,        // 索引缓冲区（不需要规定绑定点，因为索引缓冲区必须唯一，而顶点缓冲区可以将不同属性拆分到多个缓冲区）
		                              0,                   // 偏移量
		                              vk::IndexTypeValue<decltype(indices)::value_type>::value);

		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,        // 绑定到图形管线（图形/计算/光线追踪）
		                                 pipelineLayout,                          // 管线布局（描述管线要求的描述符集布局）
		                                 0,                                       // 从管线的第几个描述符集开始绑定
		                                 *descriptorSets[frameIndex],             // 具体的描述符集
		                                 nullptr                                  // 动态偏移量数组（影响描述符集中的动态 UBO，使得不同模型切换时无需切换描述符集，更改偏移量即可）
		);

		commandBuffer.drawIndexed(indices.size(),        // 索引总数（这次绘制一共要读取多少索引）
		                          1,                     // 实例数量（一个模型画几次，实例化）
		                          0,                     // 首索引偏移（从索引缓冲区的哪里开始读）
		                          0,                     // 首顶点偏移（最终读取顶点 ID = 从索引缓冲拿到的值 + 这个偏移量）
		                          0                      // 首实例偏移（定义 gl_InstanceIndex 从几开始数）
		);

		commandBuffer.endRendering();        // 结束动态渲染

		// 转换图像的布局
		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    vk::ImageLayout::ePresentSrcKHR,
		    vk::AccessFlagBits2::eColorAttachmentWrite,
		    {},
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		    vk::PipelineStageFlagBits2::eBottomOfPipe);

		commandBuffer.end();        // 结束录制
	}

	void transition_image_layout(
	    uint32_t                imageIndex,             // Swapchain 中的哪一张图
	    vk::ImageLayout         old_layout,             // 初始布局
	    vk::ImageLayout         new_layout,             // 目标布局
	    vk::AccessFlags2        src_access_mask,        // 内存操作（何种读写动作）
	    vk::AccessFlags2        dst_access_mask,        // 内存操作（何种读写动作）
	    vk::PipelineStageFlags2 src_stage_mask,         // 源流水线阶段 （时间点）
	    vk::PipelineStageFlags2 dst_stage_mask          // 目标流水线阶段（时间点）
	)
	{
		// 图像内存屏障
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,                     // 屏障执行前，必须完成的流水线阶段（屏障之中，执行图像的布局转换操作）
		    .srcAccessMask       = src_access_mask,                    // 屏障执行前，等源流水线阶段完成后，将其缓存中需要同步的数据类型写入显存（确保可见性）
		    .dstStageMask        = dst_stage_mask,                     // 屏障执行后，才能开始的流水线阶段（阻塞）
		    .dstAccessMask       = dst_access_mask,                    // 屏障执行后，目标流水线阶段缓存中的需要同步的数据设置为过期（着色器使用缓存数据时，发现数据过期会自动去显存拉取最新数据，从而完成数据同步）
		    .oldLayout           = old_layout,                         // 图像当前内存布局（内存布局就是图像像素的物理排列方式，知道内存布局才知道（x，y）对应的内存地址在哪）
		    .newLayout           = new_layout,                         // 图像转换后的内存布局
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,            // 源队列族索引（此处由于是在同一队列族内同步，所以不需要考虑图像所有权在不同队列族间的转移）（对于独占模式的图像，同一时间只能为一个队列族所占有，仅占有它的队列族才能读写，所以此处需要交接图像的所有权）
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,            // 目标队列族索引
		    .image               = swapChainImages[imageIndex],        // 需要同步的图像
		    .subresourceRange    =                                     // 图像的哪些部分需要同步
		    {
		        .aspectMask     = vk::ImageAspectFlagBits::eColor,        // 图像的哪些通道需要同步
		        .baseMipLevel   = 0,                                      // Mipmap 起始层（需要同步的 Mapmap 层级范围）
		        .levelCount     = 1,                                      // 从起点开始，连续选中多少层 Mipmap
		        .baseArrayLayer = 0,                                      // 纹理数组起始层（需要同步的纹理数组范围）
		        .layerCount     = 1                                       // 从起点开始，连续选中多少层纹理
		    }};

		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},             // 默认是全局依赖（需要等待源阶段将整个图像要同步的数据处理完），也可以选区域性依赖（移动端优化，只要源阶段将图形某位置处理完了，目标阶段就可以立即处理这个位置，无需等待源阶段将所有位置处理完）
		    .imageMemoryBarrierCount = 1,              // 图像内存屏障数量
		    .pImageMemoryBarriers    = &barrier        // 图像内存屏障（数组）起始地址
		};

		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);        // 录制屏障指令
	}

	void createSyncObjects()
	{
		assert(presentCompleteSemphores.empty() && renderFinishedSemphores.empty() && inFlightFences.empty());

		for (size_t i = 0; i < swapChainImages.size(); i++)        // 为每个交换链图像创建一个渲染完成信号量
		{
			renderFinishedSemphores.emplace_back(device, vk::SemaphoreCreateInfo());        // 某图像，已渲染完且可被显示信号（二值信号量）
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)        // 为每一帧创建同步对象
		{
			presentCompleteSemphores.emplace_back(device, vk::SemaphoreCreateInfo());        // 某工作帧，图像获取完成信号（二值信号量）

			inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});        // 某工作帧，所有工作完成标志（初始栅栏必须是已触发状态，否则会导致第一帧死锁）
		}
	}

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

	static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);        // 尝试请求至少 3 张图像

		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;        // 显卡不支持三缓冲，就只能用双缓冲
		}
		return minImageCount;
	}

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

	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));

		return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
		           vk::PresentModeKHR::eMailbox :        // 首选 Mailbox
		           vk::PresentModeKHR::eFifo;            // 垂直同步
	}

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