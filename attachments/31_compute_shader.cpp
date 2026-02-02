// #pragma warning(disable : 26813)  // 屏蔽 C26813 警告: "使用‘按位与’来检查标志是否设置"

#include <algorithm>
#include <chrono>        // 用于获取高精度时间，实现平滑旋转
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>        // 用于生成粒子的随机初始位置和颜色
#include <stdexcept>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // 导入 glfwCreateWindowSurface 函数（条件编译 glfw3.h）
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS        // 强制 GLM 使用弧度制（Vulkan 和 GLM 推荐）
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr uint32_t WIDTH                = 800;
constexpr uint32_t HEIGHT               = 600;
constexpr uint32_t PARTICLE_COUNT       = 8192;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;        // 发布时关闭验证层，保证性能
#else
constexpr bool enableValidationLayers = true;
#endif

struct UniformBufferObject
{
	float deltaTime = 1.0f;
};

struct Particle
{
	glm::vec2 position;        // 位置
	glm::vec2 velocity;        // 速度
	glm::vec4 color;           // 颜色

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		return {
		    vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),           // 属性 0，位置
		    vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)),        // 属性 1，颜色
		};
	}
};

class ComputeShaderApplication
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
	vk::raii::SwapchainKHR           swapChain      = nullptr;        //
	std::vector<vk::Image>           swapChainImages;                 // 交换链中的图像
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;          // 交换链中图像格式
	vk::Extent2D                     swapChainExtent;                 // 交换链中图像分辨率
	std::vector<vk::raii::ImageView> swapChainImageViews;             // 管线通过 imageview 接口，访问交换链中的图像

	vk::raii::PipelineLayout pipelineLayout   = nullptr;        // 管线布局
	vk::raii::Pipeline       graphicsPipeline = nullptr;        // 图形管线对象

	vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      computePipelineLayout      = nullptr;
	vk::raii::Pipeline            computePipeline            = nullptr;

	std::vector<vk::raii::Buffer>       shaderStorageBuffers;              // 着色器存储缓冲区 (SSBO)，用于存储粒子数据
	std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;        //

	std::vector<vk::raii::Buffer>       uniformBuffers;              // 统一缓冲区句柄
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;        // 统一缓冲区内存对象
	std::vector<void *>                 uniformBuffersMapped;        // 持久映射指针（避免频繁调用 map/unmap）

	vk::raii::DescriptorPool             descriptorPool = nullptr;        // 描述符池
	std::vector<vk::raii::DescriptorSet> computeDescriptorSets;           // 描述符集

	vk::raii::CommandPool                commandPool = nullptr;        // 命令池，用于分配命令缓冲
	std::vector<vk::raii::CommandBuffer> commandBuffers;               // 命令缓冲，用于记录绘图指令
	std::vector<vk::raii::CommandBuffer> computeCommandBuffers;        // 独立的计算命令缓冲

	vk::raii::Semaphore          semaphore     = nullptr;
	uint64_t                     timelineValue = 0;        // 时间轴信号量
	std::vector<vk::raii::Fence> inFlightFences;           // CPU 等待 GPU 完成的栅栏
	uint32_t                     frameIndex = 0;           // 当前帧索引（0 或 1）

	double lastFrameTime = 0.0;

	bool framebufferResized = false;        // 窗口大小是否改变的标记

	double lastTime = 0.0f;

	std::vector<const char *> requiredDeviceExtension = {
	    vk::KHRSwapchainExtensionName,        // 需要的物理设备拓展
	};

	void initWindow()
	{
		glfwInit();        // 初始化 glfw 库

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);        // 不要创建 OpenGL 上下文
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);           // 禁止窗口改变大小（暂时禁止，因为这处理起来有些复杂）

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);        // 创建窗口，返回窗口指针 (宽, 高, 标题, 显示器, 共享资源)
		glfwSetWindowUserPointer(window, this);                                      // 将当前类对象指针传入 window
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		lastTime = glfwGetTime();
	}

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto app                = reinterpret_cast<ComputeShaderApplication *>(glfwGetWindowUserPointer(window));        // 从 window 中取出当前类对象指针
		app->framebufferResized = true;
	}

	// Vulkan 初始化
	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createComputeDescriptorSetLayout();        // 创建计算管线特定的描述符布局
		createGraphicsPipeline();
		createComputePipeline();        // 创建计算管线
		createCommandPool();
		createShaderStorageBuffers();        // 创建存放粒子的 SSBO
		createUniformBuffers();
		createDescriptorPool();
		createComputeDescriptorSets();        // 分配计算描述符集合
		createCommandBuffers();
		createComputeCommandBuffers();        // 分配计算命令缓冲
		createSyncObjects();
	}

	// 主循环
	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();        // 取出上一帧积压的输入（操作系统用事件队列保存上一帧积压的输入事件）
			drawFrame();
			double currentTime = glfwGetTime();
			lastFrameTime      = (currentTime - lastTime) * 1000.0;
			lastTime           = currentTime;
		}
		device.waitIdle();        // 避免在 GPU 结束工作前关闭窗口，释放显存资源，导致 GPU 非法访问释放的资源，进而驱动崩溃
	}

	void cleanupSwapChain()
	{
		swapChainImageViews.clear();        // 清空旧的 imageView
		swapChain = nullptr;                // 通过 RAII 销毁旧的交换链
	}

	void cleanup()
	{
		glfwDestroyWindow(window);        // 销毁窗口

		glfwTerminate();        // 清理 glfw 资源
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

			    auto features = device.template getFeatures2<                 // 查询显卡支持的 Vulkan 特性
			        vk::PhysicalDeviceFeatures2,                              // 查询支持的 Vulkan 1.0 基础特性（链表头，Vulkan 规定第一个必须查询这个）
			        vk::PhysicalDeviceVulkan13Features,                       // 查询支持的 Vulkan 1.3 新特性（看是否支持动态渲染）
			        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,        // 查询动态渲染状态特性（扩展特性）
			        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();        // 查询是否支持 Timeline Semaphore

			    bool supporsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&        // 必须支持各向异性过滤
			                                   features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
			                                   features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
			                                   features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore;        // 必须支持时间轴信号量

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

	// 创建逻辑设备
	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)        // 遍历查找同时同时支持图形和显示的队列族
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			    (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
			    physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
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
		vk::StructureChain<vk::PhysicalDeviceFeatures2,
		                   vk::PhysicalDeviceVulkan11Features,
		                   vk::PhysicalDeviceVulkan13Features,
		                   vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		                   vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>
		    featureChain = {
		        {.features = {.samplerAnisotropy = true}},                   // 请求开启各项异性过滤
		        {.shaderDrawParameters = true},                              // 获取 gl_BaseVertex 访问能力（shader 中的 SV_VertexID = gl_VertexIndex - gl_BaseVertex，仅 gl_VertexIndex 常驻）
		        {.synchronization2 = true, .dynamicRendering = true},        // 开启动态渲染
		        {.extendedDynamicState = true},                              // 开启扩展动态状态
		        {.timelineSemaphore = true}                                  // 开启 Timeline Semaphore 特性
		    };

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

	// 创建交换链
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

	// 创建计算管线的描述符布局
	void createComputeDescriptorSetLayout()
	{
		std::array bindings = {
		    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),         // 绑定 0，UBO（DeltaTime）
		    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),         // 绑定 1，SSBO（上一帧粒子数据）
		    vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)};        // 绑定 2，SSBO（当前帧粒子数据）

		// 创建布局信息结构体
		vk::DescriptorSetLayoutCreateInfo layoutInfo{
		    .bindingCount = static_cast<uint32_t>(bindings.size()),
		    .pBindings    = bindings.data()};

		computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);        // 创建描述符集布局（一个描述符集布局可以有多个绑定点，每个绑定点可以绑定多个同类型的描述符）
	}

	// 创建图形管线
	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto bindingDescription    = Particle::getBindingDescription();           // 绑定点，顶点步长，顶点更新频率
		auto attributeDescriptions = Particle::getAttributeDescriptions();        // 一个顶点中有多个属性（属性描述）

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		    .vertexBindingDescriptionCount   = 1,
		    .pVertexBindingDescriptions      = &bindingDescription,        // 绑定描述指针（一个绑定点对应一个缓冲区）
		    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		    .pVertexAttributeDescriptions    = attributeDescriptions.data()        // 属性描述指针
		};

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		    .topology               = vk::PrimitiveTopology::ePointList,        // 输入装配，将每个顶点视为一个独立的点
		    .primitiveRestartEnable = false                                     // 禁用用图元重启功能
		};
		vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};        // 仅指定数量，不指定内容（就算指定了内容也会被忽略，因为后续其被指定为动态状态）

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
		    .rasterizationSamples = vk::SampleCountFlagBits::e1,        // MSAA
		    .sampleShadingEnable  = vk::False};

		// 颜色混合附件
		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		    .blendEnable         = vk::True,        // 开启混合
		    .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
		    .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		    .colorBlendOp        = vk::BlendOp::eAdd,        // 叠加混合，让粒子重叠时变亮
		    .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		    .dstAlphaBlendFactor = vk::BlendFactor::eZero,
		    .alphaBlendOp        = vk::BlendOp::eAdd,
		    .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

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

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
		        .pColorBlendState    = &colorBlending,          // 颜色混合状态
		        .pDynamicState       = &dynamicState,           // 动态状态（允许在 CommandBuffer 中动态修改）
		        .layout              = pipelineLayout,          // 管线布局
		        .renderPass          = nullptr                  // 动态渲染不需要 renderPass
		    },
		    // 动态渲染配置
		    {
		        .colorAttachmentCount    = 1,
		        .pColorAttachmentFormats = &swapChainSurfaceFormat.format,        // 颜色附件格式列表
		    }};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	void createComputePipeline()
	{
		vk::raii::ShaderModule            shaderModule = createShaderModule(readFile("shaders/slang.spv"));        // 加载并创建着色器模块
		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
		    .stage  = vk::ShaderStageFlagBits::eCompute,        // 计算着色器阶段
		    .module = shaderModule,                             // 绑定加载好的着色器模块
		    .pName  = "compMain"                                // 着色器的入口函数名
		};
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		    .setLayoutCount = 1,                                  // 使用 1 个描述符集布局
		    .pSetLayouts    = &*computeDescriptorSetLayout        // 计算着色器布局指针
		};
		computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);        // 创建管线布局对象

		vk::ComputePipelineCreateInfo pipelineInfo{
		    .stage  = computeShaderStageInfo,        // 绑定计算着色器阶段信息
		    .layout = *computePipelineLayout         // 绑定管线布局
		};

		computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);        // 创建计算管线对象
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
		    .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,        // 允许单独重置该命令池分配出的 CommandBuffer（复用）
		    .queueFamilyIndex = queueIndex                                                 // 该命令池分配的命令缓冲只能提交给序号为 queueIndex 的队列族
		};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	// 初始化粒子的 SSBO
	void createShaderStorageBuffers()
	{
		// 初始化粒子
		std::default_random_engine     rndEngine(static_cast<unsigned>(time(nullptr)));        // 使用当前时间作为种子，初始化随机数引擎
		std::uniform_real_distribution rndDist(0.0f, 1.0f);                                    // 定义一个 0.0 到 1.0 之间的均匀分布浮点数生成引擎

		// 在一个圆环形状内随机分布粒子
		std::vector<Particle> particles(PARTICLE_COUNT);
		for (auto &particle : particles)
		{
			float r           = 0.25f * sqrtf(rndDist(rndEngine));                          // 生成极坐标半径 r
			float theta       = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;        // 生成极坐标旋转角度 theta
			float x           = r * cosf(theta) * HEIGHT / WIDTH;                           // 将极坐标转为笛卡尔坐标系
			float y           = r * sinf(theta);
			particle.position = glm::vec2(x, y);                                                                    // 设置位置
			particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;                                              // 设置初速度
			particle.color    = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);        // 设置颜色，RGB 随机，A 固定在 1.0
		}
		vk::DeviceSize         bufferSize = sizeof(Particle) * PARTICLE_COUNT;
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(
		    bufferSize,
		    vk::BufferUsageFlagBits::eTransferSrc,
		    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,        // 内存属性，CPU 可见 | CPU 写入后 GPU 自动可见（无需手动 Flush）
		    stagingBuffer,
		    stagingBufferMemory);

		// 将粒子数据写入暂存缓冲区
		void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, particles.data(), (size_t) bufferSize);
		stagingBufferMemory.unmapMemory();

		// 清理旧的 SSBO 句柄（如果有的话）
		shaderStorageBuffers.clear();
		shaderStorageBuffersMemory.clear();

		// 创建 GPU 专用的 SSBO 并从暂存区拷贝数据
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::raii::Buffer       shaderStorageBufferTemp({});
			vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
			createBuffer(
			    bufferSize,
			    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,        // eStorageBuffer 表示 SSBO（一块普通显存，可读可写，完全可以手动模拟 VBO 的作用，因为 VBO 本质上也是普通显存，只不过 API 托管了顶点解析操作）
			    vk::MemoryPropertyFlagBits::eDeviceLocal,
			    shaderStorageBufferTemp,
			    shaderStorageBufferTempMemory);
			copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);        // 执行拷贝命令

			// 将创建好的 RAII 对象移动到 vector 容器中保存
			shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
			shaderStorageBuffersMemory.emplace_back(std::move(shaderStorageBufferTempMemory));
		}
	}

	// 创建 UBO 缓冲区
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

	// 创建描述符池
	void createDescriptorPool()
	{
		std::array poolSize{
		    vk::DescriptorPoolSize(                        // 描述符池的大小
		        vk::DescriptorType::eUniformBuffer,        // 描述符池存储的描述符的类型
		        MAX_FRAMES_IN_FLIGHT                       // 描述符池存储的描述符的数量
		        ),
		    vk::DescriptorPoolSize(
		        vk::DescriptorType::eStorageBuffer,
		        MAX_FRAMES_IN_FLIGHT * 2)        // 计算着色器每帧需要访问两个 SSBO，上帧 SSBO 作为输入粒子数据，当前帧 SSBO 作为输出粒子数据
		};

		vk::DescriptorPoolCreateInfo poolInfo{
		    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,        // 允许单独释放描述符池中的某一描述符集
		    .maxSets       = MAX_FRAMES_IN_FLIGHT,                                        // 描述符池能分配的描述符集的最大数量（因为描述符集这个容器本身也是要占显存的）
		    .poolSizeCount = static_cast<uint32_t>(poolSize.size()),                      // 描述符池的数量
		    .pPoolSizes    = poolSize.data()                                              // 每个描述符池的大小（数组指针）
		};

		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);        // 创建描述符池（描述符池不存放实际资源，描述符集相当于容器，描述符相当于指针，都不是实际资源）
	}

	// 创建计算管线的描述符集
	void createComputeDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);        // 为什么这里不加 *，而图形管线哪里加 * ？
		vk::DescriptorSetAllocateInfo        allocInfo{
		           .descriptorPool     = *descriptorPool,             // 指定从哪个描述符池中分配内存
		           .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,        // 要分配多少个描述符集
		           .pSetLayouts        = layouts.data()               // 指定每个集合使用什么布局
        };
		computeDescriptorSets.clear();
		computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)        // 遍历并行帧进行配置
		{
			// 获取描述符对应的 Buffer
			vk::DescriptorBufferInfo bufferInfo{
			    .buffer = uniformBuffers[i],                 // 帧对应的 UBO 缓冲区
			    .offset = 0,                                 // 从 Buffer 哪一位置开始读取
			    .range  = sizeof(UniformBufferObject)        // 读取多长的数据
			};

			// 绑定 1，上一帧的 SSBO（读取上一帧计算好的粒子位置）
			vk::DescriptorBufferInfo storageBufferInfoLastFrame(
			    shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT],
			    0,
			    sizeof(Particle) * PARTICLE_COUNT);

			// 绑定 2，当前帧的 SSBO（写入当前帧更新后的粒子数据）
			vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
			    shaderStorageBuffers[(i) % MAX_FRAMES_IN_FLIGHT],
			    0,
			    sizeof(Particle) * PARTICLE_COUNT);

			std::array descriptorWrites{
			    // 描述如何更新描述符（此结构一次只能更新一个绑定点）
			    vk::WriteDescriptorSet{
			        .dstSet          = computeDescriptorSets[i],                  // 要更新哪一个描述符集
			        .dstBinding      = 0,                                         // 描述符集布局绑定点
			        .dstArrayElement = 0,                                         // 从第 0 个元素开始写
			        .descriptorCount = 1,                                         // 更新 1 个描述符
			        .descriptorType  = vk::DescriptorType::eUniformBuffer,        // 描述符类型
			        .pBufferInfo     = &bufferInfo                                // 数据来源
			    },
			    vk::WriteDescriptorSet{
			        .dstSet          = computeDescriptorSets[i],
			        .dstBinding      = 1,
			        .dstArrayElement = 0,
			        .descriptorCount = 1,
			        .descriptorType  = vk::DescriptorType::eStorageBuffer,
			        .pBufferInfo     = &storageBufferInfoLastFrame},
			    vk::WriteDescriptorSet{
			        .dstSet          = computeDescriptorSets[i],
			        .dstBinding      = 2,
			        .dstArrayElement = 0,
			        .descriptorCount = 1,
			        .descriptorType  = vk::DescriptorType::eStorageBuffer,
			        .pBufferInfo     = &storageBufferInfoCurrentFrame}};

			device.updateDescriptorSets(descriptorWrites, {});        // 更新描述符集
		}
	}

	// 辅助函数，分配 Buffer 显存
	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)
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

	// 辅助函数，一次性命令缓冲（开始）
	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = *commandPool,                            // 从哪个命令池分配命令缓冲区
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

	// 辅助函数，一次性命令缓冲（结束并提交等待)
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

	// 辅助函数，Buffer 拷贝
	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
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

	// 辅助函数，查找内存类型
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

	void createCommandBuffers()
	{
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = commandPool,                             // 从哪个命令池分配命令缓冲
		    .level              = vk::CommandBufferLevel::ePrimary,        // 主要缓冲，可以直接提交给队列执行
		    .commandBufferCount = MAX_FRAMES_IN_FLIGHT                     // 分配(两个)命令缓冲
		};
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);        // CommandBuffers 函数返回的是命令缓冲数组
	}

	// 创建计算管线专用的命令缓冲
	void createComputeCommandBuffers()
	{
		computeCommandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = *commandPool,
		    .level              = vk::CommandBufferLevel::ePrimary,
		    .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
		};
		computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	// 录制命令缓冲
	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});        // 开始录制命令

		transition_image_layout(        // 转换 Swapchain 图像布局
		    swapChainImages[imageIndex],
		    vk::ImageLayout::eUndefined,                               // 不关心图像的原布局（因为不保留原内容）
		    vk::ImageLayout::eColorAttachmentOptimal,                  // 将图像布局切换为颜色附件最优布局
		    {},                                                        // 无需对源阶段地输出结果做任何同步处理（从源阶段缓存写入内存）
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // 颜色写入操作（动作）（真正参与同步的操作）（一个流水线阶段有多个操作，不是每个都要参与同步）
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // 上一颜色写入阶段（时间点）（该阶段一定在屏障前结束）（确保颜色写入结束后，才做图像内存布局转换）
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // 下一颜色写入阶段（时间点）（该阶段一定在屏障后开始）（确保图像内存布局转换结束后，才执行颜色写入）
		    vk::ImageAspectFlagBits::eColor);

		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);        // 定义清除颜色

		// 颜色附件信息
		vk::RenderingAttachmentInfo colorAttachmentInfo = {
		    .imageView   = swapChainImageViews[imageIndex],
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,
		    .storeOp     = vk::AttachmentStoreOp::eStore,
		    .clearValue  = clearColor};

		// 渲染信息
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},        // 渲染区域，从左上角（0，0）向右下渲染 extent 宽高大小的图
		    .layerCount           = 1,                                                    // 纹理层数
		    .colorAttachmentCount = 1,                                                    // 颜色附件数量
		    .pColorAttachments    = &colorAttachmentInfo,                                 // 链接颜色附件
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

		commandBuffer.bindVertexBuffers(0,                                         // 将 Buffer 绑定到管线的 0 号绑定点（管线创建时已经将 0 号绑定点解释为了顶点缓冲区）
		                                {shaderStorageBuffers[frameIndex]},        // 顶点缓冲区
		                                {0}                                        // 从 buffer 的第 0 个字节开始读
		);

		commandBuffer.draw(PARTICLE_COUNT, 1, 0, 0);

		commandBuffer.endRendering();        // 结束动态渲染

		// 转换 Swapchain 图像布局，准备显示
		transition_image_layout(
		    swapChainImages[imageIndex],
		    vk::ImageLayout::eColorAttachmentOptimal,
		    vk::ImageLayout::ePresentSrcKHR,
		    vk::AccessFlagBits2::eColorAttachmentWrite,
		    {},
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		    vk::PipelineStageFlagBits2::eBottomOfPipe,
		    vk::ImageAspectFlagBits::eColor);

		commandBuffer.end();        // 结束录制
	}

	// 在主循环阶段的屏障
	void transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags)
	{
		// 图像内存屏障
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,                 // 屏障执行前，必须完成的流水线阶段（屏障之中，执行图像的布局转换操作）
		    .srcAccessMask       = src_access_mask,                // 屏障执行前，等源流水线阶段完成后，将其缓存中需要同步的数据类型写入显存（确保可见性）
		    .dstStageMask        = dst_stage_mask,                 // 屏障执行后，才能开始的流水线阶段（阻塞）
		    .dstAccessMask       = dst_access_mask,                // 屏障执行后，目标流水线阶段缓存中的需要同步的数据设置为过期（着色器使用缓存数据时，发现数据过期会自动去显存拉取最新数据，从而完成数据同步）
		    .oldLayout           = old_layout,                     // 图像当前内存布局（内存布局就是图像像素的物理排列方式，知道内存布局才知道（x，y）对应的内存地址在哪）
		    .newLayout           = new_layout,                     // 图像转换后的内存布局
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,        // 源队列族索引（此处由于是在同一队列族内同步，所以不需要考虑图像所有权在不同队列族间的转移）（对于独占模式的图像，同一时间只能为一个队列族所占有，仅占有它的队列族才能读写，所以此处需要交接图像的所有权）
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,        // 目标队列族索引
		    .image               = image,                          // 需要同步的图像
		    .subresourceRange    =                                 // 图像的哪些部分需要同步
		    {
		        .aspectMask     = image_aspect_flags,        // 图像的哪些通道需要同步
		        .baseMipLevel   = 0,                         // Mipmap 起始层（需要同步的 Mapmap 层级范围）
		        .levelCount     = 1,                         // 从起点开始，连续选中多少层 Mipmap
		        .baseArrayLayer = 0,                         // 纹理数组起始层（需要同步的纹理数组范围）
		        .layerCount     = 1                          // 从起点开始，连续选中多少层纹理
		    }};

		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},             // 默认是全局依赖（需要等待源阶段将整个图像要同步的数据处理完），也可以选区域性依赖（移动端优化，只要源阶段将图形某位置处理完了，目标阶段就可以立即处理这个位置，无需等待源阶段将所有位置处理完）
		    .imageMemoryBarrierCount = 1,              // 图像内存屏障数量
		    .pImageMemoryBarriers    = &barrier        // 图像内存屏障（数组）起始地址
		};

		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);        // 录制屏障指令
	}

	// 录制命令缓冲
	void recordComputeCommandBuffer()
	{
		auto &commandBuffer = computeCommandBuffers[frameIndex];
		commandBuffer.reset();
		commandBuffer.begin({});
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);        // 管线绑定点，区分要把这个命令发给哪条管线（图形管线 / 计算管线 / 光线追踪管线）
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, {computeDescriptorSets[frameIndex]}, {});
		commandBuffer.dispatch(PARTICLE_COUNT / 256, 1, 1);
		commandBuffer.end();
	}

	// 创建每帧的同步对象
	// 栅栏是 CPU 与 GPU 的同步，
	// 信号量是对一次提交的修饰，用于同步不同提交（可跨队列同步，也可同队列同步）
	// 管线屏障是同队列的不同命令的同步
	void createSyncObjects()
	{
		inFlightFences.clear();

		vk::SemaphoreTypeCreateInfo semaphoreType{.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};        // 创建时间信号量
		semaphore     = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
		timelineValue = 0;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)        // 为每一帧创建同步对象
		{
			vk::FenceCreateInfo fenceInfo{};
			inFlightFences.emplace_back(device, fenceInfo);        // 某工作帧，所有工作完成标志（初始栅栏必须是已触发状态，否则会导致第一帧死锁）
		}
	}

	// 更新 UBO
	void updateUniformBuffer(uint32_t currentImage)
	{
		UniformBufferObject ubo{};
		ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;
		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));        // 更新（GPU 可见）特殊 CPU 内存中的 UBO
	}

	// 绘制帧
	void drawFrame()
	{
		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, nullptr, *inFlightFences[frameIndex]);        // 这里没有考量图像是否可用，理论上有风险
		auto fenceResult          = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);             // CPU 等待图像可用
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		device.resetFences(*inFlightFences[frameIndex]);        // 手动将栅栏重置为 Unsignaled 状态（表示当前帧工作处于未完成状态）

		uint64_t computeWaitValue    = timelineValue;             // Compute 等待的值
		uint64_t computeSignalValue  = ++timelineValue;           // Compute 完成后 Signal 的值
		uint64_t graphicsWaitValue   = computeSignalValue;        // Graphics 等待 Compute 完成
		uint64_t graphicsSignalValue = ++timelineValue;           // Graphics 完成后 Signal 的值

		updateUniformBuffer(frameIndex);        // 更新 UBO 缓冲区

		// 提交计算队列任务
		{
			recordComputeCommandBuffer();

			vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
			    .waitSemaphoreValueCount   = 1,
			    .pWaitSemaphoreValues      = &computeWaitValue,
			    .signalSemaphoreValueCount = 1,
			    .pSignalSemaphoreValues    = &computeSignalValue};

			vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};        // 需要等待的阶段

			const vk::SubmitInfo computeSubmitInfo{
			    .pNext                = &computeTimelineInfo,
			    .waitSemaphoreCount   = 1,
			    .pWaitSemaphores      = &*semaphore,
			    .pWaitDstStageMask    = waitStages,
			    .commandBufferCount   = 1,
			    .pCommandBuffers      = &*computeCommandBuffers[frameIndex],
			    .signalSemaphoreCount = 1,
			    .pSignalSemaphores    = &*semaphore};

			queue.submit(computeSubmitInfo, nullptr);        // 提交命令，这些命令完成后不触发 fence
		}

		// 提交图形队列任务
		{
			recordCommandBuffer(imageIndex);

			// Graphics 必须等待 Compute 完成（读取 Vertex Input 阶段）
			vk::PipelineStageFlags waitStages(vk::PipelineStageFlagBits::eVertexInput);

			vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
			    .waitSemaphoreValueCount   = 1,
			    .pWaitSemaphoreValues      = &graphicsWaitValue,
			    .signalSemaphoreValueCount = 1,
			    .pSignalSemaphoreValues    = &graphicsSignalValue};

			const vk::SubmitInfo graphicsSubmitInfo{
			    .pNext                = &graphicsTimelineInfo,
			    .waitSemaphoreCount   = 1,
			    .pWaitSemaphores      = &*semaphore,
			    .pWaitDstStageMask    = &waitStages,
			    .commandBufferCount   = 1,
			    .pCommandBuffers      = &*commandBuffers[frameIndex],
			    .signalSemaphoreCount = 1,
			    .pSignalSemaphores    = &*semaphore};

			queue.submit(graphicsSubmitInfo, nullptr);        // 提交命令，这些命令完成后不触发 fence

			vk::SemaphoreWaitInfo waitInfo{
			    .semaphoreCount = 1,
			    .pSemaphores    = &*semaphore,
			    .pValues        = &graphicsSignalValue};

			auto result = device.waitSemaphores(waitInfo, UINT64_MAX);        // 让 CPU 阻塞，直到 Graphics 任务完成，让信号量更新到 graphicsSignalValue
			if (result != vk::Result::eSuccess)
			{
				throw std::runtime_error("failed to wait for semaphore!");
			}

			try
			{
				const vk::PresentInfoKHR presentInfoKHR{
				    .waitSemaphoreCount = 0,
				    .pWaitSemaphores    = nullptr,
				    .swapchainCount     = 1,
				    .pSwapchains        = &*swapChain,
				    .pImageIndices      = &imageIndex};        // 要展示的图片
				result = queue.presentKHR(presentInfoKHR);
				if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)        // eSuboptimalKHR 表示交换链能用，但和当前窗口不完全匹配（如分辨率不同，但可拉伸），
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

	// 辅助函数，选择最小图像数量
	static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);        // 尝试请求至少 3 张图像

		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;        // 显卡不支持三缓冲，就只能用双缓冲
		}
		return minImageCount;
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
		ComputeShaderApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}