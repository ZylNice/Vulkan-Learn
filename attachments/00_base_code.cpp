// #pragma warning(disable : 26813)  // 屏蔽 C26813 警告: "使用‘按位与’来检查标志是否设置"

#include <algorithm>
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
#include <glm/glm.hpp>

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
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()        // 绑定描述（如何读取一个顶点）
	{
		return {
		    0,                                  // 绑定索引（binding）
		    sizeof(Vertex),                     // 每个顶点数据的字节跨度
		    vk::VertexInputRate::eVertex        // 数据更新频率（逐顶点/逐实例（实例化））
		};
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()        // 属性描述（如何读取一个顶点中的具体属性）
	{
		return {
		    vk::VertexInputAttributeDescription(        // 位置属性
		        0,                                      // 位置（location，对应着色器中的 layout(location = 0))
		        0,                                      // 绑定索引（binding，对应绑定描述）
		        vk::Format::eR32G32Sfloat,              // (对应 float2）
		        offsetof(Vertex, pos)                   // 自动计算 pos 成员在结构体中的偏移量
		        ),
		    vk::VertexInputAttributeDescription(        // 颜色属性
		        1,
		        0,
		        vk::Format::eR32G32B32Sfloat,        // （对应 float4）
		        offsetof(Vertex, color))};
	}
};

// const std::vector<Vertex> vertices = {
//     {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},         // 顶点 1: 红色
//     {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},          // 顶点 2: 绿色
//     {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};        // 顶点 3: 蓝色

const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

// const std::vector<Vertex> vertices = {
//     {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
//     {{0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
//     {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

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
	vk::raii::Queue                  queue          = nullptr;        // 队列（同时支持图形和显示）
	vk::raii::SwapchainKHR           swapChain      = nullptr;
	std::vector<vk::Image>           swapChainImages;               // 交换链中的图像
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;        // 交换链中图像格式
	vk::Extent2D                     swapChainExtent;               // 交换链中图像分辨率
	std::vector<vk::raii::ImageView> swapChainImageViews;           // 管线通过 imageview 接口，访问交换链中的图像

	vk::raii::PipelineLayout pipelineLayout   = nullptr;        // 管线布局
	vk::raii::Pipeline       graphicsPipeline = nullptr;        // 图形管线对象

	vk::raii::Buffer       vertexBuffer       = nullptr;        // 顶点缓冲区句柄（描述大小和用途）
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;        // 顶点缓冲区显存句柄（实际显存）

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
		createGraphicsPipeline();
		createCommandPool();
		createVertexBuffer();
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
		    [&](auto const &device) {
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

			    auto features                = device.template getFeatures2<                    // 查询显卡支持的 Vulkan 特性
                    vk::PhysicalDeviceFeatures2,                                 // 查询支持的 Vulkan 1.0 基础特性（链表头，Vulkan 规定第一个必须查询这个）
                    vk::PhysicalDeviceVulkan13Features,                          // 查询支持的 Vulkan 1.3 新特性（看是否支持动态渲染）
                    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();        // 查询动态渲染状态特性（扩展特性）
			    bool supporsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
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
		    {},                                                          // 基础特性
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

	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto                                   bindingDescription    = Vertex::getBindingDescription();
		auto                                   attributeDescriptions = Vertex::getAttributeDescriptions();
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
		    .polygonMode             = vk::PolygonMode::eFill,             // 填充三角形内部（填充模式）
		    .cullMode                = vk::CullModeFlagBits::eBack,        // 剔除背面
		    .frontFace               = vk::FrontFace::eClockwise,          // 顺时针为正面
		    .depthBiasEnable         = vk::False,
		    .lineWidth               = 1.0f};

		// 多重采样
		vk::PipelineMultisampleStateCreateInfo multisampling{
		    .rasterizationSamples = vk::SampleCountFlagBits::e1,        // 采样数为 1（关闭 MSAA）
		    .sampleShadingEnable  = vk::False};

		// 颜色混合附件
		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		    .blendEnable    = vk::False,        // 关闭混合
		    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

		// 全局颜色混合设置
		vk::PipelineColorBlendStateCreateInfo colorBlending{
		    .logicOpEnable   = vk::False,
		    .attachmentCount = 1,
		    .pAttachments    = &colorBlendAttachment};

		// 动态渲染
		std::vector                        dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{
		    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		    .pDynamicStates    = dynamicStates.data()};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		    .setLayoutCount         = 0,        // Set（描述符集）数量为 0，一个 Set 中可以包含多个 Binding（UBO/纹理）
		    .pushConstantRangeCount = 0         // 推送常量数量为 0
		};

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
		        .pColorAttachmentFormats = &swapChainSurfaceFormat.format        // 颜色附件格式列表
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

	void createVertexBuffer()
	{
		vk::BufferCreateInfo bufferInfo{
		    .size        = sizeof(vertices[0]) * vertices.size(),         // 缓冲区总字节数
		    .usage       = vk::BufferUsageFlagBits::eVertexBuffer,        // 用途（顶点缓冲区）
		    .sharingMode = vk::SharingMode::eExclusive                    // 共享模式（独占，同一时间只能被一个队列族所有）
		};

		vertexBuffer = vk::raii::Buffer(device, bufferInfo);        // 创建缓冲区句柄（仅创建了缓冲区的“元数据”对象，未分配实际显存）

		vk::MemoryRequirements memRequirements = vertexBuffer.getMemoryRequirements();        // 缓冲区在显存/内存中（考虑内存对齐后）的真实大小、可用的内存类型（由显卡决定，一定能得是显卡有的）（缓冲区数据在内存和显存中的二进制格式是完全相同的，所以内存类型的限制往往是硬件上的制约）

		vk::MemoryAllocateInfo memoryAllocateInfo        // 查找并分配显存
		    {
		        .allocationSize  = memRequirements.size,        // 缓冲区实际要分配的字节数（由于内存对齐，可能比 bufferInfo.size 要大)
		        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
		                                          vk::MemoryPropertyFlagBits::eHostVisible |           // 这块内存/显存可被 CPU(Host) 访问（对内存类型的额外要求）
		                                              vk::MemoryPropertyFlagBits::eHostCoherent        // CPU 写入后会自动将缓存同步到内存/显存，确保 GPU 能看到（对内存类型的额外要求）
		                                          )};

		vertexBufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);        // 分配实际显存

		vertexBuffer.bindMemory(*vertexBufferMemory, 0);        // 将分配的显存绑定到缓冲区句柄

		void *data = vertexBufferMemory.mapMemory(0, bufferInfo.size);        // 返回 CPU 可访问的虚拟地址指针，修改页表以建立虚拟地址到显存地址的映射关系（硬件支持）
		memcpy(data, vertices.data(), bufferInfo.size);                       // 将顶点数据拷贝到显存（通过 PCI-E 总线）
		vertexBufferMemory.unmapMemory();                                     // 恢复页表，终止虚拟地址到显存地址的映射关系
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

		commandBuffer.setViewport(0, vk::Viewport(                                          // 设置动态视口
		                                 0.0f, 0.0f,                                        // 视口矩形左上角坐标
		                                 static_cast<float>(swapChainExtent.width),         // 视口宽度
		                                 static_cast<float>(swapChainExtent.height),        // 视口高度
		                                 0.0f,                                              // 最小深度（Vulkan 的 NDC 空间与 DirectX 保持一致，与 OpenGL 不同）(Vulkan 的 NDC 的 z 轴范围是 [0, 1]，不再是标准立方体的 [-1, 1]）
		                                 1.0f                                               // 最大深度
		                                 ));

		commandBuffer.setScissor(0, vk::Rect2D(                    // 设置动态裁剪
		                                vk::Offset2D(0, 0),        // 左上角起点
		                                swapChainExtent            // 裁剪矩形宽高
		                                ));

		commandBuffer.bindVertexBuffers(
		    0,        // buffer 的 0 号绑定点（binding）
		    *vertexBuffer,
		    {0}        // 从 buffer 的第 0 个字节开始读
		);

		// commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.draw(vertices.size(), 1, 0, 0);

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
		    .srcStageMask        = src_stage_mask,                     // 屏障之前，必须完成的流水线阶段
		    .srcAccessMask       = src_access_mask,                    // 屏障之前，等源流水线阶段完成后，将其缓存中需要同步的数据类型写入显存（确保可见性）
		    .dstStageMask        = dst_stage_mask,                     // 屏障之后，必须等待的流水线阶段（阻塞）
		    .dstAccessMask       = dst_access_mask,                    // 屏障之后，目标流水线阶段缓存中的需要同步的数据设置为过期（着色器使用缓存数据时，发现数据过期会自动去显存拉取最新数据，从而完成数据同步）
		    .oldLayout           = old_layout,                         // 图像当前内存布局（内存布局就是图像像素的物理排列方式，知道内存布局才知道（x，y）对应的内存地址在哪）
		    .newLayout           = new_layout,                         // 图像转换后的内存布局
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,            // 源队列族索引（此处由于是在同一队列族内同步，所以不需要考虑图像所有权在不同队列族间的转移）（对于独占模式的图像，同一时间只能为一个队列族所占有，仅占有它的队列族才能读写，所以此处需要交接图像的所有权）
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,            // 目标队列族索引
		    .image               = swapChainImages[imageIndex],        // 需要同步的图像
		    .subresourceRange    =                                     // 图像的哪些部分需要同步
		    {
		        .aspectMask     = vk::ImageAspectFlagBits::eColor,        // 图像的哪些图层需要同步
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

		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
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