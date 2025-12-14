// #pragma warning(disable : 26813)  // 屏蔽 C26813 警告: "使用‘按位与’来检查标志是否设置"

#include <algorithm>
#include <cstdlib>
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
#include <set>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;        // 发布时关闭验证层，保证性能
#else
constexpr bool enableValidationLayers = true;
#endif

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
	vk::raii::Queue                  queue  = nullptr;        // 队列（同时支持图形和显示）

	std::vector<const char *> requiredDeviceExtension = {        // 需要的物理设备拓展
	    vk::KHRSwapchainExtensionName,
	    vk::KHRSpirv14ExtensionName,
	    vk::KHRSynchronization2ExtensionName,
	    vk::KHRCreateRenderpass2ExtensionName};

	void initWindow()
	{
		glfwInit();        // 初始化 glfw 库

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);        // 不要创建 OpenGL 上下文
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);          // 禁止窗口改变大小（暂时禁止，因为这处理起来有些复杂）

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);        // 创建窗口，返回窗口指针 (宽, 高, 标题, 显示器, 共享资源)
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();        // 取出上一帧积压的输入（操作系统用事件队列保存上一帧积压的输入事件）
		}
	}

	void cleanup()
	{
		glfwDestroyWindow(window);        // 销毁窗口

		glfwTerminate();        // 清理 glfw 资源
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

		uint32_t queueIndex = ~0; // 队列族索引，初始化为最大整数，作为无效值标记

		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) //遍历查找同时同时支持图形和显示的队列族
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
		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
		    {},                                     // 基础特性
		    {.dynamicRendering = true},             // 开启动态渲染
		    {.extendedDynamicState = true}};        // 开启扩展动态状态

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

		device        = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue = vk::raii::Queue(device, queueIndex, 0);        // 获取图形队列族的 0 号队列
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