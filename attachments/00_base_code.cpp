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
	GLFWwindow *window = nullptr;

	vk::raii::Context  context;
	vk::raii::Instance instance = nullptr;

	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

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
		vk::DebugUtilsMessageTypeFlagsEXT  messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		                                                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);        // 哪些类型的消息是需要的
		vk::DebugUtilsMessengerCreateInfoEXT  debugUtilsMessengerCreateInfoEXT{
		     .messageSeverity = severityFlags,
		     .messageType     = messageTypeFlags,
		     .pfnUserCallback = &debugCallback};
		debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
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