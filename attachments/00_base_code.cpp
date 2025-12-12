#include <memory>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

class HelloTriangeleApplication
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
	void initWindow()
	{
	}

	void initVulkan()
	{
	}

	void mainLoop()
	{
	}

	void cleanup()
	{
	}
};

int main()
{
	HelloTriangeleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}