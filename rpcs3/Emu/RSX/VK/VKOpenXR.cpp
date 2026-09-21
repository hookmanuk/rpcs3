#include "stdafx.h"
#include "VKOpenXR.h"
#include "VKHelpers.h"
#include "vkutils/commands.h"
#include "vkutils/image.h"
#include "vkutils/image_helpers.h"

#define XR_NO_PROTOTYPES
#define XR_USE_GRAPHICS_API_VULKAN
#include "../../../../3rdparty/OpenXR/include/openxr/openxr.h"
#include "../../../../3rdparty/OpenXR/include/openxr/openxr_platform.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <cstring>
#include <sstream>

LOG_CHANNEL(xr_log, "OpenXR");

namespace vk::xr
{
	namespace
	{
		struct eye_swapchain
		{
			XrSwapchain handle = XR_NULL_HANDLE;
			std::vector<XrSwapchainImageVulkanKHR> images;
			u32 acquired = umax;
		};

		struct state_t
		{
			void* loader = nullptr;
			PFN_xrGetInstanceProcAddr gipa = nullptr;

			XrInstance instance = XR_NULL_HANDLE;
			XrSystemId system = XR_NULL_SYSTEM_ID;
			XrSession session = XR_NULL_HANDLE;
			XrSpace space = XR_NULL_HANDLE;
			XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
			bool session_running = false;
			bool lost = false;

			std::vector<std::string> instance_exts;
			std::vector<std::string> device_exts;
			std::vector<s64> swapchain_formats;

			eye_swapchain eyes[2];
			u32 swapchain_w = 0;
			u32 swapchain_h = 0;
			VkFormat swapchain_format = VK_FORMAT_UNDEFINED;

			// Per-frame
			bool frame_begun = false;
			bool layers_ready = false;
			XrTime display_time = 0;

			// Presentation of the virtual stereo screen (metres, LOCAL space)
			f32 screen_distance = 2.0f;
			f32 screen_width = 3.0f;

#define XR_FN(name) PFN_##name name = nullptr
			XR_FN(xrCreateInstance);
			XR_FN(xrDestroyInstance);
			XR_FN(xrGetSystem);
			XR_FN(xrGetSystemProperties);
			XR_FN(xrGetVulkanInstanceExtensionsKHR);
			XR_FN(xrGetVulkanDeviceExtensionsKHR);
			XR_FN(xrGetVulkanGraphicsDeviceKHR);
			XR_FN(xrGetVulkanGraphicsRequirementsKHR);
			XR_FN(xrCreateSession);
			XR_FN(xrDestroySession);
			XR_FN(xrBeginSession);
			XR_FN(xrEndSession);
			XR_FN(xrCreateReferenceSpace);
			XR_FN(xrDestroySpace);
			XR_FN(xrPollEvent);
			XR_FN(xrWaitFrame);
			XR_FN(xrBeginFrame);
			XR_FN(xrEndFrame);
			XR_FN(xrEnumerateSwapchainFormats);
			XR_FN(xrCreateSwapchain);
			XR_FN(xrDestroySwapchain);
			XR_FN(xrEnumerateSwapchainImages);
			XR_FN(xrAcquireSwapchainImage);
			XR_FN(xrWaitSwapchainImage);
			XR_FN(xrReleaseSwapchainImage);
			XR_FN(xrResultToString);
#undef XR_FN
		};

		state_t g_xr;

		std::string result_string(XrResult result)
		{
			char buf[XR_MAX_RESULT_STRING_SIZE]{};
			if (g_xr.xrResultToString && g_xr.instance &&
				XR_SUCCEEDED(g_xr.xrResultToString(g_xr.instance, result, buf)))
			{
				return buf;
			}
			return fmt::format("XrResult(%d)", static_cast<s32>(result));
		}

		bool check(XrResult result, const char* what)
		{
			if (XR_SUCCEEDED(result))
			{
				return true;
			}
			xr_log.error("%s failed: %s", what, result_string(result));
			return false;
		}

		template <typename T>
		bool load_fn(T& fn, const char* name)
		{
			if (!XR_SUCCEEDED(g_xr.gipa(g_xr.instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&fn))) || !fn)
			{
				xr_log.error("Could not resolve %s", name);
				return false;
			}
			return true;
		}

		std::vector<std::string> split_extensions(const std::string& list)
		{
			std::vector<std::string> out;
			std::istringstream stream(list);
			for (std::string name; stream >> name;)
			{
				out.push_back(std::move(name));
			}
			return out;
		}

		std::string read_env(const char* name)
		{
#ifdef _WIN32
			char* buf = nullptr;
			usz sz = 0;
			if (_dupenv_s(&buf, &sz, name) == 0 && buf)
			{
				std::string v(buf);
				std::free(buf);
				return v;
			}
			return {};
#else
			if (const char* v = ::getenv(name)) return v;
			return {};
#endif
		}

		f32 env_float(const char* name, f32 fallback)
		{
			if (const std::string v = read_env(name); !v.empty())
			{
				char* end = nullptr;
				const f32 value = std::strtof(v.c_str(), &end);
				if (end != v.c_str() && value > 0.f)
				{
					return value;
				}
			}
			return fallback;
		}

		bool load_loader()
		{
#ifdef _WIN32
			// Prefer the copy beside rpcs3.exe, then the default search path.
			wchar_t exe_path[MAX_PATH]{};
			GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
			std::wstring path(exe_path);
			path = path.substr(0, path.find_last_of(L"\\/") + 1) + L"openxr_loader.dll";

			HMODULE module = LoadLibraryW(path.c_str());
			if (!module)
			{
				module = LoadLibraryW(L"openxr_loader.dll");
			}
			if (!module)
			{
				return false;
			}
			g_xr.loader = module;
			g_xr.gipa = reinterpret_cast<PFN_xrGetInstanceProcAddr>(GetProcAddress(module, "xrGetInstanceProcAddr"));
#else
			void* module = dlopen("libopenxr_loader.so.1", RTLD_NOW | RTLD_LOCAL);
			if (!module)
			{
				return false;
			}
			g_xr.loader = module;
			g_xr.gipa = reinterpret_cast<PFN_xrGetInstanceProcAddr>(dlsym(module, "xrGetInstanceProcAddr"));
#endif
			return g_xr.gipa != nullptr;
		}

		void destroy_swapchains()
		{
			for (auto& eye : g_xr.eyes)
			{
				if (eye.handle)
				{
					g_xr.xrDestroySwapchain(eye.handle);
				}
				eye = {};
			}
			g_xr.swapchain_w = g_xr.swapchain_h = 0;
			g_xr.swapchain_format = VK_FORMAT_UNDEFINED;
		}

		// The generated eyes hold display-referred (already gamma-encoded) bytes in
		// a UNORM image. Copying those bits unchanged into an sRGB swapchain image
		// makes the runtime decode them correctly. A blit would instead convert,
		// encoding a second time, so the formats must be copy-compatible.
		VkFormat choose_swapchain_format(VkFormat source)
		{
			const auto supported = [](VkFormat f)
			{
				for (const s64 v : g_xr.swapchain_formats)
				{
					if (v == static_cast<s64>(f)) return true;
				}
				return false;
			};

			VkFormat srgb = VK_FORMAT_UNDEFINED;
			switch (source)
			{
			case VK_FORMAT_B8G8R8A8_UNORM: srgb = VK_FORMAT_B8G8R8A8_SRGB; break;
			case VK_FORMAT_R8G8B8A8_UNORM: srgb = VK_FORMAT_R8G8B8A8_SRGB; break;
			case VK_FORMAT_B8G8R8A8_SRGB:
			case VK_FORMAT_R8G8B8A8_SRGB: srgb = source; break;
			default: break;
			}

			if (srgb != VK_FORMAT_UNDEFINED && supported(srgb)) return srgb;
			if (supported(source)) return source;
			return VK_FORMAT_UNDEFINED;
		}

		bool ensure_swapchains(u32 width, u32 height, VkFormat source_format)
		{
			const VkFormat format = choose_swapchain_format(source_format);
			if (format == VK_FORMAT_UNDEFINED)
			{
				static bool s_reported = false;
				if (!std::exchange(s_reported, true))
				{
					xr_log.error("No copy-compatible swapchain format for source VkFormat %d", static_cast<s32>(source_format));
				}
				return false;
			}

			if (g_xr.eyes[0].handle && g_xr.swapchain_w == width && g_xr.swapchain_h == height && g_xr.swapchain_format == format)
			{
				return true;
			}

			destroy_swapchains();

			for (auto& eye : g_xr.eyes)
			{
				XrSwapchainCreateInfo info{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
				info.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
				info.format = static_cast<s64>(format);
				info.sampleCount = 1;
				info.width = width;
				info.height = height;
				info.faceCount = 1;
				info.arraySize = 1;
				info.mipCount = 1;

				if (!check(g_xr.xrCreateSwapchain(g_xr.session, &info, &eye.handle), "xrCreateSwapchain"))
				{
					destroy_swapchains();
					return false;
				}

				u32 count = 0;
				g_xr.xrEnumerateSwapchainImages(eye.handle, 0, &count, nullptr);
				eye.images.assign(count, { XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR });
				if (!check(g_xr.xrEnumerateSwapchainImages(eye.handle, count, &count,
					reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())), "xrEnumerateSwapchainImages"))
				{
					destroy_swapchains();
					return false;
				}
			}

			g_xr.swapchain_w = width;
			g_xr.swapchain_h = height;
			g_xr.swapchain_format = format;
			xr_log.success("Eye swapchains %ux%u, VkFormat %d, %u images", width, height, static_cast<s32>(format), ::size32(g_xr.eyes[0].images));
			return true;
		}

		void poll_events()
		{
			XrEventDataBuffer event{ XR_TYPE_EVENT_DATA_BUFFER };
			while (g_xr.xrPollEvent(g_xr.instance, &event) == XR_SUCCESS)
			{
				switch (event.type)
				{
				case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
				{
					const auto& changed = *reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
					g_xr.session_state = changed.state;
					xr_log.notice("Session state %d", static_cast<s32>(changed.state));

					if (changed.state == XR_SESSION_STATE_READY)
					{
						XrSessionBeginInfo begin{ XR_TYPE_SESSION_BEGIN_INFO };
						begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
						vk::acquire_global_submit_lock();
						g_xr.session_running = check(g_xr.xrBeginSession(g_xr.session, &begin), "xrBeginSession");
						vk::release_global_submit_lock();
					}
					else if (changed.state == XR_SESSION_STATE_STOPPING)
					{
						vk::acquire_global_submit_lock();
						g_xr.xrEndSession(g_xr.session);
						vk::release_global_submit_lock();
						g_xr.session_running = false;
					}
					else if (changed.state == XR_SESSION_STATE_EXITING || changed.state == XR_SESSION_STATE_LOSS_PENDING)
					{
						g_xr.session_running = false;
						g_xr.lost = true;
					}
					break;
				}
				case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
					g_xr.session_running = false;
					g_xr.lost = true;
					break;
				default:
					break;
				}
				event = { XR_TYPE_EVENT_DATA_BUFFER };
			}
		}
	}

	bool prepare()
	{
		if (g_xr.instance)
		{
			return true;
		}

		if (read_env("RPCS3_OPENXR") == "0")
		{
			xr_log.notice("Disabled by RPCS3_OPENXR=0");
			return false;
		}

		if (!load_loader())
		{
			xr_log.notice("openxr_loader not found; headset output disabled");
			return false;
		}

		g_xr.instance = XR_NULL_HANDLE;
		if (!load_fn(g_xr.xrCreateInstance, "xrCreateInstance"))
		{
			return false;
		}

		const char* extensions[] = { XR_KHR_VULKAN_ENABLE_EXTENSION_NAME };
		XrInstanceCreateInfo create{ XR_TYPE_INSTANCE_CREATE_INFO };
		std::memcpy(create.applicationInfo.applicationName, "RPCS3", sizeof("RPCS3"));
		std::memcpy(create.applicationInfo.engineName, "RPCS3", sizeof("RPCS3"));
		create.applicationInfo.apiVersion = XR_API_VERSION_1_0;
		create.enabledExtensionCount = 1;
		create.enabledExtensionNames = extensions;

		if (!check(g_xr.xrCreateInstance(&create, &g_xr.instance), "xrCreateInstance"))
		{
			g_xr.instance = XR_NULL_HANDLE;
			return false;
		}

		bool ok = true;
#define XR_LOAD(name) ok = ok && load_fn(g_xr.name, #name)
		XR_LOAD(xrDestroyInstance);
		XR_LOAD(xrGetSystem);
		XR_LOAD(xrGetSystemProperties);
		XR_LOAD(xrGetVulkanInstanceExtensionsKHR);
		XR_LOAD(xrGetVulkanDeviceExtensionsKHR);
		XR_LOAD(xrGetVulkanGraphicsDeviceKHR);
		XR_LOAD(xrGetVulkanGraphicsRequirementsKHR);
		XR_LOAD(xrCreateSession);
		XR_LOAD(xrDestroySession);
		XR_LOAD(xrBeginSession);
		XR_LOAD(xrEndSession);
		XR_LOAD(xrCreateReferenceSpace);
		XR_LOAD(xrDestroySpace);
		XR_LOAD(xrPollEvent);
		XR_LOAD(xrWaitFrame);
		XR_LOAD(xrBeginFrame);
		XR_LOAD(xrEndFrame);
		XR_LOAD(xrEnumerateSwapchainFormats);
		XR_LOAD(xrCreateSwapchain);
		XR_LOAD(xrDestroySwapchain);
		XR_LOAD(xrEnumerateSwapchainImages);
		XR_LOAD(xrAcquireSwapchainImage);
		XR_LOAD(xrWaitSwapchainImage);
		XR_LOAD(xrReleaseSwapchainImage);
		XR_LOAD(xrResultToString);
#undef XR_LOAD

		if (!ok)
		{
			destroy();
			return false;
		}

		XrSystemGetInfo system_info{ XR_TYPE_SYSTEM_GET_INFO };
		system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
		if (!check(g_xr.xrGetSystem(g_xr.instance, &system_info, &g_xr.system), "xrGetSystem (is the headset connected?)"))
		{
			destroy();
			return false;
		}

		XrSystemProperties props{ XR_TYPE_SYSTEM_PROPERTIES };
		g_xr.xrGetSystemProperties(g_xr.instance, g_xr.system, &props);

		u32 size = 0;
		g_xr.xrGetVulkanInstanceExtensionsKHR(g_xr.instance, g_xr.system, 0, &size, nullptr);
		std::string list(size, '\0');
		g_xr.xrGetVulkanInstanceExtensionsKHR(g_xr.instance, g_xr.system, size, &size, list.data());
		g_xr.instance_exts = split_extensions(list.c_str());

		size = 0;
		g_xr.xrGetVulkanDeviceExtensionsKHR(g_xr.instance, g_xr.system, 0, &size, nullptr);
		list.assign(size, '\0');
		g_xr.xrGetVulkanDeviceExtensionsKHR(g_xr.instance, g_xr.system, size, &size, list.data());
		g_xr.device_exts = split_extensions(list.c_str());

		// Mandatory before xrCreateSession.
		XrGraphicsRequirementsVulkanKHR requirements{ XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR };
		check(g_xr.xrGetVulkanGraphicsRequirementsKHR(g_xr.instance, g_xr.system, &requirements), "xrGetVulkanGraphicsRequirementsKHR");

		g_xr.screen_distance = env_float("RPCS3_OPENXR_DISTANCE", 2.0f);
		g_xr.screen_width = env_float("RPCS3_OPENXR_WIDTH", 3.0f);

		xr_log.success("Headset '%s' found. Vulkan instance extensions: %u, device extensions: %u",
			props.systemName, ::size32(g_xr.instance_exts), ::size32(g_xr.device_exts));
		return true;
	}

	bool is_prepared()
	{
		return g_xr.instance != XR_NULL_HANDLE;
	}

	bool is_session_created()
	{
		return g_xr.session != XR_NULL_HANDLE;
	}

	const std::vector<std::string>& instance_extensions()
	{
		return g_xr.instance_exts;
	}

	const std::vector<std::string>& device_extensions()
	{
		return g_xr.device_exts;
	}

	VkPhysicalDevice get_physical_device(VkInstance instance)
	{
		if (!g_xr.instance)
		{
			return VK_NULL_HANDLE;
		}

		VkPhysicalDevice pdev = VK_NULL_HANDLE;
		check(g_xr.xrGetVulkanGraphicsDeviceKHR(g_xr.instance, g_xr.system, instance, &pdev), "xrGetVulkanGraphicsDeviceKHR");
		return pdev;
	}

	bool create_session(VkInstance instance, VkPhysicalDevice pdev, VkDevice device, u32 queue_family, u32 queue_index)
	{
		if (!g_xr.instance || g_xr.session)
		{
			return g_xr.session != XR_NULL_HANDLE;
		}

		XrGraphicsBindingVulkanKHR binding{ XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
		binding.instance = instance;
		binding.physicalDevice = pdev;
		binding.device = device;
		binding.queueFamilyIndex = queue_family;
		binding.queueIndex = queue_index;

		XrSessionCreateInfo create{ XR_TYPE_SESSION_CREATE_INFO };
		create.next = &binding;
		create.systemId = g_xr.system;

		vk::acquire_global_submit_lock();
		const XrResult result = g_xr.xrCreateSession(g_xr.instance, &create, &g_xr.session);
		vk::release_global_submit_lock();

		if (!check(result, "xrCreateSession"))
		{
			g_xr.session = XR_NULL_HANDLE;
			return false;
		}

		XrReferenceSpaceCreateInfo space{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
		space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
		space.poseInReferenceSpace.orientation.w = 1.f;
		if (!check(g_xr.xrCreateReferenceSpace(g_xr.session, &space, &g_xr.space), "xrCreateReferenceSpace"))
		{
			return false;
		}

		u32 count = 0;
		g_xr.xrEnumerateSwapchainFormats(g_xr.session, 0, &count, nullptr);
		g_xr.swapchain_formats.resize(count);
		g_xr.xrEnumerateSwapchainFormats(g_xr.session, count, &count, g_xr.swapchain_formats.data());

		xr_log.success("Session created (queue family %u, index %u). Virtual screen %.2fm wide at %.2fm.",
			queue_family, queue_index, g_xr.screen_width, g_xr.screen_distance);
		return true;
	}

	void destroy()
	{
		if (g_xr.instance)
		{
			destroy_swapchains();
			if (g_xr.space) g_xr.xrDestroySpace(g_xr.space);
			if (g_xr.session)
			{
				if (g_xr.session_running) g_xr.xrEndSession(g_xr.session);
				g_xr.xrDestroySession(g_xr.session);
			}
			if (g_xr.xrDestroyInstance) g_xr.xrDestroyInstance(g_xr.instance);
		}

		void* loader = g_xr.loader;
		const auto gipa = g_xr.gipa;
		g_xr = {};
		g_xr.loader = loader;
		g_xr.gipa = gipa;
	}

	bool begin_frame()
	{
		if (!g_xr.session || g_xr.lost)
		{
			return false;
		}

		poll_events();
		if (!g_xr.session_running)
		{
			return false;
		}

		XrFrameState frame_state{ XR_TYPE_FRAME_STATE };
		if (!check(g_xr.xrWaitFrame(g_xr.session, nullptr, &frame_state), "xrWaitFrame"))
		{
			return false;
		}

		vk::acquire_global_submit_lock();
		const XrResult result = g_xr.xrBeginFrame(g_xr.session, nullptr);
		vk::release_global_submit_lock();
		if (!check(result, "xrBeginFrame"))
		{
			return false;
		}

		g_xr.frame_begun = true;
		g_xr.layers_ready = false;
		g_xr.display_time = frame_state.predictedDisplayTime;
		return true;
	}

	bool record_eye_copies(const vk::command_buffer& cmd, vk::image* left, vk::image* right, u32 width, u32 height)
	{
		if (!g_xr.frame_begun || !left || !width || !height)
		{
			return false;
		}

		right = right ? right : left;
		width = std::min({ width, left->width(), right->width() });
		height = std::min({ height, left->height(), right->height() });

		if (!ensure_swapchains(width, height, left->format()))
		{
			return false;
		}

		vk::image* sources[2] = { left, right };
		const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		for (u32 i = 0; i < 2; ++i)
		{
			auto& eye = g_xr.eyes[i];

			XrSwapchainImageAcquireInfo acquire{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
			XrSwapchainImageWaitInfo wait{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
			wait.timeout = 100'000'000; // 100 ms

			vk::acquire_global_submit_lock();
			bool ok = check(g_xr.xrAcquireSwapchainImage(eye.handle, &acquire, &eye.acquired), "xrAcquireSwapchainImage");
			ok = ok && check(g_xr.xrWaitSwapchainImage(eye.handle, &wait), "xrWaitSwapchainImage");
			vk::release_global_submit_lock();

			if (!ok)
			{
				return false;
			}

			const VkImage target = eye.images[eye.acquired].image;
			vk::image* source = sources[i];

			source->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			vk::change_image_layout(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, range);

			VkImageCopy region{};
			region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			region.extent = { width, height, 1 };
			vkCmdCopyImage(cmd, source->value, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			// OpenXR requires color swapchain images to be released in this layout.
			vk::change_image_layout(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, range);
			source->pop_layout(cmd);
		}

		g_xr.layers_ready = true;
		return true;
	}

	void end_frame()
	{
		if (!g_xr.frame_begun)
		{
			return;
		}
		g_xr.frame_begun = false;

		vk::acquire_global_submit_lock();

		bool released = true;
		for (auto& eye : g_xr.eyes)
		{
			if (eye.acquired != umax)
			{
				XrSwapchainImageReleaseInfo release{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
				released = check(g_xr.xrReleaseSwapchainImage(eye.handle, &release), "xrReleaseSwapchainImage") && released;
				eye.acquired = umax;
			}
		}

		XrCompositionLayerQuad quads[2]{};
		const XrCompositionLayerBaseHeader* layers[2]{};
		u32 layer_count = 0;

		if (g_xr.layers_ready && released)
		{
			const f32 height = g_xr.screen_width * g_xr.swapchain_h / g_xr.swapchain_w;
			for (u32 i = 0; i < 2; ++i)
			{
				auto& quad = quads[i];
				quad.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
				quad.space = g_xr.space;
				quad.eyeVisibility = i == 0 ? XR_EYE_VISIBILITY_LEFT : XR_EYE_VISIBILITY_RIGHT;
				quad.subImage.swapchain = g_xr.eyes[i].handle;
				quad.subImage.imageRect = { { 0, 0 }, { static_cast<s32>(g_xr.swapchain_w), static_cast<s32>(g_xr.swapchain_h) } };
				quad.subImage.imageArrayIndex = 0;
				quad.pose.orientation.w = 1.f;
				quad.pose.position = { 0.f, 0.f, -g_xr.screen_distance };
				quad.size = { g_xr.screen_width, height };
				layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
			}
		}

		XrFrameEndInfo end{ XR_TYPE_FRAME_END_INFO };
		end.displayTime = g_xr.display_time;
		end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
		end.layerCount = layer_count;
		end.layers = layers;
		check(g_xr.xrEndFrame(g_xr.session, &end), "xrEndFrame");

		vk::release_global_submit_lock();

		static bool s_reported = false;
		if (layer_count && !std::exchange(s_reported, true))
		{
			xr_log.success("First stereo frame submitted to the headset.");
		}
	}
}
