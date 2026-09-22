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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <sstream>
#include <thread>

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

		// One published eye pair plus how it was rendered.
		struct slot_t
		{
			VkImage image[2]{};
			VkDeviceMemory memory[2]{};
			bool pose_valid = false;
			XrQuaternionf orientation{ 0.f, 0.f, 0.f, 1.f };
			XrVector3f eye_position[2]{};
			XrFovf eye_fov[2]{};
			bool have_fov = false;
			f32 tan_half_x = 0.f;
			f32 tan_half_y = 0.f;
		};

		struct state_t
		{
			void* loader = nullptr;
			PFN_xrGetInstanceProcAddr gipa = nullptr;

			XrInstance instance = XR_NULL_HANDLE;
			XrSystemId system = XR_NULL_SYSTEM_ID;
			XrSession session = XR_NULL_HANDLE;
			XrSpace space = XR_NULL_HANDLE;
			XrSpace view_space = XR_NULL_HANDLE;
			XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
			std::atomic<bool> session_running{ false };
			std::atomic<bool> lost{ false };

			std::vector<std::string> instance_exts;
			std::vector<std::string> device_exts;
			std::vector<s64> swapchain_formats;

			eye_swapchain eyes[2];
			u32 swapchain_w = 0;
			u32 swapchain_h = 0;
			VkFormat swapchain_format = VK_FORMAT_UNDEFINED;

			// OpenXR frame thread. It alone waits on the headset's clock; the RSX
			// thread only publishes eye pairs into these slots at each guest flip.
			VkDevice device = VK_NULL_HANDLE;
			VkPhysicalDevice physical_device = VK_NULL_HANDLE;
			VkQueue queue = VK_NULL_HANDLE;
			VkCommandPool cmd_pool = VK_NULL_HANDLE;
			VkCommandBuffer cmd = VK_NULL_HANDLE;
			VkFence fence = VK_NULL_HANDLE;
			std::thread thread;
			std::atomic<bool> stop{ false };

			std::mutex slot_mutex;
			std::condition_variable slot_cv;
			u64 commit_seq = 0;  // bumped by commit_eyes(); the frame thread paces on it
			slot_t slots[3];
			u32 slot_w = 0;
			u32 slot_h = 0;
			VkFormat slot_format = VK_FORMAT_UNDEFINED;
			s32 latest = -1;   // newest committed slot
			s32 in_use = -1;   // slot the frame thread is copying from
			s32 writing = -1;  // slot the RSX thread is filling

			// Presentation of the virtual stereo screen (metres, LOCAL space)
			f32 screen_distance = 2.0f;
			f32 screen_width = 3.0f;
			// Live screen mode (set_screen, under slot_mutex): the eyes shown as a flat
			// stereo quad instead of a projection layer.
			bool screen_mode = false;
			bool screen_world = true; // LOCAL space; false follows the head (VIEW space)
			f32 screen_x = 0.f;
			f32 screen_y = 0.f;

			// Projection mode
			bool projection = true;
			f32 eye_scale = 1.f;
			f32 fov_scale = 1.f;
			bool flip_y = false;
			bool hmd_fov = true;
			bool position_tracking = true;
			std::atomic<XrTime> last_display_time{ 0 };
			f32 ipd = 0.063f;
			// Pose the pending game frame is rendered with; declared at the next flip.
			bool render_pose_valid = false;
			XrQuaternionf render_orientation{ 0.f, 0.f, 0.f, 1.f };
			XrVector3f render_eye_position[2]{};
			XrFovf render_eye_fov[2]{};

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
			XR_FN(xrLocateSpace);
			XR_FN(xrLocateViews);
			XR_FN(xrEnumerateSwapchainFormats);
			XR_FN(xrCreateSwapchain);
			XR_FN(xrDestroySwapchain);
			XR_FN(xrEnumerateSwapchainImages);
			XR_FN(xrAcquireSwapchainImage);
			XR_FN(xrWaitSwapchainImage);
			XR_FN(xrReleaseSwapchainImage);
			XR_FN(xrResultToString);
#undef XR_FN

			void reset()
			{
				this->~state_t();
				new (this) state_t();
			}
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

	namespace
	{
		void frame_thread();
		void destroy_slots();
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
		XR_LOAD(xrLocateSpace);
		XR_LOAD(xrLocateViews);
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
		g_xr.projection = read_env("RPCS3_OPENXR_MODE") != "quad";
		g_xr.eye_scale = env_float("RPCS3_OPENXR_EYE_SCALE", 1.0f);
		g_xr.fov_scale = env_float("RPCS3_OPENXR_FOV_SCALE", 1.0f);
		g_xr.flip_y = read_env("RPCS3_OPENXR_FLIP_Y") == "1";
		g_xr.hmd_fov = read_env("RPCS3_OPENXR_FOV") != "game";
		g_xr.position_tracking = read_env("RPCS3_OPENXR_POSITION") != "0";

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

	bool create_session(VkInstance instance, VkPhysicalDevice pdev, VkDevice device, VkQueue queue, u32 queue_family, u32 queue_index)
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

		space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
		if (!check(g_xr.xrCreateReferenceSpace(g_xr.session, &space, &g_xr.view_space), "xrCreateReferenceSpace(VIEW)"))
		{
			return false;
		}

		u32 count = 0;
		g_xr.xrEnumerateSwapchainFormats(g_xr.session, 0, &count, nullptr);
		g_xr.swapchain_formats.resize(count);
		g_xr.xrEnumerateSwapchainFormats(g_xr.session, count, &count, g_xr.swapchain_formats.data());

		g_xr.device = device;
		g_xr.physical_device = pdev;
		g_xr.queue = queue;

		VkCommandPoolCreateInfo pool_info{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool_info.queueFamilyIndex = queue_family;
		VkCommandBufferAllocateInfo cmd_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmd_info.commandBufferCount = 1;
		VkFenceCreateInfo fence_info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		if (vkCreateCommandPool(device, &pool_info, nullptr, &g_xr.cmd_pool) != VK_SUCCESS ||
			(cmd_info.commandPool = g_xr.cmd_pool, vkAllocateCommandBuffers(device, &cmd_info, &g_xr.cmd) != VK_SUCCESS) ||
			vkCreateFence(device, &fence_info, nullptr, &g_xr.fence) != VK_SUCCESS)
		{
			xr_log.error("Could not create the OpenXR frame thread's Vulkan objects");
			return false;
		}

		g_xr.thread = std::thread(frame_thread);

		if (g_xr.projection)
		{
			xr_log.success("Session created (queue family %u, index %u). Projection mode: eye scale %.2f, %s FOV%s.",
				queue_family, queue_index, g_xr.eye_scale,
				g_xr.hmd_fov ? "headset" : fmt::format("game x%.2f", g_xr.fov_scale), g_xr.flip_y ? ", Y flipped" : "");
		}
		else
		{
			xr_log.success("Session created (queue family %u, index %u). Virtual screen %.2fm wide at %.2fm.",
				queue_family, queue_index, g_xr.screen_width, g_xr.screen_distance);
		}
		return true;
	}

	namespace
	{
		void destroy_slots()
		{
			for (auto& slot : g_xr.slots)
			{
				for (u32 i = 0; i < 2; ++i)
				{
					if (slot.image[i]) vkDestroyImage(g_xr.device, slot.image[i], nullptr);
					if (slot.memory[i]) vkFreeMemory(g_xr.device, slot.memory[i], nullptr);
					slot.image[i] = VK_NULL_HANDLE;
					slot.memory[i] = VK_NULL_HANDLE;
				}
			}
			g_xr.slot_w = g_xr.slot_h = 0;
			g_xr.slot_format = VK_FORMAT_UNDEFINED;
			g_xr.latest = -1;
		}

		bool create_slots(u32 width, u32 height, VkFormat format)
		{
			VkPhysicalDeviceMemoryProperties memory_props{};
			vkGetPhysicalDeviceMemoryProperties(g_xr.physical_device, &memory_props);

			for (auto& slot : g_xr.slots)
			{
				for (u32 i = 0; i < 2; ++i)
				{
					VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
					info.imageType = VK_IMAGE_TYPE_2D;
					info.format = format;
					info.extent = { width, height, 1 };
					info.mipLevels = 1;
					info.arrayLayers = 1;
					info.samples = VK_SAMPLE_COUNT_1_BIT;
					info.tiling = VK_IMAGE_TILING_OPTIMAL;
					info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
					info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
					info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
					if (vkCreateImage(g_xr.device, &info, nullptr, &slot.image[i]) != VK_SUCCESS)
					{
						destroy_slots();
						return false;
					}

					VkMemoryRequirements req{};
					vkGetImageMemoryRequirements(g_xr.device, slot.image[i], &req);
					u32 type = umax;
					for (u32 t = 0; t < memory_props.memoryTypeCount; ++t)
					{
						if ((req.memoryTypeBits & (1u << t)) &&
							(memory_props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
						{
							type = t;
							break;
						}
					}

					VkMemoryAllocateInfo alloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
					alloc.allocationSize = req.size;
					alloc.memoryTypeIndex = type;
					if (type == umax || vkAllocateMemory(g_xr.device, &alloc, nullptr, &slot.memory[i]) != VK_SUCCESS ||
						vkBindImageMemory(g_xr.device, slot.image[i], slot.memory[i], 0) != VK_SUCCESS)
					{
						destroy_slots();
						return false;
					}
				}
			}

			g_xr.slot_w = width;
			g_xr.slot_h = height;
			g_xr.slot_format = format;
			return true;
		}

		void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
			VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
		{
			VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			barrier.srcAccessMask = src_access;
			barrier.dstAccessMask = dst_access;
			barrier.oldLayout = from;
			barrier.newLayout = to;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image;
			barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
		}

		// One headset frame on the OpenXR thread: present the newest published eye
		// pair (re-presenting the previous one only after the 100 ms idle timeout).
		// Diagnostics: how long this thread holds RPCS3's global submit lock per
		// call site (RPCS3's own submits wait while it is held).
		using lock_clock = std::chrono::steady_clock;
		s64 g_lock_held_us[5]{};
		const char* const g_lock_site_names[5] = { "xrBeginFrame", "acquire+wait", "vkQueueSubmit", "release", "xrEndFrame" };

		lock_clock::time_point lock_begin()
		{
			vk::acquire_global_submit_lock();
			return lock_clock::now();
		}

		void lock_end(u32 site, lock_clock::time_point start)
		{
			const auto held = lock_clock::now() - start;
			vk::release_global_submit_lock();
			g_lock_held_us[site] += std::chrono::duration_cast<std::chrono::microseconds>(held).count();
		}

		void report_lock_hold()
		{
			static lock_clock::time_point s_window = lock_clock::now();
			const auto now = lock_clock::now();
			if (now - s_window < std::chrono::seconds(1))
			{
				return;
			}
			s_window = now;

			s64 total = 0;
			for (const s64 v : g_lock_held_us) total += v;
			if (total > 100'000)
			{
				xr_log.warning("Submit lock held %.1f ms in the last second: %s %.1f, %s %.1f, %s %.1f, %s %.1f, %s %.1f",
					total / 1000.0,
					g_lock_site_names[0], g_lock_held_us[0] / 1000.0, g_lock_site_names[1], g_lock_held_us[1] / 1000.0,
					g_lock_site_names[2], g_lock_held_us[2] / 1000.0, g_lock_site_names[3], g_lock_held_us[3] / 1000.0,
					g_lock_site_names[4], g_lock_held_us[4] / 1000.0);
			}
			for (s64& v : g_lock_held_us) v = 0;
		}

		void run_frame()
		{
			XrFrameState frame_state{ XR_TYPE_FRAME_STATE };
			if (!check(g_xr.xrWaitFrame(g_xr.session, nullptr, &frame_state), "xrWaitFrame"))
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				return;
			}
			g_xr.last_display_time.store(frame_state.predictedDisplayTime);

			const auto lt0 = lock_begin();
			const XrResult begun = g_xr.xrBeginFrame(g_xr.session, nullptr);
			lock_end(0, lt0);
			if (!check(begun, "xrBeginFrame"))
			{
				return;
			}

			slot_t meta{};
			s32 index = -1;
			bool screen_mode = !g_xr.projection;
			bool screen_world = true;
			f32 screen_pos[3] = { 0.f, 0.f, -g_xr.screen_distance };
			f32 screen_width = g_xr.screen_width;
			{
				std::lock_guard lock(g_xr.slot_mutex);
				if (g_xr.screen_mode)
				{
					screen_mode = true;
					screen_world = g_xr.screen_world;
					screen_pos[0] = g_xr.screen_x;
					screen_pos[1] = g_xr.screen_y;
					screen_pos[2] = -g_xr.screen_distance;
					screen_width = g_xr.screen_width;
				}
				index = g_xr.latest;
				if (index >= 0)
				{
					g_xr.in_use = index;
					meta = g_xr.slots[index];
				}
			}

			XrCompositionLayerQuad quads[2]{};
			XrCompositionLayerProjectionView views[2]{};
			XrCompositionLayerProjection projection{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
			const XrCompositionLayerBaseHeader* layers[2]{};
			u32 layer_count = 0;

			if (index >= 0 && frame_state.shouldRender && ensure_swapchains(g_xr.slot_w, g_xr.slot_h, g_xr.slot_format))
			{
				bool ok = true;
				for (auto& eye : g_xr.eyes)
				{
					XrSwapchainImageAcquireInfo acquire{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
					XrSwapchainImageWaitInfo wait{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
					wait.timeout = 100'000'000; // 100 ms
					const auto lt1 = lock_begin();
					ok = ok && check(g_xr.xrAcquireSwapchainImage(eye.handle, &acquire, &eye.acquired), "xrAcquireSwapchainImage");
					ok = ok && check(g_xr.xrWaitSwapchainImage(eye.handle, &wait), "xrWaitSwapchainImage");
					lock_end(1, lt1);
				}

				if (ok)
				{
					VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
					begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
					vkResetCommandBuffer(g_xr.cmd, 0);
					vkBeginCommandBuffer(g_xr.cmd, &begin);
					for (u32 i = 0; i < 2; ++i)
					{
						const VkImage target = g_xr.eyes[i].images[g_xr.eyes[i].acquired].image;
						image_barrier(g_xr.cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
							0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

						VkImageCopy region{};
						region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
						region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
						region.extent = { g_xr.slot_w, g_xr.slot_h, 1 };
						vkCmdCopyImage(g_xr.cmd, meta.image[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
							target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

						// OpenXR requires color swapchain images to be released in this layout.
						image_barrier(g_xr.cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
							VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
							VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
					}
					vkEndCommandBuffer(g_xr.cmd);

					VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
					submit.commandBufferCount = 1;
					submit.pCommandBuffers = &g_xr.cmd;
					const auto lt2 = lock_begin();
					vkQueueSubmit(g_xr.queue, 1, &submit, g_xr.fence);
					lock_end(2, lt2);

					// The copy is tiny; waiting here keeps the slot and command buffer
					// lifetimes trivial and never touches the RSX thread.
					vkWaitForFences(g_xr.device, 1, &g_xr.fence, VK_TRUE, UINT64_MAX);
					vkResetFences(g_xr.device, 1, &g_xr.fence);
				}

				const auto lt3 = lock_begin();
				for (auto& eye : g_xr.eyes)
				{
					if (eye.acquired != umax)
					{
						XrSwapchainImageReleaseInfo release{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
						ok = check(g_xr.xrReleaseSwapchainImage(eye.handle, &release), "xrReleaseSwapchainImage") && ok;
						eye.acquired = umax;
					}
				}
				lock_end(3, lt3);

				if (ok && !screen_mode && meta.pose_valid && meta.have_fov)
				{
					// Declare exactly how these eyes were rendered: the head orientation
					// the camera was rotated by, the located eye positions, and the FOV
					// the draws were projected with.
					const f32 half_x = std::atan(meta.tan_half_x);
					const f32 half_y = std::atan(meta.tan_half_y);
					for (u32 i = 0; i < 2; ++i)
					{
						auto& view = views[i];
						view.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
						view.pose.orientation = meta.orientation;
						view.pose.position = meta.eye_position[i];
						view.fov = g_xr.hmd_fov ? meta.eye_fov[i] : XrFovf{ -half_x, half_x, half_y, -half_y };
						view.subImage.swapchain = g_xr.eyes[i].handle;
						view.subImage.imageRect = { { 0, 0 }, { static_cast<s32>(g_xr.swapchain_w), static_cast<s32>(g_xr.swapchain_h) } };
						view.subImage.imageArrayIndex = 0;
					}
					projection.space = g_xr.space;
					projection.viewCount = 2;
					projection.views = views;
					layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection);

					static bool s_reported_projection = false;
					if (!std::exchange(s_reported_projection, true))
					{
						const XrFovf& f = views[0].fov;
						xr_log.success("Projection layer: game FOV %.1f x %.1f degrees; rendering %.1f x %.1f degrees (%s).",
							2.f * half_x * 57.29578f, 2.f * half_y * 57.29578f,
							(f.angleRight - f.angleLeft) * 57.29578f, (f.angleUp - f.angleDown) * 57.29578f,
							g_xr.hmd_fov ? "headset" : "game");
					}
				}
				else if (ok)
				{
					const f32 height = screen_width * g_xr.swapchain_h / g_xr.swapchain_w;
					for (u32 i = 0; i < 2; ++i)
					{
						auto& quad = quads[i];
						quad.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
						quad.eyeVisibility = i == 0 ? XR_EYE_VISIBILITY_LEFT : XR_EYE_VISIBILITY_RIGHT;
						quad.subImage.swapchain = g_xr.eyes[i].handle;
						quad.subImage.imageRect = { { 0, 0 }, { static_cast<s32>(g_xr.swapchain_w), static_cast<s32>(g_xr.swapchain_h) } };
						quad.subImage.imageArrayIndex = 0;
						quad.pose.orientation.w = 1.f;
						quad.space = screen_world ? g_xr.space : g_xr.view_space;
						quad.pose.position = { screen_pos[0], screen_pos[1], screen_pos[2] };
						quad.size = { screen_width, height };
						layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
					}
				}
			}

			{
				std::lock_guard lock(g_xr.slot_mutex);
				g_xr.in_use = -1;
			}

			XrFrameEndInfo end{ XR_TYPE_FRAME_END_INFO };
			end.displayTime = frame_state.predictedDisplayTime;
			end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
			end.layerCount = layer_count;
			end.layers = layers;
			const auto lt4 = lock_begin();
			check(g_xr.xrEndFrame(g_xr.session, &end), "xrEndFrame");
			lock_end(4, lt4);

			static bool s_reported = false;
			if (layer_count && !std::exchange(s_reported, true))
			{
				xr_log.success("First stereo frame submitted to the headset (OpenXR frame thread).");
			}
		}

		void frame_thread()
		{
			u64 presented_seq = 0;
			while (!g_xr.stop.load())
			{
				poll_events();
				if (!g_xr.session_running.load() || g_xr.lost)
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					continue;
				}

				// One headset frame per new game frame. When the game runs below the
				// headset rate, the runtime then sees the missed frames and fills them
				// itself (reprojection / motion smoothing). Re-presenting stale frames
				// would hide that. The wait is on this thread, never on RSX. After
				// 100 ms without a new frame (loading, pause) re-present anyway.
				{
					std::unique_lock lock(g_xr.slot_mutex);
					g_xr.slot_cv.wait_for(lock, std::chrono::milliseconds(100),
						[&] { return g_xr.commit_seq != presented_seq || g_xr.stop.load(); });
					presented_seq = g_xr.commit_seq;
				}
				if (g_xr.stop.load())
				{
					break;
				}
				run_frame();
				report_lock_hold();
			}
		}
	}

	void destroy()
	{
		if (g_xr.thread.joinable())
		{
			g_xr.stop.store(true);
			g_xr.slot_cv.notify_all();
			g_xr.thread.join();
		}

		if (g_xr.instance)
		{
			destroy_swapchains();
			if (g_xr.view_space) g_xr.xrDestroySpace(g_xr.view_space);
			if (g_xr.space) g_xr.xrDestroySpace(g_xr.space);
			if (g_xr.session)
			{
				if (g_xr.session_running.load()) g_xr.xrEndSession(g_xr.session);
				g_xr.xrDestroySession(g_xr.session);
			}
			if (g_xr.xrDestroyInstance) g_xr.xrDestroyInstance(g_xr.instance);
		}

		if (g_xr.device)
		{
			// The frame thread is joined; drain its (and RSX's) copies before freeing.
			if (g_xr.queue)
			{
				vk::acquire_global_submit_lock();
				vkQueueWaitIdle(g_xr.queue);
				vk::release_global_submit_lock();
			}
			destroy_slots();
			if (g_xr.fence) vkDestroyFence(g_xr.device, g_xr.fence, nullptr);
			if (g_xr.cmd_pool) vkDestroyCommandPool(g_xr.device, g_xr.cmd_pool, nullptr);
		}

		void* loader = g_xr.loader;
		const auto gipa = g_xr.gipa;
		g_xr.reset();
		g_xr.loader = loader;
		g_xr.gipa = gipa;
	}

	bool is_running()
	{
		return g_xr.session && g_xr.session_running.load() && !g_xr.lost;
	}

	bool publish_eyes(const vk::command_buffer& cmd, vk::image* left, vk::image* right, u32 width, u32 height)
	{
		if (!is_running() || !left || !width || !height)
		{
			return false;
		}

		right = right ? right : left;
		width = std::min({ width, left->width(), right->width() });
		height = std::min({ height, left->height(), right->height() });
		const VkFormat format = left->format();

		std::unique_lock lock(g_xr.slot_mutex);
		if (g_xr.slot_w != width || g_xr.slot_h != height || g_xr.slot_format != format)
		{
			// Resolution or format change: wait for the OpenXR thread to let go of
			// its slot, drain the queue, then rebuild the eye buffers.
			while (g_xr.in_use >= 0)
			{
				lock.unlock();
				std::this_thread::yield();
				lock.lock();
			}
			vk::acquire_global_submit_lock();
			vkQueueWaitIdle(g_xr.queue);
			vk::release_global_submit_lock();
			destroy_slots();
			if (!create_slots(width, height, format))
			{
				xr_log.error("Could not create %ux%u eye buffers", width, height);
				return false;
			}
		}

		s32 slot = 0;
		while (slot == g_xr.latest || slot == g_xr.in_use)
		{
			slot++;
		}
		g_xr.writing = slot;
		lock.unlock();

		const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vk::image* sources[2] = { left, right };
		for (u32 i = 0; i < 2; ++i)
		{
			const VkImage target = g_xr.slots[slot].image[i];
			vk::image* source = sources[i];

			source->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			vk::change_image_layout(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, range);

			VkImageCopy region{};
			region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			region.extent = { width, height, 1 };
			vkCmdCopyImage(cmd, source->value, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			// Left in TRANSFER_SRC for the OpenXR thread's copy into the swapchain.
			vk::change_image_layout(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, range);
			source->pop_layout(cmd);
		}
		return true;
	}

	void commit_eyes(bool have_fov, f32 tan_half_x, f32 tan_half_y)
	{
		std::unique_lock lock(g_xr.slot_mutex);
		if (g_xr.writing < 0)
		{
			return;
		}

		// The pose this frame's draws were rotated by (located at the previous flip).
		auto& slot = g_xr.slots[g_xr.writing];
		slot.pose_valid = g_xr.render_pose_valid;
		slot.orientation = g_xr.render_orientation;
		slot.eye_position[0] = g_xr.render_eye_position[0];
		slot.eye_position[1] = g_xr.render_eye_position[1];
		slot.eye_fov[0] = g_xr.render_eye_fov[0];
		slot.eye_fov[1] = g_xr.render_eye_fov[1];
		slot.have_fov = have_fov;
		slot.tan_half_x = tan_half_x;
		slot.tan_half_y = tan_half_y;

		g_xr.latest = g_xr.writing;
		g_xr.writing = -1;
		g_xr.commit_seq++;
		lock.unlock();
		g_xr.slot_cv.notify_one();
	}

	bool projection_mode() { return g_xr.projection; }
	f32 eye_scale() { return g_xr.eye_scale; }
	f32 fov_scale() { return g_xr.fov_scale; }
	bool hmd_fov() { return g_xr.hmd_fov; }
	bool flip_y() { return g_xr.flip_y; }

	f32 ipd() { return g_xr.ipd; }

	void set_screen(bool enabled, bool world_locked, f32 width, f32 x, f32 y, f32 distance)
	{
		std::lock_guard lock(g_xr.slot_mutex);
		g_xr.screen_mode = enabled;
		if (enabled)
		{
			g_xr.screen_world = world_locked;
			g_xr.screen_width = width;
			g_xr.screen_x = x;
			g_xr.screen_y = y;
			g_xr.screen_distance = distance;
		}
	}

	bool locate_render_pose(f32 quat_xyzw[4], f32 position_xyz[3], f32 eye_fov[2][4])
	{
		// A frame rendered without a located pose must not be declared with an old one.
		g_xr.render_pose_valid = false;

		const XrTime last_display = g_xr.last_display_time.load();
		if (!is_running() || !g_xr.view_space || !last_display)
		{
			return false;
		}

		// The next game frame is published at the next guest flip and shown on the
		// following headset frame; predict about one 60 Hz frame past the latest
		// display time. The compositor corrects any prediction error.
		const XrTime time = last_display + 16666667;

		XrSpaceLocation head{ XR_TYPE_SPACE_LOCATION };
		if (!XR_SUCCEEDED(g_xr.xrLocateSpace(g_xr.view_space, g_xr.space, time, &head)) ||
			!(head.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT))
		{
			return false;
		}

		XrViewLocateInfo info{ XR_TYPE_VIEW_LOCATE_INFO };
		info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
		info.displayTime = time;
		info.space = g_xr.space;
		XrViewState state{ XR_TYPE_VIEW_STATE };
		XrView located[2]{ { XR_TYPE_VIEW }, { XR_TYPE_VIEW } };
		u32 count = 0;
		if (!XR_SUCCEEDED(g_xr.xrLocateViews(g_xr.session, &info, &state, 2, &count, located)) || count != 2)
		{
			return false;
		}

		g_xr.render_orientation = head.pose.orientation;
		g_xr.render_eye_position[0] = located[0].pose.position;
		g_xr.render_eye_position[1] = located[1].pose.position;
		{
			const XrVector3f& a = located[0].pose.position;
			const XrVector3f& b = located[1].pose.position;
			const f32 d = std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
			if (d > 0.04f && d < 0.09f)
			{
				g_xr.ipd = d;
			}
		}
		for (u32 i = 0; i < 2; ++i)
		{
			const XrFovf& f = located[i].fov;
			g_xr.render_eye_fov[i] = f;
			eye_fov[i][0] = std::tan(f.angleLeft);
			eye_fov[i][1] = std::tan(f.angleRight);
			eye_fov[i][2] = std::tan(f.angleUp);
			eye_fov[i][3] = std::tan(f.angleDown);
		}
		g_xr.render_pose_valid = true;

		position_xyz[0] = position_xyz[1] = position_xyz[2] = 0.f;
		if (g_xr.position_tracking && (head.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT))
		{
			position_xyz[0] = head.pose.position.x;
			position_xyz[1] = head.pose.position.y;
			position_xyz[2] = head.pose.position.z;
		}

		quat_xyzw[0] = head.pose.orientation.x;
		quat_xyzw[1] = head.pose.orientation.y;
		quat_xyzw[2] = head.pose.orientation.z;
		quat_xyzw[3] = head.pose.orientation.w;
		return true;
	}
}
