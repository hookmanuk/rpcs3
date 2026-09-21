#pragma once

#include "VulkanAPI.h"
#include "util/types.hpp"

#include <string>
#include <vector>

namespace vk
{
	class command_buffer;
	class image;
}

// Gate 6 first light: present the generated stereo pair to an OpenXR headset.
//
// The OpenXR loader is loaded dynamically (openxr_loader.dll beside rpcs3.exe), so
// RPCS3 starts normally when it is absent. XR_KHR_vulkan_enable requires RPCS3's
// own VkInstance/VkDevice to carry the runtime's extensions, so the sequence is:
//   prepare()                      before vkCreateInstance
//   instance_extensions()          merged by vk::instance::create
//   get_physical_device()          after vkCreateInstance, selects the HMD GPU
//   device_extensions()            merged by vk::render_device::create
//   create_session()               after the device exists
// and once per guest flip:
//   begin_frame() -> record_eye_copies() into RPCS3's flip command buffer ->
//   RPCS3 submits -> end_frame()
//
// The two eyes are shown as a world-locked stereo quad (a virtual 3D screen). This
// is deliberately simpler than a projection layer: the game's fixed FOV and camera
// are not yet matched to the headset's views, and a quad cannot cause the
// FOV-mismatch discomfort a projection layer would.
namespace vk::xr
{
	// Create the XrInstance and find an HMD. Returns false (and stays inert) when
	// disabled, when the loader or runtime is missing, or when no HMD is present.
	bool prepare();
	bool is_prepared();
	bool is_session_created();

	// Space-separated runtime requirements, split. Storage is owned by this module
	// so c_str() pointers remain valid while the Vulkan objects are created.
	const std::vector<std::string>& instance_extensions();
	const std::vector<std::string>& device_extensions();

	VkPhysicalDevice get_physical_device(VkInstance instance);

	bool create_session(VkInstance instance, VkPhysicalDevice pdev, VkDevice device, u32 queue_family, u32 queue_index);
	void destroy();

	// Polls events and runs xrWaitFrame/xrBeginFrame. Returns true when a frame was
	// begun; end_frame() must then be called exactly once.
	bool begin_frame();

	// Acquires both eye swapchain images and records copies from the generated eye
	// images into them. right may equal left (mono fallback). Returns false if the
	// source cannot be presented; end_frame() still submits an empty frame.
	bool record_eye_copies(const vk::command_buffer& cmd, vk::image* left, vk::image* right, u32 width, u32 height);

	// Call after the command buffer holding the copies was submitted to the queue.
	void end_frame();
}
