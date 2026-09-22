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
// and once per guest flip (RSX thread, never blocks on the headset):
//   publish_eyes() into RPCS3's flip command buffer -> RPCS3 submits ->
//   commit_eyes(); a dedicated thread runs xrWaitFrame/xrEndFrame at the
//   headset's rate and presents the newest committed pair
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

	// Also starts the OpenXR frame thread, which submits its swapchain copies to
	// `queue` under RPCS3's global submit lock.
	bool create_session(VkInstance instance, VkPhysicalDevice pdev, VkDevice device, VkQueue queue, u32 queue_family, u32 queue_index);
	void destroy();

	// True while the headset session is running (the frame thread is presenting).
	bool is_running();

	// RSX thread, at each guest flip: record copies of the generated eyes into a
	// free eye buffer inside RPCS3's command buffer. right may equal left. If this
	// returns true, submit the command buffer, then call commit_eyes().
	bool publish_eyes(const vk::command_buffer& cmd, vk::image* left, vk::image* right, u32 width, u32 height);

	// Make the just-submitted eye pair the newest one, tagged with the pose it was
	// rendered with and (game-FOV mode) its FOV. The frame thread presents it.
	void commit_eyes(bool have_fov, f32 tan_half_x, f32 tan_half_y);

	// Projection mode (default; RPCS3_OPENXR_MODE=quad selects the virtual screen).
	bool projection_mode();
	f32 eye_scale();  // RPCS3_OPENXR_EYE_SCALE, default 1 (the game's own separation)
	f32 fov_scale();  // RPCS3_OPENXR_FOV_SCALE, game-FOV mode only
	bool hmd_fov();   // RPCS3_OPENXR_FOV=game keeps the game's FOV; default renders the headset's
	bool flip_y();    // RPCS3_OPENXR_FLIP_Y=1 if head pitch/roll come out inverted

	// Locate the head for the next game frame (predicted one 60 Hz frame after
	// the latest headset display time). commit_eyes() tags that frame with it.
	// Returns false if tracking is unavailable.
	// eye_fov receives the located per-eye tangents (left, right, up, down).
	// position_xyz receives the head position in LOCAL space (metres), zero when
	// the runtime cannot track it or RPCS3_OPENXR_POSITION=0.
	bool locate_render_pose(f32 quat_xyzw[4], f32 position_xyz[3], f32 eye_fov[2][4]);

	// Distance between the located eyes (metres), from the latest locate_render_pose.
	f32 ipd();

	// Fixed screen: show the eyes as a flat stereo quad of this width (metres),
	// centred at (x, y, -distance) in LOCAL space, or in VIEW space (following the
	// head) when !world_locked. Takes effect on the next headset frame.
	void set_screen(bool enabled, bool world_locked, f32 width, f32 x, f32 y, f32 distance);
}
