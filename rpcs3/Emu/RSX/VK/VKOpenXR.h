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
	// `queue`. If that is not `render_queue` (RPCS3's), the frame thread and the
	// runtime own it outright; otherwise they share it under RPCS3's global submit lock.
	bool create_session(VkInstance instance, VkPhysicalDevice pdev, VkDevice device, VkQueue queue, u32 queue_family, u32 queue_index,
		VkQueue render_queue);
	void destroy();

	// True while the headset session is running (the frame thread is presenting).
	bool is_running();

	// RSX thread, at each guest flip: record copies of the generated eyes into a
	// free eye buffer inside RPCS3's command buffer. right may equal left. If this
	// returns true, submit the command buffer, then call commit_eyes().
	bool publish_eyes(const vk::command_buffer& cmd, vk::image* left, vk::image* right, u32 width, u32 height);

	// After submitting a command buffer with published eyes/overlay, before commit_*:
	// marks the point on RPCS3's queue the frame thread waits for before copying them
	// (only when OpenXR has its own queue).
	void signal_published();

	// Make the just-submitted eye pair the newest one, tagged with the pose it was
	// rendered with (pose_id from locate_render_pose; 0 = none) and (game-FOV mode)
	// its FOV. The frame thread presents it.
	void commit_eyes(bool have_fov, f32 tan_half_x, f32 tan_half_y, u32 pose_id);

	// RPCS3's own overlays (home menu, dialogs, notifications) as a quad layer over
	// the eyes. publish_overlay() records a copy of `source` (premultiplied alpha)
	// into a free overlay buffer; after submitting, commit_eyes() shows it with the
	// eye pair, or commit_overlay() alone when no eyes were published (paused
	// emulation still flips for overlays). hide_overlay() when none is visible.
	bool publish_overlay(const vk::command_buffer& cmd, vk::image* source);
	void commit_overlay();
	void hide_overlay();

	// Where the overlay quad is shown: `width` metres wide, centred at
	// (x, y, -distance) in LOCAL space, or VIEW space when !world_locked.
	void set_overlay_placement(bool world_locked, f32 width, f32 x, f32 y, f32 distance);

	// Projection mode (default; RPCS3_OPENXR_MODE=quad selects the virtual screen).
	bool projection_mode();
	f32 eye_scale();  // RPCS3_OPENXR_EYE_SCALE, default 1 (the game's own separation)
	f32 fov_scale();  // RPCS3_OPENXR_FOV_SCALE, game-FOV mode only
	bool hmd_fov();   // RPCS3_OPENXR_FOV=game keeps the game's FOV; default renders the headset's
	bool flip_y();    // RPCS3_OPENXR_FLIP_Y=1 if head pitch/roll come out inverted

	// Locate the head for the next game frame (predicted one 60 Hz frame after
	// the latest headset display time). Returns its id for commit_eyes() (the
	// last few are kept), or 0 if tracking is unavailable.
	// eye_fov receives the located per-eye tangents (left, right, up, down).
	// render_fov receives them widened by margin_deg on every side: the frame is
	// rendered and declared with that FOV, so when the headset turns an older frame
	// to the current head pose it still has picture at the edges.
	// position_xyz receives the head position in LOCAL space (metres), zero when
	// the runtime cannot track it or RPCS3_OPENXR_POSITION=0.
	u32 locate_render_pose(f32 quat_xyzw[4], f32 position_xyz[3], f32 eye_fov[2][4], f32 render_fov[2][4], f32 margin_deg);

	// Where the centre of a frame rendered with pose to_id appears in a frame rendered
	// with pose from_id, as a texture-coordinate offset (u right, v down; fractions of
	// the rendered eye image). False if either pose is no longer kept.
	bool render_pose_shift(u32 from_id, u32 to_id, f32& du, f32& dv);
	// The same, exactly: a homography h (row-major 3x3) mapping a texture coordinate
	// (u, v, 1) of the to_id frame to the from_id frame's image (divide by the third).
	bool render_pose_homography(u32 from_id, u32 to_id, f32 h[9]);

	// Diagnostic: head yaw (degrees) of a pose from locate_render_pose, NaN if no longer kept.
	f32 render_pose_yaw(u32 pose_id);
	// Diagnostic: distance (mm) between the left-eye positions of two kept poses, NaN otherwise.
	f32 render_pose_step_mm(u32 from_id, u32 to_id);

	// Distance between the located eyes (metres), from the latest locate_render_pose.
	f32 ipd();

	// Fixed screen: show the eyes as a flat stereo quad of this width (metres),
	// centred at (x, y, -distance) in LOCAL space, or in VIEW space (following the
	// head) when !world_locked. Takes effect on the next headset frame.
	void set_screen(bool enabled, bool world_locked, f32 width, f32 x, f32 y, f32 distance);
}
