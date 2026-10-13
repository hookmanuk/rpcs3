#pragma once
// framegen (fork): NVIDIA DLSS Frame Generation (NGX DLSS-G, Vulkan) for the headset. Each game frame, after DLSS and
// the HUD composite, each eye's final image goes through its own DLSS-G feature, which returns the frame halfway
// between it and the previous game frame. The OpenXR frame thread shows that generated pair first and the real pair on
// the next headset frame, so a game at half the headset rate fills every display slot with a new picture.
//
// Inputs come from the DLSS context (VKDLSS.h): the eye's motion vectors and depth at render size and the eye's camera
// motion (clip to previous clip). DLSS (an NGX mode) must therefore be on at boot.
//
// Settings: Video > VR > DLSS Frame Generation (read at boot). Environment: RPCS3_DLSSG=1 turns it on, =0 off.
#include "util/types.hpp"
#include "VulkanAPI.h"

#include <memory>
#include <vector>

namespace vk
{
	class command_buffer;
	class viewable_image;
	class render_device;

	// DLSS Frame Generation is asked for at boot and NGX can run in this session.
	bool framegen_requested();
	// The instance the device is created from (for the feature's device extension query).
	void framegen_note_instance(VkInstance instance);
	// Before vkCreateInstance / vkCreateDevice: what DLSS-G needs on top of NGX's own list.
	void framegen_instance_extensions(std::vector<const char*>& extensions);
	void framegen_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions);

	// One eye's inputs for an evaluation.
	struct framegen_eye_inputs
	{
		vk::viewable_image* backbuffer = nullptr; // the final eye image (what the headset would show)
		vk::viewable_image* depth = nullptr;      // R32F at render size (VKDLSS: the resolved depth)
		vk::viewable_image* motion = nullptr;     // RG32F at render size, current to previous, render pixels
		f32 clip_to_prev_clip[16]{};              // row-major, column vectors (clip_prev = M * clip)
		bool have_motion_matrix = false;
		f32 tan_half_x = 1.f, tan_half_y = 1.f;   // the eye's projection (symmetric half tangents)
		bool depth_inverted = false;
	};

	class framegen_context
	{
	public:
		explicit framegen_context(const vk::render_device& dev);
		~framegen_context();

		// After NGX is initialised (the DLSS context's ngx_init): DLSS-G runs on this GPU. Logs the first time.
		bool available();
		// The interpolated frame between the previous call's backbuffer for this eye and this one, or null (no previous
		// frame, a reset, or a failure). Recorded into cmd; valid until the next call for the same eye.
		vk::viewable_image* evaluate(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in, bool reset);
		// Forget both eyes' history (the next evaluation generates nothing).
		void reset() { m_have_previous[0] = m_have_previous[1] = false; }

	private:
		struct feature
		{
			void* handle = nullptr; // NVSDK_NGX_Handle*
			u32 width = 0, height = 0, render_width = 0, render_height = 0;
			VkFormat format = VK_FORMAT_UNDEFINED;
		};
		bool ensure_feature(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in);
		void release_features();

		const vk::render_device& m_device;
		s32 m_available = -1; // -1 unknown, 0 no, 1 yes
		void* m_params = nullptr; // NVSDK_NGX_Parameter* of our own
		feature m_features[2];
		std::unique_ptr<vk::viewable_image> m_output[2];
		bool m_have_previous[2]{};
		bool m_disabled = false;
	};
}
