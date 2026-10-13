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

	// NGX can run in this session (DLSS on at boot): the frame generation context and its Vulkan extensions exist, so the
	// setting can turn it on and off live. RPCS3_DLSSG=0 rules it out.
	bool framegen_possible();
	// Frame generation is on now (VR > DLSS Frame Generation, live; RPCS3_DLSSG=1/0 fixes it).
	bool framegen_enabled();
	// The instance the device is created from (for the feature's device extension query).
	void framegen_note_instance(VkInstance instance);
	// Before vkCreateInstance / vkCreateDevice: what DLSS-G needs on top of NGX's own list.
	void framegen_instance_extensions(std::vector<const char*>& extensions);
	void framegen_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions);

	// Where DLSS-G's motion comes from: hybrid (default) the game's vectors where a depth check says the pixel is static
	// world, DLSS-G's own estimate elsewhere; own: DLSS-G's own estimate everywhere; game: the game's vectors everywhere.
	// RPCS3_DLSSG_MV=hybrid|own|game fixes it; otherwise the files %TEMP%/rpcs3-vrprofile/DLSSG_MV_OWN or DLSSG_MV_GAME switch it live.
	enum class framegen_motion { hybrid, own, game };
	framegen_motion framegen_motion_mode();

	// One eye's inputs for an evaluation.
	struct framegen_eye_inputs
	{
		vk::viewable_image* backbuffer = nullptr; // the final eye image (what the headset would show)
		vk::viewable_image* hudless = nullptr;    // optional: the same without the HUD (same size and format)
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
		// index / count: the index-th of count generated frames between the two (multi-frame generation; 1 / 1 at 2x).
		vk::viewable_image* evaluate(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in, bool reset, u32 index = 1, u32 count = 1);
		// The frame-rate multiplier in use: the setting (2x or 3x), within what this GPU's DLSS-G supports.
		u32 multiplier() const;
		// Forget both eyes' history (the next evaluation generates nothing).
		void reset() { m_have_previous[0] = m_have_previous[1] = false; }
		// Turned off (the setting): the features are freed and the game goes back to the full VR frame rate.
		void disable();

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
		std::unique_ptr<vk::viewable_image> m_output[2][2]; // [eye][index - 1]
		u32 m_max_frames = 1;     // generated frames per rendered frame this GPU supports (MultiFrameCountMax)
		bool m_first[2]{};        // this frame's evaluations start the history (no previous frame): nothing generated
		std::unique_ptr<vk::viewable_image> m_invalid_mv; // dev (RPCS3_DLSSG_MV=invalid): motion vectors all marked invalid
		bool m_have_previous[2]{};
		bool m_disabled = false;
	};
}
