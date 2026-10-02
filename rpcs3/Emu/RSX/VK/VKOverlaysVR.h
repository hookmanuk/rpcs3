#pragma once

// VR fork: overlay passes of the OpenXR output (declared apart from VKOverlays.h, which upstream edits).

#include "VKOverlays.h"

namespace vk
{
	// OpenXR overlay layer: draws onto a transparent target, so alpha must
	// accumulate ("over") instead of keeping the destination's. Output is
	// premultiplied, as OpenXR expects by default.
	struct ui_overlay_renderer_xr : public ui_overlay_renderer
	{
		ui_overlay_renderer_xr();
	};

	// VR fork: re-aims an image drawn with an older head pose at the current one. Each
	// output pixel (uv, v down) reads the source at H * (u, v, 1), H from
	// vk::xr::render_pose_homography; exact across the view, unlike a pixel shift.
	struct vr_homography_warp_pass : public overlay_pass
	{
		f32 homography[12] = {}; // rows of H, each padded to a vec4

		static constexpr u32 fragment_push_constants_size = sizeof(homography);

		vr_homography_warp_pass();

		std::vector<vk::glsl::program_input> get_fragment_inputs() override;

		void update_uniforms(vk::command_buffer& cmd, vk::glsl::program* program) override;

		void run(vk::command_buffer& cmd, vk::viewable_image* src, vk::image* target, const f32 h[9]);
	};
} // namespace vk
