#include "stdafx.h"
#include "VKOverlaysVR.h"
#include "VKRenderPass.h"
#include "VKPipelineCompiler.h"

#include "vkutils/image.h"
#include "vkutils/image_helpers.h"

#include "../Program/RSXOverlay.h"

namespace vk
{
	ui_overlay_renderer_xr::ui_overlay_renderer_xr()
	{
		renderpass_config.enable_blend(0,
			VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE,
			VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			VK_BLEND_OP_ADD, VK_BLEND_OP_ADD);
	}

	vr_homography_warp_pass::vr_homography_warp_pass()
	{
		vs_src =
		#include "../Program/GLSLSnippets/GenericVSPassthrough.glsl"
		;

		fs_src =
			"#version 440\n"
			"layout(set=0, binding=0) uniform sampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"layout(location=0) out vec4 ocol;\n"
			"layout(push_constant) uniform static_data\n"
			"{\n"
			"	vec4 h0;\n"
			"	vec4 h1;\n"
			"	vec4 h2;\n"
			"};\n"
			"void main()\n"
			"{\n"
			"	const vec3 p = vec3(tc0, 1.);\n"
			"	const vec3 q = vec3(dot(h0.xyz, p), dot(h1.xyz, p), dot(h2.xyz, p));\n"
			"	ocol = texture(fs0, q.xy / q.z);\n"
			"}\n";

		renderpass_config.set_depth_mask(false);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_attachment_count(1);

		m_num_usable_samplers = 1;
		m_num_uniform_buffers = 0;
	}

	std::vector<vk::glsl::program_input> vr_homography_warp_pass::get_fragment_inputs()
	{
		auto result = overlay_pass::get_fragment_inputs();
		result.push_back(
			vk::glsl::program_input::make(
				::glsl::glsl_fragment_program,
				"push_constants",
				vk::glsl::input_type_push_constant,
				0,
				0,
				glsl::push_constant_ref{ .size = fragment_push_constants_size }
			)
		);
		return result;
	}

	void vr_homography_warp_pass::update_uniforms(vk::command_buffer& cmd, vk::glsl::program* program)
	{
		vkCmdPushConstants(cmd, program->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, fragment_push_constants_size, homography);
	}

	void vr_homography_warp_pass::run(vk::command_buffer& cmd, vk::viewable_image* src, vk::image* target, const f32 h[9])
	{
		for (u32 r = 0; r < 3; ++r)
		{
			homography[r * 4 + 0] = h[r * 3 + 0];
			homography[r * 4 + 1] = h[r * 3 + 1];
			homography[r * 4 + 2] = h[r * 3 + 2];
			homography[r * 4 + 3] = 0.f;
		}

		src->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		target->change_layout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		const VkRenderPass render_pass = vk::get_renderpass(*m_device, vk::get_renderpass_key(target->format()));
		const areau viewport = { 0, 0, target->width(), target->height() };
		overlay_pass::run(cmd, viewport, target, src->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY)), render_pass);
		vk::end_renderpass(cmd);

		src->pop_layout(cmd);
	}
}
