#include "stdafx.h"
#include <set>
#include "../Common/BufferUtils.h"
#include "../Program/GLSLCommon.h"
#include "../rsx_methods.h"
#include "../Capture/rsx_stereo_inspector.h"
#include "../Capture/rsx_camera_probe.h"
#include "../Capture/rsx_vr_profile_generator.h"

#include "VKAsyncScheduler.h"
#include "VKGSRender.h"
#include "VKOpenXR.h"
#include "vkutils/buffer_object.h"
#include "vkutils/chip_class.h"

#include <vulkan/vulkan_core.h>

namespace vk
{
	VkImageViewType get_view_type(rsx::texture_dimension_extended type)
	{
		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			return VK_IMAGE_VIEW_TYPE_1D;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			return VK_IMAGE_VIEW_TYPE_2D;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			return VK_IMAGE_VIEW_TYPE_CUBE;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			return VK_IMAGE_VIEW_TYPE_3D;
		default: fmt::throw_exception("Unreachable");
		}
	}

	VkCompareOp get_compare_func(rsx::comparison_function op, bool reverse_direction = false)
	{
		switch (op)
		{
		case rsx::comparison_function::never: return VK_COMPARE_OP_NEVER;
		case rsx::comparison_function::greater: return reverse_direction ? VK_COMPARE_OP_LESS: VK_COMPARE_OP_GREATER;
		case rsx::comparison_function::less: return reverse_direction ? VK_COMPARE_OP_GREATER: VK_COMPARE_OP_LESS;
		case rsx::comparison_function::less_or_equal: return reverse_direction ? VK_COMPARE_OP_GREATER_OR_EQUAL: VK_COMPARE_OP_LESS_OR_EQUAL;
		case rsx::comparison_function::greater_or_equal: return reverse_direction ? VK_COMPARE_OP_LESS_OR_EQUAL: VK_COMPARE_OP_GREATER_OR_EQUAL;
		case rsx::comparison_function::equal: return VK_COMPARE_OP_EQUAL;
		case rsx::comparison_function::not_equal: return VK_COMPARE_OP_NOT_EQUAL;
		case rsx::comparison_function::always: return VK_COMPARE_OP_ALWAYS;
		default:
			fmt::throw_exception("Unknown compare op: 0x%x", static_cast<u32>(op));
		}
	}

	void validate_image_layout_for_read_access(
		vk::command_buffer& cmd,
		vk::image_view* view,
		VkPipelineStageFlags dst_stage,
		const rsx::sampled_image_descriptor_base* sampler_state)
	{
		switch (auto raw = view->image(); +raw->current_layout)
		{
		default:
			//case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			//ensure(sampler_state->upload_context == rsx::texture_upload_context::blit_engine_dst);
			raw->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			ensure(sampler_state->upload_context == rsx::texture_upload_context::blit_engine_src);
			raw->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			break;
		case VK_IMAGE_LAYOUT_GENERAL:
		case VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT:
			ensure(sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage);
			if (sampler_state->is_cyclic_reference) [[ unlikely ]]
			{
				// Nothing to do
				break;
			}

			// This was used in a cyclic ref before, but is missing a barrier
			// No need for a full stall, use a custom barrier instead
			VkPipelineStageFlags src_stage;
			VkAccessFlags src_access, dst_access;
			if (raw->aspect() == VK_IMAGE_ASPECT_COLOR_BIT)
			{
				src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
				src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				dst_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
				dst_stage |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			}
			else
			{
				src_stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
				src_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				dst_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
				dst_stage |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			}

			vk::insert_image_memory_barrier(
				cmd,
				raw->value,
				raw->current_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				src_stage, dst_stage,
				src_access, dst_access,
				{ raw->aspect(), 0, 1, 0, 1 });

			raw->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			ensure(sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage);
			if (!sampler_state->is_cyclic_reference) [[ likely ]]
			{
				// Standard pre-read barrier.
				raw->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				break;
			}

			// Normally this shouldn't happen. But that is only guaranteed if the attachment state never changes outside of RTT rebind interrupts.
			// If the shader changes between binds to "disable" the cyclic nature, we could end up here.
			// Draw 1 (cyllic) -> texture_barrier -> Draw 2 (no textures) -> attachment_optimal -> Draw 3 (cylic again, no new data) -> incorrect layout.
			vk::as_rtt(raw)->texture_barrier(cmd);
			break;
		}
	}
}

void VKGSRender::begin_render_pass()
{
	vk::begin_renderpass(
		*m_current_command_buffer,
		get_render_pass(),
		m_draw_fbo->value,
		{ positionu{0u, 0u}, sizeu{m_draw_fbo->width(), m_draw_fbo->height()} });
}

void VKGSRender::close_render_pass()
{
	vk::end_renderpass(*m_current_command_buffer);
}

VkRenderPass VKGSRender::get_render_pass()
{
	if (!m_cached_renderpass)
	{
		m_cached_renderpass = vk::get_renderpass(*m_device, m_current_renderpass_key);
	}

	return m_cached_renderpass;
}

void VKGSRender::invalidate_render_pass()
{
	// Regenerate renderpass key for the next draw call
	std::vector<u8> input_attachments{};
	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		input_attachments.resize(m_draw_buffers.size());
		std::iota(input_attachments.begin(), input_attachments.end(), 0);
	}

	if (const auto key = vk::get_renderpass_key(m_fbo_images, m_current_renderpass_key, input_attachments);
		key != m_current_renderpass_key)
	{
		m_current_renderpass_key = key;
		m_cached_renderpass = VK_NULL_HANDLE;
	}
}

void VKGSRender::update_draw_state()
{
	m_profiler.start();

	// Update conditional dynamic state
	if (rsx::method_registers.current_draw_clause.primitive >= rsx::primitive_type::points &&   // AMD/AMDVLK driver does not like it if you render points without setting line width for some reason
		rsx::method_registers.current_draw_clause.primitive <= rsx::primitive_type::line_strip)
	{
		const float actual_line_width =
			m_device->get_wide_lines_support() ? rsx::method_registers.line_width() * resolution_scaling_config.scale_factor() : 1.f;
		vkCmdSetLineWidth(*m_current_command_buffer, actual_line_width);
	}

	if (rsx::method_registers.blend_enabled_mask())
	{
		// Update blend constants
		auto blend_colors = rsx::get_constant_blend_colors();
		vkCmdSetBlendConstants(*m_current_command_buffer, blend_colors.data());
	}

	if (rsx::method_registers.stencil_test_enabled())
	{
		const bool two_sided_stencil = rsx::method_registers.two_sided_stencil_test_enabled();
		VkStencilFaceFlags face_flag = (two_sided_stencil) ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FRONT_AND_BACK;

		vkCmdSetStencilWriteMask(*m_current_command_buffer, face_flag, rsx::method_registers.stencil_mask());
		vkCmdSetStencilCompareMask(*m_current_command_buffer, face_flag, rsx::method_registers.stencil_func_mask());
		vkCmdSetStencilReference(*m_current_command_buffer, face_flag, rsx::method_registers.stencil_func_ref());

		if (two_sided_stencil)
		{
			vkCmdSetStencilWriteMask(*m_current_command_buffer, VK_STENCIL_FACE_BACK_BIT, rsx::method_registers.back_stencil_mask());
			vkCmdSetStencilCompareMask(*m_current_command_buffer, VK_STENCIL_FACE_BACK_BIT, rsx::method_registers.back_stencil_func_mask());
			vkCmdSetStencilReference(*m_current_command_buffer, VK_STENCIL_FACE_BACK_BIT, rsx::method_registers.back_stencil_func_ref());
		}
	}

	// The remaining dynamic state should only be set once and we have signals to enable/disable mid-renderpass
	if (!(m_current_command_buffer->flags & vk::command_buffer::cb_reload_dynamic_state))
	{
		// Dynamic state already set
		m_frame_stats.setup_time += m_profiler.duration();
		return;
	}

	if (rsx::method_registers.poly_offset_fill_enabled())
	{
		// offset_bias is the constant factor, multiplied by the implementation factor R
		// offst_scale is the slope factor, multiplied by the triangle slope factor M
		// R is implementation dependent and has to be derived empirically for supported implementations.
		// Lucky for us, only NVIDIA currently supports fixed-point 24-bit depth buffers.

		const auto polygon_offset_scale = rsx::method_registers.poly_offset_scale();
		auto polygon_offset_bias = rsx::method_registers.poly_offset_bias();

		if (m_draw_fbo->depth_format() == VK_FORMAT_D24_UNORM_S8_UINT && is_NVIDIA(vk::get_chip_family()))
		{
			// Empirically derived to be 0.5 * (2^24 - 1) for fixed type on Pascal. The same seems to apply for other NVIDIA GPUs.
			// RSX seems to be using 2^24 - 1 instead making the biases twice as large when using fixed type Z-buffer on NVIDIA.
			// Note, that the formula for floating point is complicated, but actually works out for us.
			// Since the exponent range for a polygon is around 0, and we have 23 (+1) mantissa bits, R just works out to the same range by chance \o/.
			polygon_offset_bias *= 0.5f;
		}

		vkCmdSetDepthBias(*m_current_command_buffer, polygon_offset_bias, 0.f, polygon_offset_scale);
	}
	else
	{
		// Zero bias value - disables depth bias
		vkCmdSetDepthBias(*m_current_command_buffer, 0.f, 0.f, 0.f);
	}

	if (m_device->get_depth_bounds_support())
	{
		f32 bounds_min, bounds_max;
		if (rsx::method_registers.depth_bounds_test_enabled())
		{
			// Update depth bounds min/max
			bounds_min = rsx::method_registers.depth_bounds_min();
			bounds_max = rsx::method_registers.depth_bounds_max();
		}
		else
		{
			// Avoid special case where min=max and depth bounds (incorrectly) fails
			bounds_min = std::min(0.f, rsx::method_registers.clip_min());
			bounds_max = std::max(1.f, rsx::method_registers.clip_max());
		}

		if (!m_device->get_unrestricted_depth_range_support())
		{
			bounds_min = std::clamp(bounds_min, 0.f, 1.f);
			bounds_max = std::clamp(bounds_max, 0.f, 1.f);
		}

		vkCmdSetDepthBounds(*m_current_command_buffer, bounds_min, bounds_max);
	}

	bind_viewport();

	m_current_command_buffer->flags &= ~vk::command_buffer::cb_reload_dynamic_state;
	m_graphics_state.clear(rsx::pipeline_state::polygon_offset_state_dirty | rsx::pipeline_state::depth_bounds_state_dirty);
	m_frame_stats.setup_time += m_profiler.duration();
}

void VKGSRender::load_texture_env()
{
	// Load textures
	bool check_for_cyclic_refs = false;
	auto check_surface_cache_sampler_valid = [&](auto descriptor, const auto& tex)
	{
		if (!m_texture_cache.test_if_descriptor_expired(*m_current_command_buffer, m_rtts, descriptor, tex))
		{
			check_for_cyclic_refs |= descriptor->is_cyclic_reference;
			return true;
		}

		return false;
	};

	auto get_border_color = [&](const rsx::Texture auto& tex, bool remap_colorspace)
	{
		return m_device->get_custom_border_color_support().require_border_color_remap
			? tex.remapped_border_color(remap_colorspace)
			: rsx::decode_border_color(tex.border_color(remap_colorspace));
	};

	std::lock_guard lock(m_sampler_mutex);

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!fs_sampler_state[i])
		{
			fs_sampler_state[i] = std::make_unique<vk::texture_cache::sampled_image_descriptor>();
		}

		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.fragment_textures[i];
		const auto previous_format_class = fs_sampler_state[i]->format_class;

		if (!m_samplers_dirty &&
			!m_textures_dirty[i] &&
			check_surface_cache_sampler_valid(sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_textures_dirty[i];
		m_textures_dirty[i] = false;

		if (!tex.enabled())
		{
			*sampler_state = {};
			continue;
		}

		*sampler_state = m_texture_cache.upload_texture(*m_current_command_buffer, tex, m_rtts);
		if (!sampler_state->validate())
		{
			continue;
		}

		if (sampler_state->is_cyclic_reference)
		{
			check_for_cyclic_refs |= true;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::fragment_program_state_dirty;
			}

			if (sampler_state->format_ex)
			{
				// Nothing to change, use cached sampler
				continue;
			}
		}

		sampler_state->format_ex = tex.format_ex();

		if (sampler_state->format_ex.texel_remap_control &&
			sampler_state->image_handle &&
			sampler_state->upload_context == rsx::texture_upload_context::shader_read &&
			(current_fp_metadata.bx2_texture_reads_mask & (1u << i)) == 0 &&
			!g_cfg.video.disable_hardware_texel_remapping) [[ unlikely ]]
		{
			// Check if we need to override the view format
			const auto vk_format = sampler_state->image_handle->format();
			VkFormat format_override = vk_format;;
			rsx::flags32_t flags_to_erase = 0u;
			rsx::flags32_t host_flags_to_set = 0u;

			if (sampler_state->format_ex.hw_SNORM_possible())
			{
				format_override = vk::get_compatible_snorm_format(vk_format);
				flags_to_erase = rsx::texture_control_bits::SEXT_MASK;
				host_flags_to_set = rsx::RSX_HOST_FORMAT_FEATURE_SNORM;
			}
			else if (sampler_state->format_ex.hw_SRGB_possible())
			{
				format_override = vk::get_compatible_srgb_format(vk_format);
				flags_to_erase = rsx::texture_control_bits::GAMMA_CTRL_MASK;
				host_flags_to_set = rsx::RSX_HOST_FORMAT_FEATURE_SRGB;
			}

			if (format_override != VK_FORMAT_UNDEFINED && format_override != vk_format)
			{
				sampler_state->image_handle = sampler_state->image_handle->as(format_override);
				sampler_state->format_ex.texel_remap_control &= (~flags_to_erase);
				sampler_state->format_ex.host_features |= host_flags_to_set;
			}
		}

		VkFilter mag_filter;
		vk::minification_filter min_filter;
		f32 min_lod = 0.f, max_lod = 0.f;
		f32 lod_bias = 0.f;

		const u32 texture_format = sampler_state->format_ex.format();
		VkBool32 compare_enabled = VK_FALSE;
		VkCompareOp depth_compare_mode = VK_COMPARE_OP_NEVER;

		if (texture_format >= CELL_GCM_TEXTURE_DEPTH24_D8 && texture_format <= CELL_GCM_TEXTURE_DEPTH16_FLOAT)
		{
			compare_enabled = VK_TRUE;
			depth_compare_mode = vk::get_compare_func(tex.zfunc(), true);
		}

		const f32 af_level = vk::max_aniso(tex.max_aniso());
		const auto wrap_s = vk::vk_wrap_mode(tex.wrap_s());
		const auto wrap_t = vk::vk_wrap_mode(tex.wrap_t());
		const auto wrap_r = vk::vk_wrap_mode(tex.wrap_r());

		// NOTE: In vulkan, the border color can bypass the sample swizzle stage.
		// Check the device properties to determine whether to pre-swizzle the colors or not.
		const bool sext_conv_required = (sampler_state->format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;
		vk::border_color_t border_color(VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);

		if (rsx::is_border_clamped_texture(tex))
		{
			auto color_value = get_border_color(tex, sext_conv_required);
			if (sampler_state->format_ex.host_snorm_format_active())
			{
				// Convert the border color in host space (2N - 1)
				// HW does the conversion in integer space as (x - 128) / 127 which introduces a biasing error.
				const float bias_v = 128.f / 255.f;
				const float scale_v = 255.f / 127.f;

				color4f scale{ 1.f }, bias{ 0.f };
				const auto snorm_mask = tex.argb_signed();
				if (snorm_mask & 1) { scale.a = scale_v; bias.a = -bias_v; }
				if (snorm_mask & 2) { scale.r = scale_v; bias.r = -bias_v; }
				if (snorm_mask & 4) { scale.g = scale_v; bias.g = -bias_v; }
				if (snorm_mask & 8) { scale.b = scale_v; bias.b = -bias_v; }
				color_value = (color_value + bias) * scale;
			}

			border_color = color_value;
		}

		// Check if non-point filtering can even be used on this format
		bool can_sample_linear;
		if (sampler_state->format_class == RSX_FORMAT_CLASS_COLOR) [[likely]]
		{
			// Most PS3-like formats can be linearly filtered without problem
			// Exclude textures that require SNORM conversion however
			can_sample_linear = !sext_conv_required;
		}
		else if (sampler_state->format_class != rsx::classify_format(texture_format) &&
			(texture_format == CELL_GCM_TEXTURE_A8R8G8B8 || texture_format == CELL_GCM_TEXTURE_D8R8G8B8))
		{
			// Depth format redirected to BGRA8 resample stage. Do not filter to avoid bits leaking
			can_sample_linear = false;
		}
		else
		{
			// Not all GPUs support linear filtering of depth formats
			const auto vk_format = sampler_state->image_handle ? sampler_state->image_handle->image()->format() :
				vk::get_compatible_sampler_format(m_device->get_formats_support(), sampler_state->external_subresource_desc.gcm_format);

			can_sample_linear = m_device->get_format_properties(vk_format).optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
		}

		const auto mipmap_count = tex.get_exact_mipmap_count();
		min_filter = vk::get_min_filter(tex.min_filter());

		if (can_sample_linear)
		{
			mag_filter = vk::get_mag_filter(tex.mag_filter());
		}
		else
		{
			mag_filter = VK_FILTER_NEAREST;
			min_filter.filter = VK_FILTER_NEAREST;
			min_filter.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		}

		if (min_filter.sample_mipmaps && mipmap_count > 1)
		{
			f32 actual_mipmaps;
			if (sampler_state->upload_context == rsx::texture_upload_context::shader_read)
			{
				actual_mipmaps = static_cast<f32>(mipmap_count);
			}
			else if (sampler_state->external_subresource_desc.op != rsx::deferred_request_command::nop)
			{
				actual_mipmaps = sampler_state->external_subresource_desc.exact_mip_count();
			}
			else
			{
				actual_mipmaps = 1.f;
			}

			if (actual_mipmaps > 1.f)
			{
				min_lod = tex.min_lod();
				max_lod = tex.max_lod();
				lod_bias = tex.bias();

				min_lod = std::min(min_lod, actual_mipmaps - 1.f);
				max_lod = std::min(max_lod, actual_mipmaps - 1.f);

				if (min_filter.mipmap_mode == VK_SAMPLER_MIPMAP_MODE_NEAREST)
				{
					// Round to nearest 0.5 to work around some broken games
					// Unlike openGL, sampler parameters cannot be dynamically changed on vulkan, leading to many permutations
					lod_bias = std::floor(lod_bias * 2.f + 0.5f) * 0.5f;
				}
			}
			else
			{
				min_lod = max_lod = lod_bias = 0.f;
				min_filter.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			}
		}

		if (fs_sampler_handles[i] &&
			fs_sampler_handles[i]->matches(wrap_s, wrap_t, wrap_r, false, lod_bias, af_level, min_lod, max_lod,
				min_filter.filter, mag_filter, min_filter.mipmap_mode, border_color, compare_enabled, depth_compare_mode))
		{
			continue;
		}

		fs_sampler_handles[i] = vk::get_resource_manager()->get_sampler(
			*m_device,
			fs_sampler_handles[i],
			wrap_s, wrap_t, wrap_r,
			false,
			lod_bias, af_level, min_lod, max_lod,
			min_filter.filter, mag_filter, min_filter.mipmap_mode,
			border_color, compare_enabled, depth_compare_mode);
	}

	for (u32 textures_ref = current_vp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!vs_sampler_state[i])
		{
			vs_sampler_state[i] = std::make_unique<vk::texture_cache::sampled_image_descriptor>();
		}

		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(vs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.vertex_textures[i];
		const auto previous_format_class = sampler_state->format_class;

		if (!m_samplers_dirty &&
			!m_vertex_textures_dirty[i] &&
			check_surface_cache_sampler_valid(sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_vertex_textures_dirty[i];
		m_vertex_textures_dirty[i] = false;

		if (!rsx::method_registers.vertex_textures[i].enabled())
		{
			*sampler_state = {};
			continue;
		}

		*sampler_state = m_texture_cache.upload_texture(*m_current_command_buffer, tex, m_rtts);
		if (!sampler_state->validate())
		{
			continue;
		}

		if (sampler_state->is_cyclic_reference || sampler_state->external_subresource_desc.do_not_cache)
		{
			check_for_cyclic_refs |= true;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::vertex_program_state_dirty;
			}

			if (vs_sampler_handles[i])
			{
				continue;
			}
		}

		const VkBool32 unnormalized_coords = !!(tex.format() & CELL_GCM_TEXTURE_UN);
		const auto min_lod = tex.min_lod();
		const auto max_lod = tex.max_lod();
		const auto wrap_s = vk::vk_wrap_mode(tex.wrap_s());
		const auto wrap_t = vk::vk_wrap_mode(tex.wrap_t());

		// NOTE: In vulkan, the border color can bypass the sample swizzle stage.
		// Check the device properties to determine whether to pre-swizzle the colors or not.
		const auto border_color = is_border_clamped_texture(tex)
			? vk::border_color_t(get_border_color(tex, false))
			: vk::border_color_t(VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);

		if (vs_sampler_handles[i] &&
			vs_sampler_handles[i]->matches(wrap_s, wrap_t, VK_SAMPLER_ADDRESS_MODE_REPEAT,
				unnormalized_coords, 0.f, 1.f, min_lod, max_lod, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, border_color))
		{
			continue;
		}

		vs_sampler_handles[i] = vk::get_resource_manager()->get_sampler(
			*m_device,
			vs_sampler_handles[i],
			wrap_s, wrap_t, VK_SAMPLER_ADDRESS_MODE_REPEAT,
			unnormalized_coords,
			0.f, 1.f, min_lod, max_lod,
			VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, border_color);
	}

	m_samplers_dirty.store(false);

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
	{
		// Transition our FBO to a loop-friendly format.
		// We can also convert it into an input attachment, but for now this is easier.
		auto ds = ensure(m_rtts.m_bound_depth_stencil.second, "Invalid FS export configuration.");
		ds->texture_barrier(*m_current_command_buffer);

		check_for_cyclic_refs = true;
	}

	if (check_for_cyclic_refs)
	{
		// Regenerate renderpass key
		invalidate_render_pass();
	}

	if (backend_config.supports_asynchronous_compute)
	{
		// We have to do this here, because we have to assume the CB will be dumped
		auto async_task_scheduler = g_fxo->try_get<vk::AsyncTaskScheduler>();

		if (async_task_scheduler &&
			async_task_scheduler->is_recording() &&
			!async_task_scheduler->is_host_mode())
		{
			// Sync any async scheduler tasks
			if (auto ev = async_task_scheduler->get_primary_sync_label())
			{
				ev->gpu_wait(*m_current_command_buffer, m_async_compute_dependency_info);
			}
		}
	}
}

bool VKGSRender::bind_texture_env(bool vr_right_eye)
{
	bool out_of_memory = false;

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			// Unused TIU
			continue;
		}

		if (m_fs_binding_table->ftex_location[i] == umax)
		{
			// Corrupt shader table
			break;
		}

		vk::image_view* view = nullptr;
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());

		if (rsx::method_registers.fragment_textures[i].enabled() &&
			sampler_state->validate())
		{
			// Gate 5 render-target feedback: the ordinary texture cache resolves
			// guest addresses to the authoritative left-eye surface. During the
			// right replay, substitute the isomorphic host-only surface at the
			// same guest address so post-processing does not collapse both eyes
			// back to the left intermediate.
			// The left eye either samples the surface directly (image_handle is a view of
			// the render target), or through a deferred copy of part of it, possibly
			// with a format conversion (no image_handle; NFS Most Wanted). The copy is
			// rebuilt from the matching right-eye surfaces; a raw right-eye view would
			// read the wrong region or format. A cached copy that is not a render
			// target stays on the left eye's image.
			vk::render_target* left_rtt = nullptr;
			if (vr_right_eye && sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage && sampler_state->image_handle)
			{
				left_rtt = dynamic_cast<vk::render_target*>(sampler_state->image_handle->image());
			}

			if (vr_right_eye && sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage && !sampler_state->image_handle)
			{
				auto desc = sampler_state->external_subresource_desc;
				bool complete = true;
				const auto to_right = [&](vk::image* src) -> vk::image*
				{
					auto* rtt = dynamic_cast<vk::render_target*>(src);
					auto* right = rtt ? m_vr_right_rtts.get_surface_at(rtt->base_addr) : nullptr;
					if (!right || right->format() != rtt->format() || right->width() != rtt->width() || right->height() != rtt->height())
					{
						complete = false;
						return src;
					}
					right->read_barrier(*m_current_command_buffer);
					return right->get_surface(rsx::surface_access::shader_read);
				};
				if (desc.external_handle)
				{
					desc.external_handle = to_right(desc.external_handle);
				}
				for (auto& section : desc.sections_to_copy)
				{
					section.src = to_right(section.src);
				}
				if (complete)
				{
					desc.do_not_cache = true;
					if (desc.op == rsx::deferred_request_command::copy_image_static)
					{
						desc.op = rsx::deferred_request_command::copy_image_dynamic;
					}
					view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, desc);
				}
			}
			else if (vr_right_eye && left_rtt)
			{
				if (auto* right_surface = m_vr_right_rtts.get_surface_at(sampler_state->ref_address))
				{
					if (sampler_state->is_cyclic_reference)
					{
						right_surface->texture_barrier(*m_current_command_buffer);
					}
					else
					{
						right_surface->read_barrier(*m_current_command_buffer);
					}

					auto* right_image = right_surface->get_surface(rsx::surface_access::shader_read);
					const auto aspect = sampler_state->image_handle
						? sampler_state->image_handle->info.subresourceRange.aspectMask
						: right_image->aspect();
					view = right_image->get_view(rsx::method_registers.fragment_textures[i].decoded_remap(), aspect);
				}
			}

			// Diagnostic: a right-eye sample left on the shared (left-eye) image although the
			// right-eye store holds a surface in the sampled range. Logged once per address.
			if (vr_right_eye && !view)
			{
				const u32 address = rsx::get_address(rsx::method_registers.fragment_textures[i].offset(), rsx::method_registers.fragment_textures[i].location());
				static std::set<u32> s_reported;
				if ((m_vr_right_rtts.get_surface_at(address) || sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage) && s_reported.insert(address).second)
				{
					rsx_log.warning("VR right eye: texture at 0x%x (context %d, ref 0x%x) samples the left eye (right-eye surface there: %d).",
						address, static_cast<int>(sampler_state->upload_context), sampler_state->ref_address, m_vr_right_rtts.get_surface_at(address) != nullptr);
				}
			}

			if (!view)
			{
				view = sampler_state->image_handle;
			}

			if (!view)
			{
				//Requires update, copy subresource
				if (!(view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, sampler_state->external_subresource_desc)))
				{
					out_of_memory = true;
				}
			}
			else
			{
				validate_image_layout_for_read_access(*m_current_command_buffer, view, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, sampler_state);
			}
		}

		if (view) [[likely]]
		{
			m_program->bind_uniform({ *view, *fs_sampler_handles[i] },
				vk::glsl::binding_set_index_fragment,
				m_fs_binding_table->ftex_location[i]);

			if (current_fragment_program.texture_state.redirected_textures & (1 << i))
			{
				// Stencil mirror required
				auto root_image = static_cast<vk::viewable_image*>(view->image());
				auto stencil_view = root_image->get_view(rsx::default_remap_vector, VK_IMAGE_ASPECT_STENCIL_BIT);

				if (!m_stencil_mirror_sampler)
				{
					m_stencil_mirror_sampler = std::make_unique<vk::sampler>(*m_device,
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f,
						VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
						VK_BORDER_COLOR_INT_OPAQUE_BLACK);
				}

				m_program->bind_uniform({ *stencil_view, *m_stencil_mirror_sampler },
					vk::glsl::binding_set_index_fragment,
					m_fs_binding_table->ftex_stencil_location[i]);
			}
		}
		else
		{
			const VkImageViewType view_type = vk::get_view_type(current_fragment_program.get_texture_dimension(i));
			const VkDescriptorImageInfoEx desc = { *vk::null_image_view(*m_current_command_buffer, view_type), vk::null_sampler() };
			m_program->bind_uniform(desc,
				vk::glsl::binding_set_index_fragment,
				m_fs_binding_table->ftex_location[i]);

			if (current_fragment_program.texture_state.redirected_textures & (1 << i))
			{
				m_program->bind_uniform(desc,
					vk::glsl::binding_set_index_fragment,
					m_fs_binding_table->ftex_stencil_location[i]);
			}
		}
	}

	for (u32 textures_ref = current_vp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			// Unused TIU
			continue;
		}

		if (m_vs_binding_table->vtex_location[i] == umax)
		{
			// Corrupt shader
			break;
		}

		if (!rsx::method_registers.vertex_textures[i].enabled())
		{
			const auto view_type = vk::get_view_type(current_vertex_program.get_texture_dimension(i));
			m_program->bind_uniform({ *vk::null_image_view(*m_current_command_buffer, view_type), vk::null_sampler() },
				vk::glsl::binding_set_index_vertex,
				m_vs_binding_table->vtex_location[i]);

			continue;
		}

		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(vs_sampler_state[i].get());
		auto image_ptr = sampler_state->image_handle;
		if (vr_right_eye && sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage)
		{
			if (auto* right_surface = m_vr_right_rtts.get_surface_at(sampler_state->ref_address))
			{
				right_surface->read_barrier(*m_current_command_buffer);
				auto* right_image = right_surface->get_surface(rsx::surface_access::shader_read);
				const auto aspect = image_ptr ? image_ptr->info.subresourceRange.aspectMask : right_image->aspect();
				image_ptr = right_image->get_view(rsx::method_registers.vertex_textures[i].decoded_remap(), aspect);
			}
		}

		if (!image_ptr && sampler_state->validate())
		{
			if (!(image_ptr = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, sampler_state->external_subresource_desc)))
			{
				out_of_memory = true;
			}
		}

		if (!image_ptr)
		{
			rsx_log.error("Texture upload failed to vtexture index %d. Binding null sampler.", i);
			const auto view_type = vk::get_view_type(current_vertex_program.get_texture_dimension(i));

			m_program->bind_uniform({ *vk::null_image_view(*m_current_command_buffer, view_type), vk::null_sampler() },
				vk::glsl::binding_set_index_vertex,
				m_vs_binding_table->vtex_location[i]);

			continue;
		}

		validate_image_layout_for_read_access(*m_current_command_buffer, image_ptr, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, sampler_state);

		m_program->bind_uniform({ *image_ptr, *vs_sampler_handles[i] },
			vk::glsl::binding_set_index_vertex,
			m_vs_binding_table->vtex_location[i]);
	}

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
	{
		auto ds = ensure(vr_right_eye
			? m_vr_right_rtts.m_bound_depth_stencil.second
			: m_rtts.m_bound_depth_stencil.second);
		auto view = ds->get_view(rsx::default_remap_vector, VK_IMAGE_ASPECT_DEPTH_BIT);
		m_program->bind_uniform({ *view, vk::null_sampler() }, vk::glsl::binding_set_index_fragment, m_fs_binding_table->frag_depth_input_location);
	}

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		ensure(current_fragment_program.mrt_buffers_count == m_draw_buffers.size());
		const auto remap = rsx::default_remap_vector.with_encoding(vk::VK_REMAP_IDENTITY);

		for (u32 i = 0; i < current_fragment_program.mrt_buffers_count; ++i)
		{
			auto viewable = static_cast<vk::viewable_image*>(m_fbo_images[i]);
			const auto view = viewable->get_view(remap);
			m_program->bind_uniform(*view, vk::glsl::binding_set_index_fragment, m_fs_binding_table->frag_src_location[i]);
		}
	}

	return out_of_memory;
}

bool VKGSRender::bind_interpreter_texture_env()
{
	if (current_fp_metadata.referenced_textures_mask == 0)
	{
		// Nothing to do
		return false;
	}

	std::array<VkDescriptorImageInfoEx, 68> texture_env;
	VkDescriptorImageInfoEx fallback =
	{
		*vk::null_image_view(*m_current_command_buffer, VK_IMAGE_VIEW_TYPE_1D),
		vk::null_sampler()
	};

	auto start = texture_env.begin();
	auto end = start;

	// Fill default values
	// 1D
	std::advance(end, 16);
	std::fill(start, end, fallback);
	// 2D
	start = end;
	fallback.imageView = vk::null_image_view(*m_current_command_buffer, VK_IMAGE_VIEW_TYPE_2D)->value;
	std::advance(end, 16);
	std::fill(start, end, fallback);
	// 3D
	start = end;
	fallback.imageView = vk::null_image_view(*m_current_command_buffer, VK_IMAGE_VIEW_TYPE_3D)->value;
	std::advance(end, 16);
	std::fill(start, end, fallback);
	// CUBE
	start = end;
	fallback.imageView = vk::null_image_view(*m_current_command_buffer, VK_IMAGE_VIEW_TYPE_CUBE)->value;
	std::advance(end, 16);
	std::fill(start, end, fallback);

	bool out_of_memory = false;

	auto decay_view_for_interpreter = [&](
		const rsx::image_section_attributes_t& attr,
		vk::texture_cache::sampled_image_descriptor* desc,
		vk::image_view* base,
		const rsx::texture_channel_remap_t& decoded_remap,
		bool is_msaa,
		bool is_redirected) -> vk::image_view*
	{
		if (!is_msaa && !is_redirected)
		{
			return base;
		}

		if (is_redirected && desc->image_type > rsx::texture_dimension_extended::texture_dimension_2d)
		{
			// Cannot handle redirect on 3D or cubemap with the interpreter.
			auto view_type = vk::get_view_type(desc->image_type);
			return vk::null_image_view(*m_current_command_buffer, view_type);
		}

		using deferred_subresource_t = vk::texture_cache::deferred_subresource;
		auto image = static_cast<vk::viewable_image*>(base->image());
		auto rtt = vk::try_as_rtt(base->image());

		if (is_msaa)
		{
			// MSAA resolve
			ensure(rtt);
			rtt->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_read);
			image = rtt->get_surface(rsx::surface_access::transfer_read);
		}

		if (is_redirected)
		{
			// Force bitcast
			rsx::image_section_attributes_t flatten_attrs{};
			flatten_attrs.address = desc->ref_address;
			flatten_attrs.gcm_format = desc->format_ex.format();
			flatten_attrs.width = image->width();
			flatten_attrs.height = image->height();
			flatten_attrs.depth = 1;

			const coord3u flatten_rect = { 0, 0, 0, flatten_attrs.width, flatten_attrs.height, 1 };
			auto flatten_op = deferred_subresource_t::create_copy(
				image, flatten_attrs, flatten_rect, rsx::surface_transform::identity, decoded_remap, desc->is_cyclic_reference);

			ensure(desc->ref_address);
			flatten_op.cache_range = rtt
				? rtt->get_memory_range()
				: utils::address_range32::start_length(desc->ref_address, attr.pitch * attr.height);

			return m_texture_cache.create_temporary_subresource(*m_current_command_buffer, flatten_op);
		}

		image->change_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		return image->get_view(decoded_remap, base->info.subresourceRange.aspectMask);
	};

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
			continue;

		vk::image_view* view = nullptr;
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());

		auto& tex = rsx::method_registers.fragment_textures[i];
		if (tex.enabled() && sampler_state->validate())
		{
			if (view = sampler_state->image_handle; !view)
			{
				//Requires update, copy subresource
				if (!(view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, sampler_state->external_subresource_desc)))
				{
					out_of_memory = true;
				}
			}
		}

		if (!view)
		{
			// OOM or disabled texture
			continue;
		}

		auto primary_view = view;

		// Flatten MSAA and DEPTH24S8 redirects
		if (view->image()->samples() > 1 || view->info.subresourceRange.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT)
		{
			const auto mask = (1u << i);
			const bool is_redirected = !!(current_fragment_program.texture_state.redirected_textures & mask);
			const bool is_msaa = !!(current_fragment_program.texture_state.multisampled_textures & mask);
			if (is_redirected || is_msaa)
			{
				view = decay_view_for_interpreter(
					tex.attributes(),
					sampler_state,
					view,
					tex.decoded_remap(),
					is_msaa,
					is_redirected);

				if (!view)
				{
					// OOM
					out_of_memory = true;
					continue;
				}
			}
		}

		if (primary_view == view)
		{
			validate_image_layout_for_read_access(*m_current_command_buffer, view, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, sampler_state);
		}

		const int offsets[] = { 0, 16, 48, 32 };
		auto& sampled_image_info = texture_env[offsets[static_cast<u32>(sampler_state->image_type)] + i];
		sampled_image_info = { *view, *fs_sampler_handles[i] };
	}

	m_shader_interpreter.update_fragment_textures(texture_env);
	return out_of_memory;
}

void VKGSRender::emit_geometry(u32 sub_index)
{
	auto &draw_call = rsx::method_registers.current_draw_clause;
	m_profiler.start();

	const rsx::flags32_t vertex_state_mask = rsx::vertex_base_changed | rsx::vertex_arrays_changed;
	const rsx::flags32_t state_flags = (sub_index == 0) ? rsx::vertex_arrays_changed : draw_call.execute_pipeline_dependencies(m_ctx);

	if (state_flags & rsx::vertex_arrays_changed)
	{
		m_draw_processor.analyse_inputs_interleaved(m_vertex_layout, current_vp_metadata);
	}
	else if (state_flags & rsx::vertex_base_changed)
	{
		// Rebase vertex bases instead of
		for (auto& info : m_vertex_layout.interleaved_blocks)
		{
			info->vertex_range.second = 0;
			const auto vertex_base_offset = rsx::method_registers.vertex_data_base_offset();
			info->real_offset_address = rsx::get_address(rsx::get_vertex_offset_from_base(vertex_base_offset, info->base_offset), info->memory_location);
		}
	}
	else
	{
		// Discard cached results
		for (auto& info : m_vertex_layout.interleaved_blocks)
		{
			info->vertex_range.second = 0;
		}
	}

	if ((state_flags & vertex_state_mask) && !m_vertex_layout.validate())
	{
		// No vertex inputs enabled
		// Execute remainining pipeline barriers with NOP draw
		do
		{
			draw_call.execute_pipeline_dependencies(m_ctx);
		}
		while (draw_call.next());

		draw_call.end();
		return;
	}

	// Programs data is dependent on vertex state
	auto upload_info = upload_vertex_data();
	if (!upload_info.vertex_draw_count)
	{
		// Malformed vertex setup; abort
		return;
	}

	m_frame_stats.vertex_upload_time += m_profiler.duration();

	// Faults are allowed during vertex upload. Ensure consistent CB state after uploads.
	// Queries are spawned and closed outside render pass scope for consistency reasons.
	if (m_current_command_buffer->flags & vk::command_buffer::cb_load_occluson_task)
	{
		u32 occlusion_id = m_occlusion_query_manager->allocate_query(*m_current_command_buffer);
		if (occlusion_id == umax)
		{
			// Force flush
			rsx_log.warning("[Performance Warning] Out of free occlusion slots. Forcing hard sync.");
			ZCULL_control::sync(this);

			occlusion_id = m_occlusion_query_manager->allocate_query(*m_current_command_buffer);
			if (occlusion_id == umax)
			{
				//rsx_log.error("Occlusion pool overflow");
				if (m_current_task) m_current_task->result = 1;
			}
		}

		// Before starting a query, we need to match RP scope (VK_1_0 rules).
		// We always want our queries to start outside a renderpass whenever possible.
		// We ignore this for performance reasons whenever possible of course and only do this for sensitive drivers.
		if (vk::use_strict_query_scopes() &&
			vk::is_renderpass_open(*m_current_command_buffer))
		{
			vk::end_renderpass(*m_current_command_buffer);
			emergency_query_cleanup(m_current_command_buffer);
		}

		// Begin query
		m_occlusion_query_manager->begin_query(*m_current_command_buffer, occlusion_id);

		auto& data = m_occlusion_map[m_active_query_info->driver_handle];
		data.indices.push_back(occlusion_id);
		data.set_sync_command_buffer(m_current_command_buffer);

		m_current_command_buffer->flags &= ~vk::command_buffer::cb_load_occluson_task;
		m_current_command_buffer->flags |= (vk::command_buffer::cb_has_occlusion_task | vk::command_buffer::cb_has_open_query);
	}

	VkDescriptorBufferViewEx persistent_buffer = m_persistent_attribute_storage ? *m_persistent_attribute_storage : *null_buffer_view;
	VkDescriptorBufferViewEx volatile_buffer = m_volatile_attribute_storage ? *m_volatile_attribute_storage : *null_buffer_view;
	bool update_descriptors = false;
	bool vr_render = rsx::vr::camera_probe::get().render_enabled() && m_vr_right_draw_fbo &&
		!draw_call.is_trivial_instanced_draw;
	// Gate 6: batch the right-eye draw into the current left pass's right-eye batch.
	// Programmable blending (input attachments) and conditional rendering keep the
	// per-draw replay: neither carries over into a secondary command buffer here.
	const bool vr_batch = vr_render && m_vr_batching &&
		!(current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING) &&
		!cond_render_ctrl.hw_cond_active;
	u32 vr_query_continuation = umax;
	const bool vr_suspend_query = vr_render && !vr_batch &&
		(m_current_command_buffer->flags & vk::command_buffer::cb_has_open_query);
	if (vr_suspend_query)
	{
		// A right-eye replay must not contribute samples to the guest's occlusion
		// result. Reserve another slot now so the guest query can be split into
		// two left-eye-only segments around the host draw.
		vr_query_continuation = m_occlusion_query_manager->allocate_query(*m_current_command_buffer);
		if (vr_query_continuation == umax)
		{
			// Query slots are finite. Preserve guest semantics and omit stereo for
			// this draw rather than allowing the right eye to alter its result.
			vr_render = false;
		}
	}
	const VkDescriptorBufferInfoEx guest_constants_info = m_vertex_constants_buffer_info;
	const u64 guest_constants_dynamic_offset = m_xform_constants_dynamic_offset;
	const u64 guest_constants_source_offset = guest_constants_info.offset + guest_constants_dynamic_offset;

	if (m_current_draw.subdraw_id == 0)
	{
		update_descriptors = true;

		// Allocate stream layout memory for this batch
		const u64 alloc_size = rsx::method_registers.current_draw_clause.pass_count() * 168 * (vr_render ? 2 : 1);
		m_vertex_layout_dynamic_offset = m_vertex_layout_ring_info.alloc<8>(alloc_size);
	}

	const bool vr_camera_draw = vr_render && bind_vr_eye_constants(-1.f, guest_constants_source_offset, m_xform_constants_data_size);
	m_vr_camera_draws += vr_camera_draw;
	if (vr_render && vk::xr::is_running())
	{
		if (vr_tracing())
		{
			vr_trace_copy_reads(vr_camera_draw);
			if (!vr_camera_draw && !m_vr_camera_targets.empty() && m_framebuffer_layout.color_addresses[0] == m_vr_camera_targets.back())
			{
				if (m_vr_trace_cam_count)
				{
					vr_trace_flush_cam();
				}
				m_vr_trace_other_count++;
			}
		}
		// A camera draw stamps its targets with the pose it is rotated by; any other
		// draw passes on what it samples (post-processing, the final composite). A pass
		// that samples no traced target made its output now (ICO turns this frame's
		// stencil shadow volumes into a shadow mask that way): it is current too, except
		// in a display buffer, where such draws are the HUD over an older image.
		u32 vr_pose = vr_camera_draw ? m_vr_applied_pose : vr_sampled_pose();
		if (!vr_pose && m_vr_display_target < 0)
		{
			vr_pose = m_vr_applied_pose;
		}
		vr_stamp_targets(vr_pose, vr_camera_draw);
		if (vr_camera_draw && vr_tracing())
		{
			// Reads by 3D draws (traced only; they keep their own stamp).
			const std::string before = m_vr_trace;
			vr_sampled_pose();
			if (m_vr_trace != before)
			{
				m_vr_trace.insert(before.size(), " 3d");
			}
			if (m_vr_trace_other_count || (m_vr_trace_cam_count && m_vr_trace_cam_pose != m_vr_applied_pose))
			{
				vr_trace_flush_cam();
			}
			m_vr_trace_cam_pose = m_vr_applied_pose;
			m_vr_trace_cam_count++;
		}
	}

	// HUD/menu drawn without a matrix: its own vertex context per eye (restored below).
	const bool vr_hud = vr_render && !vr_camera_draw && vk::xr::is_running() && vr_is_passthrough_hud();
	const VkDescriptorBufferInfoEx vr_saved_env_info = m_vertex_env_buffer_info;
	const u64 vr_saved_env_offset = m_vertex_env_dynamic_offset;
	// Sprites the game projected itself (ICO's flames): through the camera's eye transform.
	// Only into this frame's scene with depth test: the same program also draws ICO's pause
	// menu, which must stay in the HUD box (or as drawn).
	const u32 vr_target = m_framebuffer_layout.color_addresses[0];
	const u64 vr_listed = vr_render && !vr_camera_draw ? vr_preprojected_program() : 0;
	const bool vr_in_scene = vr_target && std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), vr_target) != m_vr_camera_targets.end();
	const u64 vr_preprojected = vr_listed && !vr_hud && rsx::method_registers.depth_test_enabled() && vr_in_scene ? vr_listed : 0;
	if (vr_listed)
	{
		// Diagnostic: each distinct way a listed program is drawn (first 16).
		static std::set<u64> s_seen;
		const u64 key = (u64{vr_target} << 3) | (rsx::method_registers.depth_test_enabled() ? 4 : 0) | (vr_hud ? 2 : 0) | (vr_in_scene ? 1 : 0);
		if (s_seen.size() < 16 && s_seen.insert(key).second)
		{
			rsx_log.notice("VR: pre-projected program %016llx into 0x%x: depth test %d, HUD %d, scene %d -> %s", vr_listed, vr_target,
				rsx::method_registers.depth_test_enabled(), vr_hud, vr_in_scene, vr_preprojected ? "eye transform" : vr_hud ? "HUD box" : "as drawn");
		}
	}
	const bool vr_hud_env = (vr_hud || vr_preprojected) && vr_hud_vertex_env(-1.f, vr_preprojected);

	// Update vertex fetch parameters
	update_vertex_env(vr_render ? sub_index * 2 : sub_index, upload_info);

	if (update_descriptors)
	{
		m_program->bind_uniform(persistent_buffer, vk::glsl::binding_set_index_vertex, m_vs_binding_table->vertex_buffers_location);
		m_program->bind_uniform(volatile_buffer, vk::glsl::binding_set_index_vertex, m_vs_binding_table->vertex_buffers_location + 1);
	}

	bool reload_state = (!m_current_draw.subdraw_id++);
	vk::renderpass_op(*m_current_command_buffer, [&](const vk::command_buffer& cmd, VkRenderPass pass, VkFramebuffer fbo)
	{
		if (get_render_pass() == pass && m_draw_fbo->value == fbo)
		{
			// Nothing to do
			return;
		}

		if (pass)
		{
			// Subpass mismatch, end it before proceeding
			vk::end_renderpass(cmd);
		}

		// Starting a new renderpass should clobber dynamic state
		m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;

		reload_state = true;
	});

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		// Subpass inter-draw dependency for input attachment reads. Preserves open renderpasses.
		for (u32 i = 0; i < current_fragment_program.mrt_buffers_count; ++i)
		{
			vk::insert_image_memory_barrier(
				*m_current_command_buffer,
				m_fbo_images[i]->value,
				m_fbo_images[i]->current_layout,
				m_fbo_images[i]->current_layout,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
				{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
				true,
				VK_DEPENDENCY_BY_REGION_BIT
			);
		}
	}

	// Bind both pipe and descriptors in one go
	// FIXME: We only need to rebind the pipeline when reload state is set. Flags?
	m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);

	if (reload_state)
	{
		update_draw_state();
		begin_render_pass();

		if (cond_render_ctrl.hw_cond_active && m_device->get_conditional_render_support())
		{
			// It is inconvenient that conditional rendering breaks other things like compute dispatch
			// TODO: If this is heavy, add refactor the resources into global and add checks around compute dispatch
			VkConditionalRenderingBeginInfoEXT info{};
			info.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT;
			info.buffer = m_cond_render_buffer->value;

			_vkCmdBeginConditionalRenderingEXT(*m_current_command_buffer, &info);
			m_current_command_buffer->flags |= vk::command_buffer::cb_has_conditional_render;
		}
	}

	// Bind the new set of descriptors for use with this draw call
	m_frame_stats.setup_time += m_profiler.duration();

	// VR fork: capture per-draw stereo evidence. This is the last point at which
	// program, constants, framebuffer layout and subdraw state all correspond to
	// the draw that is about to execute. Placing this at the constant-upload site
	// instead would miss draws that reuse an unchanged constant allocation.
	// No-op unless RPCS3_STEREO_INSPECT is set and the frame is armed.
	if (auto& inspector = rsx::vr::stereo_inspector::get(); inspector.capturing())
	{
		rsx::vr::draw_capture_input capture_in;
		capture_in.subdraw_index       = sub_index;
		capture_in.vertex_draw_count   = upload_info.vertex_draw_count;
		capture_in.indexed             = !!upload_info.index_info;
		capture_in.pass_count          = draw_call.pass_count();
		capture_in.vertex_program      = &current_vertex_program;
		capture_in.fragment_program    = &current_fragment_program;
		capture_in.framebuffer         = &m_framebuffer_layout;

		if (m_vertex_prog)
		{
			capture_in.constant_ids          = &m_vertex_prog->constant_ids;
			capture_in.has_indexed_constants = m_vertex_prog->has_indexed_constants;
			capture_in.vp_session_id         = m_vertex_prog->id;
		}
		if (m_fragment_prog)
		{
			capture_in.fp_session_id         = m_fragment_prog->id;
		}

		inspector.record_draw(capture_in);
	}

	// VR profile generation (home menu): sample this draw's vertex constants.
	if (auto& generator = rsx::vr::profile_generator::get(); generator.sampling() && m_vertex_prog)
	{
		const bool full_bank = m_shader_interpreter.is_interpreter(m_program) || m_vertex_prog->has_indexed_constants;
		generator.record_draw(full_bank ? std::span<const u16>{} : std::span<const u16>(m_vertex_prog->constant_ids),
			m_vertex_prog->id, m_framebuffer_layout.width, m_framebuffer_layout.height);
	}

	// Keep Vulkan command emission in one host-only callable. Gate 5 invokes it
	// for the guest-authoritative left target and, when armed, once more for the
	// isolated right target. Guest draw/FIFO/statistics accounting stays outside.
	const auto emit_vulkan_draw = [&]()
	{
		if (!upload_info.index_info)
		{
			if (draw_call.is_trivial_instanced_draw)
			{
				vkCmdDraw(*m_current_command_buffer, upload_info.vertex_draw_count, draw_call.pass_count(), 0, 0);
			}
			else if (draw_call.is_single_draw())
			{
				vkCmdDraw(*m_current_command_buffer, upload_info.vertex_draw_count, 1, 0, 0);
			}
			else if (m_device->get_multidraw_support())
			{
				const auto subranges = draw_call.get_subranges();
				auto ptr = utils::bless<const VkMultiDrawInfoEXT>(& subranges.front().first);
				_vkCmdDrawMultiEXT(*m_current_command_buffer, ::size32(subranges), ptr, 1, 0, sizeof(rsx::draw_range_t));
			}
			else
			{
				u32 vertex_offset = 0;
				for (const auto &range : draw_call.get_subranges())
				{
					vkCmdDraw(*m_current_command_buffer, range.count, 1, vertex_offset, 0);
					vertex_offset += range.count;
				}
			}
		}
		else
		{
			const VkIndexType index_type = std::get<1>(*upload_info.index_info);
			const VkDeviceSize offset = std::get<0>(*upload_info.index_info);

			vkCmdBindIndexBuffer(*m_current_command_buffer, m_index_buffer_ring_info.heap->value, offset, index_type);

		if (draw_call.is_trivial_instanced_draw)
		{
			vkCmdDrawIndexed(*m_current_command_buffer, upload_info.vertex_draw_count, draw_call.pass_count(), 0, 0, 0);
		}
		else if (rsx::method_registers.current_draw_clause.is_single_draw())
		{
			vkCmdDrawIndexed(*m_current_command_buffer, upload_info.vertex_draw_count, 1, 0, 0, 0);
		}
		else if (m_device->get_multidraw_support())
		{
			const auto subranges = draw_call.get_subranges();
			const auto subranges_count = ::size32(subranges);
			const auto allocation_size = subranges_count * sizeof(VkMultiDrawIndexedInfoEXT);

			m_multidraw_parameters_buffer.resize(allocation_size);
			auto base_ptr = utils::bless<VkMultiDrawIndexedInfoEXT>(m_multidraw_parameters_buffer.data());

			u32 vertex_offset = 0;
			auto _ptr = base_ptr;

			for (const auto& range : subranges)
			{
				const auto count = get_index_count(draw_call.primitive, range.count);
				_ptr->firstIndex = vertex_offset;
				_ptr->indexCount = count;
				_ptr->vertexOffset = 0;

				_ptr++;
				vertex_offset += count;
			}
			_vkCmdDrawMultiIndexedEXT(*m_current_command_buffer, subranges_count, base_ptr, 1, 0, sizeof(VkMultiDrawIndexedInfoEXT), nullptr);
		}
		else
		{
			u32 vertex_offset = 0;
			const auto subranges = draw_call.get_subranges();
			for (const auto &range : subranges)
			{
				const auto count = get_index_count(draw_call.primitive, range.count);
				vkCmdDrawIndexed(*m_current_command_buffer, count, 1, vertex_offset, 0, 0);
				vertex_offset += count;
			}
		}
		}
	};

	emit_vulkan_draw();

	if (vr_render && vr_batch)
	{
		auto* const left_fbo = m_draw_fbo;
		auto left_images = std::move(m_fbo_images);
		m_draw_fbo = m_vr_right_draw_fbo;
		m_fbo_images = m_vr_right_fbo_images;

		bind_vr_eye_constants(1.f, guest_constants_source_offset, m_xform_constants_data_size);
		// On the primary: a barrier here ends the left pass, which runs the batch first.
		bind_texture_env(true);

		if (vr_batch_begin(get_render_pass(), m_vr_right_draw_fbo))
		{
			auto* const primary = m_current_command_buffer;
			m_current_command_buffer = &m_vr_batch_cb;
			m_vr_batch_cb.flags |= vk::command_buffer::cb_reload_dynamic_state;
			if (vr_hud_env)
			{
				vr_hud_vertex_env(1.f, vr_preprojected);
			}
			update_vertex_env(sub_index * 2 + 1, upload_info);
			m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
			update_draw_state();
			emit_vulkan_draw();
			m_current_command_buffer = primary;
			m_vr_right_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);

			// A batch stays open only while the left pass is open.
			if (!vk::is_renderpass_open(*m_current_command_buffer))
			{
				vr_batch_execute();
			}
		}

		m_draw_fbo = left_fbo;
		m_fbo_images = std::move(left_images);
		m_vertex_constants_buffer_info = guest_constants_info;
		m_xform_constants_dynamic_offset = guest_constants_dynamic_offset;
		if (m_vs_binding_table->cbuf_location != umax)
		{
			m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex,
				m_vs_binding_table->cbuf_location);
		}
		bind_texture_env(false);
	}
	else if (vr_render)
	{
		vk::end_renderpass(*m_current_command_buffer);
		if (vr_suspend_query)
		{
			auto& query_data = m_occlusion_map[m_active_query_info->driver_handle];
			m_occlusion_query_manager->end_query(*m_current_command_buffer, query_data.indices.back());
			m_current_command_buffer->flags &= ~vk::command_buffer::cb_has_open_query;
		}

		auto* const left_fbo = m_draw_fbo;
		auto left_images = std::move(m_fbo_images);
		m_draw_fbo = m_vr_right_draw_fbo;
		m_fbo_images = m_vr_right_fbo_images;

		bind_vr_eye_constants(1.f, guest_constants_source_offset, m_xform_constants_data_size);
		if (vr_hud_env)
		{
			vr_hud_vertex_env(1.f, vr_preprojected);
		}
		update_vertex_env(sub_index * 2 + 1, upload_info);
		bind_texture_env(true);
		m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
		update_draw_state();
		begin_render_pass();
		emit_vulkan_draw();
		m_vr_right_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);
		vk::end_renderpass(*m_current_command_buffer);

		m_draw_fbo = left_fbo;
		m_fbo_images = std::move(left_images);
		// Restore the guest-authored allocation. Pipeline dependency processing
		// for later RSX draws must never inherit either host eye's constants.
		m_vertex_constants_buffer_info = guest_constants_info;
		m_xform_constants_dynamic_offset = guest_constants_dynamic_offset;
		if (m_vs_binding_table->cbuf_location != umax)
		{
			m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex,
				m_vs_binding_table->cbuf_location);
		}
		bind_texture_env(false);

		if (vr_suspend_query)
		{
			// Continue the same guest query after the host-only right-eye draw.
			// Result collection already sums every slot recorded for the query.
			m_occlusion_query_manager->begin_query(*m_current_command_buffer, vr_query_continuation);
			auto& query_data = m_occlusion_map[m_active_query_info->driver_handle];
			query_data.indices.push_back(vr_query_continuation);
			query_data.set_sync_command_buffer(m_current_command_buffer);
			m_current_command_buffer->flags |=
				(vk::command_buffer::cb_has_occlusion_task | vk::command_buffer::cb_has_open_query);
		}

		m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
		update_draw_state();
		begin_render_pass();
	}

	if (vr_hud_env)
	{
		// Later draws use the guest's own viewport again.
		m_vertex_env_buffer_info = vr_saved_env_info;
		m_vertex_env_dynamic_offset = vr_saved_env_offset;
		m_program->bind_uniform(m_vertex_env_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->context_buffer_location);
	}

	m_frame_stats.draw_exec_time += m_profiler.duration();
}

void VKGSRender::begin()
{
	// Save shader state now before prefetch and loading happens
	m_interpreter_state = (m_graphics_state.load() & rsx::pipeline_state::invalidate_pipeline_bits);

	rsx::thread::begin();

	if (skip_current_frame ||
		swapchain_unavailable ||
		cond_render_ctrl.disable_rendering())
	{
		return;
	}

	init_buffers(rsx::framebuffer_creation_context::context_draw);

	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		// Shaders need to be reloaded.
		m_prev_program = m_program;
		m_program = nullptr;
	}
}

void VKGSRender::end()
{
	if (skip_current_frame || !m_graphics_state.test(rsx::rtt_config_valid) || swapchain_unavailable || cond_render_ctrl.disable_rendering())
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	m_profiler.start();

	// Check for frame resource status here because it is possible for an async flip to happen between begin/end
	if (m_current_frame->flags & frame_context_state::dirty) [[unlikely]]
	{
		check_present_status();

		if (m_current_frame->swap_command_buffer) [[unlikely]]
		{
			// Borrow time by using the auxilliary context
			m_aux_frame_context.grab_resources(*m_current_frame);
			m_current_frame = &m_aux_frame_context;
		}

		ensure(!m_current_frame->swap_command_buffer);

		m_current_frame->flags &= ~frame_context_state::dirty;
	}

	analyse_current_rsx_pipeline();

	m_frame_stats.setup_time += m_profiler.duration();

	load_texture_env();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	if (!load_program())
	{
		// Program is not ready, skip drawing this
		std::this_thread::yield();
		execute_nop_draw();
		// m_rtts.on_write(); - breaks games for obvious reasons
		rsx::thread::end();
		return;
	}

	// Load program execution environment
	load_program_env();
	m_frame_stats.setup_time += m_profiler.duration();

	if (vk::xr::is_running() && m_vr_applied_pose && rsx::method_registers.blend_enabled())
	{
		vr_realign_blend_targets();
	}

	// Apply write memory barriers
	if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil))
	{
		ds->write_barrier(*m_current_command_buffer);

		if (m_graphics_state.test(rsx::zeta_address_cyclic_barrier) &&
			ds->current_layout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			// We actually need to end the subpass as a minimum. Without this, early-Z optimiazations in following draws will clobber reads from previous draws and cause flickering.
			// Since we're ending the subpass, might as well restore DCC/HiZ for extra performance
			ds->change_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
			ds->reset_surface_counters();

			// Regenerate render pass key
			invalidate_render_pass();
		}
	}

	for (auto &rtt : m_rtts.m_bound_render_targets)
	{
		if (auto surface = std::get<1>(rtt))
		{
			surface->write_barrier(*m_current_command_buffer);
		}
	}

	m_graphics_state.clear(rsx::zeta_address_cyclic_barrier);

	m_frame_stats.setup_time += m_profiler.duration();

	// Now bind the shader resources. It is important that this takes place after the barriers so that we don't end up with stale descriptors
	for (int retry = 0; retry < 3; ++retry)
	{
		if (retry > 0 && m_samplers_dirty) [[ unlikely ]]
		{
			// Reload texture env if referenced objects were invalidated during OOM handling.
			load_texture_env();

			// Do not trust fragment/vertex texture state after a texture state reset.
			// NOTE: We don't want to change the program - it's too late for that now. We just need to harmonize the state.
			m_graphics_state |= rsx::vertex_program_state_dirty | rsx::fragment_program_state_dirty;
			get_current_fragment_program(fs_sampler_state);
			get_current_vertex_program(vs_sampler_state);
			m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
		}

		const bool out_of_memory = m_shader_interpreter.is_interpreter(m_program)
			? bind_interpreter_texture_env()
			: bind_texture_env();

		if (!out_of_memory)
		{
			break;
		}

		// Handle OOM
		if (!on_vram_exhausted(rsx::problem_severity::fatal))
		{
			// It is not possible to free memory. Just use placeholder textures. Can cause graphics glitches but shouldn't crash otherwise
			break;
		}
	}

	m_texture_cache.release_uncached_temporary_subresources();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	u32 sub_index = 0;               // RSX subdraw ID
	m_current_draw.subdraw_id = 0;   // Host subdraw ID. Invalid RSX subdraws do not increment this value

	if (m_graphics_state & rsx::pipeline_state::invalidate_vk_dynamic_state)
	{
		m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
	}

	// VR fork: advance the logical RSX draw ordinal for the stereo inspector.
	// One logical clause can produce several emitted subdraws, so the ordinal is
	// assigned here and the subdraw index is recorded separately.
	rsx::vr::stereo_inspector::get().begin_draw_clause();

	auto& draw_call = rsx::method_registers.current_draw_clause;
	draw_call.begin();
	do
	{
		emit_geometry(sub_index++);

		if (draw_call.is_trivial_instanced_draw)
		{
			// We already completed. End the draw.
			draw_call.end();
		}
	}
	while (draw_call.next());

	if (m_current_command_buffer->flags & vk::command_buffer::cb_has_conditional_render)
	{
		_vkCmdEndConditionalRenderingEXT(*m_current_command_buffer);
		m_current_command_buffer->flags &= ~(vk::command_buffer::cb_has_conditional_render);
	}

	m_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);

	rsx::thread::end();
}

u32 VKGSRender::vr_sampled_pose()
{
	u32 pose = 0;
	std::string reads;
	const bool trace = vr_tracing();
	const auto stamp = [&](vk::image* image)
	{
		if (auto* rtt = dynamic_cast<vk::render_target*>(image))
		{
			pose = std::max(pose, vr_is_feedback_texture(rtt) ? m_vr_applied_pose : rtt->vr_pose);
			if (trace)
			{
				reads += rtt->vr_pose ? fmt::format("%s%x:%d", reads.empty() ? "" : ",", rtt->base_addr, static_cast<s32>(m_vr_applied_pose - rtt->vr_pose))
					: fmt::format("%s%x:-", reads.empty() ? "" : ",", rtt->base_addr);
			}
		}
	};

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		if (!(textures_ref & 1) || !sampler_state || sampler_state->upload_context != rsx::texture_upload_context::framebuffer_storage)
		{
			continue;
		}

		if (sampler_state->image_handle)
		{
			stamp(sampler_state->image_handle->image());
		}
		else
		{
			// A copy of (parts of) render targets.
			const auto& desc = sampler_state->external_subresource_desc;
			if (desc.external_handle)
			{
				stamp(desc.external_handle);
			}
			for (const auto& section : desc.sections_to_copy)
			{
				stamp(section.src);
			}
		}
	}

	if (trace && !reads.empty())
	{
		// Reads of render targets as address:frames-old (- = no 3D content traced).
		const std::string entry = fmt::format(" S{%s}", reads);
		if (!m_vr_trace.ends_with(entry))
		{
			vr_trace_flush_cam();
			m_vr_trace += entry;
		}
	}
	return pose;
}

void VKGSRender::vr_stamp_targets(u32 pose, bool camera)
{
	if (!pose)
	{
		return;
	}

	// A camera draw's image is as new as its pose. Another draw's output is at least
	// as new as what it samples (it may blend into newer content already there).
	const auto stamp = [&](vk::render_target* surface)
	{
		if (surface)
		{
			surface->vr_pose = camera ? pose : std::max(surface->vr_pose, pose);
		}
	};
	for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
	{
		stamp(std::get<1>(m_rtts.m_bound_render_targets[index]));
		if (camera)
		{
			const u32 address = m_framebuffer_layout.color_addresses[index];
			if (address && (m_vr_camera_targets.empty() || m_vr_camera_targets.back() != address))
			{
				std::erase(m_vr_camera_targets, address);
				m_vr_camera_targets.push_back(address);
				if (m_vr_camera_targets.size() > 8)
				{
					m_vr_camera_targets.erase(m_vr_camera_targets.begin());
				}
			}
		}
	}
	if (camera)
	{
		stamp(std::get<1>(m_rtts.m_bound_depth_stencil));
	}
}

bool VKGSRender::vr_is_feedback_texture(const vk::render_target* rtt) const
{
	// Drawn with an older pose than this frame's, and a full-screen image: same aspect
	// as the target being drawn, at most 8x smaller or larger (downsampled glow chains).
	if (!rtt || !rtt->vr_pose || !m_vr_applied_pose || rtt->vr_pose >= m_vr_applied_pose)
	{
		return false;
	}
	const f32 tw = rtt->get_surface_width<rsx::surface_metrics::pixels>();
	const f32 th = rtt->get_surface_height<rsx::surface_metrics::pixels>();
	const f32 ow = m_framebuffer_layout.width;
	const f32 oh = m_framebuffer_layout.height;
	if (tw <= 0.f || th <= 0.f || ow <= 0.f || oh <= 0.f)
	{
		return false;
	}
	const f32 aspect = (tw / th) / (ow / oh);
	return aspect > 0.95f && aspect < 1.05f && tw * 8.f >= ow && ow * 8.f >= tw;
}

bool VKGSRender::vr_shift_feedback_textures(rsx::fragment_program_texture_config& params)
{
	bool shifted = false;
	m_vr_params_extra_mask = 0;
	u32 homography_slot = umax; // slot holding the homography of homography_pose
	u32 homography_pose = 0;
	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		if (!(textures_ref & 1) || !sampler_state || sampler_state->upload_context != rsx::texture_upload_context::framebuffer_storage)
		{
			continue;
		}

		vk::image* image = sampler_state->image_handle ? sampler_state->image_handle->image() :
			sampler_state->external_subresource_desc.external_handle ? sampler_state->external_subresource_desc.external_handle :
			!sampler_state->external_subresource_desc.sections_to_copy.empty() ? sampler_state->external_subresource_desc.sections_to_copy.front().src : nullptr;
		const auto* rtt = dynamic_cast<const vk::render_target*>(image);
		f32 du = 0.f, dv = 0.f;
		if (!vr_is_feedback_texture(rtt) || !vk::xr::render_pose_shift(rtt->vr_pose, m_vr_applied_pose, du, dv))
		{
			continue;
		}

		static const f32 s_flip_v = []()
		{
			const char* v = std::getenv("RPCS3_VR_FEEDBACK_FLIP_V");
			return v && v[0] == '1' ? -1.f : 1.f;
		}();
		dv *= s_flip_v;

		if (!shifted)
		{
			params = current_fragment_program.texture_params;
			shifted = true;
		}

		// Exact: a homography in a slot the program does not use (one per source pose).
		if (homography_pose != rtt->vr_pose)
		{
			homography_slot = umax;
			const u16 used = current_fp_metadata.referenced_textures_mask | m_vr_params_extra_mask;
			for (u32 j = 15; j < 16; --j)
			{
				if (!(used & (1u << j)))
				{
					f32 h[9];
					if (vk::xr::render_pose_homography(rtt->vr_pose, m_vr_applied_pose, h))
					{
						auto& slot = params[j];
						slot.scale[0] = h[0]; slot.scale[1] = h[1]; slot.scale[2] = h[2];
						slot.bias[0] = h[3]; slot.bias[1] = h[4]; slot.bias[2] = h[5];
						slot.clamp_min[0] = h[6]; slot.clamp_min[1] = h[7]; slot.clamp_max[0] = h[8];
						m_vr_params_extra_mask |= static_cast<u16>(1u << j);
						homography_slot = j;
						homography_pose = rtt->vr_pose;
					}
					break;
				}
			}
		}

		if (homography_slot != umax)
		{
			params[i].bias[2] = static_cast<f32>(homography_slot);
			params[i].control |= (1u << rsx::texture_control_bits::VR_REPROJECT_BIT);
		}
		else
		{
			// No free slot: the shift that is exact at the centre of the view.
			params[i].bias[0] += du;
			params[i].bias[1] += dv;
		}

		if (vr_tracing())
		{
			vr_trace_flush_cam();
			m_vr_trace += fmt::format(" X{%x:%d %.4f,%.4f%s}", rtt->base_addr, static_cast<s32>(m_vr_applied_pose - rtt->vr_pose), du, dv,
				homography_slot != umax ? " h" : "");
		}
	}
	return shifted;
}

void VKGSRender::vr_trace_copy_reads(bool camera)
{
	// TEMPORARY diagnostic. K{address:context:age}: a texture that is not a live render
	// target view but holds render-target memory (a blit/DMA copy, or an ordinary texture
	// over a surface), with the frames since that surface's pose (- = none traced).
	std::string reads;
	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		if (!(textures_ref & 1) || !sampler_state || sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage)
		{
			continue;
		}

		const auto& tex = rsx::method_registers.fragment_textures[i];
		const u32 address = rsx::get_address(tex.offset(), tex.location());
		const auto* rtt = m_rtts.find_color_surface(address, tex.pitch());
		if (!rtt)
		{
			rtt = m_rtts.get_surface_at(address);
		}
		if (sampler_state->upload_context == rsx::texture_upload_context::shader_read && !rtt)
		{
			// Any screen-shaped ordinary texture (an image made elsewhere, e.g. on the SPUs).
			const f32 w = tex.width(), h = tex.height();
			if (w >= 64.f && h >= 32.f && w / h > 1.6f && w / h < 1.9f)
			{
				reads += fmt::format("%sM%x:%ux%u", reads.empty() ? "" : ",", address, tex.width(), tex.height());
			}
			continue;
		}

		reads += fmt::format("%s%x:%u:%s", reads.empty() ? "" : ",", address, static_cast<u32>(sampler_state->upload_context),
			rtt && rtt->vr_pose ? std::to_string(static_cast<s32>(m_vr_applied_pose - rtt->vr_pose)) : std::string("-"));
	}

	if (!reads.empty())
	{
		const std::string entry = fmt::format(" %sK{%s}", camera ? "3d" : "", reads);
		if (!m_vr_trace.ends_with(entry))
		{
			vr_trace_flush_cam();
			m_vr_trace += entry;
		}
	}
}

void VKGSRender::vr_realign_blend_targets()
{
	// Full-screen: the same aspect as the latest camera target, at most 8x smaller.
	const vk::render_target* scene = m_vr_camera_targets.empty() ? nullptr : m_rtts.get_surface_at(m_vr_camera_targets.back());
	if (!scene)
	{
		return;
	}
	const f32 sw = scene->get_surface_width<rsx::surface_metrics::pixels>();
	const f32 sh = scene->get_surface_height<rsx::surface_metrics::pixels>();

	for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
	{
		auto* surface = std::get<1>(m_rtts.m_bound_render_targets[index]);
		if (!surface || !surface->vr_pose || surface->vr_pose >= m_vr_applied_pose || surface->samples() != 1)
		{
			continue;
		}

		const f32 tw = surface->get_surface_width<rsx::surface_metrics::pixels>();
		const f32 th = surface->get_surface_height<rsx::surface_metrics::pixels>();
		const f32 aspect = sw > 0.f && sh > 0.f && th > 0.f ? (tw / th) / (sw / sh) : 0.f;
		f32 du = 0.f, dv = 0.f;
		if (aspect < 0.95f || aspect > 1.05f || tw * 8.f < sw || tw > sw * 1.05f ||
			!vk::xr::render_pose_shift(surface->vr_pose, m_vr_applied_pose, du, dv))
		{
			if (vr_tracing())
			{
				vr_trace_flush_cam();
				m_vr_trace += fmt::format(" D{%x:%d}", surface->base_addr, static_cast<s32>(m_vr_applied_pose - surface->vr_pose));
			}
			continue;
		}

		// new(x) = old(x + dx): the content moves against the head rotation.
		const auto shift = [&](vk::image* image)
		{
			const int w = static_cast<int>(image->width());
			const int h = static_cast<int>(image->height());
			const int dx = static_cast<int>(std::lround(du * w));
			const int dy = static_cast<int>(std::lround(dv * h));
			if ((!dx && !dy) || std::abs(dx) >= w || std::abs(dy) >= h)
			{
				return std::pair<int, int>{ dx, dy };
			}
			if (image->current_layout == VK_IMAGE_LAYOUT_UNDEFINED)
			{
				// Never written (a right-eye surface): nothing to move.
				return std::pair<int, int>{ dx, dy };
			}
			auto* scratch = vk::get_typeless_helper(image->format(), image->format_class(), w, h);
			if (scratch->current_layout == VK_IMAGE_LAYOUT_UNDEFINED)
			{
				// copy_image returns the image to its prior layout, which must be a real one
				// (a freshly created helper is UNDEFINED: "invalid layout" fatal error).
				if (vk::is_renderpass_open(*m_current_command_buffer))
				{
					vk::end_renderpass(*m_current_command_buffer);
				}
				vk::change_image_layout(*m_current_command_buffer, scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			}
			vk::copy_image(*m_current_command_buffer, image, scratch, areai{ 0, 0, w, h }, areai{ 0, 0, w, h });
			const areai src{ std::max(dx, 0), std::max(dy, 0), w + std::min(dx, 0), h + std::min(dy, 0) };
			const areai dst{ std::max(-dx, 0), std::max(-dy, 0), w - std::max(dx, 0), h - std::max(dy, 0) };
			vk::copy_image(*m_current_command_buffer, scratch, image, src, dst);
			return std::pair<int, int>{ dx, dy };
		};

		const auto [dx, dy] = shift(surface);
		if (auto* right = m_vr_right_rtts.get_surface_at(surface->base_addr);
			right && right->width() == surface->width() && right->height() == surface->height() && right->samples() == 1)
		{
			vr_batch_flush();
			shift(right);
		}
		surface->vr_pose = m_vr_applied_pose;
		invalidate_render_pass();

		if (vr_tracing())
		{
			vr_trace_flush_cam();
			m_vr_trace += fmt::format(" A{%x:%dpx,%dpx}", surface->base_addr, dx, dy);
		}
	}
}

bool VKGSRender::vr_is_passthrough_hud()
{
	// Into a buffer no camera draw wrote (the finished frame or a display buffer), with
	// at least one ordinary texture and no colour render target (post-processing reads those).
	const u32 target = m_framebuffer_layout.color_addresses[0];
	if (!target || std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), target) != m_vr_camera_targets.end())
	{
		return false;
	}
	// Full frame or larger: smaller buffers are intermediate passes (ICO's shadow mask).
	const vk::render_target* scene = m_vr_camera_targets.empty() ? nullptr : m_rtts.get_surface_at(m_vr_camera_targets.back());
	if (!scene || m_framebuffer_layout.width * 20 < scene->get_surface_width<rsx::surface_metrics::pixels>() * 19)
	{
		return false;
	}

	bool ordinary = false;
	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		if (!(textures_ref & 1) || !sampler_state || !rsx::method_registers.fragment_textures[i].enabled())
		{
			continue;
		}
		if (sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage)
		{
			vk::image* image = sampler_state->image_handle ? sampler_state->image_handle->image() : sampler_state->external_subresource_desc.external_handle;
			if (!image || (image->aspect() & VK_IMAGE_ASPECT_COLOR_BIT))
			{
				return false;
			}
			continue;
		}
		ordinary = true;
	}
	return ordinary;
}

u64 VKGSRender::vr_preprojected_program()
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (!profile || profile->screen_space_preprojected_programs.empty())
	{
		return 0;
	}
	const u64 hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
	const auto& list = profile->screen_space_preprojected_programs;
	return std::find(list.begin(), list.end(), hash) != list.end() ? hash : 0;
}

bool VKGSRender::vr_hud_vertex_env(f32 eye_sign, u64 preprojected_program)
{
	f32 box[4][4];
	const f32 aspect = m_framebuffer_layout.height ? static_cast<f32>(m_framebuffer_layout.width) / m_framebuffer_layout.height : 16.f / 9.f;
	if (preprojected_program ? !rsx::vr::camera_probe::get().map_vr_preprojected(box, eye_sign, preprojected_program) :
		!rsx::vr::camera_probe::get().map_vr_passthrough_hud(box, eye_sign, aspect))
	{
		return false;
	}

	// The guest's viewport matrix V (vec4 k = output k's coefficients over x, y, z, w),
	// applied after the box: column k of the result is sum_c V[k][c] * box[.][c].
	alignas(16) f32 base[24];
	m_draw_processor.fill_scale_offset_data(base, false);
	f32 combined[16];
	for (u32 k = 0; k < 4; ++k)
	{
		for (u32 r = 0; r < 4; ++r)
		{
			f32 sum = 0.f;
			for (u32 c = 0; c < 4; ++c)
			{
				sum += base[k * 4 + c] * box[r][c];
			}
			combined[k * 4 + r] = sum;
		}
	}

	const auto* ctx = &rsx::method_registers;
	const auto& gpu_limits = m_device->gpu().get_limits();
	const auto mem = m_vertex_env_allocator->alloc();
	auto buf = m_vertex_env_ring_info.map<char>(mem, 96);
	std::memcpy(buf, combined, 64);
	m_draw_processor.fill_user_clip_data(buf + 64);
	*(reinterpret_cast<u32*>(buf + 68)) = ctx->transform_branch_bits();
	*(reinterpret_cast<f32*>(buf + 72)) = ctx->point_size() * resolution_scaling_config.scale_factor();
	*(reinterpret_cast<f32*>(buf + 76)) = ctx->clip_min();
	*(reinterpret_cast<f32*>(buf + 80)) = ctx->clip_max();
	m_vertex_env_ring_info.unmap();

	m_vertex_env_buffer_info = m_vertex_env_ring_info.window<256>(mem, 96, gpu_limits.maxUniformBufferRange);
	m_vertex_env_dynamic_offset = mem - m_vertex_env_buffer_info.offset;
	m_program->bind_uniform(m_vertex_env_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->context_buffer_location);

	static bool s_reported = false, s_reported_pre = false;
	if (preprojected_program ? !std::exchange(s_reported_pre, true) : !std::exchange(s_reported, true))
	{
		if (preprojected_program)
			rsx_log.success("VR: pre-projected program %016llx drawn through the camera's eye transform (target 0x%x).", preprojected_program, m_framebuffer_layout.color_addresses[0]);
		else
			rsx_log.success("VR: HUD drawn without a matrix is mapped into the HUD box (target 0x%x).", m_framebuffer_layout.color_addresses[0]);
	}
	return true;
}
