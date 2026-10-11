#include "stdafx.h"
#include "VKGSRender.h"
#include "VKDLSS.h"

#include "../Capture/rsx_camera_probe.h"
#include "../Capture/rsx_vr_hooks.h"
#include "../Common/surface_store.h"
#include "../rsx_methods.h"

// DLSS test: VKGSRender's side of VKDLSS.h (members declared in VKGSRenderVR.inl).

void VKGSRender::dlss_prepare_framebuffer(std::vector<vk::image*>*& fbo_images)
{
	const bool was_bound = m_dlss_motion_bound;
	m_dlss_motion_bound = false;

	if (m_dlss && !m_draw_buffers.empty() && m_fbo_images.size() >= m_draw_buffers.size())
	{
		const usz colors = m_draw_buffers.size();
		const u32 address = m_surface_info[m_draw_buffers[0]].address;
		const u32 mrt = rsx::utility::get_mrt_buffers_count(m_framebuffer_layout.target);
		const bool vr_active = m_vr_multiview || rsx::vr::camera_probe::get().render_enabled();
		const auto [fbo_width, fbo_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, m_framebuffer_layout.width, m_framebuffer_layout.height);

		if (vk::image* motion = m_dlss->bind_framebuffer(*m_current_command_buffer, address, m_fbo_images[0], fbo_width, fbo_height,
				static_cast<u32>(colors), mrt, vr_active))
		{
			// Colour attachments, the motion attachment, then depth (framebuffer::depth_format() reads the last one).
			m_dlss_fbo_images.assign(m_fbo_images.begin(), m_fbo_images.begin() + colors);
			m_dlss_fbo_images.push_back(motion);
			m_dlss_fbo_images.insert(m_dlss_fbo_images.end(), m_fbo_images.begin() + colors, m_fbo_images.end());
			fbo_images = &m_dlss_fbo_images;
			m_current_renderpass_key |= vk::dlss_renderpass_motion_bit;
			m_dlss_motion_bound = true;
		}
		else if (m_dlss->wants_motion(address, vr_active))
		{
			m_dlss_refused_address = address;
		}
	}

	if (was_bound != m_dlss_motion_bound)
	{
		// The programs of the draws into this target are the motion variants (and back).
		rsx::vr::set_dlss_motion_variant(m_dlss_motion_bound);
		m_graphics_state |= rsx::pipeline_state::fragment_program_state_dirty | rsx::pipeline_state::vertex_program_state_dirty |
			rsx::pipeline_state::pipeline_config_dirty;
	}
}

bool VKGSRender::dlss_framebuffer_stale() const
{
	if (!m_dlss || m_draw_buffers.empty())
	{
		return false;
	}
	const u32 address = m_surface_info[m_draw_buffers[0]].address;
	if (!m_dlss_motion_bound && address == m_dlss_refused_address)
	{
		return false;
	}
	const bool vr_active = m_vr_multiview || rsx::vr::camera_probe::get().render_enabled();
	return m_dlss->wants_motion(address, vr_active) != m_dlss_motion_bound;
}

void VKGSRender::dlss_note_draw()
{
	if (!m_draw_buffers.empty())
	{
		const bool full_bank = m_vertex_prog->has_indexed_constants;
		m_dlss->note_draw(m_surface_info[m_draw_buffers[0]].address,
			full_bank ? nullptr : m_vertex_prog->constant_ids.data(), full_bank ? 0 : m_vertex_prog->constant_ids.size());
	}
}

void VKGSRender::dlss_after_load_program()
{
	if (!m_dlss || !m_program || m_shader_interpreter.is_interpreter(m_program) || !m_vertex_prog)
	{
		return;
	}

	dlss_note_draw();

	// Each draw's motion block differs (its own matrix history): upload the constants for every draw into the scene.
	if (dlss_draw_block_size())
	{
		m_graphics_state |= rsx::pipeline_state::transform_constants_dirty;
	}

	if (m_dlss_motion_bound)
	{
		m_dlss->before_draw(*m_current_command_buffer);
	}

	m_dlss_in_draw = true; // until the end of the draw: its constant uploads (load_program_env, then one per sub-draw) are its own

	if (static const bool s_debug = std::getenv("RPCS3_DLSS_DEBUG") != nullptr; s_debug)
	{
		// The motion attachment follows the colour attachments.
		const auto& motion_blend = m_pipeline_properties.state.att_state[std::min<usz>(m_draw_buffers.size(), std::size(m_pipeline_properties.state.att_state) - 1)];
		rsx_log.notice("DLSS debug draw: bound %d, vp ctrl 0x%x, fp ctrl 0x%x, pass key 0x%llx, attachments %u, motion write mask 0x%x, blend %u",
			m_dlss_motion_bound, current_vertex_program.ctrl, current_fragment_program.ctrl, m_current_renderpass_key,
			m_pipeline_properties.state.cs.attachmentCount, motion_blend.colorWriteMask, motion_blend.blendEnable);
	}
}

usz VKGSRender::dlss_draw_block_size() const
{
	return (m_dlss && (current_vertex_program.ctrl & RSX_SHADER_CONTROL_DLSS_MOTION) && m_program && m_vertex_prog &&
		!m_shader_interpreter.is_interpreter(m_program)) ? vk::dlss_draw_block_size : 0;
}

void VKGSRender::dlss_write_draw_block(void* dst)
{
	if (!dlss_draw_block_size())
	{
		return;
	}

	if (!m_dlss_in_draw)
	{
		// patch_transform_constants() between draws: the game rewrote constants with this program bound. The next draw
		// uploads again with its own block; this copy gets a neutral one and leaves the matrix history alone. (Inside a
		// draw, the patches come from merged draws, the sub-draws, each a draw of its own with its own matrix.)
		m_dlss->write_neutral_block(dst);
		return;
	}

	// The surface clip, or the framebuffer's size (prepare_rtts) when it reads 0 (capture replay resets the registers).
	const u16 guest_width = rsx::method_registers.surface_clip_width() ? rsx::method_registers.surface_clip_width() : static_cast<u16>(m_framebuffer_layout.width);
	const u16 guest_height = rsx::method_registers.surface_clip_height() ? rsx::method_registers.surface_clip_height() : static_cast<u16>(m_framebuffer_layout.height);
	const auto [clip_width, clip_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, guest_width, guest_height);
	const bool full_bank = m_vertex_prog->has_indexed_constants;
	m_dlss->write_draw_block(dst, full_bank ? nullptr : m_vertex_prog->constant_ids.data(), full_bank ? 0 : m_vertex_prog->constant_ids.size(),
		vr_vertex_program_hash(), clip_width, clip_height, rsx::method_registers.current_draw_clause.is_trivial_instanced_draw);
}

vk::pipeline_props VKGSRender::dlss_interpreter_pipeline() const
{
	return vk::dlss_interpreter_props(m_pipeline_properties);
}
