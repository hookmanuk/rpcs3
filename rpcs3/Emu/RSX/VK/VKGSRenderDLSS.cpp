#include "stdafx.h"
#include "VKGSRender.h"
#include "VKDLSS.h"
#include "VKFrameGen.h"
#include "VKOpenXR.h"

#include "../Capture/rsx_camera_probe.h"
#include "../Capture/rsx_vr_hooks.h"
#include "../Common/surface_store.h"
#include "../rsx_methods.h"

#include <set>

// DLSS test: VKGSRender's side of VKDLSS.h (members declared in VKGSRenderVR.inl). Upstream files only call the fork's
// existing VR hooks, which call these (plus one line each in load_program() and the render pass builder).

// ---- Lifetime (vr_init_after_device, vr_destroy_resources) ------------------------------------------------

void VKGSRender::dlss_init()
{
	// DLSS on at boot: MSAA off for the session (DLSS is the anti-aliasing; MSAA only costs time, and its motion path is
	// the least tried). Read once here, before the render targets and the backend's MSAA support are set up, as RPCS3
	// itself turns MSAA off on a device without it; the saved setting is not changed.
	if (vk::dlss_configured_mode() != vk::dlss_mode::off && g_cfg.video.antialiasing_level != msaa_level::none && !std::getenv("RPCS3_DLSS_KEEP_MSAA"))
	{
		rsx_log.notice("DLSS: MSAA off for this session (DLSS is on)");
		g_cfg.video.antialiasing_level.set(msaa_level::none);
	}
	if (vk::dlss_enabled())
	{
		m_dlss = std::make_unique<vk::dlss_context>(*m_device);
		if (vk::framegen_requested())
		{
			m_framegen = std::make_unique<vk::framegen_context>(*m_device);
		}
	}
	rsx::vr::set_dlss_session(m_dlss != nullptr);
}

void VKGSRender::dlss_destroy()
{
	rsx::vr::set_dlss_session(false);
	m_framegen.reset();
	m_dlss.reset();
}

// ---- Framebuffers (prepare_rtts through vr_update_multiview_mode, vr_renderpass_key, vr_framebuffer_images) -----

u64 VKGSRender::vr_renderpass_key(const std::vector<u8>& input_attachments)
{
	m_current_renderpass_key = vk::get_renderpass_key(m_fbo_images, input_attachments, vr_draw_view_mask()); // multiview
	dlss_prepare_framebuffer(); // the motion attachment of the scene target
	return m_current_renderpass_key;
}

std::vector<vk::image*>& VKGSRender::vr_framebuffer_images()
{
	return m_dlss_motion_bound ? m_dlss_fbo_images : m_fbo_images;
}

void VKGSRender::dlss_check_framebuffer()
{
	if (!m_draw_fbo || !dlss_framebuffer_stale())
	{
		return;
	}
	// The bound framebuffer gains or loses the motion attachment: prepare_rtts' fast paths must not keep it.
	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}
	m_draw_fbo->release();
	m_draw_fbo = nullptr;
}


void VKGSRender::dlss_prepare_framebuffer()
{
	const bool was_bound = m_dlss_motion_bound;
	m_dlss_motion_bound = false;

	if (m_dlss && !m_draw_buffers.empty() && m_fbo_images.size() >= m_draw_buffers.size())
	{
		const usz colors = m_draw_buffers.size();
		const u32 address = m_surface_info[m_draw_buffers[0]].address;
		const u32 mrt = rsx::utility::get_mrt_buffers_count(m_framebuffer_layout.target);
		const bool vr_unsupported = dlss_vr_unsupported();
		const auto [fbo_width, fbo_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, m_framebuffer_layout.width, m_framebuffer_layout.height);

		if (vk::image* motion = m_dlss->bind_framebuffer(*m_current_command_buffer, address, m_fbo_images[0], fbo_width, fbo_height,
				static_cast<u32>(colors), mrt, vr_unsupported))
		{
			// Colour attachments, the motion attachment, then depth (framebuffer::depth_format() reads the last one).
			m_dlss_fbo_images.assign(m_fbo_images.begin(), m_fbo_images.begin() + colors);
			m_dlss_fbo_images.push_back(motion);
			m_dlss_fbo_images.insert(m_dlss_fbo_images.end(), m_fbo_images.begin() + colors, m_fbo_images.end());
			m_current_renderpass_key |= vk::dlss_renderpass_motion_bit;
			m_dlss_motion_bound = true;
			m_dlss->note_scene_depth(m_framebuffer_layout.zeta_address);
		}
		else if (m_dlss->wants_motion(address, vr_unsupported))
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
	return m_dlss->wants_motion(address, dlss_vr_unsupported()) != m_dlss_motion_bound;
}

bool VKGSRender::dlss_vr_unsupported() const
{
	// VR stereo without multiview (each eye its own draw, the right one into a second surface cache): no motion.
	return rsx::vr::camera_probe::get().render_enabled() && !m_vr_multiview;
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

bool VKGSRender::dlss_in_draw() const
{
	// The draw's own upload (dlss_draw_setup), or a sub-draw's (patch_transform_constants while the draw is emitted).
	return m_dlss_uploading || rsx::method_registers.current_draw_clause.is_rendering;
}

// ---- Draws (vr_before_draw_setup: end(), after load_program_env()) ---------------------------------------------

void VKGSRender::dlss_draw_setup()
{
	if (!m_dlss || !m_program || m_shader_interpreter.is_interpreter(m_program) || !m_vertex_prog)
	{
		return;
	}

	dlss_note_draw();

	// VR multiview: each eye's constants are uploaded again in vr_setup_draw with that eye's motion block, so the shaders
	// never read a view-0 block. Only identify the draw here (its key and camera matrix, for the eyes' histories).
	if ((current_vertex_program.ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW) && dlss_draw_block_size() &&
		!rsx::method_registers.current_draw_clause.is_trivial_instanced_draw)
	{
		alignas(16) u8 unused[vk::dlss_draw_block_size];
		m_dlss->set_identify_only(true);
		m_dlss_uploading = true;
		dlss_write_draw_block(unused);
		m_dlss_uploading = false;
		m_dlss->set_identify_only(false);
		if (m_dlss_motion_bound)
		{
			m_dlss->before_draw(*m_current_command_buffer);
		}
		return;
	}

	// Each draw's motion block differs (its own matrix history): the draw's constants again, with its block, in place of
	// load_program_env()'s upload (which has a neutral block, or is an earlier draw's when the constants did not change).
	if (dlss_draw_block_size() && m_vs_binding_table->cbuf_location != umax &&
		!rsx::method_registers.current_draw_clause.is_trivial_instanced_draw)
	{
		usz mem_offset = 0;
		auto alloc_storage = [&](usz size) -> std::pair<void*, usz>
		{
			mem_offset = m_transform_constants_allocator->alloc_bytes(size);
			return std::make_pair(m_transform_constants_ring_info.map(mem_offset, size), size);
		};
		auto io_buf = rsx::io_buffer(alloc_storage);
		m_dlss_uploading = true;
		upload_transform_constants(io_buf);
		m_dlss_uploading = false;

		if (!io_buf.empty())
		{
			m_transform_constants_ring_info.unmap();
			m_xform_constants_dynamic_offset = mem_offset;
			m_vertex_constants_buffer_info = m_transform_constants_ring_info.window<16>(m_xform_constants_dynamic_offset, io_buf.size(),
				m_device->gpu().get_limits().maxUniformBufferRange);
			m_xform_constants_dynamic_offset -= m_vertex_constants_buffer_info.offset;
			m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->cbuf_location);
		}
	}

	if (m_dlss_motion_bound)
	{
		m_dlss->before_draw(*m_current_command_buffer);
	}

	if (static const bool s_debug = std::getenv("RPCS3_DLSS_DEBUG") != nullptr; s_debug)
	{
		// The motion attachment follows the colour attachments.
		const auto& motion_blend = m_pipeline_properties.state.att_state[std::min<usz>(m_draw_buffers.size(), std::size(m_pipeline_properties.state.att_state) - 1)];
		rsx_log.notice("DLSS debug draw: bound %d, vp ctrl 0x%x, fp ctrl 0x%x, pass key 0x%llx, attachments %u, motion write mask 0x%x, blend %u",
			m_dlss_motion_bound, current_vertex_program.ctrl, current_fragment_program.ctrl, m_current_renderpass_key,
			m_pipeline_properties.state.cs.attachmentCount, motion_blend.colorWriteMask, motion_blend.blendEnable);
	}
}

// ---- Constants (upload_transform_constants through vr_constants_tail_size / vr_write_constants_tail) -----------

usz VKGSRender::vr_constants_tail_size() const
{
	return vr_depth_remap_size() + dlss_draw_block_size();
}

void VKGSRender::vr_write_constants_tail(u8* dst)
{
	vr_write_depth_remap(dst, vr_depth_remap_size(), false); // the depth remap's identity
	if (m_dlss && m_vertex_prog && !m_dlss_uploading && rsx::method_registers.current_draw_clause.is_rendering)
	{
		dlss_note_draw(); // a sub-draw (merged draws): a draw of its own for the scene target detection
	}
	dlss_write_draw_block(dst + vr_depth_remap_size());
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

	if (!dlss_in_draw())
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
	alignas(16) f32 scale_offset[24];
	m_draw_processor.fill_scale_offset_data(scale_offset, false);
	m_dlss->note_viewport_xform(scale_offset);
}

void VKGSRender::dlss_write_eye_blocks(u8* left, u8* right, usz block_offset)
{
	if (static const bool s_debug = std::getenv("RPCS3_DLSS_DEBUG") != nullptr; s_debug && m_dlss_motion_bound)
	{
		// Dev: every program drawn into the scene target, once (a pass that gets no block gets no jitter either).
		static std::set<u64> s_seen;
		if (s_seen.insert(vr_vertex_program_hash()).second)
		{
			const auto& r = rsx::method_registers;
			rsx_log.notice("DLSS scene program %016llx: block %u, depth test %d write %d func %u, blend %d, instanced %d, elements %u",
				vr_vertex_program_hash(), static_cast<u32>(dlss_draw_block_size()), r.depth_test_enabled(), r.depth_write_enabled(), static_cast<u32>(r.depth_func()),
				r.blend_enabled_mask() != 0, r.current_draw_clause.is_trivial_instanced_draw, r.current_draw_clause.vr_total_elements());
		}
	}
	if (!dlss_draw_block_size() || !dlss_in_draw())
	{
		return;
	}

	const u16 guest_width = rsx::method_registers.surface_clip_width() ? rsx::method_registers.surface_clip_width() : static_cast<u16>(m_framebuffer_layout.width);
	const u16 guest_height = rsx::method_registers.surface_clip_height() ? rsx::method_registers.surface_clip_height() : static_cast<u16>(m_framebuffer_layout.height);
	const auto [clip_width, clip_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, guest_width, guest_height);
	const bool full_bank = m_vertex_prog->has_indexed_constants;
	const u16* ids = full_bank ? nullptr : m_vertex_prog->constant_ids.data();
	const usz count = full_bank ? 0 : m_vertex_prog->constant_ids.size();
	const u64 hash = vr_vertex_program_hash();
	const bool instanced = rsx::method_registers.current_draw_clause.is_trivial_instanced_draw;
	m_dlss->write_draw_block(left + block_offset, ids, count, hash, clip_width, clip_height, instanced, left, 1);
	m_dlss->write_draw_block(right + block_offset, ids, count, hash, clip_width, clip_height, instanced, right, 2);
}

void VKGSRender::dlss_hud_snapshot(u32 color_index)
{
	static const bool s_off = std::getenv("RPCS3_DLSS_HUD_IN") != nullptr; // dev: the HUD through DLSS as before
	if (!m_dlss || s_off || color_index >= m_rtts.m_bound_render_targets.size())
	{
		return;
	}
	auto* target = m_rtts.m_bound_render_targets[color_index].second;
	if (!target)
	{
		return;
	}
	// A copy needs the render pass closed; it is opened again for this draw.
	const bool open = vk::is_renderpass_open(*m_current_command_buffer);
	if (open)
	{
		close_render_pass();
	}
	m_dlss->hud_snapshot(*m_current_command_buffer, target);
	if (open)
	{
		begin_render_pass();
	}
}

// Depth-only passes (no colour target) never get the motion attachment, so their programs are not the motion variants
// and the shader adds no jitter. A game that lays the scene's depth down first and draws colour over it with a less-or-
// equal test then lost whole surfaces: MotorStorm's terrain, jittered against its unjittered depth, failed the test on a
// slope, went black, and the game's exposure swung the frame white and black. Their viewport is moved by the jitter
// instead, the same shift the shader gives the colour passes.
bool VKGSRender::dlss_jitter_viewport(VkViewport& viewport) const
{
	if (!m_dlss || m_dlss_motion_bound || !m_draw_buffers.empty() || !m_framebuffer_layout.zeta_address ||
		m_framebuffer_layout.zeta_address != m_dlss->scene_depth() || !rsx::method_registers.depth_test_enabled())
	{
		return false;
	}
	const f32* jitter = m_dlss->jitter_px();
	if (jitter[0] == 0.f && jitter[1] == 0.f)
	{
		return false;
	}
	viewport.x += jitter[0];
	viewport.y += jitter[1];
	return true;
}

// ---- Flip (vr_publish_frame) ---------------------------------------------------------------------------

void VKGSRender::framegen_on_flip(vk::viewable_image* left, vk::viewable_image* right)
{
	m_framegen_eyes[0] = m_framegen_eyes[1] = nullptr;
	if (!left || !right || left == right)
	{
		m_framegen->reset();
		return;
	}
	f32 tan_x = 1.f, tan_y = 1.f;
	rsx::vr::camera_probe::get().get_vr_fov(tan_x, tan_y);
	vk::viewable_image* outs[2]{};
	vk::viewable_image* const eyes[2] = {left, right};
	for (u32 eye = 0; eye < 2; ++eye)
	{
		vk::framegen_eye_inputs in;
		in.backbuffer = eyes[eye];
		in.tan_half_x = tan_x;
		in.tan_half_y = tan_y;
		in.depth_inverted = std::getenv("RPCS3_DLSS_DEPTH_INVERTED") != nullptr;
		if (!m_dlss->framegen_inputs(eye, in.depth, in.motion, in.clip_to_prev_clip, in.have_motion_matrix))
		{
			return;
		}
		outs[eye] = m_framegen->evaluate(*m_current_command_buffer, eye, in, m_dlss->history_reset());
	}
	if (outs[0] && outs[1])
	{
		m_framegen_eyes[0] = outs[0];
		m_framegen_eyes[1] = outs[1];
		if (static const bool s_only = std::getenv("RPCS3_DLSSG_ONLY") != nullptr; s_only)
		{
			// Dev: show only generated frames (each in place of its real frame), to look at them.
			for (u32 eye = 0; eye < 2; ++eye)
			{
				const areai r{0, 0, static_cast<s32>(outs[eye]->width()), static_cast<s32>(outs[eye]->height())};
				vk::copy_image(*m_current_command_buffer, outs[eye], eyes[eye], r, r);
			}
			m_framegen_eyes[0] = m_framegen_eyes[1] = nullptr;
		}
	}
	m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
}

void VKGSRender::dlss_on_flip(const rsx::display_flip_info_t& info, const rsx::avconf& avconfig, vk::viewable_image*& image_to_flip,
	vk::viewable_image*& image_to_flip2, bool generated_stereo, u32& buffer_width, u32& buffer_height)
{
	// The displayed image through DLSS (or a debug view); both eyes in VR multiview.
	if (!m_dlss || avconfig.stereo_enabled || (generated_stereo && !m_vr_multiview))
	{
		return;
	}

	const auto target = avconfig.aspect_convert_region({ buffer_width, buffer_height }, m_swapchain_dims);
	u32 out_w = 0, out_h = 0;
	m_dlss_refused_address = 0;
	if (static const bool s_debug = std::getenv("RPCS3_DLSS_DEBUG") != nullptr; s_debug)
	{
		rsx_log.notice("DLSS debug: flip buffer %u, %ux%u, emu_flip %d, draw calls %u, image %p", info.buffer, buffer_width, buffer_height, info.emu_flip, info.stats.draw_calls, static_cast<void*>(image_to_flip));
	}
	// VR multiview with a headset view: right_eye is the right eye's image, or null for a frame on the fixed screen.
	const bool vr_view = m_vr_multiview && (vk::xr::is_running() || vk::xr::fake_hmd());
	// The HUD outside DLSS: the image without it (taken before the frame's first HUD-box draw) in, the HUD added after.
	vk::viewable_image* const final_left = image_to_flip;
	vk::viewable_image* const final_right = image_to_flip2;
	vk::viewable_image* pre_left = m_dlss->hud_pre(image_to_flip, 0);
	vk::viewable_image* pre_right = pre_left && vr_view && image_to_flip2 ? m_dlss->hud_pre(image_to_flip, 1) : nullptr;
	if (vr_view && image_to_flip2 && !pre_right)
	{
		pre_left = nullptr;
	}
	m_dlss->hud_flip_done();
	vk::viewable_image* right = pre_right ? pre_right : image_to_flip2;
	if (vk::viewable_image* dlss_image = m_dlss->on_flip(*m_current_command_buffer, pre_left ? pre_left : image_to_flip, buffer_width, buffer_height,
			static_cast<u32>(target.width()), static_cast<u32>(target.height()), info.emu_flip, dlss_vr_unsupported(), out_w, out_h,
			vr_view ? &right : nullptr))
	{
		if (pre_left && m_dlss->output_fresh())
		{
			m_dlss->hud_composite(*m_current_command_buffer, dlss_image, final_left, 0, pre_left, buffer_width, buffer_height);
			if (pre_right && right != pre_right)
			{
				m_dlss->hud_composite(*m_current_command_buffer, right, final_right, 0, pre_right, buffer_width, buffer_height);
			}
		}
		image_to_flip = dlss_image;
		if (vr_view)
		{
			image_to_flip2 = right;
		}
		if (m_framegen && vr_view && info.emu_flip && m_dlss->output_fresh())
		{
			framegen_on_flip(dlss_image, right); // framegen: the frame halfway to this one, per eye
		}
		buffer_width = out_w;
		buffer_height = out_h;
		m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
	}
}
