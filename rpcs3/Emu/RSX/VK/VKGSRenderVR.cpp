#include "stdafx.h"
#include <numeric>
#include <set>

// VR fork: the stereo rendering members of VKGSRender. Everything here used to live in
// VKDraw.cpp, VKGSRender.cpp and VKPresent.cpp; it was moved out so that upstream's files
// carry one-line hooks only (see VKGSRenderVR.inl for the declarations).

#include "VKGSRender.h"
#include "VKOpenXR.h"
#include "VKOverlaysVR.h"
#include "VKHelpers.h"
#include "VKMultiviewVR.h"
#include "VKRenderPass.h"
#include "VKResourceManager.h"
#include "VKDMA.h"
#include "VKFramebuffer.h"
#include "vkutils/buffer_object.h"
#include "vkutils/scratch.h"

#include "../Capture/rsx_stereo_inspector.h"
#include "../Capture/rsx_camera_probe.h"
#include "../Capture/rsx_vr_profile_generator.h"
#include "../Capture/rsx_vr_eye_shape.h"
#include "../Common/BufferUtils.h"
#include "../Program/GLSLCommon.h"
#include "../rsx_methods.h"
#include "../Utils/rsx_utils.h"

#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/Overlays/overlay_manager.h"
#include "Emu/Cell/Modules/cellVideoOut.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/Memory/vm.h"
#include "Emu/System.h"
#include "Emu/system_config.h"

#include <vulkan/vulkan_core.h>
#include <unordered_set>

// Guards for what the fork adds to layouts upstream owns (see plans/8-upstream-merge-audit.md, section 2).
static_assert(rsx::texture_control_bits::VR_REPROJECT_BIT < 32, "VR_REPROJECT_BIT must fit the 32-bit texture control word");
static_assert(rsx::texture_control_bits::VR_REPROJECT_BIT > rsx::texture_control_bits::FF_16BIT_CHANNELS_BIT, "VR_REPROJECT_BIT is appended after upstream's bits");

namespace vk
{
	VkImageViewType get_view_type(rsx::texture_dimension_extended type); // VKDraw.cpp
}

extern thread_local std::chrono::steady_clock::time_point g_vr_readback_start; // VKGSRenderVRDev.cpp, dev timer
extern atomic_t<u64> g_vr_readback_pre_ns;

// Profile late_readback_sections (see rsx_camera_probe.h): readbacks whose GPU copy is still pending return at once,
// leaving the guest memory as it is, and the copy is written there at the next flips once its fence is set. The
// fence belongs to the section; a section re-created (the game's next copy into it) disposes it through the resource
// manager, so a pending entry is dropped after three flips (GT5 reuses each of its three slots every third frame).
namespace vk
{
	namespace
	{
		struct vr_late_readback_t
		{
			const event* fence;
			u32 start;
			u32 length;
			u32 flips;
		};
		std::mutex g_vr_late_mutex;
		std::vector<vr_late_readback_t> g_vr_late;
	}

	bool vr_late_readback(const event* fence, u32 start, u32 length)
	{
		if (g_vr_readback_start != std::chrono::steady_clock::time_point{})
		{
			g_vr_readback_pre_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - g_vr_readback_start).count();
			g_vr_readback_start = {};
		}
		const auto* profile = g_cfg.video.vr.enabled ? rsx::vr::camera_probe::get().profile() : nullptr;
		if (!fence || !profile || (std::find(profile->late_readback_sections.begin(), profile->late_readback_sections.end(), start) == profile->late_readback_sections.end() &&
			std::find(profile->late_readback_lengths.begin(), profile->late_readback_lengths.end(), length) == profile->late_readback_lengths.end()))
		{
			return false;
		}
		if (fence->status() == VK_EVENT_SET)
		{
			return false; // landed: the ordinary readback, which does not wait
		}
		static atomic_t<bool> s_logged{false};
		if (!s_logged.exchange(true))
		{
			rsx_log.notice("VR: readback of 0x%x+0x%x answered late (late_readback_sections/lengths).", start, length);
		}
		std::lock_guard lock(g_vr_late_mutex);
		std::erase_if(g_vr_late, [&](const vr_late_readback_t& e) { return e.start == start; });
		g_vr_late.push_back({fence, start, length, 0});
		return true;
	}

	u64 vr_cached_image_pool_limit(u64 upstream_limit)
	{
		// Upstream trims the pool by half whenever it holds more than 256 MB. A temporary copy of a 1280x720
		// target is 15 MB at 200% and 118 MB at 400% in stereo (two layers), so in VR the pool was trimmed every
		// frame and the images allocated again (GT5: ~3.5% of the RSX thread in vkFreeMemory/vkDestroyImage).
		// While stereo renders the limit grows with the scale's pixel count and the eye count, at most an eighth
		// of the GPU's own memory. Dev: RPCS3_VR_IMAGE_POOL_MB sets it.
		static const u64 s_env = []() -> u64
		{
			const char* v = std::getenv("RPCS3_VR_IMAGE_POOL_MB");
			return v ? std::strtoull(v, nullptr, 10) * 0x100000 : 0;
		}();
		if (s_env)
		{
			return s_env;
		}
		if (!rsx::vr::camera_probe::get().render_enabled())
		{
			return upstream_limit;
		}
		const u64 scale = std::max<u64>(100, g_cfg.video.resolution_scale_percent);
		const u64 wanted = upstream_limit * scale * scale / 10000 * 2;
		const u64 vram = g_render_device ? g_render_device->get_memory_mapping().device_local_total_bytes : 0;
		const u64 limit = std::max(upstream_limit, vram ? std::min(wanted, vram / 8) : upstream_limit);
		if (static u64 s_logged = 0; s_logged != limit)
		{
			s_logged = limit;
			rsx_log.notice("VR: temporary image pool limit %u MB (GPU memory %u MB).", limit >> 20, vram >> 20);
		}
		return limit;
	}

	// Framebuffer cache (VKFramebuffer.cpp).
	extern std::unordered_map<u64, std::vector<std::unique_ptr<vk::framebuffer_holder>>> g_framebuffers_cache;
	extern shared_mutex g_framebuffers_mutex;

	// remove_unused_framebuffers() (each frame end): a framebuffer idle for two frame ends goes to the resource manager,
	// which frees it once everything submitted so far has finished, instead of being destroyed at once. Upstream counts on
	// the GPU being at most two frames behind; in VR the GPU is the bottleneck and up to a swapchain's worth of frames
	// queue, so framebuffers (and the image views they own) were destroyed while queued render passes still used them
	// (validation: "vkDestroyFramebuffer(): ... currently in use by VkCommandBuffer", multiview stereo).
	bool vr_retire_unused_framebuffers()
	{
		// Dev: RPCS3_VR_FBO_RETIRE=0 destroys them at once (upstream).
		static const bool s_off = []() { const char* v = std::getenv("RPCS3_VR_FBO_RETIRE"); return v && v[0] == '0'; }();
		if (s_off)
		{
			return false;
		}
		std::lock_guard lock(g_framebuffers_mutex);
		for (auto it = g_framebuffers_cache.begin(); it != g_framebuffers_cache.end();)
		{
			auto& list = it->second;
			for (auto& fbo : list)
			{
				if (fbo->unused_check_count() >= 2)
				{
					vk::get_resource_manager()->dispose(fbo);
				}
			}
			std::erase_if(list, [](const auto& fbo) { return !fbo; });
			it = list.empty() ? g_framebuffers_cache.erase(it) : std::next(it);
		}
		return true;
	}

	void vr_complete_late_readbacks()
	{
		std::lock_guard lock(g_vr_late_mutex);
		std::erase_if(g_vr_late, [](vr_late_readback_t& e)
		{
			if (e.fence->status() == VK_EVENT_SET)
			{
				vk::flush_dma(e.start, e.length);
				return true;
			}
			return ++e.flips >= 3;
		});
	}
}

// A display buffer's memory drawn at the display buffer's size: Gran Turismo 5 also renders its
// shadow cascades (1024 wide) into the memory of the buffer it is not showing.
static bool vr_display_buffer(const rsx::thread& rsx, u32 address, u32 width, u32 height)
{
	// Display buffers are in local memory: the plain address (rsx::get_address's lookup and checks cost ~0.5% of the
	// RSX thread over the per-draw calls).
	for (const auto& buffer : rsx.display_buffers)
	{
		if (buffer.width && rsx::constants::local_mem_base + buffer.offset == address &&
			buffer.width == width && buffer.height == height)
		{
			return true;
		}
	}
	return false;
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
				reads += rtt->vr_pose ? fmt::format("%s%x:%d", reads.empty() ? "" : ",", rtt->base_addr, static_cast<s32>(m_vr_applied_pose - rtt->vr_pose)) : fmt::format("%s%x:-", reads.empty() ? "" : ",", rtt->base_addr);
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

void VKGSRender::vr_mark_3d_targets(bool camera)
{
	// 3D content passes on: a camera draw's targets hold it, and so do targets drawn from a texture that does. The
	// displayed buffer's flag tells a 2D screen (splash, menu, video: shown on the fixed screen) from a paused game
	// re-showing its last 3D frame (Pure, WipEout: stays in the headset view).
	const auto is_3d = [](vk::image* image)
	{
		const auto* rtt = dynamic_cast<vk::render_target*>(image);
		return rtt && rtt->vr_has_3d;
	};
	bool has_3d = camera;
	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; !has_3d && textures_ref; textures_ref >>= 1, ++i)
	{
		auto sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		if (!(textures_ref & 1) || !sampler_state || sampler_state->upload_context != rsx::texture_upload_context::framebuffer_storage)
		{
			continue;
		}
		if (sampler_state->image_handle)
		{
			has_3d = is_3d(sampler_state->image_handle->image());
		}
		else
		{
			const auto& desc = sampler_state->external_subresource_desc;
			has_3d = (desc.external_handle && is_3d(desc.external_handle)) ||
			         std::any_of(desc.sections_to_copy.begin(), desc.sections_to_copy.end(), [&](const auto& section)
						 {
							 return is_3d(section.src);
						 });
		}
	}
	if (!has_3d)
	{
		return;
	}
	for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
	{
		if (auto* surface = std::get<1>(m_rtts.m_bound_render_targets[index]))
		{
			surface->vr_has_3d = true;
		}
	}
	// The depth buffer too: a pass that reads it (SEGA Rally's half-width depth for its soft dust particles) makes a
	// per-eye image, which vr_eye_invariant_target must not hand the left eye's copy of.
	if (auto* depth = std::get<1>(m_rtts.m_bound_depth_stencil))
	{
		depth->vr_has_3d = true;
	}
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

		vk::image* image = sampler_state->image_handle                                        ? sampler_state->image_handle->image() :
		                   sampler_state->external_subresource_desc.external_handle           ? sampler_state->external_subresource_desc.external_handle :
		                   !sampler_state->external_subresource_desc.sections_to_copy.empty() ? sampler_state->external_subresource_desc.sections_to_copy.front().src :
		                                                                                        nullptr;
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
						slot.scale[0] = h[0];
						slot.scale[1] = h[1];
						slot.scale[2] = h[2];
						slot.bias[0] = h[3];
						slot.bias[1] = h[4];
						slot.bias[2] = h[5];
						slot.clamp_min[0] = h[6];
						slot.clamp_min[1] = h[7];
						slot.clamp_max[0] = h[8];
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

		// The rotation since the surface's pose, as a homography: exact across the view.
		// A pixel shift (the fallback) is exact only at the centre; towards the edges of
		// a ~90 degree view a rotation moves the image up to twice as far, so shifted
		// content swam against head turns and settled when the head stopped.
		f32 homography[9];
		const bool exact = vk::xr::render_pose_homography(surface->vr_pose, m_vr_applied_pose, homography);

		// new(x) = old(x + dx): the content moves against the head rotation. layer: the eye of a
		// stereo surface (multiview), else 0.
		const auto shift = [&](vk::render_target* image, u32 layer)
		{
			const int w = static_cast<int>(image->width());
			const int h = static_cast<int>(image->height());
			const int dx = static_cast<int>(std::lround(du * w));
			const int dy = static_cast<int>(std::lround(dv * h));
			if ((!dx && !dy) || std::abs(dx) >= w || std::abs(dy) >= h)
			{
				return std::pair<int, int>{dx, dy};
			}
			if (image->current_layout == VK_IMAGE_LAYOUT_UNDEFINED)
			{
				// Never written (a right-eye surface): nothing to move.
				return std::pair<int, int>{dx, dy};
			}
			if (exact)
			{
				auto& target = m_vr_warp_scratch[(static_cast<u64>(image->format()) << 40) | (static_cast<u64>(w) << 20) | static_cast<u64>(h)];
				if (!target)
				{
					target = std::make_unique<vk::viewable_image>(*m_device, m_device->get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						VK_IMAGE_TYPE_2D, image->format(), w, h, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						0, VMM_ALLOCATION_POOL_SYSTEM);
				}
				if (vk::is_renderpass_open(*m_current_command_buffer))
				{
					vk::end_renderpass(*m_current_command_buffer);
				}
				vk::get_overlay_pass<vk::vr_homography_warp_pass>()->run(*m_current_command_buffer, image, target.get(), homography, layer);
				vk::copy_image(*m_current_command_buffer, target.get(), image, areai{0, 0, w, h}, areai{0, 0, w, h}, {.dst_layer = static_cast<u8>(layer)});
				return std::pair<int, int>{dx, dy};
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
			vk::copy_image(*m_current_command_buffer, image, scratch, areai{0, 0, w, h}, areai{0, 0, w, h}, {.src_layer = static_cast<u8>(layer)});
			const areai src{std::max(dx, 0), std::max(dy, 0), w + std::min(dx, 0), h + std::min(dy, 0)};
			const areai dst{std::max(-dx, 0), std::max(-dy, 0), w - std::max(dx, 0), h - std::max(dy, 0)};
			vk::copy_image(*m_current_command_buffer, scratch, image, src, dst, {.dst_layer = static_cast<u8>(layer)});
			return std::pair<int, int>{dx, dy};
		};

		const auto [dx, dy] = shift(surface, 0);
		if (m_vr_multiview && surface->stereo_layers && surface->layers() > 1)
		{
			shift(surface, 1); // the right eye, layer 1
		}
		else if (auto* right = m_vr_right_rtts.get_surface_at(surface->base_addr);
			!m_vr_multiview && right && right->width() == surface->width() && right->height() == surface->height() && right->samples() == 1)
		{
			vr_batch_flush();
			shift(right, 0);
		}
		surface->vr_pose = m_vr_applied_pose;
		invalidate_render_pass();

		if (vr_tracing())
		{
			vr_trace_flush_cam();
			m_vr_trace += fmt::format(" A{%x:%dpx,%dpx%s}", surface->base_addr, dx, dy, exact ? " h" : "");
		}
	}
}

bool VKGSRender::vr_is_passthrough_hud()
{
	// Into a buffer no camera draw wrote (the finished frame or a display buffer), with
	// at least one ordinary texture and no colour render target (post-processing reads those).
	// A program the profile lists in hud_programs is HUD wherever it draws in full-frame buffers, also untextured
	// (fills) and without a colour target (Dragon Age: Origins' Scaleform HUD masks its minimap and portrait in the
	// depth/stencil buffer with colour writes off; left out, the masks stayed on the face while the rest moved).
	const auto* profile = rsx::vr::camera_probe::get().profile();
	const u64 hash = profile && !profile->screen_space_hud_programs.empty() ? vr_vertex_program_hash() : 0;
	const bool listed = hash && std::find(profile->screen_space_hud_programs.begin(), profile->screen_space_hud_programs.end(), hash) != profile->screen_space_hud_programs.end();
	const u32 target = m_framebuffer_layout.color_addresses[0];
	if (!target && !listed)
	{
		return false;
	}
	if (target && !listed && std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), target) != m_vr_camera_targets.end())
	{
		// Unless the profile lists this program as HUD (drawn into the scene's final image).
		return false;
	}
	// Profile hud_display_buffers_only: the HUD is only what goes into a display buffer. The frame-width test below
	// compares with the latest camera target, which can be a small view (SEGA Rally Revo's 512x512 rear-view mirror,
	// its 384- and 768-wide trackside screens): its 912x912 shadow map passes then went into the head-fixed box, and the
	// shadows slid across the ground as the head turned.
	if (target && !listed && profile && profile->screen_space_hud_display_buffers_only &&
		!vr_display_buffer(*this, target, m_framebuffer_layout.width, m_framebuffer_layout.height))
	{
		return false;
	}
	// Full frame or larger: smaller buffers are intermediate passes (ICO's shadow mask).
	// The frame is the scene or the output, whichever is narrower: Ridge Racer 7 renders
	// its scene 1408 wide and draws the HUD at the 1280 output size.
	const vk::render_target* scene = m_vr_camera_targets.empty() ? nullptr : m_rtts.get_surface_at(m_vr_camera_targets.back());
	const u32 output_width = g_fxo->get<rsx::avconf>().video_frame_size().width;
	const u32 frame_width = scene ? std::min<u32>(scene->get_surface_width<rsx::surface_metrics::pixels>(), output_width ? output_width : umax) : 0;
	if (!scene || m_framebuffer_layout.width * 20 < frame_width * 19)
	{
		return false;
	}

	// A small render target is HUD art too (Dragon's Dogma's minimap, drawn into a 256x256 target first).
	const u32 kinds = vr_sampled_textures();
	if (listed)
	{
		return !(kinds & vr_texture_view_target);
	}
	return (kinds & (vr_texture_ordinary | vr_texture_colour_target)) && !(kinds & vr_texture_view_target);
}

// Profile screen_space.unboxed_draws (vertex program ucode hash + texture 0 size), or the dev probe key
// unboxfp=<fragment program session id>, which also logs the draw's vertex program hash and texture size.
// Profile screen_space.scaled_draws: the size factor for this draw (1 = none).
f32 VKGSRender::vr_hud_draw_scale()
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (!profile || profile->screen_space_scaled_draws.empty() || !m_vertex_prog)
	{
		return 1.f;
	}
	const auto& tex = rsx::method_registers.fragment_textures[0];
	const u16 width = tex.enabled() ? tex.width() : 0;
	const u16 height = tex.enabled() ? tex.height() : 0;
	const auto& list = profile->screen_space_scaled_draws;
	if (std::none_of(list.begin(), list.end(), [&](const auto& d)
			{
				return d.width == width && d.height == height;
			}))
	{
		return 1.f;
	}
	static std::unordered_map<u32, u64> s_hashes; // vertex program session id -> ucode hash
	auto [it, added] = s_hashes.try_emplace(m_vertex_prog->id, 0);
	if (added)
	{
		it->second = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
	}
	for (const auto& d : list)
	{
		if (d.program == it->second && d.width == width && d.height == height)
		{
			return d.scale;
		}
	}
	return 1.f;
}

bool VKGSRender::vr_unboxed_draw()
{
	auto& probe = rsx::vr::camera_probe::get();
	const auto* profile = probe.profile();
	const bool dev = m_fragment_prog && probe.unboxed_fragment_program(m_fragment_prog->id);
	if (!dev && (!profile || profile->screen_space_unboxed_draws.empty()))
	{
		return false;
	}
	const auto& tex = rsx::method_registers.fragment_textures[0];
	const u16 width = tex.enabled() ? tex.width() : 0;
	const u16 height = tex.enabled() ? tex.height() : 0;
	static std::unordered_map<u32, u64> s_hashes; // vertex program session id -> ucode hash
	u64 hash = 0;
	if (m_vertex_prog)
	{
		auto [it, added] = s_hashes.try_emplace(m_vertex_prog->id, 0);
		if (added)
		{
			it->second = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
		}
		hash = it->second;
	}
	if (dev)
	{
		static std::set<u64> s_logged;
		if (s_logged.insert((hash << 16) ^ (u64{width} << 32) ^ height ^ m_fragment_prog->id).second)
		{
			rsx_log.notice("VR: left out of the HUD box (probe unboxfp=%u): vertex program %016llx, texture 0 %ux%u.", m_fragment_prog->id, hash, width, height);
		}
		return true;
	}
	for (const auto& draw : profile->screen_space_unboxed_draws)
	{
		if (draw.program == hash && draw.width == width && draw.height == height)
		{
			return true;
		}
	}
	return false;
}

u32 VKGSRender::vr_sampled_textures()
{
	u32 kinds = 0;
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
			// Depth render targets are fine (soft particles); colour ones mean post-processing. A depth one read as
			// colour (its 24 bits as RGB) is noted for the profile generator (depth_remap_programs candidates).
			if (image && !(image->aspect() & VK_IMAGE_ASPECT_COLOR_BIT) && (current_fragment_program.texture_state.redirected_textures & (1u << i)))
			{
				kinds |= vr_texture_depth_as_colour;
			}
			if (!image || (image->aspect() & VK_IMAGE_ASPECT_COLOR_BIT))
			{
				// With hud_display_buffers_only, a pass reads the scene, a displayed frame or a
				// screen-sized texture; smaller reads are the HUD's own art (Gran Turismo 5's name strips,
				// menu cards, and icon backgrounds cut from a 2048-wide blur buffer).
				const auto* profile = rsx::vr::camera_probe::get().profile();
				const auto* rtt = dynamic_cast<vk::render_target*>(image);
				if (profile && profile->screen_space_hud_display_buffers_only && rtt &&
					rsx::method_registers.fragment_textures[i].width() < g_fxo->get<rsx::avconf>().video_frame_size().width &&
					std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), rtt->base_addr) == m_vr_camera_targets.end() &&
					!vr_display_buffer(*this, rtt->base_addr, rtt->get_surface_width<rsx::surface_metrics::pixels>(), rtt->get_surface_height<rsx::surface_metrics::pixels>()))
				{
					kinds |= vr_texture_ordinary;
					continue;
				}
				kinds |= vr_texture_colour_target;
				// View-shaped: a camera target, a display buffer, or the output's or a camera view's aspect.
				const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
				const u32 w = rtt ? rtt->get_surface_width<rsx::surface_metrics::pixels>() : 0;
				const u32 h = rtt ? rtt->get_surface_height<rsx::surface_metrics::pixels>() : 0;
				const f32 out_aspect = out.height ? static_cast<f32>(out.width) / out.height : 0.f;
				if (!rtt || !h || !out_aspect ||
					std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), rtt->base_addr) != m_vr_camera_targets.end() ||
					vr_display_buffer(*this, rtt->base_addr, w, h) ||
					std::fabs((static_cast<f32>(w) / h) / out_aspect - 1.f) <= 0.05f ||
					(profile && profile->is_view_target(w, h, out_aspect)))
				{
					kinds |= vr_texture_view_target;
				}
			}
			continue;
		}
		// A read of a display buffer is the shown frame, whichever way it reached the texture cache: Killzone
		// HD blends the previous frame over the current one, and with Write/Read Color Buffers it arrives as
		// an ordinary upload from memory. Boxed as HUD, it showed as a grey panel behind the HUD.
		const auto& tex = rsx::method_registers.fragment_textures[i];
		const u32 tex_address = rsx::get_address(tex.offset(), tex.location());
		bool display_read = false;
		for (const auto& buffer : display_buffers)
		{
			display_read |= buffer.width && rsx::constants::local_mem_base + buffer.offset == tex_address;
		}
		kinds |= display_read ? vr_texture_colour_target | vr_texture_view_target : vr_texture_ordinary;
	}
	return kinds;
}

u64 VKGSRender::vr_vertex_program_hash()
{
	if (!m_vertex_prog)
	{
		return program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
	}
	if (m_vr_hash_program != m_vertex_prog)
	{
		m_vr_hash_program = m_vertex_prog;
		m_vr_hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
	}
	return m_vr_hash;
}

u64 VKGSRender::vr_preprojected_program()
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (!profile || profile->screen_space_preprojected_programs.empty())
	{
		return 0;
	}
	const u64 hash = vr_vertex_program_hash();
	const auto& list = profile->screen_space_preprojected_programs;
	return std::find(list.begin(), list.end(), hash) != list.end() ? hash : 0;
}

bool VKGSRender::vr_hud_box_matrix(f32 eye_sign, u64 preprojected_program, f32 combined[16])
{
	f32 box[4][4];
	f32 aspect = m_framebuffer_layout.height ? static_cast<f32>(m_framebuffer_layout.width) / m_framebuffer_layout.height : 16.f / 9.f;
	if (rsx::vr::camera_probe::get().hud_env_requested())
	{
		// An orthographic HUD draw: the box has the output's shape (Gran Turismo 5 draws it through a
		// 1280x720 viewport into a 2048x1080 buffer).
		const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
		aspect = out.height ? static_cast<f32>(out.width) / out.height : aspect;
	}
	if (preprojected_program ? !rsx::vr::camera_probe::get().map_vr_preprojected(box, eye_sign, preprojected_program) :
							   !rsx::vr::camera_probe::get().map_vr_passthrough_hud(box, eye_sign, aspect))
	{
		return false;
	}

	// The guest's viewport matrix V (vec4 k = output k's coefficients over x, y, z, w),
	// applied after the box: column k of the result is sum_c V[k][c] * box[.][c].
	alignas(16) f32 base[24];
	m_draw_processor.fill_scale_offset_data(base, false);
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
	return true;
}

bool VKGSRender::vr_hud_vertex_env(f32 eye_sign, u64 preprojected_program)
{
	f32 combined[16];
	if (!vr_hud_box_matrix(eye_sign, preprojected_program, combined))
	{
		return false;
	}

	const auto& gpu_limits = m_device->gpu().get_limits();
	const auto mem = m_vertex_env_allocator->alloc();
	auto buf = m_vertex_env_ring_info.map<char>(mem, 96);
	std::memcpy(buf, combined, 64);
	// The fixed-in-front box changes w with the head pose: keep the game's depth, which
	// the HUD's own layers (and the full-screen passes under them) are depth tested with.
	// Only for profiles that ask for it (SotC); it broke ICO's HUD box.
	const auto& probe = rsx::vr::camera_probe::get();
	const auto* profile = probe.profile();
	fill_vertex_env_tail(buf, !preprojected_program && probe.vr_hud_fixed() && profile && profile->screen_space_hud_keep_depth ? 1.f : 0.f);
	m_vertex_env_ring_info.unmap();

	m_vertex_env_buffer_info = m_vertex_env_ring_info.window<256>(mem, 96, gpu_limits.maxUniformBufferRange);
	m_vertex_env_dynamic_offset = mem - m_vertex_env_buffer_info.offset;
	m_program->bind_uniform(m_vertex_env_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->context_buffer_location);

	static bool s_reported = false, s_reported_pre = false;
	if (preprojected_program ? !std::exchange(s_reported_pre, true) : !std::exchange(s_reported, true))
	{
		if (preprojected_program == rsx::vr::scene_draw_program)
			rsx_log.success("VR: scene draws without a camera block drawn through the camera's eye transform (clip_space_scene_draws, target 0x%x).", m_framebuffer_layout.color_addresses[0]);
		else if (preprojected_program)
			rsx_log.success("VR: pre-projected program %016llx drawn through the camera's eye transform (target 0x%x).", preprojected_program, m_framebuffer_layout.color_addresses[0]);
		else
			rsx_log.success("VR: HUD drawn without a matrix is mapped into the HUD box (target 0x%x).", m_framebuffer_layout.color_addresses[0]);
	}
	return true;
}

// Multiview: the two eyes' contexts are consecutive entries of one allocation, so the one bound window
// holds both and the right eye's entry index is the left's plus one (update_vertex_env divides by 96).
bool VKGSRender::vr_hud_vertex_env_pair(u64 preprojected_program)
{
	f32 combined[2][16];
	if (!vr_hud_box_matrix(-1.f, preprojected_program, combined[0]))
	{
		return false;
	}
	// Each eye's scissor from its own box map (the constants pair found no box for a draw boxed here).
	VkRect2D scissors[2];
	const bool left_box = vr_box_scissor_rect(scissors[0]);
	rsx::vr::camera_probe::get().clear_box_mapped();
	if (!vr_hud_box_matrix(1.f, preprojected_program, combined[1]))
	{
		return false;
	}
	const bool right_box = vr_box_scissor_rect(scissors[1]);
	if (left_box || right_box)
	{
		m_vr_mv_scissor[0] = left_box ? scissors[0] : m_scissor;
		m_vr_mv_scissor[1] = right_box ? scissors[1] : m_scissor;
		m_vr_mv_box_scissor = true;
	}

	const auto& probe = rsx::vr::camera_probe::get();
	const auto* profile = probe.profile();
	const f32 keep_depth = !preprojected_program && probe.vr_hud_fixed() && profile && profile->screen_space_hud_keep_depth ? 1.f : 0.f;
	const auto& gpu_limits = m_device->gpu().get_limits();
	const auto mem = m_vertex_env_allocator->alloc(2);
	auto buf = m_vertex_env_ring_info.map<char>(mem, 192);
	for (u32 eye = 0; eye < 2; ++eye)
	{
		std::memcpy(buf + eye * 96, combined[eye], 64);
		fill_vertex_env_tail(buf + eye * 96, keep_depth);
	}
	m_vertex_env_ring_info.unmap();

	m_vertex_env_buffer_info = m_vertex_env_ring_info.window<256>(mem, 192, gpu_limits.maxUniformBufferRange);
	m_vertex_env_dynamic_offset = mem - m_vertex_env_buffer_info.offset;
	m_vr_draw.mv_env_right = m_vertex_env_dynamic_offset + 96;
	m_program->bind_uniform(m_vertex_env_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->context_buffer_location);
	return true;
}

VKGSRender* VKGSRender::s_vr_batch_owner = nullptr;

void VKGSRender::vr_on_end_renderpass(const vk::command_buffer& cmd)
{
	VKGSRender* const self = s_vr_batch_owner;
	if (self && self->m_vr_multiview)
	{
		return; // no right-eye batches with multiview; the query segment ended before the pass did
	}
	if (!self || !self->m_vr_batch_open || self->m_vr_batch_executing ||
		static_cast<VkCommandBuffer>(cmd) != static_cast<VkCommandBuffer>(*self->m_vr_batch_primary))
	{
		return;
	}

	self->vr_batch_execute();
}

bool VKGSRender::vr_batch_begin(VkRenderPass pass, vk::framebuffer_holder* fbo)
{
	if (m_vr_batch_open)
	{
		if (m_vr_batch_primary == m_current_command_buffer && m_vr_batch_pass == pass && m_vr_batch_fbo == fbo)
		{
			return true;
		}

		vr_batch_flush();
	}

	// This primary's secondaries: once it has been reset (its previous submission finished),
	// all of them are free and their pool is reset in one call.
	auto& batches = m_vr_primary_batches[m_current_command_buffer];
	if (!batches)
	{
		batches = std::make_unique<vr_primary_batches>();
		batches->pool.create(*m_device, m_device->get_graphics_queue_family(), 0);
	}
	if (batches->reset_id != m_current_command_buffer->reset_id)
	{
		if (batches->used)
		{
			vkResetCommandPool(*m_device, batches->pool, 0);
		}
		batches->used = 0;
		batches->reset_id = m_current_command_buffer->reset_id;
	}
	if (batches->used == batches->cbs.size())
	{
		VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		alloc.commandPool = batches->pool;
		alloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
		alloc.commandBufferCount = 1;
		VkCommandBuffer cb = VK_NULL_HANDLE;
		if (vkAllocateCommandBuffers(*m_device, &alloc, &cb) != VK_SUCCESS)
		{
			return false;
		}
		batches->cbs.push_back(cb);
	}
	const VkCommandBuffer secondary = batches->cbs[batches->used];

	VkCommandBufferInheritanceInfo inherit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
	inherit.renderPass = pass;
	inherit.subpass = 0;
	inherit.framebuffer = fbo->value;
	VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
	begin.pInheritanceInfo = &inherit;
	if (vkBeginCommandBuffer(secondary, &begin) != VK_SUCCESS)
	{
		return false;
	}

	batches->used++;
	m_vr_batch_secondary = secondary;
	m_vr_batch_cb.attach(batches->pool, secondary);
	m_vr_batch_primary = m_current_command_buffer;
	m_vr_batch_pass = pass;
	m_vr_batch_fbo = fbo;
	m_vr_batch_fbo->add_ref();
	m_vr_batch_open = true;
	return true;
}

void VKGSRender::vr_batch_flush()
{
	if (!m_vr_batch_open)
	{
		return;
	}

	if (vk::is_renderpass_open(*m_vr_batch_primary))
	{
		vk::end_renderpass(*m_vr_batch_primary); // the hook executes the batch
	}
	else
	{
		vr_batch_execute();
	}
}

void VKGSRender::vr_batch_execute()
{
	m_vr_batch_executing = true;
	m_gpuprof_batches++;
	m_vr_batch_cb.detach();

	const VkCommandBuffer secondary = m_vr_batch_secondary;
	vkEndCommandBuffer(secondary);

	auto& primary = *m_vr_batch_primary;

	// The right eye must not contribute samples to an open guest occlusion query:
	// split it into left-eye-only segments around the right-eye pass.
	const bool suspend_query = (primary.flags & vk::command_buffer::cb_has_open_query) && m_active_query_info;
	u32 continuation = umax;
	if (suspend_query)
	{
		auto& query_data = m_occlusion_map[m_active_query_info->driver_handle];
		m_occlusion_query_manager->end_query(primary, query_data.indices.back());
		primary.flags &= ~vk::command_buffer::cb_has_open_query;
		continuation = m_occlusion_query_manager->allocate_query(primary);
	}

	VkRenderPassBeginInfo rp_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
	rp_begin.renderPass = m_vr_batch_pass;
	rp_begin.framebuffer = m_vr_batch_fbo->value;
	rp_begin.renderArea.extent = {m_vr_batch_fbo->width(), m_vr_batch_fbo->height()};
	vkCmdBeginRenderPass(primary, &rp_begin, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
	vkCmdExecuteCommands(primary, 1, &secondary);
	vkCmdEndRenderPass(primary);

	if (suspend_query && continuation != umax)
	{
		m_occlusion_query_manager->begin_query(primary, continuation);
		auto& query_data = m_occlusion_map[m_active_query_info->driver_handle];
		query_data.indices.push_back(continuation);
		query_data.set_sync_command_buffer(&primary);
		primary.flags |= (vk::command_buffer::cb_has_occlusion_task | vk::command_buffer::cb_has_open_query);
	}

	// vkCmdExecuteCommands leaves the primary's pipeline, descriptors and dynamic state undefined.
	primary.invalidate_state_cache();
	primary.flags |= vk::command_buffer::cb_reload_dynamic_state;

	m_vr_batch_fbo->release();
	m_vr_batch_fbo = nullptr;
	m_vr_batch_open = false;
	m_vr_batch_executing = false;
}

const std::vector<const rsx::vr::fragment_constant_override*>* VKGSRender::find_fragment_constant_overrides()
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (!profile || profile->fragment_constant_overrides.empty() || !m_vertex_prog)
	{
		return nullptr;
	}

	// Looked up once per vertex program (the hash walks the ucode).
	if (m_fc_overrides_program != m_vertex_prog || m_fc_overrides_profile != profile)
	{
		m_fc_overrides_program = m_vertex_prog;
		m_fc_overrides_profile = profile;
		m_fc_overrides.clear();
		const u64 hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
		for (const auto& rule : profile->fragment_constant_overrides)
		{
			if (rule.program == hash)
			{
				m_fc_overrides.push_back(&rule);
			}
		}
	}

	return m_fc_overrides.empty() ? nullptr : &m_fc_overrides;
}

void VKGSRender::scale_offset_constants(void* buffer, std::span<const u16> constant_ids)
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (!profile || profile->resolution_scaled_constants.empty() || !m_vertex_prog ||
		(resolution_scaling_config.scale_percent == 100 && resolution_scaling_config.percent_y() == 100))
	{
		return;
	}

	// The rule for this vertex program, looked up once per program (the hash walks the ucode).
	if (m_scaled_constants_program != m_vertex_prog || m_scaled_constants_profile != profile)
	{
		m_scaled_constants_program = m_vertex_prog;
		m_scaled_constants_profile = profile;
		m_scaled_constants_slots = nullptr;
		const u64 hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
		for (const auto& rule : profile->resolution_scaled_constants)
		{
			if (rule.program == hash)
			{
				m_scaled_constants_slots = &rule.constant_slots;
				break;
			}
		}
	}

	if (!m_scaled_constants_slots)
	{
		return;
	}

	// Written from the guest registers, never read back from the (write-combined) buffer.
	// Taps are (x, y) pairs: x and z take the horizontal scale, y and w the vertical one (headset-shaped eyes scale them apart).
	const f32 k[4] = {100.f / resolution_scaling_config.scale_percent, 100.f / resolution_scaling_config.percent_y(),
		100.f / resolution_scaling_config.scale_percent, 100.f / resolution_scaling_config.percent_y()};
	for (const u16 slot : *m_scaled_constants_slots)
	{
		usz index = slot;
		if (!constant_ids.empty())
		{
			const auto found = std::find(constant_ids.begin(), constant_ids.end(), slot);
			if (found == constant_ids.end())
			{
				continue;
			}
			index = found - constant_ids.begin();
		}
		const auto& reg = rsx::method_registers.transform_constants[slot];
		f32 value[4];
		for (u32 c = 0; c < 4; ++c)
		{
			value[c] = std::bit_cast<f32>(reg[c]) * k[c];
		}
		std::memcpy(static_cast<u8*>(buffer) + index * 16, value, sizeof(value));
	}
}

bool VKGSRender::vr_apply_box_scissor()
{
	VkRect2D scissor;
	if (!vr_box_scissor_rect(scissor))
	{
		return false;
	}
	vkCmdSetScissor(*m_current_command_buffer, 0, 1, &scissor);
	return true;
}

// The HUD-box scissor of the eye whose constants the probe mapped last (multiview: one per eye, set later).
bool VKGSRender::vr_box_scissor_rect(VkRect2D& scissor)
{
	const f32 clip_w = rsx::method_registers.surface_clip_width();
	const f32 clip_h = rsx::method_registers.surface_clip_height();
	if (clip_w <= 0.f || clip_h <= 0.f)
	{
		return false;
	}

	f32 rect[4] =
		{
			static_cast<f32>(m_scissor.offset.x), static_cast<f32>(m_scissor.offset.y),
			static_cast<f32>(m_scissor.offset.x + m_scissor.extent.width), static_cast<f32>(m_scissor.offset.y + m_scissor.extent.height)};
	// Only the displayed part of the target: the box can reach past the guest viewport with the head
	// turned, and Gran Turismo 5 keeps its glyphs in the unshown part of its 2048x1080 buffers.
	const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
	const f32 shown_w = out.width ? std::min<f32>(clip_w, static_cast<f32>(out.width)) : clip_w;
	const f32 shown_h = out.height ? std::min<f32>(clip_h, static_cast<f32>(out.height)) : clip_h;
	if (!rsx::vr::camera_probe::get().map_box_scissor(m_viewport.width / clip_w, m_viewport.height / clip_h,
			m_viewport.width * shown_w / clip_w, m_viewport.height * shown_h / clip_h, rect))
	{
		return false;
	}

	scissor.offset.x = static_cast<s32>(std::floor(rect[0]));
	scissor.offset.y = static_cast<s32>(std::floor(rect[1]));
	scissor.extent.width = static_cast<u32>(std::ceil(rect[2]) - scissor.offset.x);
	scissor.extent.height = static_cast<u32>(std::ceil(rect[3]) - scissor.offset.y);
	return true;
}

bool VKGSRender::bind_vr_eye_constants(f32 eye_sign, u64 source_offset, usz source_size)
{
	rsx::vr::camera_probe::get().clear_box_mapped();
	if (!source_size || !m_program || m_vs_binding_table->cbuf_location == umax)
	{
		return false;
	}

	// Never read the mapped ring back: it is write-combined memory, where CPU
	// reads are uncached and cost tens of microseconds per draw (hundreds of ms
	// per second in races with many ships). Refill the constants from the guest
	// registers exactly as upload_transform_constants() does, transform them in
	// a CPU scratch buffer, then write the result once, sequentially.
	const bool full_bank = m_shader_interpreter.is_interpreter(m_program) || (m_vertex_prog && m_vertex_prog->has_indexed_constants);
	if (!full_bank && !m_vertex_prog)
	{
		return false;
	}

	const usz size = full_bank ? 8192 : m_vertex_prog->constant_ids.size() * 16;
	if (!size || size != source_size)
	{
		// The last upload does not describe this program; leave the guest allocation bound.
		{
			// Diagnostic: each program left on the game's constants this way (first 32).
			static std::set<u64> s_seen;
			const u64 hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
			if (s_seen.size() < 32 && s_seen.insert(hash).second)
			{
				rsx_log.warning("VR: eye constants not applied to program %016llx (constants %u bytes, last upload %u)", hash, size, source_size);
			}
		}
		return false;
	}

	const bool prof = m_gpuprof_enabled > 0;
	auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
	const auto lap = [&](int i)
	{
		if (!prof)
			return;
		const auto t = std::chrono::steady_clock::now();
		m_gpuprof_eye_ms[i] += std::chrono::duration<f64, std::milli>(t - t0).count();
		t0 = t;
	};
	static thread_local std::vector<u8> scratch;
	const usz remap_size = vr_depth_remap_size(); // profile depth_remap_programs: the eye's depth remap follows the constants
	scratch.resize(size + remap_size);
	const auto constant_ids = full_bank ? std::span<const u16>{} : std::span<const u16>(m_vertex_prog->constant_ids);
	// Ordinary stores, as fill_vertex_program_constants_data would but without its non-temporal (streaming)
	// stores, which suit the write-combined ring: the camera classification reads this buffer right back,
	// and reading streamed data stalled on memory (~5% of the RSX thread in Ratchet & Clank).
	const auto& guest_constants = rsx::method_registers.transform_constants;
	if (constant_ids.empty())
	{
		std::memcpy(scratch.data(), guest_constants.data(), 468 * 16);
	}
	else
	{
		u8* dst = scratch.data();
		for (const u16 index : constant_ids)
		{
			std::memcpy(dst, &guest_constants[index], 16);
			dst += 16;
		}
	}
	if (const auto& probe = rsx::vr::camera_probe::get(); probe.enabled())
	{
		probe.apply(scratch.data(), constant_ids.data(), constant_ids.size(), guest_constants.data(),
			rsx::method_registers.surface_clip_width(), rsx::method_registers.surface_clip_height());
	}
	scale_offset_constants(scratch.data(), constant_ids);
	lap(0);

	const u16* reloc = full_bank ? nullptr : m_vertex_prog->constant_ids.data();
	const usz reloc_size = full_bank ? 0 : m_vertex_prog->constant_ids.size();
	// Programs indexing a bone palette get the whole bank: match camera blocks on the slots they read directly.
	const auto* vr_profile = rsx::vr::camera_probe::get().profile();
	const rsx::vr::full_bank_direct_slots direct_slots(full_bank && m_vertex_prog && vr_profile && vr_profile->camera_slots_read_directly ? &m_vertex_prog->constant_ids : nullptr);
	// Programs kept on the game camera: the profile's game_camera_programs, or probe gamecam= (development).
	const auto& probe_game_camera = rsx::vr::camera_probe::get().game_camera_programs();
	const auto& probe_game_camera_nocolor = rsx::vr::camera_probe::get().game_camera_nocolor_programs();
	bool keep_game_camera = false;
	if (!probe_game_camera.empty() || !probe_game_camera_nocolor.empty() || (vr_profile && !vr_profile->game_camera_programs.empty()))
	{
		const u64 hash = vr_vertex_program_hash();
		const bool no_colour = std::none_of(std::begin(m_framebuffer_layout.color_write_enabled), std::end(m_framebuffer_layout.color_write_enabled), [](bool b)
								   {
									   return b;
								   }) ||
		                       !rsx::method_registers.color_write_enabled(0);
		keep_game_camera = std::find(probe_game_camera.begin(), probe_game_camera.end(), hash) != probe_game_camera.end() ||
		                   (no_colour && std::find(probe_game_camera_nocolor.begin(), probe_game_camera_nocolor.end(), hash) != probe_game_camera_nocolor.end()) ||
		                   (vr_profile && std::find(vr_profile->game_camera_programs.begin(), vr_profile->game_camera_programs.end(), hash) != vr_profile->game_camera_programs.end());
	}
	const bool classified_world = !keep_game_camera && rsx::vr::camera_probe::get().apply_render_eye(scratch.data(), reloc, reloc_size,
														   m_framebuffer_layout.width, m_framebuffer_layout.height, eye_sign);
	vr_write_depth_remap(scratch.data() + size, remap_size, true);
	lap(1);

	const u64 alignment = m_device->gpu().get_limits().minUniformBufferOffsetAlignment;
	const u64 allocation = m_transform_constants_allocator->alloc_bytes(utils::align(size + remap_size, alignment));
	void* destination = m_transform_constants_ring_info.map(allocation, size + remap_size);
	std::memcpy(destination, scratch.data(), size + remap_size);
	m_transform_constants_ring_info.unmap();

	m_xform_constants_dynamic_offset = allocation;
	m_vertex_constants_buffer_info = m_transform_constants_ring_info.window<16>(allocation, size + remap_size,
		m_device->gpu().get_limits().maxUniformBufferRange);
	m_xform_constants_dynamic_offset -= m_vertex_constants_buffer_info.offset;
	m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex,
		m_vs_binding_table->cbuf_location);
	lap(2);
	return classified_world;
}

void VKGSRender::vr_mirror_blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate)
{
	// A blit between render targets is a guest operation the right eye has to see
	// too. Pure copies its finished frame to the display buffer this way
	// (c0000000 -> c0398000 in 1024- and 256-column chunks); without the mirror the
	// right eye keeps whatever its own draws last left there, e.g. the unblurred
	// scene behind the pause menu. Mirrored: 1:1 linear copies between two
	// right-eye colour surfaces, the same staged through memory with no surface
	// (out and back, ICO), and scaled copies between two depth surfaces (ICO).
	// Anything else (texture uploads, mip chains in main memory, scaled or
	// swizzled colour blits) stays single-shot, as the left eye's texture cache is
	// what both eyes sample for non-surface addresses anyway.
	using namespace rsx::blit_engine;
	const bool src_argb8 = src.format == transfer_source_format::a8r8g8b8;
	const bool dst_argb8 = dst.format == transfer_destination_format::a8r8g8b8;
	// Multiview: the right eye is layer 1 of the ordinary surfaces, and the blitter copied it for every
	// blit between surfaces. Only the staging through memory below is still needed (from and into layer 1).
	const bool mv = m_vr_multiview;
	auto& rtts = mv ? m_rtts : m_vr_right_rtts;

	// A scaled copy between two depth surfaces: ICO halves its depth buffer
	// (0xc0f70000 -> 0xc12e0000) for a half-resolution glow and particle pass.
	if (!mv && !dst.swizzled && !dst.clip_x && !dst.clip_y && src_argb8 && dst_argb8)
	{
		const u32 src_address = vm::get_addr(src.pixels);
		const u32 dst_address = vm::get_addr(dst.pixels);
		auto* src_depth = m_vr_right_rtts.get_surface_at(src_address);
		auto* dst_depth = m_vr_right_rtts.get_surface_at(dst_address);
		if (src_depth && dst_depth && src_depth != dst_depth &&
			src_depth->base_addr == src_address && dst_depth->base_addr == dst_address &&
			(src_depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT) && (dst_depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT) &&
			src_depth->format() == dst_depth->format() && src_depth->samples() == 1 && dst_depth->samples() == 1)
		{
			// Guest pixels to the surface's resolution scale.
			const auto area = [](vk::render_target* surface, u32 w, u32 h)
			{
				const f32 kx = static_cast<f32>(surface->width()) / surface->template get_surface_width<rsx::surface_metrics::pixels>();
				const f32 ky = static_cast<f32>(surface->height()) / surface->template get_surface_height<rsx::surface_metrics::pixels>();
				return areai{0, 0, static_cast<int>(std::min<f32>(w * kx, surface->width())), static_cast<int>(std::min<f32>(h * ky, surface->height()))};
			};

			vr_batch_flush();
			src_depth->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_read);
			dst_depth->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_write);
			vk::copy_scaled_image(*m_current_command_buffer, src_depth, dst_depth,
				area(src_depth, src.width, src.height), area(dst_depth, dst.clip_width, dst.clip_height), {}, true, VK_FILTER_NEAREST);
			dst_depth->on_write_copy(rsx::get_shared_tag());
			return;
		}
	}

	if (!rsx::fcmp(dst.scale_x, 1.f) || !rsx::fcmp(dst.scale_y, 1.f) || dst.swizzled || dst.clip_x || dst.clip_y || src_argb8 != dst_argb8)
	{
		if (mv)
		{
			return; // the blitter wrote both layers
		}
		// Scaled copies between two right-eye surfaces go through the texture cache's
		// own blit on the right-eye store: Blur resolves its 4x MSAA scene
		// (0xc0af0000, 2560 pitch) and depth (0xc1220000) to 1280x720 targets with
		// 0.5x blits in 1024/1024/512-column chunks.
		vr_batch_flush();
		m_texture_cache.blit_vr_right(src, dst, interpolate, m_vr_right_rtts, *m_current_command_buffer);
		return;
	}

	const u8 bpp = src_argb8 ? 4 : 2;
	const u16 width = dst.clip_width;
	const u16 height = dst.clip_height;

	// The copy rectangle inside a right-eye surface, at the surface's resolution scale.
	const auto locate = [&](u32 address, u32 pitch, vk::render_target*& surface, areai& rect)
	{
		surface = rtts.find_color_surface(address, pitch);
		if (!surface || surface->samples() != 1 || surface->get_bpp() != bpp)
		{
			return false;
		}

		const u32 offset = address - surface->base_addr;
		const u32 x = (offset % pitch) / bpp;
		const u32 y = offset / pitch;
		const u32 surface_w = surface->template get_surface_width<rsx::surface_metrics::pixels>();
		const u32 surface_h = surface->template get_surface_height<rsx::surface_metrics::pixels>();
		if ((offset % pitch) % bpp || x + width > surface_w || y + height > surface_h)
		{
			return false;
		}

		const f32 kx = static_cast<f32>(surface->width()) / surface_w;
		const f32 ky = static_cast<f32>(surface->height()) / surface_h;
		rect = {static_cast<int>(x * kx), static_cast<int>(y * ky),
			static_cast<int>((x + width) * kx), static_cast<int>((y + height) * ky)};
		return true;
	};

	const u32 src_address = vm::get_addr(src.pixels);
	const u32 dst_address = vm::get_addr(dst.pixels);
	vk::render_target* src_surface = nullptr;
	vk::render_target* dst_surface = nullptr;
	areai src_rect, dst_rect;
	const bool src_found = locate(src_address, src.pitch, src_surface, src_rect);
	bool dst_found = locate(dst_address, dst.pitch, dst_surface, dst_rect);

	// The blit made its destination a render target on the left (the texture cache promotes a
	// copy of a surface into memory no surface holds) and draws then sample it as one: Bayonetta
	// copies its scene (0xcf460000, 1024- and 256-column chunks) to 0xce99c000 and redraws the
	// whole scene from that copy. Give the right eye the same surface, or every right-eye pass
	// after the copy samples the left eye's image and both eyes end up identical.
	if (!mv && src_found && !dst_found && !m_vr_right_rtts.find_color_surface(dst_address, dst.pitch))
	{
		if (auto* left = m_rtts.find_color_surface(dst_address, dst.pitch); left && left->samples() == 1 && !left->is_depth_surface() &&
																			left->get_bpp() == bpp && !m_vr_right_rtts.get_surface_at(left->base_addr))
		{
			const rsx::image_section_attributes_t attr{
				.address = left->base_addr,
				.gcm_format = left->get_gcm_format(),
				.pitch = left->get_rsx_pitch(),
				.width = static_cast<u16>(left->template get_surface_width<rsx::surface_metrics::pixels>()),
				.height = static_cast<u16>(left->template get_surface_height<rsx::surface_metrics::pixels>()),
				.depth = 1,
				.mipmaps = 1,
				.slice_h = static_cast<u16>(left->template get_surface_height<rsx::surface_metrics::pixels>()),
				.bpp = bpp,
				.swizzled = false,
				.edge_clamped = false};
			vr_batch_flush();
			if (auto* right = m_vr_right_rtts.create_surface_from_rsx_section(*m_current_command_buffer, attr, left->get_resolution_scaling_config());
				right && right->format() == left->format() && right->width() == left->width() && right->height() == left->height())
			{
				// The copies below fill it; what they do not cover starts as the left eye's pixels.
				right->state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
				right->old_contents.clear();
				if (vk::is_renderpass_open(*m_current_command_buffer))
				{
					vk::end_renderpass(*m_current_command_buffer);
				}
				left->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_read);
				right->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_write);
				vk::copy_image(*m_current_command_buffer, left, right, areai{0, 0, static_cast<int>(left->width()), static_cast<int>(left->height())},
					areai{0, 0, static_cast<int>(right->width()), static_cast<int>(right->height())});
				dst_found = locate(dst_address, dst.pitch, dst_surface, dst_rect);

				static bool s_reported = false;
				if (!std::exchange(s_reported, true))
				{
					rsx_log.notice("VR: right-eye surface created at 0x%x for a blit destination the left eye made a render target (%ux%u).",
						left->base_addr, attr.width, attr.height);
				}
			}
		}
	}

	if (src_found && dst_found)
	{
		if (mv || src_rect.width() != dst_rect.width() || src_rect.height() != dst_rect.height())
		{
			return; // (multiview: the blitter wrote both layers)
		}

		// Batched right-eye draws precede this copy in guest order.
		vr_batch_flush();

		src_surface->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_read);
		dst_surface->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_write);
		vk::copy_image(*m_current_command_buffer, src_surface, dst_surface, src_rect, dst_rect);
		dst_surface->on_write_copy(rsx::get_shared_tag());
		return;
	}

	// Staging through memory with no surface: ICO copies its frame out to main
	// memory (0x30900000) and back into other targets. Keep the right eye's pixels
	// in a host image keyed by the staging address, for the copy back.
	if (src_found && !rtts.find_color_surface(dst_address, dst.pitch))
	{
		// Host pixels per guest pixel of the source surface (resolution scale).
		const f32 kx = static_cast<f32>(src_rect.width()) / width;
		const u16 row_width = static_cast<u16>(dst.pitch / bpp);
		const u32 w = static_cast<u32>(row_width * kx + 0.5f), h = src_rect.height();

		// A chunk of a copy already staged: same pitch, starting inside its first row.
		auto found = std::find_if(m_vr_staged.begin(), m_vr_staged.end(), [&](const vr_staged_copy& s)
			{
				return s.pitch == dst.pitch && s.bpp == bpp && dst_address >= s.address && dst_address < s.address + s.pitch &&
			           s.image && s.image->format() == src_surface->format() && s.image->height() == h && s.image->width() == w;
			});
		if (found == m_vr_staged.end())
		{
			found = std::find_if(m_vr_staged.begin(), m_vr_staged.end(), [&](const vr_staged_copy& s)
				{
					return s.address == dst_address && s.pitch == dst.pitch;
				});
		}
		auto& staged = found != m_vr_staged.end() ? *found : m_vr_staged.emplace_back();
		const bool chunk = staged.image && dst_address != staged.address;
		if (!chunk && (!staged.image || staged.image->width() != w || staged.image->height() != h || staged.image->format() != src_surface->format()))
		{
			if (staged.image)
			{
				// Earlier copies may still be in flight.
				vk::get_resource_manager()->dispose(staged.image);
			}
			staged.image = std::make_unique<vk::image>(*m_device, m_device->get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D, src_surface->format(), w, h, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				0, VMM_ALLOCATION_POOL_SYSTEM);
		}
		if (!chunk)
		{
			staged.address = dst_address;
			staged.pitch = dst.pitch;
			staged.bpp = bpp;
			staged.width = row_width;
			staged.height = height;
		}
		const int x0 = static_cast<int>(((dst_address - staged.address) / bpp) * kx + 0.5f);
		const areai staged_rect{x0, 0, std::min<int>(x0 + src_rect.width(), static_cast<int>(w)), static_cast<int>(h)};
		const areai src_used{src_rect.x1, src_rect.y1, src_rect.x1 + staged_rect.width(), src_rect.y2};

		vr_batch_flush();
		if (staged.image->current_layout == VK_IMAGE_LAYOUT_UNDEFINED)
		{
			// copy_image returns the image to its prior layout, which must be a real one.
			if (vk::is_renderpass_open(*m_current_command_buffer))
			{
				vk::end_renderpass(*m_current_command_buffer);
			}
			vk::change_image_layout(*m_current_command_buffer, staged.image.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		}
		src_surface->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_read);
		vk::copy_image(*m_current_command_buffer, src_surface, staged.image.get(), src_used, staged_rect, {.src_layer = mv ? u8{1} : u8{0}});
		return;
	}

	if (dst_found)
	{
		const auto staged = std::find_if(m_vr_staged.begin(), m_vr_staged.end(), [&](const vr_staged_copy& s)
			{
				return s.image && s.pitch == src.pitch && src_address >= s.address && src_address < s.address + s.pitch &&
			           (src_address - s.address) / s.bpp + width <= s.width && s.height >= height &&
			           s.image->height() >= static_cast<u32>(dst_rect.height()) && s.image->format() == dst_surface->format();
			});
		if (staged != m_vr_staged.end())
		{
			const f32 kx = static_cast<f32>(staged->image->width()) / staged->width;
			const int x0 = static_cast<int>(((src_address - staged->address) / staged->bpp) * kx + 0.5f);
			const int x1 = std::min<int>(x0 + dst_rect.width(), static_cast<int>(staged->image->width()));

			vr_batch_flush();
			dst_surface->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_write);
			vk::copy_image(*m_current_command_buffer, staged->image.get(), dst_surface,
				areai{x0, 0, x1, dst_rect.height()}, areai{dst_rect.x1, dst_rect.y1, dst_rect.x1 + (x1 - x0), dst_rect.y2}, {.dst_layer = mv ? u8{1} : u8{0}});
			if (!mv)
			{
				dst_surface->on_write_copy(rsx::get_shared_tag());
			}
			return;
		}
	}
}

bool VKGSRender::vr_reprojects_older_frames() const
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	return profile && profile->reproject_older_frames;
}

// vr_before_blit(): profile texture_redirects also for blits. A blit out of the main-memory copy the SPUs process
// (Dragon Age II shows its scene by blitting it back from 0x33000000, copied there from its scene target the frame
// before) reads the render target instead, so each eye gets its own picture (main memory holds one eye: both eyes
// showed the same scene, only the sky drawn later differed). Any blit starting inside the copy is moved by the same offset.
void VKGSRender::vr_redirect_blit_source(rsx::blit_src_info& src)
{
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (!profile || profile->texture_redirects.empty() || !rsx::vr::camera_probe::get().render_enabled())
	{
		return;
	}
	const u32 address = vm::get_addr(src.pixels);
	for (const auto& [from, profile_to] : profile->texture_redirects)
	{
		if (address < from)
			continue;
		const u32 to = vr_redirect_target(profile_to, src.pitch);
		const auto* target = to ? m_rtts.find_color_surface(to, src.pitch) : nullptr;
		if (!target || target->base_addr != to || address - from >= src.pitch * target->get_surface_height<rsx::surface_metrics::samples>())
			continue;
		const s64 delta = static_cast<s64>(to) - static_cast<s64>(from);
		static bool s_reported = false;
		if (!std::exchange(s_reported, true))
		{
			rsx_log.success("VR: blits out of 0x%x read the render target 0x%x instead (profile texture_redirects).", from, to);
		}
		src.rsx_address = static_cast<u32>(src.rsx_address + delta);
		src.pixels += delta;
		return;
	}
}

// texture_redirects' `to`: the profile's address, or for "camera" (0) the newest colour target of this frame's camera
// draws with that pitch (Dragon Age II draws its scene into 0xc03c0000 and 0xc0000000 on alternate frames; the copy it
// blits back was taken from the other one, which by then holds the finished frame: read, it fed back into itself).
u32 VKGSRender::vr_redirect_target(u32 profile_to, u32 pitch)
{
	if (profile_to)
	{
		return profile_to;
	}
	for (auto it = m_vr_camera_targets.rbegin(); it != m_vr_camera_targets.rend(); ++it)
	{
		if (const auto* surface = m_rtts.find_color_surface(*it, pitch); surface && surface->base_addr == *it && (!pitch || surface->get_rsx_pitch() == pitch))
		{
			return *it;
		}
	}
	return 0;
}

void VKGSRender::vr_redirect_previous_frame_copy(rsx::blit_src_info& src)
{
	// A scene target drawn with an earlier pose than the current frame's, while a
	// twin (same size, format and pitch) was drawn in this frame: read the twin.
	auto* from = m_rtts.find_color_surface(src.rsx_address, src.pitch);
	if (!from || !from->vr_pose || from->vr_pose >= m_vr_applied_pose)
	{
		return;
	}

	for (auto it = m_vr_camera_targets.rbegin(); it != m_vr_camera_targets.rend(); ++it)
	{
		if (*it == from->base_addr)
		{
			continue;
		}

		auto* to = m_rtts.find_color_surface(*it, from->get_rsx_pitch());
		if (!to || to->base_addr != *it || to->vr_pose != m_vr_applied_pose || to->format() != from->format() ||
			to->width() != from->width() || to->height() != from->height() || to->samples() != from->samples())
		{
			continue;
		}

		const s64 delta = static_cast<s64>(to->base_addr) - static_cast<s64>(from->base_addr);
		static bool s_reported = false;
		if (!std::exchange(s_reported, true))
		{
			rsx_log.success("VR: copies of the previous frame's scene read this frame's (0x%x instead of 0x%x).", to->base_addr, from->base_addr);
		}
		if (vr_tracing())
		{
			m_vr_trace += fmt::format(" R%x>%x", src.rsx_address, static_cast<u32>(src.rsx_address + delta));
		}
		src.rsx_address = static_cast<u32>(src.rsx_address + delta);
		src.pixels += delta;
		return;
	}
}

// Locate the head for the next game frame: its camera draws are rotated by that
// pose, and the frame is declared with it when the frame thread presents it.
void VKGSRender::vr_update_view()
{
	m_vr_frame_covered.clear();
	f32 head[4];
	f32 head_position[3];
	f32 eye_fov[2][4];
	f32 render_fov[2][4];
	auto& probe = rsx::vr::camera_probe::get();
	// A game frame with no camera draws has no 3D to follow the head (splash screens,
	// videos, 2D menus): show it as the fixed screen, which respects the HUD settings,
	// instead of stretched over the whole view. Back to the headset view on the first
	// frame with a camera draw (that frame is still shown as the screen).
	// A frame with no camera draws that shows earlier 3D content (a paused game re-showing its last frame under its
	// menu) keeps that content's pose: drawn and declared with it, the compositor holds it world-fixed. With the
	// newest pose it followed the head (Kingdom Hearts' pause), and on the fixed screen it was squeezed to 16:9.
	const u32 vr_frozen_pose = !m_vr_camera_draws && m_vr_flip_has_3d ? m_vr_flip_pose : 0;
	m_vr_frames_without_camera = m_vr_camera_draws ? 0 : m_vr_frames_without_camera + 1;
	m_vr_frames_2d = m_vr_camera_draws || m_vr_flip_has_3d ? 0 : m_vr_frames_2d + 1;
	m_vr_camera_draws = 0;
	const auto* no_3d_profile = probe.profile();
	using frames_without_3d_mode = rsx::vr::title_profile::frames_without_3d_mode;
	const auto no_3d_mode = no_3d_profile ? no_3d_profile->screen_space_frames_without_3d_as_screen : frames_without_3d_mode::never;
	// A frame with one of the profile's screen_frame_draws (a composed menu), or drawn while a screen_frames_when
	// game-state word says front end, goes on the fixed screen as a whole.
	// A cinematic frame (profile reduced_scale_frames: Gran Turismo 5's pre-race views) follows the Cinematic Scenes
	// setting alone: Fixed Screen, or the headset view at Full Quality. Before, the depth-of-field pass the pre-race
	// shots share with the main menu (screen_frame_draws) could send them to the fixed screen at any setting.
	rsx::vr::note_reduced_scale_frame(m_vr_reduced_scale_draws != 0);
	m_vr_reduced_scale_draws = 0;
	const bool screen_frame = rsx::vr::cinematic_frame() ? rsx::vr::cinematic_frame_on_screen() :
		(m_vr_screen_frame_draws != 0 || rsx::vr::screen_frame_by_game_state());
	m_vr_screen_frame_draws = 0;
	const bool no_3d = screen_frame || (no_3d_mode == frames_without_3d_mode::always ? m_vr_frames_without_camera >= 3 :
																					   no_3d_mode == frames_without_3d_mode::automatic && m_vr_frames_2d >= 3);
	if (static bool s_no_3d = false; no_3d != s_no_3d)
	{
		s_no_3d = no_3d;
		rsx_log.notice("VR: %s", no_3d ? "frames without camera draws: shown as the fixed screen" : "camera draws again: headset view");
	}
	m_vr_video_on_screen = no_3d;
	// An orthographic game (profile orthographic_stereo: Fez) has no view to turn with the head: it is a diorama on the
	// fixed screen, in stereo.
	const bool orthographic = no_3d_profile && no_3d_profile->orthographic_stereo_angle > 0.f;
	const bool fixed_screen = g_cfg.video.vr.fixed_screen || !vk::xr::projection_mode() || no_3d || orthographic;
	// HUD stereo distance, and the fixed screen's distance (metres): the HUD Depth setting. The box keeps its
	// angular size (HUD Scale), so a larger depth moves it away without shrinking it.
	const f32 vr_hud_distance = rsx::vr::effective_hud_depth();
	u32 pose = vr_frozen_pose ? vk::xr::recall_render_pose(vr_frozen_pose, head, head_position, eye_fov, render_fov) : 0;
	if (!pose)
	{
		pose = vk::xr::locate_render_pose(head, head_position, eye_fov, render_fov, static_cast<f32>(rsx::vr::effective_reprojection_margin()));
	}
	const bool located = pose != 0;
	m_vr_applied_pose = 0;

	// The HUD box: the game's output aspect, fitted in the central symmetric part of
	// both eyes' views, scaled by the HUD settings, at the HUD Depth. The fixed screen and
	// RPCS3's overlays use it.
	const size2u output_size = g_fxo->get<rsx::avconf>().video_frame_size();
	const f32 aspect = output_size.width && output_size.height ? static_cast<f32>(output_size.width) / output_size.height : 16.f / 9.f;
	const f32 depth = vr_hud_distance;
	if (located)
	{
		// Headset-shaped eyes: the rendered eye extents set the vertical / horizontal resolution scale (rsx_vr_eye_shape.h).
		rsx::vr::set_eye_shape(render_fov[0][1] - render_fov[0][0], render_fov[0][2] - render_fov[0][3], aspect);
		vr_apply_eye_shape_early();
	}
	f32 box_y = 0.f;
	f32 width = 0.f;
	if (located)
	{
		const f32 fit_x = std::min({-eye_fov[0][0], eye_fov[0][1], -eye_fov[1][0], eye_fov[1][1]});
		const f32 fit_y = std::min({eye_fov[0][2], -eye_fov[0][3], eye_fov[1][2], -eye_fov[1][3]});
		box_y = std::min(fit_y, fit_x / aspect);
		width = 2.f * depth * box_y * aspect * g_cfg.video.vr.hud_scale.get() / 100.f;
		vk::xr::set_overlay_placement(g_cfg.video.vr.hud_fixed.get(), width,
			depth * box_y * aspect * g_cfg.video.vr.hud_offset_x.get() / 100.f,
			depth * box_y * g_cfg.video.vr.hud_offset_y.get() / 100.f,
			depth);
	}

	if (fixed_screen && located)
	{
		// Fixed screen: the game keeps its own camera and stereo; the HUD sliders
		// place the window where the HUD box would be (depth 0 = 2 m).
		probe.clear_vr_view();

		// The game's stereo separates far objects by a fixed fraction of the picture,
		// tuned for the profile's reference screen (WipEout: a 0.53 m, 24" TV). On a
		// wider window that would make the eyes diverge, so keep that screen's physical
		// disparity by scaling the separation by reference width / window width, then
		// by the user's strength.
		const auto* profile = probe.profile();
		const f32 reference_width = profile ? profile->reference_screen_width : 0.f;
		const f32 auto_scale = reference_width > 0.f && width > reference_width ? reference_width / width : 1.f;
		probe.set_screen_stereo_scale(auto_scale * g_cfg.video.vr.screen_depth.get() / 100.f);

		// A frame on the fixed screen as a menu, pause or video (not the user's Fixed Screen mode) is flat: both eyes
		// show the left eye's image. Its 3D parts (R&C 1's map over the frozen level, 3D menu models) otherwise kept
		// the game's stereo inside the screen, and the boxes did not line up between the eyes.
		vk::xr::set_screen_mono(no_3d && !g_cfg.video.vr.fixed_screen && vk::xr::projection_mode());
		vk::xr::set_screen(true, g_cfg.video.vr.hud_fixed.get(),
			width,
			depth * box_y * aspect * g_cfg.video.vr.hud_offset_x.get() / 100.f,
			depth * box_y * g_cfg.video.vr.hud_offset_y.get() / 100.f,
			depth);
	}
	else if (!fixed_screen && located)
	{
		vk::xr::set_screen(false, true, 0.f, 0.f, 0.f, 0.f);
		probe.set_screen_stereo_scale(1.f);
		// World Scale: a bigger world is a smaller viewer, i.e. less eye separation in game units.
		{
			// VR fork dev hook: RPCS3_VR_WOBBLE=<degrees> sweeps the rendered head yaw over that range, to
			// test world-fixed HUD boxes and trails without moving the headset.
			static const f32 s_wobble = []
			{
				const char* v = ::getenv("RPCS3_VR_WOBBLE");
				return v ? static_cast<f32>(std::atof(v)) : 0.f;
			}();
			// RPCS3_VR_YAW_FILE=<file>: a fixed rendered head yaw in degrees, read from the file (re-read every 30 frames).
			static const std::string s_yaw_file = []() -> std::string
			{
				const char* v = ::getenv("RPCS3_VR_YAW_FILE");
				return v ? v : "";
			}();
			static f32 s_yaw = 0.f;
			if (static u32 s_yaw_n = 0; !s_yaw_file.empty() && s_yaw_n++ % 30 == 0)
			{
				if (fs::file f{s_yaw_file}; f)
				{
					s_yaw = static_cast<f32>(std::atof(f.to_string().c_str()));
				}
			}
			if (s_wobble != 0.f || s_yaw != 0.f)
			{
				static u32 s_n = 0;
				const f32 a = s_wobble != 0.f ? 0.5f * s_wobble * 3.14159265f / 180.f * std::sin(++s_n * 0.05f) : s_yaw * 3.14159265f / 180.f;
				const f32 qy[4] = {0.f, std::sin(a), 0.f, std::cos(a)};
				const f32 h[4] = {head[0], head[1], head[2], head[3]};
				head[0] = qy[3] * h[0] + qy[0] * h[3] + qy[1] * h[2] - qy[2] * h[1];
				head[1] = qy[3] * h[1] - qy[0] * h[2] + qy[1] * h[3] + qy[2] * h[0];
				head[2] = qy[3] * h[2] + qy[0] * h[1] - qy[1] * h[0] + qy[2] * h[3];
				head[3] = qy[3] * h[3] - qy[0] * h[0] - qy[1] * h[1] - qy[2] * h[2];
			}
		}
		{
			// VR fork dev hook: RPCS3_VR_HEAD_OFFSET=x,y,z (metres) moves the rendered head, e.g. 0,0,0.3 leans back.
			static const std::array<f32, 3> s_offset = []
			{
				std::array<f32, 3> o{};
				if (const char* v = ::getenv("RPCS3_VR_HEAD_OFFSET"))
					std::sscanf(v, "%f,%f,%f", &o[0], &o[1], &o[2]);
				return o;
			}();
			for (u32 i = 0; i < 3; ++i)
				head_position[i] += s_offset[i];
		}
		probe.set_vr_view(head, head_position, vk::xr::eye_scale() * 100.f / g_cfg.video.vr.world_scale.get(), vk::xr::fov_scale(),
			vk::xr::flip_y(), vk::xr::ipd(), g_cfg.video.vr.camera_depth.get() / 100.f);
		probe.set_vr_eye_fov(vk::xr::hmd_fov() ? render_fov : nullptr, eye_fov,
			g_cfg.video.vr.hud_scale.get() / 100.f, g_cfg.video.vr.hud_fixed.get(),
			g_cfg.video.vr.hud_offset_x.get() / 100.f, g_cfg.video.vr.hud_offset_y.get() / 100.f,
			vr_hud_distance, vk::xr::ipd());
		m_vr_applied_pose = pose;
	}
	else
	{
		probe.clear_vr_view();
	}
}

void VKGSRender::vr_track_frame_boundary()
{
	// Per-frame poses are part of the older-frame reprojection machinery (profile
	// reproject_older_frames, Ico). Other games keep the per-flip pose update: Pure's pause
	// menu crosses a display-buffer boundary many times per flip.
	if (!vr_reprojects_older_frames())
	{
		return;
	}

	s32 index = -1;
	for (const u32 address : m_framebuffer_layout.color_addresses)
	{
		if (!address)
		{
			continue;
		}
		for (u32 i = 0; i < display_buffers_count; ++i)
		{
			if (display_buffers[i].valid() && rsx::get_address(display_buffers[i].offset, CELL_GCM_LOCATION_LOCAL) == address)
			{
				index = static_cast<s32>(i);
				break;
			}
		}
		break;
	}

	if (vr_tracing())
	{
		const u32 address = m_framebuffer_layout.color_addresses[0] ? m_framebuffer_layout.color_addresses[0] : m_framebuffer_layout.zeta_address;
		if (address != m_vr_trace_addr)
		{
			vr_trace_flush_cam();
			m_vr_trace += index >= 0 ? fmt::format(" B%x[d%d]", address, index) : fmt::format(" B%x", address);
			m_vr_trace_addr = address;
		}
	}

	if (m_vr_display_target >= 0 && index != m_vr_display_target)
	{
		// The game has finished writing a display buffer and moves on: its next frame
		// starts here, and all of it is rotated by one pose.
		if (!m_vr_frame_boundaries)
		{
			rsx_log.success("VR: head pose changes at frame boundaries (leaving display buffer %d).", m_vr_display_target);
		}
		m_vr_frame_boundaries = true;
		m_vr_flips_since_boundary = 0;
		vr_update_view();
		if (vr_tracing())
		{
			m_vr_trace += fmt::format(" Y%u(%.1f, %.1fmm)", m_vr_applied_pose, vk::xr::render_pose_yaw(m_vr_applied_pose),
				vk::xr::render_pose_step_mm(m_vr_applied_pose - 1, m_vr_applied_pose));
		}
	}

	m_vr_display_target = index;
}

// ---- Hooks called from upstream functions -----------------------------------------------
// Each of these used to be an inline block in VKDraw.cpp, VKGSRender.cpp or VKPresent.cpp.
// The upstream functions now carry one call per block; the logic is unchanged.

// emit_geometry(): decide how this draw is rendered for the right eye and keep the guest's
// own constants allocation, which the eye constants replace and the restore puts back.
void VKGSRender::vr_begin_draw()
{
	auto& d = m_vr_draw;
	const auto& draw_call = rsx::method_registers.current_draw_clause;

	// Multiview stereo: both eyes in this one draw (instanced draws included; their constants are shared between the eyes).
	d.mv = m_vr_multiview && rsx::vr::camera_probe::get().render_enabled();
	d.render = d.mv || (rsx::vr::camera_probe::get().render_enabled() && m_vr_right_draw_fbo &&
	                    !draw_call.is_trivial_instanced_draw);
	// A draw that samples a bound render target (a feedback loop) needs the texture barrier
	// between the earlier writes and its read inside the eye's own pass. In the right-eye batch
	// the barrier is recorded in the primary buffer before the batch runs, so the right eye
	// would read its target unsynchronised (Gran Turismo 5's car shadows read the 2x MSAA
	// scene target they draw into). These keep the per-draw replay.
	bool vr_feedback = !!(current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE);
	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; d.render && !vr_feedback && textures_ref; textures_ref >>= 1, ++i)
	{
		vr_feedback = (textures_ref & 1) && fs_sampler_state[i] && fs_sampler_state[i]->is_cyclic_reference;
	}
	// Gate 6: batch the right-eye draw into the current left pass's right-eye batch.
	// Programmable blending (input attachments), conditional rendering and feedback
	// loops keep the per-draw replay: none carries over into a secondary command buffer here.
	d.batch = d.render && !d.mv && m_vr_batching && !vr_feedback &&
	          !(current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING) &&
	          !cond_render_ctrl.hw_cond_active;
	d.query_continuation = umax;
	d.suspend_query = d.render && !d.mv && !d.batch &&
	                  (m_current_command_buffer->flags & vk::command_buffer::cb_has_open_query);
	if (d.suspend_query)
	{
		// A right-eye replay must not contribute samples to the guest's occlusion
		// result. Reserve another slot now so the guest query can be split into
		// two left-eye-only segments around the host draw.
		d.query_continuation = m_occlusion_query_manager->allocate_query(*m_current_command_buffer);
		if (d.query_continuation == umax)
		{
			// Query slots are finite. Preserve guest semantics and omit stereo for
			// this draw rather than allowing the right eye to alter its result.
			d.render = false;
		}
	}
	d.guest_constants_info = m_vertex_constants_buffer_info;
	d.guest_constants_dynamic_offset = m_xform_constants_dynamic_offset;
	d.guest_constants_source_offset = d.guest_constants_info.offset + d.guest_constants_dynamic_offset;

	d.camera_draw = d.hud_env = d.clear_shown = d.dynamic_state_changed = d.box_scissor = d.right_box_scissor = false;
	d.preprojected = 0;
	d.eye_constants_bound = false;
	d.mv_env_right = 0;
	d.right_t0 = {};
}

// emit_geometry(): classify the draw for the camera probe, apply the left eye's constants,
// stamp the pose, set up the HUD-box vertex context and decide the clear of the shown region.
void VKGSRender::vr_setup_draw()
{
	auto& d = m_vr_draw;
	auto& probe = rsx::vr::camera_probe::get();

	if (d.render)
	{
		// Only a view-shaped target makes a pass: Anarchy Reigns' health gauge samples a 256x256 mask the HUD
		// draws first, and with hud_skips_passes it stayed at its screen position outside the box.
		const u32 vr_kinds = vr_sampled_textures();
		probe.set_draw_samples_colour_target((vr_kinds & vr_texture_view_target) || vr_unboxed_draw());
		probe.set_draw_samples_any_colour_target((vr_kinds & vr_texture_colour_target) || vr_unboxed_draw());
		probe.set_draw_hud_scale(vr_hud_draw_scale());
		// A depth-only draw (no colour target) into the display buffers' depth surface is part of the
		// screen too: Gran Turismo 5 masks its track map with one.
		bool display_target = vr_display_buffer(*this, m_framebuffer_layout.color_addresses[0], m_framebuffer_layout.width, m_framebuffer_layout.height);
		if (!m_framebuffer_layout.color_addresses[0] && m_framebuffer_layout.zeta_address)
		{
			for (const auto& buffer : display_buffers)
			{
				display_target |= buffer.width == m_framebuffer_layout.width && buffer.height == m_framebuffer_layout.height;
			}
		}
		probe.set_draw_into_display_buffer(display_target);
		if (const auto* profile = probe.profile(); profile && (!profile->screen_space_hud_block_programs.empty() || !profile->depth_remap_programs.empty()))
		{
			probe.set_draw_program(vr_vertex_program_hash());
		}
		// Depth test that can reject something: a depth buffer bound and a compare other than ALWAYS.
		// Killzone HD draws its menus with depth test on, no depth buffer and ALWAYS; taken as a real
		// test, the HUD box kept their z while the head moved W, and text near the far plane was clipped.
		probe.set_draw_depth_test(rsx::method_registers.depth_test_enabled() && m_framebuffer_layout.zeta_address &&
								  rsx::method_registers.depth_func() != rsx::comparison_function::always);
	}
	probe.clear_hud_env_request();
	const auto vr_left_t0 = d.render && m_gpuprof_enabled > 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
	d.camera_draw = d.mv ? bind_vr_eye_constants_pair(m_xform_constants_data_size) : (d.render && bind_vr_eye_constants(-1.f, d.guest_constants_source_offset, m_xform_constants_data_size));
	if (vr_left_t0 != std::chrono::steady_clock::time_point{})
	{
		m_gpuprof_left_vr_ms += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - vr_left_t0).count();
	}
	m_vr_camera_draws += d.camera_draw;
	if (d.render && vk::xr::is_running())
	{
		if (vr_tracing())
		{
			vr_trace_copy_reads(d.camera_draw);
			if (!d.camera_draw && !m_vr_camera_targets.empty() && m_framebuffer_layout.color_addresses[0] == m_vr_camera_targets.back())
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
		u32 vr_pose = d.camera_draw ? m_vr_applied_pose : vr_sampled_pose();
		if (!vr_pose && m_vr_display_target < 0)
		{
			vr_pose = m_vr_applied_pose;
		}
		vr_stamp_targets(vr_pose, d.camera_draw);
		vr_mark_3d_targets(d.camera_draw);
		if (d.camera_draw && vr_tracing())
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
	else if (d.render && d.camera_draw && vk::xr::fake_hmd())
	{
		// The fake headset (RPCS3_VR_FAKE_HMD) keeps the camera targets the passthrough HUD tests, without the
		// pose stamping (which slowed it down by up to 90%).
		const u32 address = m_framebuffer_layout.color_addresses[0];
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

	// HUD/menu drawn without a matrix: its own vertex context per eye (restored in vr_end_draw).
	const bool vr_hud = d.render && !d.camera_draw && (vk::xr::is_running() || vk::xr::fake_hmd()) &&
	                    (probe.hud_env_requested() || vr_is_passthrough_hud());
	// Dev: probe why=<vertex hash> logs each distinct classification of that program's draws.
	if (const u64 why = probe.why_program(); why && d.render &&
											 program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program) == why)
	{
		const u32 kinds = vr_sampled_textures();
		const u32 target = m_framebuffer_layout.color_addresses[0];
		const bool in_scene = std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), target) != m_vr_camera_targets.end();
		const std::string key = fmt::format("camera %d hud %d env %d passthrough %d textures %u target 0x%x %ux%u in_scene %d box %d state %u viewport %.0fx%.0f clip %ux%u mv %d",
			d.camera_draw, vr_hud, probe.hud_env_requested(), vr_is_passthrough_hud(), kinds, target,
			m_framebuffer_layout.width, m_framebuffer_layout.height, in_scene, probe.box_mapped(), probe.vr_state_bits(),
			std::fabs(rsx::method_registers.viewport_scale_x()) * 2.f, std::fabs(rsx::method_registers.viewport_scale_y()) * 2.f,
			rsx::method_registers.surface_clip_width(), rsx::method_registers.surface_clip_height(), d.mv);
		static std::set<std::string> s_seen;
		if (s_seen.insert(fmt::format("%016llx %s", why, key)).second)
		{
			rsx_log.notice("VR why %016llx: %s", why, key);
		}
		// The first vertices' attribute 0 (float arrays only), to tell clip, NDC and pixel positions apart in programs
		// that take positions the game projected itself. At most 8 draws a probe.
		static u32 s_vertex_logs = 0;
		const auto& pos = rsx::method_registers.vertex_arrays_info[0];
		if (s_vertex_logs < 8 && pos.size() >= 3 && pos.type() == rsx::vertex_base_type::f && pos.stride() &&
			rsx::method_registers.current_draw_clause.command != rsx::draw_command::inlined_array)
		{
			s_vertex_logs++;
			const u32 base = rsx::get_address(rsx::get_vertex_offset_from_base(rsx::method_registers.vertex_data_base_offset(), pos.offset() & 0x7fffffff), pos.offset() >> 31);
			const bool indexed = rsx::method_registers.current_draw_clause.command == rsx::draw_command::indexed;
			const u32 first = rsx::method_registers.current_draw_clause.min_index();
			std::string verts;
			for (u32 v = 0; v < 4; ++v)
			{
				u32 index = first + v;
				if (indexed)
				{
					const bool u32_indices = rsx::method_registers.index_type() == rsx::index_array_type::u32;
					const u32 ia = rsx::get_address(rsx::method_registers.index_array_address(), rsx::method_registers.index_array_location()) + (first + v) * (u32_indices ? 4 : 2);
					index = u32_indices ? static_cast<u32>(vm::read32(ia)) : static_cast<u32>(vm::read16(ia));
				}
				const u32 a = base + (index + rsx::method_registers.vertex_data_base_index()) * pos.stride();
				if (!vm::check_addr(a, vm::page_readable, 16))
					break;
				fmt::append(verts, " (%g %g %g %g)", std::bit_cast<f32>(static_cast<u32>(vm::read32(a))), std::bit_cast<f32>(static_cast<u32>(vm::read32(a + 4))),
					std::bit_cast<f32>(static_cast<u32>(vm::read32(a + 8))), pos.size() >= 4 ? std::bit_cast<f32>(static_cast<u32>(vm::read32(a + 12))) : 1.f);
			}
			rsx_log.notice("VR why %016llx vertices:%s", why, verts);
		}
	}
	d.saved_env_info = m_vertex_env_buffer_info;
	d.saved_env_offset = m_vertex_env_dynamic_offset;
	// Sprites the game projected itself (ICO's flames): through the camera's eye transform.
	// Only into this frame's scene with depth test: the same program also draws ICO's pause
	// menu, which must stay in the HUD box (or as drawn). Programs listed "without_depth_test"
	// also take it with depth test off (Dante's Inferno's torch glows).
	const u32 vr_target = m_framebuffer_layout.color_addresses[0];
	const u64 vr_listed = d.render && !d.camera_draw ? vr_preprojected_program() : 0;
	const bool vr_in_scene = vr_target && std::find(m_vr_camera_targets.begin(), m_vr_camera_targets.end(), vr_target) != m_vr_camera_targets.end();
	const auto vr_untested = [&]()
	{
		const auto* profile = probe.profile();
		return profile && std::find(profile->screen_space_preprojected_untested.begin(), profile->screen_space_preprojected_untested.end(), vr_listed) !=
		                      profile->screen_space_preprojected_untested.end();
	};
	u64 vr_preprojected = vr_listed && !vr_hud && (rsx::method_registers.depth_test_enabled() || vr_untested()) && vr_in_scene ? vr_listed : 0;
	// Profile clip_space_scene_draws: every other depth-tested scene draw that is no camera draw
	// (its matrix folded with an object's, in another slot or layout, skinned from the whole bank)
	// takes the same eye transform, B^-1 * B_eye of the latest camera draw: the object part cancels.
	// Not post-processing (samples a colour render target) or the HUD.
	if (!vr_preprojected && !vr_listed && !vr_hud && d.render && !d.camera_draw && vr_in_scene && rsx::method_registers.depth_test_enabled())
	{
		if (probe.scene_draws_by_clip_space() && !(vr_sampled_textures() & vr_texture_colour_target))
		{
			vr_preprojected = rsx::vr::scene_draw_program;
		}
	}
	if (vr_listed)
	{
		// Diagnostic: each distinct way a listed program is drawn (first 16).
		static std::set<u64> s_seen;
		const u64 key = (u64{vr_target} << 3) | (rsx::method_registers.depth_test_enabled() ? 4 : 0) | (vr_hud ? 2 : 0) | (vr_in_scene ? 1 : 0);
		if (s_seen.size() < 16 && s_seen.insert(key).second)
		{
			rsx_log.notice("VR: pre-projected program %016llx into 0x%x: depth test %d, HUD %d, scene %d -> %s", vr_listed, vr_target,
				rsx::method_registers.depth_test_enabled(), vr_hud, vr_in_scene, vr_preprojected ? "eye transform" : vr_hud ? "HUD box" :
																															  "as drawn");
		}
	}
	d.preprojected = vr_preprojected;
	// Multiview: both eyes' contexts (vr_hud_vertex_env_pair sets mv_env_right); else the right eye reads the guest's.
	d.mv_env_right = m_vertex_env_dynamic_offset;
	d.hud_env = (vr_hud || vr_preprojected) && (d.mv ? vr_hud_vertex_env_pair(vr_preprojected) : vr_hud_vertex_env(-1.f, vr_preprojected));

	// A 2D screen that never clears its display buffer (Gran Turismo 5's arcade menu starts with a
	// full-screen background) leaves everything outside the HUD box stale: trails when the head
	// turns. Before the first boxed draw into a display buffer that no pass or full clear covered
	// this frame, clear the shown region in both eyes.
	d.clear_shown = false;
	// The colour target this draw writes: the first of the surface target's set (Killzone HD draws into
	// surface B, with A's write mask off; the framebuffer holds it as attachment 0 either way).
	const auto vr_rtt_indexes = rsx::utility::get_rtt_indexes(m_framebuffer_layout.target);
	const u8 vr_color_index = vr_rtt_indexes.empty() ? 0 : vr_rtt_indexes.front();
	if (d.render && !vr_rtt_indexes.empty() && m_framebuffer_layout.color_addresses[vr_color_index] && m_framebuffer_layout.color_write_enabled[vr_color_index] &&
		vr_display_buffer(*this, m_framebuffer_layout.color_addresses[vr_color_index], m_framebuffer_layout.width, m_framebuffer_layout.height))
	{
		const u32 target = m_framebuffer_layout.color_addresses[vr_color_index];
		if (std::find(m_vr_frame_covered.begin(), m_vr_frame_covered.end(), target) == m_vr_frame_covered.end())
		{
			// Boxed through the HUD-box constants (Killzone HD's menus): the game's own full-screen
			// background is boxed too, so outside the box nothing is written, and its glow pass (which
			// samples the whole display buffer and adds back onto it) fed on itself there up to white.
			const auto* profile = probe.profile();
			const bool box_after_shader = profile && profile->screen_space_hud_box_after_shader;
			const bool outside_box_clear = profile && profile->screen_space_clear_outside_box && !box_after_shader;
			const bool boxed_by_constants = outside_box_clear && !d.camera_draw && probe.box_mapped();
			if (probe.hud_env_requested() || boxed_by_constants)
			{
				d.clear_shown = true;
				m_vr_frame_covered.push_back(target);
			}
			else if ((vr_sampled_textures() & vr_texture_colour_target) || outside_box_clear)
			{
				// A pass, or (outside Gran Turismo 5's after-shader mode) any unboxed draw such as a scene
				// drawn straight into the display buffer: the buffer is this frame's, not to be cleared.
				m_vr_frame_covered.push_back(target);
			}
		}
	}
}

void VKGSRender::vr_clear_shown_region()
{
	const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
	VkClearRect rect{};
	rect.rect.extent = {std::min<u32>(static_cast<u32>(out.width * resolution_scaling_config.scale_factor()), m_draw_fbo->width()),
		std::min<u32>(static_cast<u32>(out.height * resolution_scaling_config.scale_factor_y()), m_draw_fbo->height())};
	rect.layerCount = 1;
	VkClearAttachment attachment{};
	attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	attachment.clearValue.color = {{0.f, 0.f, 0.f, 1.f}};
	vkCmdClearAttachments(*m_current_command_buffer, 1, &attachment, 1, &rect);
}

// emit_geometry(), after the pipeline bind.
void VKGSRender::vr_after_pipeline_bind(u32 sub_index, const vk::vertex_upload_info& upload_info)
{
	if (m_vr_draw.mv)
	{
		// Multiview: entry 2n serves view 0 and 2n+1 view 1 (the shader adds gl_ViewIndex). The right
		// eye's entry is written first, through the members update_vertex_env reads, so the push
		// constant left behind names the left one.
		const u64 left_xform = m_xform_constants_dynamic_offset;
		const u64 left_env = m_vertex_env_dynamic_offset;
		m_xform_constants_dynamic_offset = m_vr_mv_right_xform_offset;
		m_vertex_env_dynamic_offset = m_vr_draw.mv_env_right;
		update_vertex_env(sub_index * 2 + 1, upload_info);
		m_xform_constants_dynamic_offset = left_xform;
		m_vertex_env_dynamic_offset = left_env;
		update_vertex_env(sub_index * 2, upload_info);
	}
	else if (m_vr_draw.render)
	{
		// VR: the left eye's vertex env (a push constant) goes after the render pass change
		// above. Ending the left pass runs the right-eye batch (vkCmdExecuteCommands), which
		// leaves push constants undefined; pushed earlier, the left draw read another draw's
		// layout entry. Demon's Souls' soft particles (they sample the bound depth buffer, so
		// the render pass changes) drew with other draws' sprites or stretched in the left eye.
		update_vertex_env(sub_index * 2, upload_info);
	}

	// The right-eye secondary needs the dynamic state set again only when this draw changed it.
	m_vr_draw.dynamic_state_changed = (m_current_command_buffer->flags & vk::command_buffer::cb_reload_dynamic_state) != 0;
}

// emit_geometry(): per-draw stereo evidence and profile-generator sampling. This is the last
// point at which program, constants, framebuffer layout and subdraw state all correspond to
// the draw that is about to execute. No-op unless a capture is armed.
void VKGSRender::vr_capture_draw(u32 sub_index, const vk::vertex_upload_info& upload_info)
{
	const auto& draw_call = rsx::method_registers.current_draw_clause;

	// Dev: RPCS3_VR_GPU_CHECKPOINTS=1, a GPU checkpoint before each draw (VKMultiviewVR.cpp).
	if (vk::vr_gpu_checkpoints_enabled())
	{
		vk::vr_gpu_checkpoint_info ckpt{ .kind = 1, .frame = m_vr_ckpt_frame, .draw = m_vr_ckpt_draw++ };
		ckpt.fifo_pos = fifo_ctrl ? fifo_ctrl->get_pos() : 0;
		ckpt.vp_hash = current_vertex_program.data.empty() ? 0 : program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
		ckpt.fp_hash = current_fragment_program.ucode_hash;
		ckpt.target = m_framebuffer_layout.color_addresses[0] ? m_framebuffer_layout.color_addresses[0] : m_framebuffer_layout.zeta_address;
		ckpt.target_w = m_framebuffer_layout.width;
		ckpt.target_h = m_framebuffer_layout.height;
		vk::vr_gpu_checkpoint(*m_current_command_buffer, ckpt);
	}

	if (auto& inspector = rsx::vr::stereo_inspector::get(); inspector.capturing())
	{
		rsx::vr::draw_capture_input capture_in;
		capture_in.subdraw_index = sub_index;
		capture_in.vertex_draw_count = upload_info.vertex_draw_count;
		capture_in.indexed = !!upload_info.index_info;
		capture_in.pass_count = draw_call.pass_count();
		capture_in.vertex_program = &current_vertex_program;
		capture_in.fragment_program = &current_fragment_program;
		capture_in.framebuffer = &m_framebuffer_layout;

		if (m_vertex_prog)
		{
			capture_in.constant_ids = &m_vertex_prog->constant_ids;
			capture_in.has_indexed_constants = m_vertex_prog->has_indexed_constants;
			capture_in.vp_session_id = m_vertex_prog->id;
		}
		if (m_fragment_prog)
		{
			capture_in.fp_session_id = m_fragment_prog->id;
		}

		inspector.record_draw(capture_in);

		// The captured frame's shader sources, once per program, next to the capture: the
		// decompiled GLSL without turning on "Log shader programs" for the whole session.
		if (static const std::string dir = []() -> std::string
			{
				const char* v = std::getenv("RPCS3_STEREO_INSPECT");
				return v ? v : "";
			}();
			!dir.empty() && m_vertex_prog && m_fragment_prog)
		{
			const u64 vp_hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
			if (const std::string vp_file = dir + fmt::format("vp_%016llx.glsl", vp_hash); !fs::is_file(vp_file))
			{
				fs::write_file(vp_file, fs::rewrite, m_vertex_prog->shader.get_source());
			}
			if (const std::string fp_file = dir + fmt::format("fp_%016llx_%u.glsl", vp_hash, m_fragment_prog->id); !fs::is_file(fp_file))
			{
				fs::write_file(fp_file, fs::rewrite, m_fragment_prog->shader.get_source());
			}
		}
	}

	// VR profile generation (home menu): sample this draw's vertex constants.
	if (auto& generator = rsx::vr::profile_generator::get(); generator.sampling() && m_vertex_prog)
	{
		// Programs with indexed constants still name the slots they read directly (Demon's
		// Souls: its camera c[0..3] beside indexed bone matrices); the values come from the
		// guest registers, so they are right even while the shader interpreter draws.
		const bool full_bank = m_vertex_prog->constant_ids.empty();
		generator.record_draw(full_bank ? std::span<const u16>{} : std::span<const u16>(m_vertex_prog->constant_ids),
			m_vertex_prog->id, m_framebuffer_layout.width, m_framebuffer_layout.height, rsx::method_registers.depth_test_enabled(), vr_sampled_textures(),
			m_framebuffer_layout.color_addresses[0], program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program),
			m_vertex_prog->has_indexed_constants);
	}
}

// emit_geometry(): just before the left eye's draw commands.
void VKGSRender::vr_before_left_draw()
{
	auto& d = m_vr_draw;
	// HUD-box draws: the game's scissor follows the HUD into the box (per eye).
	d.box_scissor = d.render && !d.mv && vr_apply_box_scissor();
	if (d.mv)
	{
		// Multiview: one scissor per view, the HUD box's for a boxed draw (bind_viewport sets them on a
		// dynamic-state reload; a draw that changes them in between sets them here).
		static const bool s_box_scissor = []()
		{
			const char* v = std::getenv("RPCS3_VR_MV_SCISSOR");
			return !v || v[0] != '0';
		}();
		const bool box = m_vr_mv_box_scissor && s_box_scissor;
		const VkRect2D want[2] = {box ? m_vr_mv_scissor[0] : m_scissor, box ? m_vr_mv_scissor[1] : m_scissor};
		if (std::memcmp(want, m_vr_mv_bound_scissor, sizeof(want)) != 0)
		{
			vkCmdSetScissor(*m_current_command_buffer, 0, 2, want);
			std::memcpy(m_vr_mv_bound_scissor, want, sizeof(want));
		}
	}
	if (d.clear_shown)
	{
		vr_clear_shown_region();
	}
}

// emit_geometry(): after the left eye's draw commands. Binds the right eye's targets, constants
// and textures; returns true when the draw commands are to be emitted again for it.
bool VKGSRender::vr_begin_right_eye(u32 sub_index, const vk::vertex_upload_info& upload_info)
{
	auto& d = m_vr_draw;
	if (d.box_scissor)
	{
		vkCmdSetScissor(*m_current_command_buffer, 0, 1, &m_scissor);
	}
	if (!d.render || d.mv) // multiview: the one draw rendered both eyes
	{
		return false;
	}

	d.right_t0 = m_gpuprof_enabled > 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
	d.left_fbo = m_draw_fbo;
	m_vr_left_fbo_images.assign(m_fbo_images.begin(), m_fbo_images.end());
	m_draw_fbo = m_vr_right_draw_fbo;
	m_fbo_images.assign(m_vr_right_fbo_images.begin(), m_vr_right_fbo_images.end());

	if (d.batch)
	{
		bind_vr_eye_constants(1.f, d.guest_constants_source_offset, m_xform_constants_data_size);
		// On the primary: a barrier here ends the left pass, which runs the batch first.
		bind_texture_env(true);

		if (!vr_batch_begin(get_render_pass(), m_vr_right_draw_fbo))
		{
			vr_restore_left_eye();
			return false;
		}

		d.batch_primary = m_current_command_buffer;
		m_current_command_buffer = &m_vr_batch_cb;
		// A new secondary starts with its dynamic state reloaded (attach); within a batch only
		// a state change or the previous draw's HUD-box scissor needs it again. Reloading for
		// every draw cost ~2% of the RSX thread in Bayonetta.
		if (d.dynamic_state_changed || m_vr_batch_scissor_dirty)
		{
			m_vr_batch_cb.flags |= vk::command_buffer::cb_reload_dynamic_state;
		}
		if (d.hud_env)
		{
			vr_hud_vertex_env(1.f, d.preprojected);
		}
		update_vertex_env(sub_index * 2 + 1, upload_info);
		m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
		update_draw_state();
		m_vr_batch_scissor_dirty = vr_apply_box_scissor(); // then the next batched draw reloads the scissor
		if (d.clear_shown)
		{
			vr_clear_shown_region();
		}
		return true;
	}

	// Per-draw replay: the right eye in its own render pass.
	vk::end_renderpass(*m_current_command_buffer);
	if (d.suspend_query)
	{
		auto& query_data = m_occlusion_map[m_active_query_info->driver_handle];
		m_occlusion_query_manager->end_query(*m_current_command_buffer, query_data.indices.back());
		m_current_command_buffer->flags &= ~vk::command_buffer::cb_has_open_query;
	}

	bind_vr_eye_constants(1.f, d.guest_constants_source_offset, m_xform_constants_data_size);
	if (d.hud_env)
	{
		vr_hud_vertex_env(1.f, d.preprojected);
	}
	update_vertex_env(sub_index * 2 + 1, upload_info);
	bind_texture_env(true);
	m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
	update_draw_state();
	begin_render_pass();
	if (d.clear_shown)
	{
		vr_clear_shown_region();
	}
	d.right_box_scissor = vr_apply_box_scissor();
	return true;
}

void VKGSRender::vr_end_right_eye()
{
	auto& d = m_vr_draw;
	if (d.batch)
	{
		m_current_command_buffer = d.batch_primary;
		d.batch_primary = nullptr;
		m_vr_right_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);

		// A batch stays open only while the left pass is open.
		if (!vk::is_renderpass_open(*m_current_command_buffer))
		{
			vr_batch_execute();
		}

		vr_restore_left_eye();
		return;
	}

	if (d.right_box_scissor)
	{
		vkCmdSetScissor(*m_current_command_buffer, 0, 1, &m_scissor);
	}
	m_vr_right_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);
	vk::end_renderpass(*m_current_command_buffer);

	vr_restore_left_eye();

	if (d.suspend_query)
	{
		// Continue the same guest query after the host-only right-eye draw.
		// Result collection already sums every slot recorded for the query.
		m_occlusion_query_manager->begin_query(*m_current_command_buffer, d.query_continuation);
		auto& query_data = m_occlusion_map[m_active_query_info->driver_handle];
		query_data.indices.push_back(d.query_continuation);
		query_data.set_sync_command_buffer(m_current_command_buffer);
		m_current_command_buffer->flags |=
			(vk::command_buffer::cb_has_occlusion_task | vk::command_buffer::cb_has_open_query);
	}

	m_program->bind(*m_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
	update_draw_state();
	begin_render_pass();
}

// Back to the left eye's targets and the guest-authored constants allocation. Pipeline
// dependency processing for later RSX draws must never inherit either host eye's constants.
void VKGSRender::vr_restore_left_eye()
{
	auto& d = m_vr_draw;
	m_draw_fbo = d.left_fbo;
	m_fbo_images.assign(m_vr_left_fbo_images.begin(), m_vr_left_fbo_images.end());
	d.left_fbo = nullptr;
	m_vertex_constants_buffer_info = d.guest_constants_info;
	m_xform_constants_dynamic_offset = d.guest_constants_dynamic_offset;
	if (m_vs_binding_table->cbuf_location != umax)
	{
		m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex,
			m_vs_binding_table->cbuf_location);
	}
	bind_texture_env(false);
}

void VKGSRender::vr_end_draw()
{
	auto& d = m_vr_draw;
	if (d.right_t0 != std::chrono::steady_clock::time_point{})
	{
		m_gpuprof_right_ms += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - d.right_t0).count();
		d.right_t0 = {};
	}

	if (d.mv && d.eye_constants_bound)
	{
		// Back to the guest-authored constants allocation: later draws (or one the pair leaves alone)
		// must never inherit the eyes' transformed constants.
		m_vertex_constants_buffer_info = d.guest_constants_info;
		m_xform_constants_dynamic_offset = d.guest_constants_dynamic_offset;
		m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->cbuf_location);
	}

	if (d.hud_env)
	{
		// Later draws use the guest's own viewport again.
		m_vertex_env_buffer_info = d.saved_env_info;
		m_vertex_env_dynamic_offset = d.saved_env_offset;
		m_program->bind_uniform(m_vertex_env_buffer_info, vk::glsl::binding_set_index_vertex, m_vs_binding_table->context_buffer_location);
	}
}

// An off-aspect colour target that no camera draw reached (directly or through a texture) holds the same image in
// both eyes: a shadow map drawn from the light. The right eye samples the left eye's surface instead of its own copy,
// which the right-eye cache can evict and rebuild out of date (SEGA Rally Revo: the 912x912 shadow map's right copy
// went missing for frames and came back with older casters, so shadows appeared and vanished in one eye at a time).
// View-shaped targets (a HUD layer, a pass at the view's shape) and depth surfaces keep their per-eye copies.
bool VKGSRender::vr_eye_invariant_target(const vk::render_target* rtt) const
{
	// Dev: RPCS3_VR_NO_INVARIANT=1 keeps the per-eye copies (A/B).
	static const bool s_off = std::getenv("RPCS3_VR_NO_INVARIANT") != nullptr;
	const auto* profile = rsx::vr::camera_probe::get().profile();
	if (s_off || !profile || profile->offaspect_player_views || rtt->vr_has_3d || !(rtt->aspect() & VK_IMAGE_ASPECT_COLOR_BIT))
	{
		return false;
	}
	const size2u eye = g_fxo->get<rsx::avconf>().video_frame_size();
	const f32 output_aspect = eye.height ? static_cast<f32>(eye.width) / eye.height : 0.f;
	return !profile->is_view_target(rtt->get_surface_width<rsx::surface_metrics::pixels>(), rtt->get_surface_height<rsx::surface_metrics::pixels>(), output_aspect);
}

// bind_texture_env(): the view a fragment sampler reads. For the left eye this is the ordinary
// cache's image_handle. Gate 5 render-target feedback: the ordinary texture cache resolves
// guest addresses to the authoritative left-eye surface. During the right replay, substitute
// the isomorphic host-only surface at the same guest address so post-processing does not
// collapse both eyes back to the left intermediate. A null result makes the caller build the
// (left-eye) temporary subresource as upstream does.
vk::image_view* VKGSRender::vr_fragment_texture_view(u32 i, vk::texture_cache::sampled_image_descriptor* sampler_state, bool vr_right_eye)
{
	if (!vr_right_eye)
	{
		return sampler_state->image_handle;
	}

	vk::image_view* view = nullptr;

	// The left eye either samples the surface directly (image_handle is a view of
	// the render target), or through a deferred copy of part of it, possibly
	// with a format conversion (no image_handle; NFS Most Wanted). The copy is
	// rebuilt from the matching right-eye surfaces; a raw right-eye view would
	// read the wrong region or format. A cached copy that is not a render
	// target stays on the left eye's image.
	vk::render_target* left_rtt = nullptr;
	if (sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage && sampler_state->image_handle)
	{
		left_rtt = dynamic_cast<vk::render_target*>(sampler_state->image_handle->image());
	}

	// A cube map gathered from off-aspect targets (its faces: square 90-degree cameras, which
	// keep the game camera in both eyes) is the same image in both eyes. The right eye then
	// samples the left eye's copy, which the texture cache keeps, instead of rebuilding it on
	// every draw (Ridge Racer 7: 36 cube-map rebuilds a frame on the start grid, one per car
	// part reflecting the environment). Other off-aspect targets can hold per-eye views
	// (Ridge Racer 7's road reflection tiles, drawn with the player's camera).
	bool shared_copy = false;
	if (sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage && !sampler_state->image_handle)
	{
		const auto& desc = sampler_state->external_subresource_desc;
		const auto* profile = rsx::vr::camera_probe::get().profile();
		const bool cubemap = desc.op == rsx::deferred_request_command::cubemap_gather || desc.op == rsx::deferred_request_command::cubemap_unwrap;
		// An atlas gathered from off-aspect targets (Gran Turismo 5's shadow maps: one gather per lit
		// draw) is the same in both eyes unless the profile says off-aspect targets hold per-eye views.
		const bool atlas = desc.op == rsx::deferred_request_command::atlas_gather && profile && !profile->offaspect_player_views;
		// Likewise a mip chain gathered from off-aspect levels (MotorStorm: Pacific Rift's 2048x2048 environment map
		// and its seven downsampled levels: up to 25 right-eye rebuilds a frame in menus).
		const bool mipmaps = desc.op == rsx::deferred_request_command::mipmap_gather && profile && !profile->offaspect_player_views;
		// A copy of a few texels (GT5 binds a 3x3 dummy texture at guest address 0, the corner of the
		// final render target, for unused samplers: 146 right-eye rebuilds a frame) is no view of the scene.
		const bool dummy = (desc.op == rsx::deferred_request_command::copy_image_static || desc.op == rsx::deferred_request_command::copy_image_dynamic) &&
		                   desc.width <= 4 && desc.height <= 4;
		if (profile && dummy)
		{
			shared_copy = desc.external_handle != nullptr;
		}
		else if (const auto invariant_source = [&](vk::image* src)
					 {
						 const auto* rtt = dynamic_cast<const vk::render_target*>(src);
						 return rtt && vr_eye_invariant_target(rtt);
					 };
				 profile && (desc.external_handle || !desc.sections_to_copy.empty()) &&
				 (!desc.external_handle || invariant_source(desc.external_handle)) &&
				 std::all_of(desc.sections_to_copy.begin(), desc.sections_to_copy.end(), [&](const auto& section)
					 {
						 return !section.src || invariant_source(section.src);
					 }))
		{
			// A copy (format conversion) of eye-invariant targets: SEGA Rally Revo reads its shadow map this way.
			shared_copy = true;
		}
		else if (profile && (cubemap || atlas || mipmaps))
		{
			const size2u eye = g_fxo->get<rsx::avconf>().video_frame_size();
			const f32 output_aspect = eye.height ? static_cast<f32>(eye.width) / eye.height : 0.f;
			const auto invariant = [&](vk::image* src)
			{
				auto* rtt = dynamic_cast<vk::render_target*>(src);
				return rtt && !profile->is_view_target(rtt->get_surface_width<rsx::surface_metrics::pixels>(),
								  rtt->get_surface_height<rsx::surface_metrics::pixels>(), output_aspect);
			};
			shared_copy = desc.external_handle || !desc.sections_to_copy.empty();
			if (desc.external_handle)
			{
				shared_copy = shared_copy && invariant(desc.external_handle);
			}
			for (const auto& section : desc.sections_to_copy)
			{
				shared_copy = shared_copy && (!section.src || invariant(section.src));
			}
		}
	}

	if (!shared_copy && sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage && !sampler_state->image_handle)
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
			m_gpuprof_right_copies++;
			if (m_gpuprof_enabled > 0)
			{
				const auto* first = desc.external_handle ? desc.external_handle : (desc.sections_to_copy.empty() ? nullptr : desc.sections_to_copy.front().src);
				const auto* first_rtt = dynamic_cast<const vk::render_target*>(first);
				m_gpuprof_right_copy_kinds[fmt::format("op %d %ux%u from 0x%x %ux%u (%u sections)", static_cast<int>(desc.op), desc.width, desc.height,
					first_rtt ? first_rtt->base_addr : 0u, first ? first->width() : 0u, first ? first->height() : 0u, static_cast<u32>(desc.sections_to_copy.size()))]++;
			}
			view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, desc);
		}
	}
	else if (left_rtt && vr_eye_invariant_target(left_rtt))
	{
		// The left eye's image serves both eyes (see vr_eye_invariant_target).
	}
	else if (left_rtt)
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
			const auto aspect = sampler_state->image_handle ? sampler_state->image_handle->info.subresourceRange.aspectMask : right_image->aspect();
			view = right_image->get_view(rsx::method_registers.fragment_textures[i].decoded_remap(), aspect);
		}
	}

	// Diagnostic: a right-eye sample left on the shared (left-eye) image although the
	// right-eye store holds a surface in the sampled range. Logged once per address.
	if (!view && !shared_copy)
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
	return view;
}

// bind_texture_env(), vertex textures of the right eye: the right-eye surface where the left eye reads a render target.
vk::image_view* VKGSRender::vr_vertex_texture_view(u32 i, vk::texture_cache::sampled_image_descriptor* sampler_state, vk::image_view* image_ptr)
{
	if (sampler_state->upload_context == rsx::texture_upload_context::framebuffer_storage)
	{
		if (auto* right_surface = m_vr_right_rtts.get_surface_at(sampler_state->ref_address))
		{
			right_surface->read_barrier(*m_current_command_buffer);
			auto* right_image = right_surface->get_surface(rsx::surface_access::shader_read);
			const auto aspect = image_ptr ? image_ptr->info.subresourceRange.aspectMask : right_image->aspect();
			image_ptr = right_image->get_view(rsx::method_registers.vertex_textures[i].decoded_remap(), aspect);
		}
	}
	return image_ptr;
}

// load_texture_env(): profile texture_redirects, read the render target instead of its main-memory copy (both eyes).
VKGSRender::vr_texture_redirect VKGSRender::vr_redirect_texture(u32 i, const rsx::fragment_texture& tex)
{
	vr_texture_redirect redirect;
	if (const auto* profile = rsx::vr::camera_probe::get().profile(); profile && !profile->texture_redirects.empty() &&
																	  rsx::vr::camera_probe::get().render_enabled())
	{
		const u32 address = rsx::get_address(tex.offset(), tex.location());
		for (const auto& [from, to] : profile->texture_redirects)
		{
			if (address != from)
				continue;
			auto& regs = rsx::method_registers.registers;
			redirect.saved[0] = regs[NV4097_SET_TEXTURE_OFFSET + i * 8];
			redirect.saved[1] = regs[NV4097_SET_TEXTURE_FORMAT + i * 8];
			const u32 target = vr_redirect_target(to, 0);
			if (!target)
				break;
			regs[NV4097_SET_TEXTURE_OFFSET + i * 8] = target - rsx::constants::local_mem_base;
			regs[NV4097_SET_TEXTURE_FORMAT + i * 8] = (redirect.saved[1] & ~3u) | (CELL_GCM_LOCATION_LOCAL + 1);
			redirect.active = true;
			m_textures_dirty[i] = true; // re-checked every draw: the cached sampler names the target, not `from`
			break;
		}
	}
	return redirect;
}

void VKGSRender::vr_restore_texture(u32 i, const vr_texture_redirect& redirect)
{
	if (redirect.active)
	{
		rsx::method_registers.registers[NV4097_SET_TEXTURE_OFFSET + i * 8] = redirect.saved[0];
		rsx::method_registers.registers[NV4097_SET_TEXTURE_FORMAT + i * 8] = redirect.saved[1];
	}

	// Multiview: an eye-invariant target (see vr_eye_invariant_target) is sampled at the eye's layer like any other.
	// Its layer 1 can fall behind layer 0 (SEGA Rally Revo's shadow map: shadows in one eye only, coming and going), so
	// after each write it takes a copy of layer 0 before it is read.
	auto* sampler_state = static_cast<vk::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
	if (!m_vr_multiview || !sampler_state || sampler_state->upload_context != rsx::texture_upload_context::framebuffer_storage ||
		!rsx::vr::camera_probe::get().render_enabled())
	{
		return;
	}
	const auto sync = [&](vk::image* image)
	{
		auto* rtt = dynamic_cast<vk::render_target*>(image);
		if (!rtt || !rtt->stereo_layers || rtt->layers() < 2 || rtt->samples() > 1 || !vr_eye_invariant_target(rtt))
		{
			return;
		}
		auto [it, added] = m_vr_layer_synced.try_emplace(rtt, 0);
		if (!added && it->second == rtt->last_use_tag)
		{
			return;
		}
		it->second = rtt->last_use_tag;
		if (vk::is_renderpass_open(*m_current_command_buffer))
		{
			vk::end_renderpass(*m_current_command_buffer);
		}
		vk::vr_copy_left_to_right_layer(*m_current_command_buffer, rtt);
	};
	if (sampler_state->image_handle)
	{
		sync(sampler_state->image_handle->image());
		return;
	}
	const auto& desc = sampler_state->external_subresource_desc;
	if (desc.external_handle)
	{
		sync(desc.external_handle);
	}
	for (const auto& section : desc.sections_to_copy)
	{
		if (section.src)
		{
			sync(section.src);
		}
	}
}

// end(), right after the pipeline is analysed (before textures, program, vertices and the VR setup, so a skipped draw
// costs little): profile car_draw_limit draws the cars by distance tier by tier: full detail, only their first n draws,
// without their small draws (min_vertices), or not at all (Gran Turismo 5; RPCS3_VR_CAR_LIMIT=0 turns it off).
// Draws are grouped by transform (c[0..3]) on the 1280x720 scene target; a group's depth is the w row's translation
// (c[3].w). A car is a group of 60+ draws with 10+ draws of the car body programs; the previous frame's cars, sorted
// by depth, give each car its rank. Car-part programs (wheels, driver, glass: their own transforms) are learned: seen
// 30+ times within 3 m of a car and almost never elsewhere. Only body and car-part draws are ever skipped; a cap counts
// each transform group's own draws in its first pass, so small part groups stay under it, and a car's later pass
// (glass and other see-through parts, drawn after all the cars' bodies) is kept.
// end() (through vr_skip_far_cars) and clear_surface(): profile shared_frame_targets. Targets of a rule's size take
// turns by address: the n-th address seen is refreshed on frames where n % frames == frame % frames and keeps its
// content otherwise. Dev: RPCS3_VR_SHARED_TARGETS=0 draws everything every frame.
bool VKGSRender::vr_shared_target_skipped()
{
	const auto* profile = rsx::vr::camera_probe::get().render_enabled() ? rsx::vr::camera_probe::get().profile() : nullptr;
	static const bool s_off = []
	{
		const char* v = std::getenv("RPCS3_VR_SHARED_TARGETS");
		return v && v[0] == '0';
	}();
	if (s_off || !profile || profile->shared_frame_targets.empty())
	{
		return false;
	}
	const u32 address = m_framebuffer_layout.color_addresses[0];
	if (!address)
	{
		return false;
	}
	for (const auto& rule : profile->shared_frame_targets)
	{
		if (rule.width != m_framebuffer_layout.width || rule.height != m_framebuffer_layout.height || rule.frames < 2 ||
			!rsx::vr::profile_option_enabled(rule.option))
		{
			continue;
		}
		// Addresses in order of first sighting (a target redrawn at a new address restarts the table).
		static std::vector<u32> s_addresses;
		static const rsx::vr::title_profile* s_profile = nullptr;
		if (s_profile != profile)
		{
			s_profile = profile;
			s_addresses.clear();
		}
		auto it = std::find(s_addresses.begin(), s_addresses.end(), address);
		if (it == s_addresses.end())
		{
			if (s_addresses.size() >= 16)
			{
				s_addresses.clear();
			}
			s_addresses.push_back(address);
			it = s_addresses.end() - 1;
		}
		const u32 index = static_cast<u32>(it - s_addresses.begin());
		return (index % rule.frames) != (static_cast<u32>(int_flip_index) % rule.frames);
	}
	return false;
}

bool VKGSRender::vr_skip_far_cars()
{
	if (vr_shared_target_skipped())
	{
		execute_nop_draw();
		rsx::thread::end();
		return true;
	}
	const auto* profile = rsx::vr::camera_probe::get().render_enabled() ? rsx::vr::camera_probe::get().profile() : nullptr;
	// RPCS3_VR_CAR_LIMIT=0: off; -1: observe only (the car log, nothing skipped).
	static const s32 s_env = []
	{
		const char* v = std::getenv("RPCS3_VR_CAR_LIMIT");
		return v ? std::atoi(v) : 1;
	}();
	const bool s_off = s_env == 0;
	if (s_off || !profile || profile->car_draw_tiers.empty() || profile->car_body_programs.empty() ||
		!rsx::vr::profile_option_enabled(profile->car_draw_option) ||
		m_framebuffer_layout.width != 1280 || m_framebuffer_layout.height != 720)
	{
		return false;
	}

	constexpr f32 near_car = 3.f;
	const auto is_body = [&](u64 program)
	{
		return std::find(profile->car_body_programs.begin(), profile->car_body_programs.end(), program) != profile->car_body_programs.end();
	};

	struct group
	{
		u32 draws = 0, body_draws = 0;
		u32 last_draw = 0, pass = 0, pass_draws = 0; // a car's later pass (glass, see-through parts) comes after the other cars
		f32 depth = 0.f;
		std::vector<u64> programs;
		std::vector<u32> body_vertices; // first-pass draws' vertex counts (for keep_percent)
	};
	struct car
	{
		f32 depth = 0.f;
		std::vector<u32> cutoffs; // per tier: keep_percent's smallest kept vertex count (0: none)
	};
	struct program_stats
	{
		u32 near_count = 0, far_count = 0;
	};
	static std::unordered_map<u64, group> s_groups;
	static std::unordered_map<u64, program_stats> s_programs;
	static std::vector<f32> s_cars; // previous frame: car depths, nearest first
	static std::vector<car> s_car_info; // previous frame: per car, nearest first
	static u64 s_frame = umax, s_skipped = 0, s_frames = 0;
	static u32 s_draw = 0;

	if (s_frame != int_flip_index)
	{
		s_draw = 0;
		std::vector<f32> cars;
		std::vector<car> car_info;
		for (auto& [key, g] : s_groups)
		{
			if (g.draws >= 60 && g.body_draws >= 10 && g.depth > 0.f)
			{
				cars.push_back(g.depth);
				car& info = car_info.emplace_back();
				info.depth = g.depth;
				std::sort(g.body_vertices.begin(), g.body_vertices.end(), std::greater<>());
				u64 total = 0;
				for (const u32 v : g.body_vertices)
				{
					total += v;
				}
				for (const auto& t : profile->car_draw_tiers)
				{
					u32 cutoff = 0;
					if (t.keep_percent && t.keep_percent < 100)
					{
						u64 kept = 0;
						for (const u32 v : g.body_vertices)
						{
							kept += v;
							cutoff = v;
							if (kept * 100 >= total * t.keep_percent)
							{
								break;
							}
						}
					}
					info.cutoffs.push_back(cutoff);
				}
			}
		}
		std::sort(cars.begin(), cars.end());
		std::sort(car_info.begin(), car_info.end(), [](const car& a, const car& b) { return a.depth < b.depth; });
		s_car_info = std::move(car_info);
		for (const auto& [key, g] : s_groups)
		{
			const bool close = std::any_of(cars.begin(), cars.end(), [&](f32 d) { return std::abs(d - g.depth) < near_car; });
			for (const u64 program : g.programs)
			{
				auto& st = s_programs[program];
				(close ? st.near_count : st.far_count) += 1;
			}
		}
		s_cars = std::move(cars);
		s_groups.clear();
		s_frame = int_flip_index;
		if (++s_frames % (s_env < 0 ? 15 : 300) == 0)
		{
			u32 parts = 0;
			for (const auto& [prog, st] : s_programs)
			{
				parts += st.near_count >= 30 && st.far_count * 20 <= st.near_count;
			}
			std::string depths;
			for (const f32 d : s_cars)
			{
				fmt::append(depths, " %.0f", d);
			}
			rsx_log.notice("VR car tiers: cars at%s m, %u car-part programs, %llu draws skipped over 300 frames", depths, parts, s_skipped);
			s_skipped = 0;
		}
	}

	const auto& c = rsx::method_registers.transform_constants;
	u64 key = 0xcbf29ce484222325ull;
	for (u32 r = 0; r < 4; r++)
	{
		for (u32 i = 0; i < 4; i++)
		{
			key = (key ^ c[r][i]) * 0x100000001b3ull;
		}
	}
	const f32 depth = std::bit_cast<f32>(c[3][3]);
	const u64 program = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
	auto& g = s_groups[key];
	if (g.draws && s_draw - g.last_draw > 20)
	{
		g.pass++;
		g.pass_draws = 0;
	}
	g.last_draw = s_draw++;
	g.pass_draws++;
	g.draws++;
	const u32 elements = rsx::method_registers.current_draw_clause.vr_total_elements(); // 0: unknown (inline array)
	if (!g.pass && elements)
	{
		g.body_vertices.push_back(elements);
	}
	g.depth = depth;
	g.body_draws += is_body(program);
	if (g.programs.size() < 64 && std::find(g.programs.begin(), g.programs.end(), program) == g.programs.end())
	{
		g.programs.push_back(program);
	}

	if (depth <= 0.f || s_cars.empty() || s_env < 0)
	{
		return false;
	}
	bool car_draw = is_body(program);
	if (!car_draw)
	{
		const auto st = s_programs.find(program);
		car_draw = st != s_programs.end() && st->second.near_count >= 30 && st->second.far_count * 20 <= st->second.near_count &&
			std::any_of(s_cars.begin(), s_cars.end(), [&](f32 d) { return std::abs(d - depth) < near_car; });
	}
	if (!car_draw)
	{
		return false;
	}

	// The car's rank: the previous frame's cars nearer than this one (half a car length of slack).
	const u32 rank = static_cast<u32>(std::count_if(s_cars.begin(), s_cars.end(), [&](f32 d) { return d < depth - near_car / 2.f; }));
	const auto* tier = &profile->car_draw_tiers.back();
	const bool by_distance = std::any_of(profile->car_draw_tiers.begin(), profile->car_draw_tiers.end(), [](const auto& t) { return t.min_distance > 0.f; });
	if (by_distance)
	{
		// The nearest full_nearest cars stay complete wherever they are (the car you are chasing).
		if (rank < profile->car_full_nearest)
		{
			return false;
		}
		// Tiers by the car's view depth: the last tier whose min_distance the car has reached (a tier without one
		// covers the nearest cars).
		tier = nullptr;
		for (const auto& t : profile->car_draw_tiers)
		{
			if (depth >= t.min_distance)
			{
				tier = &t;
			}
		}
		if (!tier)
		{
			return false;
		}
	}
	else
	{
		u32 first = 0;
		for (const auto& t : profile->car_draw_tiers)
		{
			if (!t.cars || rank < first + t.cars)
			{
				tier = &t;
				break;
			}
			first += t.cars;
		}
	}
	u32 cutoff = 0;
	if (tier->keep_percent)
	{
		const usz tier_index = static_cast<usz>(tier - profile->car_draw_tiers.data());
		const car* nearest = nullptr;
		for (const car& info : s_car_info)
		{
			if (std::abs(info.depth - depth) < near_car && (!nearest || std::abs(info.depth - depth) < std::abs(nearest->depth - depth)))
			{
				nearest = &info;
			}
		}
		cutoff = nearest && tier_index < nearest->cutoffs.size() ? nearest->cutoffs[tier_index] : 0u;
	}
	// The tier limits the car's first (body) pass only: its later pass holds the glass and other see-through parts.
	if (g.pass > 0)
	{
		return false;
	}
	const bool too_many = tier->draws == 0 || (tier->draws > 0 && g.pass_draws > static_cast<u32>(tier->draws));
	const bool too_small = elements && (elements < tier->min_vertices || elements < cutoff);
	if (!too_many && !too_small)
	{
		return false;
	}
	s_skipped++;
	execute_nop_draw();
	rsx::thread::end();
	return true;
}

// end(): draws left out. Profile hidden_draws (effects switched off while VR is enabled: vertex
// program + texture 0 size), RPCS3_VR_RTDUMP prog=<hash> (dump before this program draws), and the
// dev probe key hide=<hash>[@<target>]. Returns true after finishing the draw as a no-op.
bool VKGSRender::vr_skip_draw()
{
	if (m_gpuprof_target && m_gpuprof_enabled > 0 && m_framebuffer_layout.color_addresses[0] == m_gpuprof_target)
	{
		// Dev GPU profiler: per-draw segment after the draw's textures loaded ("fmt" 0x20000 + the draw's index).
		gpuprof_mark({m_gpuprof_target, m_framebuffer_layout.width, m_framebuffer_layout.height, 0x20000u + m_gpuprof_draw});
		static u32 s_logged = 0;
		if (s_logged++ < 8)
		{
			std::string t;
			for (u32 i = 0; i < 16; ++i)
			{
				if (const auto& tex = rsx::method_registers.fragment_textures[i]; tex.enabled())
					t += fmt::format(" t%u=%08x %ux%u fmt 0x%x", i, rsx::get_address(tex.offset(), tex.location()), tex.width(), tex.height(), tex.format());
			}
			rsx_log.notice("GPU profile: draw %u into 0x%x vp %016llx:%s", m_gpuprof_draw, m_gpuprof_target,
				program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program), t);
		}
	}

	const auto hash = [&]()
	{
		return program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
	};

	if (const auto* profile = g_cfg.video.vr.enabled ? rsx::vr::camera_probe::get().profile() : nullptr; profile && !profile->hidden_draws.empty())
	{
		const auto& tex = rsx::method_registers.fragment_textures[0];
		const u16 width = tex.enabled() ? tex.width() : 0;
		const u16 height = tex.enabled() ? tex.height() : 0;
		if (std::any_of(profile->hidden_draws.begin(), profile->hidden_draws.end(), [&](const auto& d)
				{
					return d.width == width && d.height == height;
				}) &&
			std::any_of(profile->hidden_draws.begin(), profile->hidden_draws.end(), [&, h = hash()](const auto& d)
				{
					return d.program == h && d.width == width && d.height == height;
				}))
		{
			execute_nop_draw();
			rsx::thread::end();
			return true;
		}
	}

	// Profile screen_frame_draws: this frame goes on the fixed screen (see vr_update_view). The hash is the current
	// program's: m_vertex_prog (vr_vertex_program_hash's cache key) is still the previous draw's here.
	if (const auto* profile = rsx::vr::camera_probe::get().render_enabled() ? rsx::vr::camera_probe::get().profile() : nullptr;
		profile && !profile->screen_space_screen_frame_draws.empty())
	{
		const auto& tex = rsx::method_registers.fragment_textures[0];
		const u16 width = tex.enabled() ? tex.width() : 0;
		const u16 height = tex.enabled() ? tex.height() : 0;
		const auto& list = profile->screen_space_screen_frame_draws;
		if (std::any_of(list.begin(), list.end(), [&](const auto& d)
				{
					return d.width == width && d.height == height;
				}))
		{
			const u64 h = hash();
			m_vr_screen_frame_draws += std::any_of(list.begin(), list.end(), [&](const auto& d)
				{
					return d.program == h && d.width == width && d.height == height;
				});
		}
	}

	// Profile frame_rate_draws: the frame-rate words follow the VR rate while these draws are made.
	if (const auto* profile = rsx::vr::camera_probe::get().profile(); profile && !profile->frame_rate_draws.empty())
	{
		const auto& tex = rsx::method_registers.fragment_textures[0];
		const u16 width = tex.enabled() ? tex.width() : 0;
		const u16 height = tex.enabled() ? tex.height() : 0;
		const auto& list = profile->frame_rate_draws;
		if (std::any_of(list.begin(), list.end(), [&](const auto& d) { return d.width == width && d.height == height; }))
		{
			const u64 h = hash();
			for (usz i = 0; i < list.size(); ++i)
			{
				if (list[i].program == h && list[i].width == width && list[i].height == height)
				{
					rsx::vr::note_frame_rate_draw(static_cast<u32>(i));
				}
			}
		}
	}

	// Profile reduced_scale_frames: a frame with one of these draws renders at the profile's lower Resolution Scale.
	if (const auto* profile = rsx::vr::camera_probe::get().render_enabled() ? rsx::vr::camera_probe::get().profile() : nullptr;
		profile && !profile->reduced_scale_draws.empty())
	{
		const auto& tex = rsx::method_registers.fragment_textures[0];
		const u16 width = tex.enabled() ? tex.width() : 0;
		const u16 height = tex.enabled() ? tex.height() : 0;
		const auto& list = profile->reduced_scale_draws;
		if (std::any_of(list.begin(), list.end(), [&](const auto& d)
				{
					return d.width == width && d.height == height;
				}))
		{
			const u64 h = hash();
			m_vr_reduced_scale_draws += std::any_of(list.begin(), list.end(), [&](const auto& d)
				{
					return d.program == h && d.width == width && d.height == height;
				});
		}
	}

	// RPCS3_VR_RTDUMP with prog=<hash>: dump the requested surfaces of both eyes just before this program draws.
	if (m_vr_rtdump_program && m_vr_rtdump_armed && hash() == m_vr_rtdump_program &&
		!(m_vr_rtdump_skip && m_vr_rtdump_skip--))
	{
		m_vr_rtdump_program = 0;
		vr_rtdump(m_vr_rtdump_addresses, fmt::format("before program %x", hash()));
	}

	// Probe hide=<hash>[@<target>]: skip this vertex program's draws (development: finding which program draws an artefact).
	if (const auto& hidden = rsx::vr::camera_probe::get().hidden_programs(); !hidden.empty())
	{
		const u64 h = hash();
		const u32 target = m_framebuffer_layout.color_addresses[0];
		if (std::any_of(hidden.begin(), hidden.end(), [&](const auto& entry)
				{
					return entry.first == h && (!entry.second || entry.second == target);
				}))
		{
			execute_nop_draw();
			rsx::thread::end();
			return true;
		}
	}

	return false;
}

// end(), after the program environment is loaded.
void VKGSRender::vr_before_draw_setup()
{
	if (vk::xr::is_running() && m_vr_applied_pose && rsx::method_registers.blend_enabled() && vr_reprojects_older_frames())
	{
		vr_realign_blend_targets();
	}

	// Advance the logical RSX draw ordinal for the stereo inspector. One logical clause can
	// produce several emitted subdraws, so the ordinal is assigned here and the subdraw index
	// is recorded separately.
	rsx::vr::stereo_inspector::get().begin_draw_clause();
}

// flip(): the Resolution Scale to render at: the configured one, or the profile's reduced_scale_frames scale.
// vr_update_view(), once the headset's eye shape is known: while the surface caches are still empty (before the game's
// first frame) the per-axis scale applies at once. Applied by flip() at the end of the first frame instead, every surface
// of that frame was rebuilt at the new size, and ICO / Shadow of the Colossus's GPU stalled there in about one boot in four
// (its SPUs read back blits of the scene at that moment): a hung game. Later changes still go through flip().
void VKGSRender::vr_apply_eye_shape_early()
{
	if (m_rtts.m_active_memory_used || m_vr_right_rtts.m_active_memory_used)
	{
		return;
	}
	const u16 percent = vr_resolution_scale(static_cast<u16>(g_cfg.video.resolution_scale_percent));
	const rsx::surface_scaling_config_t config =
	{
		.scale_percent = rsx::vr::eye_shape_percent_x(percent),
		.min_scalable_dimension = rsx::vr::min_scalable_dimension(static_cast<u16>(g_cfg.video.min_scalable_dimension)),
		.scale_percent_y = rsx::vr::eye_shape_percent_y(percent),
	};
	if (config != resolution_scaling_config)
	{
		rsx_log.notice("VR: headset eye shape applied before the first frame: %u%% x %u%%", config.scale_percent, config.percent_y());
		resolution_scaling_config = config;
	}
}

u16 VKGSRender::vr_resolution_scale(u16 configured_percent) const
{
	return rsx::vr::effective_resolution_scale(configured_percent);
}

// VKGSRender(): Gate 6: OpenXR must be initialised before the Vulkan instance so the runtime's
// required extensions can be enabled on it. Only for the armed stereo title; everything else
// never touches the OpenXR runtime.
void VKGSRender::vr_init_before_instance()
{
	if (rsx::vr::camera_probe::get().render_enabled())
	{
		vk::xr::prepare();
	}
}

// VKGSRender(): the headset must be driven by the GPU the OpenXR runtime names.
std::string VKGSRender::vr_select_adapter(std::vector<vk::physical_device>& gpus, const std::string& adapter_name)
{
	if (const VkPhysicalDevice xr_gpu = vk::xr::get_physical_device(m_instance.handle()))
	{
		for (auto& gpu : gpus)
		{
			if (static_cast<VkPhysicalDevice>(gpu) == xr_gpu)
			{
				return gpu.get_name();
			}
		}
	}
	return adapter_name;
}

// VKGSRender(): RPCS3 submits graphics work to queue 0 of the graphics family; the OpenXR
// runtime gets a spare queue of that family when there is one.
void VKGSRender::vr_init_after_device()
{
	if (vk::xr::is_prepared())
	{
		const VkQueue xr_queue = m_device->get_xr_queue();
		vk::xr::create_session(m_instance.handle(), m_device->gpu(), *m_device,
			xr_queue ? xr_queue : m_device->get_graphics_queue(), m_device->get_graphics_queue_family(),
			xr_queue ? m_device->get_xr_queue_index() : 0, m_device->get_graphics_queue());
	}
}

// VKGSRender(): Gate 6: batched right-eye draws (RPCS3_VR_BATCH=0 keeps the per-draw replay).
void VKGSRender::vr_init_batching()
{
	// Multiview stereo: available when the device has it (and RPCS3_VR_MULTIVIEW is not 0); switched on with
	// the stereo renderer (vr_update_multiview_mode). Both render-pass hooks belong to this renderer.
	static const bool s_env_off = []()
	{
		const char* v = std::getenv("RPCS3_VR_MULTIVIEW");
		return v && v[0] == '0';
	}();
	m_vr_multiview_supported = m_device->get_multiview_support() && !s_env_off;
	rsx_log.notice("VR: multiview stereo %s (device multiview %d, per-view scissor %d)", m_vr_multiview_supported ? "available" : "unavailable: two-draw stereo",
		m_device->get_multiview_support(), m_device->get_shader_viewport_index_layer_support());
	s_vr_batch_owner = this;
	vk::g_end_renderpass_hook = &VKGSRender::vr_on_end_renderpass;
	vk::g_before_end_renderpass_hook = &VKGSRender::vr_mv_on_before_end_renderpass;

	if (!rsx::vr::camera_probe::get().render_enabled())
	{
		return;
	}

	// Not transient: the secondaries are reused every frame, and a transient pool let the driver
	// release and reallocate their memory on each begin (GlobalAlloc in every vr_batch_begin).
	m_vr_batching = true;
	if (const char* batch_env = std::getenv("RPCS3_VR_BATCH"))
	{
		m_vr_batching = std::string_view(batch_env) != "0";
	}
	rsx_log.success("Gate 6: right-eye draws %s.", m_vr_batching ? "batched per left render pass" : "replayed per draw");
}

// VKGSRender(): the headset's overlay layer has its own renderer instance and image cache; without
// its own init the built-in images (controller button icons) were missing in the headset's menus.
void VKGSRender::vr_init_overlays(vk::command_buffer& cmd)
{
	vk::get_overlay_pass<vk::ui_overlay_renderer_xr>()->init(cmd, m_texture_upload_buffer_ring_info);
}

// ~VKGSRender(): the OpenXR session references the device and its frame thread submits to
// the graphics queue: stop it before the device-wide wait and any teardown.
void VKGSRender::vr_destroy_before_wait()
{
	vk::xr::destroy();
}

// ~VKGSRender(): everything VR owns on the device (called before the surface cache is destroyed).
void VKGSRender::vr_destroy_resources()
{
	if (m_vr_right_draw_fbo)
	{
		m_vr_right_draw_fbo->release();
		m_vr_right_draw_fbo = nullptr;
	}
	m_vr_staged.clear();
	m_vr_right_rtts.destroy();
	m_xr_overlay_img.reset();
	m_vr_warp_scratch.clear();

	if (s_vr_batch_owner == this)
	{
		vk::g_end_renderpass_hook = nullptr;
		vk::g_before_end_renderpass_hook = nullptr;
		s_vr_batch_owner = nullptr;
	}
	rsx::vr::set_multiview_active(false);
	vk::g_vr_stereo_layers = false;
	m_vr_mv_right_eye.reset();
	if (m_vr_batching)
	{
		for (auto& [primary, batches] : m_vr_primary_batches)
		{
			batches->pool.destroy(); // frees its secondary command buffers
		}
		m_vr_primary_batches.clear();
	}
}

// clear_surface(): sub-viewport clears moved into the HUD box; returns the right eye's rectangle (same size, shifted).
std::optional<areai> VKGSRender::vr_map_clear_rect(u16 fb_width, u16 fb_height, u16& scissor_x, u16& scissor_y, u16& scissor_w, u16& scissor_h)
{
	std::optional<areai> vr_right_clear;
	const bool full_frame = (scissor_w == fb_width && scissor_h == fb_height);
	if (!full_frame && rsx::vr::camera_probe::get().render_enabled())
	{
		f32 rect[4] = {static_cast<f32>(scissor_x), static_cast<f32>(scissor_y), static_cast<f32>(scissor_x + scissor_w), static_cast<f32>(scissor_y + scissor_h)};
		f32 right[4];
		m_vr_clear_quads_valid = false;
		if (rsx::vr::camera_probe::get().map_subviewport_clear(resolution_scaling_config.scale_factor(), resolution_scaling_config.scale_factor_y(),
				m_framebuffer_layout.width, m_framebuffer_layout.height,
				fb_width, fb_height, rect, right, m_vr_clear_quads))
		{
			m_vr_clear_quads_valid = true;
			scissor_x = static_cast<u16>(std::floor(rect[0]));
			scissor_y = static_cast<u16>(std::floor(rect[1]));
			scissor_w = static_cast<u16>(std::ceil(rect[2]) - scissor_x);
			scissor_h = static_cast<u16>(std::ceil(rect[3]) - scissor_y);
			const int rx = static_cast<int>(std::floor(right[0]));
			const int ry = static_cast<int>(std::floor(right[1]));
			vr_right_clear = areai{rx, ry, rx + scissor_w, ry + scissor_h};
		}
	}
	return vr_right_clear;
}

// clear_surface(), after the guest's clear was recorded (cleared = the attachments list was not empty).
void VKGSRender::vr_after_clear(bool cleared, bool full_frame, bool update_color, bool update_z, u16 scissor_x, u16 scissor_y, u16 scissor_w, u16 scissor_h,
	VkImageAspectFlags vr_cleared_ds_aspects, const std::optional<areai>& vr_right_clear)
{
	// A full colour clear starts the target over, without 3D content (see vr_has_3d).
	if (full_frame && update_color)
	{
		for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
		{
			if (auto* surface = std::get<1>(m_rtts.m_bound_render_targets[index]))
			{
				surface->vr_has_3d = false;
			}
		}
	}
	if (auto* depth = std::get<1>(m_rtts.m_bound_depth_stencil); depth && full_frame && update_z)
	{
		depth->vr_has_3d = false;
	}

	if (cleared && full_frame && update_color)
	{
		m_vr_frame_covered.push_back(m_framebuffer_layout.color_addresses[0]);

		// A fully cleared target holds nothing from an older head pose.
		if (m_vr_applied_pose)
		{
			for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
			{
				if (auto* surface = std::get<1>(m_rtts.m_bound_render_targets[index]))
				{
					surface->vr_pose = m_vr_applied_pose;
				}
			}
		}
	}

	// A guest clear defines identical contents for both eyes inside the cleared
	// rectangle. Copy that rectangle of the cleared attachments into the isolated
	// right cache; subsequent eye-specific draws diverge them. Partial (scissored)
	// clears are mirrored the same way: without them a right-eye target that is
	// only ever cleared in part keeps accumulating old frames (inFamous 2 smears).
	// Attachments this clear did not touch are left alone.
	if ((update_color || update_z) && rsx::vr::camera_probe::get().render_enabled() && !m_vr_multiview &&
		m_vr_right_fbo_images.size() == m_fbo_images.size())
	{
		// Batched right-eye draws precede this clear in guest order.
		vr_batch_flush();

		for (usz i = 0; i < m_fbo_images.size(); ++i)
		{
			auto* src = m_fbo_images[i];
			auto* dst = m_vr_right_fbo_images[i];
			const bool is_color = (src->aspect() & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
			if (is_color ? !update_color : !update_z)
			{
				continue;
			}
			const int x2 = std::min<int>(scissor_x + scissor_w, std::min(src->width(), dst->width()));
			const int y2 = std::min<int>(scissor_y + scissor_h, std::min(src->height(), dst->height()));
			const areai rect = full_frame ? areai{0, 0, static_cast<int>(src->width()), static_cast<int>(src->height())} : areai{scissor_x, scissor_y, x2, y2};
			if (rect.x2 <= rect.x1 || rect.y2 <= rect.y1)
			{
				continue;
			}
			// Copy only the aspects this clear wrote. A stencil-only clear (Dragon's Dogma, before each light
			// volume pass) must not copy the left eye's depth over the right eye's.
			const VkImageAspectFlags aspects = is_color ? VkImageAspectFlags{0xFF} : (vr_cleared_ds_aspects ? vr_cleared_ds_aspects : VkImageAspectFlags{0xFF});
			if (vr_right_clear)
			{
				// The cleared area is uniform: copy it into the right eye's own rectangle.
				areai dst_rect = *vr_right_clear;
				dst_rect.x1 = std::max(dst_rect.x1, 0);
				dst_rect.y1 = std::max(dst_rect.y1, 0);
				dst_rect.x2 = std::min<int>(dst_rect.x2, dst->width());
				dst_rect.y2 = std::min<int>(dst_rect.y2, dst->height());
				const int w = std::min(dst_rect.width(), rect.width()), hgt = std::min(dst_rect.height(), rect.height());
				if (w > 0 && hgt > 0)
				{
					vk::copy_image(*m_current_command_buffer, src, dst, areai{rect.x1, rect.y1, rect.x1 + w, rect.y1 + hgt},
						areai{dst_rect.x1, dst_rect.y1, dst_rect.x1 + w, dst_rect.y1 + hgt}, {}, aspects, aspects);
				}
				continue;
			}
			vk::copy_image(*m_current_command_buffer, src, dst, rect, rect, {}, aspects, aspects);
		}
		m_vr_right_rtts.on_write({update_color, update_color, update_color, update_color}, update_z);
		m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
	}
}

// load_program_env(): texture parameters with feedback textures shifted (profile reproject_older_frames).
bool VKGSRender::vr_shift_texture_params(rsx::fragment_program_texture_config& params)
{
	return vk::xr::is_running() && vr_reprojects_older_frames() && vr_shift_feedback_textures(params);
}

// load_program_env(): profile fragment_constant_overrides, applied after the fill.
void VKGSRender::vr_apply_fragment_constant_overrides(const std::vector<const rsx::vr::fragment_constant_override*>* overrides, void* buf, usz fragment_constants_size)
{
	if (!overrides)
	{
		return;
	}

	for (const auto* rule : *overrides)
	{
		if ((rule->constant + 1) * 16 <= fragment_constants_size)
		{
			std::memcpy(static_cast<u8*>(buf) + rule->constant * 16, rule->value.data(), 16);
		}
	}
	m_fragment_constants_overridden = true;
}

// The vertex context after its scale/offset matrix (bytes 64 to 96). One place for the layout:
// load_program_env() fills the guest's viewport matrix in front of it, vr_hud_vertex_env() the
// HUD box's. Mirrors vertex_context_t in RSXDefines2.glsl.
void VKGSRender::fill_vertex_env_tail(char* buf, f32 vr_keep_depth)
{
	const auto* ctx = &rsx::method_registers;
	m_draw_processor.fill_user_clip_data(buf + 64);
	*(reinterpret_cast<u32*>(buf + 68)) = ctx->transform_branch_bits();
	*(reinterpret_cast<f32*>(buf + 72)) = ctx->point_size() * resolution_scaling_config.scale_factor();
	*(reinterpret_cast<f32*>(buf + 76)) = ctx->clip_min();
	*(reinterpret_cast<f32*>(buf + 80)) = ctx->clip_max();
	*(reinterpret_cast<f32*>(buf + 84)) = vr_keep_depth; // vertex_context_t::vr_keep_depth
	// vertex_context_t::vr_depth_scale, vr_depth_offset: the viewport's z row, from which vr_keep_depth rebuilds the game's
	// window depth (the box's w enters the HUD-box matrix's z row through the offset).
	*(reinterpret_cast<f32*>(buf + 88)) = ctx->viewport_scale_z();
	*(reinterpret_cast<f32*>(buf + 92)) = ctx->viewport_offset_z();
}

usz VKGSRender::vr_depth_remap_size() const
{
	// The compiled shaders read it (vk::vr_insert_depth_remap_vertex_end); the interpreter does not.
	return (current_vertex_program.ctrl & RSX_SHADER_CONTROL_VR_DEPTH_REMAP) && m_program && !m_shader_interpreter.is_interpreter(m_program) ? 80 : 0;
}

void VKGSRender::vr_write_depth_remap(void* dst, usz size, bool eye) const
{
	if (!size)
	{
		return;
	}
	// The matrix, then (1 / the game projection's x scale, 1 / its y scale, the render target's host width, height): the
	// ray variant (profile depth_remap_ray_texcoord) builds the game camera's ray from the remapped position and reads
	// the depth texel under the pixel at the depth texture's own size. Zero scales = no eye transform (keep the game's ray).
	f32 m[5][4] = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 0.f, 1.f}, {}};
	f32 px = 0.f, py = 0.f;
	if (eye && rsx::vr::camera_probe::get().depth_remap_matrix(reinterpret_cast<f32(&)[4][4]>(m)) &&
		rsx::vr::camera_probe::get().game_projection_scale(px, py))
	{
		// The ray's z is +1 here, but the game's view looks down -z (its pass divides by -z): x and y take the opposite
		// sign. With "++" Sonic's shadow mask put far objects (the pagoda) in shadow and lost the cars' own shadows,
		// differently per eye; "--" matches flat (Matt, 2026-10-07). Dev: RPCS3_VR_REMAP_RAY_SIGN=<x sign><y sign>.
		static const std::string s_signs = []() -> std::string
		{
			const char* v = std::getenv("RPCS3_VR_REMAP_RAY_SIGN");
			return v ? v : "--";
		}();
		m[4][0] = (s_signs.size() > 0 && s_signs[0] == '-' ? -1.f : 1.f) / px;
		m[4][1] = (s_signs.size() > 1 && s_signs[1] == '-' ? -1.f : 1.f) / py;
	}
	m[4][2] = m_draw_fbo ? static_cast<f32>(m_draw_fbo->width()) : 0.f;
	m[4][3] = m_draw_fbo ? static_cast<f32>(m_draw_fbo->height()) : 0.f;
	std::memcpy(dst, m, std::min(size, sizeof(m)));
}

// prepare_rtts(): surfaces the game reads back (see below) are copied as soon as they are left. The right eye's
// batch for the pass runs when the left pass ends, so a copy recorded after it made each read wait for
// the right eye's work too. When a listed section is about to be copied, end the left pass without
// running the batch, record the copies, then run the batch: the read waits only for the left eye.
// The RSX's own reads count too (MotorStorm: Pacific Rift samples render target memory as a plain texture
// each frame, and the texture cache flushed the target, waiting for both eyes' queued work).
// Dev, live A/B from the probe file: dev=1 keeps the old order (copies after the right eye), dev=2 leaves
// the RSX's own reads out. RPCS3_VR_NO_RSX_EARLY=1 does the latter from the start.
bool VKGSRender::vr_before_prepare_rtts()
{
	const bool vr_early_readback = rsx::vr::camera_probe::get().render_enabled();
	bool vr_copied_before_right_eye = false;
	const u32 vr_dev = rsx::vr::camera_probe::get().dev_flags();
	static const bool s_no_rsx_early_env = std::getenv("RPCS3_VR_NO_RSX_EARLY") != nullptr; // once: a getenv per target change was ~1% of the RSX thread (GT5)
	const bool s_no_rsx_early = (vr_dev & 2) || s_no_rsx_early_env;
	const bool s_copy_after_right = (vr_dev & 1) != 0;
	m_texture_cache.vr_record_flushes = vr_early_readback && !s_no_rsx_early;
	if (vr_early_readback)
	{
		std::lock_guard lock_flushed(m_texture_cache.vr_flushed_mutex);
		if (!m_texture_cache.vr_flushed_ranges.empty())
		{
			std::lock_guard lock(m_vr_readback_mutex);
			for (const auto& range : m_texture_cache.vr_flushed_ranges)
			{
				if (std::find(m_vr_readback_ranges.begin(), m_vr_readback_ranges.end(), range) == m_vr_readback_ranges.end() && m_vr_readback_ranges.size() < 64)
				{
					m_vr_readback_ranges.push_back(range);
					rsx_log.notice("VR: render target memory 0x%x-0x%x read back by the RSX or CPU; copied early from now on.", range.start, range.end);
				}
			}
			m_texture_cache.vr_flushed_ranges.clear();
		}
	}
	if (vr_early_readback && !s_copy_after_right && m_vr_batch_open && m_vr_batch_primary == m_current_command_buffer)
	{
		std::lock_guard lock(m_vr_readback_mutex);
		bool any = false;
		for (u8 i = 0; i < rsx::limits::color_buffers_count && !any; ++i)
		{
			if (m_surface_info[i].pitch)
			{
				const auto range = m_surface_info[i].get_memory_range();
				any = std::any_of(m_vr_readback_ranges.begin(), m_vr_readback_ranges.end(), [&](const auto& listed)
					{
						return listed.overlaps(range);
					});
			}
		}
		if (any)
		{
			if (vk::is_renderpass_open(*m_current_command_buffer))
			{
				m_vr_batch_executing = true; // keeps the end-of-pass hook from running the batch now
				vk::end_renderpass(*m_current_command_buffer);
				m_vr_batch_executing = false;
			}
			for (u8 i = 0; i < rsx::limits::color_buffers_count; ++i)
			{
				if (m_surface_info[i].pitch)
				{
					vr_copied_before_right_eye |= m_texture_cache.flush_listed_sections(*m_current_command_buffer, m_surface_info[i].get_memory_range(), m_vr_readback_ranges);
				}
			}
			vr_batch_flush();
		}
	}
	return vr_copied_before_right_eye;
}

// prepare_rtts(), after the left eye's targets are bound: Gate 5: bind an isomorphic, host-only
// target set for the right eye. This deliberately uses a separate surface cache: guest addresses
// remain the semantic key, but no right-eye image is ever exposed to guest memory or the ordinary
// texture cache. The path is inert unless render=1 is armed.
// vr_prepare_right_rtts(): profile clear_view_targets. The first bind of a view-sized colour target in a frame clears it
// (both multiview layers; the right eye's own target in two-draw mode), so nothing of an older frame survives where
// the game draws nothing this frame.
void VKGSRender::vr_clear_view_target(vk::render_target* surface, bool right)
{
	auto& probe = rsx::vr::camera_probe::get();
	const auto* profile = probe.profile();
	if (!surface || !profile || !profile->clear_view_targets || !probe.render_enabled() || !m_framebuffer_layout.color_addresses[0])
	{
		return;
	}
	const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
	if (!out.height || !profile->is_view_target(m_framebuffer_layout.width, m_framebuffer_layout.height, static_cast<f32>(out.width) / out.height))
	{
		return;
	}
	if (m_vr_clear_flip != static_cast<u32>(int_flip_index))
	{
		m_vr_clear_flip = static_cast<u32>(int_flip_index);
		m_vr_cleared_targets.clear();
	}
	const u32 key = m_framebuffer_layout.color_addresses[0] ^ (right ? 1u : 0u);
	if (std::find(m_vr_cleared_targets.begin(), m_vr_cleared_targets.end(), key) != m_vr_cleared_targets.end())
	{
		return;
	}
	m_vr_cleared_targets.push_back(key);
	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}
	const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, surface->layers() };
	const VkClearColorValue black{};
	surface->push_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	vkCmdClearColorImage(*m_current_command_buffer, surface->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
	surface->pop_layout(*m_current_command_buffer);
}

void VKGSRender::vr_prepare_right_rtts()
{
	if (vk::xr::is_running())
	{
		vr_track_frame_boundary();
	}

	if (gpuprof_enabled())
	{
		gpuprof_mark({m_framebuffer_layout.color_addresses[0] ? m_framebuffer_layout.color_addresses[0] : m_framebuffer_layout.zeta_address,
			m_framebuffer_layout.width, m_framebuffer_layout.height,
			m_framebuffer_layout.color_addresses[0] ? static_cast<u32>(m_framebuffer_layout.color_format) : 0x1000u + static_cast<u32>(m_framebuffer_layout.depth_format)});
	}

	vr_clear_view_target(std::get<1>(m_rtts.m_bound_render_targets[0]), false);

	if (!rsx::vr::camera_probe::get().render_enabled() || m_vr_multiview) // multiview: the right eye is layer 1 of the ordinary targets
	{
		return;
	}

	// The pending batch targets the current right-eye framebuffer; run it before
	// the right-eye surfaces are rebound or the framebuffer is released.
	vr_batch_flush();

	m_vr_right_rtts.prepare_render_target(*m_current_command_buffer,
		m_framebuffer_layout.color_format, m_framebuffer_layout.depth_format,
		m_framebuffer_layout.width, m_framebuffer_layout.height,
		m_framebuffer_layout.target, m_framebuffer_layout.aa_mode, m_framebuffer_layout.raster_type,
		m_framebuffer_layout.color_addresses, m_framebuffer_layout.zeta_address,
		m_framebuffer_layout.actual_color_pitch, m_framebuffer_layout.actual_zeta_pitch,
		resolution_scaling_config);

	// These lists are hooks for synchronizing the ordinary cache with guest
	// memory. The right-eye cache is intentionally host-only, so retaining
	// their raw surface pointers would only accumulate stale bookkeeping.
	m_vr_right_rtts.superseded_surfaces.clear();
	m_vr_right_rtts.orphaned_surfaces.clear();
	vr_clear_view_target(std::get<1>(m_vr_right_rtts.m_bound_render_targets[0]), true);

	// Initialize new right-eye surfaces as the left eye does before its first draw
	// (clear, or inherit from overlapping older surfaces). Without this a new or
	// recycled right-eye image keeps whatever it last held: Ridge Racer 7's Xevious
	// loader depth-tests its play area against a depth buffer the left eye cleared
	// and the right eye did not, so that eye stayed black.
	const auto init_right = [&](vk::render_target* surface)
	{
		if (surface && ((surface->state_flags & rsx::surface_state_flags::erase_bkgnd) || !surface->old_contents.empty()))
		{
			// Clears and copies cannot run inside a render pass.
			if (vk::is_renderpass_open(*m_current_command_buffer))
			{
				vk::end_renderpass(*m_current_command_buffer);
			}
			surface->write_barrier(*m_current_command_buffer);
		}
	};
	for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
	{
		init_right(std::get<1>(m_vr_right_rtts.m_bound_render_targets[index]));
	}
	init_right(std::get<1>(m_vr_right_rtts.m_bound_depth_stencil));

	m_vr_right_fbo_images.clear();
	for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
	{
		if (auto surface = std::get<1>(m_vr_right_rtts.m_bound_render_targets[index]))
		{
			m_vr_right_fbo_images.push_back(surface);
		}
	}
	if (auto depth = std::get<1>(m_vr_right_rtts.m_bound_depth_stencil))
	{
		m_vr_right_fbo_images.push_back(depth);
	}

	std::vector<u8> vr_input_attachments{};
	if ((current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING) &&
		!m_graphics_state.test(rsx::pipeline_state::fragment_program_state_dirty))
	{
		vr_input_attachments.resize(rsx::utility::get_rtt_indexes(m_framebuffer_layout.target).size());
		std::iota(vr_input_attachments.begin(), vr_input_attachments.end(), 0);
	}

	const u64 vr_renderpass_key = vk::get_renderpass_key(m_vr_right_fbo_images, vr_input_attachments);
	const VkRenderPass vr_renderpass = vk::get_renderpass(*m_device, vr_renderpass_key);
	const auto [vr_width, vr_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config,
		m_framebuffer_layout.width, m_framebuffer_layout.height);

	if (m_vr_right_draw_fbo)
	{
		m_vr_right_draw_fbo->release();
	}
	m_vr_right_draw_fbo = vk::get_framebuffer(*m_device, vr_width, vr_height,
		vk::to_bool32(!vr_input_attachments.empty()), vr_renderpass, m_vr_right_fbo_images);
	m_vr_right_draw_fbo->add_ref();
}

// prepare_rtts(), per surface being left: a surface the game reads back every frame (Gran Turismo 5: two
// 16x8 targets in main memory and a 128x322 one the RSX reads as data) is copied as soon as it is left, and
// that work submitted: the read then waits only for the GPU to get this far, not for both eyes' work queued
// after it. Waiting for the whole queue serialised CPU and GPU (1-1.5 ms per read in stereo, 0.1 flat).
bool VKGSRender::vr_copy_readback_sections(const rsx::gcm_framebuffer_info& info)
{
	if (info.pitch && rsx::vr::camera_probe::get().render_enabled())
	{
		std::lock_guard lock(m_vr_readback_mutex);
		if (!m_vr_readback_ranges.empty())
		{
			return m_texture_cache.flush_listed_sections(*m_current_command_buffer, info.get_memory_range(), m_vr_readback_ranges);
		}
	}
	return false;
}

void VKGSRender::vr_submit_early_copies(bool any)
{
	if (any)
	{
		if (vk::is_renderpass_open(*m_current_command_buffer))
		{
			vk::end_renderpass(*m_current_command_buffer);
		}
		flush_command_queue();
	}
}

// scaled_image_from_memory(), before the blit: profile current_frame_copies, the stereo inspector's note and the trace.
void VKGSRender::vr_before_blit(rsx::blit_src_info& src, const rsx::blit_dst_info& dst)
{
	if (gpuprof_enabled())
	{
		// Dev GPU profiler: a segment per blit, keyed by its destination ("fmt" 0x30000; size is the source's).
		gpuprof_mark({vm::get_addr(dst.pixels), src.width, src.height, 0x30000u + static_cast<u32>(src.format)});
		static u32 s_logged = 0;
		if (s_logged++ < 64)
			rsx_log.notice("GPU profile: blit %08x %ux%u fmt %u -> %08x clip %ux%u scale %.3fx%.3f", vm::get_addr(src.pixels), src.width, src.height,
				static_cast<u32>(src.format), vm::get_addr(dst.pixels), dst.clip_width, dst.clip_height, dst.scale_x, dst.scale_y);
	}

	if (vk::xr::is_running())
	{
		if (const auto* profile = rsx::vr::camera_probe::get().profile(); profile && profile->current_frame_copies)
		{
			vr_redirect_previous_frame_copy(src);
		}
	}
	vr_redirect_blit_source(src);

	if (auto& inspector = rsx::vr::stereo_inspector::get(); inspector.capturing())
	{
		inspector.record_note("blit", fmt::format("\"src\":%u,\"src_pitch\":%u,\"src_w\":%u,\"src_h\":%u,\"src_fmt\":%u,"
												  "\"dst\":%u,\"dst_pitch\":%u,\"dst_w\":%u,\"dst_h\":%u,\"dst_fmt\":%u,\"scale_x\":%f,\"scale_y\":%f,\"swizzled\":%d,\"clip_x\":%u,\"clip_y\":%u",
										  vm::get_addr(src.pixels), src.pitch, src.width, src.height, static_cast<u32>(src.format),
										  vm::get_addr(dst.pixels), dst.pitch, dst.clip_width, dst.clip_height, static_cast<u32>(dst.format), dst.scale_x, dst.scale_y,
										  dst.swizzled ? 1 : 0, dst.clip_x, dst.clip_y));
	}

	if (vk::xr::is_running() && vr_tracing())
	{
		vr_trace_flush_cam();
		m_vr_trace += fmt::format(" T%x>%x", vm::get_addr(src.pixels), vm::get_addr(dst.pixels));
	}
}

// scaled_image_from_memory(), after a successful blit: the copy carries the pose its source was
// drawn with, and the right eye gets the same copy.
void VKGSRender::vr_after_blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate)
{
	if (gpuprof_enabled())
	{
		gpuprof_mark({vm::get_addr(dst.pixels), src.width, src.height, 0x40000u + static_cast<u32>(src.format)}); // dev: after the blit
	}

	if (vk::xr::is_running())
	{
		const auto find = [&](u32 address, u32 pitch) -> vk::render_target*
		{
			if (auto* surface = m_rtts.find_color_surface(address, pitch))
			{
				return surface;
			}
			return m_rtts.get_surface_at(address);
		};
		auto* from = find(vm::get_addr(src.pixels), src.pitch);
		auto* to = find(vm::get_addr(dst.pixels), dst.pitch);
		if (from && to && from->vr_pose)
		{
			to->vr_pose = from->vr_pose;
		}
		if (from && to && from->vr_has_3d)
		{
			to->vr_has_3d = true;
		}
	}

	if (rsx::vr::camera_probe::get().render_enabled())
	{
		vr_mirror_blit(src, dst, interpolate); // (multiview: only the staging through memory)
	}
}

// frame_context_cleanup(): the headset overlay renderer's temporaries of a dirty view.
void VKGSRender::vr_remove_overlay_temp_resources(u32 uid)
{
	if (m_xr_overlay_img)
	{
		vk::get_overlay_pass<vk::ui_overlay_renderer_xr>()->remove_temp_resources(uid);
	}
}

// flip(): the right eye's image of the displayed buffer, for the side-by-side desktop view.
// Returns true when generated stereo is shown. Also records the resolution-scaled eye size,
// as rewritten by get_present_source(), for the headset.
bool VKGSRender::vr_present_right_eye(const vk::present_surface_info& present_info, const rsx::avconf& avconfig, u32 buffer_width, u32 buffer_height, vk::viewable_image* image_to_flip, vk::viewable_image*& image_to_flip2)
{
	m_vr_eye_width = present_info.width;
	m_vr_eye_height = present_info.height;
	// The picture's shape for the fixed screen: the guest buffer's, which the eye image no longer has once its axes are
	// scaled apart (headset-shaped eyes).
	vk::xr::set_screen_content_aspect(buffer_height ? static_cast<f32>(buffer_width) / buffer_height : 0.f);

	{
		static bool s_reported_present = false;
		if (!std::exchange(s_reported_present, true) || (m_vr_multiview && (m_vr_flips_since_boundary % 600) == 7))
		{
			rsx_log.success("VR: present: %ux%u (eye %ux%u) at 0x%x, native 3D %d, render %d, multiview %d", buffer_width, buffer_height, present_info.width, present_info.height,
				present_info.address, avconfig.stereo_enabled, rsx::vr::camera_probe::get().render_enabled(), m_vr_multiview);
		}
	}
	if (avconfig.stereo_enabled || !rsx::vr::camera_probe::get().render_enabled())
	{
		return false;
	}

	if (m_vr_multiview)
	{
		// Multiview stereo: the right eye is layer 1 of the display surface, copied out as a plain image.
		// get_present_source() hands back a one-layer copy when the surface's format is not the output's
		// (or an upload when no surface holds the buffer): find the surface itself then, as below.
		vk::viewable_image* stereo_image = (image_to_flip && image_to_flip->stereo_layers && image_to_flip->layers() > 1) ? image_to_flip : nullptr;
		if (!stereo_image)
		{
			for (const auto& [address, surface] : m_rtts.m_bound_render_targets)
			{
				if (address == present_info.address && surface &&
					surface->get_surface_width<rsx::surface_metrics::samples>() >= buffer_width &&
					surface->get_surface_height<rsx::surface_metrics::samples>() >= buffer_height)
				{
					stereo_image = surface->get_surface(rsx::surface_access::transfer_read);
					break;
				}
			}
			if (!stereo_image)
			{
				const auto format_bpp = rsx::get_format_block_size_in_bytes(present_info.format);
				auto overlap = m_rtts.get_merged_texture_memory_region(*m_current_command_buffer,
					present_info.address, buffer_width, buffer_height, present_info.pitch,
					format_bpp, rsx::surface_access::transfer_read);
				if (!overlap.empty())
				{
					const auto& section = overlap.back();
					auto* surface = vk::as_rtt(section.surface);
					if (section.base_address == present_info.address &&
						surface->get_surface_width<rsx::surface_metrics::samples>() >= buffer_width &&
						surface->get_surface_height<rsx::surface_metrics::samples>() >= buffer_height)
					{
						stereo_image = section.surface->get_surface(rsx::surface_access::transfer_read);
					}
				}
			}
			if (stereo_image && !(stereo_image->stereo_layers && stereo_image->layers() > 1))
			{
				stereo_image = nullptr;
			}
		}
		if (stereo_image)
		{
			image_to_flip2 = vr_mv_right_eye_image(*m_current_command_buffer, stereo_image);
		}
		static bool s_reported_multiview_stereo = false;
		if (!std::exchange(s_reported_multiview_stereo, true) || (!image_to_flip2 && (m_vr_flips_since_boundary % 300) == 1))
		{
			rsx_log.success("VR: presenting multiview stereo: display image %ux%u, layers %u, stereo %d, surface %d -> right eye %d.",
				image_to_flip ? image_to_flip->width() : 0u, image_to_flip ? image_to_flip->height() : 0u, image_to_flip ? image_to_flip->layers() : 0u,
				image_to_flip ? image_to_flip->stereo_layers : false, stereo_image != nullptr, image_to_flip2 != nullptr);
		}
		return image_to_flip2 != nullptr;
	}

	// The display target is normally still bound at flip. Prefer that exact
	// surface: resolving it through the cache merge path can reject a valid
	// right eye whose inherited memory range is larger than the display.
	// Compare in guest pixels: get_present_source() has already rewritten
	// present_info.width/height to the resolution-scaled size.
	for (const auto& [address, surface] : m_vr_right_rtts.m_bound_render_targets)
	{
		if (address == present_info.address && surface &&
			surface->get_surface_width<rsx::surface_metrics::samples>() >= buffer_width &&
			surface->get_surface_height<rsx::surface_metrics::samples>() >= buffer_height)
		{
			image_to_flip2 = surface->get_surface(rsx::surface_access::transfer_read);
			break;
		}
	}

	if (!image_to_flip2)
	{
		const auto format_bpp = rsx::get_format_block_size_in_bytes(present_info.format);
		auto right_overlap = m_vr_right_rtts.get_merged_texture_memory_region(*m_current_command_buffer,
			present_info.address, buffer_width, buffer_height, present_info.pitch,
			format_bpp, rsx::surface_access::transfer_read);
		if (!right_overlap.empty())
		{
			const auto& section = right_overlap.back();
			auto* surface = vk::as_rtt(section.surface);
			if (section.base_address == present_info.address &&
				surface->get_surface_width<rsx::surface_metrics::samples>() >= buffer_width &&
				surface->get_surface_height<rsx::surface_metrics::samples>() >= buffer_height)
			{
				image_to_flip2 = section.surface->get_surface(rsx::surface_access::transfer_read);
			}
		}
	}

	const bool generated_stereo = image_to_flip2 != nullptr;
	static bool s_reported_generated_stereo = false;
	if (generated_stereo && !std::exchange(s_reported_generated_stereo, true))
	{
		rsx_log.success("Gate 5: presenting generated stereo as side-by-side.");
	}
	return generated_stereo;
}

// flip(): Gate 6: publish the eyes to the OpenXR frame thread. The RSX thread never waits on the
// headset's clock (that stalled guest command processing and dropped frames in busy scenes); it
// records a copy into a free eye buffer, submits, and tags the pair with the pose its draws were
// rotated by. image_to_flip2 is the right eye of generated stereo, or null.
void VKGSRender::vr_publish_frame(const rsx::display_flip_info_t& info, vk::viewable_image* image_to_flip, vk::viewable_image* image_to_flip2)
{
	// Dev GPU checkpoints through the flip: 2 start, 5 eyes copied, 6 overlay drawn, 7 overlay copied, 8 end.
	const auto ckpt = [&](u32 kind)
	{
		if (vk::vr_gpu_checkpoints_enabled())
			vk::vr_gpu_checkpoint(*m_current_command_buffer, { .kind = kind, .frame = m_vr_ckpt_frame, .draw = m_vr_ckpt_draw, .fifo_pos = fifo_ctrl ? fifo_ctrl->get_pos() : 0 });
	};
	ckpt(2);
	struct ckpt_end_t
	{
		VKGSRender* r;
		decltype(ckpt)& c;
		~ckpt_end_t()
		{
			c(8);
			r->m_vr_ckpt_frame++;
			r->m_vr_ckpt_draw = 0;
		}
	} ckpt_end{this, ckpt};
	if (!vk::xr::is_running() && !vk::xr::fake_hmd())
	{
		// No headset session: back to the game's camera.
		rsx::vr::camera_probe::get().clear_vr_view();
		return;
	}

	// Videos some games show without flipping (Demon's Souls: decoded into the display
	// buffer, shown only by RPCS3's UI refresh) count as frames without camera draws, so
	// frames_without_3d_as_screen shows them on the fixed screen instead of head-locked.
	if (info.emu_flip)
	{
		m_vr_last_emu_flip_us = get_system_time();
	}
	const bool vr_video_refresh = !info.emu_flip && !Emu.IsPaused() && get_system_time() - m_vr_last_emu_flip_us > 200'000;
	// Only game flips carry a new eye pair. Flips requested by RPCS3's overlays
	// (e.g. while the home menu pauses emulation) would re-publish the old frame
	// tagged with a newer head pose, dragging the world along with the head. A video
	// frame on the fixed screen has no world to drag: it is published too.
	const bool vr_video_screen = vr_video_refresh && m_vr_video_on_screen;
	const bool xr_eyes = (info.emu_flip || vr_video_screen) && image_to_flip && vk::xr::publish_eyes(*m_current_command_buffer, image_to_flip, image_to_flip2 && !vr_video_screen ? image_to_flip2 : image_to_flip, m_vr_eye_width, m_vr_eye_height);
	ckpt(5);

	// RPCS3's own overlays (home menu, dialogs, notifications) are drawn only on
	// the desktop swapchain below; the headset gets them as a quad layer.
	bool xr_overlay = false;
	if (m_overlay_manager && m_overlay_manager->has_visible())
	{
		constexpr u32 overlay_width = 1920;
		constexpr u32 overlay_height = 1080;
		constexpr VkFormat overlay_format = VK_FORMAT_B8G8R8A8_UNORM;
		if (!m_xr_overlay_img)
		{
			m_xr_overlay_img = std::make_unique<vk::image>(*m_device, m_device->get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D, overlay_format, overlay_width, overlay_height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				0, VMM_ALLOCATION_POOL_SYSTEM);
		}

		vk::image* overlay_img = m_xr_overlay_img.get();
		const VkImageSubresourceRange overlay_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		const VkClearColorValue transparent{};
		overlay_img->change_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vkCmdClearColorImage(*m_current_command_buffer, overlay_img->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &transparent, 1, &overlay_range);
		overlay_img->change_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		const VkRenderPass overlay_pass = vk::get_renderpass(*m_device, vk::get_renderpass_key(overlay_format));
		vk::framebuffer_holder* overlay_fbo = vk::get_framebuffer(*m_device, overlay_width, overlay_height, VK_FALSE, overlay_pass, {overlay_img});
		overlay_fbo->add_ref();
		{
			auto ui_renderer = vk::get_overlay_pass<vk::ui_overlay_renderer_xr>();
			std::lock_guard lock(*m_overlay_manager);
			const areau overlay_area = {0, 0, overlay_width, overlay_height};
			for (const auto& view : m_overlay_manager->get_views())
			{
				ui_renderer->run(*m_current_command_buffer, overlay_area, overlay_fbo, overlay_pass, m_texture_upload_buffer_ring_info, *view.get());
			}
		}
		overlay_fbo->release();
		ckpt(6);

		xr_overlay = vk::xr::publish_overlay(*m_current_command_buffer, overlay_img);
		ckpt(7);
	}

	if (xr_eyes || xr_overlay)
	{
		flush_command_queue();
		if (g_cfg.video.multithreaded_rsx)
		{
			// The submit may still be queued on the offload thread.
			g_fxo->get<rsx::dma_manager>().sync();
		}
		vk::xr::signal_published();
	}

	if (xr_eyes)
	{
		// Also shows an overlay published in this flip.
		f32 tan_x = 0.f, tan_y = 0.f;
		// Rendered with the headset's FOV, the projection layer declares the headset's own per-eye FOV
		// and needs nothing from the game. Waiting for the game's first camera draw put everything before
		// it (Killzone HD's splash screens, videos and menus, already in the HUD box) on a head-locked quad,
		// stretched: the box looked ultrawide and followed the face until the first 3D frame.
		const bool have_fov = rsx::vr::camera_probe::get().get_vr_fov(tan_x, tan_y) || vk::xr::hmd_fov();
		// The pose the displayed image was drawn with, if it can be traced.
		u32 pose = m_vr_applied_pose;
		if (m_vr_frame_boundaries && info.buffer < display_buffers_count)
		{
			if (auto* surface = m_rtts.get_surface_at(rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL));
				surface && surface->vr_pose)
			{
				pose = surface->vr_pose;
			}
		}
		// Profile display_rect: a 3D frame shows only the part of the display that holds the game's view (God of
		// War's scene is inset with a black border); frames on the fixed screen show the whole picture.
		std::array<f32, 4> crop{0.f, 0.f, 1.f, 1.f};
		if (const auto* profile = rsx::vr::camera_probe::get().profile();
			profile && profile->display_rect[2] > 0.f && !m_vr_video_on_screen && !g_cfg.video.vr.fixed_screen && vk::xr::projection_mode())
		{
			const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
			if (out.width && out.height)
			{
				const auto& r = profile->display_rect;
				crop = {std::clamp(r[0] / out.width, 0.f, 1.f), std::clamp(r[1] / out.height, 0.f, 1.f),
					std::clamp((r[0] + r[2]) / out.width, 0.f, 1.f), std::clamp((r[1] + r[3]) / out.height, 0.f, 1.f)};
			}
		}
		vk::xr::commit_eyes(have_fov, tan_x, tan_y, pose, crop);
		if (vr_tracing())
		{
			vr_trace_flush_cam();
			rsx_log.notice("VR trace:%s F[d%u] declared %u(%.1f) applied %u(%.1f)", m_vr_trace, info.buffer,
				pose, vk::xr::render_pose_yaw(pose), m_vr_applied_pose, vk::xr::render_pose_yaw(m_vr_applied_pose));
		}
		m_vr_trace.clear();
		m_vr_trace_flips++;
	}
	else if (xr_overlay)
	{
		vk::xr::commit_overlay();
	}

	if (!xr_overlay)
	{
		vk::xr::hide_overlay();
	}

	// Head pose for the next game frame. Overlay flips (paused emulation) draw
	// nothing, so the pose waits for the next game flip. Games whose frames end in
	// a display buffer take it at the frame boundary instead (prepare_rtts); the
	// flip falls back to it if no boundary has been seen for two flips.
	if (info.emu_flip && info.buffer < display_buffers_count)
	{
		// Whether the frame just shown holds 3D content (vr_has_3d). A buffer the CPU wrote has no surface.
		const auto* shown = m_rtts.get_surface_at(rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL));
		m_vr_flip_has_3d = shown && shown->vr_has_3d;
		m_vr_flip_pose = shown ? shown->vr_pose : 0;
		vk::xr::keep_render_pose(m_vr_flip_pose);
	}
	else if (vr_video_refresh)
	{
		// Re-shown without game flips for 200 ms: a video decoded into the display buffer (or a stall).
		m_vr_flip_has_3d = false;
	}
	if (vr_video_refresh || (info.emu_flip && (!m_vr_frame_boundaries || ++m_vr_flips_since_boundary > 2)))
	{
		vr_update_view();
		if (vr_tracing())
		{
			m_vr_trace += fmt::format(" L%u(%.1f)", m_vr_applied_pose, vk::xr::render_pose_yaw(m_vr_applied_pose));
		}
	}
}

// flip(): the first frames of a boot can present an image the capture cannot read (Ridge Racer 7:
// smaller than the display buffer, or not 32-bit, or still owned by another queue); a screenshot
// then crashed the driver or failed image::push_layout's queue check. Skip it instead.
bool VKGSRender::vr_capturable(vk::viewable_image* image_to_flip, u32 buffer_width, u32 buffer_height)
{
	const bool capturable = image_to_flip->width() >= buffer_width && image_to_flip->height() >= buffer_height &&
	                        vk::get_format_texel_width(image_to_flip->format()) == 4 &&
	                        (image_to_flip->current_queue_family == VK_QUEUE_FAMILY_IGNORED || image_to_flip->current_queue_family == m_current_command_buffer->get_queue_family());
	if (!capturable)
	{
		rsx_log.warning("Screenshot/recording skipped: the presented image (%ux%u fmt %d) cannot be captured this frame (buffer %ux%u).",
			image_to_flip->width(), image_to_flip->height(), static_cast<int>(image_to_flip->format()), buffer_width, buffer_height);
	}
	return capturable;
}

// flip(): a screenshot while generated stereo is shown holds both eyes side by side.
bool VKGSRender::vr_side_by_side_shot(vk::viewable_image* image_to_flip, vk::viewable_image* image_to_flip2, u32 buffer_width, u32 buffer_height)
{
	if (!image_to_flip2)
	{
		return false;
	}
	const bool sbs_shot = vk::get_format_texel_width(image_to_flip2->format()) == 4 &&
	                      image_to_flip2->width() >= buffer_width && image_to_flip2->height() >= buffer_height;
	rsx_log.notice("VR screenshot: left %ux%u fmt %d, right %ux%u fmt %d, side by side %d", image_to_flip->width(), image_to_flip->height(), static_cast<int>(image_to_flip->format()),
		image_to_flip2->width(), image_to_flip2->height(), static_cast<int>(image_to_flip2->format()), sbs_shot);
	return sbs_shot;
}

// flip(): generated stereo from a display buffer larger than the region the game shows (Gran Turismo 5:
// 2048x1080 surfaces, 1280x720 shown) would put each whole surface into its half of the window, the shown
// part small in a corner. Copy the shown region of each eye into an image of that size first.
// flip(): the desktop mirror of generated stereo shows both eyes side by side in the region fitted to the eye image's
// pixel size. With headset-shaped eyes (rsx_vr_eye_shape.h) that image is narrow (1920x2160 for a 93 x 98 degree view)
// and each eye came out twice as tall as wide. Each eye now gets the shape it is shown with: the headset eye's view
// (width / height in tangents), or the game's picture when the frame is on the fixed screen (menus, videos).
void VKGSRender::vr_mirror_region(areai& region, bool generated_stereo) const
{
	if (!generated_stereo || g_cfg.video.stretch_to_display_area || !m_swapchain_dims.width || !m_swapchain_dims.height)
	{
		return;
	}
	std::array<f32, 9> rotation{};
	f32 tan_x = 0.f, tan_y = 0.f;
	const bool headset_view = rsx::vr::camera_probe::get().vr_view_rotation(rotation, tan_x, tan_y);
	const f32 eye_aspect = headset_view ? rsx::vr::eye_view_aspect() : vk::xr::screen_content_aspect();
	if (!(eye_aspect > 0.f))
	{
		return;
	}
	const f32 aspect = 2.f * eye_aspect;
	const f32 window_w = static_cast<f32>(m_swapchain_dims.width), window_h = static_cast<f32>(m_swapchain_dims.height);
	const f32 w = std::min(window_w, window_h * aspect);
	const f32 h = w / aspect;
	const s32 x = static_cast<s32>((window_w - w) * 0.5f), y = static_cast<s32>((window_h - h) * 0.5f);
	region = {x, y, x + static_cast<s32>(w), y + static_cast<s32>(h)};
}

void VKGSRender::vr_crop_for_side_by_side(rsx::simple_array<vk::viewable_image*>& calibration_src, u32 buffer_width, u32 buffer_height)
{
	if (!buffer_width || !buffer_height)
	{
		return;
	}

	for (auto& img : calibration_src)
	{
		if (img->width() <= buffer_width && img->height() <= buffer_height)
		{
			continue;
		}
		const u32 w = std::min<u32>(buffer_width, img->width()), h = std::min<u32>(buffer_height, img->height());
		auto& crop = m_vr_warp_scratch[(1ull << 63) | (static_cast<u64>(img->format()) << 40) | (static_cast<u64>(w) << 20) | h];
		if (!crop)
		{
			crop = std::make_unique<vk::viewable_image>(*m_device, m_device->get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D, img->format(), w, h, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				0, VMM_ALLOCATION_POOL_SYSTEM);
			crop->change_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		vk::copy_image(*m_current_command_buffer, img, crop.get(), areai{0, 0, static_cast<int>(w), static_cast<int>(h)},
			areai{0, 0, static_cast<int>(w), static_cast<int>(h)});
		img = crop.get();
	}
}

// ---- Multiview stereo: both eyes in one draw (plans/7-multiview-plan.md) --------------------------

// end_renderpass() hook, before vkCmdEndRenderPass: a query pair begun inside the pass ends inside it.
void VKGSRender::vr_mv_on_before_end_renderpass(const vk::command_buffer& cmd)
{
	if (VKGSRender* const self = s_vr_batch_owner; self && self->m_vr_multiview)
	{
		self->vr_mv_end_query_segment(cmd);
	}
}

// prepare_rtts(): multiview follows the stereo renderer. Every render target changes its layer count
// on a switch: finish the GPU's work and drop them all (their contents come back from guest memory, or
// are cleared, as after any other cache invalidation).
void VKGSRender::vr_update_multiview_mode()
{
	const bool want = m_vr_multiview_supported && rsx::vr::camera_probe::get().render_enabled();
	if (want == m_vr_multiview)
	{
		return;
	}

	vr_batch_flush();
	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}
	flush_command_queue(true);
	m_rtts.invalidate_all();
	if (m_draw_fbo)
	{
		m_draw_fbo->release();
		m_draw_fbo = nullptr;
	}
	m_fbo_images.clear();
	if (m_vr_mv_right_eye)
	{
		vk::get_resource_manager()->dispose(m_vr_mv_right_eye);
	}

	m_vr_multiview = want;
	vk::g_vr_stereo_layers = want;
	rsx::vr::set_multiview_active(want);
	m_vr_mv_box_scissor = false;
	m_vr_mv_open_query = umax;
	// Programs carry the multiview bit: re-evaluate them, and the framebuffer.
	m_graphics_state |= (rsx::vertex_program_state_dirty | rsx::fragment_program_state_dirty | rsx::rtt_config_dirty);
	m_samplers_dirty.store(true);
	rsx_log.notice("VR: multiview stereo %s", want ? "on (both eyes in one draw)" : "off");
}

// emit_geometry(): both eyes' constants side by side in one allocation, so the one bound window serves
// both views (the right eye's entry is one stride further: m_vr_mv_right_xform_offset). The fill and the
// camera classification are those of bind_vr_eye_constants, run for each eye; the per-eye HUD-box
// scissors are recorded for bind_viewport and vr_before_left_draw.
bool VKGSRender::bind_vr_eye_constants_pair(usz source_size)
{
	auto& d = m_vr_draw;
	auto& probe = rsx::vr::camera_probe::get();
	probe.clear_box_mapped();
	m_vr_mv_box_scissor = false;
	m_vr_mv_right_xform_offset = m_xform_constants_dynamic_offset;
	if (!source_size || !m_program || m_vs_binding_table->cbuf_location == umax)
	{
		return false;
	}

	const bool full_bank = m_shader_interpreter.is_interpreter(m_program) || (m_vertex_prog && m_vertex_prog->has_indexed_constants);
	if (!full_bank && !m_vertex_prog)
	{
		return false;
	}

	const usz size = full_bank ? 8192 : m_vertex_prog->constant_ids.size() * 16;
	if (!size || size != source_size)
	{
		// The last upload does not describe this program; leave the guest allocation bound (both views read it).
		static std::set<u64> s_seen;
		const u64 hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program);
		if (s_seen.size() < 32 && s_seen.insert(hash).second)
		{
			rsx_log.warning("VR: eye constants not applied to program %016llx (constants %u bytes, last upload %u)", hash, size, source_size);
		}
		return false;
	}

	const bool prof = m_gpuprof_enabled > 0;
	auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
	const auto lap = [&](int i)
	{
		if (!prof)
			return;
		const auto t = std::chrono::steady_clock::now();
		m_gpuprof_eye_ms[i] += std::chrono::duration<f64, std::milli>(t - t0).count();
		t0 = t;
	};
	const u64 alignment = m_device->gpu().get_limits().minUniformBufferOffsetAlignment;
	const usz remap_size = vr_depth_remap_size(); // profile depth_remap_programs: each eye's depth remap follows its constants
	const usz stride = utils::align(size + remap_size, alignment);
	static thread_local std::vector<u8> scratch;
	scratch.resize(stride * 2);
	const auto constant_ids = full_bank ? std::span<const u16>{} : std::span<const u16>(m_vertex_prog->constant_ids);
	// Ordinary stores, as in bind_vr_eye_constants: the classification reads this buffer right back.
	const auto& guest_constants = rsx::method_registers.transform_constants;
	// The game's constants for this program (the eyes' transforms start from these); run again for the rare redo below
	// instead of keeping a copy of every draw's constants.
	const auto fill_guest = [&](u8* dst)
	{
		if (constant_ids.empty())
		{
			std::memcpy(dst, guest_constants.data(), 468 * 16);
		}
		else
		{
			u8* out = dst;
			for (const u16 index : constant_ids)
			{
				std::memcpy(out, &guest_constants[index], 16);
				out += 16;
			}
		}
		if (probe.enabled())
		{
			probe.apply(dst, constant_ids.data(), constant_ids.size(), guest_constants.data(),
				rsx::method_registers.surface_clip_width(), rsx::method_registers.surface_clip_height());
		}
		scale_offset_constants(dst, constant_ids);
	};
	fill_guest(scratch.data());
	std::memcpy(scratch.data() + stride, scratch.data(), size);
	vr_write_depth_remap(scratch.data() + size, remap_size, false);
	vr_write_depth_remap(scratch.data() + stride + size, remap_size, false);
	lap(0);

	const u16* reloc = full_bank ? nullptr : m_vertex_prog->constant_ids.data();
	const usz reloc_size = full_bank ? 0 : m_vertex_prog->constant_ids.size();
	// Programs indexing a bone palette get the whole bank: match camera blocks on the slots they read directly.
	const auto* vr_profile = probe.profile();
	const rsx::vr::full_bank_direct_slots direct_slots(full_bank && m_vertex_prog && vr_profile && vr_profile->camera_slots_read_directly ? &m_vertex_prog->constant_ids : nullptr);
	// Programs kept on the game camera: the profile's game_camera_programs, or probe gamecam= (development).
	const auto& probe_game_camera = probe.game_camera_programs();
	const auto& probe_game_camera_nocolor = probe.game_camera_nocolor_programs();
	bool keep_game_camera = false;
	if (!probe_game_camera.empty() || !probe_game_camera_nocolor.empty() || (vr_profile && !vr_profile->game_camera_programs.empty()))
	{
		const u64 hash = vr_vertex_program_hash();
		const bool no_colour = std::none_of(std::begin(m_framebuffer_layout.color_write_enabled), std::end(m_framebuffer_layout.color_write_enabled), [](bool b)
								   {
									   return b;
								   }) ||
		                       !rsx::method_registers.color_write_enabled(0);
		keep_game_camera = std::find(probe_game_camera.begin(), probe_game_camera.end(), hash) != probe_game_camera.end() ||
		                   (no_colour && std::find(probe_game_camera_nocolor.begin(), probe_game_camera_nocolor.end(), hash) != probe_game_camera_nocolor.end()) ||
		                   (vr_profile && std::find(vr_profile->game_camera_programs.begin(), vr_profile->game_camera_programs.end(), hash) != vr_profile->game_camera_programs.end());
	}
	bool classified_world = false;
	// Dev (probe why=<this program>): the game's constants, to log afterwards which slots each eye changed.
	const bool why_this = probe.why_program() && probe.why_program() == vr_vertex_program_hash();
	static thread_local std::vector<u8> why_game;
	if (why_this)
	{
		why_game.assign(scratch.begin(), scratch.begin() + size);
	}
	if (!keep_game_camera)
	{
		// When the two eyes classify the draw differently (a rule that came true during the left eye's apply, such as
		// the projection becoming known, must not leave the eyes apart) both are redone from the game's constants.
		classified_world = probe.apply_render_eye(scratch.data(), reloc, reloc_size, m_framebuffer_layout.width, m_framebuffer_layout.height, -1.f);
		vr_write_depth_remap(scratch.data() + size, remap_size, true);
		bool left_box = vr_box_scissor_rect(m_vr_mv_scissor[0]);
		probe.clear_box_mapped();
		const bool right_world = probe.apply_render_eye(scratch.data() + stride, reloc, reloc_size, m_framebuffer_layout.width, m_framebuffer_layout.height, 1.f);
		vr_write_depth_remap(scratch.data() + stride + size, remap_size, true);
		bool right_box = vr_box_scissor_rect(m_vr_mv_scissor[1]);
		if (right_world != classified_world)
		{
			static u32 s_redone = 0;
			if (s_redone++ < 8)
			{
				rsx_log.notice("VR multiview: eyes classified a draw of program %016llx differently (left world %d, right world %d); redoing the left eye",
					vr_vertex_program_hash(), classified_world, right_world);
			}
			fill_guest(scratch.data());
			probe.clear_box_mapped();
			classified_world = probe.apply_render_eye(scratch.data(), reloc, reloc_size, m_framebuffer_layout.width, m_framebuffer_layout.height, -1.f);
			vr_write_depth_remap(scratch.data() + size, remap_size, true);
			left_box = vr_box_scissor_rect(m_vr_mv_scissor[0]);
			probe.clear_box_mapped();
			fill_guest(scratch.data() + stride);
			probe.apply_render_eye(scratch.data() + stride, reloc, reloc_size, m_framebuffer_layout.width, m_framebuffer_layout.height, 1.f);
			vr_write_depth_remap(scratch.data() + stride + size, remap_size, true);
			right_box = vr_box_scissor_rect(m_vr_mv_scissor[1]);
		}
		if (!left_box)
		{
			m_vr_mv_scissor[0] = m_scissor;
		}
		if (!right_box)
		{
			m_vr_mv_scissor[1] = m_scissor;
		}
		m_vr_mv_box_scissor = left_box || right_box;
	}
	lap(1);

	if (static u32 s_why_logs = 0; why_this && s_why_logs < 6)
	{
		s_why_logs++;
		std::string text;
		const usz slots = size / 16;
		for (usz i = 0; i < slots; ++i)
		{
			const f32* g = reinterpret_cast<const f32*>(why_game.data() + i * 16);
			const f32* l = reinterpret_cast<const f32*>(scratch.data() + i * 16);
			const f32* r = reinterpret_cast<const f32*>(scratch.data() + stride + i * 16);
			if (std::memcmp(g, l, 16) != 0 || std::memcmp(g, r, 16) != 0)
			{
				fmt::append(text, "\n   c[%u] game (%g %g %g %g) left (%g %g %g %g) right (%g %g %g %g)", constant_ids.empty() ? static_cast<u32>(i) : constant_ids[i],
					g[0], g[1], g[2], g[3], l[0], l[1], l[2], l[3], r[0], r[1], r[2], r[3]);
			}
		}
		rsx_log.notice("VR why %016llx eye constants (keep game camera %d, world %d), changed slots:%s", vr_vertex_program_hash(), keep_game_camera, classified_world, text.empty() ? " none" : text);
	}

	const u64 allocation = m_transform_constants_allocator->alloc_bytes(stride * 2);
	void* destination = m_transform_constants_ring_info.map(allocation, stride * 2);
	std::memcpy(destination, scratch.data(), stride * 2);
	m_transform_constants_ring_info.unmap();

	m_xform_constants_dynamic_offset = allocation;
	m_vertex_constants_buffer_info = m_transform_constants_ring_info.window<16>(allocation, stride * 2,
		m_device->gpu().get_limits().maxUniformBufferRange);
	m_xform_constants_dynamic_offset -= m_vertex_constants_buffer_info.offset;
	m_vr_mv_right_xform_offset = m_xform_constants_dynamic_offset + stride;
	m_program->bind_uniform(m_vertex_constants_buffer_info, vk::glsl::binding_set_index_vertex,
		m_vs_binding_table->cbuf_location);
	d.eye_constants_bound = true;
	lap(2);
	return classified_world;
}

// bind_viewport(): multiview pipelines carry one viewport and scissor per view; only a HUD-box draw clips
// the eyes differently (vr_before_left_draw sets them again when a draw changes them mid-pass).
bool VKGSRender::vr_bind_viewport()
{
	if (!m_vr_multiview)
	{
		return false;
	}

	const VkViewport viewports[2] = {m_viewport, m_viewport};
	// Dev: RPCS3_VR_MV_SCISSOR=0 keeps the game's scissor for both views (no HUD-box clipping).
	static const bool s_box_scissor = []()
	{
		const char* v = std::getenv("RPCS3_VR_MV_SCISSOR");
		return !v || v[0] != '0';
	}();
	const bool box = m_vr_mv_box_scissor && s_box_scissor;
	const VkRect2D scissors[2] = {box ? m_vr_mv_scissor[0] : m_scissor, box ? m_vr_mv_scissor[1] : m_scissor};
	vkCmdSetViewport(*m_current_command_buffer, 0, 2, viewports);
	vkCmdSetScissor(*m_current_command_buffer, 0, 2, scissors);
	m_vr_mv_bound_scissor[0] = scissors[0];
	m_vr_mv_bound_scissor[1] = scissors[1];
	return true;
}

// emit_geometry(), after the render pass is bound: a pending guest query begins inside the multiview pass.
void VKGSRender::vr_after_render_pass_bound()
{
	if (m_vr_multiview && (m_current_command_buffer->flags & vk::command_buffer::cb_load_occluson_task))
	{
		vr_mv_begin_query_segment();
	}
}

// bind_texture_env(): multiview shaders sample 2D textures as arrays (the layer is the eye; a one-layer
// texture clamps to layer 0).
vk::image_view* VKGSRender::vr_array_view(vk::image_view* view, rsx::texture_dimension_extended dimension)
{
	return (m_vr_multiview && view && dimension == rsx::texture_dimension_extended::texture_dimension_2d) ? view->as_array() : view;
}

vk::image_view* VKGSRender::vr_array_view(vk::image_view* view)
{
	return (m_vr_multiview && view) ? view->as_array() : view;
}

VkImageViewType VKGSRender::vr_null_view_type(rsx::texture_dimension_extended dimension)
{
	// Upstream's view type (VKDraw.cpp), with the 2D case an array while multiview is on.
	const VkImageViewType type = vk::get_view_type(dimension);
	return (m_vr_multiview && type == VK_IMAGE_VIEW_TYPE_2D) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : type;
}

// clear_surface(): with multiview, a sub-viewport clear mapped into the HUD box clears each eye's own rectangle.
bool VKGSRender::vr_clear_attachments(const std::vector<VkClearAttachment>& clear_descriptors, const VkClearRect& region, const std::optional<areai>& right_clear)
{
	// Dev: RPCS3_VR_MV_EYECLEAR=0 keeps the one clear (the left eye's rectangle in both views).
	static const bool s_eye_clear = []()
	{
		const char* v = std::getenv("RPCS3_VR_MV_EYECLEAR");
		return !v || v[0] != '0';
	}();
	if (!m_vr_multiview || !right_clear || !s_eye_clear)
	{
		return false;
	}

	if (std::exchange(m_vr_clear_quads_valid, false))
	{
		vr_mv_clear_eye_quads(clear_descriptors);
		return true;
	}
	const VkClearRect right_region = {{{right_clear->x1, right_clear->y1}, {static_cast<u32>(right_clear->width()), static_cast<u32>(right_clear->height())}}, 0, 1};
	vr_mv_clear_eye_rects(clear_descriptors, region, right_region);
	return true;
}

void VKGSRender::vr_mv_clear_eye_quads(const std::vector<VkClearAttachment>& clear_descriptors)
{
	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}

	// The nearest depth for the clear's depth range (a far clear value means a LESS-style test).
	std::optional<VkClearAttachment> near_depth;
	for (const auto& d : clear_descriptors)
	{
		if (d.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
		{
			VkClearAttachment n = d;
			n.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			n.clearValue.depthStencil.depth = d.clearValue.depthStencil.depth >= 0.5f ? 0.f : 1.f;
			near_depth = n;
		}
	}

	const u32 width = m_draw_fbo->width();
	const u32 height = m_draw_fbo->height();
	constexpr u32 bands = 256;
	std::vector<VkClearRect> inside, outside;
	inside.reserve(bands);
	outside.reserve(bands * 2);
	for (u8 eye = 0; eye < 2; ++eye)
	{
		inside.clear();
		outside.clear();
		const auto& c = m_vr_clear_quads[eye];
		const f32 poly[4][2] = {{c[0][0], c[0][1]}, {c[1][0], c[1][1]}, {c[3][0], c[3][1]}, {c[2][0], c[2][1]}};
		f32 bx1 = poly[0][0], bx2 = poly[0][0], by1 = poly[0][1], by2 = poly[0][1];
		for (const auto& p : poly)
		{
			bx1 = std::min(bx1, p[0]); bx2 = std::max(bx2, p[0]);
			by1 = std::min(by1, p[1]); by2 = std::max(by2, p[1]);
		}
		const s32 ix1 = std::clamp(static_cast<s32>(std::floor(bx1)), 0, static_cast<s32>(width));
		const s32 ix2 = std::clamp(static_cast<s32>(std::ceil(bx2)), 0, static_cast<s32>(width));
		const s32 iy1 = std::clamp(static_cast<s32>(std::floor(by1)), 0, static_cast<s32>(height));
		const s32 iy2 = std::clamp(static_cast<s32>(std::ceil(by2)), 0, static_cast<s32>(height));
		if (ix2 <= ix1 || iy2 <= iy1)
		{
			continue;
		}
		const s32 step = std::max<s32>(1, (iy2 - iy1 + bands - 1) / bands);
		for (s32 y = iy1; y < iy2; y += step)
		{
			const s32 yb = std::min(y + step, iy2);
			// The quad's x range over this band: its vertices inside it and its edges' crossings of the band's two lines.
			f32 lo = FLT_MAX, hi = -FLT_MAX;
			for (u32 i = 0; i < 4; ++i)
			{
				const f32* a = poly[i];
				const f32* b = poly[(i + 1) % 4];
				if (a[1] >= y && a[1] <= yb)
				{
					lo = std::min(lo, a[0]); hi = std::max(hi, a[0]);
				}
				for (const f32 line : {static_cast<f32>(y), static_cast<f32>(yb)})
				{
					if ((a[1] - line) * (b[1] - line) <= 0.f && a[1] != b[1])
					{
						const f32 x = a[0] + (b[0] - a[0]) * (line - a[1]) / (b[1] - a[1]);
						lo = std::min(lo, x); hi = std::max(hi, x);
					}
				}
			}
			const s32 xl = lo <= hi ? std::clamp(static_cast<s32>(std::floor(lo)), ix1, ix2) : ix2;
			const s32 xr = lo <= hi ? std::clamp(static_cast<s32>(std::ceil(hi)), xl, ix2) : ix2;
			if (xr > xl)
				inside.push_back({{{xl, y}, {static_cast<u32>(xr - xl), static_cast<u32>(yb - y)}}, 0, 1});
			if (xl > ix1)
				outside.push_back({{{ix1, y}, {static_cast<u32>(xl - ix1), static_cast<u32>(yb - y)}}, 0, 1});
			if (ix2 > xr)
				outside.push_back({{{xr, y}, {static_cast<u32>(ix2 - xr), static_cast<u32>(yb - y)}}, 0, 1});
		}

		const u8 view_mask = (eye == 0) ? 2 : 3;
		const u64 key = vk::get_renderpass_key(m_fbo_images, {}, view_mask);
		VkRenderPass pass = vk::get_renderpass(*m_device, key);
		vk::framebuffer_holder* fbo = vk::get_framebuffer(*m_device, static_cast<u16>(width), static_cast<u16>(height), VK_FALSE, pass, m_fbo_images, view_mask, 0, 2);
		fbo->add_ref();
		vk::begin_renderpass(*m_current_command_buffer, pass, fbo->value, {positionu{0u, 0u}, sizeu{width, height}});
		if (!inside.empty())
			vkCmdClearAttachments(*m_current_command_buffer, ::size32(clear_descriptors), clear_descriptors.data(), ::size32(inside), inside.data());
		if (near_depth && !outside.empty())
			vkCmdClearAttachments(*m_current_command_buffer, 1, &*near_depth, ::size32(outside), outside.data());
		vk::end_renderpass(*m_current_command_buffer);
		fbo->release();
	}
	m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
}

// clear_surface(): the partial-colour-mask route (a quad through attachment_clear_pass) per eye, as vr_clear_attachments.
bool VKGSRender::vr_clear_attachments_masked(VkRect2D rect, u32 colormask, color4f color, const std::optional<areai>& right_clear)
{
	static const bool s_eye_clear = []()
	{
		const char* v = std::getenv("RPCS3_VR_MV_EYECLEAR");
		return !v || v[0] != '0';
	}();
	if (!m_vr_multiview || !right_clear || !s_eye_clear || !m_draw_fbo)
	{
		return false;
	}

	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}
	const u32 width = m_draw_fbo->width();
	const u32 height = m_draw_fbo->height();
	const VkRect2D right_rect = {{right_clear->x1, right_clear->y1}, {static_cast<u32>(right_clear->width()), static_cast<u32>(right_clear->height())}};
	const VkRect2D* rects[2] = {&rect, &right_rect};
	auto* clear_pass = vk::get_overlay_pass<vk::attachment_clear_pass>();
	for (u8 eye = 0; eye < 2; ++eye)
	{
		const u8 view_mask = (eye == 0) ? 2 : 3;
		const u64 key = vk::get_renderpass_key(m_fbo_images, {}, view_mask);
		VkRenderPass pass = vk::get_renderpass(*m_device, key);
		vk::framebuffer_holder* fbo = vk::get_framebuffer(*m_device, static_cast<u16>(width), static_cast<u16>(height), VK_FALSE, pass, m_fbo_images, view_mask, 0, 2);
		fbo->add_ref();
		clear_pass->run(*m_current_command_buffer, fbo, *rects[eye], colormask, color, pass);
		if (vk::is_renderpass_open(*m_current_command_buffer))
		{
			vk::end_renderpass(*m_current_command_buffer);
		}
		fbo->release();
	}
	m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
	return true;
}

u8 VKGSRender::vr_image_view_mask(vk::image* image)
{
	return (m_vr_multiview && image && image->stereo_layers && image->layers() > 1) ? 1 : 0;
}

// prepare_rtts(): every draw pass renders both views (the targets' two layers).
u8 VKGSRender::vr_draw_view_mask()
{
	m_vr_draw_view_mask = 0;
	if (!m_vr_multiview || m_fbo_images.empty())
	{
		return 0;
	}

	for (const auto* image : m_fbo_images)
	{
		if (!image->stereo_layers || image->layers() < 2)
		{
			static bool s_reported = false;
			if (!std::exchange(s_reported, true))
			{
				rsx_log.error("VR multiview: a bound render target has one layer; this pass draws the left eye only");
			}
			return 0;
		}
	}
	m_vr_draw_view_mask = 1;
	return 1;
}

// Multiview: a query counts both views and takes two consecutive slots. It is begun inside the pass and
// ended when the pass ends (vr_mv_end_query_segment), which is what the specification asks for. The guest
// reads the pair's average (query_pool_manager::get_query_result): the count of a camera between the eyes.
void VKGSRender::vr_mv_begin_query_segment()
{
	u32 occlusion_id = m_occlusion_query_manager->allocate_query_pair(*m_current_command_buffer);
	if (occlusion_id == umax)
	{
		rsx_log.warning("[Performance Warning] Out of free occlusion slots. Forcing hard sync.");
		ZCULL_control::sync(this);

		occlusion_id = m_occlusion_query_manager->allocate_query_pair(*m_current_command_buffer);
		if (occlusion_id == umax)
		{
			if (m_current_task) m_current_task->result = 1;
			return;
		}
	}

	// Allocation may have ended the pass (pool replacement, sync); the query must begin inside it.
	begin_render_pass();
	m_occlusion_query_manager->begin_query_pair(*m_current_command_buffer, occlusion_id);

	auto& data = m_occlusion_map[m_active_query_info->driver_handle];
	data.indices.push_back(occlusion_id);
	data.set_sync_command_buffer(m_current_command_buffer);
	m_vr_mv_open_query = occlusion_id;

	m_current_command_buffer->flags &= ~vk::command_buffer::cb_load_occluson_task;
	m_current_command_buffer->flags |= (vk::command_buffer::cb_has_occlusion_task | vk::command_buffer::cb_has_open_query);
}

void VKGSRender::vr_mv_end_query_segment(const vk::command_buffer& cmd)
{
	auto& primary = *m_current_command_buffer;
	if (!(primary.flags & vk::command_buffer::cb_has_open_query) || !m_active_query_info || m_vr_mv_open_query == umax ||
		static_cast<VkCommandBuffer>(cmd) != static_cast<VkCommandBuffer>(primary))
	{
		return;
	}

	m_occlusion_query_manager->end_query(primary, m_vr_mv_open_query);
	m_vr_mv_open_query = umax;
	primary.flags &= ~vk::command_buffer::cb_has_open_query;
	// The guest's query goes on: the next draw begins another pair inside its pass.
	primary.flags |= vk::command_buffer::cb_load_occluson_task;
}

// Two single-view passes over the same two-layer framebuffer: view 0 clears the left eye's rectangle, view 1 the right eye's.
void VKGSRender::vr_mv_clear_eye_rects(const std::vector<VkClearAttachment>& clear_descriptors, const VkClearRect& left, const VkClearRect& right)
{
	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}

	const u32 width = m_draw_fbo->width();
	const u32 height = m_draw_fbo->height();
	const VkClearRect* rects[2] = {&left, &right};
	for (u8 eye = 0; eye < 2; ++eye)
	{
		const s32 x1 = std::max<s32>(rects[eye]->rect.offset.x, 0);
		const s32 y1 = std::max<s32>(rects[eye]->rect.offset.y, 0);
		const s32 x2 = std::min<s32>(rects[eye]->rect.offset.x + static_cast<s32>(rects[eye]->rect.extent.width), static_cast<s32>(width));
		const s32 y2 = std::min<s32>(rects[eye]->rect.offset.y + static_cast<s32>(rects[eye]->rect.extent.height), static_cast<s32>(height));
		if (x2 <= x1 || y2 <= y1)
		{
			continue;
		}
		const VkClearRect clipped = {{{x1, y1}, {static_cast<u32>(x2 - x1), static_cast<u32>(y2 - y1)}}, 0, 1};

		const u8 view_mask = (eye == 0) ? 2 : 3;
		const u64 key = vk::get_renderpass_key(m_fbo_images, {}, view_mask);
		VkRenderPass pass = vk::get_renderpass(*m_device, key);
		vk::framebuffer_holder* fbo = vk::get_framebuffer(*m_device, static_cast<u16>(width), static_cast<u16>(height), VK_FALSE, pass, m_fbo_images, view_mask, 0, 2);
		fbo->add_ref();
		vk::begin_renderpass(*m_current_command_buffer, pass, fbo->value, {positionu{0u, 0u}, sizeu{width, height}});
		vkCmdClearAttachments(*m_current_command_buffer, ::size32(clear_descriptors), clear_descriptors.data(), 1, &clipped);
		vk::end_renderpass(*m_current_command_buffer);
		fbo->release();
	}

	m_current_command_buffer->flags |= vk::command_buffer::cb_reload_dynamic_state;
}

// Layer 1 of the display surface as a plain image, for the compositor, screenshots and the headset.
vk::viewable_image* VKGSRender::vr_mv_right_eye_image(vk::command_buffer& cmd, vk::viewable_image* stereo_image)
{
	if (!m_vr_mv_right_eye || m_vr_mv_right_eye->format() != stereo_image->format() ||
		m_vr_mv_right_eye->width() != stereo_image->width() || m_vr_mv_right_eye->height() != stereo_image->height())
	{
		if (m_vr_mv_right_eye)
		{
			vk::get_resource_manager()->dispose(m_vr_mv_right_eye);
		}

		m_vr_mv_right_eye = std::make_unique<vk::viewable_image>(*m_device, m_device->get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_IMAGE_TYPE_2D, stereo_image->format(), stereo_image->width(), stereo_image->height(), 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0, VMM_ALLOCATION_POOL_SYSTEM);
		m_vr_mv_right_eye->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		m_vr_mv_right_eye->set_debug_name("VR right eye (multiview layer 1)");
	}

	m_vr_mv_right_eye->native_component_map = stereo_image->native_component_map;
	const areai whole{0, 0, static_cast<s32>(stereo_image->width()), static_cast<s32>(stereo_image->height())};
	vk::copy_image(cmd, stereo_image, m_vr_mv_right_eye.get(), whole, whole, {.src_layer = 1, .dst_layer = 0});
	return m_vr_mv_right_eye.get();
}
