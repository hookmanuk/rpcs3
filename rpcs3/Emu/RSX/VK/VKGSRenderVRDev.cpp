#include "stdafx.h"
#include <set>

// VR fork: development and diagnostic tools of VKGSRender (RPCS3_VR_GPUPROF, RPCS3_VR_RTDUMP,
// the per-frame trace, the probe's dev bits). Nothing here runs unless the matching
// environment variable or probe key is set.

#include "VKGSRender.h"
#include "VKOpenXR.h"
#include "VKHelpers.h"
#include "vkutils/buffer_object.h"

#include "../Capture/rsx_camera_probe.h"
#include "../rsx_methods.h"
#include "../Utils/rsx_utils.h"

#include "Emu/CPU/CPUThread.h"
#include "Emu/Memory/vm.h"

#include <vulkan/vulkan_core.h>

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

bool VKGSRender::gpuprof_enabled()
{
	if (m_gpuprof_enabled < 0)
	{
		const char* env = std::getenv("RPCS3_VR_GPUPROF");
		m_gpuprof_enabled = env && env[0] == '1';
		if (m_gpuprof_enabled)
		{
			VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
			info.queryType = VK_QUERY_TYPE_TIMESTAMP;
			info.queryCount = 3 * 1024;
			if (vkCreateQueryPool(*m_device, &info, nullptr, &m_gpuprof_pool) != VK_SUCCESS)
			{
				m_gpuprof_enabled = 0;
			}
			else
			{
				if (const char* target = std::getenv("RPCS3_VR_GPUPROF_TARGET"))
				{
					m_gpuprof_target = static_cast<u32>(std::strtoul(target, nullptr, 16));
				}
				rsx_log.success("GPU profile: GPU time per render target, logged every 120 frames (per draw into 0x%x).", m_gpuprof_target);
			}
		}
	}
	return m_gpuprof_enabled > 0;
}

void VKGSRender::gpuprof_mark(const gpuprof_mark_t& mark)
{
	auto& marks = m_gpuprof_marks[m_gpuprof_slot];
	if (marks.empty() || marks.size() >= 1023)
	{
		return; // before the first flip, or full
	}
	vkCmdWriteTimestamp(*m_current_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_gpuprof_pool, m_gpuprof_slot * 1024 + ::size32(marks));
	marks.push_back(mark);
	marks.back().draw = m_gpuprof_draw;
}

void VKGSRender::gpuprof_flip(const rsx::frame_statistics_t& stats)
{
	m_profiler.enabled = true; // RSX thread timings (normally only with the debug overlay)
	if (vk::is_renderpass_open(*m_current_command_buffer))
	{
		vk::end_renderpass(*m_current_command_buffer);
	}

	// End this frame: one more timestamp closes its last segment.
	auto& current = m_gpuprof_marks[m_gpuprof_slot];
	if (!current.empty())
	{
		vkCmdWriteTimestamp(*m_current_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_gpuprof_pool, m_gpuprof_slot * 1024 + ::size32(current));
		m_gpuprof_ended[m_gpuprof_slot] = true;
		m_gpuprof_end_draw[m_gpuprof_slot] = m_gpuprof_draw;
	}

	// Collect the frame recorded in the next slot (three flips ago), then reuse it.
	m_gpuprof_slot = (m_gpuprof_slot + 1) % 3;
	auto& marks = m_gpuprof_marks[m_gpuprof_slot];
	if (m_gpuprof_ended[m_gpuprof_slot] && !marks.empty())
	{
		std::vector<u64> ts(marks.size() + 1);
		if (vkGetQueryPoolResults(*m_device, m_gpuprof_pool, m_gpuprof_slot * 1024, ::size32(ts), ts.size() * sizeof(u64), ts.data(), sizeof(u64),
				VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
		{
			const f64 period_ms = m_device->gpu().get_limits().timestampPeriod / 1e6;
			for (usz i = 0; i < marks.size(); ++i)
			{
				const auto& m = marks[i];
				const u64 key = (u64{m.addr} << 32) ^ (u64{m.width} << 20) ^ (u64{m.height} << 8) ^ m.format;
				auto& sum = m_gpuprof_sum[key];
				sum.first += (ts[i + 1] - ts[i]) * period_ms;
				sum.second++;
				m_gpuprof_draws[key] += (i + 1 < marks.size() ? marks[i + 1].draw : m_gpuprof_end_draw[m_gpuprof_slot]) - m.draw;
				m_gpuprof_keys[key] = m;
			}
			m_gpuprof_total_ms += (ts.back() - ts.front()) * period_ms;
			if (++m_gpuprof_frames == 120)
			{
				std::vector<std::pair<f64, u64>> order;
				for (const auto& [key, sum] : m_gpuprof_sum)
				{
					order.emplace_back(sum.first, key);
				}
				std::sort(order.rbegin(), order.rend());
				std::string text = fmt::format("GPU profile: %.2f ms/frame over 120 frames (scale %u%%, %s); RSX thread in hard syncs %.2f ms/frame (%.1f/frame)",
					m_gpuprof_total_ms / 120, resolution_scaling_config.scale_percent, rsx::vr::camera_probe::get().render_enabled() ? "stereo" : "flat",
					m_gpuprof_sync_ms / 120, m_gpuprof_syncs / 120.);
				text += fmt::format("; %.0f draws/frame", m_gpuprof_draw_sum / 120.);
				text += fmt::format(" (%.0f right-eye batches/frame, %.1f right-eye texture rebuilds/frame)", m_gpuprof_batches / 120., m_gpuprof_right_copies / 120.);
				m_gpuprof_right_copies = 0;
				for (const auto& [kind, count] : m_gpuprof_right_copy_kinds)
				{
					rsx_log.notice("GPU profile: right-eye rebuild %s: %.1f/frame", kind, count / 120.);
				}
				m_gpuprof_right_copy_kinds.clear();
				m_gpuprof_batches = 0;
				m_gpuprof_draw_sum = 0;
				const auto rsx_ms = [&](int i)
				{
					return m_gpuprof_rsx_us[i] / 1000. / 120;
				};
				text += fmt::format("; RSX thread ms/frame: setup %.2f, vertex %.2f, textures %.2f, draw %.2f, flip %.2f = %.2f of %.2f between flips",
					rsx_ms(0), rsx_ms(1), rsx_ms(2), rsx_ms(3), rsx_ms(4), rsx_ms(0) + rsx_ms(1) + rsx_ms(2) + rsx_ms(3) + rsx_ms(4), m_gpuprof_wall_ms / 120);
				text += fmt::format("; VR on the RSX thread: left-eye constants %.2f, right-eye replay %.2f ms/frame (eye constants, both eyes: fill %.2f, apply %.2f, upload %.2f); RSX thread CPU %.2f ms/frame",
					m_gpuprof_left_vr_ms / 120, m_gpuprof_right_ms / 120, m_gpuprof_eye_ms[0] / 120, m_gpuprof_eye_ms[1] / 120, m_gpuprof_eye_ms[2] / 120, m_gpuprof_cpu_ms / 120);
				std::fill(std::begin(m_gpuprof_eye_ms), std::end(m_gpuprof_eye_ms), 0.);
				m_gpuprof_left_vr_ms = m_gpuprof_right_ms = m_gpuprof_cpu_ms = 0.;
				text += fmt::format("; flip() %.2f ms/frame, of which waiting for older frames' GPU work %.2f", m_gpuprof_flip_ms / 120, m_gpuprof_ctxwait_ms / 120);
				m_gpuprof_flip_ms = m_gpuprof_ctxwait_ms = 0.;
				std::fill(std::begin(m_gpuprof_rsx_us), std::end(m_gpuprof_rsx_us), 0);
				m_gpuprof_wall_ms = 0.;
				text += fmt::format("; guest blocked in GPU readbacks %.2f ms/frame (%.1f/frame, last at 0x%x)",
					m_gpuprof_readback_ns.exchange(0) / 1e6 / 120, m_gpuprof_readbacks.exchange(0) / 120., m_gpuprof_readback_addr.load());
				m_gpuprof_sync_ms = 0.;
				m_gpuprof_syncs = 0;
				for (usz i = 0; i < std::min<usz>(order.size(), 24); ++i)
				{
					const auto& m = m_gpuprof_keys[order[i].second];
					const auto& sum = m_gpuprof_sum[order[i].second];
					const f64 draws = m_gpuprof_draws[order[i].second] / 120.;
					text += m.format == umax ? fmt::format("\n  %7.3f ms  x%5.1f  d%6.1f  flip/present", sum.first / 120, sum.second / 120., draws) : fmt::format("\n  %7.3f ms  x%5.1f  d%6.1f  %08x %ux%u fmt 0x%x", sum.first / 120, sum.second / 120., draws, m.addr, m.width, m.height, m.format);
				}
				rsx_log.notice("%s", text);
				m_gpuprof_sum.clear();
				m_gpuprof_draws.clear();
				m_gpuprof_frames = 0;
				m_gpuprof_total_ms = 0.;
			}
		}
	}

	marks.clear();
	m_gpuprof_draw_sum += m_gpuprof_draw;
	m_gpuprof_rsx_us[0] += stats.setup_time;
	m_gpuprof_rsx_us[1] += stats.vertex_upload_time;
	m_gpuprof_rsx_us[2] += stats.textures_upload_time;
	m_gpuprof_rsx_us[3] += stats.draw_exec_time;
	m_gpuprof_rsx_us[4] += m_frame_stats.flip_time;
#ifdef _WIN32
	{
		FILETIME ctime, etime, ktime, utime;
		if (GetThreadTimes(GetCurrentThread(), &ctime, &etime, &ktime, &utime))
		{
			const u64 t = ((ktime.dwLowDateTime | static_cast<u64>(ktime.dwHighDateTime) << 32) + (utime.dwLowDateTime | static_cast<u64>(utime.dwHighDateTime) << 32));
			if (m_gpuprof_cpu_last)
				m_gpuprof_cpu_ms += (t - m_gpuprof_cpu_last) / 10000.;
			m_gpuprof_cpu_last = t;
		}
	}
#endif
	const auto now = std::chrono::steady_clock::now();
	if (m_gpuprof_last_flip != std::chrono::steady_clock::time_point{})
	{
		m_gpuprof_wall_ms += std::chrono::duration<f64, std::milli>(now - m_gpuprof_last_flip).count();
	}
	m_gpuprof_last_flip = now;
	m_gpuprof_draw = 0;
	m_gpuprof_ended[m_gpuprof_slot] = false;
	vkCmdResetQueryPool(*m_current_command_buffer, m_gpuprof_pool, m_gpuprof_slot * 1024, 1024);
	vkCmdWriteTimestamp(*m_current_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_gpuprof_pool, m_gpuprof_slot * 1024);
	marks.push_back({0, 0, 0, umax});
}

void VKGSRender::vr_trace_flush_cam()
{
	if (m_vr_trace_other_count)
	{
		m_vr_trace += fmt::format(" N x%u", m_vr_trace_other_count);
		m_vr_trace_other_count = 0;
	}
	if (m_vr_trace_cam_count)
	{
		m_vr_trace += fmt::format(" C%u x%u", m_vr_trace_cam_pose, m_vr_trace_cam_count);
		m_vr_trace_cam_count = 0;
	}
}

void VKGSRender::vr_rtdump(const std::vector<u32>& addresses, const std::string& tag)
{
	static const std::string s_rtdump = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_VR_RTDUMP");
		return v ? v : "";
	}();
	static u32 s_dump_index = 0;
	vr_batch_flush();
	std::string suffix;
	const auto dump = [&](vk::render_target* rt, const char* eye)
	{
		// 4- or 8-byte colour (8: RGBA16F, raw), or the depth aspect of a depth surface (D24: 24-bit depth in a 32-bit
		// word, D32F: float).
		const bool depth = rt && (rt->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT);
		const u32 texel = depth ? 4 : (rt ? vk::get_format_texel_width(rt->format()) : 0);
		if (!rt || (!depth && ((texel != 4 && texel != 8) || !(rt->aspect() & VK_IMAGE_ASPECT_COLOR_BIT))))
		{
			return;
		}
		const u32 w = rt->width(), h = rt->height();
		const usz size = usz{w} * h * texel;
		vk::buffer buffer(*m_device, utils::align(size, 0x100000), m_device->get_memory_mapping().host_visible_coherent,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, VMM_ALLOCATION_POOL_UNDEFINED);
		VkBufferImageCopy region{};
		region.bufferRowLength = w;
		region.imageSubresource = {static_cast<VkImageAspectFlags>(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
		region.imageExtent = {w, h, 1};
		if (vk::is_renderpass_open(*m_current_command_buffer))
			vk::end_renderpass(*m_current_command_buffer);
		rt->memory_barrier(*m_current_command_buffer, rsx::surface_access::transfer_read); // resolves an MSAA surface
		auto* image = rt->get_surface(rsx::surface_access::transfer_read);
		image->push_layout(*m_current_command_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		if (depth)
		{
			// Raw depth words (no D32F -> D16F conversion as in copy_image_to_buffer).
			vkCmdCopyImageToBuffer(*m_current_command_buffer, image->value, image->current_layout, buffer.value, 1, &region);
		}
		else
		{
			vk::copy_image_to_buffer(*m_current_command_buffer, image, &buffer, region);
		}
		image->pop_layout(*m_current_command_buffer);
		flush_command_queue(true);
		const auto src = buffer.map(0, size);
		fs::write_file(fmt::format("%s.%u%s.%s", s_rtdump, s_dump_index, suffix, eye), fs::rewrite, src, size);
		buffer.unmap();
		fs::write_file(fmt::format("%s.%u%s.%s.txt", s_rtdump, s_dump_index, suffix, eye), fs::rewrite, fmt::format("%u %u %d", w, h, static_cast<int>(image->format())));
	};
	for (const u32 a : addresses)
	{
		suffix = fmt::format(".%x", a);
		dump(m_rtts.get_surface_at(a), "left");
		dump(m_vr_right_rtts.get_surface_at(a), "right");
	}
	rsx_log.success("VR surface dump %u written (%s, %u surfaces)", s_dump_index, tag, ::size32(addresses));
	s_dump_index++;
}

// ---- Development hooks called from upstream functions -----------------------------------

// end(): GPU profiler draw marks.
void VKGSRender::vr_on_draw_begin()
{
	m_gpuprof_draw++;
	if (m_gpuprof_target && m_gpuprof_enabled > 0 && m_framebuffer_layout.color_addresses[0] == m_gpuprof_target)
	{
		// Per-draw segment: "fmt" 0x10000 + the draw's index in the frame.
		gpuprof_mark({m_gpuprof_target, m_framebuffer_layout.width, m_framebuffer_layout.height, 0x10000u + m_gpuprof_draw});
	}
}

// flip(), first thing: the pending right-eye batch, the surface dump request and the GPU profiler's frame end.
void VKGSRender::vr_flip_begin(const rsx::display_flip_info_t& info)
{
	// Gate 6: the right eye's last batched draws must land before it is presented.
	vr_batch_flush();

	// VR fork dev hook: RPCS3_VR_RTDUMP=<file>. When the file appears (consumed), both eyes' surfaces at
	// the display buffer are written raw to <file>.<n>.left / .right (+ .txt: width height format). It runs
	// before the swapchain checks, so it works with a locked desktop, where the flip skips present and
	// screenshots. The file may list other surface addresses (hex, one per line) to dump instead, as
	// <file>.<n>.<address>.left / .right.
	static const std::string s_rtdump = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_VR_RTDUMP");
		return v ? v : "";
	}();
	// A prog=<hash>#n request counts that program's draws from the start of a frame.
	if (m_vr_rtdump_program && !m_vr_rtdump_armed)
	{
		m_vr_rtdump_armed = true;
	}
	if (!s_rtdump.empty() && info.buffer < display_buffers_count && fs::is_file(s_rtdump))
	{
		std::string request;
		if (fs::file f{s_rtdump})
			request = f.to_string();
		fs::remove_file(s_rtdump);
		std::vector<u32> addresses;
		u64 program = 0;
		for (const std::string& line : fmt::split(request, {"\n", "\r", " ", ","}))
		{
			// prog=<vertex ucode hash>: dump just before that program's next draw instead of now.
			if (line.starts_with("prog="))
			{
				program = std::strtoull(line.c_str() + 5, nullptr, 16);
				const usz hash = line.find('#');
				m_vr_rtdump_skip = hash != umax ? std::max(1u, static_cast<u32>(std::strtoul(line.c_str() + hash + 1, nullptr, 10))) - 1 : 0;
				m_vr_rtdump_armed = hash == umax;
			}
			else if (const u32 a = static_cast<u32>(std::strtoul(line.c_str(), nullptr, 16)))
				addresses.push_back(a);
		}
		if (program)
		{
			m_vr_rtdump_program = program;
			m_vr_rtdump_addresses = std::move(addresses);
		}
		else
		{
			if (addresses.empty())
				addresses.push_back(rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL));
			vr_rtdump(addresses, "flip");
		}
	}

	if (gpuprof_enabled())
	{
		gpuprof_flip(info.stats);
	}
}

// on_access_violation(): sections the guest reads back (copied early from now on, see
// vr_before_prepare_rtts) and, for the GPU profiler, the time the guest thread spends blocked here.
VKGSRender::vr_readback_scope::vr_readback_scope(VKGSRender* renderer, const vk::texture_cache::thrashed_set& result, u32 address, bool is_writing)
	: r(renderer)
{
	if (rsx::vr::camera_probe::get().render_enabled())
	{
		std::lock_guard lock(r->m_vr_readback_mutex);
		for (const auto* section : result.sections_to_flush)
		{
			const auto range = section->get_section_range();
			if (std::find(r->m_vr_readback_ranges.begin(), r->m_vr_readback_ranges.end(), range) == r->m_vr_readback_ranges.end() && r->m_vr_readback_ranges.size() < 64)
			{
				r->m_vr_readback_ranges.push_back(range);
			}
		}
	}

	r->m_gpuprof_readback_addr = address;
	if (r->m_gpuprof_enabled > 0)
	{
		static atomic_t<u32> s_logged{0};
		if (s_logged++ < 6)
		{
			const auto* cpu = cpu_thread::get_current();
			rsx_log.notice("GPU profile: readback of 0x%x (%s) by %s", address, is_writing ? "write" : "read", cpu ? cpu->get_name() : std::string("host thread"));
		}
	}
}

VKGSRender::vr_readback_scope::~vr_readback_scope()
{
	if (r->m_gpuprof_enabled > 0)
	{
		r->m_gpuprof_readback_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
		r->m_gpuprof_readbacks++;
	}
}

// decode_rsx_state(): probe dev bit 0x100 (development): depth EQUAL becomes LEQUAL for draws that write colour.
rsx::comparison_function vk::vr_dev_depth_func(rsx::comparison_function func, bool writes_color)
{
	if (func == rsx::comparison_function::equal && (rsx::vr::camera_probe::get().dev_flags() & 0x100) && writes_color)
	{
		return rsx::comparison_function::less_or_equal;
	}
	return func;
}

// decode_rsx_state(): probe dev bit 0x200 (development): no stencil test for draws that write colour.
bool vk::vr_dev_skip_stencil_test(bool writes_color)
{
	return (rsx::vr::camera_probe::get().dev_flags() & 0x200) && writes_color;
}
