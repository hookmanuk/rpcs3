#include "stdafx.h"
#include "VKFrameGen.h"

#include <limits>
#include "VKDLSS.h"
#include "VKResourceManager.h"
#include "vkutils/barriers.h"
#include "vkutils/device.h"
#include "vkutils/image.h"
#include "Emu/system_config.h"
#include "Emu/RSX/Capture/rsx_camera_probe.h"
#include "Utilities/StrFmt.h"
#include "Utilities/File.h"

// DLSS-G runs in the optional plugin rpcs3_ngx.dll (MIT, rpcs3/ngx_plugin): this file has no NVIDIA code.
#include "../../../ngx_plugin/rpcs3_ngx.h"

#include <cstring>

LOG_CHANNEL(framegen_log, "FrameGen");

namespace vk
{
	namespace
	{
		VkInstance s_instance = VK_NULL_HANDLE;

		void add_extension(std::vector<const char*>& extensions, const char* name)
		{
			if (std::none_of(extensions.begin(), extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; }))
			{
				extensions.push_back(name);
			}
		}

		// Row-major, column vectors (ours) to NGX's row-major post-multiplication: the transpose.
		void to_ngx(const f32 (&m)[16], float (&out)[4][4])
		{
			for (u32 r = 0; r < 4; ++r)
				for (u32 c = 0; c < 4; ++c)
					out[r][c] = m[c * 4 + r];
		}

		bool invert(const f32 (&m)[16], f32 (&out)[16])
		{
			f64 a[4][8];
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 c = 0; c < 4; ++c)
				{
					a[r][c] = m[r * 4 + c];
					a[r][c + 4] = r == c ? 1.0 : 0.0;
				}
			}
			for (u32 c = 0; c < 4; ++c)
			{
				u32 pivot = c;
				for (u32 r = c + 1; r < 4; ++r)
					if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
						pivot = r;
				if (std::fabs(a[pivot][c]) < 1e-12)
					return false;
				if (pivot != c)
					for (u32 k = 0; k < 8; ++k)
						std::swap(a[c][k], a[pivot][k]);
				const f64 inv = 1.0 / a[c][c];
				for (u32 k = 0; k < 8; ++k)
					a[c][k] *= inv;
				for (u32 r = 0; r < 4; ++r)
				{
					if (r == c || a[r][c] == 0.0)
						continue;
					const f64 f = a[r][c];
					for (u32 k = 0; k < 8; ++k)
						a[r][k] -= f * a[c][k];
				}
			}
			for (u32 r = 0; r < 4; ++r)
				for (u32 c = 0; c < 4; ++c)
					out[r * 4 + c] = static_cast<f32>(a[r][c + 4]);
			return true;
		}

		const char* result_name(u32 r)
		{
			static thread_local std::string s;
			s = fmt::format("0x%08x", r);
			return s.c_str();
		}
	}

	framegen_motion framegen_motion_mode()
	{
		static const framegen_motion s_env = []()
		{
			const char* v = std::getenv("RPCS3_DLSSG_MV");
			const std::string_view s = v ? v : "";
			return s == "own" || s == "invalid" ? framegen_motion::own : s == "game" ? framegen_motion::game : framegen_motion::hybrid;
		}();
		if (std::getenv("RPCS3_DLSSG_MV"))
		{
			return s_env;
		}
		// The setting (VR > DLSS Frame Generation Motion, live); the dev files for headset tests win (checked every 90 calls).
		static framegen_motion s_mode = framegen_motion::hybrid;
		if (static u32 s_calls = 0; s_calls++ % 90 == 0)
		{
			const char* temp = std::getenv("TEMP");
			const std::string dir = temp ? std::string(temp) + "/rpcs3-vrprofile/" : std::string();
			const framegen_motion setting = g_cfg.video.vr.frame_generation_motion == vr_framegen_motion::game ? framegen_motion::game : framegen_motion::hybrid;
			const framegen_motion now = temp && fs::is_file(dir + "DLSSG_MV_OWN") ? framegen_motion::own : temp && fs::is_file(dir + "DLSSG_MV_GAME") ? framegen_motion::game : setting;
			if (now != s_mode)
			{
				framegen_log.notice("Frame generation motion: %s", now == framegen_motion::own ? "DLSS-G's own estimate" : now == framegen_motion::game ? "the game's vectors" : "hybrid (the game's vectors on static world)");
				s_mode = now;
			}
		}
		return s_mode;
	}

	bool framegen_possible()
	{
		const char* v = std::getenv("RPCS3_DLSSG");
		return !(v && v[0] == '0') && dlss_ngx_possible();
	}

	bool framegen_enabled()
	{
		if (const char* v = std::getenv("RPCS3_DLSSG"); v && *v)
		{
			return v[0] == '1';
		}
		return g_cfg.video.vr.frame_generation.get();
	}

	void framegen_note_instance(VkInstance instance)
	{
		s_instance = instance;
	}

	void framegen_instance_extensions(std::vector<const char*>& extensions)
	{
		if (!framegen_possible())
		{
			return;
		}
		u32 count = 0;
		VkExtensionProperties* props = nullptr;
		if (RPCS3_NGX_FAILED(ngx_plugin()->framegen_instance_extensions(&count, &props)))
		{
			framegen_log.warning("Frame generation: no instance extension requirements");
			return;
		}
		for (u32 i = 0; i < count; ++i)
		{
			framegen_log.notice("Frame generation needs instance extension %s", props[i].extensionName);
			add_extension(extensions, props[i].extensionName);
		}
	}

	void framegen_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions)
	{
		if (!framegen_possible() || !s_instance)
		{
			return;
		}
		const rpcs3_ngx_api* const ngx = ngx_plugin();

		u32 feature_supported = 0, min_architecture = 0;
		char min_os[256]{};
		if (!RPCS3_NGX_FAILED(ngx->framegen_requirements(s_instance, pdev, &feature_supported, &min_architecture, min_os, sizeof(min_os))))
		{
			framegen_log.notice("Frame generation support on this GPU: 0x%x (0 = supported), min architecture 0x%x, min OS %s",
				feature_supported, min_architecture, min_os);
		}

		u32 supported_count = 0;
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &supported_count, nullptr);
		std::vector<VkExtensionProperties> supported(supported_count);
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &supported_count, supported.data());

		u32 count = 0;
		VkExtensionProperties* props = nullptr;
		if (RPCS3_NGX_FAILED(ngx->framegen_device_extensions(s_instance, pdev, &count, &props)))
		{
			framegen_log.warning("Frame generation: no device extension requirements");
			return;
		}
		for (u32 i = 0; i < count; ++i)
		{
			const char* name = props[i].extensionName;
			if (std::strcmp(name, "VK_EXT_buffer_device_address") == 0)
			{
				continue; // Vulkan 1.2's bufferDeviceAddress (dlss_device_chain) replaces it
			}
			const bool ok = std::any_of(supported.begin(), supported.end(), [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
			framegen_log.notice("Frame generation needs device extension %s%s", name, ok ? "" : " (not supported)");
			if (ok)
			{
				// NGX's list outlives the call (static storage in the SDK).
				add_extension(extensions, name);
			}
		}
	}

	framegen_context::framegen_context(const vk::render_device& dev)
		: m_device(dev)
	{
	}

	framegen_context::~framegen_context()
	{
		rsx::vr::set_framegen_active(false);
		release_features(); // the plugin's own parameters go with the NGX session (VKDLSS's shutdown)
	}

	bool framegen_context::available()
	{
		if (m_available >= 0)
		{
			return m_available == 1;
		}
		m_available = 0;
		const rpcs3_ngx_api* const ngx = ngx_plugin();
		if (!ngx)
		{
			return false;
		}
		s32 ok = 0, needs_driver = 0, init_result = 0;
		u32 max_count = 0;
		if (const u32 r = ngx->framegen_capabilities(&ok, &needs_driver, &init_result, &max_count); RPCS3_NGX_FAILED(r))
		{
			framegen_log.error("Frame generation: NGX has no capability parameters (is DLSS on?) (%s)", result_name(r));
			return false;
		}
		if (!ok)
		{
			framegen_log.error("DLSS Frame Generation is not available (needs a newer driver: %d, init result 0x%08x): is nvngx_dlssg.dll next to rpcs3.exe?",
				needs_driver, static_cast<u32>(init_result));
			return false;
		}
		framegen_log.success("DLSS Frame Generation available (up to %u generated frames per rendered frame)", std::max(1u, max_count));
		m_max_frames = std::max(1u, max_count);
		rsx::vr::set_framegen_max_generated(std::min<u32>(m_max_frames, static_cast<u32>(std::size(m_output[0]))));
		m_available = 1;
		return true;
	}

	void framegen_context::disable()
	{
		release_features();
		reset();
		rsx::vr::set_framegen_active(false);
	}


	void framegen_context::release_features()
	{
		if (std::any_of(std::begin(m_features), std::end(m_features), [](const feature& f) { return f.handle != nullptr; }))
		{
			vkDeviceWaitIdle(m_device);
			for (auto& f : m_features)
			{
				if (f.handle)
				{
					ngx_plugin()->release(f.handle);
				}
				f = {};
			}
		}
	}

	bool framegen_context::ensure_feature(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in)
	{
		auto& f = m_features[eye];
		const u32 w = in.backbuffer->width(), h = in.backbuffer->height();
		const u32 rw = in.depth->width(), rh = in.depth->height();
		const VkFormat format = in.backbuffer->format();
		if (f.handle && f.width == w && f.height == h && f.render_width == rw && f.render_height == rh && f.format == format)
		{
			return true;
		}
		if (f.handle)
		{
			vkDeviceWaitIdle(m_device);
			ngx_plugin()->release(f.handle);
			f = {};
		}
		const rpcs3_ngx_framegen_create create{w, h, format, rw, rh};
		void* handle = nullptr;
		const u32 r = ngx_plugin()->framegen_create(cmd, &create, &handle);
		if (RPCS3_NGX_FAILED(r) || !handle)
		{
			framegen_log.error("Creating the frame generation feature failed (%s), %ux%u (render %ux%u) format %u", result_name(r), w, h, rw, rh, static_cast<u32>(format));
			m_disabled = true;
			rsx::vr::set_framegen_active(false);
			return false;
		}
		f = {handle, w, h, rw, rh, format};
		m_have_previous[eye] = false;

		for (auto& out : m_output[eye])
		{
			if (out)
			{
				vk::get_resource_manager()->dispose(out);
			}
			const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			out = std::make_unique<vk::viewable_image>(m_device, m_device.get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D, format, w, h, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL, usage,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3, VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_COLOR);
			if (!out->value)
			{
				framegen_log.error("Frame generation: no %ux%u output image", w, h);
				m_disabled = true;
				rsx::vr::set_framegen_active(false);
				return false;
			}
			out->set_debug_name(eye ? "Frame generation output (right eye)" : "Frame generation output");
		}
		rsx::vr::set_framegen_active(true); // the game at 1/multiplier of the VR frame rate from here on
		framegen_log.notice("Frame generation feature%s %ux%u (render %ux%u), format %u", eye ? " (right eye)" : "", w, h, rw, rh, static_cast<u32>(format));
		return true;
	}

	u32 framegen_context::multiplier() const
	{
		const u32 wanted = g_cfg.video.vr.frame_generation_multiplier == vr_framegen_multiplier::x3 ? 3 : 2;
		return std::min(wanted, std::min<u32>(m_max_frames, static_cast<u32>(std::size(m_output[0]))) + 1);
	}

	vk::viewable_image* framegen_context::evaluate(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in, bool reset, u32 index, u32 count)
	{
		if (m_disabled || eye > 1 || !in.backbuffer || !in.depth || !in.motion || !available() || !ensure_feature(cmd, eye, in))
		{
			return nullptr;
		}
		index = std::clamp<u32>(index, 1, static_cast<u32>(std::size(m_output[eye])));
		if (index == 1)
		{
			m_first[eye] = !m_have_previous[eye] || reset;
			m_have_previous[eye] = true;
		}
		const bool first = m_first[eye];

		vk::viewable_image* const out = m_output[eye][index - 1].get();
		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

		// framegen_motion::own: every vector marked invalid (DLSS-G's own estimate). Our vectors give objects the game moves
		// itself (MotorStorm's car, built on the SPU) the world's motion and DLSS-G ghosted their edges; its own estimate
		// ghosted every car on head turns (Matt, headset, 2026-10-13). Hybrid (the default, VKDLSS framegen_inputs) does both.
		vk::viewable_image* motion_img = in.motion;
		if (framegen_motion_mode() == framegen_motion::own)
		{
			if (!m_invalid_mv || m_invalid_mv->width() != in.motion->width() || m_invalid_mv->height() != in.motion->height())
			{
				if (m_invalid_mv)
				{
					vk::get_resource_manager()->dispose(m_invalid_mv);
				}
				m_invalid_mv = std::make_unique<vk::viewable_image>(m_device, m_device.get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32_SFLOAT, in.motion->width(), in.motion->height(), 1, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_COLOR);
			}
			m_invalid_mv->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			const VkClearColorValue invalid{.float32 = {std::numeric_limits<f32>::max(), std::numeric_limits<f32>::max(), 0.f, 0.f}};
			vkCmdClearColorImage(cmd, m_invalid_mv->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &invalid, 1, &range);
			motion_img = m_invalid_mv.get();
		}

		vk::viewable_image* const hudless = in.hudless && in.hudless->width() == in.backbuffer->width() && in.hudless->height() == in.backbuffer->height() &&
			in.hudless->format() == in.backbuffer->format() ? in.hudless : nullptr;
		in.backbuffer->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		if (hudless)
		{
			hudless->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		in.depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		motion_img->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		out->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		const vk::image_view* bb_view = in.backbuffer->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
		const vk::image_view* depth_view = in.depth->get_view(identity);
		const vk::image_view* mv_view = motion_img->get_view(identity);
		const vk::image_view* out_view = out->get_view(identity);

		rpcs3_ngx_framegen_eval opt{};
		opt.backbuffer = {bb_view->value, in.backbuffer->value, bb_view->info.format, in.backbuffer->width(), in.backbuffer->height()};
		opt.depth = {depth_view->value, in.depth->value, VK_FORMAT_R32_SFLOAT, in.depth->width(), in.depth->height()};
		opt.motion = {mv_view->value, motion_img->value, VK_FORMAT_R32G32_SFLOAT, motion_img->width(), motion_img->height()};
		opt.output = {out_view->value, out->value, out->format(), out->width(), out->height()};
		if (hudless)
		{
			const vk::image_view* hl_view = hudless->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
			opt.hudless = {hl_view->value, hudless->value, hl_view->info.format, hudless->width(), hudless->height()};
		}

		// The plugin passes these on (camera motion included in the vectors: VKDLSS's R = M_prev * M_cur^-1 per draw;
		// render pixels at scale 1, as DLSS takes them; no jitter left after DLSS).
		opt.multi_frame_count = std::max(count, 1u);
		opt.multi_frame_index = index;
		opt.reset = first ? 1 : 0;
		opt.depth_inverted = in.depth_inverted ? 1 : 0;
		opt.motion_vectors_invalid_value = std::numeric_limits<f32>::max(); // never written (0 marked every still pixel invalid: the car under a chase camera ghosted)

		// The eye's projection: symmetric half tangents, a nominal depth range (the depth flag is what DLSS-G's output
		// depends on most; the motion vectors carry the motion).
		constexpr f32 n = 0.1f, f_ = 10000.f;
		f32 proj[16]{};
		proj[0] = 1.f / std::max(in.tan_half_x, 1e-3f);
		proj[5] = 1.f / std::max(in.tan_half_y, 1e-3f);
		proj[10] = f_ / (f_ - n);
		proj[11] = -n * f_ / (f_ - n);
		proj[14] = 1.f;
		f32 proj_inv[16]{};
		invert(proj, proj_inv);
		to_ngx(proj, opt.camera_view_to_clip);
		to_ngx(proj_inv, opt.clip_to_camera_view);
		f32 identity_m[16]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
		to_ngx(identity_m, opt.clip_to_lens_clip);
		f32 prev_to_clip[16]{};
		if (in.have_motion_matrix && invert(in.clip_to_prev_clip, prev_to_clip))
		{
			to_ngx(in.clip_to_prev_clip, opt.clip_to_prev_clip);
			to_ngx(prev_to_clip, opt.prev_clip_to_clip);
		}
		else
		{
			to_ngx(identity_m, opt.clip_to_prev_clip);
			to_ngx(identity_m, opt.prev_clip_to_clip);
		}
		opt.camera_near = n;
		opt.camera_far = f_;
		opt.camera_fov = 2.f * std::atan(in.tan_half_y);
		opt.camera_aspect = in.tan_half_y > 0.f ? in.tan_half_x / in.tan_half_y : 1.f;
		if (static bool s_logged = false; hudless && !std::exchange(s_logged, true))
		{
			framegen_log.notice("Frame generation: HUD-less input on (the VR HUD box's separation)");
		}

		const u32 r = ngx_plugin()->framegen_evaluate(cmd, m_features[eye].handle, &opt);
		motion_img->pop_layout(cmd);
		in.depth->pop_layout(cmd);
		if (hudless)
		{
			hudless->pop_layout(cmd);
		}
		in.backbuffer->pop_layout(cmd);
		vk::insert_global_memory_barrier(cmd);

		if (RPCS3_NGX_FAILED(r))
		{
			static u32 s_failures = 0;
			if (s_failures++ < 10)
			{
				framegen_log.error("Frame generation evaluation failed (%s)", result_name(r));
			}
			return nullptr;
		}
		static bool s_logged = false;
		if (!s_logged)
		{
			s_logged = true;
			framegen_log.success("First generated frame (eye %u)", eye);
		}
		return first ? nullptr : out;
	}
}
