#include "stdafx.h"
#include "VKFrameGen.h"
#include "VKDLSS.h"
#include "VKResourceManager.h"
#include "vkutils/barriers.h"
#include "vkutils/device.h"
#include "vkutils/image.h"
#include "Emu/system_config.h"
#include "Utilities/StrFmt.h"

#ifdef HAVE_DLSS
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

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

#ifdef HAVE_DLSS
		NVSDK_NGX_FeatureDiscoveryInfo discovery_info(NVSDK_NGX_ProjectIdDescription& project, NVSDK_NGX_FeatureCommonInfo& common)
		{
			// As the DLSS context's NVSDK_NGX_VULKAN_Init_with_ProjectID (VKDLSS.cpp).
			project.ProjectId = "6b1f3c5e-2d4a-4e8b-9f17-3a5c7e9d1b20";
			project.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
			project.EngineVersion = "1.0";
			common = {};
			NVSDK_NGX_FeatureDiscoveryInfo info{};
			info.SDKVersion = NVSDK_NGX_Version_API;
			info.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
			info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
			info.Identifier.v.ProjectDesc = project;
			info.ApplicationDataPath = L".";
			info.FeatureInfo = &common;
			return info;
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

		const char* result_name(NVSDK_NGX_Result r)
		{
			static thread_local std::string s;
			s = fmt::format("0x%08x", static_cast<u32>(r));
			return s.c_str();
		}
#endif
	}

	bool framegen_requested()
	{
		static const bool s_requested = []()
		{
			if (const char* v = std::getenv("RPCS3_DLSSG"); v && *v)
			{
				return v[0] == '1';
			}
			return g_cfg.video.vr.frame_generation.get();
		}();
		return s_requested && dlss_ngx_possible();
	}

	void framegen_note_instance(VkInstance instance)
	{
		s_instance = instance;
	}

	void framegen_instance_extensions(std::vector<const char*>& extensions)
	{
#ifdef HAVE_DLSS
		if (!framegen_requested())
		{
			return;
		}
		NVSDK_NGX_ProjectIdDescription project{};
		NVSDK_NGX_FeatureCommonInfo common{};
		const NVSDK_NGX_FeatureDiscoveryInfo info = discovery_info(project, common);
		u32 count = 0;
		VkExtensionProperties* props = nullptr;
		if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &count, &props)))
		{
			framegen_log.warning("Frame generation: no instance extension requirements");
			return;
		}
		for (u32 i = 0; i < count; ++i)
		{
			framegen_log.notice("Frame generation needs instance extension %s", props[i].extensionName);
			add_extension(extensions, props[i].extensionName);
		}
#else
		static_cast<void>(extensions);
#endif
	}

	void framegen_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions)
	{
#ifdef HAVE_DLSS
		if (!framegen_requested() || !s_instance)
		{
			return;
		}
		NVSDK_NGX_ProjectIdDescription project{};
		NVSDK_NGX_FeatureCommonInfo common{};
		const NVSDK_NGX_FeatureDiscoveryInfo info = discovery_info(project, common);

		NVSDK_NGX_FeatureRequirement requirement{};
		if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureRequirements(s_instance, pdev, &info, &requirement)))
		{
			framegen_log.notice("Frame generation support on this GPU: 0x%x (0 = supported), min architecture 0x%x, min OS %s",
				static_cast<u32>(requirement.FeatureSupported), requirement.MinHWArchitecture, requirement.MinOSVersion);
		}

		u32 supported_count = 0;
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &supported_count, nullptr);
		std::vector<VkExtensionProperties> supported(supported_count);
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &supported_count, supported.data());

		u32 count = 0;
		VkExtensionProperties* props = nullptr;
		if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(s_instance, pdev, &info, &count, &props)))
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
#else
		static_cast<void>(pdev);
		static_cast<void>(extensions);
#endif
	}

	framegen_context::framegen_context(const vk::render_device& dev)
		: m_device(dev)
	{
	}

	framegen_context::~framegen_context()
	{
		release_features();
#ifdef HAVE_DLSS
		if (m_params)
		{
			NVSDK_NGX_VULKAN_DestroyParameters(static_cast<NVSDK_NGX_Parameter*>(m_params));
			m_params = nullptr;
		}
#endif
	}

	bool framegen_context::available()
	{
#ifdef HAVE_DLSS
		if (m_available >= 0)
		{
			return m_available == 1;
		}
		m_available = 0;
		NVSDK_NGX_Parameter* caps = nullptr;
		if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps)) || !caps)
		{
			framegen_log.error("Frame generation: NGX has no capability parameters (is DLSS on?)");
			return false;
		}
		int ok = 0, needs_driver = 0, init_result = 0;
		u32 max_count = 0;
		NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &ok);
		NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needs_driver);
		NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &init_result);
		NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &max_count);
		NVSDK_NGX_VULKAN_DestroyParameters(caps);
		if (!ok)
		{
			framegen_log.error("DLSS Frame Generation is not available (needs a newer driver: %d, init result 0x%08x): is nvngx_dlssg.dll next to rpcs3.exe?",
				needs_driver, static_cast<u32>(init_result));
			return false;
		}
		NVSDK_NGX_Parameter* params = nullptr;
		if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&params)) || !params)
		{
			framegen_log.error("Frame generation: NVSDK_NGX_VULKAN_AllocateParameters failed");
			return false;
		}
		m_params = params;
		framegen_log.success("DLSS Frame Generation available (up to %u generated frames per rendered frame)", std::max(1u, max_count));
		m_available = 1;
		return true;
#else
		return false;
#endif
	}

	void framegen_context::release_features()
	{
#ifdef HAVE_DLSS
		if (std::any_of(std::begin(m_features), std::end(m_features), [](const feature& f) { return f.handle != nullptr; }))
		{
			vkDeviceWaitIdle(m_device);
			for (auto& f : m_features)
			{
				if (f.handle)
				{
					NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(f.handle));
				}
				f = {};
			}
		}
#endif
	}

	bool framegen_context::ensure_feature(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in)
	{
#ifdef HAVE_DLSS
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
			NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(f.handle));
			f = {};
		}
		NVSDK_NGX_DLSSG_Create_Params create{};
		create.Width = w;
		create.Height = h;
		create.NativeBackbufferFormat = static_cast<unsigned int>(format);
		create.RenderWidth = rw;
		create.RenderHeight = rh;
		create.DynamicResolutionScaling = false;
		NVSDK_NGX_Handle* handle = nullptr;
		const NVSDK_NGX_Result r = NGX_VK_CREATE_DLSSG(cmd, 1, 1, &handle, static_cast<NVSDK_NGX_Parameter*>(m_params), &create);
		if (NVSDK_NGX_FAILED(r) || !handle)
		{
			framegen_log.error("Creating the frame generation feature failed (%s), %ux%u (render %ux%u) format %u", result_name(r), w, h, rw, rh, static_cast<u32>(format));
			m_disabled = true;
			return false;
		}
		f = {handle, w, h, rw, rh, format};
		m_have_previous[eye] = false;

		auto& out = m_output[eye];
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
			return false;
		}
		out->set_debug_name(eye ? "Frame generation output (right eye)" : "Frame generation output");
		framegen_log.notice("Frame generation feature%s %ux%u (render %ux%u), format %u", eye ? " (right eye)" : "", w, h, rw, rh, static_cast<u32>(format));
		return true;
#else
		static_cast<void>(cmd);
		static_cast<void>(eye);
		static_cast<void>(in);
		return false;
#endif
	}

	vk::viewable_image* framegen_context::evaluate(const vk::command_buffer& cmd, u32 eye, const framegen_eye_inputs& in, bool reset)
	{
#ifdef HAVE_DLSS
		if (m_disabled || eye > 1 || !in.backbuffer || !in.depth || !in.motion || !available() || !ensure_feature(cmd, eye, in))
		{
			return nullptr;
		}
		const bool first = !m_have_previous[eye] || reset;
		m_have_previous[eye] = true;

		vk::viewable_image* const out = m_output[eye].get();
		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

		in.backbuffer->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		in.depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		in.motion->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		out->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		const vk::image_view* bb_view = in.backbuffer->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
		const vk::image_view* depth_view = in.depth->get_view(identity);
		const vk::image_view* mv_view = in.motion->get_view(identity);
		const vk::image_view* out_view = out->get_view(identity);

		NVSDK_NGX_Resource_VK backbuffer = NVSDK_NGX_Create_ImageView_Resource_VK(bb_view->value, in.backbuffer->value, range, bb_view->info.format, in.backbuffer->width(), in.backbuffer->height(), false);
		NVSDK_NGX_Resource_VK depth = NVSDK_NGX_Create_ImageView_Resource_VK(depth_view->value, in.depth->value, range, VK_FORMAT_R32_SFLOAT, in.depth->width(), in.depth->height(), false);
		NVSDK_NGX_Resource_VK motion = NVSDK_NGX_Create_ImageView_Resource_VK(mv_view->value, in.motion->value, range, VK_FORMAT_R32G32_SFLOAT, in.motion->width(), in.motion->height(), false);
		NVSDK_NGX_Resource_VK output = NVSDK_NGX_Create_ImageView_Resource_VK(out_view->value, out->value, range, out->format(), out->width(), out->height(), true);

		NVSDK_NGX_VK_DLSSG_Eval_Params eval{};
		eval.pBackbuffer = &backbuffer;
		eval.pDepth = &depth;
		eval.pMVecs = &motion;
		eval.pOutputInterpFrame = &output;

		NVSDK_NGX_DLSSG_Opt_Eval_Params opt{};
		opt.multiFrameCount = 1;
		opt.multiFrameIndex = 1;
		opt.reset = first;
		opt.depthInverted = in.depth_inverted;
		opt.cameraMotionIncluded = true; // the motion vectors carry the camera's motion (VKDLSS: R = M_prev * M_cur^-1 per draw)
		opt.mvecScale[0] = 1.f;          // render pixels, as DLSS takes them (MVLowRes, scale 1)
		opt.mvecScale[1] = 1.f;
		opt.jitterOffset[0] = 0.f;       // the output of DLSS has no jitter left
		opt.jitterOffset[1] = 0.f;
		opt.colorBuffersHDR = false;
		opt.motionVectorsDilated = false;
		opt.menuDetectionEnabled = false;
		opt.notRenderingGameFrames = false;
		opt.orthoProjection = false;
		opt.motionVectorsInvalidValue = 0.f;

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
		to_ngx(proj, opt.cameraViewToClip);
		to_ngx(proj_inv, opt.clipToCameraView);
		f32 identity_m[16]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
		to_ngx(identity_m, opt.clipToLensClip);
		f32 prev_to_clip[16]{};
		if (in.have_motion_matrix && invert(in.clip_to_prev_clip, prev_to_clip))
		{
			to_ngx(in.clip_to_prev_clip, opt.clipToPrevClip);
			to_ngx(prev_to_clip, opt.prevClipToClip);
		}
		else
		{
			to_ngx(identity_m, opt.clipToPrevClip);
			to_ngx(identity_m, opt.prevClipToClip);
		}
		opt.cameraPos[0] = opt.cameraPos[1] = opt.cameraPos[2] = 0.f;
		opt.cameraRight[0] = 1.f; opt.cameraRight[1] = 0.f; opt.cameraRight[2] = 0.f;
		opt.cameraUp[0] = 0.f; opt.cameraUp[1] = 1.f; opt.cameraUp[2] = 0.f;
		opt.cameraFwd[0] = 0.f; opt.cameraFwd[1] = 0.f; opt.cameraFwd[2] = 1.f;
		opt.cameraNear = n;
		opt.cameraFar = f_;
		opt.cameraFOV = 2.f * std::atan(in.tan_half_y);
		opt.cameraAspectRatio = in.tan_half_y > 0.f ? in.tan_half_x / in.tan_half_y : 1.f;
		opt.mvecsSubrectSize = {in.motion->width(), in.motion->height()};
		opt.depthSubrectSize = {in.depth->width(), in.depth->height()};
		opt.backbufferSubrectSize = {in.backbuffer->width(), in.backbuffer->height()};
		opt.outputInterpSubrectSize = {out->width(), out->height()};

		const NVSDK_NGX_Result r = NGX_VK_EVALUATE_DLSSG(cmd, static_cast<NVSDK_NGX_Handle*>(m_features[eye].handle), static_cast<NVSDK_NGX_Parameter*>(m_params), &eval, &opt);
		in.motion->pop_layout(cmd);
		in.depth->pop_layout(cmd);
		in.backbuffer->pop_layout(cmd);
		vk::insert_global_memory_barrier(cmd);

		if (NVSDK_NGX_FAILED(r))
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
#else
		static_cast<void>(cmd);
		static_cast<void>(eye);
		static_cast<void>(in);
		static_cast<void>(reset);
		return nullptr;
#endif
	}
}
