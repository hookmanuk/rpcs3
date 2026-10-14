// SPDX-License-Identifier: MIT
// Copyright (c) 2026 hookmanuk
//
// rpcs3_ngx.dll: RPCS3's (VR fork) optional NVIDIA DLSS plugin. Every NGX SDK call lives here; rpcs3.exe loads this
// DLL at run time through the table in rpcs3_ngx.h. Built against the NVIDIA DLSS SDK (https://github.com/NVIDIA/DLSS),
// whose library and runtime (nvngx_dlss.dll, nvngx_dlssg.dll) are NVIDIA's, under the NVIDIA RTX SDKs license.
// This file is MIT licensed (see LICENSE in this folder) and contains no RPCS3 code.

#include "rpcs3_ngx.h"

#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

#include <cstring>
#include <string>

namespace
{
	// The fork's NGX project (also the frame generation feature's discovery identity).
	constexpr const char* project_id = "6b1f3c5e-2d4a-4e8b-9f17-3a5c7e9d1b20";

	NVSDK_NGX_Parameter* g_caps = nullptr;     // NGX's capability parameters: DLSS's creation and evaluation
	NVSDK_NGX_Parameter* g_fg_params = nullptr; // frame generation's own
	rpcs3_ngx_log_fn g_log = nullptr;

	uint32_t result(NVSDK_NGX_Result r)
	{
		return static_cast<uint32_t>(r);
	}

	NVSDK_NGX_FeatureDiscoveryInfo framegen_discovery(NVSDK_NGX_ProjectIdDescription& project, NVSDK_NGX_FeatureCommonInfo& common)
	{
		project.ProjectId = project_id;
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

	NVSDK_NGX_Resource_VK resource(const rpcs3_ngx_image& image, bool read_write)
	{
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, range, image.format, image.width, image.height, read_write);
	}

	uint32_t dlss_required_extensions(uint32_t* ic, const char*** ie, uint32_t* dc, const char*** de)
	{
		unsigned int i = 0, d = 0;
		const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_RequiredExtensions(&i, ie, &d, de);
		*ic = i;
		*dc = d;
		return result(r);
	}

	uint32_t framegen_instance_extensions(uint32_t* count, VkExtensionProperties** props)
	{
		NVSDK_NGX_ProjectIdDescription project{};
		NVSDK_NGX_FeatureCommonInfo common{};
		const NVSDK_NGX_FeatureDiscoveryInfo info = framegen_discovery(project, common);
		return result(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, count, props));
	}

	uint32_t framegen_device_extensions(VkInstance instance, VkPhysicalDevice pdev, uint32_t* count, VkExtensionProperties** props)
	{
		NVSDK_NGX_ProjectIdDescription project{};
		NVSDK_NGX_FeatureCommonInfo common{};
		const NVSDK_NGX_FeatureDiscoveryInfo info = framegen_discovery(project, common);
		return result(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, pdev, &info, count, props));
	}

	uint32_t framegen_requirements(VkInstance instance, VkPhysicalDevice pdev, uint32_t* feature_supported, uint32_t* min_architecture, char* min_os, uint32_t min_os_size)
	{
		NVSDK_NGX_ProjectIdDescription project{};
		NVSDK_NGX_FeatureCommonInfo common{};
		const NVSDK_NGX_FeatureDiscoveryInfo info = framegen_discovery(project, common);
		NVSDK_NGX_FeatureRequirement requirement{};
		const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, pdev, &info, &requirement);
		if (!NVSDK_NGX_FAILED(r))
		{
			*feature_supported = static_cast<uint32_t>(requirement.FeatureSupported);
			*min_architecture = requirement.MinHWArchitecture;
			if (min_os && min_os_size)
			{
				std::strncpy(min_os, requirement.MinOSVersion, min_os_size - 1);
				min_os[min_os_size - 1] = 0;
			}
		}
		return result(r);
	}

	uint32_t init(const wchar_t* data_path, VkInstance instance, VkPhysicalDevice pdev, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
		PFN_vkGetDeviceProcAddr gdpa, rpcs3_ngx_log_fn log)
	{
		NVSDK_NGX_FeatureCommonInfo common{};
		g_log = log;
		if (log)
		{
			common.LoggingInfo.LoggingCallback = [](const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature feature)
			{
				if (g_log)
				{
					std::string text = "[" + std::to_string(static_cast<int>(feature)) + "] " + (message ? message : "");
					while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
					{
						text.pop_back();
					}
					g_log(text.c_str());
				}
			};
			common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
		}
		NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(project_id, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", data_path, instance, pdev, device,
			gipa, gdpa, log ? &common : nullptr);
		if (NVSDK_NGX_FAILED(r))
		{
			return result(r);
		}
		r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_caps);
		if (NVSDK_NGX_FAILED(r) || !g_caps)
		{
			g_caps = nullptr;
			NVSDK_NGX_VULKAN_Shutdown1(device);
			return NVSDK_NGX_FAILED(r) ? result(r) : result(NVSDK_NGX_Result_Fail);
		}
		return result(r);
	}

	void shutdown(VkDevice device)
	{
		if (g_fg_params)
		{
			NVSDK_NGX_VULKAN_DestroyParameters(g_fg_params);
			g_fg_params = nullptr;
		}
		if (g_caps)
		{
			NVSDK_NGX_VULKAN_DestroyParameters(g_caps);
			g_caps = nullptr;
			NVSDK_NGX_VULKAN_Shutdown1(device);
		}
		g_log = nullptr;
	}

	uint32_t dlss_capabilities(int32_t* available, int32_t* needs_driver, int32_t* init_result)
	{
		if (!g_caps)
		{
			return result(NVSDK_NGX_Result_FAIL_NotInitialized);
		}
		int a = 0, n = 0, i = 0;
		NVSDK_NGX_Parameter_GetI(g_caps, NVSDK_NGX_Parameter_SuperSampling_Available, &a);
		NVSDK_NGX_Parameter_GetI(g_caps, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &n);
		NVSDK_NGX_Parameter_GetI(g_caps, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &i);
		*available = a;
		*needs_driver = n;
		*init_result = i;
		return result(NVSDK_NGX_Result_Success);
	}

	uint32_t dlss_create(VkCommandBuffer cmd, const rpcs3_ngx_dlss_create* c, void** handle)
	{
		*handle = nullptr;
		if (!g_caps)
		{
			return result(NVSDK_NGX_Result_FAIL_NotInitialized);
		}
		NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_DLAA;
		switch (c->quality)
		{
		case RPCS3_NGX_QUALITY_QUALITY: quality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
		case RPCS3_NGX_QUALITY_BALANCED: quality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
		case RPCS3_NGX_QUALITY_PERFORMANCE: quality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
		case RPCS3_NGX_QUALITY_ULTRA_PERFORMANCE: quality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
		default: break;
		}

		NVSDK_NGX_DLSS_Create_Params create{};
		create.Feature.InWidth = c->in_width;
		create.Feature.InHeight = c->in_height;
		create.Feature.InTargetWidth = c->out_width;
		create.Feature.InTargetHeight = c->out_height;
		create.Feature.InPerfQualityValue = quality;
		create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
		if (c->depth_inverted)
		{
			create.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
		}
		create.InEnableOutputSubrects = false;

		// The hints stay on the shared parameters: written every time (0 asks for NGX's default again).
		for (const char* name : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
				 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
				 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance})
		{
			NVSDK_NGX_Parameter_SetUI(g_caps, name, c->preset);
		}

		NVSDK_NGX_Handle* h = nullptr;
		const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &h, g_caps, &create);
		if (!NVSDK_NGX_FAILED(r) && !h)
		{
			return result(NVSDK_NGX_Result_Fail);
		}
		*handle = h;
		return result(r);
	}

	uint32_t dlss_evaluate(VkCommandBuffer cmd, void* handle, const rpcs3_ngx_dlss_eval* e)
	{
		if (!g_caps || !handle)
		{
			return result(NVSDK_NGX_Result_FAIL_NotInitialized);
		}
		NVSDK_NGX_Resource_VK color = resource(e->color, false);
		NVSDK_NGX_Resource_VK motion = resource(e->motion, false);
		NVSDK_NGX_Resource_VK depth = resource(e->depth, false);
		NVSDK_NGX_Resource_VK output = resource(e->output, true);

		NVSDK_NGX_VK_DLSS_Eval_Params eval{};
		eval.Feature.pInColor = &color;
		eval.Feature.pInOutput = &output;
		eval.Feature.InSharpness = e->sharpness;
		eval.pInDepth = &depth;
		eval.pInMotionVectors = &motion;
		eval.InJitterOffsetX = e->jitter_x;
		eval.InJitterOffsetY = e->jitter_y;
		eval.InRenderSubrectDimensions = {e->render_width, e->render_height};
		eval.InReset = e->reset ? 1 : 0;
		eval.InMVScaleX = 1.f;
		eval.InMVScaleY = 1.f;
		eval.InColorSubrectBase = {0, 0};
		eval.InDepthSubrectBase = {0, 0};
		eval.InMVSubrectBase = {0, 0};
		eval.InOutputSubrectBase = {0, 0};
		eval.InPreExposure = 1.f;
		eval.InExposureScale = 1.f;
		return result(NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, static_cast<NVSDK_NGX_Handle*>(handle), g_caps, &eval));
	}

	uint32_t framegen_capabilities(int32_t* available, int32_t* needs_driver, int32_t* init_result, uint32_t* max_generated)
	{
		NVSDK_NGX_Parameter* caps = nullptr;
		NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps);
		if (NVSDK_NGX_FAILED(r) || !caps)
		{
			return NVSDK_NGX_FAILED(r) ? result(r) : result(NVSDK_NGX_Result_Fail);
		}
		int a = 0, n = 0, i = 0;
		unsigned int m = 0;
		NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &a);
		NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &n);
		NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &i);
		NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &m);
		NVSDK_NGX_VULKAN_DestroyParameters(caps);
		*available = a;
		*needs_driver = n;
		*init_result = i;
		*max_generated = m;
		if (a && !g_fg_params)
		{
			r = NVSDK_NGX_VULKAN_AllocateParameters(&g_fg_params);
			if (NVSDK_NGX_FAILED(r) || !g_fg_params)
			{
				g_fg_params = nullptr;
				*available = 0;
				return NVSDK_NGX_FAILED(r) ? result(r) : result(NVSDK_NGX_Result_Fail);
			}
		}
		return result(NVSDK_NGX_Result_Success);
	}

	uint32_t framegen_create(VkCommandBuffer cmd, const rpcs3_ngx_framegen_create* c, void** handle)
	{
		*handle = nullptr;
		if (!g_fg_params)
		{
			return result(NVSDK_NGX_Result_FAIL_NotInitialized);
		}
		NVSDK_NGX_DLSSG_Create_Params create{};
		create.Width = c->width;
		create.Height = c->height;
		create.NativeBackbufferFormat = static_cast<unsigned int>(c->format);
		create.RenderWidth = c->render_width;
		create.RenderHeight = c->render_height;
		create.DynamicResolutionScaling = false;
		NVSDK_NGX_Handle* h = nullptr;
		const NVSDK_NGX_Result r = NGX_VK_CREATE_DLSSG(cmd, 1, 1, &h, g_fg_params, &create);
		if (!NVSDK_NGX_FAILED(r) && !h)
		{
			return result(NVSDK_NGX_Result_Fail);
		}
		*handle = h;
		return result(r);
	}

	uint32_t framegen_evaluate(VkCommandBuffer cmd, void* handle, const rpcs3_ngx_framegen_eval* e)
	{
		if (!g_fg_params || !handle)
		{
			return result(NVSDK_NGX_Result_FAIL_NotInitialized);
		}
		NVSDK_NGX_Resource_VK backbuffer = resource(e->backbuffer, false);
		NVSDK_NGX_Resource_VK depth = resource(e->depth, false);
		NVSDK_NGX_Resource_VK motion = resource(e->motion, false);
		NVSDK_NGX_Resource_VK output = resource(e->output, true);
		NVSDK_NGX_Resource_VK hudless{};
		if (e->hudless.view)
		{
			hudless = resource(e->hudless, false);
		}

		NVSDK_NGX_VK_DLSSG_Eval_Params eval{};
		eval.pBackbuffer = &backbuffer;
		eval.pHudless = e->hudless.view ? &hudless : nullptr;
		eval.pDepth = &depth;
		eval.pMVecs = &motion;
		eval.pOutputInterpFrame = &output;

		NVSDK_NGX_DLSSG_Opt_Eval_Params opt{};
		opt.multiFrameCount = e->multi_frame_count ? e->multi_frame_count : 1;
		opt.multiFrameIndex = e->multi_frame_index;
		opt.reset = e->reset != 0;
		opt.depthInverted = e->depth_inverted != 0;
		opt.cameraMotionIncluded = true; // the motion vectors carry the camera's motion
		opt.mvecScale[0] = 1.f;          // render pixels
		opt.mvecScale[1] = 1.f;
		opt.jitterOffset[0] = 0.f;       // the output of DLSS has no jitter left
		opt.jitterOffset[1] = 0.f;
		opt.colorBuffersHDR = false;
		opt.motionVectorsDilated = false;
		opt.menuDetectionEnabled = false;
		opt.notRenderingGameFrames = false;
		opt.orthoProjection = false;
		opt.motionVectorsInvalidValue = e->motion_vectors_invalid_value;
		std::memcpy(opt.cameraViewToClip, e->camera_view_to_clip, sizeof(opt.cameraViewToClip));
		std::memcpy(opt.clipToCameraView, e->clip_to_camera_view, sizeof(opt.clipToCameraView));
		std::memcpy(opt.clipToLensClip, e->clip_to_lens_clip, sizeof(opt.clipToLensClip));
		std::memcpy(opt.clipToPrevClip, e->clip_to_prev_clip, sizeof(opt.clipToPrevClip));
		std::memcpy(opt.prevClipToClip, e->prev_clip_to_clip, sizeof(opt.prevClipToClip));
		opt.cameraPos[0] = opt.cameraPos[1] = opt.cameraPos[2] = 0.f;
		opt.cameraRight[0] = 1.f; opt.cameraRight[1] = 0.f; opt.cameraRight[2] = 0.f;
		opt.cameraUp[0] = 0.f; opt.cameraUp[1] = 1.f; opt.cameraUp[2] = 0.f;
		opt.cameraFwd[0] = 0.f; opt.cameraFwd[1] = 0.f; opt.cameraFwd[2] = 1.f;
		opt.cameraNear = e->camera_near;
		opt.cameraFar = e->camera_far;
		opt.cameraFOV = e->camera_fov;
		opt.cameraAspectRatio = e->camera_aspect;
		opt.mvecsSubrectSize = {e->motion.width, e->motion.height};
		opt.depthSubrectSize = {e->depth.width, e->depth.height};
		opt.backbufferSubrectSize = {e->backbuffer.width, e->backbuffer.height};
		if (e->hudless.view)
		{
			opt.hudLessSubrectSize = {e->hudless.width, e->hudless.height};
		}
		opt.outputInterpSubrectSize = {e->output.width, e->output.height};
		return result(NGX_VK_EVALUATE_DLSSG(cmd, static_cast<NVSDK_NGX_Handle*>(handle), g_fg_params, &eval, &opt));
	}

	void release(void* handle)
	{
		if (handle)
		{
			NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(handle));
		}
	}

	const rpcs3_ngx_api g_api =
	{
		RPCS3_NGX_API_VERSION,
		dlss_required_extensions,
		framegen_instance_extensions,
		framegen_device_extensions,
		framegen_requirements,
		init,
		shutdown,
		dlss_capabilities,
		dlss_create,
		dlss_evaluate,
		framegen_capabilities,
		framegen_create,
		framegen_evaluate,
		release,
	};
}

extern "C" __declspec(dllexport) const rpcs3_ngx_api* rpcs3_ngx_get_api()
{
	return &g_api;
}
