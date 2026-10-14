// SPDX-License-Identifier: MIT
// Copyright (c) 2026 hookmanuk
//
// rpcs3_ngx: the interface between RPCS3 (VR fork) and its optional NVIDIA DLSS plugin, rpcs3_ngx.dll.
//
// The plugin holds every call into NVIDIA's NGX SDK (DLSS Super Resolution and DLSS Frame Generation, Vulkan), so
// rpcs3.exe contains no NVIDIA code: it loads rpcs3_ngx.dll from its own folder when present and calls the table
// rpcs3_ngx_get_api() returns. Without the plugin, DLSS is unavailable and its settings are hidden.
//
// This header is shared by both sides and is MIT licensed (see LICENSE in this folder). Plain C types only.
#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RPCS3_NGX_API_VERSION 1

// An NVSDK_NGX_Result: failures have the top 12 bits 0xBAD.
#define RPCS3_NGX_FAILED(r) ((((uint32_t)(r)) & 0xFFF00000u) == 0xBAD00000u)

// One image for NGX: a 2D single-layer, single-mip colour view.
typedef struct rpcs3_ngx_image
{
	VkImageView view;
	VkImage image;
	VkFormat format;
	uint32_t width;
	uint32_t height;
} rpcs3_ngx_image;

typedef void (*rpcs3_ngx_log_fn)(const char* message);

// DLSS Super Resolution quality modes.
enum
{
	RPCS3_NGX_QUALITY_DLAA = 0,
	RPCS3_NGX_QUALITY_QUALITY = 1,
	RPCS3_NGX_QUALITY_BALANCED = 2,
	RPCS3_NGX_QUALITY_PERFORMANCE = 3,
	RPCS3_NGX_QUALITY_ULTRA_PERFORMANCE = 4,
};

typedef struct rpcs3_ngx_dlss_create
{
	uint32_t in_width, in_height;   // render size
	uint32_t out_width, out_height; // output size
	uint32_t quality;               // RPCS3_NGX_QUALITY_*
	uint32_t preset;                // render preset 1 (A) .. 15 (O) for every mode, 0: NGX's default
	uint32_t depth_inverted;        // 1: 1 is near
} rpcs3_ngx_dlss_create;

typedef struct rpcs3_ngx_dlss_eval
{
	rpcs3_ngx_image color;  // render size (or larger: the render size is the subrect at 0,0)
	rpcs3_ngx_image motion; // RG32F, render pixels, current to previous
	rpcs3_ngx_image depth;  // R32F
	rpcs3_ngx_image output; // written (storage)
	float sharpness;
	float jitter_x, jitter_y; // render pixels
	uint32_t render_width, render_height;
	uint32_t reset; // 1: drop the history
} rpcs3_ngx_dlss_eval;

typedef struct rpcs3_ngx_framegen_create
{
	uint32_t width, height;               // backbuffer size
	VkFormat format;                      // backbuffer format
	uint32_t render_width, render_height; // depth and motion size
} rpcs3_ngx_framegen_create;

// Matrices are NGX's: float[4][4], row-major, row vectors (already transposed by the caller).
typedef struct rpcs3_ngx_framegen_eval
{
	rpcs3_ngx_image backbuffer;
	rpcs3_ngx_image hudless; // view 0: none
	rpcs3_ngx_image depth;   // R32F at render size
	rpcs3_ngx_image motion;  // RG32F at render size, current to previous, render pixels
	rpcs3_ngx_image output;  // the generated frame (storage)
	uint32_t multi_frame_count;
	uint32_t multi_frame_index;
	uint32_t reset;
	uint32_t depth_inverted;
	float motion_vectors_invalid_value;
	float camera_view_to_clip[4][4];
	float clip_to_camera_view[4][4];
	float clip_to_lens_clip[4][4];
	float clip_to_prev_clip[4][4];
	float prev_clip_to_clip[4][4];
	float camera_near, camera_far, camera_fov, camera_aspect;
} rpcs3_ngx_framegen_eval;

typedef struct rpcs3_ngx_api
{
	uint32_t version; // RPCS3_NGX_API_VERSION

	// Before vkCreateInstance / vkCreateDevice (no init needed). The lists stay valid (NGX's static storage).
	uint32_t (*dlss_required_extensions)(uint32_t* instance_count, const char*** instance_names, uint32_t* device_count, const char*** device_names);
	uint32_t (*framegen_instance_extensions)(uint32_t* count, VkExtensionProperties** props);
	uint32_t (*framegen_device_extensions)(VkInstance instance, VkPhysicalDevice pdev, uint32_t* count, VkExtensionProperties** props);
	// The frame generation requirement query: feature_supported 0 = supported.
	uint32_t (*framegen_requirements)(VkInstance instance, VkPhysicalDevice pdev, uint32_t* feature_supported, uint32_t* min_architecture, char* min_os, uint32_t min_os_size);

	// The NGX session. data_path: a writable folder for NGX's logs. log: NGX's verbose log (null: none).
	uint32_t (*init)(const wchar_t* data_path, VkInstance instance, VkPhysicalDevice pdev, VkDevice device,
		PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa, rpcs3_ngx_log_fn log);
	void (*shutdown)(VkDevice device);

	// After init. DLSS Super Resolution: available, needs a newer driver, the feature's init result.
	uint32_t (*dlss_capabilities)(int32_t* available, int32_t* needs_driver, int32_t* init_result);
	uint32_t (*dlss_create)(VkCommandBuffer cmd, const rpcs3_ngx_dlss_create* create, void** handle);
	uint32_t (*dlss_evaluate)(VkCommandBuffer cmd, void* handle, const rpcs3_ngx_dlss_eval* eval);

	// After init. DLSS Frame Generation: also the most generated frames per rendered frame.
	uint32_t (*framegen_capabilities)(int32_t* available, int32_t* needs_driver, int32_t* init_result, uint32_t* max_generated);
	uint32_t (*framegen_create)(VkCommandBuffer cmd, const rpcs3_ngx_framegen_create* create, void** handle);
	uint32_t (*framegen_evaluate)(VkCommandBuffer cmd, void* handle, const rpcs3_ngx_framegen_eval* eval);

	// Either feature (after vkDeviceWaitIdle).
	void (*release)(void* handle);
} rpcs3_ngx_api;

typedef const rpcs3_ngx_api* (*rpcs3_ngx_get_api_fn)(void);

#ifdef __cplusplus
}
#endif
