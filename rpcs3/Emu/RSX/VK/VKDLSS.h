#pragma once

// DLSS test (fork): motion vectors and sub-pixel jitter for the game's own 3D scene, and NVIDIA DLSS (NGX, Vulkan) run
// on the displayed image at flip. Flat (2D) rendering only for now: with VR stereo active nothing here is used.
//
// How it works
// - The scene target is the colour target that took the most perspective camera draws in the last frame (camera
//   blocks from the title's VR profile, or RPCS3_DLSS_CAMERA). While it is bound, its render pass carries one more
//   colour attachment, the motion image (RGBA32F: motion in UV units, window depth, coverage), which the shaders of the
//   draws into it write (program control bit RSX_SHADER_CONTROL_DLSS_MOTION: their own shader variants).
// - Per draw the CPU finds the draw's camera matrix (current frame) and the same draw's matrix in the previous frame
//   (matched by vertex program, vertex/index addresses and the n-th such draw), and uploads R = M_prev * M_cur^-1 after
//   the vertex constants. The vertex shader takes the game's clip position p and outputs p and R * p through the
//   viewport transform; the fragment shader writes the difference. Camera motion and every rigid object's own motion
//   (cars, wheels) follow from that; GPU-skinned bones and SPU-built meshes only get the matrix's motion.
// - Camera draws into the scene target are moved by a Halton(2,3) sub-pixel jitter (in the vertex shader).
// - At flip, a compute pass resolves the motion image (sample 0) into the motion vector and depth inputs, and DLSS
//   (or the debug view) turns the displayed image into the output that is presented.
//
// Settings (environment, read once):
//   RPCS3_DLSS=off|dlaa|quality|balanced|performance|ultra   DLSS mode (off by default). Output size: the window area,
//                                                            render size: the game's (Resolution Scale); the mode only
//                                                            names the preset NGX uses.
//   RPCS3_DLSS=motion       no DLSS: show the motion vectors (hue = direction, brightness = length, grey = none)
//   RPCS3_DLSS=depth        no DLSS: show the depth the motion pass wrote
//   RPCS3_DLSS=motionraw    no DLSS: motion as numbers (red/green = 128 + 4 x pixels, blue = covered), for tests
//   RPCS3_DLSS=jitter       no DLSS: jitter only (the image should shimmer by under a pixel: proves the jitter path)
//   RPCS3_DLSS_TARGET=<hex> the scene target's address (default: found per frame)
//   RPCS3_DLSS_CAMERA=<slot>[,rows|cols]  camera block when the title has no VR profile (default cols)
//   RPCS3_DLSS_JITTER_SIGN=<x sign><y sign>   jitter sign convention passed to DLSS (default "++")
//   RPCS3_DLSS_MV_SIGN=<x sign><y sign>       motion vector sign convention passed to DLSS (default "++")
//   RPCS3_DLSS_JITTER_SCALE=<f>               jitter amplitude (default 1; 0 disables jitter)
//   RPCS3_DLSS_SHARPNESS=<f>                  unused by DLSS 2.5.1+ (kept for older DLLs), default 0

#include "util/types.hpp"
#include "VulkanAPI.h"
#include "Utilities/geometry.h"

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace vk
{
	class command_buffer;
	class image;
	class viewable_image;
	class render_device;
	struct pipeline_props;

	enum class dlss_mode : u32
	{
		off = 0,
		debug_motion,
		debug_depth,
		debug_motion_raw,
		jitter_only,
		dlaa,
		quality,
		balanced,
		performance,
		ultra_performance,
	};

	// RPCS3_DLSS
	dlss_mode dlss_configured_mode();
	// Self-test (dlss_selftest): game frames with camera draws so far, and which of them (0-based) the last presented
	// image shows.
	u64 dlss_drawn_frames();
	u64 dlss_presented_frame();
	inline bool dlss_enabled() { return dlss_configured_mode() != dlss_mode::off; }
	// The mode runs NGX (not a debug or jitter-only mode).
	bool dlss_uses_ngx();

	// ---- Vulkan instance and device (vkutils/instance.cpp, vkutils/device.cpp) -----------------------------
	void dlss_instance_extensions(std::vector<const char*>& extensions);
	void dlss_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions);
	void dlss_device_features(VkPhysicalDevice pdev, VkPhysicalDeviceVulkan12Features& features);

	// ---- Render passes (VKRenderPass.cpp) -------------------------------------------------------------------
	// Key bit 44: the pass has the motion attachment after the colour attachments (before depth).
	constexpr u64 dlss_renderpass_motion_bit = 1ull << 44;
	constexpr VkFormat dlss_motion_format = VK_FORMAT_R32G32B32A32_SFLOAT;
	inline bool dlss_renderpass_has_motion(u64 key) { return !!(key & dlss_renderpass_motion_bit); }

	// ---- Pipelines (VKGSRender::load_program) -----------------------------------------------------------------
	// A pass with the motion attachment: one more blend attachment, written as src * src.a + dst * (1 - src.a), so
	// a draw whose coverage (alpha) is 0 leaves it as it was.
	void dlss_apply_pipeline_props(pipeline_props& props, u32 color_attachments);
	// The shader interpreter does not write the motion output: its pipelines leave the attachment alone.
	pipeline_props dlss_interpreter_props(const pipeline_props& props);

	// ---- Shader generation (VKVertexProgram.cpp, VKFragmentProgram.cpp) ---------------------------------------
	// The per-draw block after the vertex constants (and after the VR depth remap matrix when present): R (4 rows),
	// then (jitter x, jitter y in NDC, coverage, 0).
	constexpr u32 dlss_draw_block_slots = 5;
	constexpr usz dlss_draw_block_size = dlss_draw_block_slots * 16;
	void dlss_insert_vertex_output(std::ostream& OS, u32 ctrl);
	// block_slot: the block's first slot (the program's constant slots, plus the depth remap's 5 when present).
	void dlss_insert_vertex_end(std::ostream& OS, u32 ctrl, u32 block_slot, bool has_constants);
	void dlss_insert_fragment_input(std::ostream& OS, u32 ctrl);
	void dlss_insert_fragment_output(std::ostream& OS, u32 ctrl, u32 location);
	void dlss_insert_fragment_end(std::ostream& OS, u32 ctrl);

	class dlss_context
	{
	public:
		explicit dlss_context(const vk::render_device& dev);
		~dlss_context();

		dlss_context(const dlss_context&) = delete;
		dlss_context& operator=(const dlss_context&) = delete;

		// prepare_rtts(): the framebuffer about to be bound. Returns the motion image when it is the scene target
		// (the caller adds it to the framebuffer and sets the render pass bit), else null. Ends an open render pass
		// to clear or synchronise the motion image.
		vk::image* bind_framebuffer(const vk::command_buffer& cmd, u32 color_address, vk::image* color_surface,
			u32 fbo_width, u32 fbo_height, u32 color_attachments, u32 mrt_count, bool vr_active);
		bool motion_bound() const { return m_bound; }
		// Whether a framebuffer whose first colour target is at this address gets the motion attachment now.
		bool wants_motion(u32 color_address, bool vr_active) const;
		// Before each draw into the scene target: the motion image's clear when the framebuffer stayed bound into a
		// new frame (ends an open render pass then).
		void before_draw(const vk::command_buffer& cmd);

		// end(): counts perspective camera draws per colour target (scene target detection). constant_ids as below.
		void note_draw(u32 color_address, const u16* constant_ids, usz constant_count);

		// upload_transform_constants(): the per-draw block (dlss_draw_block_size bytes).
		// constant_ids: the program's compacted slots (empty: the full bank).
		void write_draw_block(void* dst, const u16* constant_ids, usz constant_count, u64 vp_hash,
			u32 clip_width, u32 clip_height, bool instanced);
		// The block for a constant upload outside a draw: no motion written, no jitter, no history.
		void write_neutral_block(void* dst) const;

		// flip(): runs the resolve and DLSS (or a debug view) on the displayed image. Returns the image to present
		// instead (out_w, out_h: its size), or null to present the source unchanged. emu_flip: a game frame (else a
		// UI refresh, which shows the last output again).
		vk::viewable_image* on_flip(const vk::command_buffer& cmd, vk::viewable_image* source, u32 width, u32 height,
			u32 target_width, u32 target_height, bool emu_flip, bool vr_active, u32& out_w, u32& out_h);

		// The device is going away (before vkDestroyDevice).
		void destroy();

	private:
		struct draw_record
		{
			f32 m[16]; // clip = M * v (row-major, column vectors)
		};

		struct mat4
		{
			f32 m[16];
		};

		const vk::render_device& m_device;
		dlss_mode m_mode = dlss_mode::off;
		bool m_disabled = false; // unsupported format or failed NGX setup: everything below stays off

		// Scene target
		u32 m_scene_address = 0;      // this frame's choice (from the previous frame's counts or RPCS3_DLSS_TARGET)
		u32 m_forced_address = 0;
		std::unordered_map<u32, u32> m_target_counts;
		bool m_bound = false;
		u32 m_clip_width = 0, m_clip_height = 0; // the scene draws' viewport (host pixels) this frame
		bool m_scene_drawn = false;              // a motion draw happened this frame

		// Motion image
		std::unique_ptr<vk::viewable_image> m_motion;
		u64 m_motion_cleared_frame = umax;
		u64 m_frame = 0; // game frames (emu flips)

		// Per-draw history
		std::unordered_map<u64, draw_record> m_prev_draws, m_cur_draws;
		std::unordered_map<u64, u32> m_occurrences;
		mat4 m_static_reproject{};      // the most common R of the previous frame (unmatched draws)
		bool m_static_valid = false;
		std::vector<std::pair<mat4, u32>> m_reproject_votes;
		u32 m_matched = 0, m_unmatched = 0, m_camera_draws = 0;

		// Jitter
		f32 m_jitter_px[2]{}; // this frame's, in scene pixels
		f32 m_jitter_scale = 1.f;
		u32 m_jitter_phases = 8;

		// Resolve outputs (DLSS inputs) and output
		std::unique_ptr<vk::viewable_image> m_mv, m_depth, m_debug, m_output;
		bool m_reset_history = true;
		u32 m_last_out_w = 0, m_last_out_h = 0;
		vk::viewable_image* m_last_result = nullptr; // the last frame's presented replacement (m_output or m_debug)
		u32 m_frame_camera_draws = 0;                // camera draws (any target) since the last game frame

		// NGX
		void* m_ngx_params = nullptr;  // NVSDK_NGX_Parameter*
		void* m_ngx_feature = nullptr; // NVSDK_NGX_Handle*
		bool m_ngx_ready = false;
		u32 m_feature_in[2]{}, m_feature_out[2]{};
		u32 m_feature_quality = 0;

		void update_jitter();
		bool find_camera_matrix(const u16* ids, usz count, mat4& out) const;
		bool ensure_motion_image(const vk::command_buffer& cmd, vk::image* color_surface);
		bool clear_if_new_frame(const vk::command_buffer& cmd); // true when it cleared
		bool ensure_resolve_images(u32 width, u32 height);
		bool ensure_output_image(u32 width, u32 height);
		void resolve(const vk::command_buffer& cmd, u32 width, u32 height);
		bool ngx_init();
		bool ngx_evaluate(const vk::command_buffer& cmd, vk::viewable_image* source, u32 in_w, u32 in_h, u32 out_w, u32 out_h);
		void ngx_release_feature();
	};
}
