#pragma once

// DLSS test (fork): motion vectors and sub-pixel jitter for the game's own 3D scene, and NVIDIA DLSS (NGX, Vulkan) run
// on the displayed image at flip. Flat (2D) rendering, and VR multiview (each eye its own matrix history, motion layer
// and DLSS feature). The two-draw VR stereo (no multiview) is not supported: nothing here is used then.
//
// How it works
// - The scene target is the colour target that took the most perspective camera draws in the last frame (camera
//   blocks from the title's VR profile, or RPCS3_DLSS_CAMERA). While it is bound, its render pass carries one more
//   colour attachment, the motion image (RGBA32F or 16F: motion in UV units, 1 - window depth, coverage), which the shaders of the
//   draws into it write (program control bit RSX_SHADER_CONTROL_DLSS_MOTION: their own shader variants).
// - Per draw the CPU finds the draw's camera matrix (current frame) and the same draw's matrix in the previous frame
//   (matched by vertex program, vertex/index addresses and the n-th such draw), and uploads R = M_prev * M_cur^-1 after
//   the vertex constants. The vertex shader takes the game's clip position p and outputs p and R * p through the
//   viewport transform; the fragment shader writes the difference. Camera motion and every rigid object's own motion
//   (cars, wheels) follow from that; GPU-skinned bones and SPU-built meshes only get the matrix's motion.
// - Camera draws into the scene target are moved by a Halton(2,3) sub-pixel jitter (in the vertex shader).
// - At flip, a compute pass resolves the motion image (sample 0) into the motion vector and depth inputs, and DLSS
//   (or the debug view) turns the displayed image into the output that is presented.
// - VR multiview: the motion image has the render targets' two layers (one per eye). Each eye's constants (the
//   game's, transformed for that eye by the VR camera code) carry their own block, R_eye = M_eye_prev * M_eye_cur^-1
//   with a matrix history per eye; the vertex shader reads its view's block like the rest of its constants. At
//   flip each layer is resolved on its own, and each eye goes through its own DLSS feature (its own history) into its
//   own output, which the headset is given in place of the eye image (or the debug views replace both eyes). A
//   switch between the headset view and the fixed screen (menus, videos) starts the history again.
//
// Settings (environment, read once):
//   RPCS3_DLSS=off|dlaa|quality|balanced|performance|ultra   DLSS mode (off by default). Render size: the game's
//                                                            (Resolution Scale). Output size, flat: the window area
//                                                            (the mode only names the preset NGX uses). VR: there is no
//                                                            window, each eye is upscaled by the mode's own ratio (DLAA
//                                                            1, quality 1.5, balanced 1.72, performance 2, ultra 3):
//                                                            lower Resolution Scale by as much to keep the eye size.
//   RPCS3_DLSS_VR_SCALE=<f>  VR: the eyes' upscale ratio instead of the mode's (1 to 3; DLAA too)
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
//   RPCS3_DLSS_DEPTH_INVERTED=1               the game's depth is reversed (1 near, 0 far)
//   RPCS3_DLSS_MOTION_16F=1                   RGBA16F motion image (default RGBA32F where the sample count allows)
//   RPCS3_DLSS_DEBUG=1                        per-draw and per-flip log lines (draw keys, matrices, skipped draws)
//   RPCS3_DLSS_STANDIN=1                      tests without an NVIDIA GPU: a bilinear upscale of the colour takes
//                                             NGX's place (everything around it, the per-eye outputs, sizes and
//                                             presentation, runs as with DLSS)

#include "util/types.hpp"
#include "VulkanAPI.h"
#include "Utilities/geometry.h"
#include "../Common/unordered_map.hpp"

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

	// Self-test (dlss_selftest): the camera draws of the frame the last presented image shows, per view (0 the game's
	// constants, 1 left eye, 2 right eye), in draw order. Recorded only after dlss_test_record(true).
	struct dlss_test_draw
	{
		u32 view;
		u32 index;       // the n-th camera draw of this view in the frame
		f32 cur[16];     // clip = M * v (row-major, column vectors)
		f32 prev[16];    // the same draw's M in the previous frame (or cur when it had none)
		bool matched;    // the draw was found in the previous frame
	};
	void dlss_test_record(bool enable);
	std::vector<dlss_test_draw> dlss_test_presented_draws();

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
		debug_warp, // RPCS3_DLSS=warp: the previous input moved by the motion vectors against this one (red), unmoved (green)
	};

	// RPCS3_DLSS
	dlss_mode dlss_configured_mode();
	// Self-test (dlss_selftest): game frames with camera draws so far, and which of them (0-based) the last presented
	// image shows.
	u64 dlss_drawn_frames();
	u64 dlss_presented_frame();
	// The DLSS context exists: a mode set at boot (DLSS is parked: no menu offers the setting).
	bool dlss_enabled();
	// The mode runs NGX (not a debug or jitter-only mode).
	bool dlss_uses_ngx();
	// NGX may run in this session: its Vulkan extensions are enabled at boot.
	bool dlss_ngx_possible();

	// ---- Vulkan instance and device (vkutils/instance.cpp, vkutils/device.cpp) -----------------------------
	void dlss_instance_extensions(std::vector<const char*>& extensions);
	void dlss_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions);
	// Before vkCreateDevice: the features NGX needs, set in the create info's chain.
	void dlss_device_chain(VkPhysicalDevice pdev, VkDeviceCreateInfo& info);

	// ---- Render passes (VKRenderPass.cpp) -------------------------------------------------------------------
	// Key bit 44: the pass has the motion attachment after the colour attachments (before depth).
	constexpr u64 dlss_renderpass_motion_bit = 1ull << 44;
	// RGBA32F, or RGBA16F where the device cannot multisample RGBA32F at this sample count (the window depth is stored as
	// 1 - z, which half floats keep precise near the far plane).
	VkFormat dlss_motion_format(u32 samples);
	inline bool dlss_renderpass_has_motion(u64 key) { return !!(key & dlss_renderpass_motion_bit); }
	// get_renderpass(), after the colour attachments: the motion attachment when the key has its bit (renderpass_key_blob
	// uses bits 0-43, the VR fork's multiview view mask 42-43).
	void dlss_add_motion_attachment(u64 renderpass_key, VkSampleCountFlagBits samples, std::vector<VkAttachmentDescription>& attachments,
		std::vector<VkAttachmentReference>& references, u32& attachment_count);

	// ---- Pipelines (VKGSRender::load_program) -----------------------------------------------------------------
	// A pass with the motion attachment: one more blend attachment, written as src * src.a + dst * (1 - src.a), so
	// a draw whose coverage (alpha) is 0 leaves it as it was.
	void dlss_apply_pipeline_props(pipeline_props& props, u32 color_attachments);
	// The shader interpreter does not write the motion output: its pipelines leave the attachment alone.
	void dlss_interpreter_props(pipeline_props& props);

	// ---- Shader generation (VKVertexProgram.cpp, VKFragmentProgram.cpp) ---------------------------------------
	// The per-draw block after the vertex constants (and after the VR depth remap matrix when present): R (4 rows),
	// then (jitter x, jitter y in NDC, coverage, 0).
	constexpr u32 dlss_draw_block_slots = 5;
	constexpr usz dlss_draw_block_size = dlss_draw_block_slots * 16;
	void dlss_insert_vertex_output(std::ostream& OS, u32 ctrl);
	// block_slot: the block's first slot (the program's constant slots, plus the depth remap's 6 when present).
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
			u32 fbo_width, u32 fbo_height, u32 color_attachments, u32 mrt_count, bool vr_unsupported);
		bool motion_bound() const { return m_bound; }
		// Whether a framebuffer whose first colour target is at this address gets the motion attachment now.
		// vr_unsupported: VR stereo without multiview (the two-draw stereo), which gets none.
		bool wants_motion(u32 color_address, bool vr_unsupported) const;
		// Before each draw into the scene target: the motion image's clear when the framebuffer stayed bound into a
		// new frame (ends an open render pass then).
		void before_draw(const vk::command_buffer& cmd);

		// end(): counts perspective camera draws per colour target (scene target detection). constant_ids as below.
		void note_draw(u32 color_address, const u16* constant_ids, usz constant_count);

		// The per-draw block (dlss_draw_block_size bytes). constant_ids: the program's compacted slots (empty: the full
		// bank). view 0: upload_transform_constants() (the game's constants, read from the registers); it starts a new
		// draw. Views 1 and 2 (VR multiview, bind_vr_eye_constants_pair): the same draw's left and right eye, whose
		// constants are in `constants` (laid out as the upload: compacted, or the full bank).
		// After a view-0 block: the draw's viewport transform (scale_offset_mat, 16 floats as uploaded), kept when the draw
		// had a camera matrix. The resolve gives pixels no draw covered (the sky) the camera's motion through it.
		void note_viewport_xform(const f32* scale_offset);
		// VR multiview: the next view-0 write_draw_block only identifies the draw (its eyes' blocks carry the motion).
		void set_identify_only(bool v) { m_identify_only = v; }
		// This frame's jitter in scene pixels (0, 0 when the mode has none).
		const f32* jitter_px() const { return m_jitter_px; }
		// The scene target's depth buffer (its address, from the framebuffer the motion attachment went into): depth-only
		// passes into it are jittered through the viewport (VKGSRender::dlss_jitter_viewport).
		void note_scene_depth(u32 address) { m_scene_depth = address; }
		u32 scene_depth() const { return m_scene_depth; }
		// The HUD outside DLSS (VR): the display image as it was before the frame's first HUD-box draw is DLSS's input,
		// and what the HUD draws added (the final image minus that copy) goes on top of DLSS's output. The HUD then needs
		// no jitter and no motion vectors (it had neither: DLSS moved its still text back and forth with the jitter, and
		// dragged it with the world's motion behind it when the head moved).
		// hud_snapshot: before a HUD-box draw into target (a display buffer; once until the next flip).
		void hud_snapshot(const vk::command_buffer& cmd, vk::viewable_image* target);
		// The copy of eye 0 / 1 when flip_image is the image it was taken from, else null.
		vk::viewable_image* hud_pre(const vk::image* flip_image, u32 eye) const;
		// Adds final - pre (bilinear, the region in_w x in_h of both) onto out (DLSS's output for that eye).
		void hud_composite(const vk::command_buffer& cmd, vk::viewable_image* out, vk::viewable_image* final_image, u32 final_layer,
			vk::viewable_image* pre, u32 in_w, u32 in_h);
		// At each flip, after the use: the next HUD draw takes a new copy.
		void hud_flip_done() { m_hud_pre_target = nullptr; }
		// on_flip's result was made this call (not the last output shown again, not a debug view).
		bool output_fresh() const { return m_output_fresh; }
		void write_draw_block(void* dst, const u16* constant_ids, usz constant_count, u64 vp_hash,
			u32 clip_width, u32 clip_height, bool instanced, const void* constants = nullptr, u32 view = 0);
		// The block for a constant upload outside a draw: no motion written, no jitter, no history.
		void write_neutral_block(void* dst) const;

		// flip(): runs the resolve and DLSS (or a debug view) on the displayed image. Returns the image to present
		// instead (out_w, out_h: its size), or null to present the source unchanged. emu_flip: a game frame (else a
		// UI refresh, which shows the last output again). right_eye (VR multiview, null otherwise): the right eye's
		// image (layer 0 of source is the left eye's), replaced in place by its DLSS output or debug view; both eyes
		// are replaced or neither. *right_eye null: a VR frame shown flat (the fixed screen).
		vk::viewable_image* on_flip(const vk::command_buffer& cmd, vk::viewable_image* source, u32 width, u32 height,
			u32 target_width, u32 target_height, bool emu_flip, bool vr_unsupported, u32& out_w, u32& out_h,
			vk::viewable_image** right_eye = nullptr);

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
		rsx::unordered_map<u64, draw_record> m_prev_draws, m_cur_draws; // dense: no allocation per insert (~10k a frame in GT5)
		rsx::unordered_map<u64, u32> m_occurrences;
		bool m_identify_only = false; // VR multiview: the view-0 call only identifies the draw for its eyes' blocks
		bool m_eyes_only = false;     // this frame's camera draws were identified only (VR multiview): statistics from the left eye
		// Per view (0 the game's constants, 1 and 2 the eyes):
		std::array<mat4, 3> m_static_reproject{};   // the most common R of the previous frame (unmatched draws)
		f32 m_viewport_xform[4]{1.f, 0.f, 1.f, 0.f}; // the scene draws' viewport transform: x scale, x offset, y scale, y offset
		bool m_viewport_valid = false;
		bool best_reprojection(u32 view, mat4& out) const; // this frame's most common R so far (the camera's motion)
		std::array<bool, 3> m_static_valid{};
		std::array<std::vector<std::pair<mat4, u32>>, 3> m_reproject_votes;
		std::array<u32, 3> m_view_draws{};          // camera draws this frame (the self-test's draw index)
		u64 m_draw_key = 0;                         // the current draw's identity (view 0), for its eyes' blocks
		u64 m_draw_key_alt = 0;                     // the same without the vertex and index addresses (ring-buffered geometry)
		u32 m_matched = 0, m_unmatched = 0, m_camera_draws = 0;
		std::vector<dlss_test_draw> m_test_draws;   // this frame's (dlss_test_record)

		// Jitter
		f32 m_jitter_px[2]{}; // this frame's, in scene pixels
		f32 m_jitter_scale = 1.f;
		u32 m_jitter_phases = 8;

		// Resolve outputs (DLSS inputs) and output; *_right: the right eye's (VR multiview)
		std::unique_ptr<vk::viewable_image> m_mv, m_depth, m_debug, m_output;
		std::unique_ptr<vk::viewable_image> m_mv_right, m_depth_right, m_debug_right, m_output_right;
		std::unique_ptr<vk::viewable_image> m_warp_prev[2], m_warp_prev2[2], m_warp_prev_mv[2]; // debug_warp: each eye's inputs and motion before
		std::unique_ptr<vk::viewable_image> m_hud_pre[2]; // hud_snapshot's copies (eye 0, 1)
		const vk::image* m_hud_pre_target = nullptr;      // the image they were taken from, until the flip
		u32 m_hud_pre_layers = 0;
		u32 m_scene_depth = 0;
		bool m_output_fresh = false;
		void warp_check(const vk::command_buffer& cmd, vk::viewable_image* source, u32 width, u32 height, u32 eye);
		bool m_reset_history = true;
		s32 m_last_stereo = -1;                      // the last game frame's: 1 both eyes (VR multiview), 0 one image
		u32 m_last_out_w = 0, m_last_out_h = 0;
		vk::viewable_image* m_last_result = nullptr; // the last frame's presented replacement (m_output or m_debug)
		vk::viewable_image* m_last_right = nullptr;  // and the right eye's (VR multiview debug views)
		u32 m_frame_camera_draws = 0;                // camera draws (any target) since the last game frame

		// NGX: a feature per eye (index 0 also the flat image's); each keeps its own history
		struct ngx_feature
		{
			void* handle = nullptr; // NVSDK_NGX_Handle*
			u32 in[2]{}, out[2]{};
			u32 quality = 0;
			u32 preset = 0;     // render preset letter (0: NGX's default)
			bool fresh = false; // created since its last evaluation: no history yet
		};
		void* m_ngx_params = nullptr;  // NVSDK_NGX_Parameter*
		std::array<ngx_feature, 2> m_ngx_features{};
		bool m_ngx_ready = false;

		void update_jitter();
		// At each flip: the configured mode and model (the settings or the home menu changed them).
		void refresh_mode();
		u32 m_preset_letter = 0; // the model's render preset letter (0: NGX's default), as the features were made
		// constants: as write_draw_block's (null: the registers).
		bool find_camera_matrix(const u16* ids, usz count, mat4& out, const void* constants = nullptr) const;
		bool ensure_motion_image(const vk::command_buffer& cmd, vk::image* color_surface);
		bool clear_if_new_frame(const vk::command_buffer& cmd); // true when it cleared
		bool ensure_resolve_images(u32 width, u32 height, bool stereo);
		bool ensure_output_images(u32 width, u32 height, bool stereo);
		// VR: the eyes' upscale ratio (RPCS3_DLSS_VR_SCALE, else the mode's).
		f32 vr_output_scale() const;
		void resolve(const vk::command_buffer& cmd, u32 width, u32 height, u32 eye); // eye: the motion image's layer
		bool ngx_init();
		// eye: the feature, resolve images and output to use (0 also for the flat image).
		bool ngx_evaluate(const vk::command_buffer& cmd, u32 eye, vk::viewable_image* source, u32 in_w, u32 in_h, u32 out_w, u32 out_h);
		// RPCS3_DLSS_STANDIN: the colour's bilinear upscale into the eye's output, in NGX's place.
		bool standin_evaluate(const vk::command_buffer& cmd, u32 eye, vk::viewable_image* source, u32 in_w, u32 in_h, u32 out_w, u32 out_h);
		void ngx_release_features();
	};
}
