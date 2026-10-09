	// VR fork: every VR member of VKGSRender. Textually included inside the class body of
	// VKGSRender.h (private section), so that upstream's header changes by two lines.
	// Definitions: VKGSRenderVR.cpp (rendering) and VKGSRenderVRDev.cpp (development tools).

	vk::surface_cache m_vr_right_rtts;
	vk::framebuffer_holder* m_vr_right_draw_fbo = nullptr;
	std::vector<vk::image*> m_vr_right_fbo_images;
	// The left eye's attachments while a draw's right eye uses m_fbo_images; a member so its storage is
	// reused (a moved-from m_fbo_images reallocated on every VR draw: ~5% of the RSX thread).
	std::vector<vk::image*> m_vr_left_fbo_images;

	// Headset pose per game frame. A frame must be declared with the pose its camera
	// draws were rotated by.
	// - Where a game's frames end in a display buffer, the pose changes only when the
	//   game moves on from one (a frame boundary). A flip requested from the game's
	//   vblank handler (ICO) can land mid-frame; it then cannot split a frame between
	//   two poses. Games that never render into a display buffer take it at each flip.
	// - The pose follows the image: camera draws stamp their targets
	//   (render_target::vr_pose), other draws pass on the newest stamp they sample,
	//   blits copy it, and the flip declares the displayed buffer's stamp. ICO draws
	//   each frame's scene into one of two buffers but composites the other (the
	//   previous frame's), then flips a frame later.
	u32 m_vr_applied_pose = 0;    // pose the camera draws are rotated by now
	s32 m_vr_display_target = -1; // display buffer bound as the colour target, or -1
	u32 vr_sampled_pose();        // newest pose stamp among the render targets the current draw samples
	void vr_stamp_targets(u32 pose, bool camera);
	void vr_mark_3d_targets(bool camera); // camera draws, and draws sampling 3D content, mark their targets vr_has_3d
	bool m_vr_flip_has_3d = true;         // the last displayed buffer held 3D content (vr_has_3d)
	u32 m_vr_flip_pose = 0;               // the pose the last displayed buffer's content was drawn with (0: unknown)
	u32 m_vr_frames_2d = 3;               // consecutive frames with no camera draws whose displayed buffer held no 3D (3: a new renderer starts on the fixed screen)
	u32 m_vr_screen_frame_draws = 0;      // draws this frame matching the profile's screen_frame_draws
	std::vector<u32> m_vr_camera_targets; // colour targets of recent camera draws (addresses, newest last)
	// Screen-space passes that read a full-screen target drawn with an older head
	// pose (ICO blends last frame's scene and glow into each new frame) sample it
	// shifted by the head rotation in between, so the blended copy lines up instead
	// of trailing the head.
	bool m_vr_params_shifted = false; // the last uploaded texture parameters carry such a shift
	u16 m_vr_params_extra_mask = 0;   // TIU slots holding homographies for this upload
	bool vr_is_feedback_texture(const vk::render_target* rtt) const;
	// A pass that blends onto a full-screen target still holding an older pose's image
	// (ICO's glow accumulates across frames) first moves that image by the head
	// rotation since, in both eyes.
	void vr_realign_blend_targets();
	// Profile reproject_older_frames: the two above (blend targets, feedback textures) apply.
	bool vr_reprojects_older_frames() const;
	// Profile screen_space.passthrough_hud: this draw is HUD/menu drawn without a matrix.
	bool vr_is_passthrough_hud();
	// Kinds of texture the fragment program samples: ordinary (uploaded) textures, colour render targets,
	// and among those a view-shaped one (the scene, the output, a display buffer: not a small mask or atlas).
	enum : u32
	{
		vr_texture_ordinary = 1,
		vr_texture_colour_target = 2,
		vr_texture_view_target = 4,
		vr_texture_depth_as_colour = 8 // a depth render target read as colour (a pass rebuilding positions from it)
	};
	u32 vr_sampled_textures();
	bool vr_unboxed_draw();
	f32 vr_hud_draw_scale();
	// Point the vertex context at a copy whose viewport matrix also maps the draw
	// into this eye's HUD box, or, for a profile pre-projected program (hash), from
	// the game's clip space into this eye's. Returns false if not applicable.
	bool vr_hud_vertex_env(f32 eye_sign, u64 preprojected_program = 0);
	// The box's combined matrix for one eye (the first half of vr_hud_vertex_env); false when the eye is not mapped.
	bool vr_hud_box_matrix(f32 eye_sign, u64 preprojected_program, f32 combined[16]);
	// Multiview: both eyes' vertex contexts in one allocation (the right eye's one entry after the left's: m_vr_draw.mv_env_right).
	bool vr_hud_vertex_env_pair(u64 preprojected_program);
	// The ucode hash of the current vertex program when the profile lists it as
	// pre-projected (screen_space.preprojected_programs), else 0.
	u64 vr_preprojected_program();
	// The current vertex program's ucode hash for the VR checks, cached per compiled program: several checks
	// per draw hashed the whole ucode each time.
	u64 vr_vertex_program_hash();
	const void* m_vr_hash_program = nullptr;
	u64 m_vr_hash = 0;
	bool vr_shift_feedback_textures(rsx::fragment_program_texture_config& params);
	void vr_redirect_previous_frame_copy(rsx::blit_src_info& src); // profile current_frame_copies
	void vr_redirect_blit_source(rsx::blit_src_info& src);         // profile texture_redirects, for blits
	u32 vr_redirect_target(u32 profile_to, u32 pitch);             // the profile's `to`, or "camera": this frame's scene target
	bool m_vr_frame_boundaries = false;
	u32 m_vr_flips_since_boundary = 0;
	void vr_update_view(); // locate the head and rotate the next frame's camera draws by it
	// Camera draws since the last view update, and view updates in a row without any:
	// frames with no 3D at all (splash screens, videos, menus) are shown as the fixed screen.
	u32 m_vr_camera_draws = 0;
	u32 m_vr_frames_without_camera = 3; // 3: a new renderer (boot, executable switch) starts on the fixed screen until its first camera draw
	// TEMPORARY diagnostic: per-frame trace of targets, boundaries, camera draws, blits and flips,
	// logged for 6 consecutive frames every 150 flips.
	std::string m_vr_trace;
	u32 m_vr_trace_flips = 0;
	u32 m_vr_trace_addr = 0;
	u32 m_vr_trace_cam_pose = 0;
	u32 m_vr_trace_cam_count = 0;
	u32 m_vr_trace_other_count = 0; // non-camera draws into the latest camera target
	bool vr_tracing() const
	{
		return (m_vr_trace_flips % 150) < 6;
	}
	void vr_trace_copy_reads(bool camera); // textures read from copies of render-target memory
	void vr_trace_flush_cam();
	void vr_track_frame_boundary(); // prepare_rtts: detect the move away from a display buffer

	// Dev (RPCS3_VR_GPUPROF=1): GPU time per render target. A timestamp at each render
	// target change and flip; every 120 frames the log lists the targets by GPU ms/frame.
	struct gpuprof_mark_t
	{
		u32 addr = 0;
		u32 width = 0;
		u32 height = 0;
		u32 format = 0; // umax: flip/present
		u32 draw = 0;   // draws since the flip when marked
	};
	s32 m_gpuprof_enabled = -1;
	VkQueryPool m_gpuprof_pool = VK_NULL_HANDLE;
	std::vector<gpuprof_mark_t> m_gpuprof_marks[3];
	bool m_gpuprof_ended[3]{};
	u32 m_gpuprof_end_draw[3]{};
	u32 m_gpuprof_slot = 0;
	u32 m_gpuprof_frames = 0;
	f64 m_gpuprof_total_ms = 0.;
	u32 m_gpuprof_target = 0; // RPCS3_VR_GPUPROF_TARGET=<hex address>: also time each draw into it
	u32 m_gpuprof_draw = 0;   // draws since the flip
	u64 m_gpuprof_draw_sum = 0;
	u64 m_gpuprof_batches = 0;                             // right-eye batches executed
	u64 m_gpuprof_right_copies = 0;                        // right-eye texture copies rebuilt from right-eye surfaces
	std::map<std::string, u32> m_gpuprof_right_copy_kinds; // what those copies were (op, size, first source)
	s64 m_gpuprof_rsx_us[5]{};                             // RSX thread: setup, vertex upload, texture upload, draw exec, flip
	f64 m_gpuprof_right_ms = 0.;                           // RSX thread wall time in the right-eye replay of draws
	f64 m_gpuprof_left_vr_ms = 0.;                         // and in the left eye's VR work (eye constants, classification, HUD env)
	f64 m_gpuprof_cpu_ms = 0.;                             // RSX thread CPU time (Windows: kernel + user)
	f64 m_gpuprof_eye_ms[3]{};                             // bind_vr_eye_constants: fill + scale, apply_render_eye, upload + bind
	u64 m_gpuprof_cpu_last = 0;
	std::chrono::steady_clock::time_point m_gpuprof_last_flip{};
	f64 m_gpuprof_wall_ms = 0.;
	f64 m_gpuprof_flip_ms = 0.;             // RSX thread inside flip()
	f64 m_gpuprof_ctxwait_ms = 0.;          // ... of which waiting for an older frame's GPU work (frame_context_cleanup)
	atomic_t<u64> m_gpuprof_readback_ns{0}; // guest threads blocked in GPU readbacks (on_access_violation)
	atomic_t<u32> m_gpuprof_readbacks{0};
	atomic_t<u32> m_gpuprof_readback_addr{0};
	f64 m_gpuprof_sync_ms = 0.; // RSX thread blocked in hard syncs (flush_command_queue(true))
	u32 m_gpuprof_syncs = 0;
	std::unordered_map<u64, std::pair<f64, u32>> m_gpuprof_sum; // key -> (ms, segments)
	std::unordered_map<u64, u64> m_gpuprof_draws;               // key -> draws
	std::unordered_map<u64, gpuprof_mark_t> m_gpuprof_keys;
	bool gpuprof_enabled();
	void gpuprof_mark(const gpuprof_mark_t& mark);
	void gpuprof_flip(const rsx::frame_statistics_t& stats);

	// Right-eye pixels a blit staged in memory with no surface (ICO bounces its
	// frame through main memory). One
	// image covers a full row of the pitch, so column chunks (1024 + 256) land in
	// the same image; a later blit back into a surface uses it. (inFamous 2 also
	// samples such copies, but only after the SPUs have rewritten them from the
	// left eye's data, so the raw right-eye copy is not substituted there.)
	struct vr_staged_copy
	{
		u32 address = 0;
		u32 pitch = 0;
		u8 bpp = 4;
		u16 width = 0; // guest pixels (pitch / bpp)
		u16 height = 0;
		std::unique_ptr<vk::image> image;
	};
	std::vector<vr_staged_copy> m_vr_staged;

	// Gate 6: the right-eye draws of one left render pass are recorded into a Vulkan
	// secondary command buffer and executed in a single right-eye pass when the left
	// pass ends (vk::g_end_renderpass_hook), instead of switching render passes twice
	// per draw. A batch is only open while the left pass is open. RPCS3_VR_BATCH=0
	// restores the per-draw replay.
	struct vr_secondary_cb : public vk::command_buffer_chunk
	{
		void attach(vk::command_pool& cmd_pool, VkCommandBuffer cb)
		{
			pool = &cmd_pool;
			commands = cb;
			is_open = true;
			is_pending = false;
			clear_state_cache();
			flags = cb_reload_dynamic_state;
		}

		void detach()
		{
			commands = VK_NULL_HANDLE;
			is_open = false;
			clear_state_cache();
			flags = 0;
		}
	};

	// The right-eye secondaries recorded into one primary. They are reusable once that primary
	// has been reset (its submission finished); the whole pool is then reset at once. Resetting
	// each secondary on its own made the driver allocate memory in every vkBeginCommandBuffer.
	struct vr_primary_batches
	{
		vk::command_pool pool;
		std::vector<VkCommandBuffer> cbs;
		usz used = 0;
		u64 reset_id = umax;
	};

	std::unordered_map<const vk::command_buffer_chunk*, std::unique_ptr<vr_primary_batches>> m_vr_primary_batches;
	VkCommandBuffer m_vr_batch_secondary = VK_NULL_HANDLE;
	bool m_vr_batch_scissor_dirty = false; // the last batched draw set a HUD-box scissor
	vr_secondary_cb m_vr_batch_cb;
	bool m_vr_batching = false;
	bool m_vr_batch_open = false;
	bool m_vr_batch_executing = false;
	vk::command_buffer_chunk* m_vr_batch_primary = nullptr;
	VkRenderPass m_vr_batch_pass = VK_NULL_HANDLE;
	vk::framebuffer_holder* m_vr_batch_fbo = nullptr;
	static VKGSRender* s_vr_batch_owner;
	static void vr_on_end_renderpass(const vk::command_buffer& cmd);
	bool vr_batch_begin(VkRenderPass pass, vk::framebuffer_holder* fbo);
	void vr_batch_flush();   // run any open batch now (ends the left pass if it is open)
	void vr_batch_execute(); // left pass closed: one right-eye pass executing the batch
	// RPCS3_VR_RTDUMP: write both eyes' surfaces at these addresses (development).
	void vr_rtdump(const std::vector<u32>& addresses, const std::string& tag);
	u64 m_vr_rtdump_program = 0;    // "prog=<hash>[#n]" in the request: dump before this program's next draw
	u32 m_vr_rtdump_skip = 0;       // (#n: the n-th draw of it from the next frame on, counting from 1)
	bool m_vr_rtdump_armed = false; // the next frame has started
	std::vector<u32> m_vr_rtdump_addresses;
	void vr_mirror_blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate);
	usz m_xform_constants_data_size = 0; // Exact current upload size; Gate 5 clones this host allocation per eye.
	// VR: sections the game or the RSX read back (see flush_listed_sections), copied early in stereo.
	shared_mutex m_vr_readback_mutex;
	std::vector<utils::address_range32> m_vr_readback_ranges;
	// VR: colour targets fully covered this frame (a pass or a full clear); see vr_clear_shown in emit_geometry.
	std::vector<u32> m_vr_frame_covered;
	std::unordered_map<const void*, u64> m_vr_layer_synced; // multiview: eye-invariant targets, the write tag their layer 1 last copied
	std::unique_ptr<vk::image> m_xr_overlay_img;                                    // RPCS3 overlays for the OpenXR quad layer
	std::unordered_map<u64, std::unique_ptr<vk::viewable_image>> m_vr_warp_scratch; // realign warp targets, by format and size
	bool bind_vr_eye_constants(f32 eye_sign, u64 source_offset, usz source_size);
	// HUD-box draws: the game's scissor mapped into the box for this eye (restore m_scissor after).
	bool vr_apply_box_scissor();
	// Profile resolution_scaled_constants: divide the listed slots of a filled constant block by the resolution scale.
	void scale_offset_constants(void* buffer, std::span<const u16> constant_ids);
	const void* m_scaled_constants_program = nullptr;
	const void* m_scaled_constants_profile = nullptr;
	const std::vector<u16>* m_scaled_constants_slots = nullptr;
	// Profile fragment_constant_overrides for the current vertex program, or null.
	const std::vector<const rsx::vr::fragment_constant_override*>* find_fragment_constant_overrides();
	const void* m_fc_overrides_program = nullptr;
	const void* m_fc_overrides_profile = nullptr;
	std::vector<const rsx::vr::fragment_constant_override*> m_fc_overrides;
	bool m_fragment_constants_overridden = false;
	u64 m_vr_last_emu_flip_us = 0;     // last game flip (not an overlay/UI refresh)
	bool m_vr_video_on_screen = false; // frames without camera draws are on the fixed screen
	u32 m_vr_reduced_scale_draws = 0;  // profile reduced_scale_frames draws in this game frame
	u16 vr_resolution_scale(u16 configured_percent) const; // flip(): the Resolution Scale to render at (reduced_scale_frames)
	void vr_apply_eye_shape_early(); // headset-shaped eyes: the per-axis scale at once while no surface exists yet
	void vr_mirror_region(areai& region, bool generated_stereo) const; // flip(): the desktop mirror's area for both eyes, each at its shown shape

	// ---- Multiview stereo (plans/7-multiview-plan.md): both eyes in one draw --------------------
	// Render targets have two layers (layer 0 the guest's picture, layer 1 the right eye), every draw
	// pass is a two-view multiview pass, the vertex shader picks each view's draw parameters by
	// gl_ViewIndex and guest shaders sample 2D textures as arrays. The two-draw path above stays as
	// the fallback (device without multiview, RPCS3_VR_MULTIVIEW=0).
	bool m_vr_multiview_supported = false;
	bool m_vr_multiview = false;             // active: follows camera_probe::render_enabled() (vr_update_multiview_mode)
	bool m_vr_mv_box_scissor = false;        // the current draw clips each eye to its own HUD-box rectangle
	VkRect2D m_vr_mv_scissor[2]{};           // per-eye scissors of the current draw (HUD box) ...
	VkRect2D m_vr_mv_bound_scissor[2]{};     // ... and the ones last bound
	u64 m_vr_mv_right_xform_offset = 0;      // the right eye's transform constants, relative to the bound window (bytes)
	u32 m_vr_mv_open_query = umax;           // first slot of the query pair open in the current pass
	u8 m_vr_draw_view_mask = 0;              // the render pass variant of the bound targets (1 = both views)
	std::unique_ptr<vk::viewable_image> m_vr_mv_right_eye; // layer 1 of the display surface, copied out for presentation
	static void vr_mv_on_before_end_renderpass(const vk::command_buffer& cmd);
	void vr_update_multiview_mode();         // prepare_rtts(): follow render_enabled(), dropping every surface on a change
	bool bind_vr_eye_constants_pair(usz source_size); // both eyes' constants in one allocation; records the per-eye box scissors
	bool vr_box_scissor_rect(VkRect2D& scissor);      // the HUD-box scissor of the eye whose constants were just transformed
	void vr_mv_begin_query_segment();        // begin the guest's query (two slots) inside the open multiview pass
	void vr_mv_end_query_segment(const vk::command_buffer& cmd); // the pass ends: end the open pair, continue with the next draw
	void vr_mv_clear_eye_rects(const std::vector<VkClearAttachment>& clear_descriptors, const VkClearRect& left, const VkClearRect& right);
	// A sub-viewport clear boxed into the HUD (GT5's mirror): each eye's image of the cleared rectangle, a slanted quad
	// with the head turned (vr_map_clear_rect). The clear covers the quad in horizontal bands; the rest of its bounding
	// box (which the draws' scissor allows) gets the nearest depth, so the draws, which the game clips with that
	// scissor, cannot reach past the quad.
	f32 m_vr_clear_quads[2][4][2]{};
	bool m_vr_clear_quads_valid = false;
	void vr_mv_clear_eye_quads(const std::vector<VkClearAttachment>& clear_descriptors);
	vk::viewable_image* vr_mv_right_eye_image(vk::command_buffer& cmd, vk::viewable_image* stereo_image);
	// Multiview hooks in upstream functions
	bool vr_bind_viewport();                 // bind_viewport(): one viewport and scissor per view; true when handled
	void vr_after_render_pass_bound();       // emit_geometry(): a pending guest query begins inside the multiview pass
	vk::image_view* vr_array_view(vk::image_view* view, rsx::texture_dimension_extended dimension); // bind_texture_env(): 2D samplers read arrays
	vk::image_view* vr_array_view(vk::image_view* view);
	VkImageViewType vr_null_view_type(rsx::texture_dimension_extended dimension); // bind_texture_env(): the null view an array sampler takes
	bool vr_clear_attachments(const std::vector<VkClearAttachment>& clear_descriptors, const VkClearRect& region, const std::optional<areai>& right_clear); // clear_surface(): true when cleared per eye
	bool vr_clear_attachments_masked(VkRect2D rect, u32 colormask, color4f color, const std::optional<areai>& right_clear); // clear_surface(): the masked-colour route per eye
	u8 vr_image_view_mask(vk::image* image); // the multiview variant of a pass over this image (stencil clears)
	u8 vr_draw_view_mask();                  // prepare_rtts(): the variant of the bound targets, into m_vr_draw_view_mask

	// ---- Hooks called from upstream functions (see VKGSRenderVR.cpp) -------------------------

	// Per-draw VR state shared by the emit_geometry() hooks.
	struct vr_draw_state
	{
		bool render = false;                // this draw goes to both eyes
		bool batch = false;                 // the right eye is batched into the pass's secondary command buffer
		bool camera_draw = false;           // the eye transform was applied (a camera draw)
		bool hud_env = false;               // the draw uses a per-eye vertex context (HUD box / pre-projected)
		bool clear_shown = false;           // clear the shown region in both eyes before this draw
		bool suspend_query = false;         // a guest occlusion query is split around the right-eye replay
		bool dynamic_state_changed = false; // this draw reloaded the dynamic state
		bool box_scissor = false;           // the left draw set a HUD-box scissor
		bool right_box_scissor = false;     // the right (per-draw replay) did too
		u32 query_continuation = umax;
		u64 preprojected = 0;
		bool mv = false;                    // multiview: both eyes in this one draw
		bool eye_constants_bound = false;   // bind_vr_eye_constants_pair bound the eyes' allocation (restored in vr_end_draw)
		u64 mv_env_right = 0;               // the right eye's vertex context, relative to the bound (left) window
		VkDescriptorBufferInfoEx guest_constants_info{};
		u64 guest_constants_dynamic_offset = 0;
		u64 guest_constants_source_offset = 0;
		VkDescriptorBufferInfoEx saved_env_info{};
		u64 saved_env_offset = 0;
		vk::framebuffer_holder* left_fbo = nullptr;
		vk::command_buffer_chunk* batch_primary = nullptr;
		std::chrono::steady_clock::time_point right_t0{};
	};
	vr_draw_state m_vr_draw;

	// emit_geometry()
	void vr_begin_draw(); // classification, batching decision, query split, guest constants
	void vr_setup_draw(); // probe draw flags, eye constants, pose stamping, HUD env, clear-shown
	void vr_after_pipeline_bind(u32 sub_index, const vk::vertex_upload_info& upload_info);
	void vr_capture_draw(u32 sub_index, const vk::vertex_upload_info& upload_info); // stereo inspector, profile generator
	void vr_before_left_draw();
	bool vr_begin_right_eye(u32 sub_index, const vk::vertex_upload_info& upload_info); // true: emit the draw again for the right eye
	void vr_end_right_eye();
	void vr_end_draw();
	void vr_restore_left_eye();
	void vr_clear_shown_region();

	// bind_texture_env()
	vk::image_view* vr_fragment_texture_view(u32 index, vk::texture_cache::sampled_image_descriptor* sampler_state, bool vr_right_eye);
	bool vr_eye_invariant_target(const vk::render_target* rtt) const; // an off-aspect colour target without 3D content: the left image serves both eyes
	vk::image_view* vr_vertex_texture_view(u32 index, vk::texture_cache::sampled_image_descriptor* sampler_state, vk::image_view* image_ptr);

	// load_texture_env(): profile texture_redirects
	struct vr_texture_redirect
	{
		bool active = false;
		u32 saved[2]{};
	};
	vr_texture_redirect vr_redirect_texture(u32 index, const rsx::fragment_texture& tex);
	void vr_restore_texture(u32 index, const vr_texture_redirect& redirect);

	// end()
	void vr_on_draw_begin();     // GPU profiler draw marks
	bool vr_skip_far_cars();     // dev RPCS3_VR_CAR_LIMIT=<n>: only the n nearest cars are drawn (GT5 test)
	bool vr_skip_draw();         // profile hidden_draws, probe hide=, RPCS3_VR_RTDUMP prog= (does the nop draw itself)
	void vr_before_draw_setup(); // realign blend targets, stereo inspector draw ordinal

	// VKGSRender(), ~VKGSRender()
	void vr_init_before_instance();
	std::string vr_select_adapter(std::vector<vk::physical_device>& gpus, const std::string& adapter_name);
	void vr_init_after_device();
	void vr_init_batching();
	void vr_init_overlays(vk::command_buffer& cmd);
	void vr_destroy_before_wait();
	void vr_destroy_resources();

	// on_access_violation(): sections read back by the guest, and the dev readback timer
	struct vr_readback_scope
	{
		VKGSRender* r;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		vr_readback_scope(VKGSRender* renderer, const vk::texture_cache::thrashed_set& result, u32 address, bool is_writing);
		vr_readback_scope(const vr_readback_scope&) = delete;
		~vr_readback_scope();
	};

	// clear_surface()
	std::optional<areai> vr_map_clear_rect(u16 fb_width, u16 fb_height, u16& scissor_x, u16& scissor_y, u16& scissor_w, u16& scissor_h);
	void vr_after_clear(bool cleared, bool full_frame, bool update_color, bool update_z, u16 scissor_x, u16 scissor_y, u16 scissor_w, u16 scissor_h,
		VkImageAspectFlags cleared_ds_aspects, const std::optional<areai>& right_clear);

	// load_program_env(): feedback textures drawn with an older head pose, sampled shifted by the rotation since.
	bool vr_shift_texture_params(rsx::fragment_program_texture_config& params);
	// load_program_env()
	void vr_apply_fragment_constant_overrides(const std::vector<const rsx::vr::fragment_constant_override*>* overrides, void* buf, usz size);
	// Bytes 64 to 96 of the vertex context, after its matrix (shared by load_program_env and vr_hud_vertex_env).
	void fill_vertex_env_tail(char* buf, f32 vr_keep_depth);
	// Profile depth_remap_programs: the bytes of the eye-to-game depth remap matrix that follow the current program's
	// vertex constants (64 for a listed program's compiled shaders, else 0), and their fill: the eye's, as the probe
	// recorded it for the eye just transformed, or the identity (eye false, or no camera block transformed).
	usz vr_depth_remap_size() const;
	void vr_write_depth_remap(void* dst, usz size, bool eye) const;

	// prepare_rtts()
	bool vr_before_prepare_rtts(); // early readback copies; true if any was recorded
	void vr_prepare_right_rtts();  // frame boundary, GPU profiler mark, right-eye target set
	bool vr_copy_readback_sections(const rsx::gcm_framebuffer_info& info);
	void vr_submit_early_copies(bool any);

	// scaled_image_from_memory()
	void vr_before_blit(rsx::blit_src_info& src, const rsx::blit_dst_info& dst);
	void vr_after_blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate);

	// frame_context_cleanup(), flip()
	void vr_remove_overlay_temp_resources(u32 uid);
	void vr_flip_begin(const rsx::display_flip_info_t& info); // pending batch, RPCS3_VR_RTDUMP, GPU profiler
	bool vr_present_right_eye(const vk::present_surface_info& present_info, const rsx::avconf& avconfig, u32 buffer_width, u32 buffer_height, vk::viewable_image* image_to_flip, vk::viewable_image*& image_to_flip2);
	void vr_publish_frame(const rsx::display_flip_info_t& info, vk::viewable_image* image_to_flip, vk::viewable_image* image_to_flip2);
	bool vr_capturable(vk::viewable_image* image_to_flip, u32 buffer_width, u32 buffer_height);
	bool vr_side_by_side_shot(vk::viewable_image* image_to_flip, vk::viewable_image* image_to_flip2, u32 buffer_width, u32 buffer_height);
	void vr_crop_for_side_by_side(rsx::simple_array<vk::viewable_image*>& calibration_src, u32 buffer_width, u32 buffer_height);
	u32 m_vr_eye_width = 0; // resolution-scaled eye size of the last present source
	u32 m_vr_eye_height = 0;

	// Dev GPU profiler: wall time of a scope added to one of the m_gpuprof_*_ms accumulators.
	struct gpuprof_timer
	{
		VKGSRender* r;
		f64 VKGSRender::*acc;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~gpuprof_timer()
		{
			if (r->m_gpuprof_enabled > 0)
			{
				r->*acc += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start).count();
			}
		}
	};
	// flush_command_queue(): RSX thread time blocked in hard syncs.
	struct gpuprof_sync_timer
	{
		VKGSRender* r;
		bool hard_sync;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~gpuprof_sync_timer()
		{
			if (hard_sync && r->m_gpuprof_enabled > 0)
			{
				r->m_gpuprof_sync_ms += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start).count();
				r->m_gpuprof_syncs++;
			}
		}
	};
