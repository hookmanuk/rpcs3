#pragma once

// VR fork: the glue between upstream's emulator loop and the fork's modules, so that
// RSXThread.cpp, sys_timer.cpp, lv2.cpp and System.cpp each carry one call. The dev
// namespace holds development hooks that do nothing unless their environment variable
// is set (see the comments on each).

#include "util/types.hpp"
#include "Emu/RSX/Program/ProgramStateCache.h" // vertex_program_metadata (vp_analysis_cached)

#include <string>

// Multiview stereo: the program control bit (RSXVertexProgram::ctrl, RSXFragmentProgram::ctrl) of the shader
// variants that read per-view draw parameters and sample 2D textures as arrays. Declared here, not in
// gcm_enums.h's list where upstream adds its own bits; rsx_vr_hooks.cpp checks it against those.
constexpr u32 RSX_SHADER_CONTROL_VR_MULTIVIEW = 0x00004000;
// The variants that write the game's own depth per pixel (profile screen_space.hud_exact_depth_programs).
constexpr u32 RSX_SHADER_CONTROL_VR_EXACT_DEPTH = 0x00002000;
// The variants that rebuild positions from the depth buffer as the game's camera saw them (profile depth_remap_programs).
constexpr u32 RSX_SHADER_CONTROL_VR_DEPTH_REMAP = 0x00001000;
// DLSS test (VK/VKDLSS.h): the variants that output motion vectors and jitter the scene, used while the scene target
// with its motion attachment is bound.
constexpr u32 RSX_SHADER_CONTROL_DLSS_MOTION = 0x80000000;
// Profile depth_remap_screen, passes without a camera block: with RSX_SHADER_CONTROL_VR_DEPTH_REMAP the screen-tile
// variant (programs), alone the game-view depth variant (depth_programs). A bit of the GCM flag range that upstream
// never sets (it copies only 0x0e, 0x40 and 0x80 of the game's shader control register) nor reads.
constexpr u32 RSX_SHADER_CONTROL_VR_DEPTH_REMAP_SCREEN = 0x00000200;

struct RSXVertexProgram;

class ppu_thread;

namespace fs
{
	class file;
}

namespace rsx::vr
{
	// rsx_camera_probe.cpp: the vblank rate to emulate: while a headset runs, the VR frame rate times the
	// profile's vblanks_per_frame (Unlimited: the headset's refresh rate); otherwise the configured Vblank
	// Rate (which is never modified).
	u64 effective_vblank_rate();
	// rsx::thread frame limiter: while a headset runs with a profile, a fixed Frame limit (the user's, or the online
	// config database's: God of War Collection, Ratchet & Clank Collection, Super Stardust HD carry 60) is replaced by
	// the VR rate above; the VR Frame Rate setting and the profile's max_fps choose it. Other limits pass unchanged.
	f64 vr_frame_limit(f64 limit);
	// Multiview stereo is active (rsx_camera_probe.cpp): the shader programs carry RSX_SHADER_CONTROL_VR_MULTIVIEW.
	bool multiview_active();
	// rsx::thread::get_current_vertex_program / get_current_fragment_program: sets or clears the fork's program
	// control bits: RSX_SHADER_CONTROL_VR_MULTIVIEW as multiview_active() says, RSX_SHADER_CONTROL_VR_EXACT_DEPTH and
	// RSX_SHADER_CONTROL_VR_DEPTH_REMAP as on_vertex_ucode() decided for the current vertex program.
	void set_vr_program_ctrl(u32& ctrl);
	// RSX thread wait accounting (the RPCS3_VR_FRAMESTATS line): where the RSX thread's time goes when it is not
	// emulating. fifo_empty: the game has not written commands yet (waiting on PPU/SPU); fifo_spin: the puller spins on a
	// jump-to-self or NOPs; fifo_fetch: the FIFO read stalled on a reservation; semaphore: NV406E semaphore acquire
	// (waiting on the game / SPU); gpu_fence: vk::wait_for_fence (waiting on the GPU); limiter: the frame limiter's sleep.
	enum class rsx_wait : u8 { fifo_empty, fifo_spin, fifo_fetch, semaphore, gpu_fence, limiter, count };
	void note_rsx_wait(rsx_wait kind, u64 us);
	void note_fifo_idle(u8 fifo_state, u64 us); // RSXFIFO.cpp: the FIFO left an idle state (FIFO::state) after us
	void note_eager_submit(); // VKGSRender::vr_eager_submit: one eager command buffer submission (counted per frame)
	struct rsx_wait_scope
	{
		u64 start;
		rsx_wait kind;
		explicit rsx_wait_scope(rsx_wait k);
		~rsx_wait_scope();
	};
	// DLSS test: the bound framebuffer has the motion attachment (set by the renderer before the programs are fetched);
	// set_vr_program_ctrl() then adds RSX_SHADER_CONTROL_DLSS_MOTION.
	void set_dlss_motion_variant(bool enabled);
	// DLSS: the renderer made its DLSS context at boot (DLSS was on then). Only such a session can change the DLSS mode
	// and model live (NGX's Vulkan extensions are enabled at boot), so the home menu offers them only then.
	void set_dlss_session(bool active);
	bool dlss_session();
	// The game window's camera shortcuts: moves the VR camera offset (Video > VR > Camera X/Y/Z Offset) by delta_cm on
	// axis 0 (x, right), 1 (y, up) or 2 (z, forward); axis 3 puts all three back to 0. Live, like the settings.
	void nudge_camera(u32 axis, s32 delta_cm);
	// The shader cache loader: a cached pipeline whose programs carry the DLSS motion bit, in a session without DLSS. It
	// is left on disk for a DLSS session. (Dev builds of 2026-10-11 used that bit for depth_remap_screen: such entries
	// are skipped rather than built as the wrong variant, which crashed MotorStorm's shader compiler.)
	bool skip_cached_pipeline(u32 vp_ctrl, u32 fp_ctrl);
	// rsx::thread::prefetch_vertex_program, after the ucode is analysed: whether the program is listed in the profile's
	// screen_space.hud_exact_depth_programs or depth_remap_programs. True when that changed (both programs' control
	// bits then need updating).
	bool on_vertex_ucode(const RSXVertexProgram& program);

	// prefetch_vertex_program(): a cache of vertex program analyses by the ucode in the transform program block.
	// Keyed by the entry and a hash of the instructions from the entry to the first end marker, verified by a hash of
	// the analysed program's whole instruction range. Gran Turismo 5 switches programs several hundred times a frame
	// and the analysis (a control-flow walk with heap allocations) was ~4% of the RSX thread.
	bool vp_analysis_cached(const u32* block, u32 entry, RSXVertexProgram& program, program_hash_util::vertex_program_utils::vertex_program_metadata& metadata);
	void vp_analysis_store(const u32* block, u32 entry, const RSXVertexProgram& program, const program_hash_util::vertex_program_utils::vertex_program_metadata& metadata);
	// rsx_camera_probe.cpp: whether any program is listed, and whether this vertex ucode hash is one.
	bool exact_depth_programs_listed();
	bool exact_depth_program(u64 vertex_ucode_hash);
	bool depth_remap_programs_listed();
	bool depth_remap_program(u64 vertex_ucode_hash);
	bool depth_remap_screen_program(u64 vertex_ucode_hash);
	bool depth_remap_game_view_program(u64 vertex_ucode_hash);
	// The current vertex program is one of the profile's depth_remap_programs (its shaders carry RSX_SHADER_CONTROL_VR_DEPTH_REMAP).
	bool depth_remap_active();
	// Before each draw's program is chosen: a depth_remap_programs entry with a texture0 size applies to the draws whose
	// texture unit 0 has it. True when that changed (the program state then needs choosing again).
	bool refresh_depth_remap_for_draw();
	// ... of its depth_remap_screen programs or depth_programs (RSX_SHADER_CONTROL_VR_DEPTH_REMAP_SCREEN; their depth
	// remap is the scene camera's), and of its programs only (the screen tiles).
	bool depth_remap_screen_active();
	bool depth_remap_tiles_active();

	// rsx::thread::on_init: a new boot; drop the previous game's per-frame profile cache.
	void on_boot();
	// Profile reduced_scale_frames. VKGSRender, at each game frame boundary: whether the frame had one of the listed
	// draws. VKGSRender::flip: the Resolution Scale to render at (the configured one, or the profile's lower one while
	// such frames show, until 3 frames without them).
	void note_reduced_scale_frame(bool listed_draw_seen);
	u16 effective_resolution_scale(u16 configured_percent);
	// VR Cinematic Scenes set to Fixed Screen, during the profile's reduced_scale_frames: the frame goes on the fixed screen.
	bool cinematic_frame_on_screen();
	// A frame of a scene the profile marks as cinematic (reduced_scale_frames: Gran Turismo 5's pre-race views). Such a
	// frame follows the Cinematic Scenes setting alone (vr_update_view), whatever other screen rules match it.
	bool cinematic_frame();
	// Dev: RPCS3_VR_FRAMELOG=<file> writes one CSV row per game flip (time, frame time, RSX thread CPU time, cinematic and
	// fixed-screen flags) and per headset frame submitted (OpenXR frame thread), for plans/tools/re/framelog.py.
	void set_frame_on_screen(bool on_screen); // VKGSRender::vr_update_view: this frame goes on the fixed screen
	void frame_log_xr_submit(f32 wait_frame_ms, f32 acquire_ms, f32 ready_ms, f32 copy_ms, f32 end_ms); // OpenXR frame thread, after xrEndFrame

	// rsx::thread::on_frame_end: stereo inspector, camera probe poll, game refresh rate, profile generator, dev hooks.
	void on_frame_end(u32 buffer, u32 draw_calls);
	// rsx::thread::flip: RPCS3_VR_FRAMESTATS.
	void on_flip();

	// Emulator::Load, reading a savestate. A state made from a disc game folder stores the title ID as its disc and
	// argv[0] as "/dev_bdvd/PS3_GAME/...". Upstream finds the folder in the game library (games.yml), but when the
	// library has the game as an ISO it never loads the ISO: /dev_bdvd stays unmounted and the state stops at its first
	// open file (sys_fs "Verification failed"). Given the library's path for the disc: if it is an ISO, rewrites the
	// two as a state made from that ISO stores them and returns true (load the state as an ISO archive).
	bool savestate_disc_folder_as_iso(const std::string& library_path, std::string& disc_info, std::string& argv0);

	// Emulator::Load, after the game's configuration is applied: with VR on, a game with a VR profile runs at 1280x720
	// output whatever the Resolution setting says. The profiles are made at 720p; Resolution Scale sets the image size.
	void force_vr_resolution(std::string_view title_id);

	namespace dev
	{
		// sys_timer_usleep: RPCS3_PPU_TRACE, RPCS3_PPU_WATCH, RPCS3_PPU_RWATCH, RPCS3_CALLSTACK_AT, RPCS3_PPU_SAMPLE, RPCS3_USLEEP_STATS.
		void on_usleep(ppu_thread& ppu, u64 sleep_time);

		// Emulator::Load: RPCS3_DUMP_ELF=<path> writes the decrypted executable.
		void dump_elf(const fs::file& elf_file);

		// ppu_execute_syscall: RPCS3_SYSCALL_PROFILE=1 sums the time spent in each syscall per
		// (thread, syscall, guest caller chain) and logs the top entries every 2 seconds.
		struct syscall_profile_scope
		{
			ppu_thread& ppu;
			u64 code;
			bool active = false;
			u64 start = 0;
			u64 lr = 0;
			u64 sp = 0;

			syscall_profile_scope(ppu_thread& ppu, u64 code);
			~syscall_profile_scope();
			syscall_profile_scope(const syscall_profile_scope&) = delete;
		};
	} // namespace dev
} // namespace rsx::vr
