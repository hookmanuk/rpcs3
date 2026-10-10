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
	// DLSS test: the bound framebuffer has the motion attachment (set by the renderer before the programs are fetched);
	// set_vr_program_ctrl() then adds RSX_SHADER_CONTROL_DLSS_MOTION.
	void set_dlss_motion_variant(bool enabled);
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
	// The current vertex program is one of the profile's depth_remap_programs (its shaders carry RSX_SHADER_CONTROL_VR_DEPTH_REMAP).
	bool depth_remap_active();

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
