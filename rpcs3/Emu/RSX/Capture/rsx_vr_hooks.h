#pragma once

// VR fork: the glue between upstream's emulator loop and the fork's modules, so that
// RSXThread.cpp, sys_timer.cpp, lv2.cpp and System.cpp each carry one call. The dev
// namespace holds development hooks that do nothing unless their environment variable
// is set (see the comments on each).

#include "util/types.hpp"

#include <string>

// Multiview stereo: the program control bit (RSXVertexProgram::ctrl, RSXFragmentProgram::ctrl) of the shader
// variants that read per-view draw parameters and sample 2D textures as arrays. Declared here, not in
// gcm_enums.h's list where upstream adds its own bits; rsx_vr_hooks.cpp checks it against those.
constexpr u32 RSX_SHADER_CONTROL_VR_MULTIVIEW = 0x00004000;
// The variants that write the game's own depth per pixel (profile screen_space.hud_exact_depth_programs).
constexpr u32 RSX_SHADER_CONTROL_VR_EXACT_DEPTH = 0x00002000;
// The variants that rebuild positions from the depth buffer as the game's camera saw them (profile depth_remap_programs).
constexpr u32 RSX_SHADER_CONTROL_VR_DEPTH_REMAP = 0x00001000;

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
	// Multiview stereo is active (rsx_camera_probe.cpp): the shader programs carry RSX_SHADER_CONTROL_VR_MULTIVIEW.
	bool multiview_active();
	// rsx::thread::get_current_vertex_program / get_current_fragment_program: sets or clears the fork's program
	// control bits: RSX_SHADER_CONTROL_VR_MULTIVIEW as multiview_active() says, RSX_SHADER_CONTROL_VR_EXACT_DEPTH and
	// RSX_SHADER_CONTROL_VR_DEPTH_REMAP as on_vertex_ucode() decided for the current vertex program.
	void set_vr_program_ctrl(u32& ctrl);
	// rsx::thread::prefetch_vertex_program, after the ucode is analysed: whether the program is listed in the profile's
	// screen_space.hud_exact_depth_programs or depth_remap_programs. True when that changed (both programs' control
	// bits then need updating).
	bool on_vertex_ucode(const RSXVertexProgram& program);
	// rsx_camera_probe.cpp: whether any program is listed, and whether this vertex ucode hash is one.
	bool exact_depth_programs_listed();
	bool exact_depth_program(u64 vertex_ucode_hash);
	bool depth_remap_programs_listed();
	bool depth_remap_program(u64 vertex_ucode_hash);
	// The current vertex program is one of the profile's depth_remap_programs (its shaders carry RSX_SHADER_CONTROL_VR_DEPTH_REMAP).
	bool depth_remap_active();

	// rsx::thread::on_init: a new boot; drop the previous game's per-frame profile cache.
	void on_boot();
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
