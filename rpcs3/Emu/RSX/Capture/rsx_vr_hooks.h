#pragma once

// VR fork: the glue between upstream's emulator loop and the fork's modules, so that
// RSXThread.cpp, sys_timer.cpp, lv2.cpp and System.cpp each carry one call. The dev
// namespace holds development hooks that do nothing unless their environment variable
// is set (see the comments on each).

#include "util/types.hpp"

// Multiview stereo: the program control bit (RSXVertexProgram::ctrl, RSXFragmentProgram::ctrl) of the shader
// variants that read per-view draw parameters and sample 2D textures as arrays. Declared here, not in
// gcm_enums.h's list where upstream adds its own bits; rsx_vr_hooks.cpp checks it against those.
constexpr u32 RSX_SHADER_CONTROL_VR_MULTIVIEW = 0x00004000;

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
	// rsx::thread::get_current_vertex_program / get_current_fragment_program: sets or clears
	// RSX_SHADER_CONTROL_VR_MULTIVIEW in the program's ctrl, as multiview_active() says.
	void set_multiview_ctrl(u32& ctrl);

	// rsx::thread::on_init: a new boot; drop the previous game's per-frame profile cache.
	void on_boot();
	// rsx::thread::on_frame_end: stereo inspector, camera probe poll, game refresh rate, profile generator, dev hooks.
	void on_frame_end(u32 buffer, u32 draw_calls);
	// rsx::thread::flip: RPCS3_VR_FRAMESTATS.
	void on_flip();

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
