#pragma once

// VR fork: the glue between upstream's emulator loop and the fork's modules, so that
// RSXThread.cpp, sys_timer.cpp, lv2.cpp and System.cpp each carry one call. The dev
// namespace holds development hooks that do nothing unless their environment variable
// is set (see the comments on each).

#include "util/types.hpp"

class ppu_thread;

namespace fs
{
	class file;
}

namespace rsx::vr
{
	u64 effective_vblank_rate(); // rsx_camera_probe.cpp
	// Multiview stereo is active (rsx_camera_probe.cpp): the shader programs carry RSX_SHADER_CONTROL_VR_MULTIVIEW.
	bool multiview_active();

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
