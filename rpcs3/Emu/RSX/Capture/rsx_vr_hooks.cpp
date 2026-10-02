#include "stdafx.h"
#include "rsx_vr_hooks.h"
#include "rsx_camera_probe.h"
#include "rsx_stereo_inspector.h"
#include "rsx_vr_profile_generator.h"

#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/Cell/PPUThread.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/IdManager.h"
#include "Emu/Memory/vm.h"
#include "Emu/System.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <thread>

LOG_CHANNEL(vr_dev_log, "VRDEV");

extern atomic_t<bool> g_user_asked_for_screenshot;

namespace rsx::vr
{
	void on_boot()
	{
		rsx::vr::camera_probe::get().reload_profile();
	}

	void on_frame_end(u32 buffer, u32 draw_calls)
	{
		// Finalize any armed stereo-inspector capture and arm the next one. No-op unless RPCS3_STEREO_INSPECT is set.
		rsx::vr::stereo_inspector::get().on_frame_end();
		rsx::vr::camera_probe::get().poll();
		rsx::vr::update_game_refresh_rate();
		rsx::vr::profile_generator::get().on_frame_end();

	// VR fork dev hook: RPCS3_VR_SHOT=<file>; creating the file takes a screenshot
	// (consumed), for scripted runs where the desktop cannot be captured.
	static const std::string s_shot_trigger = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_VR_SHOT");
		return v ? v : "";
	}();
	if (!s_shot_trigger.empty() && fs::is_file(s_shot_trigger) && fs::remove_file(s_shot_trigger))
	{
		g_user_asked_for_screenshot = true;
	}

	// VR fork dev hook: RPCS3_VR_MEMDUMP=<file>; creating the file writes guest main
	// memory 0x00000000-0xbfffffff to <file>.<n>.bin (mapped pages) and the
	// wall time to <file>.<n>.txt, for finding a game's clock by diffing dumps.
	static const std::string s_dump_trigger = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_VR_MEMDUMP");
		return v ? v : "";
	}();
	if (!s_dump_trigger.empty() && fs::is_file(s_dump_trigger) && fs::remove_file(s_dump_trigger))
	{
		static u32 s_dump_index = 0;
		const std::string base = fmt::format("%s.%u", s_dump_trigger, s_dump_index++);
		// Copy first (a fraction of a second) so the snapshot is close to one instant.
		std::vector<u32> pages;
		std::vector<u8> data;
		const u64 start_us = get_system_time();
		for (u32 page = 0; page < 0xc0000000u; page += 0x10000)
		{
			if (vm::check_addr(page, vm::page_readable, 0x10000))
			{
				data.insert(data.end(), vm::_ptr<u8>(page), vm::_ptr<u8>(page) + 0x10000);
				pages.push_back(page);
			}
		}
		const u64 wall_us = (start_us + get_system_time()) / 2;
		// <file>.<n>.bin: the mapped 64 KiB pages; <file>.<n>.idx: their addresses (u32 LE).
		fs::write_file(base + ".bin", fs::rewrite, data);
		fs::write_file(base + ".idx", fs::rewrite, pages.data(), pages.size() * sizeof(u32));
		fs::write_file(base + ".txt", fs::rewrite, std::to_string(wall_us));
		rsx_log.success("VR memory dump written to '%s.bin' (wall %u us)", base, wall_us);
	}

	// VR fork dev hook: RPCS3_VR_PEEK=<file> (read once at the first frame): lines "<addr hex>";
	// every RPCS3_VR_PEEK_EVERY frames (default 1) the u32 at each address is logged with the
	// wall time, e.g. to follow a game's frame pacing variables through a scene change.
	static const std::vector<u32> s_peek = []()
	{
		std::vector<u32> list;
		if (const char* v = std::getenv("RPCS3_VR_PEEK"))
		{
			if (fs::file f{v}; f)
			{
				for (const auto& line : fmt::split(f.to_string(), {"\n", "\r", " ", ","}))
				{
					if (!line.empty()) list.push_back(static_cast<u32>(std::strtoul(line.c_str(), nullptr, 16)));
				}
			}
		}
		return list;
	}();
	if (!s_peek.empty())
	{
		static const u32 s_every = [] { const char* v = std::getenv("RPCS3_VR_PEEK_EVERY"); return v ? std::max(1u, static_cast<u32>(std::strtoul(v, nullptr, 10))) : 1u; }();
		static u32 s_frame = 0;
		if (s_frame++ % s_every == 0)
		{
			std::string line = fmt::format("VR peek %u t=%.3f buf %u draws %u:", s_frame, get_system_time() / 1e6, buffer, draw_calls);
			// RPCS3_VR_PEEK_CONST=<slot>: also the vertex constant register (e.g. a camera position).
			if (static const s32 s_const = [] { const char* v = std::getenv("RPCS3_VR_PEEK_CONST"); return v ? static_cast<s32>(std::strtol(v, nullptr, 10)) : -1; }(); s_const >= 0 && s_const < 512)
			{
				const auto& c = rsx::method_registers.transform_constants[s_const];
				fmt::append(line, " c%d=(%.3f %.3f %.3f)", s_const, std::bit_cast<f32>(c[0]), std::bit_cast<f32>(c[1]), std::bit_cast<f32>(c[2]));
			}
			for (const u32 addr : s_peek)
			{
				fmt::append(line, " %x=%08x", addr, vm::check_addr(addr) ? static_cast<u32>(vm::read32(addr)) : 0u);
			}
			rsx_log.notice("%s", line);
		}
	}

	// VR fork dev hook: RPCS3_VR_POKE=<file>; creating the file writes each of its lines
	// "<addr hex> f32|u32 <value>" to guest memory (big-endian), e.g. to try a patch's
	// data change on a running game or a savestate, whose memory already holds the old value.
	static const std::string s_poke_trigger = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_VR_POKE");
		return v ? v : "";
	}();
	if (!s_poke_trigger.empty() && fs::is_file(s_poke_trigger))
	{
		std::string spec;
		if (fs::file f{s_poke_trigger}; f)
		{
			spec = f.to_string();
		}
		fs::remove_file(s_poke_trigger);
		for (const auto& line : fmt::split(spec, {"\n"}))
		{
			const auto items = fmt::split(line, {" ", "\t", "\r"});
			if (items.size() < 3)
			{
				continue;
			}
			const u32 addr = static_cast<u32>(std::strtoul(items[0].c_str(), nullptr, 16));
			// Code pages are read-only: written through the supervisor mapping, then the PPU
			// interpreter's cache is rebuilt for the word (PPU Decoder: Interpreter (static);
			// LLVM keeps its compiled code).
			const bool code = vm::check_addr(addr, vm::page_executable, 4);
			if (!code && !vm::check_addr(addr, vm::page_writable, 4))
			{
				rsx_log.error("VR poke: 0x%x is not writable", addr);
				continue;
			}
			const u32 value = items[1] == "f32" ? std::bit_cast<u32>(std::strtof(items[2].c_str(), nullptr)) :
				static_cast<u32>(std::strtoul(items[2].c_str(), nullptr, 0));
			const u32 old = vm::read32(addr);
			*vm::get_super_ptr<be_t<u32>>(addr) = value;
			if (code)
			{
				extern void ppu_register_function_at(u32 addr, u32 size, u64 ptr);
				ppu_register_function_at(addr, 4, 0);
			}
			rsx_log.success("VR poke 0x%x%s: 0x%08x -> 0x%08x (%s %s)", addr, code ? " (code)" : "", old, value, items[1], items[2]);
		}
	}

	// VR fork dev hook: RPCS3_PPU_WATCH_FILE=<file>; writing "w|r,<addr>,<len>,<code start>,<code end>"
	// (hex) to the file installs the PPU store (w) or load (r) watch at that moment, so a
	// watch can target memory located in the same run (PPU interpreter only).
	static const std::string s_watch_trigger = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_PPU_WATCH_FILE");
		return v ? v : "";
	}();
	if (!s_watch_trigger.empty() && fs::is_file(s_watch_trigger))
	{
		std::string spec;
		if (fs::file f{s_watch_trigger}; f)
		{
			spec = f.to_string();
		}
		fs::remove_file(s_watch_trigger);
		const auto items = fmt::split(spec, {","});
		if (items.size() >= 5)
		{
			extern u32 ppu_watch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end);
			extern u32 ppu_rwatch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end);
			u32 v[4]{};
			for (usz i = 0; i < 4; i++)
			{
				v[i] = static_cast<u32>(std::strtoul(items[i + 1].c_str(), nullptr, 16));
			}
			const bool load = items[0].find('r') != umax;
			const u32 count = load ? ppu_rwatch_install(v[0], v[1], v[2], v[3]) : ppu_watch_install(v[0], v[1], v[2], v[3]);
			rsx_log.success("PPU %s watch on 0x%x+0x%x: %u instructions", load ? "read" : "write", v[0], v[1], count);
		}
	}
	}

	void on_flip()
	{
	// VR fork dev hook: RPCS3_VR_FRAMESTATS=<seconds> logs the game's frame times over each window of that
	// length: frames, average FPS, 1% low and 0.1% low (the FPS of the 99th / 99.9th percentile frame time).
	// Only runs when the variable is set.
	if (static const u32 s_window = [] { const char* v = std::getenv("RPCS3_VR_FRAMESTATS"); return v ? static_cast<u32>(std::max(1, std::atoi(v))) : 0u; }(); s_window)
	{
		static std::vector<f32> s_times;
		static u64 s_last = 0, s_start = 0;
		const u64 now = get_system_time();
		if (s_last)
		{
			s_times.push_back((now - s_last) / 1000.f);
		}
		else
		{
			s_start = now;
		}
		s_last = now;
		if (now - s_start >= s_window * 1'000'000ull && s_times.size() >= 10)
		{
			std::vector<f32> sorted = s_times;
			std::sort(sorted.begin(), sorted.end());
			f64 sum = 0.;
			for (const f32 t : sorted) sum += t;
			const f32 p99 = sorted[std::min<usz>(sorted.size() - 1, sorted.size() * 99 / 100)];
			const f32 p999 = sorted[std::min<usz>(sorted.size() - 1, sorted.size() * 999 / 1000)];
			// Missed frames: longer than 1.5x the median frame time (at a fixed rate the median is the frame period).
			const f32 median = sorted[sorted.size() / 2];
			const usz late = static_cast<usz>(sorted.end() - std::upper_bound(sorted.begin(), sorted.end(), median * 1.5f));
			rsx_log.success("VR frame stats: %u frames over %.1f s: avg %.1f FPS, 1%% low %.1f, 0.1%% low %.1f (worst frame %.1f ms), median %.2f ms, late %.2f%%",
				::size32(sorted), (now - s_start) / 1e6, sorted.size() * 1000. / sum, 1000.f / p99, 1000.f / p999, sorted.back(), median, late * 100. / sorted.size());
			s_times.clear();
			s_start = now;
		}
	}
	}

	void dev::on_usleep(ppu_thread& ppu, u64 sleep_time)
	{
	// VR fork dev hooks (for finding a game's main loop).
	// RPCS3_PPU_TRACE=<hex addr>,<hex addr>,... installs trace breakpoints (PPU interpreter only).
	extern bool ppu_trace_breakpoint(u32 addr);
	if (static atomic_t<bool> s_trace_set = false; !s_trace_set.exchange(true))
	{
		if (const char* list = std::getenv("RPCS3_PPU_TRACE"))
		{
			for (const auto& item : fmt::split(list, {","}))
			{
				const u32 addr = static_cast<u32>(std::strtoul(item.c_str(), nullptr, 16));
				vr_dev_log.success("Trace breakpoint at 0x%x: %s", addr, ppu_trace_breakpoint(addr) ? "set" : "FAILED");
			}
		}
	}

	// RPCS3_PPU_WATCH=<addr>,<len>,<code start>,<code end>[,<gate addr>,<gate word>] logs the
	// store instructions writing [addr, addr+len) (PPU interpreter only). Installed once the
	// word at <gate addr> equals <gate word> (e.g. after a launcher has loaded the game).
	extern u32 ppu_watch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end);
	if (static atomic_t<bool> s_watch_set = false; !s_watch_set)
	{
		if (const char* spec = std::getenv("RPCS3_PPU_WATCH"))
		{
			std::vector<u32> v;
			for (const auto& item : fmt::split(spec, {","}))
			{
				v.push_back(static_cast<u32>(std::strtoul(item.c_str(), nullptr, 16)));
			}
			const bool gated = v.size() >= 6;
			if (v.size() >= 4 && (!gated || (vm::check_addr(v[4]) && vm::read32(v[4]) == v[5])) && !s_watch_set.exchange(true))
			{
				vr_dev_log.success("Write watch on 0x%x+0x%x: %u store instructions", v[0], v[1], ppu_watch_install(v[0], v[1], v[2], v[3]));
			}
		}
		else
		{
			s_watch_set = true;
		}
	}

	// RPCS3_PPU_RWATCH: the same for load instructions (who reads a variable).
	extern u32 ppu_rwatch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end);
	if (static atomic_t<bool> s_rwatch_set = false; !s_rwatch_set)
	{
		if (const char* spec = std::getenv("RPCS3_PPU_RWATCH"))
		{
			std::vector<u32> v;
			for (const auto& item : fmt::split(spec, {","}))
			{
				v.push_back(static_cast<u32>(std::strtoul(item.c_str(), nullptr, 16)));
			}
			if (v.size() >= 4 && !s_rwatch_set.exchange(true))
			{
				vr_dev_log.success("Read watch on 0x%x+0x%x: %u load instructions", v[0], v[1], ppu_rwatch_install(v[0], v[1], v[2], v[3]));
			}
		}
		else
		{
			s_rwatch_set = true;
		}
	}

	// RPCS3_CALLSTACK_AT=<hex address> logs the guest call stack of every 30th sleep made from that address.
	static const u32 s_callstack_at = [] { const char* v = std::getenv("RPCS3_CALLSTACK_AT"); return v ? static_cast<u32>(std::strtoul(v, nullptr, 16)) : 0u; }();
	if (static atomic_t<u32> s_callstack_hits = 0; s_callstack_at && ppu.cia - s_callstack_at <= 4 && s_callstack_hits++ % 30 == 0)
	{
		vr_dev_log.success("Sleep at 0x%x (LR 0x%x): %s", ppu.cia, ppu.lr, ppu.dump_callstack());
	}

	// Report period for the two hooks below: RPCS3_STATS_PERIOD_MS (default 5000).
	static const auto vr_stats_period_us = []() -> u64
	{
		static const u64 period = [] { const char* v = std::getenv("RPCS3_STATS_PERIOD_MS"); return v ? std::max<u64>(100, std::strtoull(v, nullptr, 10)) * 1000 : 5'000'000ull; }();
		return period;
	};

	// RPCS3_USLEEP_STATS=1 logs, every 5 s, the sleep sites (address, caller) with their
	// call count and total requested sleep: a game's frame limiter shows up as ~30 or ~60
	// calls per second sleeping a few ms each.
	// RPCS3_PPU_SAMPLE=1: a sampling profiler. Every millisecond it records each PPU
	// thread's current function (cia, which the LLVM recompiler updates at function
	// entry and at syscalls) and caller (lr); every 5 s it logs the top entries per
	// thread. Shows where a frame's time goes, e.g. what a frame limiter waits on.
	if (static const bool s_ppu_sample = std::getenv("RPCS3_PPU_SAMPLE") != nullptr; s_ppu_sample)
	{
		static std::once_flag s_started;
		std::call_once(s_started, []()
		{
			std::thread([]()
			{
				std::map<std::string, std::map<std::pair<u32, u32>, u32>> hist;
				// RPCS3_PPU_SAMPLE_STACK=<thread name part>: that thread's samples also by guest call stack.
				static const std::string s_stack_thread = [] { const char* v = std::getenv("RPCS3_PPU_SAMPLE_STACK"); return v ? std::string(v) : std::string(); }();
				std::map<std::string, u32> stacks;
				u64 last_report = get_system_time();
				while (!Emu.IsStopped())
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					idm::select<named_thread<ppu_thread>>([&](u32, named_thread<ppu_thread>& ppu)
					{
						const auto name = ppu.ppu_tname.load();
						hist[name ? *name : std::string("?")][{ppu.cia, static_cast<u32>(ppu.lr)}]++;
						if (!s_stack_thread.empty() && name && name->find(s_stack_thread) != umax)
						{
							std::string key = fmt::format("0x%x", ppu.cia);
							const auto list = ppu.dump_callstack_list();
							for (usz i = 0; i < std::min<usz>(list.size(), 12); i++)
							{
								fmt::append(key, " <- 0x%x", list[i].first);
							}
							stacks[key]++;
						}
					});
					if (const u64 now = get_system_time(); now - last_report > vr_stats_period_us())
					{
						std::string text;
						for (const auto& [name, sites] : hist)
						{
							std::vector<std::pair<std::pair<u32, u32>, u32>> sorted(sites.begin(), sites.end());
							std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
							u32 total = 0;
							for (const auto& s : sorted) total += s.second;
							text += fmt::format("\n %s (%u samples):", name, total);
							for (usz i = 0; i < std::min<usz>(sorted.size(), 6); ++i)
							{
								text += fmt::format(" 0x%x<-0x%x %u%%;", sorted[i].first.first, sorted[i].first.second, sorted[i].second * 100 / std::max(total, 1u));
							}
						}
						if (!stacks.empty())
						{
							std::vector<std::pair<std::string, u32>> sorted(stacks.begin(), stacks.end());
							std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
							text += fmt::format("\n stacks of '%s':", s_stack_thread);
							for (usz i = 0; i < std::min<usz>(sorted.size(), 5); ++i)
							{
								text += fmt::format("\n   %u: %s", sorted[i].second, sorted[i].first);
							}
							stacks.clear();
						}
						vr_dev_log.success("PPU samples over %.1f s:%s", (now - last_report) / 1e6, text);
						hist.clear();
						last_report = now;
					}
				}
			}).detach();
		});
	}

	if (static const bool s_usleep_stats = std::getenv("RPCS3_USLEEP_STATS") != nullptr; s_usleep_stats)
	{
		static std::mutex s_mutex;
		static std::map<std::pair<u32, u32>, std::pair<u64, u64>> s_sites;
		static u64 s_last_report = 0;
		std::lock_guard lock(s_mutex);
		auto& site = s_sites[{ppu.cia, static_cast<u32>(ppu.lr)}];
		site.first++;
		site.second += sleep_time;
		if (const u64 now = get_system_time(); now - s_last_report > vr_stats_period_us())
		{
			std::vector<std::pair<std::pair<u32, u32>, std::pair<u64, u64>>> sorted(s_sites.begin(), s_sites.end());
			std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
			std::string text;
			for (usz i = 0; i < std::min<usz>(sorted.size(), 12); ++i)
			{
				text += fmt::format("\n  0x%x (LR 0x%x): %u calls, %u us total, %u us each", sorted[i].first.first, sorted[i].first.second,
					sorted[i].second.first, sorted[i].second.second, sorted[i].second.second / std::max<u64>(sorted[i].second.first, 1));
			}
			vr_dev_log.success("usleep sites over %.1f s:%s", (now - s_last_report) / 1e6, text);
			s_sites.clear();
			s_last_report = now;
		}
	}

	}

	void dev::dump_elf(const fs::file& elf_file)
	{
		// VR fork dev hook: RPCS3_DUMP_ELF=<path> writes the decrypted executable
		// (for finding frame-rate patch sites with a disassembler).
		if (const char* dump = std::getenv("RPCS3_DUMP_ELF"); dump && *dump)
		{
			fs::write_file(dump, fs::rewrite, elf_file.to_vector<u8>());
			vr_dev_log.success("Decrypted executable written to '%s'", dump);
		}
	}

	dev::syscall_profile_scope::syscall_profile_scope(ppu_thread& ppu, u64 code)
		: ppu(ppu), code(code)
	{
		static const bool s_prof = std::getenv("RPCS3_SYSCALL_PROFILE") != nullptr;
		active = s_prof;
		if (active)
		{
			start = get_system_time();
			lr = ppu.lr;
			sp = ppu.gpr[1];
		}
	}

	dev::syscall_profile_scope::~syscall_profile_scope()
	{
		if (!active)
		{
			return;
		}

		struct entry { u64 count = 0, total_us = 0, max_us = 0; };
		static std::mutex s_mutex;
		static std::map<std::string, entry> s_map;
		static u64 s_window_start = 0;
		const u64 now = get_system_time();

		std::string key = fmt::format("%s | %s | lr=0x%x", ppu.ppu_tname.load() ? *ppu.ppu_tname.load() : std::string("?"), ppu_syscall_code(code), lr);
		if (vm::check_addr(static_cast<u32>(sp), vm::page_readable, 8))
		{
			if (const u64 back = vm::read64(static_cast<u32>(sp)); back && vm::check_addr(static_cast<u32>(back + 16), vm::page_readable, 8))
				fmt::append(key, " <- 0x%x", vm::read64(static_cast<u32>(back + 16)));
		}

		std::lock_guard lock(s_mutex);
		auto& e = s_map[key];
		e.count++;
		e.total_us += now - start;
		e.max_us = std::max(e.max_us, now - start);

		if (!s_window_start) s_window_start = now;
		if (now - s_window_start >= 2'000'000)
		{
			std::vector<std::pair<std::string, entry>> v(s_map.begin(), s_map.end());
			std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.total_us > b.second.total_us; });
			std::string out;
			for (usz i = 0; i < std::min<usz>(v.size(), 25); i++)
				fmt::append(out, "\n  %8.1f ms/s %6.1f calls/s max %6.2f ms  %s", v[i].second.total_us / 2000., v[i].second.count / 2., v[i].second.max_us / 1000., v[i].first);
			vr_dev_log.notice("SYSPROF window %.2f s:%s", (now - s_window_start) / 1e6, out);
			s_map.clear();
			s_window_start = now;
		}
	}
}
