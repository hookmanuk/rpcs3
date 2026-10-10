#include "stdafx.h"
#include "rsx_vr_hooks.h"
#include "rsx_camera_probe.h"
#include "rsx_stereo_inspector.h"
#include "rsx_vr_profile_generator.h"

#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/Cell/PPUThread.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/IdManager.h"
#include "Emu/Memory/vm.h"
#include "Emu/System.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "Loader/ISO.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <thread>

#include "Utilities/Thread.h"
#include "util/sysinfo.hpp"

#ifdef _WIN32
#include <Windows.h>
#include <tlhelp32.h>
#include <DbgHelp.h>
#include "Utilities/stack_trace.h"
#endif

LOG_CHANNEL(vr_dev_log, "VRDEV");
LOG_CHANNEL(sys_log, "SYS");

extern atomic_t<bool> g_user_asked_for_screenshot;
// PPUThread.cpp and PPUDevHooks.inl (global; a block-scope extern inside rsx::vr would name rsx::vr::...).
extern void ppu_register_function_at(u32 addr, u32 size, u64 ptr);
extern u32 ppu_watch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end);
extern u32 ppu_rwatch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end);
extern bool ppu_trace_breakpoint(u32 addr);

#ifdef _WIN32
namespace
{
	// VR fork dev hook: RPCS3_RSX_SAMPLE=1 samples the RSX thread's host call stack every millisecond and logs,
	// every RPCS3_STATS_PERIOD_MS (default 5000), the functions with the most samples: self (the leaf) and
	// inclusive (anywhere on the stack). Names come from rpcs3.pdb. Started from the RSX thread itself.
	void start_rsx_host_sampler()
	{
		if (!std::getenv("RPCS3_RSX_SAMPLE"))
		{
			return;
		}

		HANDLE target{};
		if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &target,
				THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0))
		{
			return;
		}

		std::thread([target]()
			{
				constexpr usz max_depth = 64;
				const char* period_env = std::getenv("RPCS3_STATS_PERIOD_MS");
				const auto period = std::chrono::milliseconds(period_env ? std::max(100, std::atoi(period_env)) : 5000);
				std::unordered_map<u64, u32> self_hits, incl_hits;
				// RPCS3_RSX_SAMPLE=2: also the most frequent stacks (leaf and its 7 callers).
				const bool with_stacks = std::atoi(std::getenv("RPCS3_RSX_SAMPLE")) >= 2;
				std::map<std::array<u64, 8>, u32> stack_hits;
				// RPCS3_RSX_SAMPLE=3: also self samples by source line (the leaf's instruction address).
				const bool with_lines = std::atoi(std::getenv("RPCS3_RSX_SAMPLE")) >= 3;
				std::unordered_map<u64, u32> line_hits;
				u32 samples = 0;
				auto last = std::chrono::steady_clock::now();

				while (!Emu.IsStopped())
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(1));

					// Nothing may allocate while the thread is suspended (it could hold the heap lock).
					u64 funcs[max_depth];
					usz depth = 0;
					u64 leaf_rip = 0;
					if (SuspendThread(target) == static_cast<DWORD>(-1))
					{
						break;
					}
					CONTEXT ctx{};
					ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
					if (GetThreadContext(target, &ctx))
					{
						leaf_rip = ctx.Rip;
						u32 misses = 0;
						while (depth < max_depth && ctx.Rip)
						{
							DWORD64 image = 0;
							if (const auto entry = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr))
							{
								misses = 0;
								funcs[depth++] = image + entry->BeginAddress;
								void* handler_data{};
								DWORD64 establisher{};
								RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, entry, &ctx, &handler_data, &establisher, nullptr);
							}
							else
							{
								// A leaf without unwind data: its return address is at [rsp].
								funcs[depth++] = ctx.Rip | (1ull << 63);
								if (++misses > 1)
								{
									break;
								}
								ctx.Rip = *reinterpret_cast<const u64*>(ctx.Rsp);
								ctx.Rsp += 8;
							}
						}
					}
					ResumeThread(target);

					if (!depth)
					{
						continue;
					}
					samples++;
					self_hits[funcs[0]]++;
					if (with_lines)
					{
						line_hits[leaf_rip]++;
					}
					if (with_stacks)
					{
						std::array<u64, 8> key{};
						std::copy_n(funcs, std::min<usz>(depth, key.size()), key.begin());
						stack_hits[key]++;
					}
					for (usz i = 0; i < depth; i++)
					{
						if (std::find(funcs, funcs + i, funcs[i]) == funcs + i)
						{
							incl_hits[funcs[i]]++;
						}
					}

					if (const auto now = std::chrono::steady_clock::now(); now - last >= period)
					{
						const auto top = [&](const std::unordered_map<u64, u32>& hits, usz count)
						{
							std::vector<std::pair<u64, u32>> sorted(hits.begin(), hits.end());
							std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
								{
									return a.second > b.second;
								});
							sorted.resize(std::min(sorted.size(), count));
							std::vector<void*> addrs;
							for (const auto& s : sorted)
							{
								addrs.push_back(reinterpret_cast<void*>(s.first & ~(1ull << 63)));
							}
							const auto names = utils::get_backtrace_symbols(addrs);
							std::string text;
							for (usz i = 0; i < sorted.size(); i++)
							{
								std::string name = i < names.size() ? names[i] : std::string("?");
								// "path\file.cpp:line function" -> "file.cpp:line function"
								if (const usz slash = name.find_last_of("\\/", name.find(' ')); slash != umax)
								{
									name = name.substr(slash + 1);
								}
								fmt::append(text, "\n   %5.1f%% %s%s", sorted[i].second * 100. / samples, (sorted[i].first >> 63) ? "(no unwind) " : "", name);
							}
							return text;
						};
						std::string stacks;
						if (with_stacks)
						{
							std::vector<std::pair<std::array<u64, 8>, u32>> sorted(stack_hits.begin(), stack_hits.end());
							std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
								{
									return a.second > b.second;
								});
							sorted.resize(std::min<usz>(sorted.size(), 40));
							for (const auto& [key, count] : sorted)
							{
								std::vector<void*> addrs;
								for (const u64 f : key)
								{
									if (f)
										addrs.push_back(reinterpret_cast<void*>(f & ~(1ull << 63)));
								}
								std::string line;
								for (std::string name : utils::get_backtrace_symbols(addrs))
								{
									// Function name only: "path:line name" -> "name", template arguments dropped.
									for (usz i = 0; i + 1 < name.size(); i++)
									{
										if (name[i] != ':' || !std::isdigit(static_cast<u8>(name[i + 1])))
											continue;
										usz j = i + 1;
										while (j < name.size() && std::isdigit(static_cast<u8>(name[j])))
											j++;
										if (j < name.size() && name[j] == ' ')
										{
											name = name.substr(j + 1);
											break;
										}
									}
									if (const usz angle = name.find('<'); angle != umax)
										name.resize(angle);
									fmt::append(line, "%s%s", line.empty() ? "" : " < ", name);
								}
								fmt::append(stacks, "\n   %5.1f%% %s", count * 100. / samples, line);
							}
							stack_hits.clear();
						}
						std::string lines;
						if (with_lines)
						{
							// Instruction addresses -> "file:line function", merged per line.
							std::vector<void*> addrs;
							std::vector<u32> counts;
							for (const auto& [rip, count] : line_hits)
							{
								addrs.push_back(reinterpret_cast<void*>(rip));
								counts.push_back(count);
							}
							// The innermost inlined frame of each address (DbgHelp inline trace): "file:line function".
							utils::get_backtrace_symbols({}); // initialises DbgHelp
							const HANDLE process = GetCurrentProcess();
							std::vector<u8> symbol_buf(sizeof(SYMBOL_INFO) + 256);
							auto* const sym = reinterpret_cast<SYMBOL_INFO*>(symbol_buf.data());
							std::unordered_map<std::string, u32> by_line;
							for (usz i = 0; i < addrs.size(); i++)
							{
								const DWORD64 addr = reinterpret_cast<DWORD64>(addrs[i]);
								std::memset(symbol_buf.data(), 0, symbol_buf.size());
								sym->SizeOfStruct = sizeof(SYMBOL_INFO);
								sym->MaxNameLen = 255;
								IMAGEHLP_LINE64 line{};
								line.SizeOfStruct = sizeof(line);
								DWORD64 disp64 = 0;
								DWORD disp32 = 0;
								DWORD context = 0, frame_index = 0;
								bool named = false, lined = false;
								if (SymAddrIncludeInlineTrace(process, addr) && SymQueryInlineTrace(process, addr, 0, addr, addr, &context, &frame_index))
								{
									named = SymFromInlineContext(process, addr, context, &disp64, sym);
									lined = SymGetLineFromInlineContext(process, addr, context, 0, &disp32, &line);
								}
								if (!named)
									named = SymFromAddr(process, addr, &disp64, sym);
								if (!lined)
									lined = SymGetLineFromAddr64(process, addr, &disp32, &line);
								std::string file = lined && line.FileName ? std::string(line.FileName) : std::string("?");
								if (const usz slash = file.find_last_of("\\/"); slash != umax)
									file = file.substr(slash + 1);
								std::string name = fmt::format("%s:%u %s", file, lined ? line.LineNumber : 0, named ? std::string(sym->Name, sym->NameLen) : std::string("?"));
								if (name.size() > 110)
									name.resize(110);
								by_line[name] += counts[i];
							}
							std::vector<std::pair<std::string, u32>> sorted(by_line.begin(), by_line.end());
							std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
								{
									return a.second > b.second;
								});
							sorted.resize(std::min<usz>(sorted.size(), 50));
							for (const auto& [name, count] : sorted)
							{
								fmt::append(lines, "\n   %5.1f%% %s", count * 100. / samples, name);
							}
							line_hits.clear();
						}
						vr_dev_log.success("RSX host samples over %.1f s (%u samples)\n  self:%s\n  inclusive:%s%s%s%s%s",
							std::chrono::duration<f64>(now - last).count(), samples, top(self_hits, 40), top(incl_hits, 60),
							stacks.empty() ? "" : "\n  stacks:", stacks, lines.empty() ? "" : "\n  lines:", lines);
						self_hits.clear();
						incl_hits.clear();
						samples = 0;
						last = now;
					}
				}
				CloseHandle(target);
			})
			.detach();
	}
} // namespace
#endif

namespace rsx::reports
{
	// Exact occlusion counts (ZCULL Accuracy "Precise"): the setting, unless the VR profile asks for approximate
	// reports while VR renders (zcull_approximate). Declared in RSXZCULL.h, used there and in VKGSRender.cpp.
	bool precise_zpass_count()
	{
		if (!g_cfg.video.precise_zpass_count)
		{
			return false;
		}
		const auto& probe = rsx::vr::camera_probe::get();
		const auto* profile = probe.profile();
		return !(profile && profile->zcull_approximate && probe.render_enabled());
	}

	// Relaxed ZCULL Sync: the setting, or the VR profile's zcull_relaxed_sync while VR renders (the RSX thread then
	// does not wait for occlusion results the game reads back each frame; in stereo each wait covers both eyes' work).
	bool relaxed_zcull_sync()
	{
		if (g_cfg.video.relaxed_zcull_sync)
		{
			return true;
		}
		const auto& probe = rsx::vr::camera_probe::get();
		const auto* profile = probe.profile();
		return profile && profile->zcull_relaxed_sync && probe.render_enabled();
	}
} // namespace rsx::reports

// The fork's RSX_SHADER_CONTROL_VR_* bits must stay clear of upstream's program control bits (gcm_enums.h) and of each
// other. After a merge from upstream, add any new RSX_SHADER_CONTROL_* bit here.
static_assert((RSX_SHADER_CONTROL_VR_MULTIVIEW & RSX_SHADER_CONTROL_VR_EXACT_DEPTH) == 0 && (RSX_SHADER_CONTROL_VR_MULTIVIEW & RSX_SHADER_CONTROL_VR_DEPTH_REMAP) == 0 &&
	(RSX_SHADER_CONTROL_VR_EXACT_DEPTH & RSX_SHADER_CONTROL_VR_DEPTH_REMAP) == 0, "VR fork: two VR control bits overlap");
static_assert(((RSX_SHADER_CONTROL_VR_MULTIVIEW | RSX_SHADER_CONTROL_VR_EXACT_DEPTH | RSX_SHADER_CONTROL_VR_DEPTH_REMAP) & (CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT | CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS |
													 RSX_SHADER_CONTROL_USED_REGS_MASK | RSX_SHADER_CONTROL_USES_KIL | RSX_SHADER_CONTROL_UNKNOWN0 | RSX_SHADER_CONTROL_UNKNOWN1 |
													 RSX_SHADER_CONTROL_FLAT_SHADING | RSX_SHADER_CONTROL_ATTRIBUTE_INTERPOLATION | RSX_SHADER_CONTROL_INSTANCED_CONSTANTS |
													 RSX_SHADER_CONTROL_INTERPRETER_MODEL | RSX_SHADER_CONTROL_8BIT_FRAMEBUFFER | RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER |
													 RSX_SHADER_CONTROL_TEXTURE_ALPHA_KILL | RSX_SHADER_CONTROL_ALPHA_TEST | RSX_SHADER_CONTROL_POLYGON_STIPPLE |
													 RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE | RSX_SHADER_CONTROL_DISABLE_EARLY_Z | RSX_SHADER_CONTROL_TEXTURE_FORMAT_CONVERT |
													 RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE | RSX_SHADER_CONTROL_ROP_MULTISAMPLED | RSX_SHADER_CONTROL_ROP_OUTPUT_REMAP |
													 RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING | RSX_SHADER_CONTROL_META_GCM_FLAGS_MASK)) == 0,
	"VR fork: a VR shader control bit collides with an upstream shader control bit");

namespace rsx::vr
{
	// on_vertex_ucode(): the decisions for the current vertex program, and those made so far by a cheap ucode
	// fingerprint (size and the first instructions), so the full hash runs once per program (bit 0 exact depth,
	// bit 1 depth remap). A collision only gives another program the exact depth, which equals its normal depth; a
	// listed program's fingerprint is confirmed by the full hash each time, as the depth remap changes what a
	// program reads. RSX thread only; cleared on boot.
	static bool s_exact_depth = false;
	static bool s_depth_remap = false;
	static std::unordered_map<u64, u8> s_program_bits_by_fingerprint;

	void on_boot()
	{
		s_program_bits_by_fingerprint.clear();
		rsx::vr::camera_probe::get().reload_profile();
	}

	void set_vr_program_ctrl(u32& ctrl)
	{
		ctrl = multiview_active() ? (ctrl | RSX_SHADER_CONTROL_VR_MULTIVIEW) : (ctrl & ~RSX_SHADER_CONTROL_VR_MULTIVIEW);
		ctrl = s_exact_depth ? (ctrl | RSX_SHADER_CONTROL_VR_EXACT_DEPTH) : (ctrl & ~RSX_SHADER_CONTROL_VR_EXACT_DEPTH);
		ctrl = s_depth_remap ? (ctrl | RSX_SHADER_CONTROL_VR_DEPTH_REMAP) : (ctrl & ~RSX_SHADER_CONTROL_VR_DEPTH_REMAP);
	}

	bool depth_remap_active()
	{
		return s_depth_remap;
	}

	namespace
	{
		struct vp_analysis_entry
		{
			u32 entry = 0;
			u32 prefix_count = 0; // instructions hashed for the key
			u32 range_first = 0, range_count = 0;
			u64 range_hash = 0;
			RSXVertexProgram program;
			program_hash_util::vertex_program_utils::vertex_program_metadata metadata{};
		};
		std::unordered_map<u64, vp_analysis_entry> s_vp_analyses;
		u32 s_vp_hits = 0, s_vp_misses = 0;

		u64 vp_hash_range(const u32* block, u32 first, u32 count)
		{
			u64 h = 0xcbf29ce484222325ull;
			const u32* w = block + first * 4;
			for (u32 i = 0; i < count * 4; ++i)
			{
				h = (h ^ w[i]) * 0x100000001b3ull;
			}
			return h;
		}

		// The key: the entry and the hash of the instructions up to the first end marker (the whole program when it
		// has no branches past it; the range hash on the stored entry catches the rest).
		u64 vp_prefix_key(const u32* block, u32 entry, u32& prefix_count)
		{
			u64 h = 0xcbf29ce484222325ull ^ entry;
			u32 i = entry;
			for (; i < rsx::max_vertex_program_instructions; ++i)
			{
				const u32* w = block + i * 4;
				h = (h ^ w[0]) * 0x100000001b3ull;
				h = (h ^ w[1]) * 0x100000001b3ull;
				h = (h ^ w[2]) * 0x100000001b3ull;
				h = (h ^ w[3]) * 0x100000001b3ull;
				if (w[3] & 1) // D3.end
				{
					i++;
					break;
				}
			}
			prefix_count = i - entry;
			return h;
		}

		bool vp_cache_enabled()
		{
			static const bool s_on = []
			{
				const char* v = std::getenv("RPCS3_VR_VP_CACHE");
				return !v || v[0] != '0';
			}();
			return s_on;
		}
	}

	bool vp_analysis_cached(const u32* block, u32 entry, RSXVertexProgram& program, program_hash_util::vertex_program_utils::vertex_program_metadata& metadata)
	{
		if (!vp_cache_enabled() || entry >= rsx::max_vertex_program_instructions)
		{
			return false;
		}
		u32 prefix_count = 0;
		const u64 key = vp_prefix_key(block, entry, prefix_count);
		const auto it = s_vp_analyses.find(key);
		if (it == s_vp_analyses.end() || it->second.entry != entry || it->second.prefix_count != prefix_count ||
			vp_hash_range(block, it->second.range_first, it->second.range_count) != it->second.range_hash)
		{
			s_vp_misses++;
			return false;
		}
		const auto& e = it->second;
		program.data = e.program.data;
		program.base_address = e.program.base_address;
		program.entry = e.program.entry;
		program.instruction_mask = e.program.instruction_mask;
		program.jump_table = e.program.jump_table;
		metadata = e.metadata;
		s_vp_hits++;
		return true;
	}

	void vp_analysis_store(const u32* block, u32 entry, const RSXVertexProgram& program, const program_hash_util::vertex_program_utils::vertex_program_metadata& metadata)
	{
		if (!vp_cache_enabled() || entry >= rsx::max_vertex_program_instructions || !metadata.ucode_length)
		{
			return;
		}
		if (s_vp_analyses.size() >= 2048)
		{
			s_vp_analyses.clear();
		}
		u32 prefix_count = 0;
		const u64 key = vp_prefix_key(block, entry, prefix_count);
		auto& e = s_vp_analyses[key];
		e.entry = entry;
		e.prefix_count = prefix_count;
		e.range_first = program.base_address;
		e.range_count = std::min<u32>(metadata.ucode_length / 16, rsx::max_vertex_program_instructions - program.base_address);
		e.range_hash = vp_hash_range(block, e.range_first, e.range_count);
		e.program.data = program.data;
		e.program.base_address = program.base_address;
		e.program.entry = program.entry;
		e.program.instruction_mask = program.instruction_mask;
		e.program.jump_table = program.jump_table;
		e.metadata = metadata;
	}

	bool on_vertex_ucode(const RSXVertexProgram& program)
	{
		// Only for a profile that lists programs (Gran Turismo 5's menu cards, Asura's Wrath's shadow mask).
		u8 bits = 0;
		if (exact_depth_programs_listed() || depth_remap_programs_listed())
		{
			const auto& ucode = program.data;
			u64 fingerprint = 0xcbf29ce484222325ull ^ ucode.size();
			for (usz i = 0; i < std::min<usz>(ucode.size(), 16); ++i)
			{
				fingerprint = (fingerprint ^ ucode[i]) * 0x100000001b3ull;
			}
			const auto decide = [&]()
			{
				const u64 hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(program);
				return static_cast<u8>((exact_depth_program(hash) ? 1 : 0) | (depth_remap_program(hash) ? 2 : 0));
			};
			if (const auto found = s_program_bits_by_fingerprint.find(fingerprint); found != s_program_bits_by_fingerprint.end())
			{
				bits = (found->second & 2) ? decide() : found->second;
			}
			else
			{
				bits = decide();
				if (s_program_bits_by_fingerprint.size() > 4096)
				{
					s_program_bits_by_fingerprint.clear();
				}
				s_program_bits_by_fingerprint.emplace(fingerprint, bits);
			}
		}
		const bool exact = (bits & 1) != 0;
		const bool remap = (bits & 2) != 0;
		const bool changed = exact != s_exact_depth || remap != s_depth_remap;
		s_exact_depth = exact;
		s_depth_remap = remap;
		return changed;
	}

	void on_frame_end(u32 buffer, u32 draw_calls)
	{
		// Finalize any armed stereo-inspector capture and arm the next one. No-op unless RPCS3_STEREO_INSPECT is set.
		rsx::vr::stereo_inspector::get().on_frame_end();
		rsx::vr::camera_probe::get().poll();
		rsx::vr::update_game_refresh_rate();
		rsx::vr::update_culling_scale();
		rsx::vr::update_option_words();
		rsx::vr::profile_generator::get().on_frame_end();

#ifdef _WIN32
		// RPCS3_RSX_SAMPLE: started from the RSX thread (it samples the calling thread).
		static std::once_flag s_sampler_started;
		std::call_once(s_sampler_started, start_rsx_host_sampler);
		// VR fork dev experiment: RPCS3_VR_RSX_CORE=<logical cpu>: the RSX thread alone on that core (its SMT sibling
		// idle), every other thread of the process kept off the pair. Checked every 2 s for threads created later.
		static const int s_rsx_core = []
		{
			const char* v = std::getenv("RPCS3_VR_RSX_CORE");
			return v ? std::atoi(v) : -1;
		}();
		if (s_rsx_core >= 0)
		{
			static u64 s_last_us = 0;
			const u64 now_us = get_system_time();
			if (now_us - s_last_us >= 2'000'000)
			{
				s_last_us = now_us;
				const u64 pair = (3ull << (s_rsx_core & ~1));
				DWORD_PTR process_mask = 0, system_mask = 0;
				GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
				const u64 others = process_mask & ~pair;
				SetThreadAffinityMask(GetCurrentThread(), 1ull << s_rsx_core);
				const DWORD me = GetCurrentThreadId(), pid = GetCurrentProcessId();
				HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
				THREADENTRY32 te{};
				te.dwSize = sizeof(te);
				u32 moved = 0;
				for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
				{
					if (te.th32OwnerProcessID != pid || te.th32ThreadID == me)
					{
						continue;
					}
					if (HANDLE h = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID))
					{
						moved += SetThreadAffinityMask(h, others) != 0;
						CloseHandle(h);
					}
				}
				CloseHandle(snap);
				static bool s_logged = false;
				if (!s_logged)
				{
					s_logged = true;
					rsx_log.success("VR dev: RSX thread pinned to cpu %d, %u other threads kept off 0x%llx", s_rsx_core, moved, pair);
				}
			}
		}
#endif

		// The dev trigger files below are checked at most every 100 ms: a file stat per frame each cost ~1% of
		// the RSX thread in test runs (they are only checked when their variables are set).
		static u64 s_dev_poll_us = 0;
		const u64 dev_now_us = get_system_time();
		const bool dev_poll = dev_now_us - s_dev_poll_us >= 100'000;
		if (dev_poll)
		{
			s_dev_poll_us = dev_now_us;
		}

		// VR fork dev hook: RPCS3_VR_SHOT=<file>; creating the file takes a screenshot
		// (consumed), for scripted runs where the desktop cannot be captured.
		static const std::string s_shot_trigger = []() -> std::string
		{
			const char* v = std::getenv("RPCS3_VR_SHOT");
			return v ? v : "";
		}();
		if (dev_poll && !s_shot_trigger.empty() && fs::is_file(s_shot_trigger) && fs::remove_file(s_shot_trigger))
		{
			g_user_asked_for_screenshot = true;
		}

		// VR fork dev hook: RPCS3_VR_SAVESTATE=<file>; creating the file saves a savestate as Ctrl+S does (the game
		// carries on), for unattended runs: with the desktop locked the game window gets no keys.
		static const std::string s_savestate_trigger = []() -> std::string
		{
			const char* v = std::getenv("RPCS3_VR_SAVESTATE");
			return v ? v : "";
		}();
		if (dev_poll && !s_savestate_trigger.empty() && fs::is_file(s_savestate_trigger) && fs::remove_file(s_savestate_trigger))
		{
			vr_dev_log.success("VR dev: savestate requested (%s)", s_savestate_trigger);
			Emu.CallFromMainThread([]()
				{
					if (!g_cfg.savestate.suspend_emu)
					{
						Emu.after_kill_callback = []()
						{
							Emu.Restart(true, false);
						};
						Emu.SetContinuousMode(true);
					}
					Emu.Kill(false, true);
				});
		}

		// VR fork dev hook: RPCS3_VR_MEMDUMP=<file>; creating the file writes guest main
		// memory 0x00000000-0xbfffffff to <file>.<n>.bin (mapped pages) and the
		// wall time to <file>.<n>.txt, for finding a game's clock by diffing dumps.
		static const std::string s_dump_trigger = []() -> std::string
		{
			const char* v = std::getenv("RPCS3_VR_MEMDUMP");
			return v ? v : "";
		}();
		if (dev_poll && !s_dump_trigger.empty() && fs::is_file(s_dump_trigger) && fs::remove_file(s_dump_trigger))
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
						if (!line.empty())
							list.push_back(static_cast<u32>(std::strtoul(line.c_str(), nullptr, 16)));
					}
				}
			}
			return list;
		}();
		if (!s_peek.empty())
		{
			static const u32 s_every = []
			{
				const char* v = std::getenv("RPCS3_VR_PEEK_EVERY");
				return v ? std::max(1u, static_cast<u32>(std::strtoul(v, nullptr, 10))) : 1u;
			}();
			static u32 s_frame = 0;
			if (s_frame++ % s_every == 0)
			{
				std::string line = fmt::format("VR peek %u t=%.3f buf %u draws %u:", s_frame, get_system_time() / 1e6, buffer, draw_calls);
				// RPCS3_VR_PEEK_CONST=<slot>: also the vertex constant register (e.g. a camera position).
				if (static const s32 s_const = []
					{
						const char* v = std::getenv("RPCS3_VR_PEEK_CONST");
						return v ? static_cast<s32>(std::strtol(v, nullptr, 10)) : -1;
					}();
					s_const >= 0 && s_const < 512)
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
		if (dev_poll && !s_poke_trigger.empty() && fs::is_file(s_poke_trigger))
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
		if (dev_poll && !s_watch_trigger.empty() && fs::is_file(s_watch_trigger))
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
		rsx::vr::note_game_flip();
		// VR fork dev hook: RPCS3_VR_FRAMESTATS=<seconds> logs the game's frame times over each window of that
		// length: frames, average FPS, 1% low and 0.1% low (the FPS of the 99th / 99.9th percentile frame time).
		// Only runs when the variable is set.
		if (static const u32 s_window = []
			{
				const char* v = std::getenv("RPCS3_VR_FRAMESTATS");
				return v ? static_cast<u32>(std::max(1, std::atoi(v))) : 0u;
			}();
			s_window)
		{
			static std::vector<f32> s_times;
			static u64 s_last = 0, s_start = 0;
			// The RSX thread's CPU time per frame (Windows): an A/B measure of its per-draw cost at a fixed rate.
			const auto rsx_cpu_us = []() -> u64
			{
#ifdef _WIN32
				// Cycle-exact (GetThreadTimes is tick-sampled), at the TSC rate.
				ULONG64 cycles = 0;
				if (static const u64 s_tsc = utils::get_tsc_freq(); s_tsc && QueryThreadCycleTime(GetCurrentThread(), &cycles))
				{
					return static_cast<u64>(static_cast<f64>(cycles) * 1e6 / static_cast<f64>(s_tsc));
				}
#endif
				return 0;
			};
			static u64 s_cpu_start = 0;
			// New frames: flips whose first game camera differs from the last one seen (a replayed or held frame repeats it).
			static u64 s_last_camera = 0;
			static u32 s_new_frames = 0;
			if (const u64 camera = rsx::vr::take_frame_camera_hash(); camera && camera != s_last_camera)
			{
				s_new_frames++;
				s_last_camera = camera;
			}
			const u64 now = get_system_time();
			if (s_last)
			{
				s_times.push_back((now - s_last) / 1000.f);
			}
			else
			{
				s_start = now;
				s_cpu_start = rsx_cpu_us();
			}
			s_last = now;
			if (now - s_start >= s_window * 1'000'000ull && s_times.size() >= 10)
			{
				std::vector<f32> sorted = s_times;
				std::sort(sorted.begin(), sorted.end());
				f64 sum = 0.;
				for (const f32 t : sorted)
					sum += t;
				const f32 p99 = sorted[std::min<usz>(sorted.size() - 1, sorted.size() * 99 / 100)];
				const f32 p999 = sorted[std::min<usz>(sorted.size() - 1, sorted.size() * 999 / 1000)];
				// Missed frames: longer than 1.5x the median frame time (at a fixed rate the median is the frame period).
				const f32 median = sorted[sorted.size() / 2];
				const usz late = static_cast<usz>(sorted.end() - std::upper_bound(sorted.begin(), sorted.end(), median * 1.5f));
				const u64 cpu = rsx_cpu_us();
				rsx_log.success("VR frame stats: %u frames over %.1f s: avg %.1f FPS, 1%% low %.1f, 0.1%% low %.1f (worst frame %.1f ms), median %.2f ms, late %.2f%%, RSX thread %.2f ms/frame, new frames %.1f/s",
					::size32(sorted), (now - s_start) / 1e6, sorted.size() * 1000. / sum, 1000.f / p99, 1000.f / p999, sorted.back(), median, late * 100. / sorted.size(),
					(cpu - s_cpu_start) / 1000. / sorted.size(), s_new_frames * 1e6 / std::max<u64>(now - s_start, 1));
				s_new_frames = 0;
				s_times.clear();
				s_start = now;
				s_cpu_start = cpu;
			}
		}
	}

	void dev::on_usleep(ppu_thread& ppu, u64 sleep_time)
	{
		// VR fork dev hooks (for finding a game's main loop).
		// RPCS3_PPU_TRACE=<hex addr>,<hex addr>,... installs trace breakpoints (PPU interpreter only).
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
		static const u32 s_callstack_at = []
		{
			const char* v = std::getenv("RPCS3_CALLSTACK_AT");
			return v ? static_cast<u32>(std::strtoul(v, nullptr, 16)) : 0u;
		}();
		if (static atomic_t<u32> s_callstack_hits = 0; s_callstack_at && ppu.cia - s_callstack_at <= 4 && s_callstack_hits++ % 30 == 0)
		{
			vr_dev_log.success("Sleep at 0x%x (LR 0x%x): %s", ppu.cia, ppu.lr, ppu.dump_callstack());
		}

		// Report period for the two hooks below: RPCS3_STATS_PERIOD_MS (default 5000).
		static const auto vr_stats_period_us = []() -> u64
		{
			static const u64 period = []
			{
				const char* v = std::getenv("RPCS3_STATS_PERIOD_MS");
				return v ? std::max<u64>(100, std::strtoull(v, nullptr, 10)) * 1000 : 5'000'000ull;
			}();
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
					// A named_thread (not a detached std::thread): it is joined when emulation ends or the process exits.
					static std::unique_ptr<named_thread<std::function<void()>>> s_sampler;
					s_sampler = std::make_unique<named_thread<std::function<void()>>>("VR PPU Sampler"sv, []()
						{
							std::map<std::string, std::map<std::pair<u32, u32>, u32>> hist;
							// RPCS3_PPU_SAMPLE_STACK=<thread name part>: that thread's samples also by guest call stack.
							static const std::string s_stack_thread = []
							{
								const char* v = std::getenv("RPCS3_PPU_SAMPLE_STACK");
								return v ? std::string(v) : std::string();
							}();
							std::map<std::string, u32> stacks;
							u64 last_report = get_system_time();
							while (!Emu.IsStopped() && thread_ctrl::state() != thread_state::aborting)
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
										std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
											{
												return a.second > b.second;
											});
										u32 total = 0;
										for (const auto& s : sorted)
											total += s.second;
										text += fmt::format("\n %s (%u samples):", name, total);
										for (usz i = 0; i < std::min<usz>(sorted.size(), 80); ++i)
										{
											text += fmt::format(" 0x%x<-0x%x %u%%;", sorted[i].first.first, sorted[i].first.second, sorted[i].second * 100 / std::max(total, 1u));
										}
									}
									if (!stacks.empty())
									{
										std::vector<std::pair<std::string, u32>> sorted(stacks.begin(), stacks.end());
										std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
											{
												return a.second > b.second;
											});
										text += fmt::format("\n stacks of '%s':", s_stack_thread);
										for (usz i = 0; i < std::min<usz>(sorted.size(), 40); ++i)
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
						});
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
				std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
					{
						return a.second.first > b.second.first;
					});
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

	void force_vr_resolution(std::string_view title_id)
	{
		// The profiles hold guest pixel sizes and the output aspect their camera views share. At 1080p God of War
		// Collection renders a 960x1080 frame stretched to 16:9: every camera view failed the output-aspect check and
		// the game never left the fixed screen.
		if (!g_cfg.video.vr.enabled || g_cfg.video.resolution == video_resolution::_720p || !title_has_profile(title_id))
		{
			return;
		}
		sys_log.warning("VR: Resolution %s replaced by %s (VR profiles are made at 720p; Resolution Scale sets the image size).",
			g_cfg.video.resolution.get(), video_resolution::_720p);
		g_cfg.video.resolution.set(video_resolution::_720p);
	}

	static atomic_t<u32> s_reduced_scale_frames = 0; // game frames left at the reduced scale

	void note_reduced_scale_frame(bool listed_draw_seen)
	{
		if (listed_draw_seen)
		{
			s_reduced_scale_frames = 3;
		}
		else if (const u32 n = s_reduced_scale_frames; n)
		{
			s_reduced_scale_frames = n - 1;
		}
	}

	u16 effective_resolution_scale(u16 configured_percent)
	{
		const auto& probe = camera_probe::get();
		const title_profile* profile = probe.profile();
		const bool reduce = s_reduced_scale_frames && profile && profile->reduced_scale_percent && profile->reduced_scale_percent < configured_percent &&
			probe.render_enabled() && g_cfg.video.vr.cinematic_scenes == vr_cinematic_scenes::fixed_screen;
		const u16 percent = reduce ? profile->reduced_scale_percent : configured_percent;
		if (static u16 s_last = 0; s_last != percent)
		{
			if (s_last)
			{
				rsx_log.notice("VR: Resolution Scale %u%% (%s).", percent, reduce ? "the profile's reduced_scale_frames show" : "configured");
			}
			s_last = percent;
		}
		return percent;
	}

	bool cinematic_frame_on_screen()
	{
		return s_reduced_scale_frames && g_cfg.video.vr.cinematic_scenes == vr_cinematic_scenes::fixed_screen;
	}

	bool cinematic_frame()
	{
		return s_reduced_scale_frames != 0;
	}

	u16 min_scalable_dimension(u16 configured)
	{
		const title_profile* profile = g_cfg.video.vr.enabled ? camera_probe::get().profile() : nullptr;
		return profile ? std::max(configured, profile->min_scalable_dimension) : configured;
	}

	bool savestate_disc_folder_as_iso(const std::string& library_path, std::string& disc_info, std::string& argv0)
	{
		constexpr std::string_view bdvd = "/dev_bdvd/";
		if (disc_info.empty() || disc_info[0] == '/' || !argv0.starts_with(bdvd) || library_path.empty() || !is_iso_file(library_path))
		{
			return false;
		}

		sys_log.notice("Savestate made from the disc folder of %s: loading it from the library's ISO ('%s')", disc_info, library_path);
		disc_info = library_path;
		argv0.erase(0, bdvd.size());
		return true;
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

		struct entry
		{
			u64 count = 0, total_us = 0, max_us = 0;
		};
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

		if (!s_window_start)
			s_window_start = now;
		if (now - s_window_start >= 2'000'000)
		{
			std::vector<std::pair<std::string, entry>> v(s_map.begin(), s_map.end());
			std::sort(v.begin(), v.end(), [](const auto& a, const auto& b)
				{
					return a.second.total_us > b.second.total_us;
				});
			std::string out;
			for (usz i = 0; i < std::min<usz>(v.size(), 25); i++)
				fmt::append(out, "\n  %8.1f ms/s %6.1f calls/s max %6.2f ms  %s", v[i].second.total_us / 2000., v[i].second.count / 2., v[i].second.max_us / 1000., v[i].first);
			vr_dev_log.notice("SYSPROF window %.2f s:%s", (now - s_window_start) / 1e6, out);
			s_map.clear();
			s_window_start = now;
		}
	}
} // namespace rsx::vr
