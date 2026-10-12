#include "stdafx.h"
#include "keyboard_pad_handler.h"
#include "Utilities/File.h"
#include "Emu/IdManager.h"
#include "Emu/Cell/PPUThread.h"
#include "Emu/Cell/SPUThread.h"
#include "Emu/Memory/vm.h"

#include <QString>

// VR fork dev hook: RPCS3_VR_KEYS=<file>. When the file appears it is consumed and
// its lines are queued, one after another:
//   <keys> <hold_ms> [gap_ms]   press keys (Qt names as in the pad config, "W+X" =
//                               together), hold, release, then wait gap_ms (default 500)
//   wait <ms>
//   motion <x> <y> <z> [g]      set the SIXAXIS sensors (0-1023, 512 level, "-" unchanged)
//                               from then on (the keyboard pad has no motion: Flower steers by tilt)
//   dump [addr..]               log every PPU and SPU thread's state at once, and 256 bytes at each hex address
// Lets scripted runs drive a game when the desktop is locked and cannot take input.
void keyboard_pad_handler::process_key_script()
{
	static const std::string path = []() -> std::string
	{
		const char* v = std::getenv("RPCS3_VR_KEYS");
		return v ? v : "";
	}();
	if (path.empty())
	{
		return;
	}

	const auto now = steady_clock::now();

	if (now - m_key_script_poll > std::chrono::milliseconds(50))
	{
		m_key_script_poll = now;

		if (fs::is_file(path))
		{
			std::string text;
			if (fs::file f{path}; f)
			{
				text = f.to_string();
			}
			fs::remove_file(path);

			auto t = m_key_script.empty() ? now : m_key_script.back().at;
			for (const std::string& raw : fmt::split(text, {"\n", "\r"}))
			{
				const auto words = fmt::split(raw, {" ", "\t"});
				if (words.empty())
				{
					continue;
				}
				const auto num = [&](usz i, u32 def) -> u32
				{
					return i < words.size() ? static_cast<u32>(std::strtoul(words[i].c_str(), nullptr, 10)) : def;
				};
				if (words[0] == "dump")
				{
					// Development: log every PPU and SPU thread's state (registers, call stack, waits), e.g. in a hang.
					idm::select<named_thread<ppu_thread>>([](u32, named_thread<ppu_thread>& t)
					{
						std::string s;
						t.dump_all(s);
						input_log.success("VR key script: thread dump %s\n%s", t.get_name(), s);
					});
					idm::select<named_thread<spu_thread>>([](u32, named_thread<spu_thread>& t)
					{
						std::string s;
						t.dump_all(s);
						input_log.success("VR key script: thread dump %s\n%s", t.get_name(), s);
					});
					// dump <addr> [<addr>..]: also 256 bytes of guest memory at each (hex) address
					for (usz i = 1; i < words.size(); i++)
					{
						const u32 addr = static_cast<u32>(std::strtoul(words[i].c_str(), nullptr, 16)) & ~15u;
						std::string s;
						for (u32 off = 0; off < 256 && vm::check_addr(addr + off); off += 16)
						{
							const be_t<u32>* p = vm::get_super_ptr<be_t<u32>>(addr + off);
							fmt::append(s, "\n%08x: %08x %08x %08x %08x", addr + off, +p[0], +p[1], +p[2], +p[3]);
						}
						input_log.success("VR key script: memory at 0x%x:%s", addr, s);
					}
					continue;
				}
				if (words[0] == "wait")
				{
					t += std::chrono::milliseconds(num(1, 0));
					m_key_script.push_back({t, {}, false});
					continue;
				}
				if (words[0] == "motion")
				{
					scripted_key_event ev{t, {}, false};
					for (usz i = 0; i < 4; ++i)
					{
						if (i + 1 < words.size() && words[i + 1] != "-")
						{
							ev.motion[i] = static_cast<s32>(std::clamp<long>(std::strtol(words[i + 1].c_str(), nullptr, 10), 0, 1023));
						}
					}
					m_key_script.push_back(std::move(ev));
					continue;
				}
				std::vector<u32> codes;
				for (const std::string& name : fmt::split(words[0], {"+"}))
				{
					if (const u32 c = GetKeyCode(QString::fromStdString(name)); c != Qt::NoButton)
					{
						codes.push_back(c);
					}
					else
					{
						input_log.error("VR key script: unknown key '%s'", name);
					}
				}
				m_key_script.push_back({t, codes, true});
				t += std::chrono::milliseconds(num(1, 150));
				m_key_script.push_back({t, codes, false});
				t += std::chrono::milliseconds(num(2, 500));
				m_key_script.push_back({t, {}, false});
			}
			input_log.notice("VR key script: queued %d events", m_key_script.size());
		}
	}

	usz done = 0;
	for (const auto& ev : m_key_script)
	{
		if (ev.at > now)
		{
			break;
		}
		for (const u32 c : ev.codes)
		{
			Key(c, ev.pressed);
		}
		for (usz i = 0; i < 4; ++i)
		{
			if (ev.motion[i] < 0)
			{
				continue;
			}
			for (auto& binding : m_bindings)
			{
				if (binding.pad)
				{
					binding.pad->m_sensors[i].m_value = static_cast<u16>(ev.motion[i]);
				}
			}
		}
		done++;
	}
	m_key_script.erase(m_key_script.begin(), m_key_script.begin() + done);
}
