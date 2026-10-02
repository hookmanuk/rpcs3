#include "stdafx.h"
#include "keyboard_pad_handler.h"
#include "Utilities/File.h"

#include <QString>

LOG_CHANNEL(input_log, "Input");

// VR fork dev hook: RPCS3_VR_KEYS=<file>. When the file appears it is consumed and
// its lines are queued, one after another:
//   <keys> <hold_ms> [gap_ms]   press keys (Qt names as in the pad config, "W+X" =
//                               together), hold, release, then wait gap_ms (default 500)
//   wait <ms>
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
				if (words[0] == "wait")
				{
					t += std::chrono::milliseconds(num(1, 0));
					m_key_script.push_back({t, {}, false});
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
		done++;
	}
	m_key_script.erase(m_key_script.begin(), m_key_script.begin() + done);
}
