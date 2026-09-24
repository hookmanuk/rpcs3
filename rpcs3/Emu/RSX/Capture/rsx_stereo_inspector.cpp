#include "stdafx.h"
#include "rsx_stereo_inspector.h"

#include "Emu/System.h"
#include "Emu/IdManager.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Core/RSXFrameBuffer.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/Utils/rsx_utils.h"

#include "util/logs.hpp"
#include "Utilities/date_time.h"

#include <sstream>
#include <iomanip>
#include <cstdlib>

LOG_CHANNEL(vr_log, "VRINSPECT");

namespace rsx::vr
{
	namespace
	{
		// JSON does not have a representation for NaN/Inf, and a camera hunt must
		// not silently lose those. Raw bits are always emitted next to this value,
		// so emitting null here is lossless overall.
		std::string json_float(f32 v)
		{
			if (!std::isfinite(v))
			{
				return "null";
			}

			std::ostringstream os;
			os << std::setprecision(9) << v;
			return os.str();
		}

		std::string json_escape(std::string_view s)
		{
			std::string out;
			out.reserve(s.size() + 2);
			for (char c : s)
			{
				switch (c)
				{
				case '"':  out += "\\\""; break;
				case '\\': out += "\\\\"; break;
				case '\n': out += "\\n"; break;
				case '\r': out += "\\r"; break;
				case '\t': out += "\\t"; break;
				default:
					if (static_cast<unsigned char>(c) < 0x20)
					{
						char buf[8];
						std::snprintf(buf, sizeof(buf), "\\u%04x", c);
						out += buf;
					}
					else
					{
						out += c;
					}
					break;
				}
			}
			return out;
		}

		// Emits one constant slot as its original guest index, raw bits, and floats.
		void append_constant(std::ostringstream& os, u32 guest_index, const u32(&slot)[4], bool first)
		{
			if (!first) os << ',';
			os << "{\"c\":" << guest_index << ",\"raw\":[";
			for (int i = 0; i < 4; ++i)
			{
				if (i) os << ',';
				os << slot[i];
			}
			os << "],\"f\":[";
			for (int i = 0; i < 4; ++i)
			{
				if (i) os << ',';
				os << json_float(std::bit_cast<f32>(slot[i]));
			}
			os << "]}";
		}
	}

	stereo_inspector& stereo_inspector::get()
	{
		static stereo_inspector instance;
		return instance;
	}

	stereo_inspector::stereo_inspector()
	{
		std::string dir;

#ifdef _WIN32
		char* buf = nullptr;
		usz buf_size = 0;
		if (_dupenv_s(&buf, &buf_size, "RPCS3_STEREO_INSPECT") == 0 && buf)
		{
			dir = buf;
			std::free(buf);
		}
#else
		if (const char* env = ::getenv("RPCS3_STEREO_INSPECT"))
		{
			dir = env;
		}
#endif

		if (dir.empty())
		{
			return;
		}

		m_out_dir = std::move(dir);
		if (m_out_dir.back() != '/' && m_out_dir.back() != '\\')
		{
			m_out_dir += '/';
		}

		m_arm_path = m_out_dir + "ARM";
		m_enabled = true;

		vr_log.success("Stereo inspector enabled. Output dir: '%s'. Create '%s' to arm one frame.",
			m_out_dir, m_arm_path);
	}

	void stereo_inspector::on_frame_end()
	{
		if (!m_enabled)
		{
			return;
		}

		m_frame_counter++;

		// Finalize the frame we were capturing.
		if (m_capturing.load())
		{
			m_capturing = false;
			close_capture();
		}

		// Arm exactly one frame when the trigger file appears.
		if (fs::is_file(m_arm_path))
		{
			if (!fs::remove_file(m_arm_path))
			{
				vr_log.error("Could not consume arm file '%s'; refusing to capture to avoid a runaway trace.", m_arm_path);
				return;
			}

			open_capture();
		}
	}

	void stereo_inspector::open_capture()
	{
		m_capture_frame = m_frame_counter;
		m_draw_ordinal = 0;
		m_vk_draw_commands = 0;
		m_seen_shaders.clear();

		const std::string title_id = Emu.GetTitleID().empty() ? Emu.GetTitle() : Emu.GetTitleID();
		const std::string path = m_out_dir + title_id + "_" + date_time::current_time_narrow() + "_stereo.jsonl";

		m_file.open(path, fs::rewrite);
		if (!m_file)
		{
			vr_log.error("Could not open capture file '%s': %s", path, fs::g_tls_error);
			return;
		}

		const auto& avconf = g_fxo->get<rsx::avconf>();
		const size2u eye_size = avconf.video_frame_size();

		std::ostringstream os;
		os << "{\"type\":\"header\""
		   << ",\"schema\":1"
		   << ",\"title_id\":\"" << json_escape(title_id) << '"'
		   << ",\"title\":\"" << json_escape(Emu.GetTitle()) << '"'
		   << ",\"app_version\":\"" << json_escape(Emu.GetAppVersion()) << '"'
		   << ",\"capture_frame\":" << m_capture_frame
		   << ",\"avconf\":{"
		   <<   "\"stereo_enabled\":" << (avconf.stereo_enabled ? "true" : "false")
		   <<   ",\"resolution_id\":" << static_cast<u32>(avconf.resolution_id)
		   <<   ",\"resolution_x\":" << avconf.resolution_x
		   <<   ",\"resolution_y\":" << avconf.resolution_y
		   <<   ",\"eye_width\":" << eye_size.width
		   <<   ",\"eye_height\":" << eye_size.height
		   <<   ",\"format\":" << static_cast<u32>(avconf.format)
		   <<   ",\"aspect\":" << static_cast<u32>(avconf.aspect)
		   << "}}";

		m_capturing = true;
		write_line(os.str());

		vr_log.success("Armed capture of frame %u -> '%s' (stereo_enabled=%d)",
			m_capture_frame, path, avconf.stereo_enabled ? 1 : 0);
	}

	void stereo_inspector::close_capture()
	{
		if (!m_file)
		{
			return;
		}

		std::ostringstream os;
		os << "{\"type\":\"footer\""
		   << ",\"capture_frame\":" << m_capture_frame
		   << ",\"logical_draws\":" << m_draw_ordinal
		   << ",\"vk_draw_commands\":" << m_vk_draw_commands
		   << ",\"unique_shaders\":" << m_seen_shaders.size()
		   << "}";

		write_line(os.str());
		m_file.close();

		vr_log.success("Capture complete: %u logical draws, %u emitted subdraws, %u unique vertex programs.",
			m_draw_ordinal, m_vk_draw_commands, static_cast<u32>(m_seen_shaders.size()));
	}

	void stereo_inspector::write_line(const std::string& line)
	{
		if (!m_file)
		{
			return;
		}

		m_file.write(line);
		m_file.write("\n", 1);
	}

	void stereo_inspector::record_note(const std::string& kind, const std::string& fields)
	{
		if (!m_enabled || !m_capturing.load())
		{
			return;
		}

		write_line(fmt::format("{\"type\":\"note\",\"kind\":\"%s\",\"after_draw\":%u,%s}", kind, m_draw_ordinal, fields));
	}

	void stereo_inspector::begin_draw_clause()
	{
		if (!m_enabled || !m_capturing.load())
		{
			return;
		}

		m_draw_ordinal++;
	}

	void stereo_inspector::emit_shader_record(const draw_capture_input& in, usz ucode_hash, usz storage_hash)
	{
		const auto& vp = *in.vertex_program;

		std::ostringstream os;
		os << "{\"type\":\"shader\""
		   << ",\"vp_storage_hash\":\"" << std::hex << storage_hash << std::dec << '"'
		   << ",\"vp_ucode_hash\":\"" << std::hex << ucode_hash << std::dec << '"'
		   << ",\"vp_session_id\":" << in.vp_session_id
		   << ",\"ctrl\":" << vp.ctrl
		   << ",\"output_mask\":" << vp.output_mask
		   << ",\"base_address\":" << vp.base_address
		   << ",\"entry\":" << vp.entry
		   << ",\"ucode_length_words\":" << vp.data.size()
		   << ",\"texture_dimensions\":" << vp.texture_state.texture_dimensions
		   << ",\"has_indexed_constants\":" << (in.has_indexed_constants ? "true" : "false")
		   << ",\"constant_ids\":[";

		if (in.constant_ids)
		{
			bool first = true;
			for (u16 id : *in.constant_ids)
			{
				if (!first) os << ',';
				os << id;
				first = false;
			}
		}

		os << "]}";
		write_line(os.str());
	}

	void stereo_inspector::record_draw(const draw_capture_input& in)
	{
		if (!m_enabled || !m_capturing.load() || !in.vertex_program || !in.framebuffer)
		{
			return;
		}

		const auto& regs = rsx::method_registers;
		const auto& vp = *in.vertex_program;
		const auto& fb = *in.framebuffer;

		const usz ucode_hash = program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(vp);
		const usz storage_hash = program_hash_util::vertex_program_storage_hash{}(vp);

		// Shader metadata is invariant per program; emit it once per capture.
		if (m_seen_shaders.insert(storage_hash).second)
		{
			emit_shader_record(in, ucode_hash, storage_hash);
		}

		m_vk_draw_commands++;

		std::ostringstream os;
		os << "{\"type\":\"draw\""
		   << ",\"frame\":" << m_capture_frame
		   << ",\"draw\":" << m_draw_ordinal
		   << ",\"subdraw\":" << in.subdraw_index
		   << ",\"vp_storage_hash\":\"" << std::hex << storage_hash << std::dec << '"'
		   << ",\"vp_ucode_hash\":\"" << std::hex << ucode_hash << std::dec << '"'
		   << ",\"vp_session_id\":" << in.vp_session_id
		   << ",\"fp_session_id\":" << in.fp_session_id;

		// Primitive / vertex signature
		os << ",\"primitive\":" << static_cast<u32>(regs.current_draw_clause.primitive)
		   << ",\"draw_command\":" << static_cast<u32>(regs.current_draw_clause.command)
		   << ",\"indexed\":" << (in.indexed ? "true" : "false")
		   << ",\"vertex_draw_count\":" << in.vertex_draw_count
		   << ",\"pass_count\":" << in.pass_count;

		// Enabled fragment textures: guest address, size and format.
		os << ",\"textures\":[";
		for (u32 i = 0, n = 0; i < rsx::limits::fragment_textures_count; ++i)
		{
			const auto& tex = regs.fragment_textures[i];
			if (!tex.enabled()) continue;
			if (n++) os << ',';
			os << "{\"unit\":" << i << ",\"address\":" << rsx::get_address(tex.offset(), tex.location())
			   << ",\"width\":" << tex.width() << ",\"height\":" << tex.height() << ",\"format\":" << static_cast<u32>(tex.format()) << '}';
		}
		os << ']';

		// Render-target identity. Guest addresses are the stable semantic key;
		// Vulkan handles are deliberately not recorded.
		os << ",\"rt\":{\"color_addresses\":[";
		for (int i = 0; i < 4; ++i)
		{
			if (i) os << ',';
			os << fb.color_addresses[i];
		}
		os << "],\"color_pitch\":[";
		for (int i = 0; i < 4; ++i)
		{
			if (i) os << ',';
			os << fb.color_pitch[i];
		}
		os << "],\"color_write_enabled\":[";
		for (int i = 0; i < 4; ++i)
		{
			if (i) os << ',';
			os << (fb.color_write_enabled[i] ? "true" : "false");
		}
		os << "],\"zeta_address\":" << fb.zeta_address
		   << ",\"zeta_pitch\":" << fb.zeta_pitch
		   << ",\"zeta_write_enabled\":" << (fb.zeta_write_enabled ? "true" : "false")
		   << ",\"width\":" << fb.width
		   << ",\"height\":" << fb.height
		   << ",\"target\":" << static_cast<u32>(fb.target)
		   << ",\"color_format\":" << static_cast<u32>(fb.color_format)
		   << ",\"depth_format\":" << static_cast<u32>(fb.depth_format)
		   << ",\"aa_mode\":" << static_cast<u32>(fb.aa_mode)
		   << "}";

		// Render state used by the draw classifier.
		os << ",\"state\":{"
		   <<   "\"viewport\":[" << regs.viewport_origin_x() << ',' << regs.viewport_origin_y()
		   <<     ',' << regs.viewport_width() << ',' << regs.viewport_height() << ']'
		   <<   ",\"scissor\":[" << regs.scissor_origin_x() << ',' << regs.scissor_origin_y()
		   <<     ',' << regs.scissor_width() << ',' << regs.scissor_height() << ']'
		   <<   ",\"depth_test\":" << (regs.depth_test_enabled() ? "true" : "false")
		   <<   ",\"depth_write\":" << (regs.depth_write_enabled() ? "true" : "false")
		   <<   ",\"depth_func\":" << static_cast<u32>(regs.depth_func())
		   <<   ",\"blend\":" << (regs.blend_enabled() ? "true" : "false")
		   << "}";

		// Transform constants, by ORIGINAL guest index (c[0]..c[467]).
		// A statically addressed program reports only what it reads. A program
		// that indexes the bank dynamically gets a conservative full-bank dump,
		// explicitly marked so the offline tool never mistakes it for evidence
		// that the shader reads all of it.
		os << ",\"constants_mode\":\"" << (in.has_indexed_constants ? "full_bank_conservative" : "static") << '"'
		   << ",\"constants\":[";

		bool first = true;
		if (in.has_indexed_constants || !in.constant_ids)
		{
			for (u32 i = 0; i < 468; ++i)
			{
				append_constant(os, i, regs.transform_constants[i], first);
				first = false;
			}
		}
		else
		{
			for (u16 id : *in.constant_ids)
			{
				if (id >= 468)
				{
					continue;
				}

				append_constant(os, id, regs.transform_constants[id], first);
				first = false;
			}
		}

		os << ']';

		// There is no per-draw RSX eye selector in this tree. Ownership must be
		// inferred offline from render-target provenance and paired draw structure.
		os << ",\"eye\":\"unknown\"}";

		write_line(os.str());
	}
}
