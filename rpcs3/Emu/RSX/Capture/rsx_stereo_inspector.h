#pragma once

// RSX Stereo Inspector (VR fork)
//
// A default-off, one-frame, per-draw evidence capture used to discover which
// vertex program and which original RSX transform constants encode WipEout's
// camera transform. This is a discovery tool only: it never alters guest state,
// never alters the RSX command stream, and emits nothing unless explicitly
// enabled and explicitly armed.
//
// Enable:  set RPCS3_STEREO_INSPECT to an existing output directory.
//          If unset, every hook below costs one relaxed bool load.
// Arm:     create a file named "ARM" inside that directory. The inspector
//          consumes it at the next frame boundary and captures exactly one frame.
//
// Output is JSON Lines so that the offline pairing tool stays decoupled from
// whatever the renderer eventually becomes. Constants are written as raw u32
// bits (authoritative) alongside decoded floats (convenience).

#include <string>
#include <vector>
#include <unordered_set>

#include "util/types.hpp"
#include "util/atomic.hpp"
#include "Utilities/File.h"

struct RSXVertexProgram;
struct RSXFragmentProgram;

namespace rsx
{
	struct framebuffer_layout;

	namespace vr
	{
		// Backend-supplied facts about one emitted subdraw. Everything else is
		// read from rsx::method_registers, which is the pristine guest bank.
		struct draw_capture_input
		{
			u32 subdraw_index = 0;
			u32 vertex_draw_count = 0;
			bool indexed = false;
			u32 pass_count = 1;

			const RSXVertexProgram* vertex_program = nullptr;
			const RSXFragmentProgram* fragment_program = nullptr;

			// Original guest constant map for the translated program.
			const std::vector<u16>* constant_ids = nullptr;
			bool has_indexed_constants = false;
			u32 vp_session_id = 0;
			u32 fp_session_id = 0;   // shaderlog/FragmentProgram<id> with "Log shader programs"

			const framebuffer_layout* framebuffer = nullptr;
		};

		class stereo_inspector
		{
		public:
			static stereo_inspector& get();

			// Hot-path gate. False unless RPCS3_STEREO_INSPECT was set.
			bool enabled() const { return m_enabled; }

			// True only during the single armed frame.
			bool capturing() const { return m_capturing.load(); }

			// Frame boundary: finalize an in-flight capture, then arm if requested.
			void on_frame_end();

			// Once per logical RSX begin/end clause, before subdraws are emitted.
			void begin_draw_clause();

			// Once per emitted subdraw, after vertex upload and before vkCmdDraw*.
			void record_draw(const draw_capture_input& in);

		private:
			stereo_inspector();

			void open_capture();
			void close_capture();
			void emit_shader_record(const draw_capture_input& in, usz ucode_hash, usz storage_hash);
			void write_line(const std::string& line);

			bool m_enabled = false;
			std::string m_out_dir;
			std::string m_arm_path;

			atomic_t<bool> m_capturing{false};
			atomic_t<bool> m_arm_next{false};

			fs::file m_file;
			u64 m_frame_counter = 0;
			u64 m_capture_frame = 0;
			u32 m_draw_ordinal = 0;
			u32 m_vk_draw_commands = 0;
			std::unordered_set<usz> m_seen_shaders;
		};
	}
}
