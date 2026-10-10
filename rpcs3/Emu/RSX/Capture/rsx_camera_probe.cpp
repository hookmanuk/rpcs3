#include "stdafx.h"
#include "rsx_camera_probe.h"
#include "rsx_vr_profile_generator.h"
#include <array>
#include <set>

#include "Emu/System.h"
#include "Emu/Memory/vm.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "Emu/IdManager.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/rsx_methods.h"

#include "util/logs.hpp"
#include "util/yaml.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

LOG_CHANNEL(vr_probe_log, "VRPROBE");

// cellVdec.cpp: open video decoders (see effective_vblank_rate).
u32 vdec_open_count();

namespace rsx::vr
{
	namespace
	{
		using mat4 = std::array<std::array<f32, 4>, 4>;

		// The address a profile target names now, or 0 (a pointer on the way is null or unreadable).
		u32 resolve_guest_address(const title_profile::guest_address& target)
		{
			u32 address = target.address;
			if (!target.deref)
			{
				return address;
			}
			for (usz level = 0; level <= target.inner_offsets.size(); level++)
			{
				if (!vm::check_addr(address, vm::page_readable, 4) || !(address = vm::_ref<be_t<u32>>(address)))
				{
					return 0;
				}
				address += level < target.inner_offsets.size() ? target.inner_offsets[level] : target.offset;
			}
			return address;
		}

		mat4 identity()
		{
			mat4 m{};
			for (int i = 0; i < 4; ++i)
				m[i][i] = 1.f;
			return m;
		}

		// Row-vector convention throughout: v' = v * M.
		mat4 mul(const mat4& a, const mat4& b)
		{
			mat4 r{};
			for (int i = 0; i < 4; ++i)
			{
				for (int j = 0; j < 4; ++j)
				{
					f32 s = 0.f;
					for (int k = 0; k < 4; ++k)
					{
						s += a[i][k] * b[k][j];
					}
					r[i][j] = s;
				}
			}
			return r;
		}

		mat4 translate(f32 x, f32 y, f32 z)
		{
			mat4 m = identity();
			m[3][0] = x;
			m[3][1] = y;
			m[3][2] = z;
			return m;
		}

		mat4 rot_x(f32 rad)
		{
			mat4 m = identity();
			const f32 c = std::cos(rad), s = std::sin(rad);
			m[1][1] = c;
			m[1][2] = s;
			m[2][1] = -s;
			m[2][2] = c;
			return m;
		}

		mat4 rot_y(f32 rad)
		{
			mat4 m = identity();
			const f32 c = std::cos(rad), s = std::sin(rad);
			m[0][0] = c;
			m[0][2] = -s;
			m[2][0] = s;
			m[2][2] = c;
			return m;
		}

		mat4 rot_z(f32 rad)
		{
			mat4 m = identity();
			const f32 c = std::cos(rad), s = std::sin(rad);
			m[0][0] = c;
			m[0][1] = s;
			m[1][0] = -s;
			m[1][1] = c;
			return m;
		}

		std::string read_env(const char* name)
		{
#ifdef _WIN32
			char* buf = nullptr;
			usz sz = 0;
			if (_dupenv_s(&buf, &sz, name) == 0 && buf)
			{
				std::string v(buf);
				std::free(buf);
				return v;
			}
			return {};
#else
			if (const char* v = ::getenv(name))
				return v;
			return {};
#endif
		}

		// The directly read slots of the current full-bank program (full_bank_direct_slots), or none.
		struct direct_slot_filter
		{
			const std::vector<u16>* ids = nullptr;
			std::array<bool, 468> read{};
		};
		thread_local direct_slot_filter t_direct_slots;

		// Locate a guest constant slot inside the transient buffer.
		// Returns nullptr when this program does not read that slot.
		f32* find_slot(void* buffer, const u16* reloc, usz reloc_size, u32 guest_index)
		{
			char* base = static_cast<char*>(buffer);

			if (reloc_size == 0)
			{
				// Full-bank upload: natural indexing.
				if (guest_index >= 468)
					return nullptr;
				if (t_direct_slots.ids && !t_direct_slots.read[guest_index])
					return nullptr;
				return reinterpret_cast<f32*>(base + guest_index * 16);
			}

			if (guest_index >= 468)
				return nullptr;

			// The relocation table is a vertex program's constant_ids, built from a std::set: sorted and unique,
			// so a binary search finds the slot. (A cache of 8 slot -> index tables, rebuilt on a miss, thrashed in
			// scenes with many large programs: ~10% of the RSX thread in Ratchet & Clank.)
			const u16* const end = reloc + reloc_size;
			const u16* const it = std::lower_bound(reloc, end, static_cast<u16>(guest_index));
			return it != end && *it == guest_index ? reinterpret_cast<f32*>(base + (it - reloc) * 16) : nullptr;
		}

		// A 4-slot matrix in the row-vector convention all the math here uses:
		// rows[k] is one row of M (clip = v * M), so column j produces clip[j].
		// A column_vectors block stores M transposed (slot i is the row DP4 reads
		// for clip[i]); it is worked on as a transposed copy and written back.
		// A DP4 program may leave the z slot out and take clip z from the w row
		// (a sky drawn on the far plane, e.g. Pure's c[26], c[27], c[29]); its z
		// row is then w, and only the slots it reads are written back.
		class matrix_block
		{
		public:
			matrix_block() = default;
			matrix_block(const matrix_block&) = delete;
			matrix_block& operator=(const matrix_block&) = delete;

			~matrix_block()
			{
				if (m_transposed)
				{
					for (u32 j = 0; j < 4; ++j)
					{
						if (!m_slots[j])
							continue;
						for (u32 i = 0; i < 4; ++i)
							m_slots[j][i] = m_local[i][j];
					}
				}
			}

			// False (and nothing bound) if the program does not read all 4 slots.
			// xyw: DP4 slots base, base+1, base+2 are clip x, y, w (no z slot).
			// explicit_slots: the 4 slots themselves, when not contiguous.
			// flat_ok: row_vectors only; a program that does not read the z slot takes no
			// input z (a 2D HUD: Demon's Souls reads c[0], c[1], c[3]). Its z row is a
			// zero scratch row, never written back.
			bool bind(void* buffer, const u16* reloc, usz reloc_size, u32 base, bool column_vectors, bool xyw = false,
				const std::array<u32, 4>* explicit_slots = nullptr, bool flat_ok = false)
			{
				release();
				column_vectors |= xyw;
				u32 slot_of[4] = {base, base + 1, xyw ? umax : base + 2, xyw ? base + 2 : base + 3};
				if (explicit_slots && (*explicit_slots)[0] != umax)
				{
					for (u32 k = 0; k < 4; ++k)
						slot_of[k] = (*explicit_slots)[k];
				}
				for (u32 k = 0; k < 4; ++k)
				{
					m_slots[k] = slot_of[k] == umax ? nullptr : find_slot(buffer, reloc, reloc_size, slot_of[k]);
					if (!m_slots[k] && !(column_vectors && k == 2) && !(flat_ok && !column_vectors && k == 2))
					{
						return false;
					}
				}

				m_transposed = column_vectors;
				m_base = base;
				// The program left out a z slot the layout has: z = w, drawn on the far plane (a sky).
				m_far_plane = !xyw && slot_of[2] != umax && !m_slots[2];
				for (u32 k = 0; k < 4; ++k)
				{
					rows[k] = m_transposed ? m_local[k] : m_slots[k];
				}
				if (!m_transposed && !rows[2])
				{
					std::fill(std::begin(m_flat_z), std::end(m_flat_z), 0.f);
					rows[2] = m_flat_z;
				}
				if (m_transposed)
				{
					for (u32 i = 0; i < 4; ++i)
						for (u32 j = 0; j < 4; ++j)
							m_local[i][j] = m_slots[m_slots[j] ? j : 3][i]; // only z may be absent: z = w
				}
				return true;
			}

			// A sky: drawn on the far plane (the program has no z slot, z = w).
			bool far_plane() const
			{
				return m_far_plane;
			}

			// The first constant slot of the bound block.
			u32 base() const
			{
				return m_base;
			}

			// The bound slot k in the buffer (null when the program lacks it), and the layout.
			const f32* slot(u32 k) const
			{
				return m_slots[k];
			}
			bool transposed() const
			{
				return m_transposed;
			}

			// Drop the binding without writing back (the block was not modified).
			void release()
			{
				m_transposed = false;
				m_far_plane = false;
				m_base = umax;
				for (f32*& r : rows)
					r = nullptr;
			}

			f32* rows[4] = {};

		private:
			f32* m_slots[4] = {};
			f32 m_local[4][4] = {};
			f32 m_flat_z[4] = {};
			u32 m_base = umax;
			bool m_transposed = false;
			bool m_far_plane = false;
		};

		// The camera position slot holds the eye point: w = 1 and, when the camera block has a finite eye
		// (clip x = y = w = 0 there), close to it. Gran Turismo 5 keeps the camera position in c[467] for most
		// programs but per-channel fog densities there in its car-shadow program (w 0.991); offsetting those
		// by the eye tinted the shadows red in one eye and green in the other.
		bool plausible_camera_position(const f32 cam[4], const f32 m[4][4], f32 eye_baseline)
		{
			if (!std::isfinite(cam[3]) || std::fabs(cam[3] - 1.f) > 1e-4f)
			{
				return false;
			}
			// Row-vector convention: clip[j] = sum_k p[k] * m[k][j] + m[3][j]. Solve clip x, y, w = 0.
			const f64 a[3][3] = {
				{m[0][0], m[1][0], m[2][0]},
				{m[0][1], m[1][1], m[2][1]},
				{m[0][3], m[1][3], m[2][3]}};
			const f64 b[3] = {-m[3][0], -m[3][1], -m[3][3]};
			const f64 det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
			                a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
			const f64 scale = std::fabs(a[0][0]) + std::fabs(a[1][1]) + std::fabs(a[2][2]) + 1e-30;
			if (std::fabs(det) < 1e-9 * scale * scale * scale)
			{
				return true; // no finite eye to compare with
			}
			f64 p[3];
			for (u32 i = 0; i < 3; ++i)
			{
				f64 t[3][3];
				for (u32 r = 0; r < 3; ++r)
					for (u32 c = 0; c < 3; ++c)
						t[r][c] = c == i ? b[r] : a[r][c];
				p[i] = (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) +
						   t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])) /
				       det;
			}
			const f64 dx = cam[0] - p[0], dy = cam[1] - p[1], dz = cam[2] - p[2];
			const f64 distance = std::sqrt(dx * dx + dy * dy + dz * dz);
			const f64 eye_distance = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
			return distance <= std::max(0.01 * eye_distance, 50.0 * eye_baseline);
		}

		bool is_perspective(f32* const r[4])
		{
			constexpr f32 eps = 1e-6f;
			return !(std::fabs(r[0][3]) < eps && std::fabs(r[1][3]) < eps &&
					 std::fabs(r[2][3]) < eps && std::fabs(r[3][3] - 1.f) < eps);
		}

		// A camera's clip w depends on the vertex position. An affine matrix whose w column is (0, 0, 0, k) passes
		// is_perspective when k != 1, but it is no projection: WipEout HD's Detonator spheres keep their world matrix
		// in c[256] with w = 5 beside the camera in c[260]; bound as the camera, the spheres kept the game's view.
		bool w_from_position(f32* const r[4])
		{
			constexpr f32 eps = 1e-6f;
			return std::fabs(r[0][3]) >= eps || std::fabs(r[1][3]) >= eps || std::fabs(r[2][3]) >= eps;
		}

		// Clip x, y and w directions (columns 0, 1, 3 of rows 0..2) mutually orthogonal.
		bool is_rigid(f32* const r[4])
		{
			const auto dot = [&](u32 i, u32 j)
			{
				return r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
			};
			const f32 n0 = std::sqrt(dot(0, 0)), n1 = std::sqrt(dot(1, 1)), n3 = std::sqrt(dot(3, 3));
			constexpr f32 tol = 0.1f;
			return n0 > 1e-8f && n1 > 1e-8f && n3 > 1e-8f &&
			       std::fabs(dot(0, 1)) <= tol * n0 * n1 &&
			       std::fabs(dot(0, 3)) <= tol * n0 * n3 &&
			       std::fabs(dot(1, 3)) <= tol * n1 * n3;
		}

		// A camera-facing sprite: the object's z axis is the view axis (input x and y do not reach clip w, input z
		// does not reach clip x or y), so the matrix is the projection times the sprite's own in-plane size. Its
		// x and y columns are orthogonal like a camera's, but their length is the projection's scale times the
		// sprite's size, not the projection's.
		bool is_screen_aligned(f32* const r[4])
		{
			const f32 n0 = std::sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0] + r[2][0] * r[2][0]);
			const f32 n1 = std::sqrt(r[0][1] * r[0][1] + r[1][1] * r[1][1] + r[2][1] * r[2][1]);
			const f32 n3 = std::sqrt(r[0][3] * r[0][3] + r[1][3] * r[1][3] + r[2][3] * r[2][3]);
			constexpr f32 tol = 1e-3f;
			return n0 > 1e-8f && n1 > 1e-8f && n3 > 1e-8f &&
			       std::fabs(r[0][3]) <= tol * n3 && std::fabs(r[1][3]) <= tol * n3 &&
			       std::fabs(r[2][0]) <= tol * n0 && std::fabs(r[2][1]) <= tol * n1;
		}

		// |clip y| / |clip x| of the block against the output aspect (10%).
		bool has_camera_aspect(f32* const r[4], f32 aspect)
		{
			const f32 nx = std::sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0] + r[2][0] * r[2][0]);
			const f32 ny = std::sqrt(r[0][1] * r[0][1] + r[1][1] * r[1][1] + r[2][1] * r[2][1]);
			return nx > 1e-8f && std::fabs((ny / nx) / aspect - 1.f) <= 0.1f;
		}

		// Bind the first perspective (and, if required, rigid and output-aspect) block of the candidates.
		bool bind_camera_block(matrix_block& block, void* buffer, const u16* reloc, usz reloc_size,
			std::span<const u32> candidates, bool column_vectors, bool require_rigid, bool xyw = false, f32 require_aspect = 0.f,
			std::span<const std::array<u32, 4>> explicit_slots = {}, std::span<const u32> nonrigid = {}, std::span<const u32> row_blocks = {},
			std::span<const u32> either_layout = {})
		{
			for (usz i = 0; i < candidates.size(); ++i)
			{
				const u32 candidate = candidates[i];
				const std::array<u32, 4>* slots = i < explicit_slots.size() ? &explicit_slots[i] : nullptr;
				const bool rows = std::find(row_blocks.begin(), row_blocks.end(), candidate) != row_blocks.end();
				const bool either = std::find(either_layout.begin(), either_layout.end(), candidate) != either_layout.end();
				for (u32 pass = 0; pass < (either ? 2u : 1u); ++pass)
				{
					// Profile either_layout_blocks: the second pass binds the block in the other layout.
					const bool columns = (column_vectors != rows) != (pass != 0);
					if (block.bind(buffer, reloc, reloc_size, candidate, columns, xyw && !rows && pass == 0, slots) && is_perspective(block.rows) &&
						w_from_position(block.rows) &&
						(!require_rigid || std::find(nonrigid.begin(), nonrigid.end(), candidate) != nonrigid.end() || is_rigid(block.rows)) &&
						(require_aspect <= 0.f || has_camera_aspect(block.rows, require_aspect)))
					{
						return true;
					}
				}
			}
			block.release();
			return false;
		}
	} // namespace

	const title_profile::stereo_rule& title_profile::stereo_for(u32 target_width, u32 output_width) const
	{
		for (const stereo_rule& rule : stereo_by_target_width)
		{
			if (target_width * rule.output_width_divisor == output_width)
			{
				return rule;
			}
		}
		return stereo;
	}

	// The running executable's file name, lower case, without extension ("shadow" for
	// shadow.self): collections run several games under one title ID (BCUS98259: ICO.self,
	// shadow.self and the menu's EBOOT.BIN), each with its own shaders and camera.
	std::string running_executable_name()
	{
		std::string name = Emu.GetBoot();
		if (const usz slash = name.find_last_of("/\\"); slash != umax)
		{
			name = name.substr(slash + 1);
		}
		if (const usz dot = name.find_last_of('.'); dot != umax && dot)
		{
			name = name.substr(0, dot);
		}
		for (char& c : name)
		{
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		}
		return name;
	}

	std::shared_ptr<const title_profile> load_title_profile(std::string_view title_id, std::string_view executable)
	{
		if (title_id.empty())
		{
			return nullptr;
		}

		// vr_profiles/<TITLE_ID>.<executable>.json for one game of a collection, else vr_profiles/<TITLE_ID>.json.
		const std::string dir = fs::get_executable_dir() + "vr_profiles/";
		std::string path = dir + std::string(title_id) + ".json";
		if (!executable.empty())
		{
			if (std::string specific = dir + std::string(title_id) + "." + std::string(executable) + ".json"; fs::is_file(specific))
			{
				path = std::move(specific);
			}
		}
		fs::file file(path);
		if (!file)
		{
			return nullptr;
		}

		// JSON is YAML flow syntax, so RPCS3's YAML reader parses it.
		const auto [root, parse_error] = yaml_load(file.to_string());
		if (!parse_error.empty())
		{
			vr_probe_log.error("VR profile '%s' is not valid JSON: %s", path, parse_error);
			return nullptr;
		}

		// emucore is built without C++ exceptions: every lookup goes through
		// get_yaml_node_value (util/yaml.cpp catches), and missing keys are checked.
		std::string error;
		const auto child = [](const YAML::Node& parent, const char* key)
		{
			return parent && parent.IsMap() ? parent[key] : YAML::Node(YAML::NodeType::Undefined);
		};
		const auto read = [&]<typename T>(const YAML::Node& parent, const char* key, T& out, bool required = true) -> bool
		{
			const YAML::Node node = child(parent, key);
			if (!node)
			{
				if (required && error.empty())
					error = fmt::format("missing \"%s\"", key);
				return false;
			}
			std::string node_error;
			T value = get_yaml_node_value<T>(node, node_error);
			if (!node_error.empty())
			{
				if (error.empty())
					error = fmt::format("\"%s\": %s", key, node_error);
				return false;
			}
			out = std::move(value);
			return true;
		};
		const auto read_rule = [&](const YAML::Node& node, title_profile::stereo_rule& rule)
		{
			read(node, "per_eye_separation", rule.per_eye_separation);
			read(node, "convergence", rule.convergence);
		};
		const auto fail = [&](std::string what)
		{
			if (error.empty())
				error = std::move(what);
		};
		// Unknown keys are only warned about, but that catches typos in optional ones.
		const auto check_keys = [&](const YAML::Node& node, const char* where, std::initializer_list<std::string_view> known)
		{
			if (!node || !node.IsMap())
				return;
			for (const auto& entry : node)
			{
				std::string key_error;
				const std::string key = get_yaml_node_value<std::string>(entry.first, key_error);
				if (std::find(known.begin(), known.end(), key) == known.end())
				{
					vr_probe_log.warning("VR profile '%s': unknown key \"%s\"%s ignored.", path, key, where);
				}
			}
		};

		if (!root || !root.IsMap())
		{
			fail("the profile is not a JSON object");
		}

		if (u32 schema = 0; read(root, "schema", schema) && schema != 1)
		{
			fail(fmt::format("schema %u is not supported (1 only)", schema));
		}

		auto profile = std::make_shared<title_profile>();
		if (read(root, "title_id", profile->title_id) && profile->title_id != title_id)
		{
			fail(fmt::format("title_id is '%s'", profile->title_id));
		}

		read(root, "app_version", profile->app_version, false);
		read(root, "name", profile->name, false);

		if (std::string layout; read(root, "matrix_layout", layout))
		{
			if (layout == "column_vectors")
			{
				profile->column_vectors = true;
			}
			else if (layout == "column_vectors_xyw")
			{
				profile->column_vectors = true;
				profile->xyw_rows = true;
			}
			else if (layout != "row_vectors")
			{
				fail(fmt::format("matrix_layout '%s' is not supported (row_vectors, column_vectors or column_vectors_xyw)", layout));
			}
		}

		if (const YAML::Node blocks = child(root, "camera_blocks"); blocks && blocks.IsSequence())
		{
			for (const auto& block : blocks)
			{
				std::string node_error;
				std::array<u32, 4> slots{umax, umax, umax, umax};
				if (block.IsSequence())
				{
					// An explicit list of the block's 4 slots.
					if (block.size() != 4)
					{
						fail("camera_blocks: a slot list needs 4 slots");
						continue;
					}
					for (u32 k = 0; k < 4; ++k)
					{
						slots[k] = get_yaml_node_value<u32>(block[k], node_error);
					}
					profile->camera_blocks.push_back(slots[0]);
				}
				else
				{
					profile->camera_blocks.push_back(get_yaml_node_value<u32>(block, node_error));
				}
				profile->camera_block_slots.push_back(slots);
				if (!node_error.empty())
					fail("camera_blocks: " + node_error);
			}
		}
		if (profile->camera_blocks.empty())
		{
			fail("camera_blocks must list at least one slot");
		}

		read(root, "output_aspect_tolerance", profile->output_aspect_tolerance);
		read(root, "camera_target_aspect", profile->camera_target_aspect, false);

		if (const YAML::Node aspects = child(root, "game_camera_aspects"); aspects && aspects.IsSequence())
		{
			for (const auto& aspect : aspects)
			{
				const f32 value = aspect.as<f32>();
				if (value <= 0.f)
				{
					fail("game_camera_aspects: expected positive |clip y| / |clip x| ratios");
					continue;
				}
				profile->game_camera_aspects.push_back(value);
			}
		}
		if (const YAML::Node widths = child(root, "game_camera_target_widths"); widths && widths.IsSequence())
		{
			for (const auto& width : widths)
			{
				std::string node_error;
				profile->game_camera_target_widths.push_back(get_yaml_node_value<u32>(width, node_error));
				if (!node_error.empty())
					fail("game_camera_target_widths: " + node_error);
			}
		}

		const YAML::Node camera_position = child(root, "camera_position");
		read(camera_position, "eye_baseline", profile->eye_baseline);
		read(camera_position, "slot", profile->camera_position_slot, false);

		const YAML::Node stereo = child(root, "stereo");
		if (std::string formula; read(stereo, "formula", formula) && formula != "clip_x_shear")
		{
			fail(fmt::format("stereo formula '%s' is not supported (clip_x_shear only)", formula));
		}
		read_rule(stereo, profile->stereo);
		if (std::string offset; read(stereo, "eye_offset", offset, false))
		{
			if (offset != "baseline" && offset != "baseline_per_w" && offset != "shear")
			{
				fail(fmt::format("stereo eye_offset '%s' is not supported (baseline, baseline_per_w or shear)", offset));
			}
			profile->stereo_eye_offset_from_baseline = offset == "baseline" || offset == "baseline_per_w";
			profile->stereo_eye_offset_per_w = offset == "baseline_per_w";
		}
		if (const YAML::Node rules = child(stereo, "by_target_width"); rules && rules.IsSequence())
		{
			for (const auto& node : rules)
			{
				title_profile::stereo_rule rule;
				read(node, "output_width_divisor", rule.output_width_divisor);
				if (!rule.output_width_divisor)
				{
					fail("output_width_divisor must be at least 1");
				}
				read_rule(node, rule);
				profile->stereo_by_target_width.push_back(rule);
			}
		}

		const YAML::Node screen_space = child(root, "screen_space");
		read(screen_space, "orthographic_block", profile->screen_space_block, false);
		if (const YAML::Node programs = child(screen_space, "hud_block_programs"); programs && programs.IsSequence())
		{
			// [{ "program": "<vertex ucode hash>", "block": 256 }]
			for (const YAML::Node& node : programs)
			{
				std::string program;
				u32 block = umax;
				read(node, "program", program);
				read(node, "block", block);
				char* end = nullptr;
				const u64 hash = std::strtoull(program.c_str(), &end, 16);
				if (program.empty() || !end || *end || block >= 468)
				{
					fail("screen_space.hud_block_programs: expected {\"program\": \"<vertex ucode hash>\", \"block\": <slot>}");
					continue;
				}
				profile->screen_space_hud_block_programs.push_back({hash, block});
			}
		}
		if (std::string layout; read(screen_space, "orthographic_block_layout", layout, false))
		{
			if (layout == "row_vectors")
				profile->screen_space_block_rows = true;
			else if (layout != "column_vectors")
				fail(fmt::format("screen_space.orthographic_block_layout '%s' is not supported (row_vectors or column_vectors)", layout));
		}
		if (const YAML::Node bare_blocks = child(screen_space, "bare_projection"); bare_blocks && bare_blocks.IsSequence())
		{
			for (const auto& node : bare_blocks)
				profile->screen_space_bare_projection_blocks.push_back(node.as<u32>());
			profile->screen_space_bare_projection = !profile->screen_space_bare_projection_blocks.empty();
		}
		else if (std::string bare; read(screen_space, "bare_projection", bare, false))
		{
			profile->screen_space_bare_projection = bare == "true";
		}
		if (std::string offset; read(screen_space, "depth_offset_projection", offset, false))
		{
			profile->screen_space_depth_offset_projection = offset == "true";
		}
		if (std::string offaspect; read(screen_space, "offaspect_projection", offaspect, false))
		{
			profile->screen_space_offaspect_projection = offaspect == "true";
		}
		if (std::string rotation; read(screen_space, "rotation_only_passthrough", rotation, false))
		{
			profile->screen_space_rotation_only_passthrough = rotation == "true";
		}
		if (std::string skips; read(screen_space, "hud_skips_passes", skips, false))
		{
			profile->screen_space_hud_skips_passes = skips == "true";
		}
		if (std::string clear; read(screen_space, "clear_outside_box", clear, false))
		{
			profile->screen_space_clear_outside_box = clear == "true";
		}
		if (std::string only; read(screen_space, "hud_display_buffers_only", only, false))
		{
			profile->screen_space_hud_display_buffers_only = only == "true";
		}
		if (std::string after; read(screen_space, "hud_box_after_shader", after, false))
		{
			profile->screen_space_hud_box_after_shader = after == "true";
		}
		if (std::string hud; read(screen_space, "passthrough_hud", hud, false))
		{
			profile->screen_space_passthrough_hud = hud == "true";
		}
		if (std::string keep; read(screen_space, "hud_keep_depth", keep, false))
		{
			profile->screen_space_hud_keep_depth = keep == "true";
		}
		if (std::string screen; read(screen_space, "frames_without_3d_as_screen", screen, false))
		{
			profile->screen_space_frames_without_3d_as_screen = screen == "true"  ? title_profile::frames_without_3d_mode::always :
			                                                    screen == "false" ? title_profile::frames_without_3d_mode::never :
			                                                                        title_profile::frames_without_3d_mode::automatic;
		}
		if (const YAML::Node programs = child(screen_space, "preprojected_programs"); programs && programs.IsSequence())
		{
			// "<vertex ucode hash>", or { "program": "<hash>", "without_depth_test": true } for a program
			// whose scene draws also take the eye transform with depth test off.
			for (const auto& program : programs)
			{
				std::string text, untested;
				if (program.IsMap())
				{
					read(program, "program", text);
					read(program, "without_depth_test", untested, false);
				}
				else
				{
					text = program.as<std::string>();
				}
				char* end = nullptr;
				const u64 hash = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end)
					fail("screen_space.preprojected_programs: '" + text + "' is not a hex program hash");
				profile->screen_space_preprojected_programs.push_back(hash);
				if (untested == "true")
					profile->screen_space_preprojected_untested.push_back(hash);
			}
		}
		if (const YAML::Node draws = child(root, "hidden_draws"); draws && draws.IsSequence())
		{
			for (const YAML::Node& node : draws)
			{
				std::string name, program, texture, hidden = "true";
				read(node, "name", name, false);
				read(node, "program", program);
				read(node, "texture", texture);
				read(node, "hidden", hidden, false);
				char* end = nullptr;
				title_profile::unboxed_draw draw{};
				draw.program = std::strtoull(program.c_str(), &end, 16);
				u32 w = 0, h = 0;
				if (program.empty() || !end || *end || std::sscanf(texture.c_str(), "%ux%u", &w, &h) != 2 || (hidden != "true" && hidden != "false"))
				{
					fail("hidden_draws: expected {\"name\": \"...\", \"program\": \"<vertex ucode hash>\", \"texture\": \"<width>x<height>\", \"hidden\": true|false}");
					continue;
				}
				draw.width = static_cast<u16>(w);
				draw.height = static_cast<u16>(h);
				if (hidden == "true")
				{
					profile->hidden_draws.push_back(draw);
				}
				vr_probe_log.notice("VR profile: %s %s.", name.empty() ? program : name, hidden == "true" ? "hidden" : "shown");
			}
		}
		if (const YAML::Node draws = child(screen_space, "scaled_draws"); draws && draws.IsSequence())
		{
			for (const YAML::Node& node : draws)
			{
				std::string program, texture, scale;
				read(node, "program", program);
				read(node, "texture", texture);
				read(node, "scale", scale);
				char* end = nullptr;
				title_profile::scaled_draw draw{};
				draw.program = std::strtoull(program.c_str(), &end, 16);
				u32 w = 0, h = 0;
				draw.scale = static_cast<f32>(std::atof(scale.c_str()));
				if (program.empty() || !end || *end || std::sscanf(texture.c_str(), "%ux%u", &w, &h) != 2 || !(draw.scale > 0.f && draw.scale <= 10.f))
				{
					fail("screen_space.scaled_draws: expected {\"program\": \"<vertex ucode hash>\", \"texture\": \"<width>x<height>\", \"scale\": <0-10>}");
					continue;
				}
				draw.width = static_cast<u16>(w);
				draw.height = static_cast<u16>(h);
				profile->screen_space_scaled_draws.push_back(draw);
			}
		}
		if (const YAML::Node draws = child(screen_space, "unboxed_draws"); draws && draws.IsSequence())
		{
			for (const YAML::Node& node : draws)
			{
				std::string program, texture;
				read(node, "program", program);
				read(node, "texture", texture);
				char* end = nullptr;
				title_profile::unboxed_draw draw{};
				draw.program = std::strtoull(program.c_str(), &end, 16);
				u32 w = 0, h = 0;
				if (program.empty() || !end || *end || std::sscanf(texture.c_str(), "%ux%u", &w, &h) != 2)
				{
					fail("screen_space.unboxed_draws: expected {\"program\": \"<vertex ucode hash>\", \"texture\": \"<width>x<height>\"}");
					continue;
				}
				draw.width = static_cast<u16>(w);
				draw.height = static_cast<u16>(h);
				profile->screen_space_unboxed_draws.push_back(draw);
			}
		}
		if (const YAML::Node reduced = child(root, "reduced_scale_frames"); reduced && reduced.IsMap())
		{
			if (std::string scale; read(reduced, "scale", scale, false))
			{
				profile->reduced_scale_percent = static_cast<u16>(std::clamp(std::atoi(scale.c_str()), 25, 800));
			}
			if (const YAML::Node draws = child(reduced, "draws"); draws && draws.IsSequence())
			{
				for (const YAML::Node& node : draws)
				{
					std::string program, texture;
					read(node, "program", program);
					read(node, "texture", texture);
					char* end = nullptr;
					title_profile::unboxed_draw draw{};
					draw.program = std::strtoull(program.c_str(), &end, 16);
					u32 w = 0, h = 0;
					if (program.empty() || !end || *end || std::sscanf(texture.c_str(), "%ux%u", &w, &h) != 2)
					{
						fail("reduced_scale_frames.draws: expected {\"program\": \"<vertex ucode hash>\", \"texture\": \"<width>x<height>\"}");
						continue;
					}
					draw.width = static_cast<u16>(w);
					draw.height = static_cast<u16>(h);
					profile->reduced_scale_draws.push_back(draw);
				}
			}
		}
		if (const YAML::Node draws = child(root, "frame_rate_draws"); draws && draws.IsSequence())
		{
			for (const YAML::Node& node : draws)
			{
				std::string program, texture;
				read(node, "program", program);
				read(node, "texture", texture);
				char* end = nullptr;
				title_profile::unboxed_draw draw{};
				draw.program = std::strtoull(program.c_str(), &end, 16);
				u32 w = 0, h = 0;
				if (program.empty() || !end || *end || std::sscanf(texture.c_str(), "%ux%u", &w, &h) != 2)
				{
					fail("frame_rate_draws: expected {\"program\": \"<vertex ucode hash>\", \"texture\": \"<width>x<height>\"}");
					continue;
				}
				draw.width = static_cast<u16>(w);
				draw.height = static_cast<u16>(h);
				u32 min_count = 1;
				read(node, "min_count", min_count, false);
				profile->frame_rate_draws.push_back(draw);
				profile->frame_rate_draw_min_counts.push_back(std::max(min_count, 1u));
			}
		}
		if (const YAML::Node draws = child(screen_space, "screen_frame_draws"); draws && draws.IsSequence())
		{
			for (const YAML::Node& node : draws)
			{
				std::string program, texture;
				read(node, "program", program);
				read(node, "texture", texture);
				char* end = nullptr;
				title_profile::unboxed_draw draw{};
				draw.program = std::strtoull(program.c_str(), &end, 16);
				u32 w = 0, h = 0;
				if (program.empty() || !end || *end || std::sscanf(texture.c_str(), "%ux%u", &w, &h) != 2)
				{
					fail("screen_space.screen_frame_draws: expected {\"program\": \"<vertex ucode hash>\", \"texture\": \"<width>x<height>\"}");
					continue;
				}
				draw.width = static_cast<u16>(w);
				draw.height = static_cast<u16>(h);
				profile->screen_space_screen_frame_draws.push_back(draw);
			}
		}
		if (const YAML::Node programs = child(screen_space, "hud_exact_depth_programs"); programs && programs.IsSequence())
		{
			for (const auto& program : programs)
			{
				const std::string text = program.as<std::string>();
				char* end = nullptr;
				const u64 hash = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end)
					fail("screen_space.hud_exact_depth_programs: '" + text + "' is not a hex program hash");
				profile->screen_space_hud_exact_depth_programs.push_back(hash);
			}
		}
		if (const YAML::Node programs = child(screen_space, "hud_programs"); programs && programs.IsSequence())
		{
			for (const auto& program : programs)
			{
				const std::string text = program.as<std::string>();
				char* end = nullptr;
				const u64 hash = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end)
					fail("screen_space.hud_programs: '" + text + "' is not a hex program hash");
				profile->screen_space_hud_programs.push_back(hash);
			}
		}
		if (std::string fills; read(screen_space, "output_pixel_draws_not_hud", fills, false))
		{
			profile->screen_space_output_pixel_draws_not_hud = fills == "true";
		}
		if (const YAML::Node cameras = child(screen_space, "boxed_cameras"); cameras && cameras.IsSequence())
		{
			for (const YAML::Node& camera : cameras)
			{
				if (!camera.IsSequence() || camera.size() != 4)
				{
					fail("screen_space.boxed_cameras: expected a list of clip-w rows, [x, y, z, w] each");
					continue;
				}
				std::array<f32, 4> w_row{};
				for (usz i = 0; i < 4; ++i)
				{
					w_row[i] = camera[i].as<f32>();
				}
				profile->screen_space_boxed_cameras.push_back(w_row);
			}
		}
		if (std::string sub; read(screen_space, "subviewport_cameras_in_box", sub, false))
		{
			profile->screen_space_subviewport_cameras_in_box = sub == "true";
		}

		read(root, "reference_screen_width", profile->reference_screen_width, false);
		if (const YAML::Node blocks = child(root, "nonrigid_camera_blocks"); blocks && blocks.IsSequence())
		{
			for (const auto& node : blocks)
			{
				profile->nonrigid_camera_blocks.push_back(node.as<u32>());
			}
		}
		if (const YAML::Node blocks = child(root, "linked_camera_blocks"); blocks && blocks.IsSequence())
		{
			for (const auto& node : blocks)
			{
				std::string node_error;
				profile->linked_camera_blocks.push_back(get_yaml_node_value<u32>(node, node_error));
				if (!node_error.empty())
					fail("linked_camera_blocks: " + node_error);
			}
		}
		if (const YAML::Node palette = child(root, "camera_palette"); palette)
		{
			if (!palette.IsSequence() || palette.size() != 2)
			{
				fail("camera_palette: expected [first block, last block]");
			}
			else
			{
				profile->camera_palette_first = palette[0].as<u32>();
				profile->camera_palette_last = palette[1].as<u32>();
				if (profile->camera_palette_last < profile->camera_palette_first || profile->camera_palette_last > 464)
				{
					fail("camera_palette: first <= last <= 464");
				}
			}
		}
		for (const char* key : {"row_vector_blocks", "column_vector_blocks"})
		{
			if (const YAML::Node blocks = child(root, key); blocks && blocks.IsSequence())
			{
				for (const auto& node : blocks)
				{
					std::string node_error;
					profile->row_vector_blocks.push_back(get_yaml_node_value<u32>(node, node_error));
					if (!node_error.empty())
						fail(std::string(key) + ": " + node_error);
				}
			}
		}
		if (const YAML::Node blocks = child(root, "either_layout_blocks"); blocks && blocks.IsSequence())
		{
			for (const auto& node : blocks)
			{
				std::string node_error;
				profile->either_layout_blocks.push_back(get_yaml_node_value<u32>(node, node_error));
				if (!node_error.empty())
					fail("either_layout_blocks: " + node_error);
			}
		}
		if (std::string rigid; read(root, "require_rigid_camera", rigid, false))
		{
			profile->require_rigid_camera = rigid == "true";
		}
		if (std::string aspect; read(root, "require_camera_aspect", aspect, false))
		{
			profile->require_camera_aspect = aspect == "true";
		}
		if (std::string direct; read(root, "camera_slots_read_directly", direct, false))
		{
			profile->camera_slots_read_directly = direct == "true";
		}
		if (std::string cache; read(root, "camera_block_cache", cache, false))
		{
			profile->camera_block_cache = cache == "true";
		}
		if (std::string clear; read(root, "clear_view_targets", clear, false))
		{
			profile->clear_view_targets = clear == "true";
		}
		if (std::string y_down; read(root, "view_y_down", y_down, false))
		{
			profile->view_y_down = y_down == "true";
		}
		if (const YAML::Node redirects = child(root, "texture_redirects"); redirects && redirects.IsSequence())
		{
			for (const YAML::Node& node : redirects)
			{
				std::string from, to;
				read(node, "from", from);
				read(node, "to", to);
				const u32 a = static_cast<u32>(std::strtoul(from.c_str(), nullptr, 16));
				// "camera": the render target this frame's camera draws went to (a game alternating two scene targets).
				const u32 b = to == "camera" ? 0u : static_cast<u32>(std::strtoul(to.c_str(), nullptr, 16));
				if (!a || (b < 0xc0000000u && to != "camera"))
				{
					fail("texture_redirects: expected {\"from\": \"<guest address>\", \"to\": \"<render target address in local memory, 0xc...>\" or \"camera\"}");
					continue;
				}
				profile->texture_redirects.emplace_back(a, b);
			}
		}
		if (const YAML::Node programs = child(root, "game_camera_programs"); programs && programs.IsSequence())
		{
			for (const auto& program : programs)
			{
				const std::string text = program.as<std::string>();
				char* end = nullptr;
				const u64 hash = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end)
					fail("game_camera_programs: '" + text + "' is not a hex program hash");
				profile->game_camera_programs.push_back(hash);
			}
		}
		for (const char* key : {"depth_remap_programs", "depth_remap_volume_programs"})
		{
			if (const YAML::Node programs = child(root, key); programs && programs.IsSequence())
			{
				for (const auto& program : programs)
				{
					const std::string text = program.as<std::string>();
					char* end = nullptr;
					const u64 hash = std::strtoull(text.c_str(), &end, 16);
					if (text.empty() || !end || *end)
						fail(std::string(key) + ": '" + text + "' is not a hex program hash");
					(std::string_view(key) == "depth_remap_programs" ? profile->depth_remap_programs : profile->depth_remap_volume_programs).push_back(hash);
				}
			}
		}
		if (std::string ray; read(root, "depth_remap_ray_texcoord", ray, false))
		{
			profile->depth_remap_ray_texcoord = std::clamp(std::atoi(ray.c_str()), -1, 9);
		}
		if (std::string uv; read(root, "depth_remap_uv", uv, false))
		{
			profile->depth_remap_uv = uv == "true";
		}
		if (std::string xyw; read(root, "depth_remap_xyw", xyw, false))
		{
			profile->depth_remap_xyw = xyw == "true";
		}
		read(root, "max_fps", profile->max_fps, false);
		read(root, "default_fps", profile->default_fps, false);
		read(root, "vblanks_per_frame", profile->vblanks_per_frame, false);
		read(root, "video_vblank_rate", profile->video_vblank_rate, false);
		if (std::string approximate; read(root, "zcull_approximate", approximate, false))
		{
			profile->zcull_approximate = approximate == "true";
		}
		if (std::string relaxed; read(root, "zcull_relaxed_sync", relaxed, false))
		{
			profile->zcull_relaxed_sync = relaxed == "true";
		}
		const YAML::Node orthographic_stereo = child(root, "orthographic_stereo");
		read(orthographic_stereo, "angle", profile->orthographic_stereo_angle, false);
		read(orthographic_stereo, "convergence", profile->orthographic_stereo_convergence, false);
		read(orthographic_stereo, "convergence_z", profile->orthographic_stereo_convergence_z, false);
		if (profile->orthographic_stereo_angle < 0.f || profile->orthographic_stereo_angle > 20.f)
		{
			fail("orthographic_stereo.angle: expected 0 to 20 degrees");
		}
		if (const YAML::Node rect = child(root, "display_rect"); rect && rect.IsSequence())
		{
			if (rect.size() != 4)
			{
				fail("display_rect: expected [x, y, width, height] in output pixels");
			}
			else
			{
				for (u32 i = 0; i < 4; ++i)
				{
					profile->display_rect[i] = rect[i].as<f32>();
				}
				if (profile->display_rect[2] <= 0.f || profile->display_rect[3] <= 0.f)
				{
					fail("display_rect: width and height must be positive");
				}
			}
		}
		if (read(root, "hud_depth", profile->hud_depth, false) && !(profile->hud_depth >= 1.f && profile->hud_depth <= 10.f))
		{
			fail("hud_depth must be between 1 and 10 (metres)");
		}
		if (std::string keep; read(root, "keep_rendered_display_buffers", keep, false))
		{
			profile->keep_rendered_display_buffers = keep == "true";
		}
		if (profile->default_fps == umax)
		{
			profile->default_fps = profile->max_fps ? profile->max_fps : 60;
		}
		if (profile->max_fps && (!profile->default_fps || profile->default_fps > profile->max_fps))
		{
			fail(fmt::format("default_fps %u is above max_fps %u", profile->default_fps, profile->max_fps));
		}
		if (!profile->vblanks_per_frame)
		{
			fail("vblanks_per_frame must be at least 1");
		}
		if (std::string scene; read(root, "clip_space_scene_draws", scene, false))
		{
			profile->clip_space_scene_draws = scene == "true";
		}
		if (std::string reproject; read(root, "reproject_older_frames", reproject, false))
		{
			profile->reproject_older_frames = reproject == "true";
		}
		// "0xADDR", "[0xPTR]" or "[0xPTR]+0xOFF"
		const auto parse_guest_address = [&](const char* key, const std::string& text, title_profile::guest_address& a) -> bool
		{
			std::string rest = text;
			usz depth = 0;
			while (rest.starts_with("["))
			{
				depth++;
				rest = rest.substr(1);
			}
			if (depth)
			{
				const usz close = rest.find(']');
				if (close == umax)
				{
					fail(std::string(key) + ": '" + text + "' has no closing ]");
					return false;
				}
				a.deref = true;
				a.address = static_cast<u32>(std::strtoul(rest.substr(0, close).c_str(), nullptr, 16));
				rest = rest.substr(close + 1);
				for (usz level = 1; level < depth; level++)
				{
					const usz inner_close = rest.find(']');
					if (!rest.starts_with("+") || inner_close == umax)
					{
						fail(std::string(key) + ": '" + text + "' expected +offset] inside the nested brackets");
						return false;
					}
					a.inner_offsets.push_back(static_cast<u32>(std::strtoul(rest.substr(1, inner_close - 1).c_str(), nullptr, 16)));
					rest = rest.substr(inner_close + 1);
				}
				if (rest.starts_with("+"))
				{
					a.offset = static_cast<u32>(std::strtoul(rest.c_str() + 1, nullptr, 16));
				}
				else if (!rest.empty())
				{
					fail(std::string(key) + ": '" + text + "' expected +offset after ]");
					return false;
				}
			}
			else
			{
				a.address = static_cast<u32>(std::strtoul(rest.c_str(), nullptr, 16));
			}
			if (!a.address)
			{
				fail(std::string(key) + ": '" + text + "' is not an address");
				return false;
			}
			return true;
		};
		const auto read_guest_addresses = [&](const char* key, std::vector<title_profile::guest_address>& list)
		{
			if (const YAML::Node targets = child(root, key); targets && targets.IsSequence())
			{
				for (const auto& target : targets)
				{
					title_profile::guest_address a;
					// "0x..." / "[0x...]+0x...", or { "address": "...", "scale": n }
					std::string text = target.IsMap() ? std::string() : target.as<std::string>();
					if (target.IsMap())
					{
						read(target, "address", text);
						read(target, "scale", a.scale, false);
						if (!(a.scale > 0.f))
						{
							fail(fmt::format("%s: scale must be positive", key));
							continue;
						}
					}
					if (parse_guest_address(key, text, a))
					{
						list.push_back(a);
					}
				}
			}
		};
		if (const YAML::Node rules = child(root, "native_rate_when"); rules && rules.IsSequence())
		{
			// [{ "address": "0xf17060", "values": [1], "rate": 60 }]
			for (const YAML::Node& node : rules)
			{
				std::string address;
				u32 rate = 60;
				read(node, "address", address);
				read(node, "rate", rate, false);
				title_profile::native_rate_rule rule;
				rule.rate = rate;
				if (!parse_guest_address("native_rate_when", address, rule.address))
				{
					continue;
				}
				if (const YAML::Node values = child(node, "values"); values && values.IsSequence())
				{
					for (const auto& v : values)
					{
						rule.values.push_back(static_cast<u32>(std::strtoul(v.as<std::string>().c_str(), nullptr, 0)));
					}
				}
				if (rule.values.empty() || rule.rate < 20)
				{
					fail("native_rate_when: expected {\"address\": \"0x...\", \"values\": [n, ...], \"rate\": 60}");
					continue;
				}
				profile->native_rate_when.push_back(std::move(rule));
			}
		}
		if (const YAML::Node rules = child(root, "culling_scale_f32"); rules && rules.IsSequence())
		{
			// [{ "address": "0x4d7180", "fov_deg": 60, "aspect": 1.7778, "margin_deg": 6, "min": 1.0, "max": 2.5 }]
			for (const YAML::Node& node : rules)
			{
				std::string address;
				read(node, "address", address);
				title_profile::culling_scale_rule rule;
				read(node, "fov_deg", rule.fov_deg, false);
				read(node, "aspect", rule.aspect, false);
				read(node, "margin_deg", rule.margin_deg, false);
				read(node, "min", rule.min, false);
				read(node, "max", rule.max, false);
				read(node, "step", rule.step, false);
				std::string mode = "angle";
				read(node, "mode", mode, false);
				rule.tangent = mode == "tangent" || mode == "tangent_x";
				rule.horizontal = mode == "tangent_x";
				if (!parse_guest_address("culling_scale_f32", address, rule.address))
				{
					continue;
				}
				if (rule.fov_deg <= 0.f || rule.fov_deg >= 180.f || rule.aspect <= 0.f || rule.min <= 0.f || rule.max < rule.min)
				{
					fail("culling_scale_f32: expected {\"address\": \"0x...\", \"fov_deg\": 60, \"aspect\": 1.7778, \"margin_deg\": 6, \"min\": 1, \"max\": 2.5}");
					continue;
				}
				profile->culling_scale_f32.push_back(std::move(rule));
			}
		}
		if (const YAML::Node rules = child(root, "option_f32"); rules && rules.IsSequence())
		{
			// [{ "address": "0x145fe04", "option": "highest_detail", "on": -8, "off": 100 }]
			for (const YAML::Node& node : rules)
			{
				std::string address;
				read(node, "address", address);
				title_profile::option_f32_rule rule;
				read(node, "option", rule.option);
				read(node, "on", rule.on);
				read(node, "off", rule.off);
				if (!parse_guest_address("option_f32", address, rule.address))
				{
					continue;
				}
				if (rule.option.empty())
				{
					fail("option_f32: expected {\"address\": \"0x...\", \"option\": \"highest_detail\", \"on\": -8, \"off\": 100}");
					continue;
				}
				profile->option_f32.push_back(std::move(rule));
			}
		}
		if (const YAML::Node rules = child(screen_space, "screen_frames_when"); rules && rules.IsSequence())
		{
			// [{ "address": "0x332b7ec0", "values": [8] }]
			for (const YAML::Node& node : rules)
			{
				std::string address;
				read(node, "address", address);
				title_profile::screen_frames_when_rule rule;
				if (!parse_guest_address("screen_space.screen_frames_when", address, rule.address))
				{
					continue;
				}
				if (const YAML::Node values = child(node, "values"); values && values.IsSequence())
				{
					for (const auto& v : values)
					{
						rule.values.push_back(static_cast<u32>(std::strtoul(v.as<std::string>().c_str(), nullptr, 0)));
					}
				}
				if (rule.values.empty())
				{
					fail("screen_space.screen_frames_when: expected {\"address\": \"0x...\", \"values\": [n, ...]}");
					continue;
				}
				profile->screen_space_screen_frames_when.push_back(std::move(rule));
			}
		}
		read_guest_addresses("game_refresh_rate_f32", profile->game_refresh_rate_f32);
		read_guest_addresses("game_frame_time_f32", profile->game_frame_time_f32);
		read_guest_addresses("game_frame_time_sq_f32", profile->game_frame_time_sq_f32);
		read_guest_addresses("game_frame_time_cube_f32", profile->game_frame_time_cube_f32);
		read_guest_addresses("game_frame_ms_u32", profile->game_frame_ms_u32);
		read_guest_addresses("game_frame_ms_f32", profile->game_frame_ms_f32);
		read_guest_addresses("game_vblank_frames_f32", profile->game_vblank_frames_f32);
		read_guest_addresses("game_fps_u32", profile->game_fps_u32);
		if (const YAML::Node rules = child(root, "resolution_scaled_constants"); rules && rules.IsSequence())
		{
			// [{ "program": "<vertex ucode hash>", "slots": [466, 467] }]
			for (const auto& node : rules)
			{
				title_profile::scaled_constants rule;
				const std::string text = node["program"] ? node["program"].as<std::string>() : std::string();
				char* end = nullptr;
				rule.program = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end)
					fail("resolution_scaled_constants: '" + text + "' is not a hex program hash");
				if (const YAML::Node slots = node["slots"]; slots && slots.IsSequence())
				{
					for (const auto& slot : slots)
					{
						const u32 value = slot.as<u32>();
						if (value >= 468)
							fail(fmt::format("resolution_scaled_constants: slot %u is out of range", value));
						rule.constant_slots.push_back(static_cast<u16>(value));
					}
				}
				profile->resolution_scaled_constants.push_back(std::move(rule));
			}
		}
		if (const YAML::Node rules = child(root, "fragment_constant_overrides"); rules && rules.IsSequence())
		{
			// [{ "program": "<vertex ucode hash>", "constant": 0, "value": [0, 0, 0, 0] }]
			for (const auto& node : rules)
			{
				check_keys(node, " in fragment_constant_overrides", {"program", "constant", "value"});
				fragment_constant_override rule;
				const std::string text = node["program"] ? node["program"].as<std::string>() : std::string();
				char* end = nullptr;
				rule.program = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end)
					fail("fragment_constant_overrides: '" + text + "' is not a hex program hash");
				rule.constant = node["constant"] ? node["constant"].as<u32>() : 0;
				const YAML::Node value = node["value"];
				if (!value || !value.IsSequence() || value.size() != 4)
				{
					fail("fragment_constant_overrides: value must be 4 numbers");
					continue;
				}
				for (u32 i = 0; i < 4; ++i)
				{
					rule.value[i] = value[i].as<f32>();
				}
				profile->fragment_constant_overrides.push_back(rule);
			}
		}
		if (std::string current; read(root, "current_frame_copies", current, false))
		{
			profile->current_frame_copies = current == "true";
		}
		if (std::string offaspect; read(root, "offaspect_player_views", offaspect, false))
		{
			profile->offaspect_player_views = offaspect == "true";
		}
		if (const YAML::Node ranges = child(root, "occlusion_depth_readback"); ranges && ranges.IsSequence())
		{
			for (const auto& node : ranges)
			{
				// "0xADDR:0xSIZE"
				const std::string text = node.as<std::string>();
				const usz colon = text.find(':');
				const u32 start = static_cast<u32>(std::strtoul(text.substr(0, colon).c_str(), nullptr, 16));
				const u32 size = colon == umax ? 0 : static_cast<u32>(std::strtoul(text.substr(colon + 1).c_str(), nullptr, 16));
				if (!size)
				{
					error += fmt::format(" occlusion_depth_readback '%s' needs a size (0xADDR:0xSIZE).", text);
					continue;
				}
				profile->occlusion_depth_readback.emplace_back(start, size);
			}
		}
		if (const YAML::Node sections = child(root, "skip_readback_sections"); sections && sections.IsSequence())
		{
			for (const auto& node : sections)
			{
				profile->skip_readback_sections.push_back(static_cast<u32>(std::strtoul(node.as<std::string>().c_str(), nullptr, 16)));
			}
		}
		if (const YAML::Node sections = child(root, "late_readback_sections"); sections && sections.IsSequence())
		{
			for (const auto& node : sections)
			{
				profile->late_readback_sections.push_back(static_cast<u32>(std::strtoul(node.as<std::string>().c_str(), nullptr, 16)));
			}
		}
		if (const YAML::Node limit = child(root, "car_draw_limit"); limit && limit.IsMap())
		{
			if (const YAML::Node tiers = limit["tiers"]; tiers && tiers.IsSequence())
			{
				for (const auto& tier : tiers)
				{
					const YAML::Node cars = tier["cars"], draws = tier["draws"], min_vertices = tier["min_vertices"], keep_percent = tier["keep_percent"], min_distance = tier["min_distance"];
					if (!cars || !cars.IsScalar() || (draws && !draws.IsScalar()) || (min_vertices && !min_vertices.IsScalar()) ||
						(keep_percent && !keep_percent.IsScalar()) || (min_distance && !min_distance.IsScalar()))
					{
						fail("car_draw_limit.tiers: expected {\"cars\": <n, 0 = the rest>, \"draws\": <-1 all, 0 none, n first n>, \"min_vertices\": <v>, \"keep_percent\": <1-99>, \"min_distance\": <m>}");
						continue;
					}
					title_profile::car_draw_tier t{};
					t.min_distance = min_distance ? static_cast<f32>(std::clamp(std::atof(min_distance.as<std::string>().c_str()), 0.0, 10000.0)) : 0.f;
					t.cars = static_cast<u32>(std::clamp(std::atoi(cars.as<std::string>().c_str()), 0, 64));
					t.draws = draws ? std::clamp(std::atoi(draws.as<std::string>().c_str()), -1, 4096) : -1;
					t.min_vertices = min_vertices ? static_cast<u32>(std::clamp(std::atoi(min_vertices.as<std::string>().c_str()), 0, 1 << 20)) : 0u;
					t.keep_percent = keep_percent ? static_cast<u32>(std::clamp(std::atoi(keep_percent.as<std::string>().c_str()), 0, 100)) : 0u;
					profile->car_draw_tiers.push_back(t);
				}
			}
			if (const YAML::Node programs = limit["body_programs"]; programs && programs.IsSequence())
			{
				for (const auto& program : programs)
				{
					const std::string text = program.as<std::string>();
					char* end = nullptr;
					const u64 hash = std::strtoull(text.c_str(), &end, 16);
					if (text.empty() || !end || *end)
						fail("car_draw_limit.body_programs: '" + text + "' is not a hex program hash");
					profile->car_body_programs.push_back(hash);
				}
			}
			if (const YAML::Node option = limit["option"]; option && option.IsScalar())
			{
				profile->car_draw_option = option.as<std::string>();
			}
			if (const YAML::Node nearest = limit["full_nearest"]; nearest && nearest.IsScalar())
			{
				profile->car_full_nearest = static_cast<u32>(std::clamp(std::atoi(nearest.as<std::string>().c_str()), 0, 64));
			}
			if (profile->car_draw_tiers.empty() || profile->car_body_programs.empty())
				fail("car_draw_limit: expected {\"tiers\": [...], \"body_programs\": [\"<vertex ucode hash>\", ...], \"option\": \"<setting>\"}");
		}
		if (const YAML::Node shared = child(root, "shared_frame_targets"); shared && shared.IsSequence())
		{
			for (const auto& node : shared)
			{
				const YAML::Node width = node["width"], height = node["height"], frames = node["frames"], option = node["option"];
				if (!width || !width.IsScalar() || !height || !height.IsScalar() || (frames && !frames.IsScalar()) || (option && !option.IsScalar()))
				{
					fail("shared_frame_targets: expected [{\"width\": <w>, \"height\": <h>, \"frames\": <n>, \"option\": \"<setting>\"}]");
					continue;
				}
				title_profile::shared_frame_target t{};
				t.option = option ? option.as<std::string>() : std::string();
				t.width = static_cast<u16>(std::clamp(std::atoi(width.as<std::string>().c_str()), 1, 8192));
				t.height = static_cast<u16>(std::clamp(std::atoi(height.as<std::string>().c_str()), 1, 8192));
				t.frames = frames ? static_cast<u32>(std::clamp(std::atoi(frames.as<std::string>().c_str()), 1, 16)) : 2u;
				profile->shared_frame_targets.push_back(t);
			}
		}
		if (const YAML::Node min_scalable = child(root, "min_scalable_dimension"); min_scalable && min_scalable.IsScalar())
		{
			profile->min_scalable_dimension = static_cast<u16>(std::clamp(std::atoi(min_scalable.as<std::string>().c_str()), 0, 4096));
		}
		if (const YAML::Node lengths = child(root, "late_readback_lengths"); lengths && lengths.IsSequence())
		{
			for (const auto& node : lengths)
			{
				profile->late_readback_lengths.push_back(static_cast<u32>(std::strtoul(node.as<std::string>().c_str(), nullptr, 0)));
			}
		}

		check_keys(root, "", {"schema", "title_id", "app_version", "name", "matrix_layout", "camera_blocks", "output_aspect_tolerance", "camera_target_aspect", "camera_position", "stereo", "screen_space", "reference_screen_width", "game_refresh_rate_f32", "game_frame_time_f32", "game_frame_time_sq_f32", "game_frame_time_cube_f32", "game_frame_ms_u32", "game_frame_ms_f32", "game_fps_u32", "game_vblank_frames_f32", "max_fps", "default_fps", "vblanks_per_frame", "video_vblank_rate", "zcull_approximate", "zcull_relaxed_sync", "display_rect", "hidden_draws", "keep_rendered_display_buffers", "hud_depth", "reproject_older_frames", "clip_space_scene_draws", "require_rigid_camera", "nonrigid_camera_blocks", "row_vector_blocks", "column_vector_blocks", "either_layout_blocks", "linked_camera_blocks", "camera_palette", "require_camera_aspect", "camera_slots_read_directly", "camera_block_cache", "clear_view_targets", "view_y_down", "texture_redirects", "game_camera_programs", "depth_remap_programs", "depth_remap_volume_programs", "depth_remap_ray_texcoord", "depth_remap_xyw", "depth_remap_uv", "reduced_scale_frames", "game_camera_target_widths", "game_camera_aspects", "current_frame_copies", "occlusion_depth_readback", "skip_readback_sections", "late_readback_sections", "late_readback_lengths", "min_scalable_dimension", "car_draw_limit", "shared_frame_targets", "offaspect_player_views", "resolution_scaled_constants", "fragment_constant_overrides", "orthographic_stereo", "frame_rate_draws", "native_rate_when", "culling_scale_f32", "option_f32"});
		check_keys(orthographic_stereo, " in orthographic_stereo", {"angle", "convergence", "convergence_z"});
		check_keys(camera_position, " in camera_position", {"slot", "eye_baseline"});
		check_keys(stereo, " in stereo", {"formula", "per_eye_separation", "convergence", "by_target_width", "eye_offset"});
		check_keys(screen_space, " in screen_space", {"orthographic_block", "orthographic_block_layout", "hud_block_programs", "bare_projection", "depth_offset_projection", "offaspect_projection", "rotation_only_passthrough", "passthrough_hud", "preprojected_programs", "hud_programs", "output_pixel_draws_not_hud", "subviewport_cameras_in_box", "boxed_cameras", "hud_keep_depth", "hud_exact_depth_programs", "hud_skips_passes", "hud_display_buffers_only", "hud_box_after_shader", "frames_without_3d_as_screen", "clear_outside_box", "unboxed_draws", "screen_frame_draws", "screen_frames_when", "scaled_draws"});
		if (const YAML::Node rules = child(stereo, "by_target_width"); rules && rules.IsSequence())
		{
			for (const auto& node : rules)
			{
				check_keys(node, " in stereo.by_target_width", {"output_width_divisor", "per_eye_separation", "convergence"});
			}
		}

		if (!error.empty())
		{
			vr_probe_log.error("VR profile '%s' is invalid: %s", path, error);
			return nullptr;
		}

		return profile;
	}

	bool title_has_profile(std::string_view title_id)
	{
		return load_title_profile(title_id, {}) != nullptr;
	}

	namespace
	{
		atomic_t<u32> g_headset_refresh_hz{0};
		atomic_t<bool> g_headset_active{false};
	} // namespace

	void set_headset_refresh_rate(u32 hz)
	{
		if (g_headset_refresh_hz.exchange(hz) != hz && hz)
		{
			vr_probe_log.notice("Headset refresh rate: %u Hz", hz);
		}
	}

	static atomic_t<bool> s_multiview_active{false};

	void set_multiview_active(bool active)
	{
		s_multiview_active = active;
	}

	bool multiview_active()
	{
		return s_multiview_active.load();
	}

	bool exact_depth_programs_listed()
	{
		const auto* profile = camera_probe::get().profile();
		return profile && !profile->screen_space_hud_exact_depth_programs.empty();
	}

	bool exact_depth_program(u64 vertex_ucode_hash)
	{
		const auto* profile = camera_probe::get().profile();
		if (!profile)
		{
			return false;
		}
		const auto& list = profile->screen_space_hud_exact_depth_programs;
		return std::find(list.begin(), list.end(), vertex_ucode_hash) != list.end();
	}

	bool depth_remap_programs_listed()
	{
		const auto* profile = camera_probe::get().profile();
		return profile && !profile->depth_remap_programs.empty();
	}

	bool depth_remap_program(u64 vertex_ucode_hash)
	{
		const auto* profile = camera_probe::get().profile();
		if (!profile)
		{
			return false;
		}
		const auto& list = profile->depth_remap_programs;
		return std::find(list.begin(), list.end(), vertex_ucode_hash) != list.end();
	}

	void set_headset_active(bool active)
	{
		g_headset_active = active;
	}

	u32 frame_rate_option_fps(u32 option)
	{
		switch (static_cast<vr_frame_rate>(option))
		{
		case vr_frame_rate::profile_default: return umax;
		case vr_frame_rate::fps_30: return 30;
		case vr_frame_rate::fps_45: return 45;
		case vr_frame_rate::fps_60: return 60;
		case vr_frame_rate::fps_72: return 72;
		case vr_frame_rate::fps_75: return 75;
		case vr_frame_rate::fps_80: return 80;
		case vr_frame_rate::fps_90: return 90;
		case vr_frame_rate::fps_120: return 120;
		case vr_frame_rate::fps_144: return 144;
		case vr_frame_rate::unlimited: return 0;
		}
		return umax;
	}

	bool profile_option_enabled(const std::string& option)
	{
		if (option.empty())
		{
			return true;
		}
		if (option == "reflections")
		{
			return g_cfg.video.vr.reduced_rate_reflections.get();
		}
		if (option == "mirror")
		{
			return g_cfg.video.vr.reduced_rate_mirror.get();
		}
		if (option == "distant_cars")
		{
			return g_cfg.video.vr.simpler_distant_cars.get();
		}
		if (option == "highest_detail")
		{
			return g_cfg.video.vr.highest_detail_models.get();
		}
		if (option == "simpler_vehicles")
		{
			return g_cfg.video.vr.simpler_vehicles.get();
		}
		if (option == "fewer_shadows")
		{
			return g_cfg.video.vr.fewer_shadows.get();
		}
		return true;
	}

	bool profile_has_option(const title_profile& profile, std::string_view option)
	{
		if (profile.car_draw_option == option && !profile.car_draw_tiers.empty())
		{
			return true;
		}
		if (std::any_of(profile.option_f32.begin(), profile.option_f32.end(), [&](const auto& r) { return r.option == option; }))
		{
			return true;
		}
		return std::any_of(profile.shared_frame_targets.begin(), profile.shared_frame_targets.end(), [&](const auto& t)
			{
				return t.option == option;
			});
	}

	bool frame_rate_option_allowed(u32 option, u32 max_fps)
	{
		const u32 fps = frame_rate_option_fps(option);
		return fps == umax || !max_fps || (fps && fps <= max_fps);
	}

	namespace
	{
		// <TITLE_ID>.json and every <TITLE_ID>.<executable>.json, with the executable ("" for the title's own).
		std::vector<std::pair<std::string, std::shared_ptr<const title_profile>>> title_profiles(std::string_view title_id)
		{
			std::vector<std::pair<std::string, std::shared_ptr<const title_profile>>> result;
			const std::string dir = fs::get_executable_dir() + "vr_profiles/";
			for (const auto& entry : fs::dir(dir))
			{
				if (entry.is_directory || !entry.name.starts_with(title_id) || !entry.name.ends_with(".json"))
				{
					continue;
				}
				const std::string_view rest = std::string_view(entry.name).substr(title_id.size());
				std::string executable;
				if (rest != ".json")
				{
					if (!rest.starts_with(".") || rest.size() <= 6)
					{
						continue;
					}
					executable = std::string(rest.substr(1, rest.size() - 6));
				}
				if (auto profile = load_title_profile(title_id, executable))
				{
					result.emplace_back(std::move(executable), std::move(profile));
				}
			}
			return result;
		}
	} // namespace

	u32 title_max_fps(std::string_view title_id)
	{
		u32 result = 0;
		bool unlimited = false;
		for (const auto& [executable, profile] : title_profiles(title_id))
		{
			unlimited |= !profile->max_fps;
			result = std::max(result, profile->max_fps);
		}
		return unlimited ? 0 : result;
	}

	std::vector<u32> title_default_fps(std::string_view title_id)
	{
		std::vector<u32> result;
		for (const auto& [executable, profile] : title_profiles(title_id))
		{
			if (std::find(result.begin(), result.end(), profile->default_fps) == result.end())
			{
				result.push_back(profile->default_fps);
			}
		}
		std::sort(result.begin(), result.end());
		return result;
	}

	std::vector<title_game_frame_rate> title_frame_rates(std::string_view title_id)
	{
		std::vector<title_game_frame_rate> result;
		for (const auto& [executable, profile] : title_profiles(title_id))
		{
			std::string name = !profile->name.empty() ? profile->name : !executable.empty() ? executable :
			                                                                                  std::string(title_id);
			result.push_back({std::move(name), profile->default_fps, profile->max_fps});
		}
		return result;
	}

	u32 frame_rate_option_for_fps(u32 fps)
	{
		for (u32 option = 0; option <= static_cast<u32>(vr_frame_rate::unlimited); ++option)
		{
			if (frame_rate_option_fps(option) == fps)
			{
				return option;
			}
		}
		return umax;
	}

	u32 effective_frame_rate()
	{
		const title_profile* profile = camera_probe::get().profile();
		if (!profile)
		{
			return 0;
		}
		u32 fps = frame_rate_option_fps(static_cast<u32>(g_cfg.video.vr.frame_rate.get()));
		if (fps == umax)
		{
			fps = profile->default_fps;
		}
		if (profile->max_fps && (!fps || fps > profile->max_fps))
		{
			fps = profile->max_fps;
		}
		return fps;
	}

	static u64 vr_vblank_rate();

	u64 effective_vblank_rate()
	{
		const u64 rate = vr_vblank_rate();
		// Profile video_vblank_rate: movies at their own pace, whatever the game runs at (applies with VR off too).
		if (const title_profile* profile = camera_probe::get().profile(); profile && profile->video_vblank_rate && rate > profile->video_vblank_rate)
		{
			static atomic_t<bool> s_capped = false;
			const bool video = vdec_open_count() != 0;
			if (s_capped.exchange(video) != video)
			{
				vr_probe_log.notice("VR: video decoder %s: vblank %u Hz", video ? "open" : "closed", video ? profile->video_vblank_rate : static_cast<u32>(rate));
			}
			if (video)
			{
				return profile->video_vblank_rate;
			}
		}
		// Profile native_rate_when: scenes the game steps per frame run at their own rate.
		if (const title_profile* profile = camera_probe::get().profile(); profile && !profile->native_rate_when.empty())
		{
			u32 cap = 0;
			for (const auto& rule : profile->native_rate_when)
			{
				const u32 address = resolve_guest_address(rule.address);
				if (address && vm::check_addr(address, vm::page_readable, 4) &&
					std::find(rule.values.begin(), rule.values.end(), static_cast<u32>(vm::_ref<be_t<u32>>(address))) != rule.values.end())
				{
					cap = cap ? std::min(cap, rule.rate) : rule.rate;
				}
			}
			static atomic_t<u32> s_native_cap = 0;
			if (s_native_cap.exchange(cap) != cap)
			{
				vr_probe_log.notice("VR: native_rate_when %s: vblank %u Hz", cap ? "on" : "off", cap && rate > cap ? cap : static_cast<u32>(rate));
			}
			if (cap && rate > cap)
			{
				return cap;
			}
		}
		return rate;
	}

	f64 vr_frame_limit(f64 limit)
	{
		if (limit <= 0. || !g_cfg.video.vr.enabled || !g_headset_active.load() || !camera_probe::get().profile())
		{
			return limit;
		}
		const f64 vr = static_cast<f64>(effective_vblank_rate());
		static atomic_t<u32> s_logged{0};
		if (const u32 key = static_cast<u32>(limit * 100.) ^ (static_cast<u32>(vr) << 16); s_logged.exchange(key) != key && vr != limit)
		{
			vr_probe_log.notice("VR: Frame limit %.2f replaced by the VR rate %.0f (VR Frame Rate setting).", limit, vr);
		}
		return vr;
	}

	static u64 vr_vblank_rate()
	{
		const u64 configured = g_cfg.video.vblank_rate;
		if (!g_cfg.video.vr.enabled || !g_headset_active.load())
		{
			return configured;
		}
		const title_profile* profile = camera_probe::get().profile();
		if (!profile)
		{
			return configured;
		}
		if (const u32 fps = effective_frame_rate())
		{
			return u64{fps} * profile->vblanks_per_frame;
		}
		// Unlimited: a game frame per headset refresh, if the runtime reports it (Killzone 2 flips
		// every second vblank: 180 Hz for a 90 Hz headset).
		const u32 headset = g_headset_refresh_hz.load();
		return headset ? u64{headset} * profile->vblanks_per_frame : configured;
	}

	f32 effective_hud_depth()
	{
		if (const u64 configured = g_cfg.video.vr.hud_depth.get())
		{
			return std::max<u64>(configured, 100) / 100.f;
		}
		const title_profile* profile = camera_probe::get().profile();
		return profile && profile->hud_depth > 0.f ? profile->hud_depth : 2.f;
	}

	u32 effective_reprojection_margin()
	{
		const s64 configured = g_cfg.video.vr.reprojection_margin.get();
		if (configured >= 0)
		{
			return static_cast<u32>(configured);
		}
		// Below the headset's refresh rate the headset turns older frames to the current pose.
		const u32 fps = effective_frame_rate();
		const u32 headset = g_headset_refresh_hz.load();
		return fps && (!headset || fps < headset) ? 10 : 0;
	}

	bool occlusion_depth_readback(u32 start, u32 end, void* memory)
	{
		profile_generator::get().note_readback(start, end - start + 1);
		// Dev: RPCS3_VR_FLUSH_LOG=1 counts the texture-cache flushes (GPU readbacks, each a wait for the GPU) per
		// range and logs the busiest every 2 s.
		if (static const bool s_flush_log = std::getenv("RPCS3_VR_FLUSH_LOG") != nullptr; s_flush_log)
		{
			static std::mutex s_mutex;
			static std::map<std::pair<u32, u32>, u32> s_counts;
			static u64 s_last = get_system_time();
			std::lock_guard lock(s_mutex);
			s_counts[{start, end - start + 1}]++;
			if (const u64 now = get_system_time(); now - s_last >= 2'000'000)
			{
				std::vector<std::pair<u32, std::pair<u32, u32>>> top;
				u32 total = 0;
				for (const auto& [k, n] : s_counts)
				{
					top.emplace_back(n, k);
					total += n;
				}
				std::sort(top.rbegin(), top.rend());
				std::string text;
				for (usz i = 0; i < std::min<usz>(top.size(), 8); ++i)
				{
					fmt::append(text, " 0x%x+0x%x x%u;", top[i].second.first, top[i].second.second, top[i].first);
				}
				vr_probe_log.notice("VR flushes over %.1f s: %u (%u ranges):%s", (now - s_last) / 1e6, total, ::size32(top), text);
				s_counts.clear();
				s_last = now;
			}
		}
		// Dev: RPCS3_VR_SKIP_FLUSH=<hex start>[,<hex start>...] skips the readback of sections starting there (memory left as it is).
		static const std::vector<u32> s_skip = []()
		{
			std::vector<u32> r;
			for (const char* v = std::getenv("RPCS3_VR_SKIP_FLUSH"); v && *v;)
			{
				char* e = nullptr;
				r.push_back(static_cast<u32>(std::strtoul(v, &e, 16)));
				v = (e && *e == ',') ? e + 1 : nullptr;
			}
			return r;
		}();
		if (!s_skip.empty() && std::find(s_skip.begin(), s_skip.end(), start) != s_skip.end())
		{
			return true;
		}
		camera_probe& probe = camera_probe::get();
		const title_profile* profile = probe.profile();
		if (profile && !profile->skip_readback_sections.empty() &&
			std::find(profile->skip_readback_sections.begin(), profile->skip_readback_sections.end(), start) != profile->skip_readback_sections.end())
		{
			static atomic_t<bool> s_logged{false};
			if (!s_logged.exchange(true))
			{
				vr_probe_log.notice("VR: readback of the section at 0x%x skipped (skip_readback_sections).", start);
			}
			return true;
		}
		if (!profile || profile->occlusion_depth_readback.empty() || !probe.render_enabled())
		{
			return false;
		}
		for (const auto& [base, size] : profile->occlusion_depth_readback)
		{
			if (start < base + size && end >= base)
			{
				static atomic_t<bool> s_logged{false};
				if (!s_logged.exchange(true))
				{
					vr_probe_log.notice("VR: occlusion depth readback at 0x%x..0x%x answered with far depth (occlusion_depth_readback).", base, base + size - 1);
				}
				std::memset(memory, 0xff, end - start + 1);
				return true;
			}
		}
		return false;
	}

	bool screen_frame_by_game_state()
	{
		const title_profile* profile = camera_probe::get().profile();
		if (!profile || profile->screen_space_screen_frames_when.empty())
		{
			return false;
		}

		for (const auto& rule : profile->screen_space_screen_frames_when)
		{
			const u32 address = resolve_guest_address(rule.address);
			if (!address || !vm::check_addr(address, vm::page_readable, 4))
			{
				continue;
			}
			const u32 value = vm::_ref<be_t<u32>>(address);
			if (std::find(rule.values.begin(), rule.values.end(), value) != rule.values.end())
			{
				return true;
			}
		}
		return false;
	}

	// Smoothed host time between the game's flips (seconds; 0 = not measured yet), for update_game_refresh_rate.
	static atomic_t<f64> g_game_flip_interval = 0.0;

	void note_game_flip()
	{
		static u64 s_last_us = 0;
		const u64 now_us = get_system_time();
		if (const f64 dt = s_last_us ? (now_us - s_last_us) / 1e6 : 0.0; dt > 0.001 && dt < 0.2)
		{
			const f64 old = g_game_flip_interval.load();
			g_game_flip_interval = old > 0.0 ? old * 0.9 + dt * 0.1 : dt;
		}
		s_last_us = now_us;
	}

	// RPCS3_VR_FRAMESTATS: the frame's first game camera (apply_render_eye), | 1 so a camera draw never reads as none.
	static atomic_t<u64> g_frame_camera_hash = 0;

	u64 take_frame_camera_hash()
	{
		return g_frame_camera_hash.exchange(0);
	}

	static atomic_t<u64> g_frame_rate_draw_time = 0; // get_system_time() of the last frame with enough frame_rate_draws matches
	static std::array<atomic_t<u32>, 8> g_frame_rate_draw_counts{}; // this frame's matches per frame_rate_draws entry

	void note_frame_rate_draw(u32 index)
	{
		if (index < g_frame_rate_draw_counts.size())
		{
			g_frame_rate_draw_counts[index]++;
		}
	}

	bool camera_probe::vr_view_rotation(std::array<f32, 9>& rotation, f32& tan_x, f32& tan_y) const
	{
		if (!m_vr_view || !m_vr_hmd_fov)
		{
			return false;
		}
		rotation = m_vr_rot;
		tan_x = tan_y = 0.f;
		for (const auto& t : m_vr_eye_fov)
		{
			tan_x = std::max({tan_x, std::fabs(t[0]), std::fabs(t[1])});
			tan_y = std::max({tan_y, std::fabs(t[2]), std::fabs(t[3])});
		}
		return tan_x > 0.f && tan_y > 0.f;
	}

	void update_culling_scale()
	{
		const title_profile* profile = camera_probe::get().profile();
		if (!profile || profile->culling_scale_f32.empty())
		{
			return;
		}
		std::array<f32, 9> R{};
		f32 tx = 0.f, ty = 0.f;
		// No headset view this frame (fixed screen: menus, Gran Turismo 5's pre-race views and replays; no headset):
		// the game's own culling, so a patch that widens the game's projection with the scale shows the game's own
		// framing on the screen instead of a zoomed-out one.
		const bool head_view = camera_probe::get().vr_view_rotation(R, tx, ty);
		for (const auto& rule : profile->culling_scale_f32)
		{
			const u32 address = resolve_guest_address(rule.address);
			if (!address || !vm::check_addr(address, vm::page_readable, 4))
			{
				continue;
			}
			if (!head_view)
			{
				be_t<f32>& value = *vm::get_super_ptr<f32>(address);
				if (value != rule.min)
				{
					value = rule.min;
				}
				continue;
			}
			// The headset frustum's corners turned by the head: the game's frustum (half-height h, half-width
			// h x aspect, in tangents) must hold each, so h >= max(|y / z|, |x / z| / aspect). A corner at or behind the
			// camera's plane takes the maximum. Both rotation directions are tested (the basis holds the transpose).
			f32 h = 0.f, hx = 0.f; // hx: the horizontal tangent alone
			bool behind = false;
			for (u32 transpose = 0; transpose < 2; ++transpose)
			{
				const auto m = [&](u32 r, u32 c)
				{
					return transpose ? R[c * 3 + r] : R[r * 3 + c];
				};
				for (const f32 sx : {-tx, tx})
				{
					for (const f32 sy : {-ty, ty})
					{
						const f32 x = m(0, 0) * sx + m(0, 1) * sy + m(0, 2);
						const f32 y = m(1, 0) * sx + m(1, 1) * sy + m(1, 2);
						const f32 z = m(2, 0) * sx + m(2, 1) * sy + m(2, 2);
						if (z <= 0.05f)
						{
							behind = true;
							continue;
						}
						h = std::max({h, std::fabs(y / z), std::fabs(x / z) / rule.aspect});
						hx = std::max(hx, std::fabs(x / z));
					}
				}
			}
			const f32 half_deg = behind ? 90.f : std::atan(rule.horizontal ? hx : h) * 57.29578f + rule.margin_deg;
			// Tangent mode: the projection's scales are divided by the word, so the game's half-height tangent becomes
			// tan(fov / 2) x scale (89 degrees at most: a corner behind the camera takes the maximum). tangent_x: the
			// half-width tangent, tan(fov / 2) x aspect x scale.
			const f32 game_tan = std::tan(rule.fov_deg / 2.f / 57.29578f) * (rule.horizontal ? rule.aspect : 1.f);
			const f32 wanted = std::clamp(rule.tangent ? std::tan(std::min(half_deg, 89.f) / 57.29578f) / game_tan :
				2.f * half_deg / rule.fov_deg, rule.min, rule.max);
			// Up at once (newly visible scenery must not be missing); down slowly, so a glance back and forth does not
			// make the scenery at the edges pop in and out. Written past the page protection, as a game patch is: the
			// word can sit in a patch's code cave (SEGA Rally: the code segment's tail, read-only on a fresh boot; a plain
			// write there froze the emulator on the title screen).
			be_t<f32>& value = *vm::get_super_ptr<f32>(address);
			const f32 current = value;
			f32 next = !(current >= rule.min && current <= rule.max) || wanted >= current ? wanted : std::max(wanted, current - 0.01f);
			if (rule.step > 0.f)
			{
				// Stepped: up to the next step at once; down only when the head no longer needs the step below this one.
				const f32 stepped = std::min(rule.max, rule.min + std::ceil((wanted - rule.min) / rule.step - 1e-4f) * rule.step);
				const bool valid = current >= rule.min && current <= rule.max;
				next = !valid || stepped > current || stepped < current - rule.step - 1e-4f ? stepped : current;
			}
			if (next != current)
			{
				value = next;
			}
			static u32 s_logged = 0;
			if (s_logged < 3 && std::fabs(next - current) > 0.2f)
			{
				s_logged++;
				vr_probe_log.notice("Culling scale at 0x%x: %.2f -> %.2f (headset frustum %.1f degrees off the game's axis)", address, current, next, half_deg);
			}
		}
	}

	void update_option_words()
	{
		const title_profile* profile = camera_probe::get().profile();
		if (!profile)
		{
			return;
		}
		for (const auto& rule : profile->option_f32)
		{
			const u32 address = resolve_guest_address(rule.address);
			if (!address || !vm::check_addr(address, vm::page_readable, 4))
			{
				continue;
			}
			// Past the page protection, as culling_scale_f32 (the word can sit in a patch's code cave).
			be_t<f32>& value = *vm::get_super_ptr<f32>(address);
			const f32 wanted = profile_option_enabled(rule.option) ? rule.on : rule.off;
			if (value != wanted)
			{
				value = wanted;
			}
		}
	}

	void update_game_refresh_rate()
	{
		const title_profile* profile = camera_probe::get().profile();
		if (!profile || (profile->game_refresh_rate_f32.empty() && profile->game_frame_time_f32.empty() && profile->game_frame_ms_u32.empty() &&
							profile->game_frame_time_sq_f32.empty() && profile->game_frame_time_cube_f32.empty() && profile->game_frame_ms_f32.empty() && profile->game_vblank_frames_f32.empty() &&
							profile->game_fps_u32.empty()))
		{
			return;
		}

		// The address a target names now, or 0 (pointer not set up yet, or not writable).
		const auto resolve = [](const title_profile::guest_address& target) -> u32
		{
			const u32 address = resolve_guest_address(target);
			return address && vm::check_addr(address, vm::page_writable, 4) ? address : 0;
		};

		// The rate the game really runs at. A PC that cannot keep the VR rate shows each frame twice (the runtime's
		// ASW / motion smoothing halves it), and a frame-locked game told the nominal rate then runs in slow motion
		// (Tales of Xillia at half speed on slower PCs). The time between game flips, smoothed, replaces the nominal
		// rate when it is more than 3% slower. Long gaps (loading) are ignored, and so is a rate under 40% of the VR rate
		// (a stall, not the runtime's halving): Need for Speed: Hot Pursuit's loading measured 7 FPS, and the 0.14 s step
		// written then lay outside the frame-time window, so it was never replaced and the race ran 12x fast.
		// Dev: RPCS3_VR_NOMINAL_RATE=1 keeps the nominal rate.
		const f32 frames_per_vblank = 1.f / static_cast<f32>(std::max<u32>(profile->vblanks_per_frame, 1));
		// Profile frame_rate_draws: the game's own 60 Hz values unless a frame had one of those draws (min_count times) in
		// the last 2 s.
		for (usz i = 0; i < g_frame_rate_draw_counts.size(); ++i)
		{
			const u32 count = g_frame_rate_draw_counts[i].exchange(0);
			if (i < profile->frame_rate_draw_min_counts.size() && count >= profile->frame_rate_draw_min_counts[i])
			{
				g_frame_rate_draw_time = get_system_time();
			}
		}
		const bool native_only = !profile->frame_rate_draws.empty() && get_system_time() - g_frame_rate_draw_time.load() > 2'000'000;
		if (static bool s_native = false; native_only != s_native && !profile->frame_rate_draws.empty())
		{
			s_native = native_only;
			static u32 s_switches = 0;
			if (s_switches++ < 20)
			{
				vr_probe_log.notice("Game frame-rate words: %s (profile frame_rate_draws)", native_only ? "the game's own 60 Hz" : "the VR rate");
			}
		}
		const f32 nominal_rate = native_only ? 60.f / frames_per_vblank : static_cast<f32>(effective_vblank_rate());
		static const bool s_nominal_only = std::getenv("RPCS3_VR_NOMINAL_RATE") != nullptr;
		const f64 frame_interval = g_game_flip_interval.load();
		f32 rate = nominal_rate;
		if (!s_nominal_only && !native_only && frame_interval > 0.0 && nominal_rate > 0.f)
		{
			const f32 measured_fps = static_cast<f32>(1.0 / frame_interval);
			const f32 full_fps = nominal_rate * frames_per_vblank;
			const bool slow = measured_fps < full_fps * 0.97f && measured_fps >= full_fps * 0.4f;
			if (slow)
			{
				rate = measured_fps / frames_per_vblank;
			}
			static bool s_slow = false;
			static u32 s_switches = 0;
			if (slow != s_slow)
			{
				s_slow = slow;
				if (s_switches++ < 20)
				{
					vr_probe_log.notice("Game frame rate: %s (%.1f FPS measured, VR rate %.1f FPS)", slow ? "below the VR rate, game timing follows the measured rate" : "back at the VR rate",
						measured_fps, nominal_rate * frames_per_vblank);
				}
			}
		}
		for (const auto& target : profile->game_refresh_rate_f32)
		{
			const u32 address = resolve(target);
			if (!address)
			{
				continue;
			}

			// Only over a value that looks like a refresh rate: the game has set it up.
			be_t<f32>& value = *vm::_ptr<be_t<f32>>(address);
			const f32 current = value;
			if (current >= 20.f && current <= 1000.f && current != rate)
			{
				value = rate;
				static u32 s_logged = 0;
				if (s_logged++ < 4)
				{
					vr_probe_log.notice("Game refresh rate at 0x%x: %.2f -> %.2f Hz (effective vblank rate)", address, current, rate);
				}
			}
		}

		// One vblank in 60 Hz frames, over a value that looks like it (2.0 at 30 Hz down to 0.1 at 600 Hz).
		const f32 vblank_frames = 60.f / std::max(rate, 1.f);
		for (const auto& target : profile->game_vblank_frames_f32)
		{
			const u32 address = resolve(target);
			if (!address)
			{
				continue;
			}
			be_t<f32>& value = *vm::_ptr<be_t<f32>>(address);
			const f32 current = value;
			const f32 scaled = vblank_frames * target.scale;
			if (current >= 0.1f * target.scale && current <= 2.f * target.scale && current != scaled)
			{
				value = scaled;
				static u32 s_logged = 0;
				if (s_logged++ < 4)
				{
					vr_probe_log.notice("Game vblank length at 0x%x: %.4f -> %.4f (%g x frames of 60 Hz, vblank %.2f Hz)", address, current, scaled, target.scale, rate);
				}
			}
		}

		// The game's frames per second: one frame every vblanks_per_frame vblanks.
		const f32 fps = rate / static_cast<f32>(std::max<u32>(profile->vblanks_per_frame, 1));
		if (fps < 1.f)
		{
			return;
		}
		const f32 frame_time = 1.f / fps;
		for (const auto& target : profile->game_frame_time_f32)
		{
			const u32 address = resolve(target);
			if (!address)
			{
				continue;
			}

			// Only over a value that looks like a frame time (1/500 to 1/10 s).
			be_t<f32>& value = *vm::_ptr<be_t<f32>>(address);
			const f32 current = value;
			if (current >= 0.002f && current <= 0.1f && current != frame_time)
			{
				value = frame_time;
				static u32 s_logged = 0;
				if (s_logged++ < 4)
				{
					vr_probe_log.notice("Game frame time at 0x%x: %.6f -> %.6f s (%.2f FPS)", address, current, frame_time, fps);
				}
			}
		}
		// Powers of the frame time, over values in the matching range (1/500 to 1/10 s, squared or cubed).
		const auto write_power = [&](const std::vector<title_profile::guest_address>& targets, u32 power)
		{
			const f32 wanted = power == 2 ? frame_time * frame_time : frame_time * frame_time * frame_time;
			const f32 low = power == 2 ? 0.002f * 0.002f : 0.002f * 0.002f * 0.002f;
			const f32 high = power == 2 ? 0.01f : 0.001f;
			for (const auto& target : targets)
			{
				if (const u32 address = resolve(target))
				{
					be_t<f32>& value = *vm::_ptr<be_t<f32>>(address);
					if (const f32 current = value; current >= low && current <= high && current != wanted)
					{
						value = wanted;
					}
				}
			}
		};
		write_power(profile->game_frame_time_sq_f32, 2);
		write_power(profile->game_frame_time_cube_f32, 3);
		const u32 frame_ms = static_cast<u32>(std::lround(1000.f / fps));
		for (const auto& target : profile->game_frame_ms_u32)
		{
			const u32 address = resolve(target);
			if (!address)
			{
				continue;
			}

			// Only over a value that looks like milliseconds per frame.
			be_t<u32>& value = *vm::_ptr<be_t<u32>>(address);
			const u32 current = value;
			if (current >= 1 && current <= 100 && current != frame_ms)
			{
				value = frame_ms;
				static u32 s_logged = 0;
				if (s_logged++ < 4)
				{
					vr_probe_log.notice("Game frame milliseconds at 0x%x: %u -> %u (%.2f FPS)", address, current, frame_ms, fps);
				}
			}
		}
		const f32 frame_ms_f = 1000.f / fps;
		for (const auto& target : profile->game_frame_ms_f32)
		{
			const u32 address = resolve(target);
			if (!address)
			{
				continue;
			}

			// Only over a value that looks like milliseconds per frame.
			be_t<f32>& value = *vm::_ptr<be_t<f32>>(address);
			const f32 current = value;
			if (current >= 2.f && current <= 100.f && current != frame_ms_f)
			{
				value = frame_ms_f;
				static u32 s_logged = 0;
				if (s_logged++ < 4)
				{
					vr_probe_log.notice("Game frame milliseconds (float) at 0x%x: %.3f -> %.3f (%.2f FPS)", address, current, frame_ms_f, fps);
				}
			}
		}
		const u32 whole_fps = static_cast<u32>(std::lround(fps));
		for (const auto& target : profile->game_fps_u32)
		{
			const u32 address = resolve(target);
			if (!address)
			{
				continue;
			}

			// Only over a value that looks like a frame rate.
			be_t<u32>& value = *vm::_ptr<be_t<u32>>(address);
			const u32 current = value;
			if (current >= 10 && current <= 1000 && current != whole_fps)
			{
				value = whole_fps;
				static u32 s_logged = 0;
				if (s_logged++ < 4)
				{
					vr_probe_log.notice("Game frame rate at 0x%x: %u -> %u FPS", address, current, whole_fps);
				}
			}
		}
	}

	const title_profile* camera_probe::profile() const
	{
		if (m_profile_fast_valid.load())
		{
			return m_profile_fast.load();
		}
		const title_profile* result = profile_slow();
		m_profile_fast = result;
		m_profile_fast_valid = true;
		return result;
	}

	const title_profile* camera_probe::profile_slow() const
	{
		const std::string& title = Emu.GetTitleID();
		const std::string& boot = Emu.GetBoot();
		std::lock_guard lock(m_profile_mutex);
		// Called several times per draw: only rebuild the key when the title or boot path changed
		// (building it every call cost Ridge Racer 7 about a tenth of the RSX thread's work).
		if (!m_profile_title.empty() && title == m_profile_title_id && boot == m_profile_boot)
		{
			return m_profile.get();
		}
		m_profile_title_id = title;
		m_profile_boot = boot;
		const std::string executable = running_executable_name();
		if (title + "|" + executable != m_profile_title)
		{
			m_profile_title = title + "|" + executable;
			m_profile = load_title_profile(title, executable);
			if (m_profile)
			{
				const title_profile& p = *m_profile;
				std::string blocks;
				for (const u32 block : p.camera_blocks)
				{
					blocks += fmt::format("%sc[%u]", blocks.empty() ? "" : " ", block);
				}
				vr_probe_log.success("VR profile loaded for %s: camera blocks %s, camera position c[%d] baseline %.9g, "
									 "stereo sep %.9g conv %.9g (%u width rules), screen space c[%d]%s, aspect tolerance %.9g, reference screen %.9g m.",
					m_profile_title, blocks, static_cast<s32>(p.camera_position_slot), p.eye_baseline,
					p.stereo.per_eye_separation, p.stereo.convergence, ::size32(p.stereo_by_target_width),
					static_cast<s32>(p.screen_space_block), p.screen_space_bare_projection ? " + bare projections" : "",
					p.output_aspect_tolerance, p.reference_screen_width);
				for (const auto& rule : p.stereo_by_target_width)
				{
					vr_probe_log.notice("VR profile stereo rule: targets 1/%u of output width: sep %.9g conv %.9g",
						rule.output_width_divisor, rule.per_eye_separation, rule.convergence);
				}
				if (!m_profile->app_version.empty() && Emu.GetAppVersion() != m_profile->app_version)
				{
					vr_probe_log.warning("VR profile for %s was made for version %s; this is version %s.",
						title, m_profile->app_version, Emu.GetAppVersion());
				}
			}
		}
		return m_profile.get();
	}

	void camera_probe::reload_profile()
	{
		std::lock_guard lock(m_profile_mutex);
		m_profile_title.clear();
		m_profile_fast_valid = false;
	}

	camera_probe::camera_probe()
	{
		if (const std::string audit = read_env("RPCS3_VR_AUDIT"); !audit.empty())
		{
			// Yaw in the clip basis (right, up or down, forward): v = Q u.
			m_audit_yaw_deg = static_cast<f32>(std::atof(audit.c_str()));
			const f32 a = m_audit_yaw_deg * 3.14159265358979323846f / 180.f;
			const f32 c = std::cos(a), s = std::sin(a);
			// "pitch:<deg>" pitches instead (up or down, per the clip basis's Y).
			if (audit.starts_with("pitch:"))
			{
				m_audit_yaw_deg = static_cast<f32>(std::atof(audit.c_str() + 6));
				const f32 p = m_audit_yaw_deg * 3.14159265358979323846f / 180.f;
				const f32 cp = std::cos(p), sp = std::sin(p);
				m_audit_rot = {1.f, 0.f, 0.f, 0.f, cp, -sp, 0.f, sp, cp};
				vr_probe_log.success("Rotation audit: right eye pitched by %f degrees, no stereo separation.", m_audit_yaw_deg);
			}
			else
			{
				m_audit_rot = {c, 0.f, s, 0.f, 1.f, 0.f, -s, 0.f, c};
				vr_probe_log.success("Rotation audit: right eye yawed by %f degrees, no stereo separation.", m_audit_yaw_deg);
			}

			if (const std::string fov = read_env("RPCS3_VR_AUDIT_FOV"); !fov.empty())
			{
				m_audit_fov_tan = static_cast<f32>(std::atof(fov.c_str()));
				vr_probe_log.success("Rotation audit: both eyes remapped onto a symmetric frustum, tan %f (projection A=B=%f).",
					m_audit_fov_tan, 1.f / m_audit_fov_tan);
			}
		}

		m_config_path = read_env("RPCS3_VR_PROBE_FILE");
		const std::string cfg = read_env("RPCS3_VR_PROBE");

		if (cfg.empty() && m_config_path.empty())
		{
			// Gate 5 development default: render any profiled title in
			// stereo without requiring the launcher to inject an environment
			// variable. Explicit probe configuration still overrides this.
			parse("render=1");
			return;
		}

		m_enabled = true;

		if (!m_config_path.empty())
		{
			vr_probe_log.success("Camera probe subsystem enabled; watching '%s' (re-read each frame).", m_config_path);
			poll();
		}

		if (!cfg.empty())
		{
			parse(cfg);
		}
	}

	bool camera_probe::render_enabled() const
	{
		if (!m_active.load() || !m_render_enabled || !g_cfg.video.vr.enabled)
		{
			return false;
		}

		// Do not allocate/replay right-eye resources for unrelated titles merely
		// because the development default is armed.
		if (!m_title.empty() && Emu.GetTitleID() != m_title)
		{
			return false;
		}
		return profile() != nullptr;
	}

	bool camera_probe::scene_draws_by_clip_space() const
	{
		const title_profile* p = profile();
		return m_scene_override >= 0 ? m_scene_override != 0 : p && p->clip_space_scene_draws;
	}

	void camera_probe::reset_params()
	{
		m_scene_override = -1;
		m_base = umax;
		m_cam_slot = umax;
		m_yaw = m_pitch = m_roll = 0.f;
		m_tx = m_ty = m_tz = 0.f;
		m_eye = 0.f;
		m_have_xform = false;
		m_require_cam = false;
		m_column_vectors = false;
		m_stereo_sep = 0.f;
		m_stereo_conv = 0.f;
		m_have_stereo = false;
		m_render_enabled = false;
		m_hidden_programs.clear();
		m_why_program = 0;
		m_game_camera_programs.clear();
		m_game_camera_nocolor_programs.clear();
		m_dev_flags = 0;
		m_unbox_fp.clear();
		m_render_camera_right = {};
		m_render_camera_right_valid = false;
		m_have_raw = false;
		m_raw_slot = 0;
		m_raw_comp = 0;
		m_raw_add = 0.f;
		m_title.clear();
	}

	void camera_probe::poll()
	{
		// Frame boundary: the next profile() call rechecks the title and boot path.
		m_profile_fast_valid = false;

		// VR fork dev hook: RPCS3_VR_PROFILE_RELOAD=1. Editing vr_profiles/<TITLE_ID>[.<executable>].json
		// while the game runs applies it within half a second. An invalid edit is logged and leaves
		// the game in 2D until the file is fixed. Off by default: release runs never watch the file.
		static const bool s_profile_reload = read_env("RPCS3_VR_PROFILE_RELOAD") == "1";
		if (const u64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); s_profile_reload && now_ms - m_profile_file_check_ms >= 500)
		{
			m_profile_file_check_ms = now_ms;
			if (const std::string title = Emu.GetTitleID(); !title.empty())
			{
				const std::string dir = fs::get_executable_dir() + "vr_profiles/";
				u64 stamp = 1;
				for (const std::string& path : {dir + title + ".json", dir + title + "." + running_executable_name() + ".json"})
				{
					if (fs::stat_t st{}; fs::get_stat(path, st) && !st.is_directory)
					{
						stamp = stamp * 31 + (static_cast<u64>(st.mtime) ^ (static_cast<u64>(st.size) << 40));
					}
				}
				// 0 = not seen yet: the first check only records the stamp.
				if (m_profile_file_stamp != 0 && stamp != m_profile_file_stamp)
				{
					vr_probe_log.success("VR profile file changed on disk; reloading it for %s.", title);
					reload_profile();
				}
				m_profile_file_stamp = stamp;
			}
		}

		if (!m_enabled || m_config_path.empty())
		{
			return;
		}

		// The probe file is checked at most every 100 ms (a stat per frame cost ~1% of the RSX thread).
		if (const u64 now_us = get_system_time(); now_us - m_config_poll_us < 100'000)
		{
			return;
		}
		else
		{
			m_config_poll_us = now_us;
		}

		fs::stat_t st{};
		if (!fs::get_stat(m_config_path, st) || st.is_directory)
		{
			// File removed -> stop perturbing, keep watching.
			if (m_active.load())
			{
				m_active = false;
				m_description.clear();
				vr_probe_log.success("Probe file gone; perturbation DISARMED (baseline).");
			}
			m_config_stamp = 0;
			return;
		}

		if (m_report_pending && m_active.load())
		{
			m_report_pending = false;
			vr_probe_log.success("Classifier report (one frame, '%s'): perturbed=%u, rejected off-aspect target=%u, rejected no perspective block=%u",
				m_description, m_stat_perturbed.load(), m_stat_rejected_aspect.load(), m_stat_rejected_no_perspective.load());
		}

		const u64 stamp = static_cast<u64>(st.mtime) ^ (static_cast<u64>(st.size) << 32);
		if (stamp == m_config_stamp)
		{
			return;
		}
		m_config_stamp = stamp;

		std::string cfg;
		if (fs::file f{m_config_path})
		{
			cfg = f.to_string();
		}

		// Trim whitespace/newlines.
		while (!cfg.empty() && (cfg.back() == '\n' || cfg.back() == '\r' || cfg.back() == ' ' || cfg.back() == '\t'))
		{
			cfg.pop_back();
		}

		if (cfg.empty())
		{
			m_active = false;
			m_description.clear();
			vr_probe_log.success("Probe file empty; perturbation DISARMED (baseline).");
			return;
		}

		parse(cfg);
	}

	void camera_probe::parse(const std::string& cfg)
	{
		reset_params();

		// Parse "key=value,key=value"
		usz pos = 0;
		while (pos < cfg.size())
		{
			const usz comma = cfg.find(',', pos);
			const std::string tok = cfg.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
			pos = (comma == std::string::npos) ? cfg.size() : comma + 1;

			const usz eq = tok.find('=');
			if (eq == std::string::npos)
				continue;

			const std::string k = tok.substr(0, eq);
			const std::string v = tok.substr(eq + 1);

			auto as_f = [&]()
			{
				return static_cast<f32>(std::atof(v.c_str()));
			};
			auto as_u = [&]()
			{
				return static_cast<u32>(std::atoi(v.c_str()));
			};

			if (k == "base")
				m_base = as_u();
			else if (k == "cam")
				m_cam_slot = as_u();
			else if (k == "yaw")
			{
				m_yaw = as_f();
				m_have_xform = true;
			}
			else if (k == "pitch")
			{
				m_pitch = as_f();
				m_have_xform = true;
			}
			else if (k == "roll")
			{
				m_roll = as_f();
				m_have_xform = true;
			}
			else if (k == "tx")
			{
				m_tx = as_f();
				m_have_xform = true;
			}
			else if (k == "ty")
			{
				m_ty = as_f();
				m_have_xform = true;
			}
			else if (k == "tz")
			{
				m_tz = as_f();
				m_have_xform = true;
			}
			else if (k == "eye")
			{
				m_eye = as_f();
				m_have_xform = true;
			}
			else if (k == "slot")
			{
				m_raw_slot = as_u();
				m_have_raw = true;
			}
			else if (k == "comp")
			{
				m_raw_comp = as_u();
			}
			else if (k == "add")
			{
				m_raw_add = as_f();
			}
			else if (k == "reqcam")
				m_require_cam = (as_u() != 0);
			else if (k == "layout")
				m_column_vectors = v == "columns";
			else if (k == "stereo")
			{
				m_stereo_sep = as_f();
				m_have_stereo = true;
			}
			else if (k == "conv")
			{
				m_stereo_conv = as_f();
			}
			else if (k == "render")
			{
				m_render_enabled = (as_u() != 0);
			}
			else if (k == "scene")
			{
				m_scene_override = as_u() != 0;
			}
			else if (k == "dev")
			{
				m_dev_flags = as_u();
			}
			else if (k == "unboxfp")
			{
				for (const auto& id : fmt::split(v, {"+"}))
				{
					m_unbox_fp.push_back(static_cast<u32>(std::strtoul(id.c_str(), nullptr, 10)));
				}
			}
			else if (k == "hide")
			{
				for (usz start = 0; start < v.size();)
				{
					const usz plus = v.find('+', start);
					const std::string item = v.substr(start, plus == umax ? umax : plus - start);
					const usz at = item.find('@');
					m_hidden_programs.emplace_back(std::strtoull(item.substr(0, at).c_str(), nullptr, 16),
						at == umax ? 0u : static_cast<u32>(std::strtoul(item.substr(at + 1).c_str(), nullptr, 16)));
					start = plus == umax ? v.size() : plus + 1;
				}
			}
			else if (k == "why")
				m_why_program = std::strtoull(v.c_str(), nullptr, 16);
			else if (k == "gamecam")
			{
				for (const auto& id : fmt::split(v, {"+"}))
				{
					(id.ends_with("@nocolor") ? m_game_camera_nocolor_programs : m_game_camera_programs).push_back(std::strtoull(id.c_str(), nullptr, 16));
				}
			}
			else if (k == "title")
				m_title = v;
		}

		if (!m_have_xform && !m_have_raw && !m_have_stereo && !m_render_enabled)
		{
			m_active = false;
			m_description.clear();
			vr_probe_log.warning("Probe config '%s' requests no perturbation; DISARMED (baseline).", cfg);
			return;
		}

		m_stat_perturbed = 0;
		m_stat_rejected_aspect = 0;
		m_stat_rejected_no_perspective = 0;
		m_report_pending = true;

		m_enabled = true;
		m_active = true;
		m_description = cfg;

		vr_probe_log.success("Camera probe ARMED for title '%s': %s", m_title.empty() ? "(any profiled)" : m_title, cfg);
		vr_probe_log.warning("This modifies the transient per-draw constant copy only. "
							 "Guest state is untouched.");
	}

	full_bank_direct_slots::full_bank_direct_slots(const std::vector<u16>* ids)
	{
		if (!ids || ids->empty())
			return;
		t_direct_slots.ids = ids;
		t_direct_slots.read.fill(false);
		for (u16 id : *ids)
		{
			if (id < 468)
				t_direct_slots.read[id] = true;
		}
	}

	full_bank_direct_slots::~full_bank_direct_slots()
	{
		t_direct_slots.ids = nullptr;
	}

	bool camera_probe::apply_render_eye(void* buffer, const u16* reloc, usz reloc_size,
		u16 surface_w, u16 surface_h, f32 eye_sign) const
	{
		if (!render_enabled() || !buffer || (eye_sign != -1.f && eye_sign != 1.f))
		{
			return false;
		}

		const title_profile& profile = *ensure(this->profile());

		if (std::find(profile.game_camera_target_widths.begin(), profile.game_camera_target_widths.end(), surface_w) != profile.game_camera_target_widths.end())
		{
			return false;
		}

		if (profile.orthographic_stereo_angle > 0.f)
		{
			return apply_orthographic_eye(profile, buffer, reloc, reloc_size, surface_w, surface_h, eye_sign);
		}

		// The position policy has a wider domain than the matrix policy. Locate a
		// perspective block first so the camera position's offset follows the exact
		// camera right axis used by this draw (not a global axis or a stale prior draw).
		const u32 base_override[2] = {m_base, m_base + 4};
		const std::span<const u32> camera_blocks = m_base != umax ? std::span<const u32>(base_override) : std::span<const u32>(profile.camera_blocks);
		matrix_block block;
		f32* const* const rows = block.rows;
		const size2u output_eye = g_fxo->get<rsx::avconf>().video_frame_size();
		const f32 camera_aspect = profile.require_camera_aspect && output_eye.height ? static_cast<f32>(output_eye.width) / output_eye.height : 0.f;

		// Profile camera_block_cache: a draw whose camera block and everything else the eye transform below reads
		// equal this eye's previous plain scene-camera draw gets that draw's results copied instead of rebuilt.
		static const int s_block_cache_env = []
		{
			const char* v = std::getenv("RPCS3_VR_BLOCK_CACHE");
			return v ? (v[0] == '1' ? 1 : 0) : -1;
		}();
		const bool block_cache = s_block_cache_env < 0 ? profile.camera_block_cache : s_block_cache_env == 1;
		const u32 cache_eye = eye_sign < 0.f ? 0 : 1;
		const auto build_key = [&](u32 base, bool far_plane, const f32* cam, const f32 (&game)[4][4])
		{
			std::array<u32, 56> cache_key{};
			usz k = 0;
			const auto put = [&](f32 v)
			{
				cache_key[k++] = std::bit_cast<u32>(v);
			};
			const u64 profile_id = reinterpret_cast<u64>(&profile);
			cache_key[k++] = static_cast<u32>(profile_id);
			cache_key[k++] = static_cast<u32>(profile_id >> 32);
			cache_key[k++] = base;
			cache_key[k++] = surface_w | (u32{surface_h} << 16);
			cache_key[k++] = output_eye.width | (output_eye.height << 16);
			cache_key[k++] = (m_vr_view ? 1u : 0u) | (m_vr_hmd_fov ? 2u : 0u) | (m_vr_proj_valid ? 4u : 0u) | (m_vr_flip_y ? 8u : 0u) |
			                 (far_plane ? 16u : 0u) | (m_render_camera_right_valid ? 32u : 0u) | (m_draw_into_display_buffer ? 64u : 0u) |
			                 (m_draw_samples_colour_target ? 128u : 0u) | (m_draw_samples_any_colour_target ? 256u : 0u) |
			                 (m_draw_depth_test ? 512u : 0u) | (rsx::method_registers.depth_test_enabled() ? 1024u : 0u) | (cam ? 2048u : 0u);
			cache_key[k++] = static_cast<u32>(m_draw_program);
			cache_key[k++] = static_cast<u32>(m_draw_program >> 32);
			put(m_draw_hud_scale);
			put(rsx::method_registers.viewport_scale_x());
			put(rsx::method_registers.viewport_scale_y());
			put(rsx::method_registers.viewport_offset_x());
			put(rsx::method_registers.viewport_offset_y());
			cache_key[k++] = u32{rsx::method_registers.surface_clip_width()} | (u32{rsx::method_registers.surface_clip_height()} << 16);
			for (u32 r = 0; r < 4; ++r)
				for (u32 c = 0; c < 4; ++c)
					put(game[r][c]);
			for (const f32 v : m_vr_rot)
				put(v);
			for (const f32 v : m_vr_head_units)
				put(v);
			for (const f32 v : m_vr_eye_fov[cache_eye])
				put(v);
			put(m_vr_proj_x);
			put(m_vr_proj_y);
			put(m_vr_eye_scale);
			put(m_vr_fov_scale);
			put(m_screen_stereo_scale);
			for (u32 i = 0; i < 3; ++i)
				put(cam ? cam[i] : 0.f);
			return cache_key;
		};

		// Fast path: the previous stored draw of this eye told where its block and camera position sit in this
		// program's constants. The key is built from those words as they are now; a match writes the stored
		// result straight back without binding the block (~0.3 ms a frame in Gran Turismo 5 at 2,600 draws).
		if (block_cache && m_base == umax && camera_blocks.size() == 1 && !rsx::vr::depth_remap_active())
		{
			const auto& fast = m_eye_fast[cache_eye];
			const auto& entry = m_eye_block_cache[cache_eye];
			if (fast.valid && entry.valid && fast.reloc == reloc && fast.reloc_size == reloc_size && fast.direct_ids == t_direct_slots.ids)
			{
				char* base_ptr = static_cast<char*>(buffer);
				f32 game[4][4];
				for (u32 k = 0; k < 4; ++k)
				{
					const f32* slot = reinterpret_cast<const f32*>(base_ptr + fast.row_off[k]);
					for (u32 i = 0; i < 4; ++i)
					{
						// Row-vector convention: a transposed layout stores column j of M in slot j.
						if (fast.transposed)
							game[i][k] = slot[i];
						else
							game[k][i] = slot[i];
					}
				}
				f32* const cam = fast.cam_off != umax ? reinterpret_cast<f32*>(base_ptr + fast.cam_off) : nullptr;
				if (build_key(fast.base, false, cam, game) == entry.key)
				{
					for (u32 k = 0; k < 4; ++k)
					{
						f32* slot = reinterpret_cast<f32*>(base_ptr + fast.row_off[k]);
						for (u32 i = 0; i < 4; ++i)
						{
							slot[i] = fast.transposed ? entry.rows[i][k] : entry.rows[k][i];
						}
					}
					if (cam)
					{
						std::copy(std::begin(entry.cam), std::end(entry.cam), cam);
					}
					m_vr_proj_x = entry.proj_x;
					m_vr_proj_y = entry.proj_y;
					m_vr_proj_valid = true;
					m_proj_refreshed = entry.proj_refreshed;
					if (entry.proj_refreshed)
					{
						m_scene_proj_x = entry.proj_x;
						m_scene_proj_y = entry.proj_y;
					}
					m_render_camera_right = entry.camera_right;
					m_render_camera_right_valid = entry.camera_right_valid;
					f32 out_rows[4][4];
					f32* out_ptrs[4] = {out_rows[0], out_rows[1], out_rows[2], out_rows[3]};
					for (u32 r = 0; r < 4; ++r)
						for (u32 c = 0; c < 4; ++c)
							out_rows[r][c] = entry.rows[r][c];
					finish_eye_block(profile, buffer, reloc, reloc_size, eye_sign, true, false, game, out_ptrs);
					m_eye_fast_hits++;
					return true;
				}
			}
			m_eye_fast_misses++;
		}

		if (!bind_camera_block(block, buffer, reloc, reloc_size, camera_blocks, profile.column_vectors, profile.require_rigid_camera, profile.xyw_rows, camera_aspect,
				m_base != umax ? std::span<const std::array<u32, 4>>() : std::span<const std::array<u32, 4>>(profile.camera_block_slots),
				m_base != umax ? std::span<const u32>() : std::span<const u32>(profile.nonrigid_camera_blocks),
				m_base != umax ? std::span<const u32>() : std::span<const u32>(profile.row_vector_blocks),
				m_base != umax ? std::span<const u32>() : std::span<const u32>(profile.either_layout_blocks)))
		{
			apply_vr_screen_space(profile, buffer, reloc, reloc_size, surface_w, surface_h, eye_sign);
			return false;
		}

		if (static const bool s_frame_stats = std::getenv("RPCS3_VR_FRAMESTATS") != nullptr; s_frame_stats && !g_frame_camera_hash)
		{
			u64 h = 0xcbf29ce484222325ull;
			for (u32 i = 0; i < 4; ++i)
				for (u32 j = 0; j < 4; ++j)
					h = (h ^ std::bit_cast<u32>(rows[i][j])) * 0x100000001b3ull;
			g_frame_camera_hash = h | 1;
		}

		// Profile game_camera_aspects: a view rendered for a texture keeps the game's camera.
		if (!profile.game_camera_aspects.empty())
		{
			const f32 nx = std::sqrt(rows[0][0] * rows[0][0] + rows[1][0] * rows[1][0] + rows[2][0] * rows[2][0]);
			const f32 ny = std::sqrt(rows[0][1] * rows[0][1] + rows[1][1] * rows[1][1] + rows[2][1] * rows[2][1]);
			if (nx > 1e-8f && std::any_of(profile.game_camera_aspects.begin(), profile.game_camera_aspects.end(), [&](f32 aspect)
								  {
									  return std::fabs((ny / nx) / aspect - 1.f) <= 0.0025f;
								  }))
			{
				block.release();
				return false;
			}
		}

		// A camera draw through part of a view target (Gran Turismo 5's rear-view mirror, 448x86 at the
		// top of the screen) is a picture on the screen: into the HUD box, with the game's camera.
		if (profile.screen_space_subviewport_cameras_in_box && profile.screen_space_hud_box_after_shader &&
			m_vr_view && m_vr_hmd_fov && m_vr_proj_valid && output_eye.height &&
			profile.is_view_target(surface_w, surface_h, static_cast<f32>(output_eye.width) / output_eye.height) &&
			viewport_inside_shown_region())
		{
			block.release();
			m_hud_env_request = true;
			return false;
		}

		// Profile boxed_cameras: 3D screen elements (menu panels) drawn with their own fixed camera.
		if (!profile.screen_space_boxed_cameras.empty() && m_vr_view && m_vr_hmd_fov && m_vr_proj_valid &&
			std::any_of(profile.screen_space_boxed_cameras.begin(), profile.screen_space_boxed_cameras.end(), [&](const std::array<f32, 4>& w_row)
				{
					for (u32 r = 0; r < 4; ++r)
					{
						if (std::fabs(rows[r][3] - w_row[r]) > 1e-3f * std::max(1.f, std::fabs(w_row[r])))
							return false;
					}
					return true;
				}))
		{
			block.release();
			m_hud_env_request = true;
			return false;
		}

		// A perspective draw straight into a display buffer is part of the 2D screen (Gran Turismo 5's
		// menu cards, whose 3D scenes render elsewhere): it goes into the HUD box with the rest.
		if (profile.screen_space_hud_box_after_shader && profile.screen_space_hud_display_buffers_only && m_draw_into_display_buffer &&
			!(profile.screen_space_hud_skips_passes && m_draw_samples_colour_target) && m_vr_view && m_vr_hmd_fov && m_vr_proj_valid)
		{
			block.release();
			m_hud_env_request = true;
			return false;
		}

		f32 game_block[4][4];
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				game_block[r][c] = rows[r][c];
			}
		}

		std::array<u32, 56> cache_key{};
		f32* cache_cam = nullptr;
		if (block_cache)
		{
			const u32 cache_cam_slot = m_cam_slot != umax ? m_cam_slot : profile.camera_position_slot;
			cache_cam = cache_cam_slot != umax ? find_slot(buffer, reloc, reloc_size, cache_cam_slot) : nullptr;
			cache_key = build_key(block.base(), block.far_plane(), cache_cam, game_block);

			if (const auto& entry = m_eye_block_cache[cache_eye]; entry.valid && entry.key == cache_key && !rsx::vr::depth_remap_active())
			{
				for (u32 r = 0; r < 4; ++r)
					for (u32 c = 0; c < 4; ++c)
						rows[r][c] = entry.rows[r][c];
				if (cache_cam)
				{
					std::copy(std::begin(entry.cam), std::end(entry.cam), cache_cam);
				}
				m_vr_proj_x = entry.proj_x;
				m_vr_proj_y = entry.proj_y;
				m_vr_proj_valid = true;
				m_proj_refreshed = entry.proj_refreshed;
				if (entry.proj_refreshed)
				{
					m_scene_proj_x = entry.proj_x;
					m_scene_proj_y = entry.proj_y;
				}
				m_render_camera_right = entry.camera_right;
				m_render_camera_right_valid = entry.camera_right_valid;
				finish_eye_block(profile, buffer, reloc, reloc_size, eye_sign, true, false, game_block, rows);
				return true;
			}
		}

		// A full-screen pass building view rays from a translation-free camera block
		// (see screen_space_rotation_only_passthrough): it follows head rotation, but
		// takes no eye offset, head translation or stereo shear.
		bool rotation_only_pass = false;
		if (profile.screen_space_rotation_only_passthrough)
		{
			constexpr f32 eps = 1e-5f;
			rotation_only_pass = std::fabs(rows[3][0]) < eps && std::fabs(rows[3][1]) < eps && std::fabs(rows[3][3]) < eps &&
			                     !rsx::method_registers.depth_test_enabled();
		}

		const auto& avconf = g_fxo->get<rsx::avconf>();
		const size2u eye = avconf.video_frame_size();
		if (!surface_w || !surface_h || !eye.width || !eye.height)
		{
			return false;
		}

		const f32 target_aspect = static_cast<f32>(surface_w) / surface_h;
		const f32 output_aspect = static_cast<f32>(eye.width) / eye.height;
		const bool output_aspect_match = profile.is_view_target(surface_w, surface_h, output_aspect);
		static_cast<void>(target_aspect);
		// Profile offaspect_player_views: a draw into an off-aspect target with the player's camera (its projection has the output
		// aspect) is part of the player's view sampled at screen position: Ridge Racer 7 renders its
		// road reflections with the game camera into 128x128 tiles of a 512x128 target and the road
		// looks them up at its own clip position. It takes the same rotation, eye offset and headset
		// FOV as the scene, but keeps its viewport (no undo_viewport), and it does not refresh the
		// camera-right axis or the stored eye block, which stay those of real camera views. Other
		// off-aspect cameras (cube-map faces, shadow maps) keep the game camera in both eyes.
		const bool screen_sampled_view = profile.offaspect_player_views && !output_aspect_match && has_camera_aspect(rows, output_aspect);
		const bool view_draw = output_aspect_match || screen_sampled_view;

		// Rotation-invariance audit (see m_audit_yaw_deg): the same classifier and
		// clip-space rotation as the headset path, with the left eye unrotated and
		// no eye offsets, so the only difference between the eyes is a known yaw.
		if (m_audit_yaw_deg != 0.f && !m_vr_view)
		{
			if (!view_draw)
			{
				return false;
			}

			static constexpr std::array<f32, 9> identity3 = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
			apply_vr_rotation(rows, eye_sign > 0.f ? m_audit_rot : identity3, {});

			// The headset path's FOV remap, onto a symmetric frustum (l, r, u, d).
			if (m_audit_fov_tan > 0.f && m_vr_proj_valid)
			{
				const f32 t[4] = {-m_audit_fov_tan, m_audit_fov_tan, m_audit_fov_tan, -m_audit_fov_tan};
				remap_to_eye_fov(rows, t, m_vr_proj_x, m_vr_proj_y, screen_sampled_view);
			}
			if (output_aspect_match)
			{
				store_eye_block(eye_sign, game_block, rows);
			}
			apply_linked_camera_blocks(profile, buffer, reloc, reloc_size, game_block, rows);
			if (m_vr_proj_valid && !m_audit_logged)
			{
				m_audit_logged = true;
				vr_probe_log.success("Rotation audit: game projection A=%f B=%f (tools/rotation_audit.py --proj).",
					m_vr_proj_x, m_vr_proj_y);
			}
			return true;
		}

		if (rotation_only_pass)
		{
			if (!view_draw || !m_vr_view)
			{
				block.release();
				return false;
			}
			apply_vr_rotation(rows, m_vr_rot, {});
			if (m_vr_hmd_fov && m_vr_proj_valid)
			{
				remap_to_eye_fov(rows, m_vr_eye_fov[eye_sign < 0.f ? 0 : 1], m_vr_proj_x, m_vr_proj_y, screen_sampled_view);
			}
			return true;
		}

		// A camera block that is a bare projection (no view rotation or translation
		// folded in) draws in the game camera's own space. In WipEout that is the
		// main-menu particle cloud, and no race camera block has this form. The
		// depth-offset form (W = z + d) is Blur's 3D HUD, 42.65 units ahead; Blur's
		// light volumes are exact bare projections, so the two are separate options.
		// When the profile says so, it is part of the screen and goes into the same
		// fixed box as the HUD instead of following the head.
		// A bare-projection quad that samples a render target is a pass (The Darkness: its post-processing and HDR
		// luminance chain, which sample per-eye targets in screen space): left as drawn in desktop stereo too.
		// Sheared, the luminance passes read and wrote shifted texels and the exposure blew the image out red.
		// In desktop stereo every bare-projection draw without a depth test that can reject is screen-space work
		// too (no HUD box there): The Darkness builds its colour-grading LUT with depth func ALWAYS quads; shifted,
		// the LUT slices were written a few texels off and the tone map turned the image pink.
		// A bare_projection block list limits all this to bare projections bound from those blocks.
		const bool bare_projection_block = profile.screen_space_bare_projection_blocks.empty() ||
			std::find(profile.screen_space_bare_projection_blocks.begin(), profile.screen_space_bare_projection_blocks.end(), block.base()) !=
				profile.screen_space_bare_projection_blocks.end();
		if (!m_vr_view && profile.screen_space_bare_projection && bare_projection_block &&
			((profile.screen_space_hud_skips_passes && m_draw_samples_any_colour_target) || !m_draw_depth_test))
		{
			constexpr f32 eps = 1e-5f;
			if (std::fabs(rows[0][1]) < eps && std::fabs(rows[0][2]) < eps && std::fabs(rows[0][3]) < eps &&
				std::fabs(rows[1][0]) < eps && std::fabs(rows[1][2]) < eps && std::fabs(rows[1][3]) < eps &&
				std::fabs(rows[2][0]) < eps && std::fabs(rows[2][1]) < eps &&
				std::fabs(rows[3][0]) < eps && std::fabs(rows[3][1]) < eps && std::fabs(rows[3][3]) < eps &&
				std::fabs(rows[2][3]) > eps)
			{
				block.release();
				return false;
			}
		}

		// An off-aspect bare projection (God of War's HUD and menus) is screen space whether or not the headset view
		// is on: on the fixed screen (no view) it is left as drawn, not taken for a camera draw. Counted as one there,
		// a menu frame went back to the headset view, where the same draws are boxed and count as none, so the pause
		// and Power Up menus switched between the two every few frames (shown twice, offset).
		if (output_aspect_match && !m_vr_view && profile.screen_space_offaspect_projection)
		{
			constexpr f32 eps = 1e-5f;
			if (std::fabs(rows[0][1]) < eps && std::fabs(rows[0][2]) < eps && std::fabs(rows[0][3]) < eps &&
				std::fabs(rows[1][0]) < eps && std::fabs(rows[1][2]) < eps && std::fabs(rows[1][3]) < eps &&
				std::fabs(rows[2][0]) < eps && std::fabs(rows[2][1]) < eps &&
				std::fabs(rows[3][0]) < eps && std::fabs(rows[3][1]) < eps && std::fabs(rows[3][3]) < eps &&
				std::fabs(rows[2][3]) > eps && std::fabs(rows[0][0]) > eps &&
				std::fabs(std::fabs(rows[1][1] / rows[0][0]) / output_aspect - 1.f) > 0.1f)
			{
				block.release();
				return false;
			}
		}

		if (output_aspect_match && m_vr_view && m_vr_hmd_fov &&
			(profile.screen_space_bare_projection || profile.screen_space_depth_offset_projection || profile.screen_space_offaspect_projection))
		{
			constexpr f32 eps = 1e-5f;
			const bool depth_offset = std::fabs(rows[3][3]) >= eps;
			// A plane at a fixed depth may also be shifted in x/y (NFS Most Wanted's HUD:
			// a pixel-origin plane 1108.5 units ahead), so the x/y translation is only
			// required to be zero for the bare projection.
			const bool camera_space =
				std::fabs(rows[0][1]) < eps && std::fabs(rows[0][2]) < eps && std::fabs(rows[0][3]) < eps &&
				std::fabs(rows[1][0]) < eps && std::fabs(rows[1][2]) < eps && std::fabs(rows[1][3]) < eps &&
				std::fabs(rows[2][0]) < eps && std::fabs(rows[2][1]) < eps &&
				(depth_offset || (std::fabs(rows[3][0]) < eps && std::fabs(rows[3][1]) < eps)) &&
				std::fabs(rows[2][3]) > eps;
			// God of War HD: the scene and the HUD are both bare projections; only the HUD's
			// square-pixel aspect (B/A 1.33) is not the output's (1.78).
			const bool off_aspect = camera_space && !depth_offset && std::fabs(rows[0][0]) > eps &&
			                        std::fabs(std::fabs(rows[1][1] / rows[0][0]) / output_aspect - 1.f) > 0.1f;
			const bool bare_projection = camera_space &&
			                             (depth_offset ? profile.screen_space_depth_offset_projection :
														 ((profile.screen_space_bare_projection && bare_projection_block) || (profile.screen_space_offaspect_projection && off_aspect)));
			if (bare_projection && profile.screen_space_hud_skips_passes && m_draw_samples_any_colour_target)
			{
				// A full-screen pass drawn with the projection (The Darkness composites its 1024x576 scene into
				// the display buffer this way): it samples per-eye targets in screen space, so leave it as drawn.
				block.release();
				return false;
			}
			if (bare_projection)
			{
				// Still the game's projection, so it keeps the FOV cache valid on
				// screens with no other camera draws. Not an off-aspect HUD's projection.
				if (!off_aspect)
				{
					m_vr_proj_x = std::fabs(rows[0][0] / rows[2][3]);
					m_vr_proj_y = std::fabs(rows[1][1] / rows[2][3]);
					m_vr_proj_valid = true;
				}
				map_vr_screen_box(rows, eye_sign, output_aspect);
				return false;
			}
		}

		// The camera position is a camera-world point in the native oracle. Unlike
		// the matrix it changes on the cascade route too. A cascade's camera block is
		// a perspective projection but its clip-X column is not camera right
		// (WipEout native capture dot=+0.007); use the most recent real camera view
		// for that global axis. Output-aspect camera draws establish/refresh it.
		// Head rotation first, so camera right and the eye offsets below follow
		// the rotated view.
		// A sky on the far plane is infinitely far: it turns with the head but takes no head
		// translation or eye offset (a finite eye offset gives it the parallax of the dome's
		// real size, contradicting its far-plane depth).
		const bool at_infinity = block.far_plane() && m_vr_view;
		// A camera-facing sprite in a nonrigid_camera_blocks block (Bayonetta's c[24] flames and tree cards) carries
		// its own size in the projection's scale: it must neither replace the cached projection nor size its eye offset.
		const bool sprite_block = std::find(profile.nonrigid_camera_blocks.begin(), profile.nonrigid_camera_blocks.end(), block.base()) !=
		                              profile.nonrigid_camera_blocks.end() && is_screen_aligned(rows);
		// The sprite's clip x per view unit: the cached projection's x scale times its w scale (taken before the rotation).
		const f32 sprite_x_per_unit = sprite_block && m_vr_proj_valid ?
			m_vr_proj_x * std::sqrt(rows[0][3] * rows[0][3] + rows[1][3] * rows[1][3] + rows[2][3] * rows[2][3]) : 0.f;
		// A depth_remap_programs pass draws its own volume (Sonic's shadow cascades: a projection times a non-uniform
		// view-space scale), whose matrix is no camera projection: it must not replace the cached one, which the pass's
		// ray remap reads (taken from it, Sonic's rebuilt view rays were off by ~3x away from the view's centre).
		const bool remap_volume = std::find(profile.depth_remap_programs.begin(), profile.depth_remap_programs.end(), m_draw_program) !=
		                              profile.depth_remap_programs.end() ||
		                          std::find(profile.depth_remap_volume_programs.begin(), profile.depth_remap_volume_programs.end(), m_draw_program) !=
		                              profile.depth_remap_volume_programs.end();
		m_proj_refreshed = false;
		if (view_draw && m_vr_view)
		{
			apply_vr_rotation(rows, m_vr_rot, at_infinity ? std::array<f32, 3>{} : m_vr_head_units, !sprite_block && !remap_volume);
		}
		if (output_aspect_match && m_proj_refreshed)
		{
			m_scene_proj_x = m_vr_proj_x;
			m_scene_proj_y = m_vr_proj_y;
		}

		if (output_aspect_match)
		{
			m_render_camera_right = {rows[0][0], rows[1][0], rows[2][0]};
			const f32 length = std::sqrt(m_render_camera_right[0] * m_render_camera_right[0] +
										 m_render_camera_right[1] * m_render_camera_right[1] +
										 m_render_camera_right[2] * m_render_camera_right[2]);
			if (length > 1e-8f)
			{
				for (f32& v : m_render_camera_right)
					v /= length;
				m_render_camera_right_valid = true;
			}
		}

		const u32 cam_slot = m_cam_slot != umax ? m_cam_slot : profile.camera_position_slot;
		if (f32* cam = cam_slot != umax ? find_slot(buffer, reloc, reloc_size, cam_slot) : nullptr;
			cam && m_render_camera_right_valid && !at_infinity && plausible_camera_position(cam, game_block, profile.eye_baseline))
		{
			const f32 half_eye_baseline = profile.eye_baseline * 0.5f * (m_vr_view ? m_vr_eye_scale : m_screen_stereo_scale);
			cam[0] += eye_sign * half_eye_baseline * m_render_camera_right[0];
			cam[1] += eye_sign * half_eye_baseline * m_render_camera_right[1];
			cam[2] += eye_sign * half_eye_baseline * m_render_camera_right[2];
		}

		if (!view_draw)
		{
			return false;
		}

		// The profile's measured shear for this target width (WipEout's half-resolution
		// pass uses 75% of the full-resolution shear, not 50%). Infinity layers stay
		// on the converged rule until their program/pass keys are profile data.
		const auto& stereo_rule = profile.stereo_for(surface_w, eye.width);
		const f32 per_eye_sep = stereo_rule.per_eye_separation;
		const f32 convergence = stereo_rule.convergence;
		// On a fixed screen larger than the one the game tuned its stereo for, the
		// whole separation is scaled down so far objects keep the same physical
		// disparity (see set_screen_stereo_scale). The headset path uses eye_scale.
		const f32 sep = eye_sign * per_eye_sep * (m_vr_view ? 1.f : m_screen_stereo_scale);

		if (m_vr_view)
		{
			// Headset eyes are parallel: keep the formula's eye translation
			// (clip.x -= sep*conv, the same offset as the camera position's) and drop
			// its convergence image shift (clip.x += sep*clip.w).
			if (!at_infinity)
			{
				if (profile.stereo_eye_offset_from_baseline)
				{
					// Move the eye by half the baseline along the camera's right: clip.x
					// changes by that distance times the length of its x row.
					f32 clip_x_per_unit = sprite_x_per_unit > 0.f ? sprite_x_per_unit :
						std::sqrt(rows[0][0] * rows[0][0] + rows[1][0] * rows[1][0] + rows[2][0] * rows[2][0]);
					if (profile.stereo_eye_offset_per_w && sprite_x_per_unit <= 0.f)
					{
						// The x row is the projection's x scale times any object scale, the w row the w scale (1 for
						// w = +-z) times the same object scale: their ratio is the object-free x scale.
						if (const f32 w_len = std::sqrt(rows[0][3] * rows[0][3] + rows[1][3] * rows[1][3] + rows[2][3] * rows[2][3]); w_len > 1e-12f)
						{
							clip_x_per_unit /= w_len;
						}
					}
					rows[3][0] -= eye_sign * profile.eye_baseline * 0.5f * m_vr_eye_scale * clip_x_per_unit;
				}
				else
				{
					rows[3][0] -= sep * m_vr_eye_scale * convergence;
				}
			}

			if (m_vr_hmd_fov && m_vr_proj_valid)
			{
				remap_to_eye_fov(rows, m_vr_eye_fov[eye_sign < 0.f ? 0 : 1], m_vr_proj_x, m_vr_proj_y, screen_sampled_view);
			}
			else if (m_vr_fov_scale != 1.f)
			{
				const f32 zoom = 1.f / m_vr_fov_scale;
				for (u32 r = 0; r < 4; ++r)
				{
					rows[r][0] *= zoom;
					rows[r][1] *= zoom;
				}
			}
			finish_eye_block(profile, buffer, reloc, reloc_size, eye_sign, output_aspect_match, remap_volume, game_block, rows);
			// A plain scene-camera draw: its results serve the next draws with the same inputs (camera_block_cache).
			if (block_cache && output_aspect_match && !screen_sampled_view && !remap_volume && !sprite_block && m_vr_proj_valid &&
				m_audit_yaw_deg == 0.f && !rsx::vr::depth_remap_active() && profile.linked_camera_blocks.empty() && !profile.camera_palette_last)
			{
				auto& entry = m_eye_block_cache[cache_eye];
				entry.key = cache_key;
				for (u32 r = 0; r < 4; ++r)
					for (u32 c = 0; c < 4; ++c)
						entry.rows[r][c] = rows[r][c];
				if (cache_cam)
				{
					std::copy(cache_cam, cache_cam + 3, entry.cam);
				}
				entry.proj_x = m_vr_proj_x;
				entry.proj_y = m_vr_proj_y;
				entry.proj_refreshed = m_proj_refreshed;
				entry.camera_right = m_render_camera_right;
				entry.camera_right_valid = m_render_camera_right_valid;
				entry.valid = true;
				// The fast path above: the block's slots in this program's constants (all four present, not a sky).
				auto& fast = m_eye_fast[cache_eye];
				fast.valid = !block.far_plane() && block.slot(0) && block.slot(1) && block.slot(2) && block.slot(3) && m_base == umax && camera_blocks.size() == 1;
				if (fast.valid)
				{
					const char* base_ptr = static_cast<const char*>(buffer);
					fast.reloc = reloc;
					fast.reloc_size = reloc_size;
					fast.direct_ids = t_direct_slots.ids;
					fast.base = block.base();
					fast.transposed = block.transposed();
					for (u32 k = 0; k < 4; ++k)
						fast.row_off[k] = static_cast<u32>(reinterpret_cast<const char*>(block.slot(k)) - base_ptr);
					fast.cam_off = cache_cam ? static_cast<u32>(reinterpret_cast<const char*>(cache_cam) - base_ptr) : umax;
				}
			}
			return true;
		}

		for (u32 r = 0; r < 4; ++r)
		{
			rows[r][0] += sep * rows[r][3];
		}
		rows[3][0] -= sep * convergence;
		if (output_aspect_match)
		{
			store_eye_block(eye_sign, game_block, rows);
		}
		finish_eye_block(profile, buffer, reloc, reloc_size, eye_sign, output_aspect_match, remap_volume, game_block, rows);
		return true;
	}

	bool camera_probe::apply_orthographic_eye(const title_profile& profile, void* buffer, const u16* reloc, usz reloc_size,
		u16 surface_w, u16 surface_h, f32 eye_sign) const
	{
		// An orthographic game (Fez) has no eye position to move: each eye's view is turned by the profile's angle about
		// the convergence depth instead, so clip x moves by tan(angle) x (depth - convergence) in world units. Only
		// depth-tested draws into view targets (the world): full-screen passes and the HUD stay as drawn and sample
		// each eye's own targets. The picture goes on the fixed screen (VKGSRender::vr_update_view).
		const size2u eye = g_fxo->get<rsx::avconf>().video_frame_size();
		if (!m_draw_depth_test || !surface_w || !surface_h || !eye.width || !eye.height ||
			!profile.is_view_target(surface_w, surface_h, static_cast<f32>(eye.width) / eye.height))
		{
			return false;
		}
		const auto func = rsx::method_registers.depth_func();
		// Depth grows away from the viewer unless the test passes greater depths.
		const f32 depth_sign = func == rsx::comparison_function::greater || func == rsx::comparison_function::greater_or_equal ? -1.f : 1.f;
		const f32 tan_angle = std::tan(profile.orthographic_stereo_angle * 0.017453292f) * m_screen_stereo_scale;
		for (const u32 base : profile.camera_blocks)
		{
			const bool rows_layout = std::find(profile.row_vector_blocks.begin(), profile.row_vector_blocks.end(), base) != profile.row_vector_blocks.end();
			matrix_block block;
			if (!block.bind(buffer, reloc, reloc_size, base, profile.column_vectors != rows_layout) || is_perspective(block.rows))
			{
				continue;
			}
			f32* const* const rows = block.rows;
			const f32 x_scale = std::sqrt(rows[0][0] * rows[0][0] + rows[1][0] * rows[1][0] + rows[2][0] * rows[2][0]);
			const f32 z_scale = std::sqrt(rows[0][2] * rows[0][2] + rows[1][2] * rows[1][2] + rows[2][2] * rows[2][2]);
			if (x_scale < 1e-8f || z_scale < 1e-12f)
			{
				continue;
			}
			// clip x += k (clip z - z0), z0 the clip z of the convergence depth.
			const f32 k = eye_sign * depth_sign * tan_angle * x_scale / z_scale;
			const f32 z0 = profile.orthographic_stereo_convergence_z > 0.f ? profile.orthographic_stereo_convergence_z :
				rows[3][2] + depth_sign * profile.orthographic_stereo_convergence * z_scale;
			for (u32 r = 0; r < 4; ++r)
			{
				rows[r][0] += k * (rows[r][2] - z0 * rows[r][3]);
			}
			return true;
		}
		return false;
	}

	void camera_probe::finish_eye_block(const title_profile& profile, void* buffer, const u16* reloc, usz reloc_size, f32 eye_sign,
		bool output_aspect_match, bool remap_volume, const f32 (&game)[4][4], f32* const rows[4]) const
	{
		const u32 eye = eye_sign < 0.f ? 0 : 1;
		if (output_aspect_match)
		{
			store_eye_block(eye_sign, game, rows);
		}
		// The scene camera's matrices per eye, for the depth remap of a depth_remap_programs pass drawn through its own
		// volume matrix (Sonic's shadow cascades: a projection times a non-uniform scale). Its eye offset is sized by its
		// own clip x per unit, which that scale distorts: built from it, the remap shifted the rebuilt positions
		// sideways, a different way in each eye (shadows on the green lumps at each eye's outer side, Matt 2026-10-07).
		if (output_aspect_match && !remap_volume)
		{
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 c = 0; c < 4; ++c)
				{
					m_scene_game_block[eye][r][c] = game[r][c];
					m_scene_eye_block[eye][r][c] = rows[r][c];
				}
			}
			m_scene_block_valid[eye] = true;
		}
		m_remap_scene_eye = remap_volume && m_scene_block_valid[eye] ? static_cast<s32>(eye) : -1;
		if (m_remap_scene_eye >= 0)
		{
			// The volume gets the scene's eye transform as a clip-space map X = scene game^-1 * scene eye (row vectors):
			// its rows = its game matrix * X, so it covers the same pixels as the scene points it shades.
			f64 a[4][8];
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 k = 0; k < 4; ++k)
				{
					a[r][k] = m_scene_game_block[eye][r][k];
					a[r][k + 4] = r == k ? 1.0 : 0.0;
				}
			}
			bool ok = true;
			for (u32 k = 0; k < 4 && ok; ++k)
			{
				u32 pivot = k;
				for (u32 r = k + 1; r < 4; ++r)
				{
					if (std::fabs(a[r][k]) > std::fabs(a[pivot][k]))
						pivot = r;
				}
				if (std::fabs(a[pivot][k]) < 1e-12)
				{
					ok = false;
					break;
				}
				for (u32 j = 0; j < 8; ++j)
					std::swap(a[k][j], a[pivot][j]);
				const f64 inv = 1.0 / a[k][k];
				for (u32 j = 0; j < 8; ++j)
					a[k][j] *= inv;
				for (u32 r = 0; r < 4; ++r)
				{
					if (r == k)
						continue;
					const f64 f = a[r][k];
					for (u32 j = 0; j < 8; ++j)
						a[r][j] -= f * a[k][j];
				}
			}
			if (ok)
			{
				f64 x[4][4]{};
				for (u32 r = 0; r < 4; ++r)
					for (u32 k = 0; k < 4; ++k)
						for (u32 j = 0; j < 4; ++j)
							x[r][k] += a[r][j + 4] * m_scene_eye_block[eye][j][k];
				for (u32 r = 0; r < 4; ++r)
				{
					f64 out[4]{};
					for (u32 k = 0; k < 4; ++k)
						for (u32 j = 0; j < 4; ++j)
							out[k] += static_cast<f64>(game[r][j]) * x[j][k];
					for (u32 k = 0; k < 4; ++k)
						rows[r][k] = static_cast<f32>(out[k]);
				}
			}
		}
		apply_linked_camera_blocks(profile, buffer, reloc, reloc_size, game, rows);
		m_remap_scene_eye = -1;
	}

	void camera_probe::apply_linked_camera_blocks(const title_profile& profile, void* buffer, const u16* reloc, usz reloc_size,
		const f32 (&game)[4][4], f32* const rows[4]) const
	{
		if (rsx::vr::depth_remap_active())
		{
			if (m_remap_scene_eye >= 0)
			{
				f32* scene_rows[4];
				for (u32 r = 0; r < 4; ++r)
				{
					scene_rows[r] = m_scene_eye_block[m_remap_scene_eye][r];
				}
				store_depth_remap(m_scene_game_block[m_remap_scene_eye], scene_rows);
			}
			else
			{
				store_depth_remap(game, rows);
			}
		}
		if (profile.linked_camera_blocks.empty() && !profile.camera_palette_last)
		{
			return;
		}

		// X = game^-1 * eye (row vectors: clip = v * M), by Gauss-Jordan with partial pivoting.
		f64 a[4][8];
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				a[r][c] = game[r][c];
				a[r][c + 4] = r == c ? 1.0 : 0.0;
			}
		}
		for (u32 c = 0; c < 4; ++c)
		{
			u32 pivot = c;
			for (u32 r = c + 1; r < 4; ++r)
			{
				if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
					pivot = r;
			}
			if (std::fabs(a[pivot][c]) < 1e-12)
			{
				return;
			}
			if (pivot != c)
			{
				for (u32 k = 0; k < 8; ++k)
					std::swap(a[c][k], a[pivot][k]);
			}
			const f64 inv = 1.0 / a[c][c];
			for (u32 k = 0; k < 8; ++k)
				a[c][k] *= inv;
			for (u32 r = 0; r < 4; ++r)
			{
				if (r == c || a[r][c] == 0.0)
					continue;
				const f64 f = a[r][c];
				for (u32 k = 0; k < 8; ++k)
					a[r][k] -= f * a[c][k];
			}
		}
		f64 x[4][4];
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				f64 sum = 0.0;
				for (u32 k = 0; k < 4; ++k)
					sum += a[r][k + 4] * rows[k][c];
				x[r][c] = sum;
			}
		}

		for (const u32 base : profile.linked_camera_blocks)
		{
			const bool block_rows = std::find(profile.row_vector_blocks.begin(), profile.row_vector_blocks.end(), base) != profile.row_vector_blocks.end();
			matrix_block linked;
			// The bound camera block itself (a program whose camera sits where others keep the previous frame's: Flower's
			// sky draws through c[260]) already has the eye transform. Row layout: the same slots. Column layout binds a
			// transposed copy, and the camera block, written back last, wins over this one.
			if (!linked.bind(buffer, reloc, reloc_size, base, profile.column_vectors != block_rows, false) || linked.rows[0] == rows[0])
			{
				continue;
			}
			f64 m[4][4];
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 c = 0; c < 4; ++c)
				{
					f64 sum = 0.0;
					for (u32 k = 0; k < 4; ++k)
						sum += linked.rows[r][k] * x[k][c];
					m[r][c] = sum;
				}
			}
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 c = 0; c < 4; ++c)
					linked.rows[r][c] = static_cast<f32>(m[r][c]);
			}
		}

		if (!profile.camera_palette_last)
		{
			return;
		}
		// The camera's projection: clip z = a * clip w + b (row vectors: z column = a * w column + b in the translation row).
		f64 ww = 0.0, zw = 0.0;
		for (u32 r = 0; r < 3; ++r)
		{
			ww += f64{game[r][3]} * game[r][3];
			zw += f64{game[r][2]} * game[r][3];
		}
		if (ww < 1e-12)
		{
			return;
		}
		const f64 pa = zw / ww;
		const f64 pb = game[3][2] - pa * game[3][3];
		// The palette is read through the index register, not directly: look it up in the whole bank.
		const auto* const direct_ids = std::exchange(t_direct_slots.ids, nullptr);
		for (u32 base = profile.camera_palette_first; base <= profile.camera_palette_last; base += 4)
		{
			matrix_block bone;
			if (!bone.bind(buffer, reloc, reloc_size, base, profile.column_vectors, false))
			{
				continue;
			}
			f64 len = 0.0, residual = 0.0;
			for (u32 r = 0; r < 3; ++r)
			{
				len += f64{bone.rows[r][3]} * bone.rows[r][3];
				residual = std::max(residual, std::fabs(bone.rows[r][2] - pa * bone.rows[r][3]));
			}
			len = std::sqrt(len);
			if (len < 1e-6 || residual > 1e-3 * len ||
				std::fabs(bone.rows[3][2] - pa * bone.rows[3][3] - pb) > 1e-3 * std::max(1.0, std::fabs(pb)))
			{
				continue;
			}
			f64 m[4][4];
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 c = 0; c < 4; ++c)
				{
					f64 sum = 0.0;
					for (u32 k = 0; k < 4; ++k)
						sum += bone.rows[r][k] * x[k][c];
					m[r][c] = sum;
				}
			}
			for (u32 r = 0; r < 4; ++r)
			{
				for (u32 c = 0; c < 4; ++c)
					bone.rows[r][c] = static_cast<f32>(m[r][c]);
			}
		}
		t_direct_slots.ids = direct_ids;
	}

	void camera_probe::set_vr_view(const f32 q[4], const f32 pos[3], f32 eye_scale,
		f32 fov_scale, bool flip_y, f32 ipd, f32 camera_depth)
	{
		// OpenXR rotation matrix (right, up, back basis) from the quaternion.
		const f32 x = q[0], y = q[1], z = q[2], w = q[3];
		const f32 r[3][3] =
			{
				{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
				{2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
				{2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)},
			};

		// The draw's clip basis is (NDC x = right, NDC y = up unless flip_y,
		// clip w = forward). Conjugate by S = diag(1, sy, -1), which is its own
		// inverse, and transpose: a camera turned by R sees view vectors by R^T.
		const title_profile* profile = this->profile();
		if (profile && profile->view_y_down)
		{
			flip_y = !flip_y;
		}
		const f32 s[3] = {1.f, flip_y ? -1.f : 1.f, -1.f};
		for (u32 i = 0; i < 3; ++i)
		{
			for (u32 j = 0; j < 3; ++j)
			{
				m_vr_rot[i * 3 + j] = s[j] * r[j][i] * s[i];
			}
		}

		// Head translation, in the same clip basis as the rotation. The game's own
		// eye separation (the profile's eye_baseline, in world units) stands for ipd
		// metres of real separation, so that ratio is the world scale, and the
		// eye_scale knob scales both together.
		const f32 eye_baseline = profile ? profile->eye_baseline : 0.f;
		const f32 units_per_metre = ipd > 0.01f ? eye_baseline * eye_scale / ipd : 0.f;
		for (u32 i = 0; i < 3; ++i)
		{
			// + camera_depth is forward, which is -Z in LOCAL space and +forward here.
			m_vr_head_m[i] = s[i] * pos[i] + (i == 2 ? camera_depth : 0.f);
			m_vr_head_units[i] = m_vr_head_m[i] * units_per_metre;
		}

		m_vr_eye_scale = eye_scale;
		m_vr_fov_scale = fov_scale;
		m_vr_flip_y = flip_y;
		m_vr_view = true;
	}

	void camera_probe::set_vr_eye_fov(const f32 (*tangents)[4], const f32 (*visible)[4], f32 hud_scale, bool hud_fixed,
		f32 hud_offset_x, f32 hud_offset_y, f32 hud_depth, f32 ipd)
	{
		m_vr_hud_parallax = hud_depth > 0.f ? ipd / (2.f * hud_depth) : 0.f;
		if (static std::pair<f32, f32> s_logged{-1.f, -1.f}; std::fabs(s_logged.first - hud_depth) > 1e-3f || std::fabs(s_logged.second - ipd) > 1e-4f)
		{
			s_logged = {hud_depth, ipd};
			vr_probe_log.notice("VR: HUD box at %.2f m, IPD %.1f mm (HUD parallax %.4f per eye).", hud_depth, ipd * 1000.f, m_vr_hud_parallax);
		}
		m_vr_hud_depth = hud_depth;
		// Only a usable frustum counts: the HUD box divides by its width and height. Boxing from the first
		// frame (not after the game's first camera draw) reached frames drawn before the headset had reported
		// its views, and the non-finite box scissor that followed lost the Vulkan device at boot.
		const auto usable = [](const f32(*fov)[4])
		{
			for (u32 e = 0; e < 2; ++e)
			{
				const f32* t = fov[e];
				if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2]) || !std::isfinite(t[3]) ||
					t[1] - t[0] < 0.01f || t[2] - t[3] < 0.01f || -t[0] <= 0.f || t[1] <= 0.f || t[2] <= 0.f || -t[3] <= 0.f)
				{
					return false;
				}
			}
			return true;
		};
		m_vr_hmd_fov = tangents != nullptr && usable(tangents) && (!visible || usable(visible));
		m_vr_hud_scale = hud_scale;
		m_vr_hud_fixed = hud_fixed;
		m_vr_hud_offset_x = hud_offset_x;
		m_vr_hud_offset_y = hud_offset_y;
		if (m_vr_hmd_fov)
		{
			for (u32 e = 0; e < 2; ++e)
			{
				for (u32 i = 0; i < 4; ++i)
				{
					m_vr_eye_fov[e][i] = tangents[e][i];
					m_vr_eye_fov_visible[e][i] = visible ? visible[e][i] : tangents[e][i];
				}
			}
		}
	}

	void camera_probe::set_screen_stereo_scale(f32 scale)
	{
		m_screen_stereo_scale = scale;
	}

	void camera_probe::clear_vr_view()
	{
		m_vr_view = false;
	}

	bool camera_probe::get_vr_fov(f32& tan_half_x, f32& tan_half_y) const
	{
		if (!m_vr_proj_valid)
		{
			return false;
		}
		tan_half_x = m_vr_fov_scale / m_vr_proj_x;
		tan_half_y = m_vr_fov_scale / m_vr_proj_y;
		return true;
	}

	void camera_probe::apply_vr_screen_space(const title_profile& profile, void* buffer, const u16* reloc, usz reloc_size,
		u16 surface_w, u16 surface_h, f32 eye_sign) const
	{
		// With the headset FOV, the eye image spans far more than the game's
		// frustum. Screen-space draws (HUD, menus) must not stretch with it or
		// leave the view, so they are mapped into a fixed box inside the headset
		// frustum. Classifier: an output-aspect draw whose profile screen-space
		// block is orthographic. In WipEout's Gate 3 capture every HUD draw reads
		// c[256..259] as an orthographic pixel matrix, while every post-process
		// pass (bloom chain, full-screen composite) reads no c[256..259] at all -
		// so post-processing is never touched.
		// The box needs only the headset view, not a game camera: requiring one (m_vr_proj_valid) left
		// menus full-view until the game first drew 3D, and boxed after, so Killzone HD's main menu
		// looked different from run to run depending on what had been drawn before it.
		// The HUD block: the profile's, or a listed program's own (hud_block_programs).
		u32 hud_block = profile.screen_space_block;
		for (const auto& entry : profile.screen_space_hud_block_programs)
		{
			if (entry.program == m_draw_program)
			{
				hud_block = entry.block;
				break;
			}
		}
		if (!m_vr_view || !m_vr_hmd_fov || hud_block == umax ||
			(profile.screen_space_hud_skips_passes && m_draw_samples_colour_target) ||
			(profile.screen_space_hud_display_buffers_only && !m_draw_into_display_buffer))
		{
			return;
		}

		const auto& avconf = g_fxo->get<rsx::avconf>();
		const size2u eye = avconf.video_frame_size();
		// The HUD may be drawn at the output's aspect while the scene renders at another one
		// (Anarchy Reigns: scene 1024x720, HUD on the 1280x720 targets after the upscale).
		const bool output_target = surface_w && surface_h && eye.width && eye.height &&
		                           std::fabs((static_cast<f32>(surface_w) / surface_h) / (static_cast<f32>(eye.width) / eye.height) - 1.f) <= profile.output_aspect_tolerance;
		if (!surface_w || !surface_h || !eye.width || !eye.height ||
			!(output_target || profile.is_view_target(surface_w, surface_h, static_cast<f32>(eye.width) / eye.height)))
		{
			return;
		}

		matrix_block block;
		const bool hud_rows = profile.screen_space_block_rows ||
		                      std::find(profile.row_vector_blocks.begin(), profile.row_vector_blocks.end(), hud_block) != profile.row_vector_blocks.end();
		if (!block.bind(buffer, reloc, reloc_size, hud_block, profile.column_vectors && !hud_rows, false, nullptr, true))
		{
			return;
		}

		if (is_perspective(block.rows))
		{
			block.release();
			return;
		}

		if (profile.screen_space_output_pixel_draws_not_hud &&
			std::fabs(std::fabs(block.rows[0][0]) * eye.width * 0.5f - 1.f) < 1e-3f &&
			std::fabs(std::fabs(block.rows[1][1]) * eye.height * 0.5f - 1.f) < 1e-3f)
		{
			block.release();
			return;
		}

		if (profile.screen_space_hud_box_after_shader)
		{
			block.release();
			m_hud_env_request = true;
			return;
		}

		if (m_draw_hud_scale != 1.f)
		{
			// Resize about the game screen's centre (NDC origin): scale clip x and y.
			for (u32 r = 0; r < 4; ++r)
			{
				block.rows[r][0] *= m_draw_hud_scale;
				block.rows[r][1] *= m_draw_hud_scale;
			}
		}
		map_vr_screen_box(block.rows, eye_sign, static_cast<f32>(eye.width) / eye.height);
	}

	bool camera_probe::map_box_scissor(f32 host_scale_x, f32 host_scale_y, f32 host_width, f32 host_height, f32 rect[4], f32 (*corners)[2]) const
	{
		// rect: x1, y1, x2, y2 in host pixels, replaced by the bounds of its image in the box.
		const f32 vsx = rsx::method_registers.viewport_scale_x(), vsy = rsx::method_registers.viewport_scale_y();
		const f32 vox = rsx::method_registers.viewport_offset_x(), voy = rsx::method_registers.viewport_offset_y();
		if (!m_box_mapped || std::fabs(vsx) < 1e-6f || std::fabs(vsy) < 1e-6f || host_scale_x <= 0.f || host_scale_y <= 0.f)
		{
			return false;
		}

		f32 out[4] = {host_width, host_height, 0.f, 0.f};
		for (u32 corner = 0; corner < 4; ++corner)
		{
			// Host pixel -> guest window -> game NDC (inverse viewport), through the box, and back.
			const f32 gx = rect[(corner & 1) ? 2 : 0] / host_scale_x;
			const f32 gy = rect[(corner & 2) ? 3 : 1] / host_scale_y;
			const f32 p[4] = {(gx - vox) / vsx, (gy - voy) / vsy, 0.f, 1.f};
			f32 o[4] = {};
			for (u32 c = 0; c < 4; ++c)
			{
				for (u32 r = 0; r < 4; ++r)
				{
					o[c] += p[r] * m_box_map[r][c];
				}
			}
			if (o[3] <= 1e-4f)
			{
				// Part of the box is behind the viewer: leave the game's scissor.
				return false;
			}
			const f32 x = (o[0] / o[3] * vsx + vox) * host_scale_x;
			const f32 y = (o[1] / o[3] * vsy + voy) * host_scale_y;
			if (corners)
			{
				corners[corner][0] = x;
				corners[corner][1] = y;
			}
			out[0] = std::min(out[0], x);
			out[1] = std::min(out[1], y);
			out[2] = std::max(out[2], x);
			out[3] = std::max(out[3], y);
		}
		if (!std::isfinite(out[0]) || !std::isfinite(out[1]) || !std::isfinite(out[2]) || !std::isfinite(out[3]))
		{
			return false;
		}
		rect[0] = std::clamp(out[0], 0.f, host_width);
		rect[1] = std::clamp(out[1], 0.f, host_height);
		rect[2] = std::clamp(out[2], rect[0], host_width);
		rect[3] = std::clamp(out[3], rect[1], host_height);
		return true;
	}

	void camera_probe::map_vr_screen_box(f32* const rows[4], f32 eye_sign, f32 aspect) const
	{
		if (!m_box_identity_pass)
		{
			// The mapping is the same linear map on every row: record it (from identity rows)
			// for map_box_scissor.
			f32 id[4][4] = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
			f32* const id_rows[4] = {id[0], id[1], id[2], id[3]};
			m_box_identity_pass = true;
			map_vr_screen_box(id_rows, eye_sign, aspect);
			m_box_identity_pass = false;
			std::memcpy(m_box_map, id, sizeof(id));
			m_box_mapped = true;
		}

		// Applied after the shader, the box works on the whole shown target: a draw through part of it
		// (a sub-viewport) is first put where its viewport places it.
		if (m_hud_env_request)
		{
			undo_viewport(rows, true, true);
		}

		// The game camera's FOV changes with speed and camera mode (and can exceed
		// the headset's), so the HUD is not tied to it. It becomes a fixed box with
		// the output aspect, fitted inside the central symmetric part of this
		// eye's headset frustum and scaled by the HUD scale. The mapping is linear
		// in (X, Y, W), so it also carries perspective blocks (W != 1) whose game
		// NDC image belongs to the screen, e.g. the menu background.
		// Sized from the visible frustum, mapped into the rendered one (wider by the
		// reprojection margin).
		const f32* t = m_vr_eye_fov[eye_sign < 0.f ? 0 : 1];
		const f32* v = m_vr_eye_fov_visible[eye_sign < 0.f ? 0 : 1];
		f32 fit_x = std::min(-v[0], v[1]);
		f32 fit_y = std::min(v[2], -v[3]);
		if (m_vr_hud_fixed)
		{
			// One box for both eyes, so the fixed HUD carries no stray disparity.
			const f32* o = m_vr_eye_fov_visible[eye_sign < 0.f ? 1 : 0];
			fit_x = std::min({fit_x, -o[0], o[1]});
			fit_y = std::min({fit_y, o[2], -o[3]});
		}
		const f32 box_y = std::min(fit_y, fit_x / aspect);
		const f32 tx = m_vr_hud_scale * box_y * aspect;
		const f32 ty = m_vr_hud_scale * box_y;
		// Box centre, in view tangents (y in the clip basis, which is down when flip_y).
		// At a finite depth each eye sees the HUD shifted away from its own side:
		// the eye sits at eye_sign * ipd/2, so the point appears at -eye_sign * ipd/(2d).
		const f32 parallax = -eye_sign * m_vr_hud_parallax;
		const f32 cx = m_vr_hud_offset_x * box_y * aspect;
		const f32 cy = m_vr_hud_offset_y * box_y * (m_vr_flip_y ? -1.f : 1.f);

		// Eye frustum remap of a view direction (tan x, tan y, 1) to clip space.
		const f32 fx = 2.f / (t[1] - t[0]);
		const f32 ox = -(t[1] + t[0]) / (t[1] - t[0]);
		const f32 fy = 2.f / (t[2] - t[3]);
		const f32 oy = -(t[2] + t[3]) / (t[2] - t[3]) * (m_vr_flip_y ? -1.f : 1.f);

		if (!m_vr_hud_fixed)
		{
			// Head-locked: the box stays at the centre of the view.
			for (u32 r = 0; r < 4; ++r)
			{
				rows[r][0] = (rows[r][0] * tx + rows[r][3] * (cx + parallax)) * fx + rows[r][3] * ox;
				rows[r][1] = (rows[r][1] * ty + rows[r][3] * cy) * fy + rows[r][3] * oy;
			}
			undo_viewport(rows, true);
			return;
		}

		// Fixed in front: the box is a direction in LOCAL space (straight ahead),
		// seen through the head rotation this frame is rendered with, exactly as
		// the world is. u = (tan x, tan y, 1) in the clip basis, u' = R^T u.
		// Rotating changes W. Keep Z/W on the same depth as for a camera draw
		// (Z' = Z + (c/e)(W' - W), see apply_vr_rotation); c/e is 0 for the
		// orthographic HUD, whose column 3 is (0, 0, 0, 1).
		f32 k = 0.f;
		if (const f32 d33 = rows[0][3] * rows[0][3] + rows[1][3] * rows[1][3] + rows[2][3] * rows[2][3]; d33 > 1e-12f)
		{
			k = (rows[0][2] * rows[0][3] + rows[1][2] * rows[1][3] + rows[2][2] * rows[2][3]) / d33;
		}

		// The box hangs at a known distance, so the head's own position moves across
		// it: a point at distance d is (u * d - head), divided through by d to keep
		// w on the scale the depth test expects. At infinity it is a pure direction.
		const f32 lean = m_vr_hud_depth > 0.f ? 1.f / m_vr_hud_depth : 0.f;

		const auto& R = m_vr_rot;
		for (u32 r = 0; r < 4; ++r)
		{
			const f32 w = rows[r][3];
			const f32 u0 = rows[r][0] * tx + w * (cx - lean * m_vr_head_m[0]);
			const f32 u1 = rows[r][1] * ty + w * (cy - lean * m_vr_head_m[1]);
			const f32 u2 = w * (1.f - lean * m_vr_head_m[2]);

			// The eye offset is along the head's own right axis, i.e. after rotation.
			const f32 v0 = R[0] * u0 + R[1] * u1 + R[2] * u2 + parallax * u2;
			const f32 v1 = R[3] * u0 + R[4] * u1 + R[5] * u2;
			const f32 v2 = R[6] * u0 + R[7] * u1 + R[8] * u2;

			rows[r][2] += k * (v2 - rows[r][3]);
			rows[r][0] = v0 * fx + v2 * ox;
			rows[r][1] = v1 * fy + v2 * oy;
			rows[r][3] = v2;
			// Without depth test z only clips: a HUD on the far plane (Demon's Souls: z = w = 1)
			// left the depth range as W changed and vanished. Mid-range instead.
			if (!m_draw_depth_test)
			{
				rows[r][2] = 0.5f * v2;
			}
		}
		undo_viewport(rows, true);
	}

	void camera_probe::remap_to_eye_fov(f32* const rows[4], const f32* t, f32 A, f32 B, bool keep_viewport) const
	{
		// Re-project from the game frustum onto this eye's headset frustum.
		// Game NDC x = A * (x/f); headset NDC x = (2*(x/f) - (r+l)) / (r-l).
		// Both are linear in clip space: X' = X*sx + W*ox (likewise Y).
		const f32 sx = 2.f / (A * (t[1] - t[0]));
		const f32 ox = -(t[1] + t[0]) / (t[1] - t[0]);
		const f32 sy = 2.f / (B * (t[2] - t[3]));
		const f32 oy = -(t[2] + t[3]) / (t[2] - t[3]) * (m_vr_flip_y ? -1.f : 1.f);
		for (u32 r = 0; r < 4; ++r)
		{
			rows[r][0] = rows[r][0] * sx + rows[r][3] * ox;
			rows[r][1] = rows[r][1] * sy + rows[r][3] * oy;
		}
		if (!keep_viewport)
		{
			undo_viewport(rows);
		}
	}

	bool camera_probe::viewport_inside_shown_region() const
	{
		// The shown part of the target: Gran Turismo 5 draws its menus and HUD through a 1280x720 viewport into
		// 2048x1080 buffers, so against the whole buffer that viewport looked like a sub-viewport too: the menus'
		// full-screen colour and depth clears went into the HUD box like the mirror's. Turned with the head, the boxed
		// depth clear no longer covered the menu cards, and each card failed its depth test against its own depth of
		// the frame before: the cards were cut along a line (Matt, headset, 2026-10-03).
		const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
		const f32 shown_w = out.width ? std::min<f32>(rsx::method_registers.surface_clip_width(), static_cast<f32>(out.width)) : rsx::method_registers.surface_clip_width();
		const f32 shown_h = out.height ? std::min<f32>(rsx::method_registers.surface_clip_height(), static_cast<f32>(out.height)) : rsx::method_registers.surface_clip_height();
		return std::fabs(rsx::method_registers.viewport_scale_x()) * 2.f < shown_w * 0.9f &&
		       std::fabs(rsx::method_registers.viewport_scale_y()) * 2.f < shown_h * 0.9f;
	}

	bool camera_probe::map_subviewport_clear(f32 host_scale_x, f32 host_scale_y, u32 surface_w, u32 surface_h, f32 host_width, f32 host_height, f32 rect[4], f32 right_rect[4], f32 (*quads)[4][2]) const
	{
		const title_profile* p = profile();
		const size2u shown = g_fxo->get<rsx::avconf>().video_frame_size();
		if (!p || !p->screen_space_subviewport_cameras_in_box || !p->screen_space_hud_box_after_shader ||
			!m_vr_view || !m_vr_hmd_fov || !m_vr_proj_valid || !shown.height ||
			!p->is_view_target(surface_w, surface_h, static_cast<f32>(shown.width) / shown.height) ||
			!viewport_inside_shown_region())
		{
			return false;
		}
		// Each eye's box differs by the HUD parallax; a clear also writes depth, so any cleared area
		// the eye's own mirror does not cover would stay black.
		const bool request = std::exchange(m_hud_env_request, true);
		f32 box[4][4];
		std::copy(rect, rect + 4, right_rect);
		const f32 aspect = static_cast<f32>(shown.width) / shown.height;
		const bool mapped = map_vr_passthrough_hud(box, 1.f, aspect) && map_box_scissor(host_scale_x, host_scale_y, host_width, host_height, right_rect, quads ? quads[1] : nullptr) &&
		                    map_vr_passthrough_hud(box, -1.f, aspect) && map_box_scissor(host_scale_x, host_scale_y, host_width, host_height, rect, quads ? quads[0] : nullptr);
		m_hud_env_request = request;
		return mapped;
	}

	bool camera_probe::map_vr_passthrough_hud(f32 m[4][4], f32 eye_sign, f32 aspect) const
	{
		const title_profile* p = profile();
		if (!p || !(p->screen_space_passthrough_hud || m_hud_env_request) || !m_vr_view || !m_vr_hmd_fov)
		{
			return false;
		}
		f32 r0[4] = {1.f, 0.f, 0.f, 0.f};
		f32 r1[4] = {0.f, 1.f, 0.f, 0.f};
		f32 r2[4] = {0.f, 0.f, 1.f, 0.f};
		f32 r3[4] = {0.f, 0.f, 0.f, 1.f};
		f32* const rows[4] = {r0, r1, r2, r3};
		map_vr_screen_box(rows, eye_sign, aspect);
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 col = 0; col < 4; ++col)
			{
				m[r][col] = rows[r][col];
			}
		}
		return true;
	}

	void camera_probe::store_eye_block(f32 eye_sign, const f32 (&game)[4][4], f32* const rows[4]) const
	{
		const u32 eye = eye_sign < 0.f ? 0 : 1;
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				m_vr_last_block[eye][r][c] = game[r][c];
				m_vr_last_eye_block[eye][r][c] = rows[r][c];
			}
		}
		m_vr_last_block_valid[eye] = true;
	}

	void camera_probe::store_depth_remap(const f32 (&game)[4][4], f32* const rows[4]) const
	{
		// Row vectors (clip = v * M): the game's clip position of the point an eye drew at clip e is
		// e * T, T = eye^-1 * game. The depth buffer holds window depth d = z / w * sz + oz (the viewport's
		// z scale and offset), so in (NDC x, NDC y, d, 1) the map is V^-1 * T * V, V: z' = z * sz + w * oz.
		const f64 sz = rsx::method_registers.viewport_scale_z();
		const f64 oz = rsx::method_registers.viewport_offset_z();
		if (std::fabs(sz) < 1e-9)
		{
			return;
		}
		f64 a[4][8];
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				a[r][c] = rows[r][c];
				a[r][c + 4] = r == c ? 1.0 : 0.0;
			}
		}
		for (u32 c = 0; c < 4; ++c)
		{
			u32 pivot = c;
			for (u32 r = c + 1; r < 4; ++r)
			{
				if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
					pivot = r;
			}
			if (std::fabs(a[pivot][c]) < 1e-12)
			{
				return;
			}
			if (pivot != c)
			{
				for (u32 k = 0; k < 8; ++k)
					std::swap(a[c][k], a[pivot][k]);
			}
			const f64 inv = 1.0 / a[c][c];
			for (u32 k = 0; k < 8; ++k)
				a[c][k] *= inv;
			for (u32 r = 0; r < 4; ++r)
			{
				if (r == c || a[r][c] == 0.0)
					continue;
				const f64 f = a[r][c];
				for (u32 k = 0; k < 8; ++k)
					a[r][k] -= f * a[c][k];
			}
		}
		f64 t[4][4];
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				f64 sum = 0.0;
				for (u32 k = 0; k < 4; ++k)
					sum += a[r][k + 4] * game[k][c];
				t[r][c] = sum;
			}
		}
		// V^-1 * T: row 2 / sz, row 3 - oz / sz * row 2; then * V on each row: z' = z * sz + w * oz.
		for (u32 c = 0; c < 4; ++c)
		{
			t[3][c] -= oz / sz * t[2][c];
			t[2][c] /= sz;
		}
		for (u32 r = 0; r < 4; ++r)
		{
			m_depth_remap[r][0] = static_cast<f32>(t[r][0]);
			m_depth_remap[r][1] = static_cast<f32>(t[r][1]);
			m_depth_remap[r][2] = static_cast<f32>(t[r][2] * sz + t[r][3] * oz);
			m_depth_remap[r][3] = static_cast<f32>(t[r][3]);
		}
		m_depth_remap_valid = true;
	}

	bool camera_probe::game_projection_scale(f32& x, f32& y) const
	{
		if (m_scene_proj_x > 0.f && m_scene_proj_y > 0.f)
		{
			x = m_scene_proj_x;
			y = m_scene_proj_y;
			return true;
		}
		if (!m_vr_proj_valid || m_vr_proj_x <= 0.f || m_vr_proj_y <= 0.f)
		{
			return false;
		}
		x = m_vr_proj_x;
		y = m_vr_proj_y;
		return true;
	}

	bool camera_probe::depth_remap_matrix(f32 (&out)[4][4]) const
	{
		if (!m_depth_remap_valid)
		{
			return false;
		}
		std::memcpy(out, m_depth_remap, sizeof(out));
		return true;
	}

	bool camera_probe::map_vr_preprojected(f32 m[4][4], f32 eye_sign, u64 program_hash) const
	{
		const title_profile* p = profile();
		const u32 eye = eye_sign < 0.f ? 0 : 1;
		if (!p || !m_vr_last_block_valid[eye] || (program_hash == scene_draw_program ? !scene_draws_by_clip_space() : std::find(p->screen_space_preprojected_programs.begin(), p->screen_space_preprojected_programs.end(), program_hash) == p->screen_space_preprojected_programs.end()))
		{
			return false;
		}

		// Dev (probe why=<this program>): the camera blocks it is mapped through, a few times.
		if (static u32 s_logged = 0; m_why_program == program_hash && s_logged < 4)
		{
			s_logged++;
			const auto& g = m_vr_last_block[eye];
			const auto& e = m_vr_last_eye_block[eye];
			vr_probe_log.notice("VR why %016llx eye %u game block (%g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g) eye block (%g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g)",
				program_hash, eye, g[0][0], g[0][1], g[0][2], g[0][3], g[1][0], g[1][1], g[1][2], g[1][3], g[2][0], g[2][1], g[2][2], g[2][3], g[3][0], g[3][1], g[3][2], g[3][3],
				e[0][0], e[0][1], e[0][2], e[0][3], e[1][0], e[1][1], e[1][2], e[1][3], e[2][0], e[2][1], e[2][2], e[2][3], e[3][0], e[3][1], e[3][2], e[3][3]);
		}

		// A pre-projected vertex c is a point in the game's clip space: c * B^-1 is that
		// point (homogeneous, in the space B was applied to), and * B_eye draws it for
		// this eye exactly as the camera draws were, eye offset and head position
		// included. Inverse by Gauss-Jordan with partial pivoting.
		f64 a[4][8];
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				a[r][c] = m_vr_last_block[eye][r][c];
				a[r][c + 4] = r == c ? 1.0 : 0.0;
			}
		}
		for (u32 c = 0; c < 4; ++c)
		{
			u32 pivot = c;
			for (u32 r = c + 1; r < 4; ++r)
			{
				if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
					pivot = r;
			}
			if (std::fabs(a[pivot][c]) < 1e-12)
			{
				return false;
			}
			if (pivot != c)
			{
				for (u32 k = 0; k < 8; ++k)
					std::swap(a[c][k], a[pivot][k]);
			}
			const f64 inv = 1.0 / a[c][c];
			for (u32 k = 0; k < 8; ++k)
				a[c][k] *= inv;
			for (u32 r = 0; r < 4; ++r)
			{
				if (r == c || a[r][c] == 0.0)
					continue;
				const f64 f = a[r][c];
				for (u32 k = 0; k < 8; ++k)
					a[r][k] -= f * a[c][k];
			}
		}
		for (u32 r = 0; r < 4; ++r)
		{
			for (u32 c = 0; c < 4; ++c)
			{
				f64 sum = 0.0;
				for (u32 k = 0; k < 4; ++k)
				{
					sum += a[r][k + 4] * m_vr_last_eye_block[eye][k][c];
				}
				m[r][c] = static_cast<f32>(sum);
			}
		}
		return true;
	}

	void camera_probe::undo_viewport(f32* const rows[4], bool displayed_region, bool inverse) const
	{
		// Target NDC = NDC * k + o, with k = scale / (clip size / 2) and o the offset from
		// the target centre; a viewport covering the target exactly is k = +-1, o = 0.
		// The mapping above wants target NDC = its NDC (in the viewport's own y sense),
		// so X' = (X - o*W) / k.
		f32 half_w = rsx::method_registers.surface_clip_width() / 2.f;
		f32 half_h = rsx::method_registers.surface_clip_height() / 2.f;
		if (displayed_region)
		{
			// The HUD box belongs to the part of the target that is shown. Gran Turismo 5 draws its
			// HUD through a 1280x720 viewport into a 2048x1080 buffer and displays only that corner;
			// measured against the whole buffer the box came out 1.6x too large, off to the lower right.
			const size2u out = g_fxo->get<rsx::avconf>().video_frame_size();
			const f32 vx0 = rsx::method_registers.viewport_offset_x() - std::fabs(rsx::method_registers.viewport_scale_x());
			const f32 vy0 = rsx::method_registers.viewport_offset_y() - std::fabs(rsx::method_registers.viewport_scale_y());
			const f32 vx1 = rsx::method_registers.viewport_offset_x() + std::fabs(rsx::method_registers.viewport_scale_x());
			const f32 vy1 = rsx::method_registers.viewport_offset_y() + std::fabs(rsx::method_registers.viewport_scale_y());
			if (out.width && out.height && (half_w * 2.f > out.width || half_h * 2.f > out.height) &&
				vx0 > -0.5f && vy0 > -0.5f && vx1 < out.width + 0.5f && vy1 < out.height + 0.5f)
			{
				half_w = std::min<f32>(half_w, out.width / 2.f);
				half_h = std::min<f32>(half_h, out.height / 2.f);
			}
		}
		if (half_w <= 0.f || half_h <= 0.f)
		{
			return;
		}
		const f32 kx = rsx::method_registers.viewport_scale_x() / half_w;
		const f32 ky = rsx::method_registers.viewport_scale_y() / half_h;
		const f32 ox = (rsx::method_registers.viewport_offset_x() - half_w) / half_w;
		const f32 oy = (rsx::method_registers.viewport_offset_y() - half_h) / half_h;
		const f32 ax = std::fabs(kx), ay = std::fabs(ky);
		{
			// Diagnostic: each distinct viewport scale seen by a remapped draw (first 16).
			static std::set<u64> s_seen;
			const u64 key = (static_cast<u64>(std::lround(ax * 1000.f)) << 32) | static_cast<u32>(std::lround(ay * 1000.f));
			if (s_seen.size() < 16 && s_seen.insert(key).second)
			{
				vr_probe_log.notice("VR viewport: scale %.4f x %.4f, offset %.4f, %.4f (clip %.0fx%.0f)", kx, ky, ox, oy, half_w * 2.f, half_h * 2.f);
			}
		}
		// A box applied after the shader also takes small sub-viewports (a rear-view mirror).
		const f32 min_scale = m_hud_env_request ? 0.01f : 0.25f;
		if (ax < min_scale || ay < min_scale || ax > 4.f || ay > 4.f ||
			(std::fabs(ax - 1.f) < 1e-3f && std::fabs(ay - 1.f) < 1e-3f && std::fabs(ox) < 1e-3f && std::fabs(oy) < 1e-3f))
		{
			return;
		}
		// In the viewport's own orientation: divide by |k|; the offset in NDC units of
		// that orientation is o / sign(k).
		const f32 sox = ox / (kx < 0.f ? -1.f : 1.f);
		const f32 soy = oy / (ky < 0.f ? -1.f : 1.f);
		for (u32 r = 0; r < 4; ++r)
		{
			if (inverse)
			{
				rows[r][0] = rows[r][0] * ax + sox * rows[r][3];
				rows[r][1] = rows[r][1] * ay + soy * rows[r][3];
				continue;
			}
			rows[r][0] = (rows[r][0] - sox * rows[r][3]) / ax;
			rows[r][1] = (rows[r][1] - soy * rows[r][3]) / ay;
		}

		static bool s_reported = false;
		if (!std::exchange(s_reported, true))
		{
			vr_probe_log.success("VR: camera draws use a viewport %.4fx%.4f the render target's (offset %.4f, %.4f); the headset mapping compensates.",
				ax, ay, sox, soy);
		}
	}

	void camera_probe::apply_vr_rotation(f32* const rows[4], const std::array<f32, 9>& R, const std::array<f32, 3>& head, bool refresh_projection) const
	{
		// Row-vector camera block M (clip = v * M). For M = L * P with L affine and
		// P a perspective projection (x' = a*x, y' = b*y, z' = c*z + d, w' = e*z):
		//   col0 = a*L.col0, col1 = b*L.col1, col2 = c*L.col2 (+ d in row 3), col3 = e*L.col2
		// so for rows 0..2, c/e = col2.col3 / col3.col3 exactly, for any affine L.
		// When L is rigid (orthogonal columns, uniform scale), |col0|/|col3| = a/|e|
		// and |col1|/|col3| = b/|e|; those are cached from such blocks and reused
		// for non-rigid ones. The view rotation is then applied in clip space:
		//   u = (X/A, Y/B, W),  u' = R^T u,  X' = A*u'x,  Y' = B*u'y,  W' = u'z,
		//   Z' = Z + (c/e) * (W' - W)
		const auto dot = [&](u32 i, u32 j)
		{
			return rows[0][i] * rows[0][j] + rows[1][i] * rows[1][j] + rows[2][i] * rows[2][j];
		};

		const f32 n0 = std::sqrt(dot(0, 0));
		const f32 n1 = std::sqrt(dot(1, 1));
		const f32 d33 = dot(3, 3);
		const f32 n3 = std::sqrt(d33);
		if (n0 < 1e-8f || n1 < 1e-8f || n3 < 1e-8f)
		{
			return;
		}

		constexpr f32 tol = 1e-3f;
		if (refresh_projection &&
			std::fabs(dot(0, 1)) <= tol * n0 * n1 &&
			std::fabs(dot(0, 3)) <= tol * n0 * n3 &&
			std::fabs(dot(1, 3)) <= tol * n1 * n3)
		{
			m_vr_proj_x = n0 / n3;
			m_vr_proj_y = n1 / n3;
			m_vr_proj_valid = true;
			m_proj_refreshed = true;
		}

		if (!m_vr_proj_valid)
		{
			return;
		}

		const f32 k = dot(2, 3) / d33;
		const f32 A = m_vr_proj_x;
		const f32 B = m_vr_proj_y;

		for (u32 r = 0; r < 4; ++r)
		{
			// u is the view vector in the clip basis. Rows 0..2 carry the object's
			// x/y/z, which a camera move does not touch; row 3 is the point term,
			// so the head offset is subtracted there and there only.
			const f32 t = r == 3 ? 1.f : 0.f;
			const f32 u0 = rows[r][0] / A - t * head[0];
			const f32 u1 = rows[r][1] / B - t * head[1];
			const f32 u2 = rows[r][3] - t * head[2];

			const f32 v0 = R[0] * u0 + R[1] * u1 + R[2] * u2;
			const f32 v1 = R[3] * u0 + R[4] * u1 + R[5] * u2;
			const f32 v2 = R[6] * u0 + R[7] * u1 + R[8] * u2;

			rows[r][2] += k * (v2 - rows[r][3]);
			rows[r][0] = A * v0;
			rows[r][1] = B * v1;
			rows[r][3] = v2;
		}
	}

	void camera_probe::apply(void* buffer, const u16* reloc, usz reloc_size, const void* guest_constants, u16 surface_w, u16 surface_h) const
	{
		if (!m_active.load() || !buffer)
		{
			return;
		}

		// Title gate (title=). Unset, experiments run on whatever is booted, so an
		// unprofiled game can be investigated with explicit base= and cam=.
		if (!m_title.empty() && Emu.GetTitleID() != m_title)
		{
			return;
		}

		// --- raw single-component probe (positive / negative controls) -------
		if (m_have_raw)
		{
			if (f32* slot = find_slot(buffer, reloc, reloc_size, m_raw_slot);
				slot && m_raw_comp < 4)
			{
				slot[m_raw_comp] += m_raw_add;
			}
		}

		if (!m_have_xform && !m_have_stereo)
		{
			return;
		}

		// --- matrix probe ----------------------------------------------------
		// Slots come from base=/cam= or else the title's profile.
		const title_profile* profile = this->profile();
		const u32 base_override[2] = {m_base, m_base + 4};
		const std::span<const u32> camera_blocks = m_base != umax ? std::span<const u32>(base_override) : profile ? std::span<const u32>(profile->camera_blocks) :
		                                                                                                            std::span<const u32>();
		const u32 cam_slot = m_cam_slot != umax ? m_cam_slot : profile ? profile->camera_position_slot :
		                                                                 umax;
		// Probe default for unprofiled titles: RPCS3's own 2% output-aspect match.
		const f32 aspect_tolerance = profile ? profile->output_aspect_tolerance : 0.02f;
		// layout=columns selects the DP4 layout for an unprofiled game.
		const bool column_vectors = m_column_vectors || (profile && profile->column_vectors);
		if (camera_blocks.empty())
		{
			return;
		}

		if (m_require_cam && (cam_slot == umax || !find_slot(buffer, reloc, reloc_size, cam_slot)))
		{
			return;
		}

		// Classifier rule 1: only camera-view render targets, i.e. targets that share
		// the output aspect ratio. Off-aspect targets (512x256 cascades, 512x512 and
		// 256x256 maps) are eye-invariant in the native oracle.
		{
			const auto& avconf = g_fxo->get<rsx::avconf>();
			const size2u eye = avconf.video_frame_size();
			if (!surface_w || !surface_h || !eye.width || !eye.height)
			{
				return;
			}

			const f32 target_aspect = static_cast<f32>(surface_w) / surface_h;
			const f32 output_aspect = static_cast<f32>(eye.width) / eye.height;
			const f32 view_aspect = profile && profile->camera_target_aspect > 0.f ? profile->camera_target_aspect : output_aspect;
			if (std::fabs(target_aspect / view_aspect - 1.f) > aspect_tolerance)
			{
				if (find_slot(buffer, reloc, reloc_size, camera_blocks[0]))
				{
					m_stat_rejected_aspect++;
				}
				return;
			}
		}

		// Select the camera block: the first of {base, base+4} that is present in
		// this program and is a PERSPECTIVE matrix. Orthographic blocks (HUD,
		// shadow cascades, env maps) and affine world matrices are left alone.
		matrix_block block;
		f32* const* const rows = block.rows;
		if (!bind_camera_block(block, buffer, reloc, reloc_size, camera_blocks, column_vectors, profile && profile->require_rigid_camera,
				m_base == umax && profile && profile->xyw_rows,
				profile && profile->require_camera_aspect ? static_cast<f32>(g_fxo->get<rsx::avconf>().video_frame_size().width) / g_fxo->get<rsx::avconf>().video_frame_size().height : 0.f,
				m_base == umax && profile ? std::span<const std::array<u32, 4>>(profile->camera_block_slots) : std::span<const std::array<u32, 4>>(),
				m_base == umax && profile ? std::span<const u32>(profile->nonrigid_camera_blocks) : std::span<const u32>(),
				m_base == umax && profile ? std::span<const u32>(profile->row_vector_blocks) : std::span<const u32>(),
				m_base == umax && profile ? std::span<const u32>(profile->either_layout_blocks) : std::span<const u32>()))
		{
			if (find_slot(buffer, reloc, reloc_size, camera_blocks[0]))
			{
				m_stat_rejected_no_perspective++;
			}
			return;
		}

		m_stat_perturbed++;

		// The native stereo shear (WipEout's), applied in clip space:
		//   clip.x += sep * (clip.w - conv)
		// Column 0 of the row-vector matrix produces clip.x and column 3 produces
		// clip.w, so: col0 += sep * col3, then the constant term (row 3) -= sep*conv.
		if (m_have_stereo)
		{
			for (int r = 0; r < 4; ++r)
			{
				rows[r][0] += m_stereo_sep * rows[r][3];
			}
			rows[3][0] -= m_stereo_sep * m_stereo_conv;
		}

		if (!m_have_xform)
		{
			return;
		}

		mat4 M{};
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				M[i][j] = rows[i][j];
			}
		}

		// Camera pivot from the pristine guest bank (w must be 1 for a point).
		f32 cam[3] = {0.f, 0.f, 0.f};
		if (guest_constants && cam_slot < 512)
		{
			const u32* bank = static_cast<const u32*>(guest_constants);
			const u32* c = bank + cam_slot * 4;
			std::memcpy(&cam[0], &c[0], sizeof(f32));
			std::memcpy(&cam[1], &c[1], sizeof(f32));
			std::memcpy(&cam[2], &c[2], sizeof(f32));
		}

		// Translation offset, optionally including an eye shift along the camera
		// right axis derived from the matrix itself (column 0 of the upper 3x3).
		f32 tx = m_tx, ty = m_ty, tz = m_tz;
		if (m_eye != 0.f)
		{
			f32 right[3] = {M[0][0], M[1][0], M[2][0]};
			const f32 len = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
			if (len > 1e-8f)
			{
				tx += m_eye * right[0] / len;
				ty += m_eye * right[1] / len;
				tz += m_eye * right[2] / len;
			}
		}

		constexpr f32 deg2rad = 3.14159265358979323846f / 180.f;
		mat4 R = identity();
		if (m_pitch != 0.f)
			R = mul(R, rot_x(m_pitch * deg2rad));
		if (m_yaw != 0.f)
			R = mul(R, rot_y(m_yaw * deg2rad));
		if (m_roll != 0.f)
			R = mul(R, rot_z(m_roll * deg2rad));

		// Rotate about the camera pivot, then move the world opposite to the
		// requested camera translation.
		mat4 D = translate(-cam[0], -cam[1], -cam[2]);
		D = mul(D, R);
		D = mul(D, translate(cam[0], cam[1], cam[2]));
		D = mul(D, translate(-tx, -ty, -tz));

		const mat4 Mp = mul(D, M);

		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				rows[i][j] = Mp[i][j];
			}
		}
	}
} // namespace rsx::vr
