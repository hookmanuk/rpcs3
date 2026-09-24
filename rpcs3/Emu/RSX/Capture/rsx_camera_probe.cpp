#include "stdafx.h"
#include "rsx_camera_probe.h"
#include <set>

#include "Emu/System.h"
#include "Emu/Memory/vm.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "Emu/IdManager.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/rsx_methods.h"

#include "util/logs.hpp"
#include "util/yaml.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

LOG_CHANNEL(vr_probe_log, "VRPROBE");

namespace rsx::vr
{
	namespace
	{
		using mat4 = std::array<std::array<f32, 4>, 4>;

		mat4 identity()
		{
			mat4 m{};
			for (int i = 0; i < 4; ++i) m[i][i] = 1.f;
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
			m[3][0] = x; m[3][1] = y; m[3][2] = z;
			return m;
		}

		mat4 rot_x(f32 rad)
		{
			mat4 m = identity();
			const f32 c = std::cos(rad), s = std::sin(rad);
			m[1][1] = c;  m[1][2] = s;
			m[2][1] = -s; m[2][2] = c;
			return m;
		}

		mat4 rot_y(f32 rad)
		{
			mat4 m = identity();
			const f32 c = std::cos(rad), s = std::sin(rad);
			m[0][0] = c; m[0][2] = -s;
			m[2][0] = s; m[2][2] = c;
			return m;
		}

		mat4 rot_z(f32 rad)
		{
			mat4 m = identity();
			const f32 c = std::cos(rad), s = std::sin(rad);
			m[0][0] = c;  m[0][1] = s;
			m[1][0] = -s; m[1][1] = c;
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
			if (const char* v = ::getenv(name)) return v;
			return {};
#endif
		}

		// Locate a guest constant slot inside the transient buffer.
		// Returns nullptr when this program does not read that slot.
		f32* find_slot(void* buffer, const u16* reloc, usz reloc_size, u32 guest_index)
		{
			char* base = static_cast<char*>(buffer);

			if (reloc_size == 0)
			{
				// Full-bank upload: natural indexing.
				if (guest_index >= 468) return nullptr;
				return reinterpret_cast<f32*>(base + guest_index * 16);
			}

			for (usz i = 0; i < reloc_size; ++i)
			{
				if (reloc[i] == guest_index)
				{
					return reinterpret_cast<f32*>(base + i * 16);
				}
			}
			return nullptr;
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
						if (!m_slots[j]) continue;
						for (u32 i = 0; i < 4; ++i)
							m_slots[j][i] = m_local[i][j];
					}
				}
			}

			// False (and nothing bound) if the program does not read all 4 slots.
			// xyw: DP4 slots base, base+1, base+2 are clip x, y, w (no z slot).
			// explicit_slots: the 4 slots themselves, when not contiguous.
			bool bind(void* buffer, const u16* reloc, usz reloc_size, u32 base, bool column_vectors, bool xyw = false,
				const std::array<u32, 4>* explicit_slots = nullptr)
			{
				release();
				column_vectors |= xyw;
				u32 slot_of[4] = { base, base + 1, xyw ? umax : base + 2, xyw ? base + 2 : base + 3 };
				if (explicit_slots && (*explicit_slots)[0] != umax)
				{
					for (u32 k = 0; k < 4; ++k) slot_of[k] = (*explicit_slots)[k];
				}
				for (u32 k = 0; k < 4; ++k)
				{
					m_slots[k] = slot_of[k] == umax ? nullptr : find_slot(buffer, reloc, reloc_size, slot_of[k]);
					if (!m_slots[k] && !(column_vectors && k == 2))
					{
						return false;
					}
				}

				m_transposed = column_vectors;
				for (u32 k = 0; k < 4; ++k)
				{
					rows[k] = m_transposed ? m_local[k] : m_slots[k];
				}
				if (m_transposed)
				{
					for (u32 i = 0; i < 4; ++i)
						for (u32 j = 0; j < 4; ++j)
							m_local[i][j] = m_slots[m_slots[j] ? j : 3][i]; // only z may be absent: z = w
				}
				return true;
			}

			// Drop the binding without writing back (the block was not modified).
			void release()
			{
				m_transposed = false;
				for (f32*& r : rows) r = nullptr;
			}

			f32* rows[4] = {};

		private:
			f32* m_slots[4] = {};
			f32 m_local[4][4] = {};
			bool m_transposed = false;
		};

		bool is_perspective(f32* const r[4])
		{
			constexpr f32 eps = 1e-6f;
			return !(std::fabs(r[0][3]) < eps && std::fabs(r[1][3]) < eps &&
				std::fabs(r[2][3]) < eps && std::fabs(r[3][3] - 1.f) < eps);
		}

		// Clip x, y and w directions (columns 0, 1, 3 of rows 0..2) mutually orthogonal.
		bool is_rigid(f32* const r[4])
		{
			const auto dot = [&](u32 i, u32 j) { return r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j]; };
			const f32 n0 = std::sqrt(dot(0, 0)), n1 = std::sqrt(dot(1, 1)), n3 = std::sqrt(dot(3, 3));
			constexpr f32 tol = 0.1f;
			return n0 > 1e-8f && n1 > 1e-8f && n3 > 1e-8f &&
				std::fabs(dot(0, 1)) <= tol * n0 * n1 &&
				std::fabs(dot(0, 3)) <= tol * n0 * n3 &&
				std::fabs(dot(1, 3)) <= tol * n1 * n3;
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
			std::span<const std::array<u32, 4>> explicit_slots = {})
		{
			for (usz i = 0; i < candidates.size(); ++i)
			{
				const u32 candidate = candidates[i];
				const std::array<u32, 4>* slots = i < explicit_slots.size() ? &explicit_slots[i] : nullptr;
				if (block.bind(buffer, reloc, reloc_size, candidate, column_vectors, xyw, slots) && is_perspective(block.rows) &&
					(!require_rigid || is_rigid(block.rows)) && (require_aspect <= 0.f || has_camera_aspect(block.rows, require_aspect)))
				{
					return true;
				}
			}
			block.release();
			return false;
		}
	}

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

	std::shared_ptr<const title_profile> load_title_profile(std::string_view title_id)
	{
		if (title_id.empty())
		{
			return nullptr;
		}

		const std::string path = fs::get_executable_dir() + "vr_profiles/" + std::string(title_id) + ".json";
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
				if (required && error.empty()) error = fmt::format("missing \"%s\"", key);
				return false;
			}
			std::string node_error;
			T value = get_yaml_node_value<T>(node, node_error);
			if (!node_error.empty())
			{
				if (error.empty()) error = fmt::format("\"%s\": %s", key, node_error);
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
			if (error.empty()) error = std::move(what);
		};
		// Unknown keys are only warned about, but that catches typos in optional ones.
		const auto check_keys = [&](const YAML::Node& node, const char* where, std::initializer_list<std::string_view> known)
		{
			if (!node || !node.IsMap()) return;
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
				std::array<u32, 4> slots{ umax, umax, umax, umax };
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
				if (!node_error.empty()) fail("camera_blocks: " + node_error);
			}
		}
		if (profile->camera_blocks.empty())
		{
			fail("camera_blocks must list at least one slot");
		}

		read(root, "output_aspect_tolerance", profile->output_aspect_tolerance);
		read(root, "camera_target_aspect", profile->camera_target_aspect, false);

		if (const YAML::Node widths = child(root, "game_camera_target_widths"); widths && widths.IsSequence())
		{
			for (const auto& width : widths)
			{
				std::string node_error;
				profile->game_camera_target_widths.push_back(get_yaml_node_value<u32>(width, node_error));
				if (!node_error.empty()) fail("game_camera_target_widths: " + node_error);
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
		if (std::string bare; read(screen_space, "bare_projection", bare, false))
		{
			profile->screen_space_bare_projection = bare == "true";
		}
		if (std::string offset; read(screen_space, "depth_offset_projection", offset, false))
		{
			profile->screen_space_depth_offset_projection = offset == "true";
		}
		if (std::string rotation; read(screen_space, "rotation_only_passthrough", rotation, false))
		{
			profile->screen_space_rotation_only_passthrough = rotation == "true";
		}
		if (std::string hud; read(screen_space, "passthrough_hud", hud, false))
		{
			profile->screen_space_passthrough_hud = hud == "true";
		}
		if (const YAML::Node programs = child(screen_space, "preprojected_programs"); programs && programs.IsSequence())
		{
			for (const auto& program : programs)
			{
				const std::string text = program.as<std::string>();
				char* end = nullptr;
				const u64 hash = std::strtoull(text.c_str(), &end, 16);
				if (text.empty() || !end || *end) fail("screen_space.preprojected_programs: '" + text + "' is not a hex program hash");
				profile->screen_space_preprojected_programs.push_back(hash);
			}
		}

		read(root, "reference_screen_width", profile->reference_screen_width, false);
		if (std::string rigid; read(root, "require_rigid_camera", rigid, false))
		{
			profile->require_rigid_camera = rigid == "true";
		}
		if (std::string aspect; read(root, "require_camera_aspect", aspect, false))
		{
			profile->require_camera_aspect = aspect == "true";
		}
		read(root, "max_fps", profile->max_fps, false);
		if (const YAML::Node targets = child(root, "game_refresh_rate_f32"); targets && targets.IsSequence())
		{
			for (const auto& target : targets)
			{
				// "0xADDR", "[0xPTR]" or "[0xPTR]+0xOFF"
				const std::string text = target.as<std::string>();
				title_profile::guest_address a;
				std::string rest = text;
				if (rest.starts_with("["))
				{
					const usz close = rest.find(']');
					if (close == umax)
					{
						fail("game_refresh_rate_f32: '" + text + "' has no closing ]");
						continue;
					}
					a.deref = true;
					a.address = static_cast<u32>(std::strtoul(rest.substr(1, close - 1).c_str(), nullptr, 16));
					rest = rest.substr(close + 1);
					if (rest.starts_with("+"))
					{
						a.offset = static_cast<u32>(std::strtoul(rest.c_str() + 1, nullptr, 16));
					}
					else if (!rest.empty())
					{
						fail("game_refresh_rate_f32: '" + text + "' expected +offset after ]");
						continue;
					}
				}
				else
				{
					a.address = static_cast<u32>(std::strtoul(rest.c_str(), nullptr, 16));
				}
				if (!a.address)
				{
					fail("game_refresh_rate_f32: '" + text + "' is not an address");
					continue;
				}
				profile->game_refresh_rate_f32.push_back(a);
			}
		}
		if (std::string current; read(root, "current_frame_copies", current, false))
		{
			profile->current_frame_copies = current == "true";
		}

		check_keys(root, "", { "schema", "title_id", "app_version", "matrix_layout", "camera_blocks", "output_aspect_tolerance", "camera_target_aspect",
			"camera_position", "stereo", "screen_space", "reference_screen_width", "game_refresh_rate_f32", "max_fps", "require_rigid_camera", "require_camera_aspect",
			"game_camera_target_widths", "current_frame_copies" });
		check_keys(camera_position, " in camera_position", { "slot", "eye_baseline" });
		check_keys(stereo, " in stereo", { "formula", "per_eye_separation", "convergence", "by_target_width" });
		check_keys(screen_space, " in screen_space", { "orthographic_block", "bare_projection", "depth_offset_projection", "rotation_only_passthrough", "passthrough_hud", "preprojected_programs" });
		if (const YAML::Node rules = child(stereo, "by_target_width"); rules && rules.IsSequence())
		{
			for (const auto& node : rules)
			{
				check_keys(node, " in stereo.by_target_width", { "output_width_divisor", "per_eye_separation", "convergence" });
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
		return load_title_profile(title_id) != nullptr;
	}

	namespace
	{
		atomic_t<u32> g_headset_refresh_hz{0};
	}

	void set_headset_refresh_rate(u32 hz)
	{
		if (g_headset_refresh_hz.exchange(hz) != hz && hz)
		{
			vr_probe_log.notice("Headset refresh rate: %u Hz", hz);
		}
	}

	u64 effective_vblank_rate()
	{
		const u64 configured = g_cfg.video.vblank_rate;
		const u32 headset = g_headset_refresh_hz.load();
		if (!headset || !g_cfg.video.vr.enabled || !g_cfg.video.vr.match_headset_rate)
		{
			return configured;
		}

		const title_profile* profile = camera_probe::get().profile();
		return profile && profile->syncs_to_headset() ? headset : configured;
	}

	u32 effective_reprojection_margin()
	{
		const s64 configured = g_cfg.video.vr.reprojection_margin.get();
		if (configured >= 0)
		{
			return static_cast<u32>(configured);
		}
		const title_profile* profile = camera_probe::get().profile();
		return profile && profile->max_fps != 0 ? 10 : 0;
	}

	void update_game_refresh_rate()
	{
		const title_profile* profile = camera_probe::get().profile();
		if (!profile || profile->game_refresh_rate_f32.empty())
		{
			return;
		}

		const f32 rate = static_cast<f32>(effective_vblank_rate());
		for (const auto& target : profile->game_refresh_rate_f32)
		{
			u32 address = target.address;
			if (target.deref)
			{
				if (!vm::check_addr(address, vm::page_readable, 4))
				{
					continue;
				}
				const u32 base = vm::_ref<be_t<u32>>(address);
				if (!base)
				{
					continue;
				}
				address = base + target.offset;
			}
			if (!vm::check_addr(address, vm::page_writable, 4))
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
	}

	const title_profile* camera_probe::profile() const
	{
		const std::string& title = Emu.GetTitleID();
		std::lock_guard lock(m_profile_mutex);
		if (title != m_profile_title)
		{
			m_profile_title = title;
			m_profile = load_title_profile(title);
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
					title, blocks, static_cast<s32>(p.camera_position_slot), p.eye_baseline,
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
	}

	camera_probe& camera_probe::get()
	{
		static camera_probe instance;
		return instance;
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
				m_audit_rot = { 1.f, 0.f, 0.f, 0.f, cp, -sp, 0.f, sp, cp };
				vr_probe_log.success("Rotation audit: right eye pitched by %f degrees, no stereo separation.", m_audit_yaw_deg);
			}
			else
			{
				m_audit_rot = { c, 0.f, s, 0.f, 1.f, 0.f, -s, 0.f, c };
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

	void camera_probe::reset_params()
	{
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
		if (!m_enabled || m_config_path.empty())
		{
			return;
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
			if (eq == std::string::npos) continue;

			const std::string k = tok.substr(0, eq);
			const std::string v = tok.substr(eq + 1);

			auto as_f = [&]() { return static_cast<f32>(std::atof(v.c_str())); };
			auto as_u = [&]() { return static_cast<u32>(std::atoi(v.c_str())); };

			if      (k == "base")  m_base = as_u();
			else if (k == "cam")   m_cam_slot = as_u();
			else if (k == "yaw")   { m_yaw = as_f();   m_have_xform = true; }
			else if (k == "pitch") { m_pitch = as_f(); m_have_xform = true; }
			else if (k == "roll")  { m_roll = as_f();  m_have_xform = true; }
			else if (k == "tx")    { m_tx = as_f();    m_have_xform = true; }
			else if (k == "ty")    { m_ty = as_f();    m_have_xform = true; }
			else if (k == "tz")    { m_tz = as_f();    m_have_xform = true; }
			else if (k == "eye")   { m_eye = as_f();   m_have_xform = true; }
			else if (k == "slot")  { m_raw_slot = as_u(); m_have_raw = true; }
			else if (k == "comp")  { m_raw_comp = as_u(); }
			else if (k == "add")   { m_raw_add = as_f(); }
			else if (k == "reqcam") m_require_cam = (as_u() != 0);
			else if (k == "layout") m_column_vectors = v == "columns";
			else if (k == "stereo") { m_stereo_sep = as_f(); m_have_stereo = true; }
			else if (k == "conv")   { m_stereo_conv = as_f(); }
			else if (k == "render") { m_render_enabled = (as_u() != 0); }
			else if (k == "title") m_title = v;
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

		// The position policy has a wider domain than the matrix policy. Locate a
		// perspective block first so the camera position's offset follows the exact
		// camera right axis used by this draw (not a global axis or a stale prior draw).
		const u32 base_override[2] = { m_base, m_base + 4 };
		const std::span<const u32> camera_blocks = m_base != umax ? std::span<const u32>(base_override) : std::span<const u32>(profile.camera_blocks);
		matrix_block block;
		f32* const* const rows = block.rows;
		const size2u output_eye = g_fxo->get<rsx::avconf>().video_frame_size();
		const f32 camera_aspect = profile.require_camera_aspect && output_eye.height ? static_cast<f32>(output_eye.width) / output_eye.height : 0.f;
		if (!bind_camera_block(block, buffer, reloc, reloc_size, camera_blocks, profile.column_vectors, profile.require_rigid_camera, profile.xyw_rows, camera_aspect,
			m_base != umax ? std::span<const std::array<u32, 4>>() : std::span<const std::array<u32, 4>>(profile.camera_block_slots)))
		{
			apply_vr_screen_space(profile, buffer, reloc, reloc_size, surface_w, surface_h, eye_sign);
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

		// Rotation-invariance audit (see m_audit_yaw_deg): the same classifier and
		// clip-space rotation as the headset path, with the left eye unrotated and
		// no eye offsets, so the only difference between the eyes is a known yaw.
		if (m_audit_yaw_deg != 0.f && !m_vr_view)
		{
			if (!output_aspect_match)
			{
				return false;
			}

			static constexpr std::array<f32, 9> identity3 = { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f };
			apply_vr_rotation(rows, eye_sign > 0.f ? m_audit_rot : identity3, {});

			// The headset path's FOV remap, onto a symmetric frustum (l, r, u, d).
			if (m_audit_fov_tan > 0.f && m_vr_proj_valid)
			{
				const f32 t[4] = { -m_audit_fov_tan, m_audit_fov_tan, m_audit_fov_tan, -m_audit_fov_tan };
				remap_to_eye_fov(rows, t, m_vr_proj_x, m_vr_proj_y);
			}
			store_eye_block(eye_sign, game_block, rows);
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
			if (!output_aspect_match || !m_vr_view)
			{
				block.release();
				return false;
			}
			apply_vr_rotation(rows, m_vr_rot, {});
			if (m_vr_hmd_fov && m_vr_proj_valid)
			{
				remap_to_eye_fov(rows, m_vr_eye_fov[eye_sign < 0.f ? 0 : 1], m_vr_proj_x, m_vr_proj_y);
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
		if (output_aspect_match && m_vr_view && m_vr_hmd_fov &&
			(profile.screen_space_bare_projection || profile.screen_space_depth_offset_projection))
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
			const bool bare_projection = camera_space &&
				(depth_offset ? profile.screen_space_depth_offset_projection : profile.screen_space_bare_projection);
			if (bare_projection)
			{
				// Still the game's projection, so it keeps the FOV cache valid on
				// screens with no other camera draws.
				m_vr_proj_x = std::fabs(rows[0][0] / rows[2][3]);
				m_vr_proj_y = std::fabs(rows[1][1] / rows[2][3]);
				m_vr_proj_valid = true;
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
		if (output_aspect_match && m_vr_view)
		{
			apply_vr_rotation(rows, m_vr_rot, m_vr_head_units);
		}

		if (output_aspect_match)
		{
			m_render_camera_right = { rows[0][0], rows[1][0], rows[2][0] };
			const f32 length = std::sqrt(m_render_camera_right[0] * m_render_camera_right[0] +
				m_render_camera_right[1] * m_render_camera_right[1] +
				m_render_camera_right[2] * m_render_camera_right[2]);
			if (length > 1e-8f)
			{
				for (f32& v : m_render_camera_right) v /= length;
				m_render_camera_right_valid = true;
			}
		}

		const u32 cam_slot = m_cam_slot != umax ? m_cam_slot : profile.camera_position_slot;
		if (f32* cam = cam_slot != umax ? find_slot(buffer, reloc, reloc_size, cam_slot) : nullptr; cam && m_render_camera_right_valid)
		{
			const f32 half_eye_baseline = profile.eye_baseline * 0.5f * (m_vr_view ? m_vr_eye_scale : m_screen_stereo_scale);
			cam[0] += eye_sign * half_eye_baseline * m_render_camera_right[0];
			cam[1] += eye_sign * half_eye_baseline * m_render_camera_right[1];
			cam[2] += eye_sign * half_eye_baseline * m_render_camera_right[2];
		}

		if (!output_aspect_match)
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
			rows[3][0] -= sep * m_vr_eye_scale * convergence;

			if (m_vr_hmd_fov && m_vr_proj_valid)
			{
				remap_to_eye_fov(rows, m_vr_eye_fov[eye_sign < 0.f ? 0 : 1], m_vr_proj_x, m_vr_proj_y);
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
			store_eye_block(eye_sign, game_block, rows);
			return true;
		}

		for (u32 r = 0; r < 4; ++r)
		{
			rows[r][0] += sep * rows[r][3];
		}
		rows[3][0] -= sep * convergence;
		store_eye_block(eye_sign, game_block, rows);
		return true;
	}

	void camera_probe::set_vr_view(const f32 q[4], const f32 pos[3], f32 eye_scale,
		f32 fov_scale, bool flip_y, f32 ipd, f32 camera_depth)
	{
		// OpenXR rotation matrix (right, up, back basis) from the quaternion.
		const f32 x = q[0], y = q[1], z = q[2], w = q[3];
		const f32 r[3][3] =
		{
			{ 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w) },
			{ 2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
			{ 2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y) },
		};

		// The draw's clip basis is (NDC x = right, NDC y = up unless flip_y,
		// clip w = forward). Conjugate by S = diag(1, sy, -1), which is its own
		// inverse, and transpose: a camera turned by R sees view vectors by R^T.
		const f32 s[3] = { 1.f, flip_y ? -1.f : 1.f, -1.f };
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
		const title_profile* profile = this->profile();
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
		m_vr_hud_depth = hud_depth;
		m_vr_hmd_fov = tangents != nullptr;
		m_vr_hud_scale = hud_scale;
		m_vr_hud_fixed = hud_fixed;
		m_vr_hud_offset_x = hud_offset_x;
		m_vr_hud_offset_y = hud_offset_y;
		if (tangents)
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
		if (!m_vr_view || !m_vr_hmd_fov || !m_vr_proj_valid || profile.screen_space_block == umax)
		{
			return;
		}

		const auto& avconf = g_fxo->get<rsx::avconf>();
		const size2u eye = avconf.video_frame_size();
		if (!surface_w || !surface_h || !eye.width || !eye.height ||
			!profile.is_view_target(surface_w, surface_h, static_cast<f32>(eye.width) / eye.height))
		{
			return;
		}

		matrix_block block;
		if (!block.bind(buffer, reloc, reloc_size, profile.screen_space_block, profile.column_vectors))
		{
			return;
		}

		if (is_perspective(block.rows))
		{
			block.release();
			return;
		}

		map_vr_screen_box(block.rows, eye_sign, static_cast<f32>(eye.width) / eye.height);
	}

	void camera_probe::map_vr_screen_box(f32* const rows[4], f32 eye_sign, f32 aspect) const
	{
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
			fit_x = std::min({ fit_x, -o[0], o[1] });
			fit_y = std::min({ fit_y, o[2], -o[3] });
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
			undo_viewport(rows);
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
		}
		undo_viewport(rows);
	}

	void camera_probe::remap_to_eye_fov(f32* const rows[4], const f32* t, f32 A, f32 B) const
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
		undo_viewport(rows);
	}

	bool camera_probe::map_vr_passthrough_hud(f32 m[4][4], f32 eye_sign, f32 aspect) const
	{
		const title_profile* p = profile();
		if (!p || !p->screen_space_passthrough_hud || !m_vr_view || !m_vr_hmd_fov)
		{
			return false;
		}
		f32 r0[4] = { 1.f, 0.f, 0.f, 0.f };
		f32 r1[4] = { 0.f, 1.f, 0.f, 0.f };
		f32 r2[4] = { 0.f, 0.f, 1.f, 0.f };
		f32 r3[4] = { 0.f, 0.f, 0.f, 1.f };
		f32* const rows[4] = { r0, r1, r2, r3 };
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

	bool camera_probe::map_vr_preprojected(f32 m[4][4], f32 eye_sign, u64 program_hash) const
	{
		const title_profile* p = profile();
		const u32 eye = eye_sign < 0.f ? 0 : 1;
		if (!p || !m_vr_last_block_valid[eye] || std::find(p->screen_space_preprojected_programs.begin(),
			p->screen_space_preprojected_programs.end(), program_hash) == p->screen_space_preprojected_programs.end())
		{
			return false;
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
				if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
			}
			if (std::fabs(a[pivot][c]) < 1e-12)
			{
				return false;
			}
			if (pivot != c)
			{
				for (u32 k = 0; k < 8; ++k) std::swap(a[c][k], a[pivot][k]);
			}
			const f64 inv = 1.0 / a[c][c];
			for (u32 k = 0; k < 8; ++k) a[c][k] *= inv;
			for (u32 r = 0; r < 4; ++r)
			{
				if (r == c || a[r][c] == 0.0) continue;
				const f64 f = a[r][c];
				for (u32 k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
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

	void camera_probe::undo_viewport(f32* const rows[4]) const
	{
		// Target NDC = NDC * k + o, with k = scale / (clip size / 2) and o the offset from
		// the target centre; a viewport covering the target exactly is k = +-1, o = 0.
		// The mapping above wants target NDC = its NDC (in the viewport's own y sense),
		// so X' = (X - o*W) / k.
		const f32 half_w = rsx::method_registers.surface_clip_width() / 2.f;
		const f32 half_h = rsx::method_registers.surface_clip_height() / 2.f;
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
		if (ax < 0.25f || ay < 0.25f || ax > 4.f || ay > 4.f ||
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

	void camera_probe::apply_vr_rotation(f32* const rows[4], const std::array<f32, 9>& R, const std::array<f32, 3>& head) const
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
		if (std::fabs(dot(0, 1)) <= tol * n0 * n1 &&
			std::fabs(dot(0, 3)) <= tol * n0 * n3 &&
			std::fabs(dot(1, 3)) <= tol * n1 * n3)
		{
			m_vr_proj_x = n0 / n3;
			m_vr_proj_y = n1 / n3;
			m_vr_proj_valid = true;
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
		const u32 base_override[2] = { m_base, m_base + 4 };
		const std::span<const u32> camera_blocks = m_base != umax ? std::span<const u32>(base_override)
			: profile ? std::span<const u32>(profile->camera_blocks) : std::span<const u32>();
		const u32 cam_slot = m_cam_slot != umax ? m_cam_slot : profile ? profile->camera_position_slot : umax;
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
			m_base == umax && profile ? std::span<const std::array<u32, 4>>(profile->camera_block_slots) : std::span<const std::array<u32, 4>>()))
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
		f32 cam[3] = { 0.f, 0.f, 0.f };
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
			f32 right[3] = { M[0][0], M[1][0], M[2][0] };
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
		if (m_pitch != 0.f) R = mul(R, rot_x(m_pitch * deg2rad));
		if (m_yaw   != 0.f) R = mul(R, rot_y(m_yaw   * deg2rad));
		if (m_roll  != 0.f) R = mul(R, rot_z(m_roll  * deg2rad));

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
}
