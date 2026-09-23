#include "stdafx.h"
#include "rsx_vr_profile_generator.h"
#include "rsx_camera_probe.h"

#include "Emu/System.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/system_config.h"
#include "Emu/IdManager.h"
#include "Emu/localized_string_id.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/Overlays/overlay_message.h"
#include "Utilities/File.h"

#include "util/logs.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <optional>

LOG_CHANNEL(vr_gen_log, "VRGEN");

namespace rsx::vr
{
	namespace
	{
		// Sampling: wait for the menu to close and the game to settle, then sample
		// one frame every half second over ten seconds of play, so a moving scene
		// contributes many viewpoints (and objects that only appear along the way).
		constexpr u32 settle_frames = 30;
		constexpr u64 sample_period_us = 500'000;
		constexpr u64 play_time_us = 10'000'000;
		constexpr u64 max_frame_gap_us = 200'000; // longer gaps (loading, stalls) are not play time
		constexpr u32 min_frames = 5;

		constexpr f64 rigid_tolerance = 0.1;    // require_rigid_camera's test
		constexpr f64 aspect_tolerance = 0.1;   // B/A against the output aspect
		constexpr f64 human_ipd = 0.064;         // metres
		constexpr f64 reference_near_plane = 0.1; // metres; a common engine near plane (Pure's is exactly 0.1)
		constexpr f64 convergence_in_baselines = 40.0;

		// A 4-slot block as DP4 rows: clip[i] = dot(row_i, (v, 1)).
		using mat4 = std::array<std::array<f64, 4>, 4>;

		struct block_result
		{
			mat4 m{};
			bool z_missing = false;
		};

		struct slot_reader
		{
			const std::vector<u16>& ids;
			const std::vector<std::array<f32, 4>>& values;
			bool full_bank;

			const std::array<f32, 4>* get(u32 slot) const
			{
				if (full_bank)
				{
					return slot < values.size() ? &values[slot] : nullptr;
				}
				for (usz i = 0; i < ids.size(); ++i)
				{
					if (ids[i] == slot) return &values[i];
				}
				return nullptr;
			}
		};

		// columns: slot i is row i, and z may be absent (z = w, a far-plane sky).
		// rows: slot k is row k of M in clip = v * M, i.e. the transpose.
		std::optional<block_result> read_block(const slot_reader& r, u32 base, bool columns)
		{
			const std::array<f32, 4>* s[4];
			for (u32 k = 0; k < 4; ++k) s[k] = r.get(base + k);

			block_result out;
			if (columns)
			{
				if (!s[0] || !s[1] || !s[3]) return std::nullopt;
				out.z_missing = !s[2];
				const std::array<f32, 4>* rows[4] = { s[0], s[1], s[2] ? s[2] : s[3], s[3] };
				for (u32 i = 0; i < 4; ++i)
					for (u32 j = 0; j < 4; ++j)
						out.m[i][j] = (*rows[i])[j];
				return out;
			}

			for (u32 k = 0; k < 4; ++k)
			{
				if (!s[k]) return std::nullopt;
			}
			for (u32 i = 0; i < 4; ++i)
				for (u32 j = 0; j < 4; ++j)
					out.m[i][j] = (*s[j])[i];
			return out;
		}

		f64 len3(const std::array<f64, 4>& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
		f64 dot3(const std::array<f64, 4>& a, const std::array<f64, 4>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

		bool is_perspective(const mat4& m)
		{
			constexpr f64 eps = 1e-6;
			return !(std::fabs(m[3][0]) < eps && std::fabs(m[3][1]) < eps && std::fabs(m[3][2]) < eps && std::fabs(m[3][3] - 1.0) < eps);
		}

		// Largest |cos| between the clip x, y and w directions (0 = rigid).
		f64 rigidity(const mat4& m)
		{
			const f64 nx = len3(m[0]), ny = len3(m[1]), nw = len3(m[3]);
			if (nx < 1e-8 || ny < 1e-8 || nw < 1e-8) return 9.0;
			return std::max({ std::fabs(dot3(m[0], m[1])) / (nx * ny), std::fabs(dot3(m[0], m[3])) / (nx * nw),
				std::fabs(dot3(m[1], m[3])) / (ny * nw) });
		}

		// Projection scales A = |x|/|w|, B = |y|/|w|.
		std::pair<f64, f64> projection(const mat4& m)
		{
			const f64 nw = len3(m[3]);
			return { len3(m[0]) / nw, len3(m[1]) / nw };
		}

		bool aspect_matches(const mat4& m, f64 output_aspect, f64 tolerance)
		{
			const auto [a, b] = projection(m);
			return a > 1e-8 && std::fabs((b / a) / output_aspect - 1.0) <= tolerance;
		}

		// A plausible camera: perspective, rigid, and square pixels at the output aspect.
		bool is_camera(const mat4& m, f64 output_aspect)
		{
			return is_perspective(m) && rigidity(m) <= rigid_tolerance && aspect_matches(m, output_aspect, aspect_tolerance);
		}

		// The eye point in the block's input space: clip x = y = w = 0.
		std::optional<std::array<f64, 3>> eye_point(const mat4& m)
		{
			const f64 a[3][3] = { { m[0][0], m[0][1], m[0][2] }, { m[1][0], m[1][1], m[1][2] }, { m[3][0], m[3][1], m[3][2] } };
			const f64 b[3] = { -m[0][3], -m[1][3], -m[3][3] };
			const f64 det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
				a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
			if (std::fabs(det) < 1e-12) return std::nullopt;

			std::array<f64, 3> x{};
			for (u32 c = 0; c < 3; ++c)
			{
				f64 t[3][3];
				for (u32 i = 0; i < 3; ++i)
					for (u32 j = 0; j < 3; ++j)
						t[i][j] = j == c ? b[i] : a[i][j];
				x[c] = (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) +
					t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])) / det;
			}
			return x;
		}

		// Near plane distance in the block's view-depth units (GL clip: z = -w at near).
		std::optional<f64> near_plane(const mat4& m)
		{
			const f64 ww = dot3(m[3], m[3]);
			if (ww < 1e-12) return std::nullopt;
			const f64 k = dot3(m[2], m[3]) / ww;
			const f64 t = m[2][3] - k * m[3][3];
			if (std::fabs(k + 1.0) < 1e-9) return std::nullopt;
			const f64 n = -t / (k + 1.0) / std::sqrt(ww);
			return n > 0.0 ? std::optional<f64>(n) : std::nullopt;
		}

		// A projection with no view rotation or translation folded in (drawn in the
		// camera's own space, e.g. WipEout's menu particle cloud): screen space.
		bool is_bare_projection(const mat4& m)
		{
			constexpr f64 eps = 1e-5;
			return std::fabs(m[1][0]) < eps && std::fabs(m[2][0]) < eps && std::fabs(m[3][0]) < eps &&
				std::fabs(m[0][1]) < eps && std::fabs(m[2][1]) < eps && std::fabs(m[3][1]) < eps &&
				std::fabs(m[0][2]) < eps && std::fabs(m[1][2]) < eps &&
				std::fabs(m[0][3]) < eps && std::fabs(m[1][3]) < eps && std::fabs(m[3][3]) < eps &&
				std::fabs(m[3][2]) > eps;
		}

		f64 median(std::vector<f64> v)
		{
			std::sort(v.begin(), v.end());
			return v[v.size() / 2];
		}

		std::string fmt_number(f64 v)
		{
			return fmt::format("%.6g", v);
		}
	}

	profile_generator& profile_generator::get()
	{
		static profile_generator instance;
		return instance;
	}

	void profile_generator::request()
	{
		state expected = state::idle;
		if (!m_state.compare_exchange(expected, state::waiting))
		{
			return;
		}

		{
			std::lock_guard lock(m_mutex);
			m_samples.clear();
		}
		m_frame_counter = 0;
		m_frames_sampled = 0;
		vr_gen_log.success("VR profile generation requested for %s; sampling gameplay after the menu closes.", Emu.GetTitleID());
		rsx::overlays::queue_message(localized_string_id::VR_PROFILE_GENERATING, 10'000'000);
	}

	void profile_generator::record_draw(std::span<const u16> constant_ids, u32 program_id, u16 surface_w, u16 surface_h)
	{
		const auto& bank = rsx::method_registers.transform_constants;

		draw_sample s;
		s.program = program_id;
		s.width = surface_w;
		s.height = surface_h;
		s.full_bank = constant_ids.empty();

		const auto push = [&](u32 index)
		{
			std::array<f32, 4> v;
			for (u32 k = 0; k < 4; ++k) v[k] = std::bit_cast<f32>(bank[index][k]);
			s.values.push_back(v);
		};

		if (s.full_bank)
		{
			s.values.reserve(468);
			for (u32 i = 0; i < 468; ++i) push(i);
		}
		else
		{
			s.ids.assign(constant_ids.begin(), constant_ids.end());
			s.values.reserve(s.ids.size());
			for (const u16 id : s.ids) push(id);
		}

		std::lock_guard lock(m_mutex);
		m_samples.push_back(std::move(s));
	}

	void profile_generator::on_frame_end()
	{
		switch (m_state.load())
		{
		case state::waiting:
			if (++m_frame_counter >= settle_frames)
			{
				m_frame_counter = 0;
				m_played_us = 0;
				m_next_sample_us = sample_period_us;
				m_last_frame_us = get_system_time();
				m_state = state::sampling;
				m_sample_this_frame = true;
			}
			break;
		case state::sampling:
		{
			if (m_sample_this_frame.exchange(false))
			{
				std::lock_guard lock(m_mutex);
				// A frame only counts if the game drew in it (it may still be paused).
				if (!m_samples.empty() && m_samples.back().program != umax)
				{
					m_frames_sampled++;
					draw_sample marker;
					marker.program = umax;
					m_samples.push_back(std::move(marker));
				}
			}

			// Play time: frame-to-frame time while the game runs (not paused or stalled).
			const u64 now = get_system_time();
			const u64 gap = now - m_last_frame_us;
			m_last_frame_us = now;
			if (!Emu.IsPaused() && gap < max_frame_gap_us)
			{
				m_played_us += gap;
			}

			if (m_played_us >= play_time_us && m_frames_sampled >= min_frames)
			{
				m_state = state::analysing;
				finish();
				m_state = state::idle;
				break;
			}

			if (m_played_us >= m_next_sample_us)
			{
				m_next_sample_us += sample_period_us;
				m_sample_this_frame = true;
			}
			break;
		}
		default:
			break;
		}
	}

	void profile_generator::finish()
	{
		std::vector<draw_sample> samples;
		{
			std::lock_guard lock(m_mutex);
			samples = std::move(m_samples);
			m_samples.clear();
		}

		const std::string title = Emu.GetTitleID();
		const auto fail = [&](const std::string& why)
		{
			vr_gen_log.error("VR profile generation for %s failed: %s", title, why);
			rsx::overlays::queue_message(localized_string_id::VR_PROFILE_FAILED, 6'000'000);
		};

		const size2u eye = g_fxo->get<rsx::avconf>().video_frame_size();
		if (!eye.width || !eye.height)
		{
			fail("no output size");
			return;
		}
		const f64 output_aspect = static_cast<f64>(eye.width) / eye.height;

		std::vector<const draw_sample*> views;
		for (const auto& s : samples)
		{
			if (s.program == umax || !s.height) continue;
			if (std::fabs((static_cast<f64>(s.width) / s.height) / output_aspect - 1.0) <= 0.02)
			{
				views.push_back(&s);
			}
		}
		vr_gen_log.notice("Sampled %u frames over %.1f s of play: %u draws, %u on output-aspect (%ux%u) targets.",
			m_frames_sampled, m_played_us / 1'000'000.0, ::size32(samples) - m_frames_sampled, ::size32(views), eye.width, eye.height);

		// 1. Camera block candidates, both layouts. Programs that read the whole
		// bank (indexed constants) are left out: every slot "exists" there.
		std::map<std::pair<bool, u32>, u32> candidates;
		for (const draw_sample* s : views)
		{
			if (s->full_bank) continue;
			const slot_reader r{ s->ids, s->values, false };
			for (const u16 base : s->ids)
			{
				for (const bool columns : { false, true })
				{
					if (const auto b = read_block(r, base, columns); b && is_camera(b->m, output_aspect))
					{
						candidates[{ columns, base }]++;
					}
				}
			}
		}
		if (candidates.empty())
		{
			fail("no perspective camera matrix in the sampled draws (sample during 3D gameplay, not a menu)");
			return;
		}

		std::vector<std::pair<std::pair<bool, u32>, u32>> ranked(candidates.begin(), candidates.end());
		std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		for (usz i = 0; i < std::min<usz>(ranked.size(), 8); ++i)
		{
			vr_gen_log.notice("Candidate %s c[%u]: %u draws", ranked[i].first.first ? "column_vectors" : "row_vectors",
				ranked[i].first.second, ranked[i].second);
		}

		// Both layouts of one block can pass the camera test equally (ICO's bare
		// projection in c[60]). A camera's w row is the unit view direction; the
		// transposed reading usually is not, so prefer the layout closer to 1.
		bool columns = ranked.front().first.first;
		if (ranked.size() >= 2 && ranked[0].second == ranked[1].second && ranked[0].first.second == ranked[1].first.second)
		{
			const u32 base = ranked[0].first.second;
			const auto w_error = [&](bool as_columns)
			{
				std::vector<f64> errors;
				for (const draw_sample* s : views)
				{
					const slot_reader r{ s->ids, s->values, s->full_bank };
					if (const auto b = read_block(r, base, as_columns); b && is_camera(b->m, output_aspect))
					{
						errors.push_back(std::fabs(len3(b->m[3]) - 1.0));
					}
				}
				return errors.empty() ? 1e9 : median(errors);
			};
			const f64 rows_error = w_error(false), columns_error = w_error(true);
			columns = columns_error < rows_error;
			vr_gen_log.notice("Both layouts match c[%u] equally: |w row| is off unit length by %.4f (rows), %.4f (columns); using %s.",
				base, rows_error, columns_error, columns ? "column_vectors" : "row_vectors");
		}
		std::vector<u32> blocks;
		for (const auto& [key, count] : ranked)
		{
			if (key.first != columns || count < 2 || blocks.size() >= 4) continue;
			const bool overlaps = std::any_of(blocks.begin(), blocks.end(), [&](u32 b) { return key.second + 3 >= b && key.second <= b + 3; });
			if (!overlaps) blocks.push_back(key.second);
		}

		// 2. Stray matches: data in a listed block that passes the perspective
		// test but is no camera. If any, require rigid camera blocks.
		bool require_rigid = false;
		for (const draw_sample* s : views)
		{
			const slot_reader r{ s->ids, s->values, s->full_bank };
			for (const u32 base : blocks)
			{
				const auto b = read_block(r, base, columns);
				if (!b || !is_perspective(b->m)) continue;
				if (rigidity(b->m) > 0.3 && !aspect_matches(b->m, output_aspect, 0.5))
				{
					if (!require_rigid)
					{
						vr_gen_log.notice("Program %u holds non-camera data in c[%u]: require_rigid_camera.", s->program, base);
					}
					require_rigid = true;
				}
				break;
			}
		}

		// 3. Coverage, projection and camera position, with the final rule.
		std::map<u32, std::pair<u32, u32>> programs; // program -> (covered draws, uncovered draws)
		std::map<u16, std::vector<f64>> scale_a_by_width;
		std::vector<f64> near_planes;
		std::map<u16, std::vector<f64>> bare_scale_a_by_width;
		std::vector<f64> bare_near_planes;
		bool bare_projection = false;
		std::map<u32, u32> position_hits;
		u32 eye_points = 0;
		u32 covered_draws = 0;
		for (const draw_sample* s : views)
		{
			const slot_reader r{ s->ids, s->values, s->full_bank };
			std::optional<block_result> cam;
			u32 cam_base = 0;
			for (const u32 base : blocks)
			{
				if (auto b = read_block(r, base, columns); b && is_perspective(b->m) && (!require_rigid || rigidity(b->m) <= rigid_tolerance))
				{
					cam = b;
					cam_base = base;
					break;
				}
			}

			auto& prog = programs[s->program];
			if (!cam)
			{
				prog.second++;
				continue;
			}
			prog.first++;
			covered_draws++;

			if (is_bare_projection(cam->m))
			{
				bare_projection = true;
				const f64 a = projection(cam->m).first;
				bare_scale_a_by_width[s->width].push_back(a);
				if (const auto n = near_plane(cam->m)) bare_near_planes.push_back(*n);
			}
			else if (rigidity(cam->m) <= rigid_tolerance)
			{
				const f64 a = projection(cam->m).first;
				scale_a_by_width[s->width].push_back(a);
				if (const auto n = near_plane(cam->m)) near_planes.push_back(*n);
			}

			// A bare projection's eye point is always the origin: it would match any
			// (0, 0, 0, 1) constant, not a camera position.
			if (const auto e = is_bare_projection(cam->m) ? std::nullopt : eye_point(cam->m))
			{
				eye_points++;
				const f64 tolerance = 0.05 * std::max(1.0, std::sqrt((*e)[0] * (*e)[0] + (*e)[1] * (*e)[1] + (*e)[2] * (*e)[2]));
				const auto check = [&](u32 slot, const std::array<f32, 4>& v)
				{
					if (slot >= cam_base && slot < cam_base + 4) return;
					if (std::fabs(v[3] - 1.0) > 1e-4) return;
					const f64 d = std::sqrt((v[0] - (*e)[0]) * (v[0] - (*e)[0]) + (v[1] - (*e)[1]) * (v[1] - (*e)[1]) + (v[2] - (*e)[2]) * (v[2] - (*e)[2]));
					if (d < tolerance) position_hits[slot]++;
				};
				if (s->full_bank)
				{
					for (u32 i = 0; i < s->values.size(); ++i) check(i, s->values[i]);
				}
				else
				{
					for (usz i = 0; i < s->ids.size(); ++i) check(s->ids[i], s->values[i]);
				}
			}
		}

		// Every camera draw a bare projection: the game keeps the view in another
		// block (ICO), so the projection is the camera, not screen space.
		if (scale_a_by_width.empty() && !bare_scale_a_by_width.empty())
		{
			vr_gen_log.notice("Every camera draw is a bare projection: the view is applied elsewhere, so the projection is the camera.");
			scale_a_by_width = std::move(bare_scale_a_by_width);
			near_planes = std::move(bare_near_planes);
			bare_projection = false;
		}

		if (covered_draws < 10)
		{
			fail(fmt::format("only %u draws matched a camera block", covered_draws));
			return;
		}
		if (scale_a_by_width.empty())
		{
			fail(fmt::format("%u draws matched a camera block, but none is a rigid camera", covered_draws));
			return;
		}

		// The projection comes from the main camera view: the output width, or else
		// the width most camera draws use (ICO renders 3D below the output size).
		u16 main_width = static_cast<u16>(eye.width);
		if (!scale_a_by_width.contains(main_width))
		{
			usz most = 0;
			for (const auto& [width, values] : scale_a_by_width)
			{
				if (values.size() > most) { most = values.size(); main_width = width; }
			}
			vr_gen_log.notice("No rigid camera draws at the output width %u: using the projection of %u-wide targets (%u draws).",
				eye.width, main_width, static_cast<u32>(most));
		}
		const std::vector<f64>& scale_a = scale_a_by_width[main_width];

		for (const auto& [program, counts] : programs)
		{
			if (counts.second && !counts.first)
			{
				vr_gen_log.notice("Program %u: %u output-aspect draws not covered (full-screen pass, HUD, or a camera the profile misses).",
					program, counts.second);
			}
		}

		const f64 a = median(scale_a);

		u32 position_slot = umax;
		u32 best_hits = 0;
		for (const auto& [slot, hits] : position_hits)
		{
			if (hits > best_hits) { best_hits = hits; position_slot = slot; }
		}
		// Most draws fold an object matrix into the camera block, so their eye point
		// is in object space and matches nothing; a real camera position still
		// matches exactly on the draws drawn in world space (Pure: ~11% in a race).
		if (best_hits < std::max(10u, eye_points / 20))
		{
			position_slot = umax;
		}
		vr_gen_log.notice("Camera position: best c[%d] matches %u of %u eye points.", static_cast<s32>(position_slot), best_hits, eye_points);

		// 4. HUD: an orthographic block with pixel-sized scales.
		std::map<u32, u32> hud_hits;
		for (const draw_sample* s : views)
		{
			if (s->full_bank) continue;
			const slot_reader r{ s->ids, s->values, false };
			for (const u16 base : s->ids)
			{
				const auto b = read_block(r, base, columns);
				if (!b || b->z_missing || is_perspective(b->m)) continue;
				const f64 sx = std::fabs(b->m[0][0]), sy = std::fabs(b->m[1][1]);
				if (sx > 0 && sx < 0.01 && sy > 0 && sy < 0.01) hud_hits[base]++;
			}
		}
		u32 hud_block = umax;
		u32 hud_best = 1;
		for (const auto& [base, hits] : hud_hits)
		{
			if (hits > hud_best) { hud_best = hits; hud_block = base; }
		}

		// 5. World scale. Nothing in the constants says how big a unit is; the near
		// plane is the best cue (engines put it a similar real distance from the eye).
		// Pure: 0.1 units, metres. WipEout: 0.542 units -> 0.347 (native 3D: 0.240).
		// World Scale in the VR settings corrects the rest in the headset.
		f64 baseline = human_ipd;
		if (!near_planes.empty())
		{
			const f64 n = median(near_planes);
			baseline = std::clamp(human_ipd * n / reference_near_plane, human_ipd / 20.0, human_ipd * 20.0);
			vr_gen_log.notice("Near plane %.4f units: eye_baseline %.4f (world units per metre %.3f).", n, baseline, baseline / human_ipd);
		}

		// 6. Stereo. The headset keeps only sep * conv, the half-baseline in clip
		// units, so it scales with the projection: targets whose own projection
		// differs (WipEout's half-resolution pass, 0.75x) get their own rule.
		const f64 convergence = convergence_in_baselines * baseline;
		const f64 separation = a * baseline * 0.5 / convergence;
		std::vector<std::pair<u32, f64>> width_rules; // divisor, separation
		for (const auto& [width, values] : scale_a_by_width)
		{
			if (width == main_width || !width || eye.width % width || values.size() < 3) continue;
			const f64 ratio = median(values) / a;
			if (std::fabs(ratio - 1.0) <= 0.02) continue;
			width_rules.emplace_back(eye.width / width, separation * ratio);
			vr_gen_log.notice("Targets %u wide use %.3fx the projection: own stereo rule.", width, ratio);
		}

		std::string blocks_text;
		for (const u32 b : blocks) blocks_text += fmt::format("%s%u", blocks_text.empty() ? "" : ", ", b);

		std::string json = "{\n";
		json += "  \"schema\": 1,\n";
		json += fmt::format("  \"title_id\": \"%s\",\n", title);
		if (!Emu.GetAppVersion().empty()) json += fmt::format("  \"app_version\": \"%s\",\n", Emu.GetAppVersion());
		json += "\n";
		json += fmt::format("  \"matrix_layout\": \"%s\",\n", columns ? "column_vectors" : "row_vectors");
		json += fmt::format("  \"camera_blocks\": [%s],\n", blocks_text);
		if (require_rigid) json += "  \"require_rigid_camera\": true,\n";
		json += "  \"output_aspect_tolerance\": 0.02,\n\n";
		json += "  \"camera_position\": {\n";
		if (position_slot != umax) json += fmt::format("    \"slot\": %u,\n", position_slot);
		json += fmt::format("    \"eye_baseline\": %s\n  },\n\n", fmt_number(baseline));
		json += "  \"stereo\": {\n    \"formula\": \"clip_x_shear\",\n";
		json += fmt::format("    \"per_eye_separation\": %s,\n    \"convergence\": %s", fmt_number(separation), fmt_number(convergence));
		if (!width_rules.empty())
		{
			json += ",\n    \"by_target_width\": [\n";
			for (usz i = 0; i < width_rules.size(); ++i)
			{
				json += fmt::format("      { \"output_width_divisor\": %u, \"per_eye_separation\": %s, \"convergence\": %s }%s\n",
					width_rules[i].first, fmt_number(width_rules[i].second), fmt_number(convergence), i + 1 < width_rules.size() ? "," : "");
			}
			json += "    ]";
		}
		json += "\n  }";
		if (hud_block != umax || bare_projection)
		{
			json += ",\n\n  \"screen_space\": {\n";
			if (hud_block != umax) json += fmt::format("    \"orthographic_block\": %u%s\n", hud_block, bare_projection ? "," : "");
			if (bare_projection) json += "    \"bare_projection\": true\n";
			json += "  }";
		}
		json += "\n}\n";

		const std::string dir = fs::get_executable_dir() + "vr_profiles/";
		const std::string path = dir + title + ".json";
		if (!fs::create_path(dir) || !fs::write_file(path, fs::rewrite, json))
		{
			fail(fmt::format("cannot write '%s' (%s)", path, fs::g_tls_error));
			return;
		}

		vr_gen_log.success("VR profile written to '%s': %s c[%s]%s, camera position %s, HUD %s, projection A %.4f (%u of %u camera-view draws covered).",
			path, columns ? "column_vectors" : "row_vectors", blocks_text, require_rigid ? " (rigid)" : "",
			position_slot != umax ? fmt::format("c[%u]", position_slot) : "none",
			hud_block != umax ? fmt::format("c[%u]", hud_block) : "none", a, covered_draws, ::size32(views));

		camera_probe::get().reload_profile();
		if (!camera_probe::get().profile())
		{
			fail("the written profile did not load (see the VRPROBE error above)");
			return;
		}

		// Turn VR on for this game and keep it: the custom config is saved with it.
		// Stereo starts now; the headset needs the game to be restarted.
		if (!g_cfg.video.vr.enabled)
		{
			g_cfg.video.vr.enabled.set(true);
			Emu.CallFromMainThread([title]()
			{
				Emulator::SaveSettings(g_cfg.to_string(), title);
			});
		}

		rsx::overlays::queue_message(localized_string_id::VR_PROFILE_CREATED, 8'000'000);
	}
}
