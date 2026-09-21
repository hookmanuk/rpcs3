#include "stdafx.h"
#include "rsx_camera_probe.h"

#include "Emu/System.h"
#include "Utilities/File.h"
#include "Emu/IdManager.h"
#include "Emu/RSX/Utils/rsx_utils.h"

#include "util/logs.hpp"

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
	}

	camera_probe& camera_probe::get()
	{
		static camera_probe instance;
		return instance;
	}

	camera_probe::camera_probe()
	{
		m_config_path = read_env("RPCS3_VR_PROBE_FILE");
		const std::string cfg = read_env("RPCS3_VR_PROBE");

		if (cfg.empty() && m_config_path.empty())
		{
			// Gate 5 development default: render WipEout's profiled title in
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
		if (!m_active.load() || !m_render_enabled)
		{
			return false;
		}

		// Do not allocate/replay right-eye resources for unrelated titles merely
		// because the development default is armed.
		return m_title.empty() || Emu.GetTitleID() == m_title;
	}

	void camera_probe::reset_params()
	{
		m_base = 256;
		m_cam_slot = 465;
		m_yaw = m_pitch = m_roll = 0.f;
		m_tx = m_ty = m_tz = 0.f;
		m_eye = 0.f;
		m_have_xform = false;
		m_require_cam = false;
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
		m_title = "BCES00664";
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

		vr_probe_log.success("Camera probe ARMED for title '%s': %s", m_title, cfg);
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

		if (!m_title.empty() && Emu.GetTitleID() != m_title)
		{
			return false;
		}

		const auto is_perspective = [](f32* const r[4])
		{
			constexpr f32 eps = 1e-6f;
			return !(std::fabs(r[0][3]) < eps && std::fabs(r[1][3]) < eps &&
				std::fabs(r[2][3]) < eps && std::fabs(r[3][3] - 1.f) < eps);
		};

		// The position policy has a wider domain than the matrix policy. Locate a
		// perspective block first so c[465]'s offset follows the exact camera
		// right axis used by this draw (not a global axis or a stale prior draw).
		f32* rows[4] = {};
		bool have_perspective = false;
		for (const u32 candidate : { m_base, m_base + 4 })
		{
			f32* r[4];
			bool present = true;
			for (u32 k = 0; k < 4 && present; ++k)
			{
				r[k] = find_slot(buffer, reloc, reloc_size, candidate + k);
				present = r[k] != nullptr;
			}

			if (present && is_perspective(r))
			{
				for (u32 k = 0; k < 4; ++k) rows[k] = r[k];
				have_perspective = true;
				break;
			}
		}

		if (!have_perspective)
		{
			return false;
		}

		const auto& avconf = g_fxo->get<rsx::avconf>();
		const size2u eye = avconf.video_frame_size();
		if (!surface_w || !surface_h || !eye.width || !eye.height)
		{
			return false;
		}

		const f32 target_aspect = static_cast<f32>(surface_w) / surface_h;
		const f32 output_aspect = static_cast<f32>(eye.width) / eye.height;
		const bool output_aspect_match = std::fabs(target_aspect / output_aspect - 1.f) <= 0.02f;

		// c[465] is a camera-world point in the native oracle. Unlike the
		// matrix it changes on the cascade route too. A cascade's c[260] is a
		// perspective projection but its clip-X column is not camera right
		// (native capture dot=+0.007); use the most recent real camera view for
		// that global axis. Output-aspect camera draws establish/refresh it.
		// Head rotation first, so camera right and the eye offsets below follow
		// the rotated view.
		if (output_aspect_match && m_vr_view)
		{
			apply_vr_rotation(rows);
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

		if (f32* cam = find_slot(buffer, reloc, reloc_size, m_cam_slot); cam && m_render_camera_right_valid)
		{
			const f32 half_eye_baseline = 0.120002f * (m_vr_view ? m_vr_eye_scale : 1.f);
			cam[0] += eye_sign * half_eye_baseline * m_render_camera_right[0];
			cam[1] += eye_sign * half_eye_baseline * m_render_camera_right[1];
			cam[2] += eye_sign * half_eye_baseline * m_render_camera_right[2];
		}

		if (!output_aspect_match)
		{
			return false;
		}

		// Gate 4 fitted two resolution families. The half-resolution pass uses
		// 75% of the full-resolution shear, not 50%, so select the measured
		// value by target width. Infinity layers are deliberately left on the
		// converged path until their six program/pass keys are made profile data;
		// guessing them would turn a measured policy into a heuristic.
		const f32 per_eye_sep = surface_w * 2u == eye.width ? 0.03047f : 0.040625f;
		constexpr f32 convergence = 2.878f;
		const f32 sep = eye_sign * per_eye_sep;

		if (m_vr_view)
		{
			// Headset eyes are parallel: keep the formula's eye translation
			// (clip.x -= sep*conv, the same 0.120-unit offset as c[465]) and drop
			// its convergence image shift (clip.x += sep*clip.w).
			rows[3][0] -= sep * m_vr_eye_scale * convergence;

			if (m_vr_fov_scale != 1.f)
			{
				const f32 zoom = 1.f / m_vr_fov_scale;
				for (u32 r = 0; r < 4; ++r)
				{
					rows[r][0] *= zoom;
					rows[r][1] *= zoom;
				}
			}
			return true;
		}

		for (u32 r = 0; r < 4; ++r)
		{
			rows[r][0] += sep * rows[r][3];
		}
		rows[3][0] -= sep * convergence;
		return true;
	}

	void camera_probe::set_vr_view(const f32 q[4], f32 eye_scale, f32 fov_scale, bool flip_y)
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

		m_vr_eye_scale = eye_scale;
		m_vr_fov_scale = fov_scale;
		m_vr_view = true;
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

	void camera_probe::apply_vr_rotation(f32* const rows[4]) const
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
		const auto& R = m_vr_rot;

		for (u32 r = 0; r < 4; ++r)
		{
			const f32 u0 = rows[r][0] / A;
			const f32 u1 = rows[r][1] / B;
			const f32 u2 = rows[r][3];

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

		// Title gate: never perturb a title we have not profiled.
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
		if (m_require_cam && !find_slot(buffer, reloc, reloc_size, m_cam_slot))
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
			if (std::fabs(target_aspect / output_aspect - 1.f) > 0.02f)
			{
				if (find_slot(buffer, reloc, reloc_size, m_base))
				{
					m_stat_rejected_aspect++;
				}
				return;
			}
		}

		// Select the camera block: the first of {base, base+4} that is present in
		// this program and is a PERSPECTIVE matrix. Orthographic blocks (HUD,
		// shadow cascades, env maps) and affine world matrices are left alone.
		const auto is_perspective = [](f32* const r[4])
		{
			constexpr f32 eps = 1e-6f;
			return !(std::fabs(r[0][3]) < eps && std::fabs(r[1][3]) < eps &&
			         std::fabs(r[2][3]) < eps && std::fabs(r[3][3] - 1.f) < eps);
		};

		f32* rows[4] = {};
		bool found = false;
		for (const u32 candidate : { m_base, m_base + 4 })
		{
			f32* r[4];
			bool present = true;
			for (u32 k = 0; k < 4 && present; ++k)
			{
				r[k] = find_slot(buffer, reloc, reloc_size, candidate + k);
				present = r[k] != nullptr;
			}

			if (present && is_perspective(r))
			{
				for (u32 k = 0; k < 4; ++k) rows[k] = r[k];
				found = true;
				break;
			}
		}

		if (!found)
		{
			if (find_slot(buffer, reloc, reloc_size, m_base))
			{
				m_stat_rejected_no_perspective++;
			}
			return;
		}

		m_stat_perturbed++;

		// WipEout's native stereo shear, applied in clip space:
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
		if (guest_constants && m_cam_slot < 512)
		{
			const u32* bank = static_cast<const u32*>(guest_constants);
			const u32* c = bank + m_cam_slot * 4;
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
