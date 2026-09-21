#pragma once

// RSX camera probe (VR fork, Gate 4)
//
// Causal test rig for the camera seam that the Gate 3 inspector found by
// correlation. It perturbs ONLY the transient per-draw copy of the vertex
// transform constants; rsx::method_registers.transform_constants is never
// touched, so guest simulation, FIFO progression and savestates are unaffected
// and every effect is reversible by simply not setting the environment variable.
//
// Enable:  RPCS3_VR_PROBE="<key>=<value>,..."        one fixed probe, read once
//          RPCS3_VR_PROBE_FILE="<path>"              re-read at every frame end,
//                                                    so a whole probe series can
//                                                    run against ONE booted scene
//                                                    (removes scene variation as
//                                                    a confound). Empty/missing
//                                                    file => no perturbation.
//
//   base=256        base slot of the 4x4 world->clip matrix (c[base..base+3])
//   cam=465         slot holding camera world position (w==1), used as the
//                   pivot for rotations and to report the derived right axis
//
//   yaw=<deg>       rotate the world about the camera pivot (world Y)
//   pitch=<deg>     ... world X
//   roll=<deg>      ... world Z
//   tx,ty,tz=<f>    move the camera by this world-space offset
//
//   eye=<f>         move the camera along its OWN right axis, derived from the
//                   matrix itself. This is the probe that replicates what the
//                   game does natively between its two eyes.
//
//   slot=<n>,comp=<0..3>,add=<f>
//                   raw single-component add. Used for positive controls (a
//                   slot we believe is camera state) and negative controls (a
//                   slot we believe is not).
//
//   stereo=<sep>    apply WipEout's own clip-space stereo shear to the camera
//   conv=<c>        block:  clip.x += sep * (clip.w - conv).
//                   This is the exact per-eye transform the game applies in its
//                   native 3D mode (fitted over 641 draws, residual < 4e-6). It is
//                   a post-multiply in clip space, so it is independent of any
//                   world/object matrix folded into the block.
//
//   Draw classifier (always on for matrix probes). A draw is perturbed only if
//     1. its render target has the output aspect ratio (a camera view), and
//     2. it carries a PERSPECTIVE camera block (column 3 != (0,0,0,1)),
//        auto-selected per program: c[256..259] if that is perspective
//        (single-matrix route), else c[260..263] (two-matrix route, where
//        c[256..259] is an affine world matrix).
//   Scored against the native-3D oracle's own per-eye labels this is exact:
//   641 true positives, 0 false positives, 0 false negatives. Rule 2 alone
//   excludes the HUD (orthographic) but NOT the 512x256 cascade passes, which
//   carry a perspective c[260] block the game keeps shared between eyes; rule 1
//   is what excludes them (58 draws).
//
//   reqcam=0|1      additionally require the program to read the camera slot
//                   (default 0; superseded by the perspective test, which is
//                   strictly more precise - kept for experiments).
//
//   title=<id>      only act on this title id (default BCES00664)
//
//   render=1         enable the Gate 5 renderer interface. The renderer calls
//                    apply_render_eye() twice on two *new* host constant
//                    allocations; it never reuses this probe's in-place
//                    experiment path. eye_sign is -1 for the native left eye
//                    and +1 for the native right eye. This applies both
//                    camera policies measured in Gate 4: the per-eye clip-X
//                    shear and the c[465] camera-position offset.
//
// Convention: RSX vertex programs emit clip position as
//     o = v.x*c[base] + v.y*c[base+1] + v.z*c[base+2] + c[base+3]
// i.e. a row-vector convention where c[base..base+3] are the ROWS of the
// world->clip matrix M, and o = v * M. A camera delta D is therefore applied as
// M' = D * M. Column 0 of M's upper 3x3 is the world-space direction that maps
// to clip X, which is the camera right axis.

#include <string>
#include <array>

#include "util/types.hpp"
#include "util/atomic.hpp"

namespace rsx::vr
{
	class camera_probe
	{
	public:
		static camera_probe& get();

		// Hot-path gate. False unless a perturbation is currently configured.
		bool enabled() const { return m_active.load(); }
		bool render_enabled() const;

		// Frame boundary: re-read RPCS3_VR_PROBE_FILE if it changed.
		void poll();

		// Perturb a freshly filled transient constant buffer in place.
		// reloc_table_data/size describe the original guest index of each 16-byte
		// slot in the buffer; an empty table means the buffer holds the full
		// 468-slot bank at its natural indices.
		// guest_constants points at the pristine 512x u32[4] guest bank.
		// surface_w/h: the draw's render-target size, used by the draw classifier.
		void apply(void* buffer, const u16* reloc_table_data, usz reloc_table_size,
			const void* guest_constants, u16 surface_w, u16 surface_h) const;

		// Apply WipEout's profiled native-eye transform to a freshly cloned
		// constant buffer. Returns true only when this is a perspective,
		// output-aspect world draw and its camera matrix was sheared. c[465]
		// is adjusted independently whenever the draw carries both it and a
		// usable perspective camera block (including the shared shadow route).
		bool apply_render_eye(void* buffer, const u16* reloc_table_data, usz reloc_table_size,
			u16 surface_w, u16 surface_h, f32 eye_sign) const;

		// Human-readable description of the active probe, for logging.
		const std::string& description() const { return m_description; }

	private:
		camera_probe();
		void parse(const std::string& cfg);
		void reset_params();

		bool m_enabled = false;          // subsystem on (default render path or probe config)
		atomic_t<bool> m_active{false};  // a perturbation is configured right now

		std::string m_config_path;
		u64 m_config_stamp = 0;

		std::string m_description;
		std::string m_title;

		u32 m_base = 256;
		u32 m_cam_slot = 465;

		f32 m_yaw = 0.f, m_pitch = 0.f, m_roll = 0.f;
		f32 m_tx = 0.f, m_ty = 0.f, m_tz = 0.f;
		f32 m_eye = 0.f;
		f32 m_stereo_sep = 0.f;
		f32 m_stereo_conv = 0.f;
		bool m_have_stereo = false;
		bool m_render_enabled = false;

		// c[465] is global camera state. Shadow cascades carry a perspective
		// c[260] block whose clip-X axis is *not* camera right, so retain the
		// most recent output-aspect camera-right axis for their c[465] policy.
		// RSX draw processing is serialized on the renderer thread.
		mutable std::array<f32, 3> m_render_camera_right{};
		mutable bool m_render_camera_right_valid = false;

		bool m_have_xform = false;
		bool m_require_cam = false;

		// One-frame classifier report, logged on the first full frame after arming.
		mutable atomic_t<u32> m_stat_perturbed{0};
		mutable atomic_t<u32> m_stat_rejected_aspect{0};
		mutable atomic_t<u32> m_stat_rejected_no_perspective{0};
		bool m_report_pending = false;

		// Raw single-component probe
		bool m_have_raw = false;
		u32 m_raw_slot = 0;
		u32 m_raw_comp = 0;
		f32 m_raw_add = 0.f;
	};
}
