#include "stdafx.h"
#include "rsx_vr_eye_shape.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/system_config.h"
#include "util/atomic.hpp"

#include <cmath>
#include <cstdlib>

LOG_CHANNEL(vr_eye_log, "VREYE");

namespace rsx::vr
{
	namespace
	{
		// Vertical / horizontal scale ratio for equal pixels per degree; 0 until the headset view has been seen.
		atomic_t<f32> g_eye_shape_ratio{0.f};

		// Percents in steps of 25, as RPCS3's own Resolution Scale: scaled sizes of guest surfaces (multiples of 4) stay
		// whole. At 251% x 358% a 320x180 target was 803 wide and its 160x90 half 401: the blit engine's halving asked
		// for 402 and every such blit was refused (Sonic's bloom chain).
		u16 snap(f32 percent, u16 max)
		{
			return static_cast<u16>(std::clamp(std::lround(percent / 25.f) * 25l, 25l, static_cast<long>(max)));
		}

		bool eye_shape_enabled()
		{
			static const int s_env = []
			{
				const char* v = std::getenv("RPCS3_VR_EYE_SHAPE");
				return v ? (v[0] == '1' ? 1 : 0) : -1;
			}();
			return s_env < 0 ? g_cfg.video.vr.eye_shape.get() : s_env == 1;
		}

		// The ratio in use, or 1 (same scale for both axes).
		f32 active_ratio()
		{
			const f32 ratio = g_eye_shape_ratio.load();
			return ratio > 0.f && eye_shape_enabled() ? ratio : 1.f;
		}
	}

	void set_eye_shape(f32 tan_width, f32 tan_height, f32 output_aspect)
	{
		if (!(tan_width > 0.f) || !(tan_height > 0.f) || !(output_aspect > 0.f))
		{
			return;
		}
		// Pixels per tangent across: width_px / tan_width; down: height_px / tan_height, with height_px / width_px the
		// output's 1 / aspect at equal scales. Equal density needs scale_y / scale_x = aspect * tan_height / tan_width.
		f32 ratio = std::clamp(output_aspect * tan_height / tan_width, 0.25f, 4.f);
		// Within 1%: the same (the eyes' extents move a little with the reprojection margin and pose prediction; every
		// change rebuilds the scaled surfaces).
		if (std::fabs(ratio - 1.f) < 0.01f)
		{
			ratio = 1.f;
		}
		const f32 current = g_eye_shape_ratio.load();
		if (current > 0.f && std::fabs(ratio / current - 1.f) < 0.01f)
		{
			return;
		}
		g_eye_shape_ratio = ratio;
		vr_eye_log.notice("Headset eye shape: %.3f x %.3f tangents at output aspect %.3f: vertical / horizontal scale %.3f%s", tan_width,
			tan_height, output_aspect, ratio, eye_shape_enabled() ? "" : " (Headset Eye Shape off: not applied)");
	}

	u16 eye_shape_percent_x(u16 percent)
	{
		const f32 ratio = active_ratio();
		if (ratio == 1.f)
		{
			return percent;
		}
		// Same pixel count: x / sqrt(ratio), y * sqrt(ratio). Never exactly 100 while y differs: RPCS3 takes a 100%
		// horizontal scale for an unscaled surface in places.
		const f32 x = percent / std::sqrt(ratio);
		const u16 snapped = snap(x, 800);
		return snapped == 100 && eye_shape_percent_y(percent) != 100 ? (x < 100.f ? 75 : 125) : snapped;
	}

	u16 eye_shape_percent_y(u16 percent)
	{
		const f32 ratio = active_ratio();
		if (ratio == 1.f)
		{
			return percent;
		}
		return snap(percent * std::sqrt(ratio), 1600);
	}

	void encode_wpos_x(f32& wpos_bias_x, const rsx::surface_scaling_config_t& config, u32 window_height)
	{
		if (config.percent_y() == config.scale_percent || window_height <= config.min_scalable_dimension)
		{
			return;
		}
		wpos_bias_x += 4096.f * config.scale_percent;
	}
}
