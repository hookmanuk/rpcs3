#pragma once

#include "util/types.hpp"

namespace rsx
{
	struct surface_scaling_config_t;
}

// VR fork: headset-shaped eyes.
//
// Each eye shows the game's frame (its output aspect, 16:9) stretched over the headset's eye view, which is relatively
// taller: about 100 x 89 degrees, 2.1x the game's height in tangents but 1.45x its width. With one Resolution Scale for
// both axes the eye got ~1.45x the pixels per degree across that it got up and down, and thin far geometry seen edge-on
// (Sonic & All-Stars Racing Transformed's track bend, roofs) fell between the rows: stair-stepped edges and lines. The
// scale is split per axis so both get the same pixels per degree: the axis that had fewer keeps the configured scale
// and the other drops to match (fewer pixels for the same detail: 300% -> 150% x 300% on a 0.925:1 view).
// RPCS3's surfaces then carry a vertical scale (surface_scaling_config_t::scale_percent_y) next to the horizontal one.
namespace rsx::vr
{
	// The headset's rendered eye extents in tangents (right - left, up - down) and the game's output aspect, each frame
	// in the headset view: sets the vertical / horizontal scale ratio (1 when the eyes render the game's own FOV).
	void set_eye_shape(f32 tan_width, f32 tan_height, f32 output_aspect);
	// The rendered eye's shape (width / height in tangents) from the last set_eye_shape; 0 before the headset view.
	f32 eye_view_aspect();

	// The scale percents for x and y from the configured one: both the same until set_eye_shape has a ratio, or with
	// Video > VR > Headset Eye Shape off (dev: RPCS3_VR_EYE_SHAPE=0/1 forces it).
	u16 eye_shape_percent_x(u16 percent);
	u16 eye_shape_percent_y(u16 percent);

	// The fragment window position (WPOS) is gl_FragCoord over the resolution scale. Its context holds one scale (the
	// vertical one); with the axes scaled apart this carries the horizontal percent in wpos_bias[0] as
	// 4096 x percent + bias (bias 0 or -0.5: exact in a float), which get_wpos decodes (RSXFragmentPrologue.glsl, and the
	// interpreter). Nothing changes while both axes have the same scale.
	void encode_wpos_x(f32& wpos_bias_x, const rsx::surface_scaling_config_t& config, u32 window_height);
}
