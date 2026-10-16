#pragma once

// VR fork: profile boosted_targets render a surface at a scale above the renderer's (VKGSRender::vr_select_target_scale).
// A texture-cache gather that copies such a surface into a texture is sized by the largest scale among its surfaces,
// which each piece's copy is scaled by (texture_cache_helpers.h, two call sites).

#include "Emu/RSX/RSXThread.h"

namespace rsx::vr
{
	template <typename surface_store_list_type>
	static const rsx::surface_scaling_config_t& gather_scaling_config(const surface_store_list_type& fbos)
	{
		const rsx::surface_scaling_config_t* best = &rsx::get_current_renderer()->resolution_scaling_config;
		for (const auto& fbo : fbos)
		{
			const auto& config = fbo.surface->resolution_scaling_config;
			if (config.scale_percent > best->scale_percent || config.percent_y() > best->percent_y())
				best = &config;
		}
		return *best;
	}
}
