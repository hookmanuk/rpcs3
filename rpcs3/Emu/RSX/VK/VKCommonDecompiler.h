#pragma once
#include "../Program/GLSLTypes.h"

namespace vk
{
	// VR fork: the varying of RSX_SHADER_CONTROL_VR_EXACT_DEPTH, after the RSX's (0-15)
	constexpr int vr_exact_depth_location = 16;

	using namespace ::glsl;

	int get_varying_register_location(std::string_view varying_register_name);

	int get_texture_index(std::string_view name);
}
