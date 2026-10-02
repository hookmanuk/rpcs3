#pragma once

// VR fork: declarations shared by VKGSRender.h and the VR translation units
// (VKGSRenderVR.cpp, VKGSRenderVRDev.cpp). Kept out of VKGSRender.h so that
// upstream's header carries two include lines for the whole fork.

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Capture/rsx_camera_probe.h"
#include "../gcm_enums.h"

namespace rsx
{
	struct context;
}

namespace vk
{
	// decode_rsx_state (VKGSRender.cpp): probe dev bit 0x100 turns depth EQUAL into LEQUAL and
	// bit 0x200 drops the stencil test, for draws that write colour (development toggles).
	rsx::comparison_function vr_dev_depth_func(rsx::comparison_function func, bool writes_color);
	bool vr_dev_skip_stencil_test(bool writes_color);
}
