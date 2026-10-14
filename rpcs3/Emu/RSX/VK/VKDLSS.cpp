#include "stdafx.h"
#include "VKDLSS.h"
#include "VKFrameGen.h"

#include "VKCompute.h"
#include "VKFramebuffer.h"
#include "VKHelpers.h"
#include "VKPipelineCompiler.h"
#include "VKRenderPass.h"
#include "VKResourceManager.h"
#include "vkutils/barriers.h"
#include "vkutils/device.h"
#include "vkutils/image.h"
#include "vkutils/image_helpers.h"
#include "vkutils/sampler.h"

#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/Capture/rsx_camera_probe.h"
#include "Emu/RSX/Capture/rsx_vr_hooks.h" // RSX_SHADER_CONTROL_DLSS_MOTION
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "Utilities/StrFmt.h"

#include <set>

#ifdef HAVE_DLSS
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>

LOG_CHANNEL(dlss_log, "DLSS");

namespace vk
{
	namespace
	{
		const char* env(const char* name)
		{
			const char* v = std::getenv(name);
			return (v && *v) ? v : nullptr;
		}

		f32 env_f32(const char* name, f32 def)
		{
			const char* v = env(name);
			return v ? static_cast<f32>(std::atof(v)) : def;
		}

		// "<x sign><y sign>", e.g. "+-": the multipliers for x and y.
		std::array<f32, 2> env_signs(const char* name)
		{
			std::array<f32, 2> s{1.f, 1.f};
			if (const char* v = env(name))
			{
				for (int i = 0; i < 2 && v[i]; ++i)
				{
					s[i] = v[i] == '-' ? -1.f : 1.f;
				}
			}
			return s;
		}

		// The varyings after the RSX's (0-15) and the VR fork's (16 exact depth, 17-22 depth remap).
		constexpr int dlss_cur_location = 23; // after the depth remap's 6 (17 to 22)
		constexpr int dlss_prev_location = 24;
		constexpr int dlss_cover_location = 25;

		f32 halton(u32 index, u32 base)
		{
			f32 f = 1.f, r = 0.f;
			while (index > 0)
			{
				f /= static_cast<f32>(base);
				r += f * static_cast<f32>(index % base);
				index /= base;
			}
			return r;
		}

		bool invert4(const f32 m[16], f32 out[16])
		{
			f32 inv[16];
			inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
			inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
			inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
			inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
			inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
			inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
			inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
			inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
			inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
			inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
			inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
			inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
			inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
			inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
			inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
			inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

			const f64 det = static_cast<f64>(m[0]) * inv[0] + static_cast<f64>(m[1]) * inv[4] + static_cast<f64>(m[2]) * inv[8] + static_cast<f64>(m[3]) * inv[12];
			if (!std::isfinite(det) || std::abs(det) < 1e-30)
			{
				return false;
			}

			const f32 inv_det = static_cast<f32>(1.0 / det);
			for (int i = 0; i < 16; ++i)
			{
				out[i] = inv[i] * inv_det;
			}
			return true;
		}

		void mul4(const f32 a[16], const f32 b[16], f32 out[16])
		{
			for (int r = 0; r < 4; ++r)
			{
				for (int c = 0; c < 4; ++c)
				{
					out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c] + a[r * 4 + 3] * b[3 * 4 + c];
				}
			}
		}

		constexpr f32 identity4[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};

		// The same motion within float rounding: the main view's draws, whose motions differ by rounding (up to ~1e-3) during a head
		// turn. At near_equal's 1e-4 they split into several votes, and a pass with its own camera (Gran Turismo 5's rear-view
		// mirror, ~290 draws without head motion) won: every draw falling back on the camera's motion lost the head turn, and
		// GT5's cars smeared in frame generation in proportion to head speed.
		bool same_motion(const f32 a[16], const f32 b[16])
		{
			for (int i = 0; i < 16; ++i)
			{
				if (std::abs(a[i] - b[i]) > 2e-3f * std::max(1.f, std::abs(a[i])))
				{
					return false;
				}
			}
			return true;
		}

		bool near_equal(const f32 a[16], const f32 b[16])
		{
			for (int i = 0; i < 16; ++i)
			{
				if (std::abs(a[i] - b[i]) > 1e-4f * std::max(1.f, std::abs(a[i])))
				{
					return false;
				}
			}
			return true;
		}

		u64 mix(u64 h, u64 v)
		{
			h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			return h;
		}

		// A camera block as the profile describes it.
		struct camera_block_desc
		{
			u32 slots[4];
			bool columns; // clip[i] = dot(c[slot i], v)
		};

		std::vector<camera_block_desc> camera_blocks_now()
		{
			std::vector<camera_block_desc> result;

			if (const char* v = env("RPCS3_DLSS_CAMERA"))
			{
				const u32 base = static_cast<u32>(std::strtoul(v, nullptr, 0));
				const bool rows = std::strstr(v, "row") != nullptr;
				result.push_back({{base, base + 1, base + 2, base + 3}, !rows});
				return result;
			}

			const auto* profile = rsx::vr::camera_probe::get().profile();
			if (!profile || profile->xyw_rows)
			{
				return result;
			}

			for (usz i = 0; i < profile->camera_blocks.size(); ++i)
			{
				const u32 base = profile->camera_blocks[i];
				camera_block_desc d{{base, base + 1, base + 2, base + 3}, profile->column_vectors};
				if (i < profile->camera_block_slots.size() && profile->camera_block_slots[i][0] != umax)
				{
					for (int s = 0; s < 4; ++s)
					{
						d.slots[s] = profile->camera_block_slots[i][s];
					}
				}
				if (std::find(profile->row_vector_blocks.begin(), profile->row_vector_blocks.end(), base) != profile->row_vector_blocks.end())
				{
					d.columns = !d.columns;
				}
				result.push_back(d);
			}
			return result;
		}

		// The resolve: motion image (sample 0) -> motion vectors in render pixels, depth, and the debug view.
		template <bool MSAA>
		struct dlss_resolve_pass : public compute_task
		{
			const vk::image_view* m_in = nullptr;
			const vk::image_view* m_mv = nullptr;
			const vk::image_view* m_depth = nullptr;
			const vk::image_view* m_debug = nullptr;
			std::unique_ptr<vk::sampler> m_sampler;

			dlss_resolve_pass()
			{
				m_src = std::string(
					"#version 450\n"
					"layout(local_size_x = 16, local_size_y = 16) in;\n") +
					(MSAA ? "layout(set=0, binding=0) uniform sampler2DMS motion_src;\n" : "layout(set=0, binding=0) uniform sampler2D motion_src;\n") +
					"layout(set=0, binding=1, rg32f) uniform writeonly image2D mv_out;\n"
					"layout(set=0, binding=2, r32f) uniform writeonly image2D depth_out;\n"
					"layout(set=0, binding=3, rgba8) uniform writeonly image2D debug_out;\n"
					"layout(set=0, binding=4) uniform sampler2D prev2_color;\n"
					"layout(set=0, binding=5) uniform sampler2D prev_mv_tex;\n"
					"layout(push_constant) uniform pc_block\n"
					"{\n"
					"	vec4 sizes;  // source region w, h; output w, h\n"
					"	vec4 params; // mv sign x, y; debug mode (1 motion, 2 depth); 1: the camera's motion below is valid\n"
					"	vec4 vp;     // the scene's viewport transform: x scale, x offset, y scale, y offset\n"
					"	vec4 r0, r1, r2, r3; // the camera's R (rows), game clip space: this frame -> the previous frame\n"
					"};\n"
					"vec3 hue(float h)\n"
					"{\n"
					"	return clamp(abs(fract(h + vec3(0., 2. / 3., 1. / 3.)) * 6. - 3.) - 1., 0., 1.);\n"
					"}\n"
					"void main()\n"
					"{\n"
					"	const ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
					"	if (p.x >= int(sizes.z) || p.y >= int(sizes.w)) return;\n"
					"	const vec2 uv = (vec2(p) + 0.5) / sizes.zw;\n"
					"	const ivec2 sp = clamp(ivec2(uv * sizes.xy), ivec2(0), ivec2(sizes.xy) - 1);\n"
					"	const vec4 m = texelFetch(motion_src, sp, 0);\n"
					"	const bool covered = m.w > 0.;\n"
					"	// Not covered (the sky, drawn without depth): the camera's motion at the far plane (VR head turns move it).\n"
					"	vec2 cam = vec2(0.);\n"
					"	if (!covered && params.w > 0.5)\n"
					"	{\n"
					"		const vec2 ndc = (vec2(sp) + 0.5) / sizes.xy * 2. - 1.;\n"
					"		const vec4 g = vec4((ndc.x - vp.y) / vp.x, (ndc.y - vp.w) / vp.z, 1., 1.);\n"
					"		const vec4 pg = vec4(dot(r0, g), dot(r1, g), dot(r2, g), dot(r3, g));\n"
					"		if (pg.w > 1e-6) cam = (vec2(vp.x * pg.x / pg.w + vp.y, vp.z * pg.y / pg.w + vp.w) - ndc) * 0.5;\n"
					"	}\n"
					"	vec2 mv = (covered ? m.xy : cam) * sizes.zw * params.xy;\n"
					"	float d = covered ? 1. - m.z : 1.;\n"
					"	// Never NaN, infinite or absurd (a degenerate matrix in a draw's R): NGX keeps such a pixel in its history and\n"
					"	// spreads it into black blocks (Matt, headset, MotorStorm paused with the head moving).\n"
					"	if (!(abs(mv.x) <= sizes.z * 4.) || !(abs(mv.y) <= sizes.w * 4.)) mv = vec2(0.);\n"
					"	d = (d >= 0. && d <= 1.) ? d : 1.;\n"
					"	imageStore(mv_out, p, vec4(mv, 0., 0.));\n"
					"	imageStore(depth_out, p, vec4(d));\n"
					"	if (params.z == 1.)\n"
					"	{\n"
					"		const float len = length(mv);\n"
					"		vec3 c = covered ? hue(atan(mv.y, mv.x) / 6.2831853 + 0.5) * clamp(len / 16., 0.15, 1.) : len > 0.05 ? hue(atan(mv.y, mv.x) / 6.2831853 + 0.5) * clamp(len / 16., 0.15, 1.) * 0.6 : vec3(0.25);\n"
					"		if (covered && len < 0.05) c = vec3(0.05);\n"
					"		imageStore(debug_out, p, vec4(c, 1.));\n"
					"	}\n"
					"	else if (params.z == 3.)\n"
					"	{\n"
					"		// Numeric: 128 + 4 x motion (pixels) in red and green, coverage in blue (self-test decoding)\n"
					"		imageStore(debug_out, p, vec4(clamp((128. + mv * 4.) / 255., 0., 1.), covered ? 1. : 0., 1.));\n"
					"	}\n"
					"	else if (params.z == 2.)\n"
					"	{\n"
					"		const float v = covered ? pow(clamp(1. - d, 0., 1.), 0.25) : 0.;\n"
					"		imageStore(debug_out, p, vec4(v, covered ? v : 0.2, v, 1.));\n"
					"	}\n"
					"}\n";
				ssbo_count = 0;
				use_push_constants = true;
				push_constants_size = 112;
				create();
			}

			std::vector<glsl::program_input> get_inputs() override
			{
				auto result = compute_task::get_inputs();
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "motion_src", vk::glsl::input_type_texture, 0, 0));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "mv_out", vk::glsl::input_type_storage_texture, 0, 1));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "depth_out", vk::glsl::input_type_storage_texture, 0, 2));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "debug_out", vk::glsl::input_type_storage_texture, 0, 3));
				return result;
			}

			void bind_resources(const vk::command_buffer& /*cmd*/) override
			{
				if (!m_sampler)
				{
					m_sampler = std::make_unique<vk::sampler>(*vk::get_current_renderer(),
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
				}

				m_program->bind_uniform({*m_in, *m_sampler, VK_IMAGE_LAYOUT_GENERAL}, 0, 0);
				m_program->bind_uniform({*m_mv}, 0, 1);
				m_program->bind_uniform({*m_depth}, 0, 2);
				m_program->bind_uniform({*m_debug}, 0, 3);
			}

			void run(const vk::command_buffer& cmd, const vk::image_view* src, const vk::image_view* mv, const vk::image_view* depth, const vk::image_view* debug,
				const f32 constants[28])
			{
				m_in = src;
				m_mv = mv;
				m_depth = depth;
				m_debug = debug;

				if (vk::is_renderpass_open(cmd))
				{
					vk::end_renderpass(cmd);
				}

				load_program(cmd);
				vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 112, constants);
				vkCmdDispatch(cmd, utils::aligned_div(static_cast<u32>(constants[2]), 16u), utils::aligned_div(static_cast<u32>(constants[3]), 16u), 1);
			}
		};

		// Frame generation (VKFrameGen.h): the game's motion vectors where they describe a static point, "invalid" elsewhere.
		// A static point at the previous position (p + mv) had, last frame, the view depth R predicts (k = w' / w, from the
		// draw: the motion image's w); a point that moved by itself (MotorStorm's car, built on the SPU: its vectors are the
		// world's) finds something at another depth there. Depth ratios from window depth: 1 - z is about n / w.
		struct dlss_framegen_check_pass : public compute_task
		{
			const vk::image_view* m_motion = nullptr;
			const vk::image_view* m_mv = nullptr;
			const vk::image_view* m_depth = nullptr;
			const vk::image_view* m_prev_depth = nullptr;
			const vk::image_view* m_out = nullptr;
			std::unique_ptr<vk::sampler> m_nearest;

			dlss_framegen_check_pass()
			{
				m_src =
					"#version 450\n"
					"layout(local_size_x = 16, local_size_y = 16) in;\n"
					"layout(set=0, binding=0) uniform sampler2D motion_src;\n"
					"layout(set=0, binding=1) uniform sampler2D mv_tex;\n"
					"layout(set=0, binding=2) uniform sampler2D depth_tex;\n"
					"layout(set=0, binding=3) uniform sampler2D prev_depth_tex;\n"
					"layout(set=0, binding=4, rg32f) uniform writeonly image2D mv_out;\n"
					"layout(push_constant) uniform pc_block\n"
					"{\n"
					"	vec4 sizes;  // motion image region w, h; output w, h\n"
					"	vec4 params; // tolerance, 1: the previous depth is valid, invalid value\n"
					"};\n"
					"void main()\n"
					"{\n"
					"	const ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
					"	if (p.x >= int(sizes.z) || p.y >= int(sizes.w)) return;\n"
					"	const vec2 uv = (vec2(p) + 0.5) / sizes.zw;\n"
					"	const ivec2 sp = clamp(ivec2(uv * sizes.xy), ivec2(0), ivec2(sizes.xy) - 1);\n"
					"	const float k = texelFetch(motion_src, sp, 0).w;\n"
					"	vec2 mv = texelFetch(mv_tex, p, 0).xy;\n"
					"	const float d = texelFetch(depth_tex, p, 0).x;\n"
					"	const ivec2 q = ivec2(floor(vec2(p) + 0.5 + mv));\n"
					"	if (params.y > 0.5 && k > 0. && d < 0.9999 && all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, ivec2(sizes.zw))))\n"
					"	{\n"
					"		const float dp = texelFetch(prev_depth_tex, q, 0).x;\n"
					"		const float found = (1. - d) / max(1. - dp, 1e-9); // view depth there last frame over this pixel's\n"
					"		if (dp >= 0.9999 || abs(found / k - 1.) > params.x) mv = vec2(params.z);\n"
					"	}\n"
					"	imageStore(mv_out, p, vec4(mv, 0., 0.));\n"
					"}\n";
				ssbo_count = 0;
				use_push_constants = true;
				push_constants_size = 32;
				create();
			}

			std::vector<glsl::program_input> get_inputs() override
			{
				auto result = compute_task::get_inputs();
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "motion_src", vk::glsl::input_type_texture, 0, 0));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "mv_tex", vk::glsl::input_type_texture, 0, 1));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "depth_tex", vk::glsl::input_type_texture, 0, 2));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "prev_depth_tex", vk::glsl::input_type_texture, 0, 3));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "mv_out", vk::glsl::input_type_storage_texture, 0, 4));
				return result;
			}

			void bind_resources(const vk::command_buffer& /*cmd*/) override
			{
				if (!m_nearest)
				{
					m_nearest = std::make_unique<vk::sampler>(*vk::get_current_renderer(),
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
				}
				m_program->bind_uniform({*m_motion, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 0);
				m_program->bind_uniform({*m_mv, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 1);
				m_program->bind_uniform({*m_depth, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 2);
				m_program->bind_uniform({*m_prev_depth, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 3);
				m_program->bind_uniform({*m_out}, 0, 4);
			}

			void run(const vk::command_buffer& cmd, const vk::image_view* motion, const vk::image_view* mv, const vk::image_view* depth,
				const vk::image_view* prev_depth, const vk::image_view* out, const f32 constants[8])
			{
				m_motion = motion;
				m_mv = mv;
				m_depth = depth;
				m_prev_depth = prev_depth;
				m_out = out;
				if (vk::is_renderpass_open(cmd))
				{
					vk::end_renderpass(cmd);
				}
				load_program(cmd);
				vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, constants);
				vkCmdDispatch(cmd, utils::aligned_div(static_cast<u32>(constants[2]), 16u), utils::aligned_div(static_cast<u32>(constants[3]), 16u), 1);
			}
		};

		// debug_warp: the previous input sampled through this frame's motion vectors (render pixels, this frame -> the
		// previous), against this input. Red: what is left after the warp (wrong motion vectors light up); green: unwarped.
		struct dlss_warp_check_pass : public compute_task
		{
			const vk::image_view* m_mv = nullptr;
			const vk::image_view* m_cur = nullptr;
			const vk::image_view* m_prev = nullptr;
			const vk::image_view* m_out = nullptr;
			const vk::image_view* m_prev2 = nullptr;
			const vk::image_view* m_prev_mv = nullptr;
			std::unique_ptr<vk::sampler> m_nearest, m_linear;

			dlss_warp_check_pass()
			{
				m_src =
					"#version 450\n"
					"layout(local_size_x = 16, local_size_y = 16) in;\n"
					"layout(set=0, binding=0) uniform sampler2D mv_tex;\n"
					"layout(set=0, binding=1) uniform sampler2D cur_color;\n"
					"layout(set=0, binding=2) uniform sampler2D prev_color;\n"
					"layout(set=0, binding=3, rgba8) uniform writeonly image2D debug_out;\n"
					"layout(set=0, binding=4) uniform sampler2D prev2_color;\n"
					"layout(set=0, binding=5) uniform sampler2D prev_mv_tex;\n"
					"layout(push_constant) uniform pc_block { vec4 sizes; }; // w, h, gain, 1: green unwarped\n"
					"void main()\n"
					"{\n"
					"	const ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
					"	if (p.x >= int(sizes.x) || p.y >= int(sizes.y)) return;\n"
					"	const vec2 mv = texelFetch(mv_tex, p, 0).xy;\n"
					"	const vec3 c = texelFetch(cur_color, p, 0).rgb;\n"
					"	const vec3 q = texture(prev_color, (vec2(p) + 0.5 + mv) / sizes.xy).rgb;\n"
					"	// Timing: red, this frame's vectors on this pair; green, on the pair before (vectors a frame late);\n"
					"	// blue, the previous frame's vectors on this pair (vectors a frame early). RPCS3_DLSS_WARP_PLAIN=1: green unwarped.\n"
					"	const vec3 pr = texelFetch(prev_color, p, 0).rgb;\n"
					"	const vec3 g2 = texture(prev2_color, (vec2(p) + 0.5 + mv) / sizes.xy).rgb;\n"
					"	const vec2 mv1 = texelFetch(prev_mv_tex, p, 0).xy;\n"
					"	const vec3 b1 = texture(prev_color, (vec2(p) + 0.5 + mv1) / sizes.xy).rgb;\n"
					"	const float green = sizes.w > 0.5 ? length(c - pr) : length(pr - g2);\n"
					"	imageStore(debug_out, p, vec4(clamp(length(c - q) * sizes.z, 0., 1.), clamp(green * sizes.z, 0., 1.), clamp(length(c - b1) * sizes.z, 0., 1.), 1.));\n"
					"}\n";
				ssbo_count = 0;
				use_push_constants = true;
				push_constants_size = 16;
				create();
			}

			std::vector<glsl::program_input> get_inputs() override
			{
				auto result = compute_task::get_inputs();
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "mv_tex", vk::glsl::input_type_texture, 0, 0));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "cur_color", vk::glsl::input_type_texture, 0, 1));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "prev_color", vk::glsl::input_type_texture, 0, 2));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "debug_out", vk::glsl::input_type_storage_texture, 0, 3));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "prev2_color", vk::glsl::input_type_texture, 0, 4));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "prev_mv_tex", vk::glsl::input_type_texture, 0, 5));
				return result;
			}

			void bind_resources(const vk::command_buffer& /*cmd*/) override
			{
				if (!m_nearest)
				{
					m_nearest = std::make_unique<vk::sampler>(*vk::get_current_renderer(),
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
					m_linear = std::make_unique<vk::sampler>(*vk::get_current_renderer(),
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
				}
				m_program->bind_uniform({*m_mv, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 0);
				m_program->bind_uniform({*m_cur, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 1);
				m_program->bind_uniform({*m_prev, *m_linear, VK_IMAGE_LAYOUT_GENERAL}, 0, 2);
				m_program->bind_uniform({*m_out}, 0, 3);
				m_program->bind_uniform({*m_prev2, *m_linear, VK_IMAGE_LAYOUT_GENERAL}, 0, 4);
				m_program->bind_uniform({*m_prev_mv, *m_nearest, VK_IMAGE_LAYOUT_GENERAL}, 0, 5);
			}

			void run(const vk::command_buffer& cmd, const vk::image_view* mv, const vk::image_view* cur, const vk::image_view* prev, const vk::image_view* out,
				const vk::image_view* prev2, const vk::image_view* prev_mv, u32 width, u32 height, f32 gain, f32 plain)
			{
				m_prev2 = prev2;
				m_prev_mv = prev_mv;
				m_mv = mv;
				m_cur = cur;
				m_prev = prev;
				m_out = out;
				if (vk::is_renderpass_open(cmd))
				{
					vk::end_renderpass(cmd);
				}
				load_program(cmd);
				const f32 pc[4] = {static_cast<f32>(width), static_cast<f32>(height), gain, plain};
				vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
				vkCmdDispatch(cmd, utils::aligned_div(width, 16u), utils::aligned_div(height, 16u), 1);
			}
		};

		// The HUD on top of DLSS's output: out += final - pre, both sampled bilinearly over the input region.
		struct dlss_hud_composite_pass : public compute_task
		{
			const vk::image_view* m_final = nullptr;
			const vk::image_view* m_pre = nullptr;
			const vk::image_view* m_out = nullptr;
			std::unique_ptr<vk::sampler> m_linear;

			dlss_hud_composite_pass()
			{
				m_src =
					"#version 450\n"
					"layout(local_size_x = 16, local_size_y = 16) in;\n"
					"layout(set=0, binding=0) uniform sampler2D final_tex;\n"
					"layout(set=0, binding=1) uniform sampler2D pre_tex;\n"
					"layout(set=0, binding=2, rgba8) uniform image2D out_img;\n"
					"layout(push_constant) uniform pc_block { vec4 sizes; vec4 scales; vec4 h0, h1, h2; }; // out w, h; final u, v; pre u, v (region / texture); gain\n"
					"void main()\n"
					"{\n"
					"	const ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
					"	if (p.x >= int(sizes.x) || p.y >= int(sizes.y)) return;\n"
					"	// h (rows): this output's uv to the HUD's frame (frame generation: a generated frame at another head pose).\n"
					"	const vec3 q = vec3(dot(h0.xyz, vec3((vec2(p) + 0.5) / sizes.xy, 1.)), dot(h1.xyz, vec3((vec2(p) + 0.5) / sizes.xy, 1.)), dot(h2.xyz, vec3((vec2(p) + 0.5) / sizes.xy, 1.)));\n"
					"	if (q.z <= 1e-6) return;\n"
					"	const vec2 uv = q.xy / q.z;\n"
					"	if (any(lessThan(uv, vec2(0.))) || any(greaterThan(uv, vec2(1.)))) return;\n"
					"	const vec3 f = texture(final_tex, uv * sizes.zw).rgb;\n"
					"	const vec3 b = texture(pre_tex, uv * scales.xy).rgb;\n"
					"	const vec3 d = abs(f - b);\n"
					"	if (all(lessThan(d, vec3(0.5 / 255.)))) return;\n"
					"	// final = pre (1 - a) + hud a, so the HUD over DLSS's output is final + (1 - a) (out - pre). The HUD's\n"
					"	// coverage a is not known: estimated from how much it changed the pixel (any clear change: all of it).\n"
					"	const float a = clamp(max(d.r, max(d.g, d.b)) * scales.z, 0., 1.);\n"
					"	const vec4 o = imageLoad(out_img, p);\n"
					"	imageStore(out_img, p, vec4(clamp(f + (1. - a) * (o.rgb - b), 0., 1.), o.a));\n"
					"}\n";
				ssbo_count = 0;
				use_push_constants = true;
				push_constants_size = 80;
				create();
			}

			std::vector<glsl::program_input> get_inputs() override
			{
				auto result = compute_task::get_inputs();
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "final_tex", vk::glsl::input_type_texture, 0, 0));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "pre_tex", vk::glsl::input_type_texture, 0, 1));
				result.push_back(glsl::program_input::make(::glsl::program_domain::glsl_compute_program, "out_img", vk::glsl::input_type_storage_texture, 0, 2));
				return result;
			}

			void bind_resources(const vk::command_buffer& /*cmd*/) override
			{
				if (!m_linear)
				{
					m_linear = std::make_unique<vk::sampler>(*vk::get_current_renderer(),
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
				}
				m_program->bind_uniform({*m_final, *m_linear, VK_IMAGE_LAYOUT_GENERAL}, 0, 0);
				m_program->bind_uniform({*m_pre, *m_linear, VK_IMAGE_LAYOUT_GENERAL}, 0, 1);
				m_program->bind_uniform({*m_out}, 0, 2);
			}

			void run(const vk::command_buffer& cmd, const vk::image_view* final_view, const vk::image_view* pre, const vk::image_view* out, const f32 pc[20])
			{
				m_final = final_view;
				m_pre = pre;
				m_out = out;
				if (vk::is_renderpass_open(cmd))
				{
					vk::end_renderpass(cmd);
				}
				load_program(cmd);
				vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 80, pc);
				vkCmdDispatch(cmd, utils::aligned_div(static_cast<u32>(pc[0]), 16u), utils::aligned_div(static_cast<u32>(pc[1]), 16u), 1);
			}
		};

		// The motion image drops its framebuffers when it is finally destroyed (like a render target).
		class dlss_motion_image : public vk::viewable_image
		{
		public:
			using viewable_image::viewable_image;

			~dlss_motion_image() override
			{
				if (value)
				{
					vk::remove_framebuffers_with_image(this);
				}
			}
		};

		std::unique_ptr<vk::viewable_image> make_image(const vk::render_device& dev, VkFormat format, u32 w, u32 h, VkSampleCountFlagBits samples, VkImageUsageFlags usage)
		{
			return std::make_unique<vk::viewable_image>(dev, dev.get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D, format, w, h, 1, 1, 1, samples, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL, usage,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3, VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_COLOR);
		}

		bool extension_listed(const std::vector<VkExtensionProperties>& list, const char* name)
		{
			return std::any_of(list.begin(), list.end(), [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
		}

		void add_extension(std::vector<const char*>& extensions, const char* name)
		{
			if (std::none_of(extensions.begin(), extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; }))
			{
				extensions.push_back(name);
			}
		}

#ifdef HAVE_DLSS
		const char* ngx_result_name(NVSDK_NGX_Result r)
		{
			static thread_local std::string s;
			s = fmt::format("0x%08x", static_cast<u32>(r));
			return s.c_str();
		}
#endif
	} // namespace

	namespace
	{
		atomic_t<u64> s_drawn_frames{0};    // game frames that had camera draws
		u32 s_rgba32f_samples = 0, s_rgba16f_samples = 0; // sample counts each motion format supports (dlss_context)
		bool s_test_record = false;                           // dlss_test_record()
		std::mutex s_test_mutex;
		std::vector<dlss_test_draw> s_test_presented;         // dlss_test_presented_draws()
		atomic_t<u64> s_presented_frame{0}; // which of them the last presented output shows
	}

	u64 dlss_drawn_frames() { return s_drawn_frames.load(); }

	void dlss_test_record(bool enable)
	{
		s_test_record = enable;
	}

	std::vector<dlss_test_draw> dlss_test_presented_draws()
	{
		std::lock_guard lock(s_test_mutex);
		return s_test_presented;
	}

	VkFormat dlss_motion_format(u32 samples)
	{
		return (s_rgba32f_samples & samples) || !(s_rgba16f_samples & samples) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
	}
	u64 dlss_presented_frame() { return s_presented_frame.load(); }

	dlss_mode dlss_configured_mode()
	{
		// RPCS3_DLSS (any mode, debug views included; fixed for the session), else the game's (or global) setting
		// VR / DLSS (test), which the context reads again at each flip (the settings and the home menu change it live).
		static const char* const s_env = env("RPCS3_DLSS");
		const auto parse = [](std::string s) -> dlss_mode
		{
			s = fmt::to_lower(s);
			std::erase(s, ' ');
			if (s == "motion" || s == "mv") return dlss_mode::debug_motion;
			if (s == "depth") return dlss_mode::debug_depth;
			if (s == "motionraw") return dlss_mode::debug_motion_raw;
			if (s == "warp") return dlss_mode::debug_warp;
			if (s == "jitter") return dlss_mode::jitter_only;
			if (s == "dlaa" || s == "1" || s == "on") return dlss_mode::dlaa;
			if (s == "quality") return dlss_mode::quality;
			if (s == "balanced") return dlss_mode::balanced;
			if (s == "performance") return dlss_mode::performance;
			if (s == "ultra" || s == "ultra_performance" || s == "ultraperformance") return dlss_mode::ultra_performance;
			return dlss_mode::off;
		};
		return parse(s_env ? std::string(s_env) : g_cfg.video.vr.dlss.to_string());
	}

	bool dlss_enabled()
	{
		// With the SDK the context exists from boot (Off does nothing per draw), so the settings and the home menu can turn
		// DLSS on live; RPCS3_DLSS=off rules it out. Without an NVIDIA GPU or the runtime NGX fails to start and DLSS
		// stays off. (Parked 2026-10-11 to boot-only; live again 2026-10-13, Matt.)
#ifdef HAVE_DLSS
		return env("RPCS3_DLSS") == nullptr || dlss_configured_mode() != dlss_mode::off;
#else
		return dlss_configured_mode() != dlss_mode::off;
#endif
	}

	bool dlss_ngx_possible()
	{
#ifdef HAVE_DLSS
		return env("RPCS3_DLSS") == nullptr || dlss_uses_ngx();
#else
		return false;
#endif
	}

	bool dlss_uses_ngx()
	{
		switch (dlss_configured_mode())
		{
		case dlss_mode::dlaa:
		case dlss_mode::quality:
		case dlss_mode::balanced:
		case dlss_mode::performance:
		case dlss_mode::ultra_performance:
			return true;
		default:
			return false;
		}
	}

	// ---- Instance and device --------------------------------------------------------------------------------

	void dlss_instance_extensions(std::vector<const char*>& extensions)
	{
#ifdef HAVE_DLSS
		if (!dlss_ngx_possible())
		{
			return;
		}

		u32 count = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
		std::vector<VkExtensionProperties> supported(count);
		vkEnumerateInstanceExtensionProperties(nullptr, &count, supported.data());

		unsigned int ic = 0, dc = 0;
		const char** ie = nullptr;
		const char** de = nullptr;
		if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ic, &ie, &dc, &de)))
		{
			dlss_log.error("NVSDK_NGX_VULKAN_RequiredExtensions failed");
			return;
		}
		for (unsigned i = 0; i < ic; ++i)
		{
			if (extension_listed(supported, ie[i]))
			{
				add_extension(extensions, ie[i]);
			}
			else
			{
				dlss_log.warning("Instance extension %s (DLSS) is not supported", ie[i]);
			}
		}
		framegen_instance_extensions(extensions); // framegen: DLSS-G's own (when asked for at boot)
#else
		static_cast<void>(extensions);
#endif
	}

	void dlss_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions)
	{
#ifdef HAVE_DLSS
		if (!dlss_ngx_possible())
		{
			return;
		}

		u32 count = 0;
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &count, nullptr);
		std::vector<VkExtensionProperties> supported(count);
		vkEnumerateDeviceExtensionProperties(pdev, nullptr, &count, supported.data());

		unsigned int ic = 0, dc = 0;
		const char** ie = nullptr;
		const char** de = nullptr;
		if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ic, &ie, &dc, &de)))
		{
			return;
		}
		for (unsigned i = 0; i < dc; ++i)
		{
			if (std::strcmp(de[i], "VK_EXT_buffer_device_address") == 0)
			{
				// Vulkan 1.2's bufferDeviceAddress (dlss_device_chain) replaces it; both together are invalid.
				continue;
			}
			if (extension_listed(supported, de[i]))
			{
				add_extension(extensions, de[i]);
			}
			else
			{
				dlss_log.warning("Device extension %s (DLSS) is not supported", de[i]);
			}
		}
		framegen_device_extensions(pdev, extensions); // framegen: DLSS-G's own (when asked for at boot)
#else
		static_cast<void>(pdev);
		static_cast<void>(extensions);
#endif
	}

	void dlss_device_chain(VkPhysicalDevice pdev, VkDeviceCreateInfo& info)
	{
#ifdef HAVE_DLSS
		if (!dlss_ngx_possible() || !pdev)
		{
			return;
		}

		// NGX's Vulkan path uses buffer device addresses: Vulkan 1.2's feature, in the chain's 1.2 features (RPCS3 always
		// chains them).
		VkPhysicalDeviceVulkan12Features supported{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
		VkPhysicalDeviceFeatures2 f2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &supported};
		vkGetPhysicalDeviceFeatures2(pdev, &f2);
		for (auto* next = static_cast<VkBaseOutStructure*>(const_cast<void*>(info.pNext)); next; next = next->pNext)
		{
			if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES && supported.bufferDeviceAddress)
			{
				reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(next)->bufferDeviceAddress = VK_TRUE;
			}
		}
#else
		static_cast<void>(pdev);
		static_cast<void>(info);
#endif
	}

	// ---- Pipelines --------------------------------------------------------------------------------------------

	void dlss_apply_pipeline_props(pipeline_props& props, u32 color_attachments)
	{
		if (!dlss_renderpass_has_motion(props.renderpass_key) || color_attachments >= std::size(props.state.att_state))
		{
			return;
		}

		props.state.set_attachment_count(color_attachments + 1);
		VkPipelineColorBlendAttachmentState& a = props.state.att_state[color_attachments];
		a = {};
		a.blendEnable = VK_TRUE;
		a.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		a.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		a.colorBlendOp = VK_BLEND_OP_ADD;
		a.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		a.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		a.alphaBlendOp = VK_BLEND_OP_ADD;
		a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	}

	void dlss_add_motion_attachment(u64 renderpass_key, VkSampleCountFlagBits samples, std::vector<VkAttachmentDescription>& attachments,
		std::vector<VkAttachmentReference>& references, u32& attachment_count)
	{
		if (!dlss_renderpass_has_motion(renderpass_key))
		{
			return;
		}

		VkAttachmentDescription motion = {};
		motion.format = dlss_motion_format(samples);
		motion.samples = samples;
		motion.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		motion.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		motion.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		motion.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		motion.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
		motion.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
		attachments.push_back(motion);
		references.push_back({attachment_count++, VK_IMAGE_LAYOUT_GENERAL});
	}

	void dlss_interpreter_props(pipeline_props& props)
	{
		if (dlss_renderpass_has_motion(props.renderpass_key) && props.state.cs.attachmentCount > 0)
		{
			props.state.att_state[props.state.cs.attachmentCount - 1].colorWriteMask = 0;
		}
	}

	// ---- Shader generation ----------------------------------------------------------------------------------

	void dlss_insert_vertex_output(std::ostream& OS, u32 ctrl)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_DLSS_MOTION))
		{
			return;
		}
		OS << "layout(location=" << dlss_cur_location << ") out vec4 dlss_cur;   // DLSS test: this frame's position\n";
		OS << "layout(location=" << dlss_prev_location << ") out vec4 dlss_prev;  // the same point last frame\n";
		OS << "layout(location=" << dlss_cover_location << ") flat out float dlss_cover;\n";
	}

	void dlss_insert_vertex_end(std::ostream& OS, u32 ctrl, u32 block_slot, bool has_constants)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_DLSS_MOTION))
		{
			return;
		}

		OS << "	// DLSS test: motion (the game's clip position p and R * p, both through the viewport transform) and jitter\n";
		if ((ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS) || !has_constants)
		{
			// Instanced constants come from another buffer, without the block: no motion written, no jitter.
			OS << "	dlss_cur = vr_pre_xform * scale_offset_mat;\n";
			OS << "	dlss_prev = dlss_cur;\n";
			OS << "	dlss_cover = 0.;\n";
			return;
		}

		OS << "	{\n";
		OS << "		const uint dlss_base = get_draw_params().xform_constants_offset + " << block_slot << "u;\n";
		OS << "		const vec4 dlss_jitter = vc[dlss_base + 4u];\n";
		OS << "		const vec4 dlss_prev_game = vec4(dot(vc[dlss_base], vr_pre_xform), dot(vc[dlss_base + 1u], vr_pre_xform), dot(vc[dlss_base + 2u], vr_pre_xform), dot(vc[dlss_base + 3u], vr_pre_xform));\n";
		OS << "		dlss_cur = vr_pre_xform * scale_offset_mat;\n";
		OS << "		dlss_prev = dlss_prev_game * scale_offset_mat;\n";
		OS << "		dlss_cover = dlss_jitter.z;\n";
		OS << "		gl_Position.xy += dlss_jitter.xy * gl_Position.w;\n";
		OS << "	}\n";
	}

	void dlss_insert_fragment_input(std::ostream& OS, u32 ctrl)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_DLSS_MOTION))
		{
			return;
		}
		OS << "layout(location=" << dlss_cur_location << ") in vec4 dlss_cur; // DLSS test\n";
		OS << "layout(location=" << dlss_prev_location << ") in vec4 dlss_prev;\n";
		OS << "layout(location=" << dlss_cover_location << ") flat in float dlss_cover;\n";
	}

	void dlss_insert_fragment_output(std::ostream& OS, u32 ctrl, u32 location)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_DLSS_MOTION))
		{
			return;
		}
		OS << "layout(location=" << location << ") out vec4 dlss_motion; // DLSS test: (motion in UV, depth, coverage)\n";
	}

	void dlss_insert_fragment_end(std::ostream& OS, u32 ctrl)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_DLSS_MOTION))
		{
			return;
		}
		OS << "	// DLSS test: motion towards the previous frame's position, in UV units\n";
		OS << "	{\n";
		OS << "		const vec2 dlss_c = dlss_cur.xy / dlss_cur.w;\n";
		OS << "		const vec2 dlss_p = dlss_prev.w > 1e-6 ? dlss_prev.xy / dlss_prev.w : dlss_c;\n";
		OS << "		// w: covered, as this point's view depth last frame over this frame's (R's w over p's: frame generation's\n";
		OS << "		// static-world check, VKFrameGen.h); 0: not covered.\n";
		OS << "		const float dlss_k = dlss_prev.w > 1e-6 && dlss_cur.w > 1e-6 ? clamp(dlss_prev.w / dlss_cur.w, 1e-3, 1e3) : 1.;\n";
		OS << "		dlss_motion = vec4((dlss_p - dlss_c) * 0.5, 1. - gl_FragCoord.z, dlss_cover > 0.5 ? dlss_k : 0.);\n";
		OS << "	}\n";
	}

	// ---- Context ------------------------------------------------------------------------------------------

	dlss_context::dlss_context(const vk::render_device& dev)
		: m_device(dev)
		, m_mode(dlss_configured_mode())
	{
		if (const char* v = env("RPCS3_DLSS_TARGET"))
		{
			m_forced_address = static_cast<u32>(std::strtoul(v, nullptr, 16));
		}
		m_jitter_scale = env_f32("RPCS3_DLSS_JITTER_SCALE", 1.f);

		// The sample counts each motion format supports as a blended colour attachment.
		const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
		const auto sample_counts = [&](VkFormat format) -> u32
		{
			VkImageFormatProperties fp{};
			if ((dev.get_format_properties(format).optimalTilingFeatures & needed) != needed ||
				vkGetPhysicalDeviceImageFormatProperties(dev.gpu(), format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, &fp) != VK_SUCCESS)
			{
				return 0;
			}
			return fp.sampleCounts;
		};
		s_rgba32f_samples = sample_counts(VK_FORMAT_R32G32B32A32_SFLOAT);
		s_rgba16f_samples = sample_counts(VK_FORMAT_R16G16B16A16_SFLOAT);
		if (env("RPCS3_DLSS_MOTION_16F"))
		{
			s_rgba32f_samples = 0; // test: the RGBA16F motion image
		}
		if (!s_rgba32f_samples && !s_rgba16f_samples)
		{
			dlss_log.error("Neither RGBA32F nor RGBA16F colour attachments with blending are supported: DLSS motion vectors are off");
			m_disabled = true;
		}

		if (m_mode == dlss_mode::debug_motion || m_mode == dlss_mode::debug_depth || m_mode == dlss_mode::debug_motion_raw || m_mode == dlss_mode::debug_warp || m_mode == dlss_mode::jitter_only)
		{
			m_jitter_phases = 8;
		}
		update_jitter();

		dlss_log.notice("DLSS test: mode %u (RPCS3_DLSS or VR / DLSS (test)), scene target %s", static_cast<u32>(m_mode),
			m_forced_address ? fmt::format("0x%x", m_forced_address) : std::string("found per frame"));
#ifndef HAVE_DLSS
		if (dlss_uses_ngx())
		{
			dlss_log.error("This build has no DLSS SDK (HAVE_DLSS): motion vectors and jitter run, the image is shown without DLSS%s",
				env("RPCS3_DLSS_STANDIN") ? " (RPCS3_DLSS_STANDIN: upscaled)" : "");
		}
#endif
	}

	dlss_context::~dlss_context()
	{
		destroy();
	}

	void dlss_context::destroy()
	{
		ngx_release_features();
#ifdef HAVE_DLSS
		if (m_ngx_params)
		{
			NVSDK_NGX_VULKAN_DestroyParameters(static_cast<NVSDK_NGX_Parameter*>(m_ngx_params));
			m_ngx_params = nullptr;
		}
		if (m_ngx_ready)
		{
			NVSDK_NGX_VULKAN_Shutdown1(m_device);
			m_ngx_ready = false;
		}
#endif
		m_last_result = nullptr;
		m_last_right = nullptr;
		m_motion.reset();
		m_mv.reset();
		m_depth.reset();
		m_debug.reset();
		m_output.reset();
		m_mv_right.reset();
		m_depth_right.reset();
		m_debug_right.reset();
		m_output_right.reset();
		for (u32 e = 0; e < 2; ++e)
		{
			m_warp_prev[e].reset();
			m_warp_prev2[e].reset();
			m_warp_prev_mv[e].reset();
			m_hud_pre[e].reset();
		}
		m_hud_pre_target = nullptr;
	}

	static f32 dlss_mode_upscale(dlss_mode mode);

	void dlss_context::refresh_mode()
	{
		if (env("RPCS3_DLSS"))
		{
			return; // fixed for the session
		}
		const dlss_mode mode = dlss_configured_mode();
		if (mode == m_mode)
		{
			return;
		}
		dlss_log.notice("DLSS mode %u -> %u (setting changed)", static_cast<u32>(m_mode), static_cast<u32>(mode));
		// The Resolution Scale follows, so the output keeps its size (DLAA -> Quality: 1 / 1.5 of it; Off counts as 1).
		if (const f32 from = dlss_mode_upscale(m_mode), to = dlss_mode_upscale(mode); from != to && !env("RPCS3_DLSS_VR_SCALE"))
		{
			const u32 scale = static_cast<u32>(g_cfg.video.resolution_scale_percent);
			const u32 scaled = std::clamp<u32>(static_cast<u32>(std::lround(scale * from / to / 5.f) * 5), 25, 800);
			g_cfg.video.resolution_scale_percent.set(scaled);
			dlss_log.notice("DLSS: Resolution Scale %u%% -> %u%% (upscale %.2f -> %.2f)", scale, scaled, from, to);
		}
		m_mode = mode;
		m_reset_history = true;
		m_jitter_phases = 8;
		m_last_result = nullptr;
		m_last_right = nullptr;
		m_last_out_w = m_last_out_h = 0;
		m_hud_pre_target = nullptr;
		if (mode == dlss_mode::off)
		{
			ngx_release_features();
		}
		update_jitter();
	}

	void dlss_context::update_jitter()
	{
		if (m_mode == dlss_mode::off || m_jitter_scale == 0.f || m_mode == dlss_mode::debug_motion || m_mode == dlss_mode::debug_depth || m_mode == dlss_mode::debug_motion_raw || m_mode == dlss_mode::debug_warp)
		{
			m_jitter_px[0] = m_jitter_px[1] = 0.f;
			return;
		}
		const u32 index = static_cast<u32>(m_frame % m_jitter_phases) + 1;
		// The title's VR profile can scale it down (dlss_jitter; RPCS3_DLSS_JITTER_SCALE overrides).
		const auto* profile = rsx::vr::camera_probe::get().profile();
		const f32 scale = std::getenv("RPCS3_DLSS_JITTER_SCALE") || !profile ? m_jitter_scale : profile->dlss_jitter;
		m_jitter_px[0] = (halton(index, 2) - 0.5f) * scale;
		m_jitter_px[1] = (halton(index, 3) - 0.5f) * scale;
	}

	bool dlss_context::find_camera_matrix(const u16* ids, usz count, mat4& out, const void* constants) const
	{
		static const auto* s_profile = static_cast<const void*>(nullptr);
		static std::vector<camera_block_desc> s_blocks;
		const auto* profile = rsx::vr::camera_probe::get().profile();
		if (profile != s_profile || (s_blocks.empty() && env("RPCS3_DLSS_CAMERA")))
		{
			s_profile = profile;
			s_blocks = camera_blocks_now();
		}

		const auto& bank = rsx::method_registers.transform_constants;
		for (const auto& b : s_blocks)
		{
			bool readable = true;
			for (u32 s : b.slots)
			{
				if (s >= bank.size() || (count && std::find(ids, ids + count, static_cast<u16>(s)) == ids + count))
				{
					readable = false;
					break;
				}
			}
			if (!readable)
			{
				continue;
			}

			f32 c[4][4];
			for (int i = 0; i < 4; ++i)
			{
				if (constants)
				{
					// The upload's layout: the slot's position among the program's constants, or the slot (full bank).
					const usz index = count ? static_cast<usz>(std::find(ids, ids + count, static_cast<u16>(b.slots[i])) - ids) : b.slots[i];
					std::memcpy(c[i], static_cast<const u8*>(constants) + index * 16, 16);
					continue;
				}
				for (int j = 0; j < 4; ++j)
				{
					c[i][j] = std::bit_cast<f32>(bank[b.slots[i]][j]);
				}
			}

			for (int r = 0; r < 4; ++r)
			{
				for (int k = 0; k < 4; ++k)
				{
					out.m[r * 4 + k] = b.columns ? c[r][k] : c[k][r];
				}
			}

			bool finite = true;
			for (f32 v : out.m)
			{
				finite &= std::isfinite(v);
			}

			// Perspective: clip w depends on the position (an orthographic HUD or shadow block has w = 1).
			if (finite && (std::abs(out.m[12]) + std::abs(out.m[13]) + std::abs(out.m[14])) > 1e-6f)
			{
				return true;
			}
		}
		return false;
	}

	void dlss_context::note_draw(u32 color_address, const u16* constant_ids, usz constant_count)
	{
		if (!color_address || m_mode == dlss_mode::off)
		{
			return;
		}
		mat4 m;
		if (find_camera_matrix(constant_ids, constant_count, m))
		{
			m_frame_camera_draws++; // counted when disabled too: game frames still end at the flip (presented unchanged)
			if (!m_forced_address && !m_disabled)
			{
				m_target_counts[color_address]++;
			}
		}
	}

	bool dlss_context::ensure_motion_image(const vk::command_buffer& cmd, vk::image* color_surface)
	{
		const u32 w = color_surface->width();
		const u32 h = color_surface->height();
		const auto samples = static_cast<VkSampleCountFlagBits>(color_surface->samples());
		const u32 layers = color_surface->layers(); // 2 in VR multiview: one per eye, as the render targets

		if (m_motion && m_motion->width() == w && m_motion->height() == h && m_motion->samples() == color_surface->samples() && m_motion->layers() == layers)
		{
			return true;
		}

		if (m_motion)
		{
			// Destroyed when the GPU is done with it; its framebuffers go with it.
			vk::get_resource_manager()->dispose(m_motion);
		}

		if (!((s_rgba32f_samples | s_rgba16f_samples) & samples))
		{
			dlss_log.error("RGBA32F or RGBA16F with %u samples is not supported: DLSS motion vectors are off", static_cast<u32>(samples));
			m_disabled = true;
			return false;
		}
		const VkFormat format = dlss_motion_format(samples);

		m_motion = std::make_unique<dlss_motion_image>(m_device, m_device.get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_IMAGE_TYPE_2D, format, w, h, 1, 1, layers, samples, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_CREATE_ALLOW_NULL_RPCS3, VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_COLOR);

		if (!m_motion->value)
		{
			dlss_log.error("Out of memory for the motion image (%ux%u, %u samples)", w, h, static_cast<u32>(samples));
			m_motion.reset();
			return false;
		}

		m_motion->stereo_layers = layers == 2;
		m_motion->set_debug_name("DLSS motion");
		m_motion->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_motion_cleared_frame = umax;
		m_reset_history = true;
		dlss_log.notice("Motion image %ux%u, %u samples, %s%s (scene target 0x%x)", w, h, static_cast<u32>(samples),
			format == VK_FORMAT_R32G32B32A32_SFLOAT ? "RGBA32F" : "RGBA16F", layers == 2 ? ", one layer per eye" : "", m_scene_address);
		return true;
	}

	bool dlss_context::wants_motion(u32 color_address, bool vr_unsupported) const
	{
		if (m_mode == dlss_mode::off || m_disabled || vr_unsupported || !color_address)
		{
			return false;
		}
		return color_address == (m_forced_address ? m_forced_address : m_scene_address);
	}

	vk::image* dlss_context::bind_framebuffer(const vk::command_buffer& cmd, u32 color_address, vk::image* color_surface,
		u32 fbo_width, u32 fbo_height, u32 color_attachments, u32 mrt_count, bool vr_unsupported)
	{
		static_cast<void>(fbo_width);
		static_cast<void>(fbo_height);
		m_bound = false;

		if (m_mode == dlss_mode::off || m_disabled || vr_unsupported || !color_surface)
		{
			return nullptr;
		}

		const u32 target = m_forced_address ? m_forced_address : m_scene_address;
		if (!target || color_address != target || color_attachments == 0 || color_attachments != mrt_count || color_attachments >= 4)
		{
			return nullptr;
		}

		if (!ensure_motion_image(cmd, color_surface))
		{
			return nullptr;
		}

		if (vk::is_renderpass_open(cmd))
		{
			vk::end_renderpass(cmd);
		}

		if (!clear_if_new_frame(cmd))
		{
			// Blending reads what an earlier pass wrote.
			const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, m_motion->layers()};
			vk::insert_image_memory_barrier(cmd, m_motion->value, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, range);
		}

		m_bound = true;
		return m_motion.get();
	}

	bool dlss_context::clear_if_new_frame(const vk::command_buffer& cmd)
	{
		if (m_motion_cleared_frame == m_frame)
		{
			return false;
		}

		if (vk::is_renderpass_open(cmd))
		{
			vk::end_renderpass(cmd);
		}

		// First use this frame: no motion, far depth (stored as 1 - z), not covered. Both eyes' layers in VR multiview.
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, m_motion->layers()};
		vk::insert_image_memory_barrier(cmd, m_motion->value, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, range);
		const VkClearColorValue clear{.float32 = {0.f, 0.f, 0.f, 0.f}};
		vkCmdClearColorImage(cmd, m_motion->value, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
		vk::insert_image_memory_barrier(cmd, m_motion->value, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, range);
		m_motion_cleared_frame = m_frame;
		return true;
	}

	void dlss_context::before_draw(const vk::command_buffer& cmd)
	{
		if (m_bound && m_motion)
		{
			clear_if_new_frame(cmd);
		}
	}

	void dlss_context::note_viewport_xform(const f32* so)
	{
		// scale_offset_mat as uploaded: the shader's vec4 * mat4 takes float 4k + j as output k's coefficient of input j.
		if (m_draw_key && so && so[0] != 0.f && so[5] != 0.f)
		{
			m_viewport_xform[0] = so[0];
			m_viewport_xform[1] = so[3];
			m_viewport_xform[2] = so[5];
			m_viewport_xform[3] = so[7];
			m_viewport_valid = true;
		}
	}

	bool dlss_context::framegen_inputs(const vk::command_buffer& cmd, u32 eye, bool checked, vk::viewable_image*& depth, vk::viewable_image*& motion,
		f32 (&clip_to_prev)[16], bool& have_matrix)
	{
		depth = eye ? m_depth_right.get() : m_depth.get();
		motion = eye ? m_mv_right.get() : m_mv.get();
		mat4 r{};
		have_matrix = best_reprojection(eye + 1, r); // views 1 and 2: the eyes
		std::memcpy(clip_to_prev, r.m, sizeof(clip_to_prev));
		if (!depth || !motion)
		{
			return false;
		}

		const u32 e = eye ? 1 : 0;
		auto& prev_depth = m_fg_prev_depth[e];
		auto& out = m_fg_mv[e];
		const u32 w = depth->width(), h = depth->height();
		bool have_prev = m_fg_have_prev[e];
		for (auto* image : {&prev_depth, &out})
		{
			if (!*image || (*image)->width() != w || (*image)->height() != h)
			{
				if (*image)
				{
					vk::get_resource_manager()->dispose(*image);
				}
				const bool is_depth = image == &prev_depth;
				*image = std::make_unique<vk::viewable_image>(m_device, m_device.get_memory_mapping().device_local, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					VK_IMAGE_TYPE_2D, is_depth ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32_SFLOAT, w, h, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
					0, VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_COLOR);
				(*image)->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL); // a copy returns it to this, never to undefined
				have_prev = false;
			}
		}

		// The motion image keeps the draws' depth ratios until the next frame's first motion draw clears it.
		if (checked && m_motion && m_motion->samples() == 1)
		{
			const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			const u32 layer = m_motion->layers() > 1 ? eye : u32{umax};
			m_motion->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			depth->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			motion->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			prev_depth->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			out->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			vk::insert_global_memory_barrier(cmd);
			const f32 src_w = static_cast<f32>(std::min<u32>(m_clip_width ? m_clip_width : m_motion->width(), m_motion->width()));
			const f32 src_h = static_cast<f32>(std::min<u32>(m_clip_height ? m_clip_height : m_motion->height(), m_motion->height()));
			static const f32 s_tolerance = env_f32("RPCS3_DLSSG_DEPTH_TOLERANCE", 0.04f);
			const f32 constants[8] = {src_w, src_h, static_cast<f32>(w), static_cast<f32>(h), s_tolerance, have_prev ? 1.f : 0.f, std::numeric_limits<f32>::max(), 0.f};
			vk::get_compute_task<dlss_framegen_check_pass>()->run(cmd, m_motion->get_layer_view(identity, VK_IMAGE_ASPECT_COLOR_BIT, layer), motion->get_view(identity),
				depth->get_view(identity), prev_depth->get_view(identity), out->get_view(identity), constants);
			vk::insert_global_memory_barrier(cmd);
			motion = out.get();
		}

		// This frame's depth for the next frame's check.
		const areai rect{0, 0, static_cast<s32>(w), static_cast<s32>(h)};
		vk::copy_image(cmd, depth, prev_depth.get(), rect, rect);
		m_fg_have_prev[e] = true;
		return true;
	}

	bool dlss_context::best_reprojection(u32 view, mat4& out) const
	{
		u32 best = 0;
		for (const auto& [m, n] : m_reproject_votes[std::min(view, 2u)])
		{
			if (n > best)
			{
				best = n;
				out = m;
			}
		}
		if (best >= 3)
		{
			return true;
		}
		if (m_static_valid[std::min(view, 2u)])
		{
			out = m_static_reproject[std::min(view, 2u)];
			return true;
		}
		return false;
	}

	void dlss_context::write_neutral_block(void* dst) const
	{
		f32* out = static_cast<f32*>(dst);
		std::memcpy(out, identity4, sizeof(identity4));
		out[16] = out[17] = out[18] = out[19] = 0.f;
	}

	void dlss_context::write_draw_block(void* dst, const u16* constant_ids, usz constant_count, u64 vp_hash,
		u32 clip_width, u32 clip_height, bool instanced, const void* constants, u32 view)
	{
		view = std::min(view, 2u);
		f32* out = static_cast<f32*>(dst);
		std::memcpy(out, identity4, sizeof(identity4));
		out[16] = out[17] = out[18] = out[19] = 0.f;

		if (view == 0)
		{
			m_draw_key = 0; // set below when this draw has a camera matrix
		}
		else if (!m_draw_key)
		{
			return; // the draw itself had none: neither have its eyes
		}

		static const bool s_debug = env("RPCS3_DLSS_DEBUG") != nullptr;
		if (!m_bound || instanced || !clip_width || !clip_height)
		{
			if (s_debug)
				dlss_log.notice("Draw skipped: bound %d, instanced %d, clip %ux%u", m_bound, instanced, clip_width, clip_height);
			return;
		}

		mat4 cur;
		if (!find_camera_matrix(constant_ids, constant_count, cur, constants))
		{
			if (s_debug)
				dlss_log.notice("Draw skipped: no camera matrix (%u constant ids)", static_cast<u32>(constant_count));
			// Dev (RPCS3_DLSS_RSTATS=1): each program drawn into the scene without a camera matrix (its pixels get the far
			// background's motion), once.
			if (static std::set<u64> s_skipped; env("RPCS3_DLSS_RSTATS") && view != 2 && s_skipped.insert(vp_hash).second)
			{
				const auto& regs = rsx::method_registers;
				dlss_log.notice("No camera matrix: program %016llx, %u constant ids, %u elements, depth test %d write %d, blend %d",
					vp_hash, static_cast<u32>(constant_count), regs.current_draw_clause.vr_total_elements(), regs.depth_test_enabled(), regs.depth_write_enabled(), regs.blend_enabled_mask() != 0);
			}
			return;
		}

		m_scene_drawn = true;
		m_clip_width = clip_width;
		m_clip_height = clip_height;

		const auto& regs = rsx::method_registers;
		u64 key = 0, key_alt = 0;
		if (view == 0)
		{
			m_camera_draws++;

			// Draw identity: program, vertex and index addresses, range, and how many such draws came before it.
			key = mix(0xcbf29ce484222325ull, vp_hash);
			for (u32 i = 0; i < 16; ++i)
			{
				if (regs.vertex_arrays_info[i].size())
				{
					key = mix(key, (u64{i} << 32) | regs.vertex_arrays_info[i].offset());
				}
			}
			key = mix(key, regs.index_array_address());
			const auto& clause = regs.current_draw_clause;
			// (min_index() asserts outside the draw loop; the element count and the addresses identify the draw well enough.)
			key = mix(key, clause.vr_total_elements());
			key = mix(key, m_occurrences[key]++);
			m_draw_key = key;
			// Without the addresses: games that build their geometry each frame into a ring buffer (EDGE on the SPUs:
			// Sonic & All-Stars Racing Transformed) move every draw's vertices every frame, so the full key never
			// repeats. Program, element count and repeat number pair most draws with their previous frame's; static
			// geometry shares the camera's matrix, so a wrong pairing among those changes nothing.
			const u64 alt_base = mix(mix(0x84222325cbf29ce4ull, vp_hash), clause.vr_total_elements());
			key_alt = mix(alt_base, m_occurrences[alt_base]++);
			m_draw_match_base = mix(0x2545f4914f6cdd1dull, vp_hash); // the program only: GT5's cars change element counts as the SPUs cull
			m_draw_key_alt = key_alt;
			m_draw_stat_key = vp_hash ^ (u64{clause.vr_total_elements()} << 40);
			if (m_identify_only)
			{
				m_eyes_only = true; // statistics from the left eye
				return; // VR multiview: no view-0 history; the eyes' blocks follow
			}
		}
		else
		{
			// An eye of the draw whose view-0 block was just written: its own history.
			key = mix(m_draw_key, 0x5eed0000ull + view);
			key_alt = mix(m_draw_key_alt, 0x5eed0000ull + view);
		}

		f32 r[16];
		std::memcpy(r, identity4, sizeof(r));
		bool have_r = false;

		bool by_alt = false;
		auto found = m_prev_draws.find(key);
		const f32* prev_m = found != m_prev_draws.end() ? found->second.m : nullptr;
		const u64 match_key = mix(m_draw_match_base, 0x5eed0000ull + view);
		// Profile dlss_pair_nearest. Dev RPCS3_DLSS_PAIR (any game): =order by draw order; =nearest by matrix; =alt by draw order on
		// odd frames and by matrix on even ones (a same-run A/B with tools/re/mvcheck.py and parity3.py).
		const auto* pair_profile = rsx::vr::camera_probe::get().profile();
		static const char* const s_pair_env = env("RPCS3_DLSS_PAIR");
		const bool by_order = s_pair_env ? (std::string_view(s_pair_env) == "order" || (std::string_view(s_pair_env) == "alt" && (m_frame & 1)))
			: (!pair_profile || !pair_profile->dlss_pair_nearest);
		if (!prev_m && !by_order)
		{
			// Where this frame's camera motion puts the draw's matrix last frame (exact for a static object: a paused scene's
			// cars at any head speed), then the previous frame's matrix of this program nearest to that, when near (5 % of the
			// matrix's scale) and clearly the nearest (the next one twice as far).
			f32 cam[16];
			std::memcpy(cam, identity4, sizeof(cam));
			u32 cam_votes = 0;
			for (const auto& [m, n] : m_reproject_votes[view])
			{
				if (n > cam_votes) { cam_votes = n; std::memcpy(cam, m.m, sizeof(cam)); }
			}
			if (cam_votes < 3 && m_static_valid[view])
			{
				std::memcpy(cam, m_static_reproject[view].m, sizeof(cam));
			}
			f32 expected[16];
			mul4(cam, cur.m, expected);
			if (const auto list = m_prev_matrices.find(match_key); list != m_prev_matrices.end())
			{
				f32 scale = 1.f;
				for (const f32 v : expected) scale = std::max(scale, std::abs(v));
				f32 best_d = std::numeric_limits<f32>::max(), next_d = std::numeric_limits<f32>::max();
				const f32* best = nullptr;
				for (const auto& m : list->second)
				{
					f32 d = 0.f;
					for (u32 i = 0; i < 16; ++i) d = std::max(d, std::abs(m.m[i] - expected[i]));
					if (d < best_d) { next_d = best_d; best_d = d; best = m.m; }
					else if (d < next_d) { next_d = d; }
				}
				if (best && best_d <= 0.05f * scale && next_d >= 2.f * best_d)
				{
					prev_m = best;
					by_alt = true;
				}
			}
		}
		else if (!prev_m)
		{
			found = m_prev_draws.find(key_alt);
			by_alt = found != m_prev_draws.end();
			prev_m = by_alt ? found->second.m : nullptr;
		}
		const u32 pair_kind = !prev_m ? 3u : !by_alt ? 0u : by_order ? 2u : 1u; // address, nearest, draw order, none
		if (prev_m)
		{
			f32 inv[16];
			if (invert4(cur.m, inv))
			{
				mul4(prev_m, inv, r);
				have_r = true;
				m_matched += view == (m_eyes_only ? 1u : 0u);

				// Votes for the camera-only reprojection (the same for every static object), from the main view only: a draw
				// through a viewport covering under a quarter of the target is an inset with its own camera (GT5's rear-view
				// mirror, 448x86 of 1280x720, which does not turn with the head).
				const f32 vp_area = 4.f * std::abs(regs.viewport_scale_x() * regs.viewport_scale_y());
				const f32 target_area = static_cast<f32>(regs.surface_clip_width()) * regs.surface_clip_height();
				auto& votes = m_reproject_votes[view];
				bool voted = vp_area < 0.25f * target_area;
				for (auto& [m, n] : votes)
				{
					if (voted)
					{
						break;
					}
					if (same_motion(m.m, r))
					{
						n++;
						voted = true;
						break;
					}
				}
				if (!voted && votes.size() < 16)
				{
					mat4 v;
					std::memcpy(v.m, r, sizeof(r));
					votes.emplace_back(v, 1u);
				}
			}
		}

		// Dev (RPCS3_DLSS_RSTATS=1): paired draws whose motion is not the camera's (in a race only the cars should be).
		if (static const bool s_rstats = env("RPCS3_DLSS_RSTATS") != nullptr; s_rstats && have_r && view == (m_eyes_only ? 1u : 0u) && m_static_valid[view])
		{
			f32 d = 0.f;
			for (u32 i = 0; i < 16; ++i)
			{
				d = std::max(d, std::abs(r[i] - m_static_reproject[view].m[i]));
			}
			const bool moved = d > 0.01f;
			auto& st = m_rstats[m_draw_stat_key];
			st.n++;
			st.moved += moved;
			st.alt |= by_alt;
			m_rstats_n[by_alt]++;
			m_rstats_moved[by_alt] += moved;
		}

		if (!have_r)
		{
			m_unmatched += view == (m_eyes_only ? 1u : 0u);
			// Profile dlss_pair_nearest: this frame's camera motion once the main view has settled it (3 votes); the previous
			// frame's is a frame late on every head turn. Otherwise the previous frame's (measured better for Sonic).
			const mat4* fresh = nullptr;
			if (!by_order)
			{
				u32 best = 2;
				for (const auto& [m, n] : m_reproject_votes[view])
				{
					if (n > best) { best = n; fresh = &m; }
				}
			}
			if (fresh)
			{
				std::memcpy(r, fresh->m, sizeof(r));
			}
			else if (m_static_valid[view])
			{
				std::memcpy(r, m_static_reproject[view].m, sizeof(r));
			}
		}

		// Dev RPCS3_DLSS_STATIC_CHECK=1: in a paused scene every draw's motion is the camera's; logs per program, every 120
		// frames, how many draws' motion was not (by how each was paired), and the camera-motion votes.
		if (static const bool s_static_check = env("RPCS3_DLSS_STATIC_CHECK") != nullptr; s_static_check && view == (m_eyes_only ? 1u : 0u))
		{
			static_check_draw e{vp_hash, pair_kind, {}};
			std::memcpy(e.r.m, r, sizeof(r));
			m_static_check.push_back(e);
		}
		if (s_debug)
		{
			const auto prev_it = m_prev_draws.find(key);
			dlss_log.notice("Draw key %016llx view %u: M row0 (%.4f %.4f %.4f %.4f), previous %s, R row0 (%.4f %.4f %.4f %.4f)", key, view, cur.m[0], cur.m[1], cur.m[2], cur.m[3],
				prev_it != m_prev_draws.end() ? fmt::format("(%.4f %.4f %.4f %.4f)", prev_it->second.m[0], prev_it->second.m[1], prev_it->second.m[2], prev_it->second.m[3]) : std::string("none"),
				r[0], r[1], r[2], r[3]);
		}

		if (s_test_record)
		{
			dlss_test_draw t{view, m_view_draws[view]};
			std::memcpy(t.cur, cur.m, sizeof(t.cur));
			const auto prev_it = m_prev_draws.find(key);
			std::memcpy(t.prev, prev_it != m_prev_draws.end() ? prev_it->second.m : cur.m, sizeof(t.prev));
			t.matched = prev_it != m_prev_draws.end();
			m_test_draws.push_back(t);
		}
		m_view_draws[view]++;

		draw_record rec;
		std::memcpy(rec.m, cur.m, sizeof(rec.m));
		m_cur_draws[key] = rec;
		m_cur_draws[key_alt] = rec;
		if (auto& list = m_cur_matrices[match_key]; list.size() < 64 && std::none_of(list.begin(), list.end(), [&](const mat4& m) { return near_equal(m.m, cur.m); }))
		{
			list.push_back(cur);
		}

		// Coverage: opaque surfaces (depth writes) and unblended depth-tested ones (the sky). Blended draws without depth
		// writes (particles, glass) keep the motion of what is behind them.
		const bool depth_write = regs.depth_test_enabled() && regs.depth_write_enabled();
		const bool opaque_test = regs.depth_test_enabled() && !regs.blend_enabled_mask();
		const f32 cover = (depth_write || opaque_test) ? 1.f : 0.f;

		std::memcpy(out, r, sizeof(r));
		// Dev: RPCS3_DLSS_JITTER_BITS=mask,value jitters only programs whose hash & mask == value (bisecting a draw
		// the jitter breaks); with RPCS3_DLSS_DEBUG each jittered program is logged once.
		static const auto s_bits = []() -> std::pair<u64, u64>
		{
			u64 mask = 0, value = 0;
			if (const char* v = env("RPCS3_DLSS_JITTER_BITS")) std::sscanf(v, "%llx,%llx", &mask, &value);
			return {mask, value};
		}();
		const bool jitter_this = (vp_hash & s_bits.first) == s_bits.second;
		if (static std::set<u64> s_logged; s_debug && view != 2 && s_logged.insert(vp_hash).second)
		{
			dlss_log.notice("Camera program %016llx: viewport %.0fx%.0f of %ux%u, %u elements, depth test %d write %d func %u, blend %d",
				vp_hash, 2.f * std::abs(regs.viewport_scale_x()), 2.f * std::abs(regs.viewport_scale_y()), regs.surface_clip_width(), regs.surface_clip_height(),
				regs.current_draw_clause.vr_total_elements(), regs.depth_test_enabled(), regs.depth_write_enabled(), static_cast<u32>(regs.depth_func()), regs.blend_enabled_mask() != 0);
		}
		out[16] = jitter_this ? 2.f * m_jitter_px[0] / static_cast<f32>(clip_width) : 0.f;
		out[17] = jitter_this ? 2.f * m_jitter_px[1] / static_cast<f32>(clip_height) : 0.f;
		out[18] = cover;
		out[19] = 0.f;
	}

	bool dlss_context::ensure_resolve_images(u32 width, u32 height, bool stereo)
	{
		if (m_mv && m_mv->width() == width && m_mv->height() == height && (!stereo || m_mv_right))
		{
			return true;
		}

		m_last_result = nullptr;
		m_last_right = nullptr;
		for (auto* image : {&m_mv, &m_depth, &m_debug, &m_mv_right, &m_depth_right, &m_debug_right})
		{
			if (*image) vk::get_resource_manager()->dispose(*image);
		}

		// TRANSFER_SRC: the debug views are presented (and copied for screenshots and the VR mirror).
		const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		bool ok = true;
		for (u32 eye = 0; eye < (stereo ? 2u : 1u); ++eye)
		{
			auto& mv = eye ? m_mv_right : m_mv;
			auto& depth = eye ? m_depth_right : m_depth;
			auto& debug = eye ? m_debug_right : m_debug;
			mv = make_image(m_device, VK_FORMAT_R32G32_SFLOAT, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
			depth = make_image(m_device, VK_FORMAT_R32_SFLOAT, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
			debug = make_image(m_device, VK_FORMAT_R8G8B8A8_UNORM, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
			ok &= mv->value && depth->value && debug->value;
		}

		if (!ok)
		{
			dlss_log.error("Out of memory for the DLSS inputs (%ux%u)", width, height);
			for (auto* image : {&m_mv, &m_depth, &m_debug, &m_mv_right, &m_depth_right, &m_debug_right})
			{
				image->reset();
			}
			return false;
		}
		m_reset_history = true;
		return true;
	}

	bool dlss_context::ensure_output_images(u32 width, u32 height, bool stereo)
	{
		if (m_output && m_output->width() == width && m_output->height() == height && (!stereo || m_output_right))
		{
			return true;
		}

		m_last_result = nullptr;
		m_last_right = nullptr;
		for (auto* image : {&m_output, &m_output_right})
		{
			if (*image) vk::get_resource_manager()->dispose(*image);
		}

		// RGBA8: the present path (screenshots) expects 4 bytes per pixel. TRANSFER_SRC: presented, copied for
		// screenshots and into the headset's eye buffers. TRANSFER_DST: RPCS3_DLSS_STANDIN's blit.
		const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		bool ok = true;
		for (u32 eye = 0; eye < (stereo ? 2u : 1u); ++eye)
		{
			auto& output = eye ? m_output_right : m_output;
			output = make_image(m_device, VK_FORMAT_R8G8B8A8_UNORM, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
			if (!output->value)
			{
				ok = false;
				break;
			}
			output->set_debug_name(eye ? "DLSS output (right eye)" : "DLSS output");
		}

		if (!ok)
		{
			dlss_log.error("Out of memory for the DLSS output (%ux%u%s)", width, height, stereo ? ", two eyes" : "");
			m_output.reset();
			m_output_right.reset();
			return false;
		}
		return true;
	}

	// The render scale each mode stands for (NGX's own: 1/1.5, 0.58, 0.5, 1/3 per axis), as an upscale.
	static f32 dlss_mode_upscale(dlss_mode mode)
	{
		switch (mode)
		{
		case dlss_mode::quality: return 1.5f;
		case dlss_mode::balanced: return 1.72f;
		case dlss_mode::performance: return 2.f;
		case dlss_mode::ultra_performance: return 3.f;
		default: return 1.f;
		}
	}

	f32 dlss_context::vr_output_scale() const
	{
		if (const f32 forced = env_f32("RPCS3_DLSS_VR_SCALE", 0.f); forced > 0.f)
		{
			return std::clamp(forced, 1.f, 3.f);
		}
		return dlss_mode_upscale(m_mode);
	}

	void dlss_context::resolve(const vk::command_buffer& cmd, u32 width, u32 height, u32 eye)
	{
		// Every earlier write (motion attachment, the displayed image) before the compute reads.
		vk::insert_global_memory_barrier(cmd);

		vk::viewable_image* const mv_image = eye ? m_mv_right.get() : m_mv.get();
		vk::viewable_image* const depth_image = eye ? m_depth_right.get() : m_depth.get();
		vk::viewable_image* const debug_image = eye ? m_debug_right.get() : m_debug.get();
		mv_image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		depth_image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		debug_image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		static const auto mv_sign = env_signs("RPCS3_DLSS_MV_SIGN");
		const f32 debug_mode = m_mode == dlss_mode::debug_motion ? 1.f : m_mode == dlss_mode::debug_depth ? 2.f : m_mode == dlss_mode::debug_motion_raw ? 3.f : 0.f;
		const f32 src_w = static_cast<f32>(std::min<u32>(m_clip_width ? m_clip_width : m_motion->width(), m_motion->width()));
		const f32 src_h = static_cast<f32>(std::min<u32>(m_clip_height ? m_clip_height : m_motion->height(), m_motion->height()));
		f32 constants[28] = {src_w, src_h, static_cast<f32>(width), static_cast<f32>(height), mv_sign[0], mv_sign[1], debug_mode, 0.f,
			m_viewport_xform[0], m_viewport_xform[1], m_viewport_xform[2], m_viewport_xform[3]};
		// The camera's motion for pixels no draw covered: this frame's R of the eye (VR: views 1 and 2), or the flat one.
		if (mat4 r; m_viewport_valid && best_reprojection(m_motion->layers() > 1 ? eye + 1 : 0, r))
		{
			std::memcpy(constants + 12, r.m, sizeof(r.m));
			constants[7] = 1.f;
		}

		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const vk::image_view* mv = mv_image->get_view(identity);
		const vk::image_view* depth = depth_image->get_view(identity);
		const vk::image_view* debug = debug_image->get_view(identity);
		// One layer of the motion image: the eye's (a one-layer image: its only one).
		const u32 layer = m_motion->layers() > 1 ? eye : u32{umax};

		if (m_motion->samples() > 1)
		{
			const vk::image_view* src = m_motion->get_layer_view(rsx::default_remap_vector.with_encoding(VK_REMAP_VIEW_MULTISAMPLED), VK_IMAGE_ASPECT_COLOR_BIT, layer);
			vk::get_compute_task<dlss_resolve_pass<true>>()->run(cmd, src, mv, depth, debug, constants);
		}
		else
		{
			const vk::image_view* src = m_motion->get_layer_view(identity, VK_IMAGE_ASPECT_COLOR_BIT, layer);
			vk::get_compute_task<dlss_resolve_pass<false>>()->run(cmd, src, mv, depth, debug, constants);
		}

		vk::insert_global_memory_barrier(cmd);
	}

	void dlss_context::warp_check(const vk::command_buffer& cmd, vk::viewable_image* source, u32 width, u32 height, u32 eye)
	{
		const u32 e = eye ? 1 : 0;
		auto& prev = m_warp_prev[e];
		auto& prev2 = m_warp_prev2[e];
		auto& prev_mv = m_warp_prev_mv[e];
		vk::viewable_image* const mv_image = eye ? m_mv_right.get() : m_mv.get();
		vk::viewable_image* const debug_image = eye ? m_debug_right.get() : m_debug.get();
		if (!source || !mv_image || !debug_image || source->width() < width || source->height() < height)
		{
			return;
		}
		const bool fresh = !prev || prev->width() != width || prev->height() != height || prev->format() != source->format();
		if (fresh)
		{
			for (auto* image : {&prev, &prev2, &prev_mv})
			{
				if (*image) vk::get_resource_manager()->dispose(*image);
			}
			const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			prev = make_image(m_device, source->format(), width, height, VK_SAMPLE_COUNT_1_BIT, usage);
			prev2 = make_image(m_device, source->format(), width, height, VK_SAMPLE_COUNT_1_BIT, usage);
			prev_mv = make_image(m_device, VK_FORMAT_R32G32_SFLOAT, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
		}
		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const areai rect = {0, 0, static_cast<s32>(width), static_cast<s32>(height)};

		if (!fresh)
		{
			source->push_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			for (auto* image : {prev.get(), prev2.get(), prev_mv.get(), mv_image, debug_image})
			{
				image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			}
			vk::insert_global_memory_barrier(cmd);
			static const f32 s_plain = env_f32("RPCS3_DLSS_WARP_PLAIN", 0.f);
			vk::get_compute_task<dlss_warp_check_pass>()->run(cmd, mv_image->get_view(identity), source->get_view(identity), prev->get_view(identity),
				debug_image->get_view(identity), prev2->get_view(identity), prev_mv->get_view(identity), width, height, env_f32("RPCS3_DLSS_WARP_GAIN", 8.f), s_plain);
			vk::insert_global_memory_barrier(cmd);
			source->pop_layout(cmd);
		}
		// History: prev -> prev2, this input -> prev, these vectors -> prev_mv.
		prev->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		prev2->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vk::copy_image(cmd, prev.get(), prev2.get(), rect, rect);
		source->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		prev->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vk::copy_image(cmd, source, prev.get(), rect, rect);
		source->pop_layout(cmd);
		mv_image->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		prev_mv->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vk::copy_image(cmd, mv_image, prev_mv.get(), rect, rect);
		for (auto* image : {prev.get(), prev2.get(), prev_mv.get(), mv_image})
		{
			image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		}
	}

	void dlss_context::hud_snapshot(const vk::command_buffer& cmd, vk::viewable_image* target)
	{
		if (!target || target == m_hud_pre_target || target->samples() > 1 || m_mode == dlss_mode::off || m_disabled)
		{
			return;
		}
		const u32 layers = std::min<u32>(target->layers(), 2);
		for (u32 eye = 0; eye < layers; ++eye)
		{
			auto& pre = m_hud_pre[eye];
			if (!pre || pre->width() != target->width() || pre->height() != target->height() || pre->format() != target->format())
			{
				if (pre) vk::get_resource_manager()->dispose(pre);
				pre = make_image(m_device, target->format(), target->width(), target->height(), VK_SAMPLE_COUNT_1_BIT,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
				if (!pre->value)
				{
					pre.reset();
					return;
				}
			}
		}

		target->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		for (u32 eye = 0; eye < layers; ++eye)
		{
			auto& pre = m_hud_pre[eye];
			pre->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			VkImageCopy copy{};
			copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eye, 1};
			copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			copy.extent = {target->width(), target->height(), 1};
			vkCmdCopyImage(cmd, target->value, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pre->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
			pre->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		}
		target->pop_layout(cmd);
		m_hud_pre_target = target;
		m_hud_pre_layers = layers;
	}

	vk::viewable_image* dlss_context::hud_pre(const vk::image* flip_image, u32 eye) const
	{
		if (!flip_image || flip_image != m_hud_pre_target || eye >= m_hud_pre_layers)
		{
			return nullptr;
		}
		return m_hud_pre[eye].get();
	}

	void dlss_context::hud_composite(const vk::command_buffer& cmd, vk::viewable_image* out, vk::viewable_image* final_image, u32 final_layer,
		vk::viewable_image* pre, u32 in_w, u32 in_h, const f32* homography)
	{
		if (!out || !final_image || !pre || !in_w || !in_h)
		{
			return;
		}
		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		vk::insert_global_memory_barrier(cmd);
		final_image->push_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		pre->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		out->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		const vk::image_view* final_view = final_image->layers() > 1 ?
			final_image->get_layer_view(identity, VK_IMAGE_ASPECT_COLOR_BIT, final_layer) : final_image->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
		static const f32 identity_h[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
		const f32* h = homography ? homography : identity_h;
		const f32 pc[20] = {static_cast<f32>(out->width()), static_cast<f32>(out->height()),
			static_cast<f32>(in_w) / final_image->width(), static_cast<f32>(in_h) / final_image->height(),
			static_cast<f32>(in_w) / pre->width(), static_cast<f32>(in_h) / pre->height(), env_f32("RPCS3_DLSS_HUD_GAIN", 20.f), 0.f,
			h[0], h[1], h[2], 0.f, h[3], h[4], h[5], 0.f, h[6], h[7], h[8], 0.f};
		vk::get_compute_task<dlss_hud_composite_pass>()->run(cmd, final_view, pre->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT), out->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT), pc);
		final_image->pop_layout(cmd);
		vk::insert_global_memory_barrier(cmd);
	}

	vk::viewable_image* dlss_context::on_flip(const vk::command_buffer& cmd, vk::viewable_image* source, u32 width, u32 height,
		u32 target_width, u32 target_height, bool emu_flip, bool vr_unsupported, u32& out_w, u32& out_h,
		vk::viewable_image** right_eye)
	{
		m_output_fresh = false;
		refresh_mode();
		if (m_mode == dlss_mode::off)
		{
			return nullptr;
		}

		const bool debug_view = m_mode == dlss_mode::debug_motion || m_mode == dlss_mode::debug_depth || m_mode == dlss_mode::debug_motion_raw || m_mode == dlss_mode::debug_warp;
		// VR multiview: both eyes are shown (right_eye), and the motion image has a layer per eye. A VR frame without
		// the right eye is shown on the fixed screen (menus, videos, frames_without_3d_as_screen).
		const bool vr = right_eye != nullptr;
		const bool stereo = vr && *right_eye;

		if (!emu_flip || !m_frame_camera_draws)
		{
			// A UI refresh, or a flip without any 3D drawn since the last one (a game showing the same frame again,
			// a capture replay's empty loop): show the last output again; history, jitter and frame count stay.
			if (m_last_result && m_last_out_w && m_last_out_h && !m_disabled && !vr_unsupported && (!stereo || m_last_right))
			{
				out_w = m_last_out_w;
				out_h = m_last_out_h;
				if (stereo)
				{
					*right_eye = m_last_right;
				}
				return m_last_result;
			}
			return nullptr;
		}
		m_frame_camera_draws = 0;

		// The headset view and the fixed screen show different cameras (the VR one, the game's own): a switch between
		// them is a cut for the history.
		if (const s32 stereo_now = stereo ? 1 : 0; m_last_stereo != stereo_now)
		{
			if (m_last_stereo >= 0)
			{
				dlss_log.notice("%s: DLSS history starts again", stereo ? "Headset view" : "One image (fixed screen or flat)");
			}
			m_last_stereo = stereo_now;
			m_reset_history = true;
		}

		vk::viewable_image* result = nullptr;
		vk::viewable_image* result_right = nullptr;
		const bool usable = !m_disabled && !vr_unsupported && source && width && height && m_motion && m_scene_drawn &&
			m_motion_cleared_frame == m_frame && (!stereo || m_motion->layers() == 2);

		if (usable && m_mode != dlss_mode::jitter_only && ensure_resolve_images(width, height, stereo))
		{
			resolve(cmd, width, height, 0);
			if (stereo)
			{
				resolve(cmd, width, height, 1);
			}
			if (m_mode == dlss_mode::debug_warp)
			{
				warp_check(cmd, source, width, height, 0);
				if (stereo)
				{
					warp_check(cmd, *right_eye, width, height, 1);
				}
			}

			if (debug_view)
			{
				result = m_debug.get();
				result_right = stereo ? m_debug_right.get() : nullptr;
				out_w = m_last_out_w = width;
				out_h = m_last_out_h = height;
			}
			else
			{
				// Flat: output at the window's size for this image (upscale up to 3x, never down: DLAA then). VR: no
				// window (the headset's eye buffers take the eyes' size), the mode's ratio.
				f32 scale = vr ? vr_output_scale() :
					std::min(static_cast<f32>(target_width) / static_cast<f32>(width), static_cast<f32>(target_height) / static_cast<f32>(height));
				if ((m_mode == dlss_mode::dlaa && !vr) || !std::isfinite(scale))
				{
					scale = 1.f;
				}
				scale = std::clamp(scale, 1.f, 3.f);
				const u32 ow = static_cast<u32>(std::lround(width * scale));
				const u32 oh = static_cast<u32>(std::lround(height * scale));

				// Both eyes or neither (the headset takes the pair).
				if (ensure_output_images(ow, oh, stereo) && ngx_evaluate(cmd, 0, source, width, height, ow, oh) &&
					(!stereo || ngx_evaluate(cmd, 1, *right_eye, width, height, ow, oh)))
				{
					result = m_output.get();
					result_right = stereo ? m_output_right.get() : nullptr;
					m_output_fresh = true;
					out_w = m_last_out_w = ow;
					out_h = m_last_out_h = oh;
					m_reset_history = false;
				}
				else
				{
					m_reset_history = true;
					m_last_out_w = m_last_out_h = 0;
				}

				const f32 ratio = static_cast<f32>(ow) / static_cast<f32>(width);
				m_jitter_phases = std::max(8u, static_cast<u32>(std::ceil(8.f * ratio * ratio)));
			}
		}
		else if (!usable)
		{
			m_reset_history = true;
			m_last_out_w = m_last_out_h = 0;
		}

		// ---- The frame is over: history, scene target, statistics, next jitter ----
		static const bool s_debug = env("RPCS3_DLSS_DEBUG") != nullptr;
		if (m_frame % 300 == 0 || s_debug)
		{
			dlss_log.notice("Frame %llu: scene 0x%x, camera draws %u (matched %u, unmatched %u; eyes %u + %u), static reprojection %s, clip %ux%u, output %s",
				m_frame, m_forced_address ? m_forced_address : m_scene_address, m_camera_draws, m_matched, m_unmatched, m_view_draws[1], m_view_draws[2],
				m_static_valid[m_eyes_only ? 1 : 0] ? "found" : "none", m_clip_width, m_clip_height, result ? (result_right ? "replaced (both eyes)" : "replaced") : "unchanged");
		}

		if (m_frame % 120 == 0 && (m_rstats_n[0] || m_rstats_n[1]))
		{
			std::vector<std::pair<u64, rstat>> top(m_rstats.begin(), m_rstats.end());
			std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.moved > b.second.moved; });
			std::string worst;
			for (usz i = 0; i < std::min<usz>(top.size(), 6) && top[i].second.moved; ++i)
			{
				worst += fmt::format(" %016llx/%u: %u of %u%s;", top[i].first & 0xffffffffffull, static_cast<u32>(top[i].first >> 40), top[i].second.moved, top[i].second.n, top[i].second.alt ? " (fallback)" : "");
			}
			dlss_log.notice("Pairing (120 frames): by address %u (%u moved), by the fallback %u (%u moved); most moved (program low bits/elements):%s",
				m_rstats_n[0], m_rstats_moved[0], m_rstats_n[1], m_rstats_moved[1], worst);
			m_rstats.clear();
			m_rstats_n[0] = m_rstats_n[1] = m_rstats_moved[0] = m_rstats_moved[1] = 0;
		}

		m_prev_draws.swap(m_cur_draws);
		m_cur_draws.clear();
		m_prev_matrices.swap(m_cur_matrices);
		m_cur_matrices.clear();
		m_occurrences.clear();
		m_draw_key = 0;

		if (!m_static_check.empty())
		{
			const u32 v = m_eyes_only ? 1u : 0u;
			const mat4* cam = nullptr;
			u32 best = 0;
			for (const auto& [m, n] : m_reproject_votes[v]) if (n > best) { best = n; cam = &m; }
			if (static u32 s_vote_log = 0; cam && m_frame % 30 == 0 && s_vote_log++ < 40)
			{
				std::vector<std::pair<u32, f32>> clusters;
				for (const auto& [m, n] : m_reproject_votes[v])
				{
					f32 di = 0.f;
					for (u32 i = 0; i < 16; ++i) di = std::max(di, std::abs(m.m[i] - identity4[i]));
					clusters.emplace_back(n, di);
				}
				std::sort(clusters.begin(), clusters.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
				std::string line;
				for (usz i = 0; i < std::min<usz>(clusters.size(), 6); ++i) fmt::append(line, " %u draws |R-I| %.4f;", clusters[i].first, clusters[i].second);
				dlss_log.notice("Motion votes frame %llu (%u clusters):%s", m_frame, static_cast<u32>(clusters.size()), line);
			}
			if (cam)
			{
				for (const auto& e : m_static_check)
				{
					f32 d = 0.f;
					for (u32 i = 0; i < 16; ++i) d = std::max(d, std::abs(e.r.m[i] - cam->m[i]));
					auto& st = m_static_check_stats[e.program];
					st[e.kind]++;
					st[4 + e.kind] += d > 0.003f;
				}
			}
			m_static_check.clear();
			if (m_frame % 120 == 0)
			{
				std::vector<std::pair<u64, std::array<u32, 8>>> top(m_static_check_stats.begin(), m_static_check_stats.end());
				std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second[4] + a.second[5] + a.second[6] + a.second[7] > b.second[4] + b.second[5] + b.second[6] + b.second[7]; });
				std::string line;
				u32 tot[8]{};
				for (const auto& [p, st] : top) for (u32 i = 0; i < 8; ++i) tot[i] += st[i];
				for (usz i = 0; i < std::min<usz>(top.size(), 8); ++i)
				{
					const auto& st = top[i].second;
					fmt::append(line, " %016llx: addr %u/%u nearest %u/%u order %u/%u none %u/%u;", top[i].first, st[4], st[0], st[5], st[1], st[6], st[2], st[7], st[3]);
				}
				dlss_log.notice("Static check (120 frames, wrong/draws): all addr %u/%u nearest %u/%u order %u/%u none %u/%u; worst programs:%s", tot[4], tot[0], tot[5], tot[1], tot[6], tot[2], tot[7], tot[3], line);
				m_static_check_stats.clear();
			}
		}
		for (u32 view = 0; view < 3; ++view)
		{
			m_static_valid[view] = false;
			u32 best = 0;
			for (const auto& [m, n] : m_reproject_votes[view])
			{
				if (n > best && n >= 3)
				{
					best = n;
					m_static_reproject[view] = m;
					m_static_valid[view] = true;
				}
			}
			m_reproject_votes[view].clear();
			m_view_draws[view] = 0;
		}
		m_matched = m_unmatched = m_camera_draws = 0;
		{
			std::lock_guard lock(s_test_mutex);
			s_test_presented.swap(m_test_draws);
			m_test_draws.clear();
		}

		if (!m_forced_address)
		{
			// The scene target: the target with the most perspective camera draws (counted for any target this frame).
			u32 best_count = 0, best_address = 0;
			for (const auto& [address, n] : m_target_counts)
			{
				if (n > best_count)
				{
					best_count = n;
					best_address = address;
				}
			}
			if (best_count && best_address != m_scene_address) // a frame without camera draws (menus, loading) keeps the target
			{
				dlss_log.notice("Scene target 0x%x (%u camera draws)", best_address, best_count);
				m_scene_address = best_address;
				m_reset_history = true;
			}
			m_target_counts.clear();
		}

		m_last_result = result;
		m_last_right = result_right;
		if (!result)
		{
			m_last_out_w = m_last_out_h = 0;
		}
		if (stereo && result_right)
		{
			*right_eye = result_right;
		}
		s_presented_frame = s_drawn_frames.load();
		s_drawn_frames++;

		m_scene_drawn = false;
		m_frame++;
		update_jitter();
		return result;
	}

	// ---- NGX --------------------------------------------------------------------------------------------

	bool dlss_context::ngx_init()
	{
#ifdef HAVE_DLSS
		if (m_ngx_ready)
		{
			return true;
		}
		if (m_disabled)
		{
			return false;
		}

		const std::string dir = fs::get_cache_dir() + "dlss/";
		fs::create_path(dir);
		const std::wstring wdir(dir.begin(), dir.end());

		const VkPhysicalDevice pdev = m_device.gpu();
		const VkInstance instance = m_device.gpu();
		NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID("6b1f3c5e-2d4a-4e8b-9f17-3a5c7e9d1b20", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
			wdir.c_str(), instance, pdev, m_device, vkGetInstanceProcAddr, vkGetDeviceProcAddr);
		if (NVSDK_NGX_FAILED(r))
		{
			dlss_log.error("NVSDK_NGX_VULKAN_Init failed (%s): is nvngx_dlss.dll next to rpcs3.exe, and the driver recent?", ngx_result_name(r));
			m_disabled = true;
			return false;
		}
		m_ngx_ready = true;

		NVSDK_NGX_Parameter* params = nullptr;
		r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&params);
		if (NVSDK_NGX_FAILED(r) || !params)
		{
			dlss_log.error("NVSDK_NGX_VULKAN_GetCapabilityParameters failed (%s)", ngx_result_name(r));
			m_disabled = true;
			return false;
		}
		m_ngx_params = params;

		int available = 0, needs_driver = 0;
		NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
		NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
		if (!available)
		{
			int feature_result = 0;
			NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &feature_result);
			dlss_log.error("DLSS is not available (needs a newer driver: %d, init result 0x%08x)", needs_driver, static_cast<u32>(feature_result));
			m_disabled = true;
			return false;
		}

		dlss_log.success("DLSS (NGX) ready");
		return true;
#else
		return false;
#endif
	}

	void dlss_context::ngx_release_features()
	{
#ifdef HAVE_DLSS
		if (std::any_of(m_ngx_features.begin(), m_ngx_features.end(), [](const ngx_feature& f) { return f.handle != nullptr; }))
		{
			vkDeviceWaitIdle(m_device);
			for (auto& f : m_ngx_features)
			{
				if (f.handle)
				{
					NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(f.handle));
				}
				f = {};
			}
		}
#endif
	}

	bool dlss_context::standin_evaluate(const vk::command_buffer& cmd, u32 eye, vk::viewable_image* source, u32 in_w, u32 in_h, u32 out_w, u32 out_h)
	{
		vk::viewable_image* const output = eye ? m_output_right.get() : m_output.get();
		if (!output || !source)
		{
			return false;
		}

		static bool s_reported = false;
		if (!std::exchange(s_reported, true))
		{
			dlss_log.warning("RPCS3_DLSS_STANDIN: a bilinear upscale stands in for DLSS (%ux%u -> %ux%u)", in_w, in_h, out_w, out_h);
		}

		// Layer 0 of the source: the left eye of a multiview surface, the only layer otherwise.
		source->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		output->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.srcOffsets[1] = {static_cast<s32>(std::min(in_w, source->width())), static_cast<s32>(std::min(in_h, source->height())), 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.dstOffsets[1] = {static_cast<s32>(out_w), static_cast<s32>(out_h), 1};
		vkCmdBlitImage(cmd, source->value, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, output->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		source->pop_layout(cmd);
		output->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		vk::insert_global_memory_barrier(cmd);
		return true;
	}

	bool dlss_context::ngx_evaluate(const vk::command_buffer& cmd, u32 eye, vk::viewable_image* source, u32 in_w, u32 in_h, u32 out_w, u32 out_h)
	{
		static const bool s_standin = env("RPCS3_DLSS_STANDIN") != nullptr;
		if (s_standin)
		{
			return standin_evaluate(cmd, eye, source, in_w, in_h, out_w, out_h);
		}

#ifdef HAVE_DLSS
		if (!ngx_init())
		{
			return false;
		}

		NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_DLAA;
		switch (m_mode)
		{
		case dlss_mode::quality: quality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
		case dlss_mode::balanced: quality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
		case dlss_mode::performance: quality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
		case dlss_mode::ultra_performance: quality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
		default: break;
		}

		auto* params = static_cast<NVSDK_NGX_Parameter*>(m_ngx_params);
		ngx_feature& feature = m_ngx_features[eye];

		// The render preset for every mode: VR / DLSS Model (test), or RPCS3_DLSS_PRESET=<letter> (310: J, K transformer;
		// L, M lighter; E, F the old convolutional ones, deprecated). Default: NGX's choice per mode (K for DLAA, Quality,
		// Balanced; M for Performance; L for Ultra Performance). A change makes the features again.
		u32 letter = 0;
		if (const char* preset = env("RPCS3_DLSS_PRESET"); preset && preset[0] >= 'A' && preset[0] <= 'O')
		{
			letter = static_cast<u8>(preset[0]);
		}
		else if (const vr_dlss_model model = g_cfg.video.vr.dlss_model; model != vr_dlss_model::default_model)
		{
			letter = static_cast<u8>(fmt::format("%s", model)[0]);
		}

		if (!feature.handle || feature.in[0] != in_w || feature.in[1] != in_h || feature.out[0] != out_w || feature.out[1] != out_h ||
			feature.quality != static_cast<u32>(quality) || feature.preset != letter)
		{
			if (feature.handle)
			{
				vkDeviceWaitIdle(m_device);
				NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(feature.handle));
				feature = {};
			}

			NVSDK_NGX_DLSS_Create_Params create{};
			create.Feature.InWidth = in_w;
			create.Feature.InHeight = in_h;
			create.Feature.InTargetWidth = out_w;
			create.Feature.InTargetHeight = out_h;
			create.Feature.InPerfQualityValue = quality;
			create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
			if (env("RPCS3_DLSS_DEPTH_INVERTED"))
			{
				create.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
			}
			create.InEnableOutputSubrects = false;

			// The hints stay on the shared parameters: written every time (0 asks for NGX's default again).
			for (const char* name : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
					 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
					 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance})
			{
				NVSDK_NGX_Parameter_SetUI(params, name, letter ? letter - 'A' + 1 : 0);
			}
			if (letter)
			{
				dlss_log.notice("render preset %c", static_cast<char>(letter));
			}

			NVSDK_NGX_Handle* handle = nullptr;
			const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &handle, params, &create);
			if (NVSDK_NGX_FAILED(r) || !handle)
			{
				dlss_log.error("Creating the DLSS feature failed (%s), %ux%u -> %ux%u", ngx_result_name(r), in_w, in_h, out_w, out_h);
				m_disabled = true;
				return false;
			}

			feature.handle = handle;
			feature.in[0] = in_w;
			feature.in[1] = in_h;
			feature.out[0] = out_w;
			feature.out[1] = out_h;
			feature.quality = static_cast<u32>(quality);
			feature.preset = letter;
			feature.fresh = true;
			dlss_log.notice("DLSS feature%s %ux%u -> %ux%u (quality %u), colour %ux%u format %u", eye ? " (right eye)" : "", in_w, in_h, out_w, out_h, feature.quality, source->width(), source->height(), static_cast<u32>(source->format()));
		}

		vk::viewable_image* const mv_image = eye ? m_mv_right.get() : m_mv.get();
		vk::viewable_image* const depth_image = eye ? m_depth_right.get() : m_depth.get();
		vk::viewable_image* const output_image = eye ? m_output_right.get() : m_output.get();

		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

		source->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		mv_image->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		depth_image->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		output_image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		const vk::image_view* color_view = source->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
		const vk::image_view* mv_view = mv_image->get_view(identity);
		const vk::image_view* depth_view = depth_image->get_view(identity);
		const vk::image_view* out_view = output_image->get_view(identity);

		NVSDK_NGX_Resource_VK color = NVSDK_NGX_Create_ImageView_Resource_VK(color_view->value, source->value, range, color_view->info.format, source->width(), source->height(), false);
		NVSDK_NGX_Resource_VK motion = NVSDK_NGX_Create_ImageView_Resource_VK(mv_view->value, mv_image->value, range, VK_FORMAT_R32G32_SFLOAT, mv_image->width(), mv_image->height(), false);
		NVSDK_NGX_Resource_VK depth = NVSDK_NGX_Create_ImageView_Resource_VK(depth_view->value, depth_image->value, range, VK_FORMAT_R32_SFLOAT, depth_image->width(), depth_image->height(), false);
		NVSDK_NGX_Resource_VK output = NVSDK_NGX_Create_ImageView_Resource_VK(out_view->value, output_image->value, range, output_image->format(), output_image->width(), output_image->height(), true);

		// Jitter in render pixels: the scene's jitter scaled from its viewport to the displayed image.
		static const auto jitter_sign = env_signs("RPCS3_DLSS_JITTER_SIGN");
		const f32 sx = m_clip_width ? static_cast<f32>(in_w) / static_cast<f32>(m_clip_width) : 1.f;
		const f32 sy = m_clip_height ? static_cast<f32>(in_h) / static_cast<f32>(m_clip_height) : 1.f;

		NVSDK_NGX_VK_DLSS_Eval_Params eval{};
		eval.Feature.pInColor = &color;
		eval.Feature.pInOutput = &output;
		eval.Feature.InSharpness = env_f32("RPCS3_DLSS_SHARPNESS", 0.f);
		eval.pInDepth = &depth;
		eval.pInMotionVectors = &motion;
		eval.InJitterOffsetX = m_jitter_px[0] * sx * jitter_sign[0];
		eval.InJitterOffsetY = m_jitter_px[1] * sy * jitter_sign[1];
		eval.InRenderSubrectDimensions = {in_w, in_h};
		eval.InReset = (m_reset_history || feature.fresh) ? 1 : 0;
		feature.fresh = false;
		eval.InMVScaleX = 1.f;
		eval.InMVScaleY = 1.f;
		eval.InColorSubrectBase = {0, 0};
		eval.InDepthSubrectBase = {0, 0};
		eval.InMVSubrectBase = {0, 0};
		eval.InOutputSubrectBase = {0, 0};
		eval.InPreExposure = 1.f;
		eval.InExposureScale = 1.f;

		const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, static_cast<NVSDK_NGX_Handle*>(feature.handle), params, &eval);
		source->pop_layout(cmd);

		// NGX recorded its own pipeline and descriptor state; make the results visible to the present passes.
		vk::insert_global_memory_barrier(cmd);

		if (NVSDK_NGX_FAILED(r))
		{
			static u32 s_failures = 0;
			if (s_failures++ < 10)
			{
				dlss_log.error("DLSS evaluation failed (%s)", ngx_result_name(r));
			}
			return false;
		}
		return true;
#else
		static_cast<void>(cmd);
		static_cast<void>(eye);
		static_cast<void>(source);
		static_cast<void>(in_w);
		static_cast<void>(in_h);
		static_cast<void>(out_w);
		static_cast<void>(out_h);
		return false;
#endif
	}
}
