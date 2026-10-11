#include "stdafx.h"
#include "VKDLSS.h"

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
#include "Utilities/File.h"
#include "Utilities/StrFmt.h"

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

		// The varyings after the RSX's (0-15) and the VR fork's (16 exact depth, 17-21 depth remap).
		constexpr int dlss_cur_location = 22;
		constexpr int dlss_prev_location = 23;
		constexpr int dlss_cover_location = 24;

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
					"layout(push_constant) uniform pc_block\n"
					"{\n"
					"	vec4 sizes;  // source region w, h; output w, h\n"
					"	vec4 params; // mv sign x, y; debug mode (1 motion, 2 depth); unused\n"
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
					"	const bool covered = m.w > 0.5;\n"
					"	const vec2 mv = covered ? m.xy * sizes.zw * params.xy : vec2(0.);\n"
					"	const float d = covered ? 1. - m.z : 1.;\n"
					"	imageStore(mv_out, p, vec4(mv, 0., 0.));\n"
					"	imageStore(depth_out, p, vec4(d));\n"
					"	if (params.z == 1.)\n"
					"	{\n"
					"		const float len = length(mv);\n"
					"		vec3 c = covered ? hue(atan(mv.y, mv.x) / 6.2831853 + 0.5) * clamp(len / 16., 0.15, 1.) : vec3(0.25);\n"
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
				push_constants_size = 32;
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
				const f32 constants[8])
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
				vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, constants);
				vkCmdDispatch(cmd, utils::aligned_div(static_cast<u32>(constants[2]), 16u), utils::aligned_div(static_cast<u32>(constants[3]), 16u), 1);
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
		atomic_t<u64> s_presented_frame{0}; // which of them the last presented output shows
	}

	u64 dlss_drawn_frames() { return s_drawn_frames.load(); }

	VkFormat dlss_motion_format(u32 samples)
	{
		return (s_rgba32f_samples & samples) || !(s_rgba16f_samples & samples) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
	}
	u64 dlss_presented_frame() { return s_presented_frame.load(); }

	dlss_mode dlss_configured_mode()
	{
		static const dlss_mode s_mode = []
		{
			const char* v = env("RPCS3_DLSS");
			if (!v)
			{
				return dlss_mode::off;
			}
			const std::string s = fmt::to_lower(v);
			if (s == "motion" || s == "mv") return dlss_mode::debug_motion;
			if (s == "depth") return dlss_mode::debug_depth;
			if (s == "motionraw") return dlss_mode::debug_motion_raw;
			if (s == "jitter") return dlss_mode::jitter_only;
			if (s == "dlaa" || s == "1" || s == "on") return dlss_mode::dlaa;
			if (s == "quality") return dlss_mode::quality;
			if (s == "balanced") return dlss_mode::balanced;
			if (s == "performance") return dlss_mode::performance;
			if (s == "ultra" || s == "ultra_performance") return dlss_mode::ultra_performance;
			return dlss_mode::off;
		}();
		return s_mode;
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
		if (!dlss_uses_ngx())
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
#else
		static_cast<void>(extensions);
#endif
	}

	void dlss_device_extensions(VkPhysicalDevice pdev, std::vector<const char*>& extensions)
	{
#ifdef HAVE_DLSS
		if (!dlss_uses_ngx())
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
				// Vulkan 1.2's bufferDeviceAddress (dlss_device_features) replaces it; both together are invalid.
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
#else
		static_cast<void>(pdev);
		static_cast<void>(extensions);
#endif
	}

	void dlss_device_features(VkPhysicalDevice pdev, VkPhysicalDeviceVulkan12Features& features)
	{
#ifdef HAVE_DLSS
		if (!dlss_uses_ngx())
		{
			return;
		}

		// NGX's Vulkan path uses buffer device addresses.
		VkPhysicalDeviceVulkan12Features supported{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
		VkPhysicalDeviceFeatures2 f2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &supported};
		vkGetPhysicalDeviceFeatures2(pdev, &f2);
		if (supported.bufferDeviceAddress)
		{
			features.bufferDeviceAddress = VK_TRUE;
		}
#else
		static_cast<void>(pdev);
		static_cast<void>(features);
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

	pipeline_props dlss_interpreter_props(const pipeline_props& props)
	{
		pipeline_props result = props;
		if (dlss_renderpass_has_motion(result.renderpass_key) && result.state.cs.attachmentCount > 0)
		{
			result.state.att_state[result.state.cs.attachmentCount - 1].colorWriteMask = 0;
		}
		return result;
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
		OS << "		dlss_motion = vec4((dlss_p - dlss_c) * 0.5, 1. - gl_FragCoord.z, dlss_cover);\n";
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

		if (m_mode == dlss_mode::debug_motion || m_mode == dlss_mode::debug_depth || m_mode == dlss_mode::debug_motion_raw || m_mode == dlss_mode::jitter_only)
		{
			m_jitter_phases = 8;
		}
		update_jitter();

		dlss_log.notice("DLSS test: mode %u (RPCS3_DLSS), scene target %s", static_cast<u32>(m_mode),
			m_forced_address ? fmt::format("0x%x", m_forced_address) : std::string("found per frame"));
#ifndef HAVE_DLSS
		if (dlss_uses_ngx())
		{
			dlss_log.error("This build has no DLSS SDK (HAVE_DLSS): motion vectors and jitter run, the image is shown without DLSS");
		}
#endif
	}

	dlss_context::~dlss_context()
	{
		destroy();
	}

	void dlss_context::destroy()
	{
		ngx_release_feature();
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
		m_motion.reset();
		m_mv.reset();
		m_depth.reset();
		m_debug.reset();
		m_output.reset();
	}

	void dlss_context::update_jitter()
	{
		if (m_mode == dlss_mode::off || m_jitter_scale == 0.f || m_mode == dlss_mode::debug_motion || m_mode == dlss_mode::debug_depth || m_mode == dlss_mode::debug_motion_raw)
		{
			m_jitter_px[0] = m_jitter_px[1] = 0.f;
			return;
		}
		const u32 index = static_cast<u32>(m_frame % m_jitter_phases) + 1;
		m_jitter_px[0] = (halton(index, 2) - 0.5f) * m_jitter_scale;
		m_jitter_px[1] = (halton(index, 3) - 0.5f) * m_jitter_scale;
	}

	bool dlss_context::find_camera_matrix(const u16* ids, usz count, mat4& out) const
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
		if (!color_address)
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

		if (m_motion && m_motion->width() == w && m_motion->height() == h && m_motion->samples() == color_surface->samples())
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
			VK_IMAGE_TYPE_2D, format, w, h, 1, 1, 1, samples, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_CREATE_ALLOW_NULL_RPCS3, VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_COLOR);

		if (!m_motion->value)
		{
			dlss_log.error("Out of memory for the motion image (%ux%u, %u samples)", w, h, static_cast<u32>(samples));
			m_motion.reset();
			return false;
		}

		m_motion->set_debug_name("DLSS motion");
		m_motion->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_motion_cleared_frame = umax;
		m_reset_history = true;
		dlss_log.notice("Motion image %ux%u, %u samples, %s (scene target 0x%x)", w, h, static_cast<u32>(samples),
			format == VK_FORMAT_R32G32B32A32_SFLOAT ? "RGBA32F" : "RGBA16F", m_scene_address);
		return true;
	}

	bool dlss_context::wants_motion(u32 color_address, bool vr_active) const
	{
		if (m_mode == dlss_mode::off || m_disabled || vr_active || !color_address)
		{
			return false;
		}
		return color_address == (m_forced_address ? m_forced_address : m_scene_address);
	}

	vk::image* dlss_context::bind_framebuffer(const vk::command_buffer& cmd, u32 color_address, vk::image* color_surface,
		u32 fbo_width, u32 fbo_height, u32 color_attachments, u32 mrt_count, bool vr_active)
	{
		static_cast<void>(fbo_width);
		static_cast<void>(fbo_height);
		m_bound = false;

		if (m_mode == dlss_mode::off || m_disabled || vr_active || !color_surface)
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
			const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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

		// First use this frame: no motion, far depth (stored as 1 - z), not covered.
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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

	void dlss_context::write_neutral_block(void* dst) const
	{
		f32* out = static_cast<f32*>(dst);
		std::memcpy(out, identity4, sizeof(identity4));
		out[16] = out[17] = out[18] = out[19] = 0.f;
	}

	void dlss_context::write_draw_block(void* dst, const u16* constant_ids, usz constant_count, u64 vp_hash,
		u32 clip_width, u32 clip_height, bool instanced)
	{
		f32* out = static_cast<f32*>(dst);
		std::memcpy(out, identity4, sizeof(identity4));
		out[16] = out[17] = out[18] = out[19] = 0.f;

		static const bool s_debug = env("RPCS3_DLSS_DEBUG") != nullptr;
		if (!m_bound || instanced || !clip_width || !clip_height)
		{
			if (s_debug)
				dlss_log.notice("Draw skipped: bound %d, instanced %d, clip %ux%u", m_bound, instanced, clip_width, clip_height);
			return;
		}

		mat4 cur;
		if (!find_camera_matrix(constant_ids, constant_count, cur))
		{
			if (s_debug)
				dlss_log.notice("Draw skipped: no camera matrix (%u constant ids)", static_cast<u32>(constant_count));
			return;
		}

		m_scene_drawn = true;
		m_clip_width = clip_width;
		m_clip_height = clip_height;
		m_camera_draws++;

		// Draw identity: program, vertex and index addresses, range, and how many such draws came before it.
		const auto& regs = rsx::method_registers;
		u64 key = mix(0xcbf29ce484222325ull, vp_hash);
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

		f32 r[16];
		std::memcpy(r, identity4, sizeof(r));
		bool have_r = false;

		if (const auto found = m_prev_draws.find(key); found != m_prev_draws.end())
		{
			f32 inv[16];
			if (invert4(cur.m, inv))
			{
				mul4(found->second.m, inv, r);
				have_r = true;
				m_matched++;

				// Votes for the camera-only reprojection (the same for every static object).
				bool voted = false;
				for (auto& [m, n] : m_reproject_votes)
				{
					if (near_equal(m.m, r))
					{
						n++;
						voted = true;
						break;
					}
				}
				if (!voted && m_reproject_votes.size() < 16)
				{
					mat4 v;
					std::memcpy(v.m, r, sizeof(r));
					m_reproject_votes.emplace_back(v, 1u);
				}
			}
		}

		if (!have_r)
		{
			m_unmatched++;
			if (m_static_valid)
			{
				std::memcpy(r, m_static_reproject.m, sizeof(r));
			}
		}

		if (s_debug)
		{
			const auto prev_it = m_prev_draws.find(key);
			dlss_log.notice("Draw key %016llx: M row0 (%.4f %.4f %.4f %.4f), previous %s, R row0 (%.4f %.4f %.4f %.4f)", key, cur.m[0], cur.m[1], cur.m[2], cur.m[3],
				prev_it != m_prev_draws.end() ? fmt::format("(%.4f %.4f %.4f %.4f)", prev_it->second.m[0], prev_it->second.m[1], prev_it->second.m[2], prev_it->second.m[3]) : std::string("none"),
				r[0], r[1], r[2], r[3]);
		}

		draw_record rec;
		std::memcpy(rec.m, cur.m, sizeof(rec.m));
		m_cur_draws[key] = rec;

		// Coverage: opaque surfaces (depth writes) and unblended depth-tested ones (the sky). Blended draws without depth
		// writes (particles, glass) keep the motion of what is behind them.
		const bool depth_write = regs.depth_test_enabled() && regs.depth_write_enabled();
		const bool opaque_test = regs.depth_test_enabled() && !regs.blend_enabled_mask();
		const f32 cover = (depth_write || opaque_test) ? 1.f : 0.f;

		std::memcpy(out, r, sizeof(r));
		out[16] = 2.f * m_jitter_px[0] / static_cast<f32>(clip_width);
		out[17] = 2.f * m_jitter_px[1] / static_cast<f32>(clip_height);
		out[18] = cover;
		out[19] = 0.f;
	}

	bool dlss_context::ensure_resolve_images(u32 width, u32 height)
	{
		if (m_mv && m_mv->width() == width && m_mv->height() == height)
		{
			return true;
		}

		m_last_result = nullptr;
		if (m_mv) vk::get_resource_manager()->dispose(m_mv);
		if (m_depth) vk::get_resource_manager()->dispose(m_depth);
		if (m_debug) vk::get_resource_manager()->dispose(m_debug);

		const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		m_mv = make_image(m_device, VK_FORMAT_R32G32_SFLOAT, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
		m_depth = make_image(m_device, VK_FORMAT_R32_SFLOAT, width, height, VK_SAMPLE_COUNT_1_BIT, usage);
		m_debug = make_image(m_device, VK_FORMAT_R8G8B8A8_UNORM, width, height, VK_SAMPLE_COUNT_1_BIT, usage);

		if (!m_mv->value || !m_depth->value || !m_debug->value)
		{
			dlss_log.error("Out of memory for the DLSS inputs (%ux%u)", width, height);
			m_mv.reset();
			m_depth.reset();
			m_debug.reset();
			return false;
		}
		m_reset_history = true;
		return true;
	}

	bool dlss_context::ensure_output_image(u32 width, u32 height)
	{
		if (m_output && m_output->width() == width && m_output->height() == height)
		{
			return true;
		}

		m_last_result = nullptr;
		if (m_output) vk::get_resource_manager()->dispose(m_output);

		// RGBA8: the present path (screenshots) expects 4 bytes per pixel.
		m_output = make_image(m_device, VK_FORMAT_R8G8B8A8_UNORM, width, height, VK_SAMPLE_COUNT_1_BIT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		if (!m_output->value)
		{
			m_output.reset();
			return false;
		}
		m_output->set_debug_name("DLSS output");
		return true;
	}

	void dlss_context::resolve(const vk::command_buffer& cmd, u32 width, u32 height)
	{
		// Every earlier write (motion attachment, the displayed image) before the compute reads.
		vk::insert_global_memory_barrier(cmd);

		m_mv->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_depth->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_debug->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		static const auto mv_sign = env_signs("RPCS3_DLSS_MV_SIGN");
		const f32 debug_mode = m_mode == dlss_mode::debug_motion ? 1.f : m_mode == dlss_mode::debug_depth ? 2.f : m_mode == dlss_mode::debug_motion_raw ? 3.f : 0.f;
		const f32 src_w = static_cast<f32>(std::min<u32>(m_clip_width ? m_clip_width : m_motion->width(), m_motion->width()));
		const f32 src_h = static_cast<f32>(std::min<u32>(m_clip_height ? m_clip_height : m_motion->height(), m_motion->height()));
		const f32 constants[8] = {src_w, src_h, static_cast<f32>(width), static_cast<f32>(height), mv_sign[0], mv_sign[1], debug_mode, 0.f};

		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const vk::image_view* mv = m_mv->get_view(identity);
		const vk::image_view* depth = m_depth->get_view(identity);
		const vk::image_view* debug = m_debug->get_view(identity);

		if (m_motion->samples() > 1)
		{
			const vk::image_view* src = m_motion->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_VIEW_MULTISAMPLED), VK_IMAGE_ASPECT_COLOR_BIT);
			vk::get_compute_task<dlss_resolve_pass<true>>()->run(cmd, src, mv, depth, debug, constants);
		}
		else
		{
			const vk::image_view* src = m_motion->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
			vk::get_compute_task<dlss_resolve_pass<false>>()->run(cmd, src, mv, depth, debug, constants);
		}

		vk::insert_global_memory_barrier(cmd);
	}

	vk::viewable_image* dlss_context::on_flip(const vk::command_buffer& cmd, vk::viewable_image* source, u32 width, u32 height,
		u32 target_width, u32 target_height, bool emu_flip, bool vr_active, u32& out_w, u32& out_h)
	{
		if (m_mode == dlss_mode::off)
		{
			return nullptr;
		}

		if (!emu_flip || !m_frame_camera_draws)
		{
			// A UI refresh, or a flip without any 3D drawn since the last one (a game showing the same frame again,
			// a capture replay's empty loop): show the last output again; history, jitter and frame count stay.
			if (m_last_result && m_last_out_w && m_last_out_h && !m_disabled && !vr_active)
			{
				out_w = m_last_out_w;
				out_h = m_last_out_h;
				return m_last_result;
			}
			return nullptr;
		}
		m_frame_camera_draws = 0;

		vk::viewable_image* result = nullptr;
		const bool usable = !m_disabled && !vr_active && source && width && height && m_motion && m_scene_drawn &&
			m_motion_cleared_frame == m_frame;

		if (usable && m_mode != dlss_mode::jitter_only && ensure_resolve_images(width, height))
		{
			resolve(cmd, width, height);

			if (m_mode == dlss_mode::debug_motion || m_mode == dlss_mode::debug_depth || m_mode == dlss_mode::debug_motion_raw)
			{
				result = m_debug.get();
				out_w = m_last_out_w = width;
				out_h = m_last_out_h = height;
			}
			else
			{
				// Output at the window's size for this image (upscale up to 3x, never down: DLAA then).
				f32 scale = std::min(static_cast<f32>(target_width) / static_cast<f32>(width), static_cast<f32>(target_height) / static_cast<f32>(height));
				if (m_mode == dlss_mode::dlaa || !std::isfinite(scale))
				{
					scale = 1.f;
				}
				scale = std::clamp(scale, 1.f, 3.f);
				const u32 ow = static_cast<u32>(std::lround(width * scale));
				const u32 oh = static_cast<u32>(std::lround(height * scale));

				if (ensure_output_image(ow, oh) && ngx_evaluate(cmd, source, width, height, ow, oh))
				{
					result = m_output.get();
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
			dlss_log.notice("Frame %llu: scene 0x%x, camera draws %u (matched %u, unmatched %u), static reprojection %s, clip %ux%u, output %s",
				m_frame, m_forced_address ? m_forced_address : m_scene_address, m_camera_draws, m_matched, m_unmatched,
				m_static_valid ? "found" : "none", m_clip_width, m_clip_height, result ? "replaced" : "unchanged");
		}

		m_prev_draws.swap(m_cur_draws);
		m_cur_draws.clear();
		m_occurrences.clear();

		m_static_valid = false;
		u32 best = 0;
		for (const auto& [m, n] : m_reproject_votes)
		{
			if (n > best && n >= 3)
			{
				best = n;
				m_static_reproject = m;
				m_static_valid = true;
			}
		}
		m_reproject_votes.clear();
		m_matched = m_unmatched = m_camera_draws = 0;

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
		if (!result)
		{
			m_last_out_w = m_last_out_h = 0;
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

	void dlss_context::ngx_release_feature()
	{
#ifdef HAVE_DLSS
		if (m_ngx_feature)
		{
			vkDeviceWaitIdle(m_device);
			NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(m_ngx_feature));
			m_ngx_feature = nullptr;
		}
#endif
	}

	bool dlss_context::ngx_evaluate(const vk::command_buffer& cmd, vk::viewable_image* source, u32 in_w, u32 in_h, u32 out_w, u32 out_h)
	{
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
		if (!m_ngx_feature || m_feature_in[0] != in_w || m_feature_in[1] != in_h || m_feature_out[0] != out_w || m_feature_out[1] != out_h ||
			m_feature_quality != static_cast<u32>(quality))
		{
			ngx_release_feature();

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

			NVSDK_NGX_Handle* handle = nullptr;
			const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &handle, params, &create);
			if (NVSDK_NGX_FAILED(r) || !handle)
			{
				dlss_log.error("Creating the DLSS feature failed (%s), %ux%u -> %ux%u", ngx_result_name(r), in_w, in_h, out_w, out_h);
				m_disabled = true;
				return false;
			}

			m_ngx_feature = handle;
			m_feature_in[0] = in_w;
			m_feature_in[1] = in_h;
			m_feature_out[0] = out_w;
			m_feature_out[1] = out_h;
			m_feature_quality = static_cast<u32>(quality);
			m_reset_history = true;
			dlss_log.notice("DLSS feature %ux%u -> %ux%u (quality %u)", in_w, in_h, out_w, out_h, m_feature_quality);
		}

		const auto identity = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
		const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

		source->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_mv->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_depth->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_output->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		const vk::image_view* color_view = source->get_view(identity, VK_IMAGE_ASPECT_COLOR_BIT);
		const vk::image_view* mv_view = m_mv->get_view(identity);
		const vk::image_view* depth_view = m_depth->get_view(identity);
		const vk::image_view* out_view = m_output->get_view(identity);

		NVSDK_NGX_Resource_VK color = NVSDK_NGX_Create_ImageView_Resource_VK(color_view->value, source->value, range, color_view->info.format, source->width(), source->height(), false);
		NVSDK_NGX_Resource_VK motion = NVSDK_NGX_Create_ImageView_Resource_VK(mv_view->value, m_mv->value, range, VK_FORMAT_R32G32_SFLOAT, m_mv->width(), m_mv->height(), false);
		NVSDK_NGX_Resource_VK depth = NVSDK_NGX_Create_ImageView_Resource_VK(depth_view->value, m_depth->value, range, VK_FORMAT_R32_SFLOAT, m_depth->width(), m_depth->height(), false);
		NVSDK_NGX_Resource_VK output = NVSDK_NGX_Create_ImageView_Resource_VK(out_view->value, m_output->value, range, m_output->format(), m_output->width(), m_output->height(), true);

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
		eval.InReset = m_reset_history ? 1 : 0;
		eval.InMVScaleX = 1.f;
		eval.InMVScaleY = 1.f;
		eval.InColorSubrectBase = {0, 0};
		eval.InDepthSubrectBase = {0, 0};
		eval.InMVSubrectBase = {0, 0};
		eval.InOutputSubrectBase = {0, 0};
		eval.InPreExposure = 1.f;
		eval.InExposureScale = 1.f;

		const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, static_cast<NVSDK_NGX_Handle*>(m_ngx_feature), params, &eval);
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
		static_cast<void>(source);
		static_cast<void>(in_w);
		static_cast<void>(in_h);
		static_cast<void>(out_w);
		static_cast<void>(out_h);
		return false;
#endif
	}
}
