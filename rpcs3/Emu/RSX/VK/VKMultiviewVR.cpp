#include "stdafx.h"
#include "VKMultiviewVR.h"

#include "VKHelpers.h"
#include "VKProgramPipeline.h"
#include "vkutils/descriptors.h"
#include "vkutils/device.h"
#include "vkutils/instance.h"
#include "vkutils/query_pool.hpp"
#include "VKQueryPool.h"
#include "VKRenderPass.h"
#include "VKRenderTargets.h"
#include "vkutils/image.h"

#include "Emu/RSX/Program/GLSLTypes.h"
#include "Emu/RSX/Program/ShaderInterpreter.h"
#include "Emu/RSX/Program/ShaderParam.h"
#include "Utilities/StrFmt.h"
#include "Emu/RSX/gcm_enums.h" // RSX_SHADER_CONTROL_INSTANCED_CONSTANTS
#include "../Capture/rsx_camera_probe.h" // profile depth_remap_ray_texcoord, depth_remap_xyw

#include <bit>
#include <type_traits>

// VR fork: multiview stereo, the bodies of the hooks in upstream's Vulkan files (see VKMultiviewVR.h), and the
// multiview members of upstream classes (image_view::as_array, the query pool's pairs), defined here so that
// the upstream files carry only their declarations.

static_assert(std::is_same_v<std::underlying_type_t<program_common::interpreter::compiler_option>, u32>,
	"VR fork: the interpreter's compiler options grew past 32 bits; move vk::COMPILER_OPT_VR_MULTIVIEW above them");

namespace
{
	// Merge guards. The multiview variants depend on upstream GLSL by its text: the _VR_MULTIVIEW blocks at the
	// end of RSXFragmentTextureOps.glsl and RSXFragmentTextureDepthConversion.glsl redefine the macros below with
	// a layer coordinate (copies of upstream's with the layer added), and the interpreter variants patch the call
	// sites below. A merge from upstream that changes one of them stops the build here instead of leaving the VR
	// copy behind: update the copy (or the patch in vr_insert_interpreter_*) to match, then the text here.
	constexpr std::string_view upstream_texture_ops =
#include "Emu/RSX/Program/GLSLSnippets/RSXProg/RSXFragmentTextureOps.glsl"
		;
	constexpr std::string_view upstream_depth_conversion =
#include "Emu/RSX/Program/GLSLSnippets/RSXProg/RSXFragmentTextureDepthConversion.glsl"
		;
	constexpr std::string_view upstream_vertex_interpreter =
#include "Emu/RSX/Program/GLSLInterpreter/VertexInterpreter.glsl"
		;
	constexpr std::string_view upstream_fragment_interpreter =
#include "Emu/RSX/Program/GLSLInterpreter/FragmentInterpreter.glsl"
		;

	constexpr bool has(std::string_view text, std::string_view line)
	{
		return text.find(line) != std::string_view::npos;
	}

	static_assert(has(upstream_texture_ops, "#define TEX2D(index, coord2) _process_texel(texture(TEX_NAME(index), COORD_SCALE2(index, coord2)), TEX_FLAGS(index))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_BIAS(index, coord2, bias) _process_texel(texture(TEX_NAME(index), COORD_SCALE2(index, coord2), bias), TEX_FLAGS(index))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_LOD(index, coord2, lod) _process_texel(textureLod(TEX_NAME(index), COORD_SCALE2(index, coord2), lod), TEX_FLAGS(index))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_GRAD(index, coord2, dpdx, dpdy) _process_texel(textureGrad(TEX_NAME(index), COORD_SCALE2(index, coord2), dpdx, dpdy), TEX_FLAGS(index))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_PROJ(index, coord4) _process_texel(texture(TEX_NAME(index), COORD_PROJ2(index, coord4.xyw)), TEX_FLAGS(index))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_SHADOW(index, coord3) texture(TEX_NAME(index), SHADOW_COORD(index, coord3))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_SHADOWPROJ(index, coord4) texture(TEX_NAME(index), SHADOW_COORD_PROJ(index, coord4))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_SHADOW(index, coord3) texture(TEX_NAME(index), vec3(COORD_SCALE2(index, coord3.xy), coord3.z))"));
	static_assert(has(upstream_texture_ops, "#define TEX2D_SHADOWPROJ(index, coord4) texture(TEX_NAME(index), COORD_PROJ3_SHADOW(index, coord4))"));
	static_assert(has(upstream_depth_conversion, "#define ZS_READ(index, coord) vec2(texture(TEX_NAME(index), coord).r, float(texture(TEX_NAME_STENCIL(index), coord).x))"));
	static_assert(has(upstream_depth_conversion, "#define TEX2D_Z24X8_RGBA8(index, coord2) _process_texel(convert_z24x8_to_rgba8(ZS_READ(index, COORD_SCALE2(index, coord2)), TEX_PARAM(index).remap, TEX_FLAGS(index)), TEX_FLAGS(index))"));
	static_assert(has(upstream_vertex_interpreter, "	gl_Position = pos;"));
	static_assert(has(upstream_fragment_interpreter, "texture(SAMPLER2D(ur0), coord.xy, bias)"));
	static_assert(has(upstream_fragment_interpreter, "textureLod(SAMPLER2D(ur0), coord.xy, lod)"));
} // namespace

namespace vk
{
	// While set, new render targets get two layers (layer 1 = right eye). See VKRenderTargets.h.
	bool g_vr_stereo_layers = false;

	// ---- Render passes ----------------------------------------------------------------------------

	const void* vr_renderpass_multiview_info(u32 key_view_mask)
	{
		// The views of one pass are spatially correlated (a hint), so the correlation mask is the view mask.
		static constexpr u32 view_masks[4] = {0u, 0b11u, 0b01u, 0b10u};
		static constexpr VkRenderPassMultiviewCreateInfo infos[4] =
			{
				{},
				{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO, .subpassCount = 1, .pViewMasks = &view_masks[1], .correlationMaskCount = 1, .pCorrelationMasks = &view_masks[1]},
				{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO, .subpassCount = 1, .pViewMasks = &view_masks[2], .correlationMaskCount = 1, .pCorrelationMasks = &view_masks[2]},
				{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO, .subpassCount = 1, .pViewMasks = &view_masks[3], .correlationMaskCount = 1, .pCorrelationMasks = &view_masks[3]},
			};

		const u32 index = key_view_mask & 3;
		return index ? &infos[index] : nullptr;
	}

	// ---- Images -----------------------------------------------------------------------------------

	// Render-target reloads from guest memory, and atlases and mip chains built from render-target sections: the
	// memory load writes layer 0, and layer 1 takes a copy (before the sections write each eye's pixels; left
	// alone, layer 1 kept whatever the pooled image last held: Kingdom Hearts II's hair flickered in the right eye).
	void vr_copy_left_to_right_layer(vk::command_buffer& cmd, vk::image* image, u8 mipmaps)
	{
		if (!image->stereo_layers || image->layers() < 2)
		{
			return;
		}
		const areai whole{0, 0, static_cast<s32>(image->width()), static_cast<s32>(image->height())};
		vk::copy_image(cmd, image, image, whole, whole, {.mipmap_count = mipmaps, .src_layer = 0, .dst_layer = 1});
	}

	image_view* image_view::as_array()
	{
		if (info.viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY)
		{
			return this;
		}

		if (!m_resource)
		{
			return this; // a view without its image (framebuffer attachments) is never sampled
		}

		if (!m_array_view)
		{
			// Through the constructor that keeps the image: descriptors read the view's image for its layout and id.
			VkImageSubresourceRange range = info.subresourceRange;
			range.baseArrayLayer = 0;
			range.layerCount = m_resource->layers();
			m_array_view = std::make_unique<vk::image_view>(m_device, m_resource, info.format, VK_IMAGE_VIEW_TYPE_2D_ARRAY, info.components, range);
		}

		return m_array_view.get();
	}

	// ---- Occlusion queries: a query begun in a two-view pass takes two consecutive slots -------------

	u32 query_pool_manager::allocate_query_pair(vk::command_buffer& cmd)
	{
		if (m_pool_lifetime_counter < 2)
		{
			if (vk::is_renderpass_open(cmd))
			{
				vk::end_renderpass(cmd);
			}

			// A pool holds one reference per slot until the slot is freed; a slot this pool will not hand out any
			// more keeps it alive for good (the discard pile grew: "Are we leaking??"). Drop those first.
			for (; m_pool_lifetime_counter && m_current_query_pool; --m_pool_lifetime_counter)
			{
				m_current_query_pool->release();
			}

			reallocate_pool(cmd);
		}

		// Two consecutive free slots at the front of the list: the list starts in order, a pair is freed as a
		// pair (free_query), and a lone slot in the way is rotated to the back. No allocation: this runs for
		// every query segment (a pass end ends the open pair; the next draw begins another).
		for (usz tries = 0, count = m_available_slots.size(); tries < count && m_available_slots.size() >= 2; ++tries)
		{
			const u32 first = m_available_slots.front();
			m_available_slots.pop_front();
			if (m_available_slots.front() == first + 1)
			{
				m_available_slots.pop_front();
				m_pool_lifetime_counter -= 2;
				return first;
			}
			m_available_slots.push_back(first);
		}

		return ~0u;
	}

	void query_pool_manager::begin_query_pair(vk::command_buffer& cmd, u32 index)
	{
		// The head is begun as a single query (one begin covers both views' slots, index and index + 1);
		// the second slot only records the pool its result is read from, and is freed with the head.
		ensure(query_slot_status[index + 1].active == false);
		begin_query(cmd, index);
		query_slot_status[index].pair_head = true;
		query_slot_status[index + 1].pool = m_current_query_pool.get();
		query_slot_status[index + 1].active = true;
	}

	// ---- Shader generation ------------------------------------------------------------------------

	std::string_view vr_sampler_type(std::string_view type, u32 ctrl)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW))
		{
			return type;
		}

		// The layer is the eye (gl_ViewIndex); a one-layer texture clamps to layer 0.
		if (type == "sampler2D")
			return "sampler2DArray";
		if (type == "sampler2DMS")
			return "sampler2DMSArray";
		if (type == "sampler2DShadow")
			return "sampler2DArrayShadow";
		return type;
	}

	void vr_set_sampler_type(std::string& type, u32 ctrl)
	{
		if (const std::string_view array_type = vr_sampler_type(type, ctrl); array_type.data() != type.data())
		{
			type = array_type;
		}
	}

	void vr_insert_fragment_extensions(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW)
		{
			// gl_ViewIndex selects the eye's layer of every 2D texture
			OS << "#extension GL_EXT_multiview: require\n";
		}
	}

	void vr_insert_fragment_defines(std::ostream& OS, ::glsl::shader_properties& props, u32 ctrl)
	{
		props.vr_multiview = !!(ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW);
		if (props.vr_multiview)
		{
			OS << "#define _VR_MULTIVIEW\n\n";
		}
	}

	std::string_view vr_vertex_function(FUNCTION f, u32 ctrl)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW))
		{
			return {};
		}

		// 2D vertex textures are array samplers; the layer is the eye
		switch (f)
		{
		case FUNCTION::VERTEX_TEXTURE_FETCH2D:
			return "textureLod($t, vec3($0.xy, float(gl_ViewIndex)), 0)";
		case FUNCTION::VERTEX_TEXTURE_FETCH2DMS:
			return "texelFetch($t, ivec3(ivec2($0.xy * textureSize($t).xy), gl_ViewIndex), 0)";
		default:
			return {};
		}
	}

	void vr_insert_vertex_extensions(std::ostream& OS, u32 ctrl, bool viewport_index)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW))
		{
			return;
		}

		// Both eyes in one draw; each view reads its own draw parameters (and scissor)
		OS << "#extension GL_EXT_multiview : require\n";
		if (viewport_index)
		{
			OS << "#extension GL_ARB_shader_viewport_layer_array : require\n";
		}
		OS << "\n";
	}

	void vr_insert_vertex_draw_params(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW)
		{
			// The two views' entries are adjacent (left, right): each view reads its own.
			OS << "#undef get_draw_params\n"
				  "#define get_draw_params() draw_parameters[draw_parameters_offset + uint(gl_ViewIndex)]\n\n";
		}
	}

	void vr_insert_vertex_main_end(std::ostream& OS, u32 ctrl, bool viewport_index)
	{
		if ((ctrl & RSX_SHADER_CONTROL_VR_MULTIVIEW) && viewport_index)
		{
			// Each view clips to its own scissor (the HUD box differs per eye)
			OS << "	gl_ViewportIndex = int(gl_ViewIndex);\n\n";
		}
	}

	// The varyings after the RSX's (locations 0-15): exact depth, then the depth remap's 5 (17 to 21)
	static constexpr int vr_exact_depth_location = 16;
	static constexpr int vr_depth_remap_location = 17;

	void vr_insert_exact_depth_vertex_output(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_EXACT_DEPTH)
		{
			OS << "layout(location=" << vr_exact_depth_location << ") out vec2 vr_exact_depth;\n";
		}
	}

	void vr_insert_exact_depth_vertex_end(std::ostream& OS, u32 ctrl)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_VR_EXACT_DEPTH))
		{
			return;
		}
		// The game's window depth at this vertex, weighted by the game's own w (vr_pre_xform: the position before the
		// viewport or HUD-box matrix). The rasterizer interpolates varyings in the space gl_Position spans, a linear map
		// of the game's clip space, so x / y is the game's depth at each pixel exactly. Unrestricted depth range: the
		// viewport maps the zclip output [0, 1] onto [z_near, z_far].
		const bool unrestricted = vk::get_current_renderer()->get_unrestricted_depth_range_support();
		OS << "	{\n";
		OS << "		const float vr_d = gl_Position.w != 0. ? gl_Position.z / gl_Position.w : 0.;\n";
		OS << (unrestricted ? "		vr_exact_depth = vec2((z_near + vr_d * (z_far - z_near)) * vr_pre_xform.w, vr_pre_xform.w);\n"
		                    : "		vr_exact_depth = vec2(vr_d * vr_pre_xform.w, vr_pre_xform.w);\n");
		OS << "	}\n";
	}

	void vr_insert_depth_remap_vertex_output(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_DEPTH_REMAP)
		{
			OS << "layout(location=" << vr_depth_remap_location << ") flat out vec4 vr_depth_remap[5];\n";
		}
	}

	void vr_insert_depth_remap_vertex_end(std::ostream& OS, u32 ctrl, u32 constant_slots)
	{
		if (!(ctrl & RSX_SHADER_CONTROL_VR_DEPTH_REMAP))
		{
			return;
		}
		// Instanced constants come from another buffer, without the matrix: the identity (the game's own reconstruction).
		if (ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS)
		{
			OS << "	vr_depth_remap = vec4[4](vec4(1., 0., 0., 0.), vec4(0., 1., 0., 0.), vec4(0., 0., 1., 0.), vec4(0., 0., 0., 1.), vec4(0.));\n";
			return;
		}
		OS << "	for (uint vr_i = 0; vr_i < 5; ++vr_i) vr_depth_remap[vr_i] = vc[get_draw_params().xform_constants_offset + " << constant_slots << "u + vr_i];\n";
	}

	void vr_insert_depth_remap_fragment_input(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_DEPTH_REMAP)
		{
			OS << "layout(location=" << vr_depth_remap_location << ") flat in vec4 vr_depth_remap[5]; // VR fork\n";
		}
	}

	void vr_insert_depth_remap_fragment_start(std::ostream& OS, u32 ctrl, u32 in_register_mask, u32 depth_mask)
	{
		const u32 tc_mask = (in_register_mask >> 4) & 0x3ff; // FragmentProgramDecompiler's in_tc0 (bit 4) to in_tc9
		if (!(ctrl & RSX_SHADER_CONTROL_VR_DEPTH_REMAP) || !tc_mask || !depth_mask)
		{
			return;
		}
		const std::string tc = "tc" + std::to_string(std::countr_zero(tc_mask));
		const std::string unit = std::to_string(std::countr_zero(depth_mask));
		const auto* profile = rsx::vr::camera_probe::get().profile();
		if (profile && !profile->depth_remap_uv && profile->depth_remap_ray_texcoord >= 0 && (tc_mask >> profile->depth_remap_ray_texcoord & 1))
		{
			// Ray variant (profile depth_remap_ray_texcoord): the pass rebuilds a view-space position as its view ray
			// (a varying built with the game's view, not a camera block) scaled to the depth's view distance, with its
			// texture coordinates from the clip position. The clip position stays the eye's (it picks the texels), the
			// depth texel under the pixel is read at the depth texture's own size (passes drawn below its resolution)
			// and remapped to the game camera, and the ray becomes the game camera's ray to that point: (x / A, y / B, 1)
			// of its game NDC, the camera probe's view frame (u = (X/A, Y/B, W), see remap_to_eye_fov; only the ray's
			// direction up to sign matters, as such passes divide by its z). Sonic & All-Stars Racing Transformed's shadow cascades.
			const std::string ray = "tc" + std::to_string(profile->depth_remap_ray_texcoord);
			const std::string w = profile->depth_remap_xyw ? tc + ".z" : tc + ".w";
			OS << "\n	// VR fork: depth remap, ray variant (profile depth_remap_programs + depth_remap_ray_texcoord)\n";
			OS << "	vec4 vr_remap_ray;\n";
			OS << "	vec2 vr_remap_zs;\n";
			OS << "	{\n";
			OS << "		const vec2 vr_scale = vec2(textureSize(tex" << unit << ", 0).xy) / max(vr_depth_remap[4].zw, vec2(1.));\n";
			OS << "#ifdef _VR_MULTIVIEW\n";
			OS << "		const ivec3 vr_px = ivec3(ivec2(gl_FragCoord.xy * vr_scale), gl_ViewIndex);\n";
			OS << "#else\n";
			OS << "		const ivec2 vr_px = ivec2(gl_FragCoord.xy * vr_scale);\n";
			OS << "#endif\n";
			OS << "		const float vr_w = " << w << " != 0. ? " << w << " : 1.;\n";
			OS << "		const vec4 vr_eye = vec4(" << tc << ".xy / vr_w, texelFetch(tex" << unit << ", vr_px, 0).r, 1.);\n";
			OS << "		const vec4 vr_g = vr_eye.x * vr_depth_remap[0] + vr_eye.y * vr_depth_remap[1] + vr_eye.z * vr_depth_remap[2] + vr_depth_remap[3];\n";
			OS << "		const float vr_gw = vr_g.w != 0. ? vr_g.w : 1.;\n";
			OS << "		vr_remap_zs = vec2(vr_g.w != 0. ? clamp(vr_g.z / vr_g.w, 0., 1.) : vr_eye.z, float(texelFetch(tex" << unit << "_stencil, vr_px, 0).x));\n";
			OS << "		vr_remap_ray = vr_depth_remap[4].x != 0. ? vec4(vr_g.x / vr_gw * vr_depth_remap[4].x, vr_g.y / vr_gw * vr_depth_remap[4].y, 1., 0.) : " << ray << ";\n";
			OS << "	}\n";
			OS << "#define " << ray << " vr_remap_ray\n";
			OS << "#undef TEX2D_Z24X8_RGBA8\n";
			OS << "#ifdef _VR_MULTIVIEW\n";
			OS << "#define TEX2D_Z24X8_RGBA8(index, coord2) _process_texel(convert_z24x8_to_rgba8((index == " << unit << ") ? vr_remap_zs : ZS_READ2D(index, COORD_SCALE2(index, coord2)), TEX_PARAM(index).remap, TEX_FLAGS(index)), TEX_FLAGS(index))\n";
			OS << "#else\n";
			OS << "#define TEX2D_Z24X8_RGBA8(index, coord2) _process_texel(convert_z24x8_to_rgba8((index == " << unit << ") ? vr_remap_zs : ZS_READ(index, COORD_SCALE2(index, coord2)), TEX_PARAM(index).remap, TEX_FLAGS(index)), TEX_FLAGS(index))\n";
			OS << "#endif\n\n";
			return;
		}
		// The eye's (NDC x, NDC y, window depth) at this pixel, times the matrix: the game's, homogeneous. The texture
		// coordinate holds the pass's clip position (the eye's: the camera block took the eye transform), and the
		// depth texture is the pass's depth buffer, the size of its render target.
		OS << "\n	// VR fork: depth remap (profile depth_remap_programs), before the program reads its position and depth\n";
		OS << "	vec4 vr_remap_position;\n";
		OS << "	vec2 vr_remap_zs;\n";
		OS << "	{\n";
		OS << "#ifdef _VR_MULTIVIEW\n";
		OS << "		const ivec3 vr_px = ivec3(ivec2(gl_FragCoord.xy), gl_ViewIndex);\n";
		OS << "#else\n";
		OS << "		const ivec2 vr_px = ivec2(gl_FragCoord.xy);\n";
		OS << "#endif\n";
		OS << "		const float vr_w = " << tc << ".w != 0. ? " << tc << ".w : 1.;\n";
		if (profile && profile->depth_remap_uv)
		{
			// Screen uv (y down) to NDC, remap to the game camera, and back to uv (w = 1: the program divides by it).
			OS << "		const vec2 vr_uv = " << tc << ".xy / vr_w;\n";
			OS << "		const vec4 vr_eye = vec4(vr_uv.x * 2. - 1., 1. - vr_uv.y * 2., texelFetch(tex" << unit << ", vr_px, 0).r, 1.);\n";
		}
		else
		{
			OS << "		const vec4 vr_eye = vec4(" << tc << ".xy / vr_w, texelFetch(tex" << unit << ", vr_px, 0).r, 1.);\n";
		}
		OS << "		vr_remap_position = vr_eye.x * vr_depth_remap[0] + vr_eye.y * vr_depth_remap[1] + vr_eye.z * vr_depth_remap[2] + vr_depth_remap[3];\n";
		OS << "		vr_remap_zs = vec2(vr_remap_position.w != 0. ? clamp(vr_remap_position.z / vr_remap_position.w, 0., 1.) : vr_eye.z, float(texelFetch(tex" << unit << "_stencil, vr_px, 0).x));\n";
		if (profile && profile->depth_remap_uv)
		{
			OS << "		const vec2 vr_ndc_g = vr_remap_position.xy / (vr_remap_position.w != 0. ? vr_remap_position.w : 1.);\n";
			if (in_register_mask & 1) // FragmentProgramDecompiler's in_wpos: move the window position by the NDC change
			{
				OS << "		const vec2 vr_dn = vec2(dFdx(vr_eye.x), dFdy(vr_eye.y));\n";
				OS << "		const vec2 vr_k = vec2(abs(vr_dn.x) > 1e-12 ? dFdx(wpos.x) / vr_dn.x : 0., abs(vr_dn.y) > 1e-12 ? dFdy(wpos.y) / vr_dn.y : 0.);\n";
				OS << "		wpos.xy += (vr_ndc_g - vr_eye.xy) * vr_k;\n";
			}
			OS << "		vr_remap_position = vec4((vr_ndc_g.x + 1.) * 0.5, (1. - vr_ndc_g.y) * 0.5, " << tc << ".z, 1.);\n";
		}
		OS << "	}\n";
		OS << "#define " << tc << " vr_remap_position\n";
		OS << "#undef TEX2D_Z24X8_RGBA8\n";
		OS << "#ifdef _VR_MULTIVIEW\n";
		OS << "#define TEX2D_Z24X8_RGBA8(index, coord2) _process_texel(convert_z24x8_to_rgba8((index == " << unit << ") ? vr_remap_zs : ZS_READ2D(index, COORD_SCALE2(index, coord2)), TEX_PARAM(index).remap, TEX_FLAGS(index)), TEX_FLAGS(index))\n";
		OS << "#else\n";
		OS << "#define TEX2D_Z24X8_RGBA8(index, coord2) _process_texel(convert_z24x8_to_rgba8((index == " << unit << ") ? vr_remap_zs : ZS_READ(index, COORD_SCALE2(index, coord2)), TEX_PARAM(index).remap, TEX_FLAGS(index)), TEX_FLAGS(index))\n";
		OS << "#endif\n\n";
	}

	void vr_insert_exact_depth_fragment_input(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_EXACT_DEPTH)
		{
			OS << "layout(location=" << vr_exact_depth_location << ") in vec2 vr_exact_depth;\n";
		}
	}

	void vr_insert_exact_depth_fragment_end(std::ostream& OS, u32 ctrl)
	{
		if (ctrl & RSX_SHADER_CONTROL_VR_EXACT_DEPTH)
		{
			OS << "	gl_FragDepth = vr_exact_depth.y != 0. ? clamp(vr_exact_depth.x / vr_exact_depth.y, 0., 1.) : gl_FragCoord.z;\n\n";
		}
	}

	void vr_insert_interpreter_vertex(std::ostream& OS, const std::string& vertex_interpreter, bool viewport_index)
	{
		if (!viewport_index)
		{
			OS << vertex_interpreter;
			return;
		}

		// Each view clips to its own scissor (the HUD box differs per eye)
		const std::string text = fmt::replace_all(vertex_interpreter, "	gl_Position = pos;\n", "	gl_Position = pos;\n	gl_ViewportIndex = int(gl_ViewIndex);\n");
		if (text.size() == vertex_interpreter.size())
		{
			// Guard for merges from upstream: without it both views would clip to the left eye's scissor.
			rsx_log.error("VR multiview: the vertex interpreter's gl_Position write changed; update vr_insert_interpreter_vertex");
		}
		OS << text;
	}

	void vr_insert_interpreter_fragment_extensions(std::ostream& OS, u64 compiler_options)
	{
		if (compiler_options & COMPILER_OPT_VR_MULTIVIEW)
		{
			OS << "#extension GL_EXT_multiview : require\n\n"; // gl_ViewIndex picks the eye's layer
		}
	}

	std::string_view vr_interpreter_sampler_type(std::string_view type, u64 compiler_options)
	{
		return vr_sampler_type(type, (compiler_options & COMPILER_OPT_VR_MULTIVIEW) ? RSX_SHADER_CONTROL_VR_MULTIVIEW : 0);
	}

	void vr_insert_interpreter_fragment(std::ostream& OS, const std::string& fragment_interpreter, u64 compiler_options)
	{
		if (!(compiler_options & COMPILER_OPT_VR_MULTIVIEW))
		{
			OS << fragment_interpreter;
			return;
		}

		// 2D textures are arrays sampled at layer gl_ViewIndex
		std::string fs_text = fragment_interpreter;
		const usz sites = fs_text.size();
		fs_text = fmt::replace_all(fs_text, "texture(SAMPLER2D(ur0), coord.xy, bias)", "texture(SAMPLER2D(ur0), vec3(coord.xy, float(gl_ViewIndex)), bias)");
		fs_text = fmt::replace_all(fs_text, "textureLod(SAMPLER2D(ur0), coord.xy, lod)", "textureLod(SAMPLER2D(ur0), vec3(coord.xy, float(gl_ViewIndex)), lod)");
		ensure(fs_text.size() > sites); // the interpreter's 2D sampling sites changed: update the replacements
		OS << fs_text;
	}
} // namespace vk

// ---- Push descriptors ---------------------------------------------------------------------------------

namespace vk
{
	u32 vr_push_descriptor_limit(VkPhysicalDevice dev)
	{
		static const bool s_allowed = []
		{
			const char* v = std::getenv("RPCS3_VK_PUSH_DESCRIPTORS");
			return !v || v[0] != '0';
		}();
		if (!s_allowed || !supported_extensions(supported_extensions::device, nullptr, dev).is_supported(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
		{
			return 0;
		}
		VkPhysicalDevicePushDescriptorPropertiesKHR push_props{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR };
		VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &push_props };
		vkGetPhysicalDeviceProperties2(dev, &props2);
		if (push_props.maxPushDescriptors)
		{
			rsx_log.notice("Push descriptors available (up to %u per set)", push_props.maxPushDescriptors);
		}
		return push_props.maxPushDescriptors;
	}

	VkDescriptorSetLayout vr_push_descriptor_layout(glsl::descriptor_table_t& table, const rsx::simple_array<VkDescriptorSetLayoutBinding>& bindings)
	{
		auto& push = table.m_vr_push;
		push.enabled = false;
		if (!push.candidate)
		{
			return VK_NULL_HANDLE;
		}
		u32 total = 0;
		for (const auto& binding : bindings)
		{
			total += binding.descriptorCount;
		}
		if (!total || total > g_render_device->get_push_descriptor_limit())
		{
			return VK_NULL_HANDLE;
		}
		// Push sets take no update-after-bind flags: they are written when the command is recorded.
		const VkDescriptorSetLayoutCreateInfo info
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
			.bindingCount = ::size32(bindings),
			.pBindings = bindings.data()
		};
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		CHECK_RESULT(vkCreateDescriptorSetLayout(*g_render_device, &info, nullptr, &layout));
		push.enabled = true;
		return layout;
	}

	void vr_push_descriptor_table(glsl::descriptor_table_t& table, const command_buffer& cmd, VkPipelineBindPoint bind_point,
		VkPipelineLayout layout, u32 set_index, bool still_bound)
	{
		auto& push = table.m_vr_push;
		if (still_bound && push.writes_valid && !table.m_any_descriptors_dirty && push.last_cmd == static_cast<VkCommandBuffer>(cmd))
		{
			return;
		}

		const u32 slots = ::size32(table.m_descriptor_slots);
		if (push.writes.size() != slots)
		{
			push.writes.resize(slots);
			push.arrays.resize(slots);
			push.writes_valid = false;
		}

		if (!push.writes_valid || table.m_any_descriptors_dirty)
		{
			for (u32 i = 0; i < slots; ++i)
			{
				if (push.writes_valid && !table.m_descriptors_dirty[i])
				{
					continue;
				}

				auto& write = push.writes[i];
				write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1, .descriptorType = table.m_descriptor_types[i] };
				const auto& slot = table.m_descriptor_slots[i];

				if (auto ptr = std::get_if<VkDescriptorImageInfoEx>(&slot))
				{
					write.pImageInfo = ptr;
				}
				else if (auto ptr = std::get_if<VkDescriptorBufferInfoEx>(&slot))
				{
					write.pBufferInfo = ptr;
				}
				else if (auto ptr = std::get_if<VkDescriptorBufferViewEx>(&slot))
				{
					write.pTexelBufferView = &ptr->view;
				}
				else if (auto ptr = std::get_if<glsl::descriptor_image_array_t>(&slot))
				{
					push.arrays[i] = ptr->map(FN(static_cast<VkDescriptorImageInfo>(x)));
					write.descriptorCount = ::size32(push.arrays[i]);
					write.pImageInfo = push.arrays[i].data();
				}
				else
				{
					fmt::throw_exception("Unexpected descriptor structure at index %u", i);
				}

				table.m_descriptors_dirty[i] = false;
			}

			push.writes_valid = true;
			table.m_any_descriptors_dirty = false;
		}

		_vkCmdPushDescriptorSetKHR(cmd, bind_point, layout, set_index, ::size32(push.writes), push.writes.data());
		push.last_cmd = cmd;
	}
}
