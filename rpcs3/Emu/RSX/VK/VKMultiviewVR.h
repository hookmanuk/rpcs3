#pragma once

// VR fork: multiview stereo (both eyes in one draw: plans/9-multiview-plan.md). The pieces that upstream's
// Vulkan files (render passes, the texture cache, the shader generators and interpreter) call, kept here so
// that each upstream site carries one line. Bodies in VKMultiviewVR.cpp, except the one template.

#include "util/types.hpp"
#include "Emu/RSX/Capture/rsx_vr_hooks.h" // rsx::vr::multiview_active(), RSX_SHADER_CONTROL_VR_MULTIVIEW

#include <ostream>
#include <string>
#include <string_view>

enum class FUNCTION; // Emu/RSX/Program/ShaderParam.h

namespace glsl
{
	struct shader_properties;
}

namespace vk
{
	class command_buffer;
	class image;

	// The shader interpreter's multiview variant: a compiler option above upstream's
	// (program_common::interpreter::compiler_option; VKMultiviewVR.cpp checks that they stay below it).
	constexpr u64 COMPILER_OPT_VR_MULTIVIEW = 1ull << 32;

	// ---- Render passes ------------------------------------------------------------------------------

	// get_renderpass(): the VkRenderPassMultiviewCreateInfo of a key's view mask (0 an ordinary pass, 1 both
	// views, 2 view 0 only, 3 view 1 only), for VkRenderPassCreateInfo::pNext; nullptr for an ordinary pass.
	const void* vr_renderpass_multiview_info(u32 key_view_mask);

	// ---- Images -------------------------------------------------------------------------------------

	// Copies layer 0 of a stereo image into layer 1 (guest memory is the same picture for both eyes).
	// Does nothing to an image without stereo layers.
	void vr_copy_left_to_right_layer(vk::command_buffer& cmd, vk::image* image, u8 mipmaps = 1);

	// Texture cache: a copy made from a stereo render target holds both eyes, in two layers. The layer count
	// of a temporary built from desc's sources (a texture_cache::deferred_subresource): 2 when one is stereo.
	template <typename Desc>
	u16 vr_temporary_layers(const Desc& desc)
	{
		if (!rsx::vr::multiview_active())
		{
			return 1;
		}
		if (desc.external_handle && desc.external_handle->stereo_layers && desc.external_handle->layers() > 1)
		{
			return 2;
		}
		for (const auto& section : desc.sections_to_copy)
		{
			if (section.src && section.src->stereo_layers && section.src->layers() > 1)
			{
				return 2;
			}
		}
		return 1;
	}

	// ---- Shader generation: the variants with RSX_SHADER_CONTROL_VR_MULTIVIEW ------------------------

	// The GLSL type of a sampler in a multiview program: 2D samplers are arrays whose layer is the eye
	// (sampler2D, sampler2DMS and sampler2DShadow); every other type, and every type outside the
	// multiview variant, is returned unchanged.
	std::string_view vr_sampler_type(std::string_view type, u32 ctrl);
	void vr_set_sampler_type(std::string& type, u32 ctrl);

	// VKFragmentDecompilerThread::insertHeader(): GL_EXT_multiview (gl_ViewIndex selects the eye's layer),
	// after upstream's required extensions.
	void vr_insert_fragment_extensions(std::ostream& OS, u32 ctrl);
	// VKFragmentDecompilerThread::insertGlobalFunctions(): props.vr_multiview and #define _VR_MULTIVIEW
	// (the array forms of the texture macros in the GLSL snippets).
	void vr_insert_fragment_defines(std::ostream& OS, ::glsl::shader_properties& props, u32 ctrl);

	// VKVertexDecompilerThread::getFunction(): the array forms of the 2D vertex texture fetches; empty
	// when f is not one of them or ctrl is not the multiview variant.
	std::string_view vr_vertex_function(FUNCTION f, u32 ctrl);
	// VKVertexDecompilerThread::insertHeader(): the multiview extensions (after upstream's), and, after the
	// variable redirection block, get_draw_params() redefined so that each view reads its own entry.
	void vr_insert_vertex_extensions(std::ostream& OS, u32 ctrl, bool viewport_index);
	void vr_insert_vertex_draw_params(std::ostream& OS, u32 ctrl);
	// VKVertexDecompilerThread::insertMainEnd(): each view clips to its own scissor (the HUD box differs per eye).
	void vr_insert_vertex_main_end(std::ostream& OS, u32 ctrl, bool viewport_index);

	// Shader interpreter (VKShaderInterpreter.cpp), with COMPILER_OPT_VR_MULTIVIEW in compiler_options.
	void vr_insert_interpreter_vertex(std::ostream& OS, const std::string& vertex_interpreter, bool viewport_index);
	void vr_insert_interpreter_fragment_extensions(std::ostream& OS, u64 compiler_options);
	std::string_view vr_interpreter_sampler_type(std::string_view type, u64 compiler_options);
	void vr_insert_interpreter_fragment(std::ostream& OS, const std::string& fragment_interpreter, u64 compiler_options);
} // namespace vk
