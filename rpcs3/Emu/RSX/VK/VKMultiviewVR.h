#pragma once

// VR fork: multiview stereo (both eyes in one draw: plans/9-multiview-plan.md). The pieces that upstream's
// Vulkan files (render passes, the texture cache, the shader generators and interpreter) call, kept here so
// that each upstream site carries one line. Bodies in VKMultiviewVR.cpp, except the one template.

#include "util/types.hpp"
#include "VulkanAPI.h"
#include "Emu/RSX/Common/simple_array.hpp"
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

	namespace glsl
	{
		struct descriptor_table_t;
	}

	// ---- Push descriptors (VK_KHR_push_descriptor) -----------------------------------------------------
	// The fragment set of separate-object graphics pipelines (textures; rewritten on about half the draws in
	// Gran Turismo 5) is written into the command buffer instead of allocated, updated at submit and bound.
	// Any GPU without the extension, or a set larger than its limit, keeps the allocated path.
	// RPCS3_VK_PUSH_DESCRIPTORS=0 turns it off.

	// get_physical_device_properties_1(): maxPushDescriptors, or 0 when unsupported or turned off.
	u32 vr_push_descriptor_limit(VkPhysicalDevice dev);
	// create_descriptor_set_layout(): a push layout for a candidate set within the limit (sets m_vr_push.enabled),
	// else VK_NULL_HANDLE (the caller creates the ordinary layout).
	VkDescriptorSetLayout vr_push_descriptor_layout(glsl::descriptor_table_t& table, const rsx::simple_array<VkDescriptorSetLayoutBinding>& bindings);
	// program::bind(): writes the set's descriptors. Skipped when nothing changed since the last push into this
	// command buffer and the program's pipeline was still bound there (the pushed state is intact).
	void vr_push_descriptor_table(glsl::descriptor_table_t& table, const command_buffer& cmd, VkPipelineBindPoint bind_point,
		VkPipelineLayout layout, u32 set_index, bool still_bound);

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

	// Exact depth (RSX_SHADER_CONTROL_VR_EXACT_DEPTH): the vertex shader passes (window depth x w, w) in the game's own
	// clip space; interpolated over the triangle their ratio is the game's depth at each pixel, also when the fixed HUD
	// box tilts the draw. The fragment shader writes it to gl_FragDepth.
	void vr_insert_exact_depth_vertex_output(std::ostream& OS, u32 ctrl);
	void vr_insert_exact_depth_vertex_end(std::ostream& OS, u32 ctrl); // after apply_zclip_xform; needs vr_pre_xform
	void vr_insert_exact_depth_fragment_input(std::ostream& OS, u32 ctrl);
	void vr_insert_exact_depth_fragment_end(std::ostream& OS, u32 ctrl);

	// Depth remap (RSX_SHADER_CONTROL_VR_DEPTH_REMAP, profile depth_remap_programs): the vertex shader passes the
	// eye-to-game matrix that follows the program's vertex constants (constant_slots: the slots the CPU fills, 512 for
	// a full bank) to the fragment shader. At the start of fs_main, the fragment shader reads the eye's depth at its
	// pixel and maps it, with the screen position of its lowest texture coordinate (the pass's clip position), to the
	// game's: the program then reads the game's position from that coordinate and the game's depth from its lowest
	// depth texture read as colour (in_register_mask: the decompiler's inputs read, bit 4 tc0; depth_mask: those textures).
	void vr_insert_depth_remap_vertex_output(std::ostream& OS, u32 ctrl);
	void vr_insert_depth_remap_vertex_end(std::ostream& OS, u32 ctrl, u32 constant_slots);
	void vr_insert_depth_remap_fragment_input(std::ostream& OS, u32 ctrl);
	void vr_insert_depth_remap_fragment_start(std::ostream& OS, u32 ctrl, u32 in_register_mask, u32 depth_mask);

	// Shader interpreter (VKShaderInterpreter.cpp), with COMPILER_OPT_VR_MULTIVIEW in compiler_options.
	void vr_insert_interpreter_vertex(std::ostream& OS, const std::string& vertex_interpreter, bool viewport_index);
	void vr_insert_interpreter_fragment_extensions(std::ostream& OS, u64 compiler_options);
	std::string_view vr_interpreter_sampler_type(std::string_view type, u64 compiler_options);
	void vr_insert_interpreter_fragment(std::ostream& OS, const std::string& fragment_interpreter, u64 compiler_options);
} // namespace vk
