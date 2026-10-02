#pragma once

#include "vkutils/framebuffer_object.hpp"

namespace vk
{
	struct framebuffer_holder : public vk::framebuffer, public rsx::ref_counted
	{
		using framebuffer::framebuffer;
	};

	// VR fork (multiview): view_mask is the render pass variant the framebuffer is created for (incompatible
	// variants must not share one), base_layer/layer_count select the attachment layers (0 = layer 0 only;
	// a multiview pass takes both layers of a stereo image).
	vk::framebuffer_holder* get_framebuffer(VkDevice dev, u16 width, u16 height, VkBool32 has_input_attachments, VkRenderPass renderpass, const std::vector<vk::image*>& image_list,
		u8 view_mask = 0, u8 base_layer = 0, u8 layer_count = 0);
	vk::framebuffer_holder* get_framebuffer(VkDevice dev, u16 width, u16 height, VkBool32 has_input_attachments, VkRenderPass renderpass, VkFormat format, VkImage attachment);

	void remove_unused_framebuffers();
	void clear_framebuffer_cache();

	void remove_framebuffers_with_image(const vk::image* image);
	void remove_framebuffers_with_image(VkImage handle);
}
