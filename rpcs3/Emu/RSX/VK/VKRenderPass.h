#pragma once

#include "VulkanAPI.h"
#include "Utilities/geometry.h"

namespace vk
{
	class image;
	class command_buffer;

	// VR fork (multiview): view_mask 0 = ordinary pass, 1 = both views (0b11), 2 = view 0 only, 3 = view 1 only.
	u64 get_renderpass_key(const std::vector<vk::image*>& images, const std::vector<u8>& input_attachment_ids = {}, u8 view_mask = 0);
	u8 get_renderpass_view_mask(u64 renderpass_key);
	u64 get_renderpass_key(const std::vector<vk::image*>& images, u64 previous_key, const std::vector<u8>& input_attachment_ids = {});
	u64 get_renderpass_key(VkFormat surface_format, u8 color_attachment_count = 1, u8 sample_count = 1);
	u64 get_renderpass_key(VkFormat color_format, VkFormat depth_format, u8 color_attachment_count = 1, u8 sample_count = 1);
	VkRenderPass get_renderpass(VkDevice dev, u64 renderpass_key);

	bool renderpass_has_input_attachments(u64 renderpass_key);

	void clear_renderpass_cache(VkDevice dev);

	// Renderpass scope management helpers.
	// NOTE: These are not thread safe by design.
	void begin_renderpass(VkDevice dev, const vk::command_buffer& cmd, u64 renderpass_key, VkFramebuffer target, const coordu& framebuffer_region);
	void begin_renderpass(const vk::command_buffer& cmd, VkRenderPass pass, VkFramebuffer target, const coordu& framebuffer_region);
	void end_renderpass(const vk::command_buffer& cmd);

	// VR fork (Gate 6): called after every end_renderpass, so batched right-eye draws
	// run exactly where the left render pass ends (before anything recorded after it).
	extern void (*g_end_renderpass_hook)(const vk::command_buffer& cmd);
	// VR fork (multiview): called before vkCmdEndRenderPass, so a query begun inside the pass can end inside it.
	extern void (*g_before_end_renderpass_hook)(const vk::command_buffer& cmd);
	bool is_renderpass_open(const vk::command_buffer& cmd);

	using renderpass_op_callback_t = std::function<void(const vk::command_buffer&, VkRenderPass, VkFramebuffer)>;
	void renderpass_op(const vk::command_buffer& cmd, const renderpass_op_callback_t& op);
}
