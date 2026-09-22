#pragma once

#include "upscalers/upscaling.h"

#include "vkutils/descriptors.h"
#include "vkutils/data_heap.h"
#include "vkutils/ex.h"
#include "vkutils/instance.h"
#include "vkutils/sync.h"
#include "vkutils/swapchain.h"

#include "VKGSRenderTypes.hpp"
#include "VKTextureCache.h"
#include "VKRenderTargets.h"
#include "VKFormats.h"
#include "VKOverlays.h"
#include "VKProgramBuffer.h"
#include "VKFramebuffer.h"
#include "VKShaderInterpreter.h"
#include "VKQueryPool.h"

#include "Emu/RSX/GSRender.h"
#include "Emu/RSX/Host/RSXDMAWriter.h"
#include <functional>
#include <initializer_list>

using namespace vk::vmm_allocation_pool_; // clang workaround.
using namespace vk::upscaling_flags_;     // ditto

using vs_binding_table_t = decltype(VKVertexProgram::binding_table);
using fs_binding_table_t = decltype(VKFragmentProgram::binding_table);

namespace vk
{
	using host_data_t = rsx::host_gpu_context_t;
}

class VKGSRender : public GSRender, public ::rsx::reports::ZCULL_control
{
private:
	enum frame_context_state : u32
	{
		dirty = 1
	};

	enum flush_queue_state : u32
	{
		ok = 0,
		flushing = 1,
		deadlock = 2
	};

private:
	const VKFragmentProgram *m_fragment_prog = nullptr;
	const VKVertexProgram *m_vertex_prog = nullptr;
	vk::glsl::program *m_program = nullptr;
	vk::glsl::program *m_prev_program = nullptr;
	vk::pipeline_props m_pipeline_properties;

	const vs_binding_table_t* m_vs_binding_table = nullptr;
	const fs_binding_table_t* m_fs_binding_table = nullptr;

	vk::texture_cache m_texture_cache;
	vk::surface_cache m_rtts;
	// Gate 5: host-only mirror of guest render targets. The ordinary cache is
	// authoritative for guest memory; this cache never participates in guest
	// readback, queries, or FIFO accounting.
	vk::surface_cache m_vr_right_rtts;

	std::unique_ptr<vk::buffer> null_buffer;
	std::unique_ptr<vk::buffer_view> null_buffer_view;

	std::unique_ptr<vk::upscaler> m_upscaler;
	output_scaling_mode m_output_scaling{output_scaling_mode::bilinear};

	std::unique_ptr<vk::buffer> m_cond_render_buffer;
	u64 m_cond_render_sync_tag = 0;

	shared_mutex m_sampler_mutex;
	atomic_t<bool> m_samplers_dirty = { true };
	std::unique_ptr<vk::sampler> m_stencil_mirror_sampler;
	std::array<vk::sampler*, rsx::limits::fragment_textures_count> fs_sampler_handles{};
	std::array<vk::sampler*, rsx::limits::vertex_textures_count> vs_sampler_handles{};

	std::unique_ptr<vk::buffer_view> m_persistent_attribute_storage;
	std::unique_ptr<vk::buffer_view> m_volatile_attribute_storage;

	VkDependencyInfoKHR m_async_compute_dependency_info {};
	VkMemoryBarrier2KHR m_async_compute_memory_barrier {};

	std::pair<const vs_binding_table_t*, const fs_binding_table_t*> get_binding_table() const;

public:
	//vk::fbo draw_fbo;
	std::unique_ptr<vk::vertex_cache> m_vertex_cache;
	std::unique_ptr<vk::shader_cache> m_shaders_cache;

private:
	std::unique_ptr<vk::program_cache> m_prog_buffer;

	std::unique_ptr<vk::swapchain_base> m_swapchain;
	vk::instance m_instance;
	vk::render_device *m_device;

	//Vulkan internals
	std::unique_ptr<vk::query_pool_manager> m_occlusion_query_manager;
	bool m_occlusion_query_active = false;
	rsx::reports::occlusion_query_info *m_active_query_info = nullptr;
	std::vector<vk::occlusion_data> m_occlusion_map;

	shared_mutex m_secondary_cb_guard;
	vk::command_pool m_secondary_command_buffer_pool;
	vk::command_buffer_chain<VK_MAX_ASYNC_CB_COUNT> m_secondary_cb_list;

	vk::command_pool m_command_buffer_pool;
	vk::command_buffer_chain<VK_MAX_ASYNC_CB_COUNT> m_primary_cb_list;
	vk::command_buffer_chunk* m_current_command_buffer = nullptr;

	std::unique_ptr<vk::buffer> m_host_object_data;
	vk::framebuffer_holder* m_draw_fbo = nullptr;
	vk::framebuffer_holder* m_vr_right_draw_fbo = nullptr;
	std::vector<vk::image*> m_vr_right_fbo_images;

	// Gate 6: the right-eye draws of one left render pass are recorded into a Vulkan
	// secondary command buffer and executed in a single right-eye pass when the left
	// pass ends (vk::g_end_renderpass_hook), instead of switching render passes twice
	// per draw. A batch is only open while the left pass is open. RPCS3_VR_BATCH=0
	// restores the per-draw replay.
	struct vr_secondary_cb : public vk::command_buffer_chunk
	{
		void attach(vk::command_pool& cmd_pool, VkCommandBuffer cb)
		{
			pool = &cmd_pool;
			commands = cb;
			is_open = true;
			is_pending = false;
			clear_state_cache();
			flags = cb_reload_dynamic_state;
		}

		void detach()
		{
			commands = VK_NULL_HANDLE;
			is_open = false;
			clear_state_cache();
			flags = 0;
		}
	};

	struct vr_batch_slot
	{
		VkCommandBuffer cb = VK_NULL_HANDLE;
		vk::command_buffer_chunk* owner = nullptr; // primary it was executed in
		u64 owner_reset_id = 0;                    // free once the owner has been reset (GPU done)
	};

	vk::command_pool m_vr_batch_pool;
	std::vector<vr_batch_slot> m_vr_batch_slots;
	vr_secondary_cb m_vr_batch_cb;
	bool m_vr_batching = false;
	bool m_vr_batch_open = false;
	bool m_vr_batch_executing = false;
	usz m_vr_batch_slot = 0;
	vk::command_buffer_chunk* m_vr_batch_primary = nullptr;
	VkRenderPass m_vr_batch_pass = VK_NULL_HANDLE;
	vk::framebuffer_holder* m_vr_batch_fbo = nullptr;
	static VKGSRender* s_vr_batch_owner;
	static void vr_on_end_renderpass(const vk::command_buffer& cmd);
	bool vr_batch_begin(VkRenderPass pass, vk::framebuffer_holder* fbo);
	void vr_batch_flush();   // run any open batch now (ends the left pass if it is open)
	void vr_batch_execute(); // left pass closed: one right-eye pass executing the batch

	sizeu m_swapchain_dims{};
	bool swapchain_unavailable = false;
	bool should_reinitialize_swapchain = false;

	u64 m_last_heap_sync_time = 0;
	u32 m_texbuffer_view_size = 0;

	vk::data_heap m_attrib_ring_info;                         // Vertex data
	vk::data_heap m_fragment_constants_ring_info;             // Fragment program constants
	vk::data_heap m_transform_constants_ring_info;            // Transform program constants
	vk::data_heap m_fragment_env_ring_info;                   // Fragment environment params
	vk::data_heap m_vertex_env_ring_info;                     // Vertex environment params
	vk::data_heap m_fragment_texture_params_ring_info;        // Fragment texture params
	vk::data_heap m_vertex_layout_ring_info;                  // Vertex layout structure
	vk::data_heap m_index_buffer_ring_info;                   // Index data
	vk::data_heap m_texture_upload_buffer_ring_info;          // Texture upload heap
	vk::data_heap m_raster_env_ring_info;                     // Raster control such as polygon and line stipple
	vk::data_heap m_instancing_buffer_ring_info;              // Instanced rendering data (constants indirection table + instanced constants)

	vk::data_heap m_fragment_instructions_buffer;             // Interpreter FP block
	vk::data_heap m_vertex_instructions_buffer;               // Interpreter VP block

	rsx::simple_array<vk::data_heap*> m_flushable_data_heaps; // List of heaps that can be 'dirty' and need manual flush

	VkDescriptorBufferInfoEx m_vertex_env_buffer_info {};
	VkDescriptorBufferInfoEx m_fragment_env_buffer_info {};
	VkDescriptorBufferInfoEx m_vertex_layout_stream_info {};
	VkDescriptorBufferInfoEx m_vertex_constants_buffer_info {};
	VkDescriptorBufferInfoEx m_fragment_constants_buffer_info {};
	VkDescriptorBufferInfoEx m_fragment_texture_params_buffer_info {};
	VkDescriptorBufferInfoEx m_raster_env_buffer_info {};
	VkDescriptorBufferInfoEx m_instancing_indirection_buffer_info {};
	VkDescriptorBufferInfoEx m_instancing_constants_array_buffer_info{};

	VkDescriptorBufferInfoEx m_vertex_instructions_buffer_info {};
	VkDescriptorBufferInfoEx m_fragment_instructions_buffer_info {};

	rsx::simple_array<u8> m_multidraw_parameters_buffer;
	u64 m_xform_constants_dynamic_offset = 0;          // We manage transform_constants dynamic offset manually to alleviate performance penalty of doing a hot-patch of constants.
	usz m_xform_constants_data_size = 0;               // Exact current upload size; Gate 5 clones this host allocation per eye.
	u64 m_vertex_env_dynamic_offset = 0;
	u64 m_vertex_layout_dynamic_offset = 0;
	u64 m_fragment_constants_dynamic_offset = 0;
	u64 m_fragment_env_dynamic_offset = 0;
	u64 m_texture_parameters_dynamic_offset = 0;
	u64 m_stipple_array_dynamic_offset = 0;

	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 96>> m_vertex_env_allocator;
	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 16>> m_transform_constants_allocator;
	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 16>> m_fragment_constants_allocator;

	std::vector<vk::frame_context_t> m_frame_context_storage;
	u32 m_max_async_frames = 0u;
	// Temp frame context to use if the real frame queue is overburdened. Only used for storage
	vk::frame_context_t m_aux_frame_context;

	u32 m_current_queue_index = 0;
	vk::frame_context_t* m_current_frame = nullptr;
	std::deque<vk::frame_context_t*> m_queued_frames;

	VkViewport m_viewport {};
	VkRect2D m_scissor {};

	std::vector<u8> m_draw_buffers;

	shared_mutex m_flush_queue_mutex;
	vk::flush_request_task m_flush_requests;

	ullong m_last_cond_render_eval_hint = 0;

	// Offloader thread deadlock recovery
	rsx::atomic_bitmask_t<flush_queue_state> m_queue_status;
	utils::address_range32 m_offloader_fault_range;
	rsx::invalidation_cause m_offloader_fault_cause;

	vk::draw_call_t m_current_draw {};
	u64 m_current_renderpass_key = 0;
	VkRenderPass m_cached_renderpass = VK_NULL_HANDLE;
	std::vector<vk::image*> m_fbo_images;

	std::unique_ptr<vk::image> m_overlay_recording_img;
	std::unique_ptr<vk::image> m_xr_overlay_img; // RPCS3 overlays for the OpenXR quad layer

	//Vertex layout
	rsx::vertex_input_layout m_vertex_layout;

	vk::shader_interpreter m_shader_interpreter;
	u32 m_interpreter_state;

#if defined(HAVE_X11) && defined(HAVE_VULKAN)
	Display *m_display_handle = nullptr;
#endif

public:
	u64 get_cycles() final;
	~VKGSRender() override;

	VKGSRender(utils::serial* ar) noexcept;
	VKGSRender() noexcept : VKGSRender(nullptr) {}

private:
	void prepare_rtts(rsx::framebuffer_creation_context context);

	void close_and_submit_command_buffer(
		vk::fence* fence = nullptr,
		VkSemaphore wait_semaphore = VK_NULL_HANDLE,
		VkSemaphore signal_semaphore = VK_NULL_HANDLE,
		VkPipelineStageFlags pipeline_stage_flags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	void flush_command_queue(bool hard_sync = false, bool do_not_switch = false);
	void queue_swap_request();
	void frame_context_cleanup(vk::frame_context_t *ctx);
	void advance_queued_frames();
	void present(vk::frame_context_t *ctx);
	bool reinitialize_swapchain();

	vk::viewable_image* get_present_source(vk::present_surface_info* info, const rsx::avconf& avconfig);

	void begin_render_pass();
	void close_render_pass();
	VkRenderPass get_render_pass();
	void invalidate_render_pass();

	void update_draw_state();
	void check_present_status();

	vk::vertex_upload_info upload_vertex_data();
	rsx::simple_array<u8> m_scratch_mem;

	bool load_program();
	void load_program_env();
	void update_vertex_env(u32 id, const vk::vertex_upload_info& vertex_info);
	void upload_transform_constants(const rsx::io_buffer& buffer);
	bool bind_vr_eye_constants(f32 eye_sign, u64 source_offset, usz source_size);

	void load_texture_env();
	bool bind_texture_env(bool vr_right_eye = false);
	bool bind_interpreter_texture_env();

public:
	void init_buffers(rsx::framebuffer_creation_context context, bool skip_reading = false);
	void set_viewport();
	void set_scissor(bool clip_viewport);
	void bind_viewport();

	// Sync
	void write_barrier(u32 address, u32 range) override;
	void sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload) override;
	bool release_GCM_label(u32 type, u32 address, u32 data) override;

	void begin_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	void end_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	bool check_occlusion_query_status(rsx::reports::occlusion_query_info* query) override;
	void get_occlusion_query_result(rsx::reports::occlusion_query_info* query) override;
	void discard_occlusion_query(rsx::reports::occlusion_query_info* query) override;

	// External callback in case we need to suddenly submit a commandlist unexpectedly, e.g in a violation handler
	void emergency_query_cleanup(vk::command_buffer* commands);

	// External callback to handle out of video memory problems
	bool on_vram_exhausted(rsx::problem_severity severity);

	// Handle pool creation failure due to fragmentation
	void on_descriptor_pool_fragmentation(bool is_fatal);

	// Conditional rendering
	void begin_conditional_rendering(const std::vector<rsx::reports::occlusion_query_info*>& sources) override;
	void end_conditional_rendering() override;

	// Host sync object
	std::pair<volatile vk::host_data_t*, VkBuffer> map_host_object_data() const;
	void on_guest_texture_read(const vk::command_buffer& cmd);

	// GRAPH backend
	void patch_transform_constants(rsx::context* ctx, u32 index, u32 count) override;

	// Misc
	bool is_current_program_interpreted() const override;

protected:
	void clear_surface(u32 mask) override;
	void begin() override;
	void end() override;
	void emit_geometry(u32 sub_index) override;

	void on_init_thread() override;
	void on_exit() override;
	void flip(const rsx::display_flip_info_t& info) override;

	void renderctl(u32 request_code, void* args) override;

	void do_local_task(rsx::FIFO::state state) override;
	bool scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate) override;
	void notify_tile_unbound(u32 tile) override;

	bool on_access_violation(u32 address, bool is_writing) override;
	void on_invalidate_memory_range(const utils::address_range32 &range, rsx::invalidation_cause cause) override;
	void on_semaphore_acquire_wait() override;
};
