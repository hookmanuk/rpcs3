// DLSS test: a headless self-test of the motion vector path (VK/VKDLSS.h) without a game, Qt or an NVIDIA GPU.
//
// It writes a synthetic RSX capture (a floor, a static cube and a moving cube under a perspective camera whose matrix is
// the vertex constants c[0..3], DP4 rows like Gran Turismo 5), boots it with the Vulkan renderer into an X11 window
// (Xvfb + lavapipe in the cloud), moves the camera and the cube between replays by rewriting their constants in the
// FIFO, and saves screenshots of what is presented (with RPCS3_DLSS=motionraw: the motion vectors as numbers).
// The expected motion of chosen points is written next to the screenshots (expected.txt) for dlss_selftest_check.py.
//
//   dlss_selftest <output dir> [frames] [camera|object|both|static]
//
// Linux only (X11). Build: cmake -DBUILD_RPCS3_GUI=OFF -DBUILD_DLSS_SELFTEST=ON ...

#include "stdafx.h"

#include "Emu/System.h"
#include "Emu/system_config.h"
#include "Emu/emu_callbacks.h"
#include "Emu/IdManager.h"
#include "Emu/RSX/GSFrameBase.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/Capture/rsx_replay.h"
#include "Emu/RSX/Program/Assembler/FPASM.h"
#include "Emu/RSX/Program/RSXVertexProgram.h"
#include "Emu/RSX/VK/VKGSRender.h"
#include "Emu/RSX/VK/VKDLSS.h"
#include "Emu/Io/Null/NullKeyboardHandler.h"
#include "Emu/Io/Null/NullMouseHandler.h"
#include "Emu/Io/Null/null_camera_handler.h"
#include "Emu/Io/Null/null_music_handler.h"
#include "Emu/Audio/Null/NullAudioBackend.h"
#include "Emu/Memory/vm.h"
#include "Emu/Cell/Modules/cellMsgDialog.h"
#include "Emu/Cell/Modules/cellOskDialog.h"
#include "Emu/Cell/Modules/cellSaveData.h"
#include "Emu/Cell/Modules/sceNp.h"
#include "Emu/Cell/Modules/sceNpTrophy.h"
#include "Emu/Audio/audio_device_enumerator.h"
#include "util/video_source.h"
#include "util/serialization_ext.hpp"
#include "util/logs.hpp"
#include "Utilities/File.h"
#include "Utilities/Thread.h"

#include <X11/Xlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <set>

LOG_CHANNEL(selftest_log, "SELFTEST");

extern atomic_t<bool> g_user_asked_for_screenshot; // RSXThread.cpp

namespace
{
	constexpr u32 width = 1280;
	constexpr u32 height = 720;
	constexpr u32 pitch = width * 4;
	constexpr u32 color_offset = 0;
	constexpr u32 depth_offset = 0x00400000;
	constexpr u32 vertex_offset = 0x01000000;
	constexpr u32 fp_offset = 0x01100000; // one fragment program per object, 0x100 apart

	std::string g_out_dir;
	std::string g_scenario = "both";
	u32 g_frames = 24;

	// ---- Main thread queue (Emu.CallFromMainThread) ------------------------------------------------------------
	std::mutex g_queue_mutex;
	std::condition_variable g_queue_cv;
	std::deque<std::pair<std::function<void()>, atomic_t<u32>*>> g_queue;

	void post_to_main(std::function<void()> func, atomic_t<u32>* wake_up)
	{
		{
			std::lock_guard lock(g_queue_mutex);
			g_queue.emplace_back(std::move(func), wake_up);
		}
		g_queue_cv.notify_one();
	}

	// ---- Math (row-major, column vectors: clip = M * v) ---------------------------------------------------------
	using mat4 = std::array<f32, 16>;

	mat4 mul(const mat4& a, const mat4& b)
	{
		mat4 r{};
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int k = 0; k < 4; ++k)
					r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
		return r;
	}

	mat4 translate(f32 x, f32 y, f32 z)
	{
		return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
	}

	mat4 scale(f32 x, f32 y, f32 z)
	{
		return {x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1};
	}

	mat4 perspective(f32 fovy, f32 aspect, f32 n, f32 f)
	{
		const f32 t = 1.f / std::tan(fovy / 2.f);
		return {t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), 2 * f * n / (n - f), 0, 0, -1, 0};
	}

	mat4 look_at(f32 ex, f32 ey, f32 ez, f32 cx, f32 cy, f32 cz)
	{
		f32 fx = cx - ex, fy = cy - ey, fz = cz - ez;
		const f32 fl = std::sqrt(fx * fx + fy * fy + fz * fz);
		fx /= fl, fy /= fl, fz /= fl;
		// s = f x up(0,1,0)
		f32 sx = -fz, sy = 0, sz = fx;
		const f32 sl = std::sqrt(sx * sx + sz * sz);
		sx /= sl, sz /= sl;
		// u = s x f
		const f32 ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
		const mat4 r = {sx, sy, sz, 0, ux, uy, uz, 0, -fx, -fy, -fz, 0, 0, 0, 0, 1};
		return mul(r, translate(-ex, -ey, -ez));
	}

	struct scene_frame
	{
		std::array<mat4, 3> mvp; // floor, static cube, moving cube
		std::array<std::array<f32, 3>, 3> probe_points; // world points to check: on the floor, the static cube, the moving cube
	};

	scene_frame scene_at(u32 frame)
	{
		const bool move_camera = g_scenario == "camera" || g_scenario == "both";
		const bool move_object = g_scenario == "object" || g_scenario == "both";
		const f32 t = static_cast<f32>(frame);

		const f32 yaw = move_camera ? 0.02f * t : 0.f;
		const f32 ex = 7.f * std::sin(yaw), ez = 7.f * std::cos(yaw);
		const mat4 vp = mul(perspective(1.0472f, static_cast<f32>(width) / height, 0.1f, 100.f), look_at(ex, 2.5f, ez, 0.f, 0.5f, 0.f));

		const f32 cube_x = move_object ? -1.f + 0.05f * static_cast<f32>(frame % 40) : 1.f;

		scene_frame s;
		s.mvp[0] = mul(vp, scale(6.f, 1.f, 6.f));
		s.mvp[1] = mul(vp, translate(-2.f, 0.5f, -0.5f));
		s.mvp[2] = mul(vp, translate(cube_x, 0.5f, 1.5f));
		s.probe_points = {{{3.f, 0.f, -3.f}, {-2.f, 1.f, -0.5f}, {cube_x, 1.f, 1.5f}}}; // floor, tops of the cubes
		return s;
	}

	// The window position (pixels, y down) of a world point, as RSX's viewport below maps it.
	std::array<f32, 2> project(const mat4& vp_only, const std::array<f32, 3>& p)
	{
		f32 c[4];
		for (int i = 0; i < 4; ++i)
			c[i] = vp_only[i * 4 + 0] * p[0] + vp_only[i * 4 + 1] * p[1] + vp_only[i * 4 + 2] * p[2] + vp_only[i * 4 + 3];
		return {(c[0] / c[3]) * (width / 2.f) + width / 2.f, (c[1] / c[3]) * -(height / 2.f) + height / 2.f};
	}

	mat4 view_projection_at(u32 frame)
	{
		const bool move_camera = g_scenario == "camera" || g_scenario == "both";
		const f32 yaw = move_camera ? 0.02f * static_cast<f32>(frame) : 0.f;
		return mul(perspective(1.0472f, static_cast<f32>(width) / height, 0.1f, 100.f), look_at(7.f * std::sin(yaw), 2.5f, 7.f * std::cos(yaw), 0.f, 0.5f, 0.f));
	}

	// ---- Capture builder -----------------------------------------------------------------------------------------
	struct capture_builder
	{
		rsx::frame_capture_data frame;
		u32 fifo_words = 0; // FIFO position of the next word (alloc_write_fifo's layout)

		// One method with its arguments; returns the FIFO word index of the first argument.
		u32 method(u32 reg, std::initializer_list<u32> args)
		{
			return method(reg, std::vector<u32>(args));
		}

		u32 method(u32 reg, const std::vector<u32>& args)
		{
			ensure(!args.empty() && args.size() < 0x800);
			const u32 header = (static_cast<u32>(args.size()) << 18) | (reg << 2);
			const u32 first_arg = fifo_words + 1;
			for (usz i = 0; i < args.size(); ++i)
			{
				rsx::frame_capture_data::replay_command cmd;
				cmd.rsx_command = {i == 0 ? header : 0u, args[i]};
				frame.replay_commands.push_back(std::move(cmd));
			}
			fifo_words += 1 + static_cast<u32>(args.size());
			return first_arg;
		}

		u32 methodf(u32 reg, std::initializer_list<f32> args)
		{
			std::vector<u32> v;
			for (f32 f : args)
				v.push_back(std::bit_cast<u32>(f));
			return method(reg, v);
		}

		u64 add_memory(u32 offset, u32 location, std::vector<u8> data)
		{
			rsx::frame_capture_data::memory_block_data block_data{std::move(data)};
			const u64 data_id = frame.memory_indexer++;
			frame.memory_data_map.emplace(std::move(block_data), data_id);

			rsx::frame_capture_data::memory_block block{};
			block.offset = offset;
			block.location = location;
			block.data_state = data_id;
			const u64 block_id = frame.memory_indexer++;
			frame.memory_map.emplace(block, block_id);
			return block_id;
		}
	};

	// RSX vertex program: o[0].<c> = DP4(v[0], c[k]) for k = 0..3.
	std::vector<u32> make_vertex_program()
	{
		std::vector<u32> words;
		for (u32 k = 0; k < 4; ++k)
		{
			SRC src0{}, src1{}, src2{};
			src0.reg_type = RSX_VP_REGISTER_TYPE_INPUT;
			src1.reg_type = RSX_VP_REGISTER_TYPE_CONSTANT;
			src2.reg_type = RSX_VP_REGISTER_TYPE_TEMP;
			for (SRC* s : {&src0, &src1, &src2})
			{
				s->swz_x = 0, s->swz_y = 1, s->swz_z = 2, s->swz_w = 3;
			}

			D0 d0{};
			d0.dst_tmp = 0x3f;
			d0.cond = 7; // TR: always write (0 = never)
			d0.vec_result = 1;

			D1 d1{};
			d1.input_src = 0;
			d1.const_src = k;
			d1.vec_opcode = RSX_VEC_OPCODE_DP4;
			d1.sca_opcode = RSX_SCA_OPCODE_NOP;
			d1.src0h = src0.src0h;

			D2 d2{};
			d2.src0l = src0.src0l;
			d2.src1 = src1.src1;
			d2.src2h = src2.src2h;

			D3 d3{};
			d3.end = k == 3;
			d3.dst = 0;
			d3.sca_dst_tmp = 0x3f;
			d3.vec_writemask_x = k == 0;
			d3.vec_writemask_y = k == 1;
			d3.vec_writemask_z = k == 2;
			d3.vec_writemask_w = k == 3;
			d3.src2l = src2.src2l;

			words.insert(words.end(), {d0.HEX, d1.HEX, d2.HEX, d3.HEX});
		}
		return words;
	}

	std::vector<u8> fragment_program(f32 r, f32 g, f32 b)
	{
		const auto ir = rsx::assembler::FPIR::from_source(fmt::format("MOV R0, #{%f, %f, %f, 1.0}", r, g, b));
		const auto words = ir.compile();
		std::vector<u8> bytes(words.size() * 4);
		std::memcpy(bytes.data(), words.data(), bytes.size());
		return bytes;
	}

	void push_be_vertex(std::vector<u8>& out, f32 x, f32 y, f32 z)
	{
		for (f32 f : {x, y, z})
		{
			const be_t<f32> v = f;
			const u8* p = reinterpret_cast<const u8*>(&v);
			out.insert(out.end(), p, p + 4);
		}
	}

	// Unit cube centred at the origin (36 vertices), then a floor quad at y = 0 spanning [-1, 1] (6 vertices).
	std::vector<u8> make_vertices()
	{
		std::vector<u8> v;
		const f32 c[8][3] = {{-.5f, -.5f, -.5f}, {.5f, -.5f, -.5f}, {.5f, .5f, -.5f}, {-.5f, .5f, -.5f}, {-.5f, -.5f, .5f}, {.5f, -.5f, .5f}, {.5f, .5f, .5f}, {-.5f, .5f, .5f}};
		const int faces[6][4] = {{0, 1, 2, 3}, {5, 4, 7, 6}, {4, 0, 3, 7}, {1, 5, 6, 2}, {3, 2, 6, 7}, {4, 5, 1, 0}};
		for (const auto& f : faces)
		{
			for (int i : {f[0], f[1], f[2], f[0], f[2], f[3]})
			{
				push_be_vertex(v, c[i][0], c[i][1], c[i][2]);
			}
		}
		const f32 q[4][3] = {{-1, 0, -1}, {1, 0, -1}, {1, 0, 1}, {-1, 0, 1}};
		for (int i : {0, 1, 2, 0, 2, 3})
		{
			push_be_vertex(v, q[i][0], q[i][1], q[i][2]);
		}
		return v;
	}

	// FIFO word index of each object's 16 constants (patched at every flip).
	std::array<u32, 3> g_constant_words{};

	std::string build_capture(const std::string& path)
	{
		capture_builder b;
		auto& f = b.frame;
		f.reg_state.init();

		// Memory: vertices and the three fragment programs (local memory).
		std::unordered_set<u64> memory;
		memory.insert(b.add_memory(vertex_offset, CELL_GCM_LOCATION_LOCAL, make_vertices()));
		const std::array<std::array<f32, 3>, 3> colors = {{{0.35f, 0.45f, 0.35f}, {0.2f, 0.4f, 0.9f}, {0.9f, 0.3f, 0.2f}}};
		for (u32 i = 0; i < 3; ++i)
		{
			memory.insert(b.add_memory(fp_offset + i * 0x100, CELL_GCM_LOCATION_LOCAL, fragment_program(colors[i][0], colors[i][1], colors[i][2])));
		}

		// Display buffers: one, the scene's colour target.
		rsx::frame_capture_data::display_buffers_state db{};
		db.count = 2;
		db.buffers[0] = {width, height, pitch, color_offset};
		db.buffers[1] = {width, height, pitch, color_offset};
		f.display_buffers_map.emplace(db, 1);

		// Surface
		const u32 first = b.method(NV4097_SET_CONTEXT_DMA_COLOR_A, {CELL_GCM_CONTEXT_DMA_MEMORY_FRAME_BUFFER});
		b.frame.replay_commands.front().memory_state = memory;
		b.frame.replay_commands.front().display_buffer_state = 1;
		static_cast<void>(first);
		b.method(NV4097_SET_CONTEXT_DMA_ZETA, {CELL_GCM_CONTEXT_DMA_MEMORY_FRAME_BUFFER});
		b.method(NV4097_SET_SURFACE_FORMAT, {CELL_GCM_SURFACE_A8R8G8B8 | (CELL_GCM_SURFACE_Z24S8 << 5) | (CELL_GCM_SURFACE_PITCH << 8) | (11u << 16) | (10u << 24)});
		b.method(NV4097_SET_SURFACE_PITCH_A, {pitch});
		b.method(NV4097_SET_SURFACE_COLOR_AOFFSET, {color_offset});
		b.method(NV4097_SET_SURFACE_ZETA_OFFSET, {depth_offset});
		b.method(NV4097_SET_SURFACE_PITCH_Z, {pitch});
		b.method(NV4097_SET_SURFACE_COLOR_TARGET, {CELL_GCM_SURFACE_TARGET_0});
		b.method(NV4097_SET_SURFACE_CLIP_HORIZONTAL, {width << 16});
		b.method(NV4097_SET_SURFACE_CLIP_VERTICAL, {height << 16});
		b.method(NV4097_SET_WINDOW_OFFSET, {0});
		b.method(NV4097_SET_SHADER_WINDOW, {height | (CELL_GCM_WINDOW_ORIGIN_TOP << 12) | (CELL_GCM_WINDOW_PIXEL_CENTER_HALF << 16)});

		// Viewport: NDC y up -> window y down
		b.method(NV4097_SET_VIEWPORT_HORIZONTAL, {width << 16});
		b.method(NV4097_SET_VIEWPORT_VERTICAL, {height << 16});
		b.method(NV4097_SET_SCISSOR_HORIZONTAL, {width << 16});
		b.method(NV4097_SET_SCISSOR_VERTICAL, {height << 16});
		b.methodf(NV4097_SET_VIEWPORT_OFFSET, {width / 2.f, height / 2.f, 0.5f, 0.f});
		b.methodf(NV4097_SET_VIEWPORT_SCALE, {width / 2.f, -(height / 2.f), 0.5f, 0.f});
		b.methodf(NV4097_SET_CLIP_MIN, {0.f});
		b.methodf(NV4097_SET_CLIP_MAX, {1.f});

		// State
		b.method(NV4097_SET_COLOR_MASK, {0x01010101});
		b.method(NV4097_SET_DEPTH_TEST_ENABLE, {1});
		b.method(NV4097_SET_DEPTH_FUNC, {CELL_GCM_LESS});
		b.method(NV4097_SET_DEPTH_MASK, {1});
		b.method(NV4097_SET_CULL_FACE_ENABLE, {0});
		b.method(NV4097_SET_BLEND_ENABLE, {0});
		b.method(NV4097_SET_ALPHA_TEST_ENABLE, {0});
		b.method(NV4097_SET_STENCIL_TEST_ENABLE, {0});

		// Clear
		b.method(NV4097_SET_COLOR_CLEAR_VALUE, {0xff202838});
		b.method(NV4097_SET_ZSTENCIL_CLEAR_VALUE, {0xffffff00});
		b.method(NV4097_CLEAR_SURFACE, {0xf3});

		// Vertex program
		const auto vp = make_vertex_program();
		b.method(NV4097_SET_TRANSFORM_PROGRAM_LOAD, {0});
		b.method(NV4097_SET_TRANSFORM_PROGRAM, vp);
		b.method(NV4097_SET_TRANSFORM_PROGRAM_START, {0});
		b.method(NV4097_SET_VERTEX_ATTRIB_INPUT_MASK, {1});
		b.method(NV4097_SET_VERTEX_ATTRIB_OUTPUT_MASK, {0});
		b.method(NV4097_SET_SHADER_CONTROL, {0x02000440}); // 2 registers, 32-bit exports
		b.method(NV4097_SET_SHADER_PACKER, {0});

		// Vertex array 0: float3 positions, big-endian, stride 12
		for (u32 i = 1; i < 16; ++i)
		{
			b.method(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + i, {CELL_GCM_VERTEX_F});
		}
		b.method(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT, {CELL_GCM_VERTEX_F | (3u << 4) | (12u << 8)});
		b.method(NV4097_SET_VERTEX_DATA_BASE_OFFSET, {0});
		b.method(NV4097_SET_VERTEX_DATA_BASE_INDEX, {0});

		const scene_frame s0 = scene_at(0);
		const struct { u32 first; u32 count; } ranges[3] = {{36, 6}, {0, 36}, {0, 36}};
		for (u32 i = 0; i < 3; ++i)
		{
			b.method(NV4097_SET_SHADER_PROGRAM, {(fp_offset + i * 0x100) | (CELL_GCM_LOCATION_LOCAL + 1)});
			b.method(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET, {vertex_offset}); // local memory
			b.method(NV4097_SET_TRANSFORM_CONSTANT_LOAD, {0});
			std::vector<u32> c;
			for (f32 v : s0.mvp[i])
				c.push_back(std::bit_cast<u32>(v));
			g_constant_words[i] = b.method(NV4097_SET_TRANSFORM_CONSTANT, c);
			b.method(NV4097_SET_BEGIN_END, {CELL_GCM_PRIMITIVE_TRIANGLES});
			b.method(NV4097_DRAW_ARRAYS, {ranges[i].first | ((ranges[i].count - 1) << 24)});
			b.method(NV4097_SET_BEGIN_END, {0});
		}

		// Save, as RSXThread does (uncompressed)
		fs::file file(path, fs::rewrite);
		ensure(file, "Cannot write the capture");
		utils::serial save;
		save.m_file_handler = make_uncompressed_serialization_file_handler(std::move(file));
		save(f);
		save.m_file_handler->finalize(save);
		return path;
	}

	// ---- Frame ---------------------------------------------------------------------------------------------------
	std::atomic<u32> g_flips{0};
	u32 g_drawn_seen = 0;
	std::set<u32> g_shot_frames;

	void patch_constants(u32 next_frame)
	{
		// The next replay of the capture draws frame `next_frame`: rewrite the three constant blocks in the FIFO.
		const scene_frame s = scene_at(next_frame);
		const u32 fifo = rsx::get_address(0x10000000, CELL_GCM_LOCATION_MAIN);
		if (!fifo)
		{
			return;
		}
		for (u32 i = 0; i < 3; ++i)
		{
			auto ptr = vm::ptr<u32>::make(fifo + g_constant_words[i] * 4);
			for (u32 j = 0; j < 16; ++j)
			{
				ptr[j] = std::bit_cast<u32>(s.mvp[i][j]);
			}
		}
	}

	class x11_frame final : public GSFrameBase
	{
		Display* m_display = nullptr;
		Window m_window = 0;

	public:
		x11_frame()
		{
			m_display = XOpenDisplay(nullptr);
			ensure(m_display, "XOpenDisplay failed (is DISPLAY set? run under xvfb-run)");
			m_window = XCreateSimpleWindow(m_display, DefaultRootWindow(m_display), 0, 0, width, height, 0, 0, 0);
			XMapWindow(m_display, m_window);
			XFlush(m_display);
		}

		~x11_frame() override
		{
			if (m_display)
			{
				XDestroyWindow(m_display, m_window);
				XCloseDisplay(m_display);
			}
		}

		void close() override {}
		void reset() override {}
		bool shown() override { return true; }
		void hide() override {}
		void show() override {}
		void toggle_fullscreen() override {}
		void delete_context(draw_context_t) override {}
		draw_context_t make_context() override { return nullptr; }
		void set_current(draw_context_t) override {}

		void flip(draw_context_t, bool skip_frame) override
		{
			if (skip_frame)
			{
				return;
			}

			++g_flips;

			// A replay loop that drew (the renderer counts frames with camera draws; some loops flip without drawing):
			// the next drawn frame shows scene_at(drawn).
			const u32 drawn = static_cast<u32>(vk::dlss_drawn_frames());
			if (drawn != g_drawn_seen)
			{
				g_drawn_seen = drawn;
				patch_constants(drawn);
			}

			// Screenshots of the last frames (taken at the next present).
			if (drawn + 6 >= g_frames && drawn < g_frames + 1)
			{
				g_user_asked_for_screenshot = true;
			}

			while (XPending(m_display))
			{
				XEvent e;
				XNextEvent(m_display, &e);
			}
		}

		int client_width() override { return width; }
		int client_height() override { return height; }
		f64 client_display_rate() override { return 60.; }
		bool has_alpha() override { return false; }
		display_handle_t handle() const override { return std::make_pair(m_display, m_window); }
		bool can_consume_frame() const override { return false; }
		void present_frame(std::vector<u8>&&, u32, u32, u32, bool) const override {}

		void take_screenshot(std::vector<u8>&& data, u32 w, u32 h, bool is_bgra) override
		{
			// The presented image shows drawn frame k = scene_at(k).
			const u32 frame = static_cast<u32>(vk::dlss_presented_frame());
			if (!g_shot_frames.insert(frame).second)
			{
				return;
			}
			if (is_bgra)
			{
				for (usz i = 0; i + 3 < data.size(); i += 4)
					std::swap(data[i], data[i + 2]);
			}
			for (usz i = 3; i < data.size(); i += 4)
				data[i] = 255;
			const std::string path = fmt::format("%s/frame_%03u.png", g_out_dir, frame);
			stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h), 4, data.data(), static_cast<int>(w * 4));
			selftest_log.success("Screenshot %s (%ux%u)", path, w, h);

			// Expected motion (pixels, towards the previous frame) of the probe points in this frame.
			const scene_frame cur = scene_at(frame), prev = scene_at(frame ? frame - 1 : 0);
			const mat4 vp_cur = view_projection_at(frame), vp_prev = view_projection_at(frame ? frame - 1 : 0);
			std::string text;
			for (u32 i = 0; i < 3; ++i)
			{
				const auto pc = project(vp_cur, cur.probe_points[i]);
				const auto pp = project(vp_prev, prev.probe_points[i]);
				text += fmt::format("%u %u %f %f %f %f\n", frame, i, pc[0], pc[1], pp[0] - pc[0], pp[1] - pc[1]);
			}
			fs::write_file(g_out_dir + "/expected.txt", fs::create + fs::write + fs::append, text);
		}

		void update_title(double) override {}
	};

	void set_callbacks()
	{
		auto& cb = g_emu_callbacks;
		cb.call_from_main_thread = [](std::function<void()> func, atomic_t<u32>* wake_up) { post_to_main(std::move(func), wake_up); };
		cb.on_run = [](bool) {};
		cb.on_pause = []() {};
		cb.on_resume = []() {};
		cb.on_stop = []() {};
		cb.on_ready = []() {};
		cb.on_missing_fw = []() {};
		cb.on_emulation_stop_no_response = [](std::shared_ptr<atomic_t<bool>>, int) {};
		cb.on_save_state_progress = [](std::shared_ptr<atomic_t<bool>>, stx::shared_ptr<utils::serial>, stx::atomic_ptr<std::string>*, std::shared_ptr<void>) {};
		cb.enable_disc_eject = [](bool) {};
		cb.enable_disc_insert = [](bool) {};
		cb.try_to_quit = [](bool, std::function<void()> on_exit) { if (on_exit) on_exit(); return true; };
		cb.handle_taskbar_progress = [](s32, s32) {};
		cb.init_kb_handler = []() { ensure(g_fxo->init<KeyboardHandlerBase, NullKeyboardHandler>(Emu.DeserialManager())); };
		cb.init_mouse_handler = []() { ensure(g_fxo->init<MouseHandlerBase, NullMouseHandler>(Emu.DeserialManager())); };
		cb.init_pad_handler = [](std::string_view) {};
		cb.update_emu_settings = []() {};
		cb.save_emu_settings = []() {};
		cb.close_gs_frame = []() {};
		cb.get_gs_frame = []() -> std::unique_ptr<GSFrameBase> { return std::make_unique<x11_frame>(); };
		cb.get_camera_handler = []() -> std::shared_ptr<camera_handler_base> { return std::make_shared<null_camera_handler>(); };
		cb.get_music_handler = []() -> std::shared_ptr<music_handler_base> { return std::make_shared<null_music_handler>(); };
		cb.init_gs_render = [](utils::serial* ar) { g_fxo->init<rsx::thread, named_thread<VKGSRender>>(ar); };
		cb.get_audio = []() -> std::shared_ptr<AudioBackend> { return std::make_shared<NullAudioBackend>(); };
		cb.get_audio_enumerator = [](u64) -> std::shared_ptr<audio_device_enumerator> { return nullptr; };
		cb.get_msg_dialog = []() -> std::shared_ptr<MsgDialogBase> { return nullptr; };
		cb.get_osk_dialog = []() -> std::shared_ptr<OskDialogBase> { return nullptr; };
		cb.get_save_dialog = []() -> std::unique_ptr<SaveDialogBase> { return nullptr; };
		cb.get_sendmessage_dialog = []() -> std::shared_ptr<SendMessageDialogBase> { return nullptr; };
		cb.get_recvmessage_dialog = []() -> std::shared_ptr<RecvMessageDialogBase> { return nullptr; };
		cb.get_trophy_notification_dialog = []() -> std::unique_ptr<TrophyNotificationBase> { return nullptr; };
		cb.get_localized_string = [](localized_string_id, const char*) -> std::string { return {}; };
		cb.get_localized_u32string = [](localized_string_id, const char*) -> std::u32string { return {}; };
		cb.get_localized_setting = [](const cfg::_base*, u32) -> std::string { return {}; };
		cb.get_photo_path = [](std::string_view) -> std::string { return {}; };
		cb.play_sound = [](const std::string&, std::optional<f32>) {};
		cb.get_image_info = [](const std::string&, std::string&, s32&, s32&, s32&) { return false; };
		cb.get_scaled_image = [](const std::string&, s32, s32, s32&, s32&, u8*, bool) { return false; };
		cb.get_font_dirs = []() { return std::vector<std::string>{}; };
		cb.on_install_pkgs = [](const std::vector<std::string>&, bool) { return false; };
		cb.add_breakpoint = [](u32) {};
		cb.display_sleep_control_supported = []() { return false; };
		cb.enable_display_sleep = [](bool) {};
		cb.check_microphone_permissions = []() {};
		cb.make_video_source = []() -> std::unique_ptr<video_source> { return nullptr; };
		cb.enable_gamemode = [](bool) {};
		cb.get_database_config = [](const std::string&) -> std::string { return {}; };
	}
}

[[noreturn]] void report_fatal_error(std::string_view text, bool, bool)
{
	std::fprintf(stderr, "FATAL: %.*s\n", static_cast<int>(text.size()), text.data());
	std::fflush(stderr);
	std::_Exit(2);
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::fprintf(stderr, "usage: %s <output dir> [frames] [camera|object|both|static]\n", argv[0]);
		return 1;
	}

	g_out_dir = argv[1];
	if (argc > 2) g_frames = static_cast<u32>(std::atoi(argv[2]));
	if (argc > 3) g_scenario = argv[3];
	fs::create_path(g_out_dir);
	fs::remove_file(g_out_dir + "/expected.txt");

	XInitThreads();

	static auto log_file = logs::make_file_listener(g_out_dir + "/RPCS3.log", 512ull * 1024 * 1024);
	static_cast<void>(log_file);

	set_callbacks();
	Emu.SetSupportedRenderers({video_renderer::null, video_renderer::vulkan});
	Emu.SetDefaultRenderer(video_renderer::vulkan);
	Emu.SetDefaultGraphicsAdapter(std::getenv("DLSS_SELFTEST_ADAPTER") ? std::getenv("DLSS_SELFTEST_ADAPTER") : "llvmpipe (LLVM 20.1.2, 256 bits)");
	Emu.SetHasGui(false);
	Emu.SetHeadless(false);
	Emu.SetUsr("00000001");
	Emu.Init();

	// Settings for the run: Vulkan, synchronous shader compilation, no frame limit, the debug output (validation).
	g_cfg.video.renderer.set(video_renderer::vulkan);
	g_cfg.video.shadermode.set(shader_mode::recompiler);
	g_cfg.video.frame_limit.set(frame_limit_type::none);
	g_cfg.video.vsync.set(vsync_mode::off);
	g_cfg.video.resolution.set(video_resolution::_720p);
	g_cfg.video.debug_output.set(std::getenv("DLSS_SELFTEST_VALIDATION") != nullptr);
	g_cfg.video.write_color_buffers.set(false);
	g_cfg.video.log_programs.set(std::getenv("DLSS_SELFTEST_LOG_PROGRAMS") != nullptr);
	g_cfg.video.debug_program_analyser.set(std::getenv("DLSS_SELFTEST_LOG_PROGRAMS") != nullptr);
	g_cfg.video.disable_on_disk_shader_cache.set(true);
	g_cfg.audio.renderer.set(audio_renderer::null);
	g_cfg.misc.show_shader_compilation_hint.set(false);
	fs::write_file(fs::get_config_dir(true) + "config.yml", fs::rewrite, g_cfg.to_string());

	const std::string capture = build_capture(g_out_dir + "/dlss_selftest.rrc");
	selftest_log.notice("Capture written: %s", capture);

	if (!Emu.BootRsxCapture(capture))
	{
		std::fprintf(stderr, "BootRsxCapture failed\n");
		return 3;
	}

	if (const char* trace = std::getenv("DLSS_SELFTEST_TRACE"))
	{
		logs::set_level(trace, logs::level::trace);
	}

	// Main thread: run queued calls until enough frames were shown and their screenshots written.
	const auto start = std::chrono::steady_clock::now();
	while (true)
	{
		std::unique_lock lock(g_queue_mutex);
		g_queue_cv.wait_for(lock, std::chrono::milliseconds(20), [] { return !g_queue.empty(); });
		while (!g_queue.empty())
		{
			auto [func, wake_up] = std::move(g_queue.front());
			g_queue.pop_front();
			lock.unlock();
			func();
			if (wake_up)
			{
				*wake_up = true;
				wake_up->notify_one();
			}
			lock.lock();
		}
		lock.unlock();

		if (vk::dlss_drawn_frames() >= g_frames + 2)
		{
			break;
		}
		if (std::chrono::steady_clock::now() - start > std::chrono::seconds(std::getenv("DLSS_SELFTEST_TIMEOUT") ? std::atoi(std::getenv("DLSS_SELFTEST_TIMEOUT")) : 300))
		{
			std::fprintf(stderr, "Timeout after %u flips\n", g_flips.load());
			std::fflush(stderr);
			std::_Exit(4);
		}
	}

	std::printf("Done: %u flips\n", g_flips.load());
	std::fflush(stdout);
	// Exit without a full emulator shutdown (the replay thread loops forever).
	std::_Exit(0);
}
