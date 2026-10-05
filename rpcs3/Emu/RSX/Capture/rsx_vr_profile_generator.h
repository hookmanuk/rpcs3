#pragma once

// VR profile generator (VR fork)
//
// Builds bin/vr_profiles/<TITLE_ID>.json from the running game, so a title can
// get VR without offline analysis. Started from the home menu's VR tab: after
// the menu closes it samples ten seconds of gameplay (the vertex
// constants of every draw), finds the camera blocks and their matrix layout,
// the camera position slot, the HUD block and the game's frame rate, writes the
// profile, reloads it and turns VR on. The method is plans/5-vr-profile-playbook.md step 3
// (plans/tools/profile_survey.py); the choices it makes are logged in detail.

#include <array>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "util/types.hpp"
#include "util/atomic.hpp"

#include <map>

namespace rsx::vr
{
	class profile_generator
	{
	public:
		static profile_generator& get();

		// Home menu: sample gameplay once the menu has closed.
		void request();

		// Hot-path gate for record_draw().
		bool sampling() const
		{
			return m_sample_this_frame.load();
		}

		// One draw's vertex constants, by original guest index. constant_ids empty
		// means the program reads the whole bank (indexed constants). textures: what the
		// fragment program samples (bit 0 ordinary textures, bit 1 colour render targets).
		// target: colour address 0; ucode: the vertex program's ucode hash (profile hud_programs).
		// indexed: the program indexes its constants (bone palettes); its constant_ids are then the
		// slots it reads directly.
		void record_draw(std::span<const u16> constant_ids, u32 program_id, u16 surface_w, u16 surface_h, bool depth_test, u32 textures,
			u32 target, u64 ucode, bool indexed);

		// Frame boundary (game flips only).
		void on_frame_end();

		// vk texture cache flush (camera_probe hook): a GPU readback of [start, start + length) while sampling.
		// Small ranges the game reads back every frame become late_readback_lengths suggestions.
		void note_readback(u32 start, u32 length);

	private:
		struct draw_sample
		{
			u32 program = 0;
			u16 width = 0;
			u16 height = 0;
			bool full_bank = false;
			bool indexed = false;
			bool depth_test = false;
			bool viewport_y_down = false; // the viewport's y scale is positive: NDC +Y is down the screen
			u8 textures = 0; // bit 0 ordinary textures, bit 1 colour render targets, bit 2 a view-shaped one, bit 3 depth read as colour
			u32 target = 0;  // colour address 0
			u64 ucode = 0;   // vertex program ucode hash
			std::vector<u16> ids;
			std::vector<std::array<f32, 4>> values;
			// Indexed programs: per directly read 4-slot block, the run of blocks after it in the bank with the same
			// projection (clip z = a * w + b): a palette of per-bone clip matrices (Kingdom Hearts: c[256 + 4k]).
			std::vector<std::pair<u16, u16>> palettes;
		};

		void finish();

		enum class state : u32
		{
			idle,
			waiting,
			sampling,
			analysing
		};
		atomic_t<state> m_state{state::idle};
		atomic_t<bool> m_sample_this_frame{false};
		u32 m_frame_counter = 0;
		u32 m_frames_sampled = 0;
		u32 m_flips = 0; // game frames within the play time (frame rate)
		u64 m_played_us = 0;
		u64 m_next_sample_us = 0;
		u64 m_last_frame_us = 0;

		std::mutex m_mutex;
		std::vector<draw_sample> m_samples;
		std::map<std::pair<u32, u32>, u32> m_readbacks; // (start, length) -> count over the play time, under m_mutex
	};
} // namespace rsx::vr
