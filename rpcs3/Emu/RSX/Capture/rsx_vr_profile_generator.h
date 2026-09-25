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

namespace rsx::vr
{
	class profile_generator
	{
	public:
		static profile_generator& get();

		// Home menu: sample gameplay once the menu has closed.
		void request();

		// Hot-path gate for record_draw().
		bool sampling() const { return m_sample_this_frame.load(); }

		// One draw's vertex constants, by original guest index. constant_ids empty
		// means the program reads the whole bank (indexed constants).
		void record_draw(std::span<const u16> constant_ids, u32 program_id, u16 surface_w, u16 surface_h);

		// Frame boundary (game flips only).
		void on_frame_end();

	private:
		struct draw_sample
		{
			u32 program = 0;
			u16 width = 0;
			u16 height = 0;
			bool full_bank = false;
			std::vector<u16> ids;
			std::vector<std::array<f32, 4>> values;
		};

		void finish();

		enum class state : u32 { idle, waiting, sampling, analysing };
		atomic_t<state> m_state{ state::idle };
		atomic_t<bool> m_sample_this_frame{ false };
		u32 m_frame_counter = 0;
		u32 m_frames_sampled = 0;
		u32 m_flips = 0;         // game frames within the play time (frame rate)
		u64 m_played_us = 0;
		u64 m_next_sample_us = 0;
		u64 m_last_frame_us = 0;

		std::mutex m_mutex;
		std::vector<draw_sample> m_samples;
	};
}
