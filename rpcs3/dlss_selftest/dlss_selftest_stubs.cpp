// DLSS test: definitions the emulator core expects from the Qt GUI and the input layer, which the headless self-test
// does not build. None of them is used by an RSX capture replay.
#include "stdafx.h"
#include "Input/pad_thread.h"
#include "Input/product_info.h"
#include "Utilities/Thread.h"
#include "Input/ps_move_tracker.h"
#include "Emu/Io/pad_config.h"

#include <functional>
#include <thread>

namespace pad
{
	atomic_t<pad_thread*> g_pad_thread = nullptr;
	shared_mutex g_pad_mutex;
}

void pad_thread::SetRumble(u32, u8, u8) {}
void pad_thread::SetIntercepted(bool) {}
s32 pad_thread::AddLddPad() { return -1; }
void pad_thread::UnregisterLddPad(u32) {}

cfg_input_configurations g_cfg_input_configs;

namespace input
{
	std::vector<product_info> get_products_by_class(int) { return {}; }
}

template <> ps_move_tracker<false>::ps_move_tracker() {}
template <> ps_move_tracker<false>::~ps_move_tracker() {}
template <> void ps_move_tracker<false>::set_image_data(const void*, u64, u32, u32, s32) {}
template <> void ps_move_tracker<false>::process_image() {}
template <> void ps_move_tracker<false>::set_active(u32, bool) {}
template <> void ps_move_tracker<false>::set_hue(u32, u16) {}
template <> void ps_move_tracker<false>::set_hue_threshold(u32, u16) {}
template <> void ps_move_tracker<false>::set_saturation_threshold(u32, u16) {}
template <> std::tuple<u8, u8, u8> ps_move_tracker<false>::hsv_to_rgb(u16, f32, f32) { return {}; }
template <> std::tuple<s16, f32, f32> ps_move_tracker<false>::rgb_to_hsv(f32, f32, f32) { return {}; }

void qt_events_aware_op(int repeat_duration_ms, std::function<bool()> wrapped_op)
{
	while (!wrapped_op())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(repeat_duration_ms ? repeat_duration_ms : 1));
	}
}

std::string g_input_config_override;
