// VR fork: members of keyboard_pad_handler for the scripted key presses (RPCS3_VR_KEYS).
// Textually included in the private section of keyboard_pad_handler.h; the body is in
// keyboard_pad_handler_vr.cpp.

	// VR fork dev hook: scripted key presses from RPCS3_VR_KEYS=<file> (see process_key_script).
	struct scripted_key_event
	{
		steady_clock::time_point at;
		std::vector<u32> codes;
		bool pressed = false;
	};
	std::vector<scripted_key_event> m_key_script;
	steady_clock::time_point m_key_script_poll{};
	void process_key_script();
