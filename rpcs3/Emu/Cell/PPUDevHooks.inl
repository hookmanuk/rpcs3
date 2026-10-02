// VR fork: development breakpoints for the PPU interpreter (RPCS3_PPU_TRACE, RPCS3_PPU_WATCH,
// RPCS3_PPU_RWATCH, RPCS3_PPU_WATCH_FILE). Textually included by PPUThread.cpp, next to
// ppu_breakpoint(), because they use its file-local ppu_read() and ppu_cache().

// VR fork dev hook: a trace breakpoint logs the guest call stack (each distinct
// stack once, then every 300th hit) and continues. Interpreter only.
static void ppu_trace_break(ppu_thread& ppu, ppu_opcode_t, be_t<u32>* this_op, ppu_intrp_func* next_fn)
{
	const u32 addr = vm::get_addr(this_op);
	ppu.cia = addr;

	static std::mutex s_mutex;
	static std::map<u32, u64> s_hits;
	static std::set<std::string> s_seen;
	{
		std::lock_guard lock(s_mutex);
		const u64 hits = ++s_hits[addr];
		std::string key = fmt::format("0x%x LR 0x%x", addr, static_cast<u32>(ppu.lr));
		const auto list = ppu.dump_callstack_list();
		for (usz i = 0; i < std::min<usz>(list.size(), 10); i++)
		{
			fmt::append(key, " <- 0x%x", list[i].first);
		}
		if (s_seen.insert(key).second || hits % 300 == 0)
		{
			// RPCS3_PPU_TRACE_REGS=r18,f13,...: also log these registers.
			static const std::string s_regs = []
			{
				const char* v = std::getenv("RPCS3_PPU_TRACE_REGS");
				return v ? std::string(v) : std::string();
			}();
			std::string regs;
			for (const auto& r : fmt::split(s_regs, {","}))
			{
				if (r.size() < 2)
					continue;
				const u32 n = static_cast<u32>(std::strtoul(r.c_str() + 1, nullptr, 10)) % 32;
				if (r[0] == 'f')
					fmt::append(regs, " f%u=%g", n, ppu.fpr[n]);
				else
					fmt::append(regs, " r%u=0x%llx", n, ppu.gpr[n]);
			}
			ppu_log.success("TRACE %s [%s, r3=0x%llx r4=0x%llx%s, hit %u]", key, ppu.get_name(), ppu.gpr[3], ppu.gpr[4], regs, hits);
		}
	}

	return ppu_cache(addr)(ppu, {*this_op}, this_op, next_fn);
}

// VR fork dev hook: a write watch. Every store instruction in a code range is
// replaced by this check, which logs each distinct store address (with the
// value and call stack) that writes into [s_watch_addr, s_watch_addr + s_watch_len).
// Interpreter only.
static u32 s_watch_addr = 0;
static u32 s_watch_len = 0;

static void ppu_watch_break(ppu_thread& ppu, ppu_opcode_t, be_t<u32>* this_op, ppu_intrp_func* next_fn)
{
	const u32 addr = vm::get_addr(this_op);
	const ppu_opcode_t op{*this_op};
	const u64 base = op.ra ? ppu.gpr[op.ra] : 0;
	u32 ea = 0, size = 4;
	bool is_float = false;
	switch (op.main)
	{
	case 36:
	case 37: ea = static_cast<u32>(base + op.simm16); break; // stw(u)
	case 38:
	case 39:
		ea = static_cast<u32>(base + op.simm16);
		size = 1;
		break; // stb(u)
	case 44:
	case 45:
		ea = static_cast<u32>(base + op.simm16);
		size = 2;
		break; // sth(u)
	case 47:
		ea = static_cast<u32>(base + op.simm16);
		size = (32 - op.rs) * 4;
		break; // stmw
	case 52:
	case 53:
		ea = static_cast<u32>(base + op.simm16);
		is_float = true;
		break; // stfs(u)
	case 54:
	case 55:
		ea = static_cast<u32>(base + op.simm16);
		size = 8;
		break; // stfd(u)
	case 62:
		ea = static_cast<u32>(base + (op.simm16 & ~3));
		size = 8;
		break; // std(u)
	case 31:
	{
		ea = static_cast<u32>(base + ppu.gpr[op.rb]);
		switch (op.opcode >> 1 & 0x3ff)
		{
		case 231:
		case 487:
			ea &= ~15u;
			size = 16;
			break;                  // stvx(l)
		case 199: ea &= ~3u; break; // stvewx
		case 647:
		case 679:
		case 775:
		case 807: size = 16; break; // stvlx, stvrx(l) (approximate range)
		case 214: size = 8; break;  // stdcx.
		case 918: size = 2; break;  // sthbrx
		case 663:
		case 695: is_float = true; break; // stfsx, stfsux
		case 727:
		case 759:
		case 149:
		case 181: size = 8; break; // stfdx(u), stdx(u)
		case 215:
		case 247: size = 1; break; // stbx(u)
		case 407:
		case 439: size = 2; break; // sthx(u)
		default: break;            // stwx(u)
		}
		break;
	}
	default: size = 0; break;
	}

	if (size && ea < s_watch_addr + s_watch_len && ea + size > s_watch_addr)
	{
		static std::mutex s_mutex;
		static std::map<u32, u64> s_hits;
		std::lock_guard lock(s_mutex);
		const u64 hits = ++s_hits[addr];
		if (static const u64 s_every = []
			{
				const char* v = std::getenv("RPCS3_PPU_WATCH_EVERY");
				return v ? std::max<u64>(1, std::strtoull(v, nullptr, 10)) : 1000ull;
			}();
			hits == 1 || hits % s_every == 0)
		{
			std::string key = fmt::format("0x%x LR 0x%x", addr, static_cast<u32>(ppu.lr));
			const auto list = ppu.dump_callstack_list();
			for (usz i = 0; i < std::min<usz>(list.size(), 8); i++)
			{
				fmt::append(key, " <- 0x%x", list[i].first);
			}
			std::string value;
			if (is_float)
			{
				value = fmt::format("f%u=%g", op.frs, ppu.fpr[op.frs]);
			}
			else if (size == 16)
			{
				const v128 v = ppu.vr[op.vs];
				value = fmt::format("v%u=(%g %g %g %g)", op.vs, v._f[3], v._f[2], v._f[1], v._f[0]);
			}
			else
			{
				value = fmt::format("r%u=0x%llx", op.rs, ppu.gpr[op.rs]);
			}
			ppu_log.success("WATCH store to 0x%x (%u bytes) %s at %s [%s, hit %u]", ea, size, value, key, ppu.get_name(), hits);
		}
	}

	return ppu_cache(addr)(ppu, {*this_op}, this_op, next_fn);
}

// VR fork dev hook: a read watch, as the write watch but for load instructions: logs
// each distinct load address (with the loaded value and call stack) that reads from
// [s_rwatch_addr, s_rwatch_addr + s_rwatch_len). Interpreter only.
static u32 s_rwatch_addr = 0;
static u32 s_rwatch_len = 0;

static void ppu_rwatch_break(ppu_thread& ppu, ppu_opcode_t, be_t<u32>* this_op, ppu_intrp_func* next_fn)
{
	const u32 addr = vm::get_addr(this_op);
	const ppu_opcode_t op{*this_op};
	const u64 base = op.ra ? ppu.gpr[op.ra] : 0;
	u32 ea = 0, size = 4;
	switch (op.main)
	{
	case 32:
	case 33: ea = static_cast<u32>(base + op.simm16); break; // lwz(u)
	case 34:
	case 35:
		ea = static_cast<u32>(base + op.simm16);
		size = 1;
		break; // lbz(u)
	case 40:
	case 41:
	case 42:
	case 43:
		ea = static_cast<u32>(base + op.simm16);
		size = 2;
		break; // lhz(u), lha(u)
	case 46:
		ea = static_cast<u32>(base + op.simm16);
		size = (32 - op.rd) * 4;
		break; // lmw
	case 48:
	case 49: ea = static_cast<u32>(base + op.simm16); break; // lfs(u)
	case 50:
	case 51:
		ea = static_cast<u32>(base + op.simm16);
		size = 8;
		break; // lfd(u)
	case 58:
		ea = static_cast<u32>(base + (op.simm16 & ~3));
		size = (op.opcode & 3) == 2 ? 4 : 8;
		break; // ld(u), lwa
	case 31:
	{
		ea = static_cast<u32>(base + ppu.gpr[op.rb]);
		switch (op.opcode >> 1 & 0x3ff)
		{
		case 87:
		case 119: size = 1; break; // lbzx(u)
		case 103:
		case 359:
			ea &= ~15u;
			size = 16;
			break; // lvx(l)
		case 519:
		case 551: size = 16; break; // lvlx, lvrx (approximate range)
		case 790: size = 2; break;  // lhbrx
		case 84: size = 8; break;   // ldarx
		case 279:
		case 311:
		case 343:
		case 375: size = 2; break; // lhzx(u), lhax(u)
		case 21:
		case 53:
		case 599:
		case 631: size = 8; break; // ldx(u), lfdx(u)
		default: break;            // lwzx(u), lfsx(u)
		}
		break;
	}
	default: size = 0; break;
	}

	if (size && ea < s_rwatch_addr + s_rwatch_len && ea + size > s_rwatch_addr)
	{
		static std::mutex s_mutex;
		static std::map<u32, u64> s_hits;
		std::lock_guard lock(s_mutex);
		const u64 hits = ++s_hits[addr];
		if (static const u64 s_every = []
			{
				const char* v = std::getenv("RPCS3_PPU_WATCH_EVERY");
				return v ? std::max<u64>(1, std::strtoull(v, nullptr, 10)) : 1000ull;
			}();
			hits == 1 || hits % s_every == 0)
		{
			std::string key = fmt::format("0x%x LR 0x%x", addr, static_cast<u32>(ppu.lr));
			const auto list = ppu.dump_callstack_list();
			for (usz i = 0; i < std::min<usz>(list.size(), 8); i++)
			{
				fmt::append(key, " <- 0x%x", list[i].first);
			}
			const u32 value = vm::check_addr(ea & ~3u) ? static_cast<u32>(vm::read32(ea & ~3u)) : 0u;
			ppu_log.success("RWATCH load from 0x%x (%u bytes, word 0x%x) at %s [%s, hit %u] r3=0x%x r4=0x%x r24..r31=0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", ea, size, value, key, ppu.get_name(), hits,
				static_cast<u32>(ppu.gpr[3]), static_cast<u32>(ppu.gpr[4]), static_cast<u32>(ppu.gpr[24]), static_cast<u32>(ppu.gpr[25]), static_cast<u32>(ppu.gpr[26]), static_cast<u32>(ppu.gpr[27]),
				static_cast<u32>(ppu.gpr[28]), static_cast<u32>(ppu.gpr[29]), static_cast<u32>(ppu.gpr[30]), static_cast<u32>(ppu.gpr[31]));
		}
	}

	return ppu_cache(addr)(ppu, {*this_op}, this_op, next_fn);
}

// Installs the read watch on every load instruction in [start, end). Returns the count.
extern u32 ppu_rwatch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end)
{
	if (g_cfg.core.ppu_decoder == ppu_decoder_type::llvm)
	{
		return 0;
	}

	s_rwatch_addr = watch_addr;
	s_rwatch_len = watch_len;
	u32 count = 0;
	for (u32 addr = start; addr < end; addr += 4)
	{
		if (!vm::check_addr(addr, vm::page_executable))
		{
			continue;
		}
		const ppu_opcode_t op{vm::read32(addr)};
		bool load = (op.main >= 32 && op.main <= 35) || (op.main >= 40 && op.main <= 43) || op.main == 46 || (op.main >= 48 && op.main <= 51) || op.main == 58;
		if (op.main == 31)
		{
			switch (op.opcode >> 1 & 0x3ff)
			{
			case 23:
			case 55:
			case 87:
			case 119:
			case 279:
			case 311:
			case 343:
			case 375:
			case 21:
			case 53:
			case 535:
			case 567:
			case 599:
			case 631:
			case 20:
			case 84:
			case 534:
			case 790:
			case 103:
			case 359:
			case 519:
			case 551:
				load = true;
				break;
			default: break;
			}
		}
		if (load && ppu_read(addr) != &ppu_rwatch_break && ppu_read(addr) != &ppu_watch_break && ppu_read(addr) != &ppu_trace_break)
		{
			write_to_ptr_unsafe<ppu_intrp_func_t>(ppu_ptr(addr), &ppu_rwatch_break);
			count++;
		}
	}
	return count;
}

// Installs the watch on every store instruction in [start, end). Returns the count.
extern u32 ppu_watch_install(u32 watch_addr, u32 watch_len, u32 start, u32 end)
{
	if (g_cfg.core.ppu_decoder == ppu_decoder_type::llvm)
	{
		return 0;
	}

	s_watch_addr = watch_addr;
	s_watch_len = watch_len;
	u32 count = 0;
	for (u32 addr = start; addr < end; addr += 4)
	{
		if (!vm::check_addr(addr, vm::page_executable))
		{
			continue;
		}
		const ppu_opcode_t op{vm::read32(addr)};
		bool store = (op.main >= 36 && op.main <= 39) || op.main == 44 || op.main == 45 || op.main == 47 || (op.main >= 52 && op.main <= 55) || (op.main == 62 && (op.opcode & 3) < 2);
		if (op.main == 31)
		{
			switch (op.opcode >> 1 & 0x3ff)
			{
			case 151:
			case 183:
			case 215:
			case 247:
			case 407:
			case 439:
			case 149:
			case 181:
			case 663:
			case 695:
			case 727:
			case 759:
			case 231:
			case 487:
			case 199:
			case 150:
			case 214:
			case 662:
			case 918:
			case 983:
			case 647:
			case 679:
			case 775:
			case 807:
				store = true;
				break;
			default: break;
			}
		}
		if (store && ppu_read(addr) != &ppu_watch_break && ppu_read(addr) != &ppu_trace_break)
		{
			write_to_ptr_unsafe<ppu_intrp_func_t>(ppu_ptr(addr), &ppu_watch_break);
			count++;
		}
	}
	return count;
}

extern bool ppu_trace_breakpoint(u32 addr)
{
	if (addr % 4 || !vm::check_addr(addr, vm::page_executable) || g_cfg.core.ppu_decoder == ppu_decoder_type::llvm || ppu_read(addr) == &ppu_trace_break)
	{
		return false;
	}

	write_to_ptr_unsafe<ppu_intrp_func_t>(ppu_ptr(addr), &ppu_trace_break);
	return true;
}
