#include "stdafx.h"
#include "SPUAsyncCompile.h"
#include "SPUThread.h"
#include "SPURecompiler.h"
#include "SPUInterpreter.h"
#include "Emu/system_config.h"
#include "Emu/IdManager.h"
#include "Emu/RSX/Capture/rsx_camera_probe.h"
#include "Utilities/Thread.h"
#include "Utilities/lockless.h"

#include <cstdlib>
#include <memory>
#include <mutex>



// VR fork: see SPUAsyncCompile.h.
//
// Why: an SPU stops for the whole analysis and compile of a program it reaches for the first time (milliseconds), at
// whatever point of the guest code that is. Gran Turismo 5's SPU job queues lost their start (SPURS ready count) when
// that pause fell between a queue finding itself empty and it storing its ready count 0, and the game froze: on a fresh
// install (an empty SPU cache) after the intro and at race starts, also on official RPCS3. With the SPU interpreter, or
// every program precompiled from the SPU cache, there is no such pause and no freeze. This keeps the SPU running
// (interpreted) while its new code compiles on a worker thread, then hands it back to the compiled code.

namespace
{
	struct spu_async_job
	{
		std::vector<be_t<u32>> ls; // The SPU's local storage when it reached the code
		u32 pc = 0;
		atomic_t<u32> done = 0;
	};

	struct spu_async_worker
	{
		lf_queue<std::shared_ptr<spu_async_job>> queue;

		void operator()()
		{
			std::unique_ptr<spu_recompiler_base> compiler;

			while (thread_ctrl::state() != thread_state::aborting)
			{
				for (auto slice = queue.pop_all(); slice; slice.pop_front())
				{
					spu_async_job& job = **slice.get();

					if (thread_ctrl::state() != thread_state::aborting)
					{
						if (!compiler)
						{
							compiler = spu_recompiler_base::make_fast_llvm_recompiler();
							compiler->init();
						}

						if (auto program = compiler->analyse(job.ls.data(), job.pc); !program.data.empty())
						{
							compiler->compile(std::move(program));
						}
					}

					job.done.release(1);
					job.done.notify_all();
				}

				thread_ctrl::wait_on(queue.get_wait_atomic(), 0);
			}
		}
	};

	struct spu_async_compile_pool
	{
		static constexpr u32 worker_count = 2;

		named_thread_group<spu_async_worker> workers{"SPUA."sv, worker_count};
		atomic_t<u32> next = 0;

		void push(std::shared_ptr<spu_async_job> job)
		{
			(workers.begin() + (next++ % worker_count))->queue.push(std::move(job));
		}

		spu_async_compile_pool& operator=(thread_state state)
		{
			if (state == thread_state::created)
			{
				// The workers already run
				return *this;
			}

			for (u32 i = 0; i < worker_count; i++)
			{
				*(workers.begin() + i) = state;
			}

			return *this;
		}
	};

	bool enabled()
	{
		if (g_cfg.core.spu_decoder != spu_decoder_type::llvm)
		{
			return false;
		}

		// Development switch: RPCS3_SPU_ASYNC=0/1 forces it off/on
		static const int s_env = []
		{
			const char* v = std::getenv("RPCS3_SPU_ASYNC");
			return v ? (*v == '1' ? 1 : 0) : -1;
		}();

		if (s_env >= 0)
		{
			return s_env == 1;
		}

		// Per game: VR profile spu_async_compile
		const auto* profile = rsx::vr::camera_probe::get().profile();
		return profile && profile->spu_async_compile;
	}

	spu_async_compile_pool& pool()
	{
		if (auto* p = g_fxo->try_get<spu_async_compile_pool>())
		{
			return *p;
		}

		static std::mutex s_mutex;
		std::lock_guard lock(s_mutex);
		g_fxo->need<spu_async_compile_pool>();
		return g_fxo->get<spu_async_compile_pool>();
	}
}

bool spu_async_dispatch(spu_thread& spu)
{
	if (!enabled())
	{
		return false;
	}

	const auto* table = g_fxo->try_get<spu_interpreter_rt>();

	if (!table)
	{
		return false;
	}

	auto job = std::make_shared<spu_async_job>();
	job->pc = spu.pc;
	job->ls.assign(spu._ptr<be_t<u32>>(0), spu._ptr<be_t<u32>>(0) + SPU_LS_SIZE / 4);
	pool().push(job);

	// Interpret (as spu_recompiler_base::old_interpreter) until the compile is done, checking between short runs.
	const u8* ls = spu._ptr<u8>(0);

	while (!job->done)
	{
		if (spu.state && spu.check_state())
		{
			break;
		}

		for (u32 i = 0; i < 256 && !spu.state; i++)
		{
			const u32 op = *reinterpret_cast<const be_t<u32>*>(ls + spu.pc);

			if (table->decode(op)(spu, {op}))
			{
				spu.pc += 4;
			}
		}
	}

	// Back to the gateway, which runs the compiled code at spu.pc (or comes back here for code still not compiled).
	spu_runtime::g_escape(&spu);
	return true;
}
