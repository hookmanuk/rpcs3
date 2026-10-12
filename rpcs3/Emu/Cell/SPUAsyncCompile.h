#pragma once

class spu_thread;

// VR fork: an SPU that reaches code nobody compiled yet runs it through the interpreter while the code compiles on a
// background thread, instead of stopping until the compile is done. Called by spu_recompiler_base::dispatch() before
// it compiles; true: handled (the SPU went back to the gateway, which picks up the compiled code), false: compile as
// usual.
bool spu_async_dispatch(spu_thread& spu);
