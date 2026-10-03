#pragma once

// VR fork: the VR release number. Every VR release takes the next one and it never restarts, also not when the
// release is built on a newer upstream version: 0.0.42-vr7 is followed by 0.0.43-vr8 (GitHub release tags
// v<upstream>-vrN). It leads the version postfix: "0.0.42-vr7-<commit> Alpha". Bump it for each release (see
// plans/4-next-steps.md, release procedure; plans/tools/package_release.py stops on a number already released).
// A preprocessor define rather than a constant because rpcs3_version.cpp joins it with the
// generated RPCS3_GIT_VERSION string literal at compile time.
#define RPCS3_VR_VERSION "vr8"
