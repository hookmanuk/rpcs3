#pragma once

// VR fork: builds branched from upstream 0.0.42 are 0.0.42-vr1, -vr2, ... (GitHub release tags v0.0.42-vrN).
// The tag leads the version postfix: "0.0.42-vr6-<commit> Alpha". Bump this for each release; after merging
// a newer upstream version, restart at vr1 (see plans/4-next-steps.md, release procedure).
// A preprocessor define rather than a constant because rpcs3_version.cpp joins it with the
// generated RPCS3_GIT_VERSION string literal at compile time.
#define RPCS3_VR_VERSION "vr6"
