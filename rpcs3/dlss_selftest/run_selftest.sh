#!/bin/bash
# DLSS test: runs dlss_selftest over its scenarios and checks the motion vectors (dlss_selftest_check.py).
#
#   run_selftest.sh <dlss_selftest binary> <output dir> [frames]
#
# Needs Xvfb (xvfb-run), a Vulkan driver (lavapipe is enough) and python3 with numpy and Pillow.
# Extra settings come from the environment, e.g. DLSS_SELFTEST_VALIDATION=1 (Khronos validation layer),
# RPCS3_DLSS_MOTION_16F=1 (RGBA16F motion image). DLSS_SELFTEST_MSAA_RUNS (default "1 4") lists the scene target's
# sample counts to test (2 is Gran Turismo 5's, which lavapipe does not support).
set -u
BIN=$(realpath "$1")
OUT=$(realpath -m "$2")
FRAMES=${3:-24}
CHECK=$(dirname "$(realpath "$0")")/dlss_selftest_check.py
mkdir -p "$OUT"
failed=0
for msaa in ${DLSS_SELFTEST_MSAA_RUNS:-1 4}; do
	for scenario in static object camera both; do
		dir="$OUT/${scenario}_msaa${msaa}"
		rm -rf -- "$dir"
		(cd "$(dirname "$BIN")" && env RPCS3_DLSS=motionraw RPCS3_DLSS_CAMERA=0 DLSS_SELFTEST_MSAA=$msaa \
			timeout 300 xvfb-run -a -s "-screen 0 1280x720x24" "$BIN" "$dir" "$FRAMES" "$scenario" > "$dir.stdout" 2>&1)
		result=$(python3 "$CHECK" "$dir" | tail -1)
		echo "$scenario, ${msaa}x: $result"
		if ! grep -q "^\([1-9][0-9]*\) of \1 " <<< "$result"; then
			failed=1
		fi
	done
done
exit $failed
