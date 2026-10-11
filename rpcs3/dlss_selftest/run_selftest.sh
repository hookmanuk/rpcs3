#!/bin/bash
# DLSS test: runs dlss_selftest over its scenarios and checks the motion vectors (dlss_selftest_check.py).
#
#   run_selftest.sh <dlss_selftest binary> <output dir> [frames]
#
# Needs Xvfb (xvfb-run), a Vulkan driver (lavapipe is enough) and python3 with numpy and Pillow.
# Extra settings come from the environment, e.g. DLSS_SELFTEST_VALIDATION=1 (Khronos validation layer),
# RPCS3_DLSS_MOTION_16F=1 (RGBA16F motion image). DLSS_SELFTEST_MSAA_RUNS (default "1 4") lists the scene target's
# sample counts to test (2 is Gran Turismo 5's, which lavapipe does not support). DLSS_SELFTEST_MODES (default
# "flat vr") lists the modes: flat, and VR multiview with the fake headset (both eyes checked). With VR, the DLSS output
# path for the headset is checked too (DLSS_SELFTEST_UPSCALE=0 skips it): RPCS3_DLSS_STANDIN puts a bilinear upscale
# in NGX's place, and each eye must come out as its own eye upscaled 2x (dlss_selftest_upscale_check.py).
set -u
BIN=$(realpath "$1")
OUT=$(realpath -m "$2")
FRAMES=${3:-24}
CHECK=$(dirname "$(realpath "$0")")/dlss_selftest_check.py
UPSCALE_CHECK=$(dirname "$(realpath "$0")")/dlss_selftest_upscale_check.py
mkdir -p "$OUT"
failed=0
for mode in ${DLSS_SELFTEST_MODES:-flat vr}; do
	vr=""
	[ "$mode" = vr ] && vr=1
	for msaa in ${DLSS_SELFTEST_MSAA_RUNS:-1 4}; do
		for scenario in static object camera both; do
			dir="$OUT/${mode}_${scenario}_msaa${msaa}"
			rm -rf -- "$dir"
			(cd "$(dirname "$BIN")" && env RPCS3_DLSS=motionraw RPCS3_DLSS_CAMERA=0 DLSS_SELFTEST_MSAA=$msaa ${vr:+DLSS_SELFTEST_VR=1} \
				timeout 300 xvfb-run -a -s "-screen 0 1280x720x24" "$BIN" "$dir" "$FRAMES" "$scenario" > "$dir.stdout" 2>&1)
			result=$(python3 "$CHECK" "$dir" | tail -1)
			echo "$mode, $scenario, ${msaa}x: $result"
			if ! grep -q "^\([1-9][0-9]*\) of \1 " <<< "$result"; then
				failed=1
			fi
		done
	done
done
if [[ " ${DLSS_SELFTEST_MODES:-flat vr} " == *" vr "* && ${DLSS_SELFTEST_UPSCALE:-1} != 0 ]]; then
	# Without jitter, so that both runs render the same eyes. The runs share the test profile: one at a time.
	for run in reference:jitter standin:performance; do
		dir="$OUT/vr_upscale_${run%%:*}"
		standin=""
		[ "${run%%:*}" = standin ] && standin=1
		rm -rf -- "$dir"
		(cd "$(dirname "$BIN")" && env RPCS3_DLSS=${run#*:} RPCS3_DLSS_JITTER_SCALE=0 RPCS3_DLSS_CAMERA=0 DLSS_SELFTEST_VR=1 \
			${standin:+RPCS3_DLSS_STANDIN=1} timeout 300 xvfb-run -a -s "-screen 0 1280x720x24" "$BIN" "$dir" "$FRAMES" both > "$dir.stdout" 2>&1)
	done
	result=$(python3 "$UPSCALE_CHECK" "$OUT/vr_upscale_reference" "$OUT/vr_upscale_standin" 2 | tail -1)
	echo "vr, DLSS output per eye (stand-in): $result"
	if ! grep -q "^\([1-9][0-9]*\) of \1 " <<< "$result"; then
		failed=1
	fi
fi
exit $failed
