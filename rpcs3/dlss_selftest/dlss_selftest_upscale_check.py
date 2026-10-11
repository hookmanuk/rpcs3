#!/usr/bin/env python3
# DLSS test: checks the VR DLSS output path with RPCS3_DLSS_STANDIN (a bilinear upscale in NGX's place).
#
#   dlss_selftest_upscale_check.py <reference dir> <stand-in dir> <scale>
#
# Both runs are dlss_selftest VR runs of the same scenario without jitter: the reference shows the eyes as rendered
# (RPCS3_DLSS=jitter, RPCS3_DLSS_JITTER_SCALE=0), the stand-in run through the per-eye DLSS outputs. The screenshots
# hold both eyes side by side. Each eye of the stand-in run must be its own reference eye upscaled by <scale> (not the
# other eye): the per-eye inputs, outputs, sizes and presentation work as they will with DLSS.
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def eyes(img):
    w, h = img.size
    return [img.crop((e * w // 2, 0, (e + 1) * w // 2, h)) for e in range(2)]


def main():
    ref_dir, test_dir, scale = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    frames = sorted(p.name for p in ref_dir.glob("frame_*.png") if (test_dir / p.name).exists())
    if not frames:
        print("FAIL: no frames in both runs")
        return 1

    failures = 0
    for name in frames:
        ref = eyes(Image.open(ref_dir / name).convert("RGB"))
        test = eyes(Image.open(test_dir / name).convert("RGB"))
        expected_size = (round(ref[0].size[0] * scale), round(ref[0].size[1] * scale))
        notes = []
        ok = test[0].size == expected_size
        if not ok:
            notes.append(f"eye size {test[0].size}, expected {expected_size}")
        else:
            for e in range(2):
                got = np.asarray(test[e]).astype(np.float32)
                own = np.abs(np.asarray(ref[e].resize(expected_size, Image.BILINEAR)).astype(np.float32) - got)
                other = np.abs(np.asarray(ref[1 - e].resize(expected_size, Image.BILINEAR)).astype(np.float32) - got)
                eye_ok = own.mean() < 1.0 and np.percentile(own, 99) <= 8 and own.mean() < other.mean()
                ok &= eye_ok
                notes.append(f"eye {e}: mean diff {own.mean():.2f} (other eye {other.mean():.2f})")
        failures += not ok
        print(f"{name}: {'ok' if ok else 'FAIL'}, {'; '.join(notes)}")

    print(f"{len(frames) - failures} of {len(frames)} frames upscaled per eye as expected ({scale}x)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
