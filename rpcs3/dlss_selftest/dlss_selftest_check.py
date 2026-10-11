#!/usr/bin/env python3
"""DLSS test: checks dlss_selftest's screenshots taken with RPCS3_DLSS=motionraw.

Each screenshot encodes the motion vector of every pixel: red/green = 128 + 4 x motion (pixels, towards the previous
frame), blue = 255 where the motion pass covered the pixel. expected.txt holds, per frame and probe point, the point's
window position and its expected motion. A probe passes when the decoded motion at its pixel is within the tolerance.

  dlss_selftest_check.py <output dir> [tolerance pixels]
"""
import os
import sys

import numpy as np
from PIL import Image


def main():
    out_dir = sys.argv[1]
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    names = ["floor", "static cube", "moving cube"]
    failures = 0
    checked = 0

    with open(os.path.join(out_dir, "expected.txt")) as f:
        rows = [line.split() for line in f if line.strip()]

    for row in rows:
        frame, probe = int(row[0]), int(row[1])
        x, y, ex, ey = map(float, row[2:6])
        path = os.path.join(out_dir, "frame_%03d.png" % frame)
        if not os.path.exists(path):
            continue
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
        h, w, _ = img.shape
        # The probe sits on an edge or top face; sample a few pixels just inside it (below the top edge) and take
        # the median of the covered ones.
        samples = []
        for dy in (2, 3, 4, 5):
            for dx in (-1, 0, 1):
                px, py = int(round(x)) + dx, int(round(y)) + dy
                if 0 <= px < w and 0 <= py < h and img[py, px, 2] > 127:
                    samples.append(((img[py, px, 0] - 128.0) / 4.0, (img[py, px, 1] - 128.0) / 4.0))
        checked += 1
        if not samples:
            print("frame %3d %-12s at (%7.1f, %7.1f): NOT COVERED (expected %+.2f, %+.2f)" % (frame, names[probe], x, y, ex, ey))
            failures += 1
            continue
        mx = float(np.median([s[0] for s in samples]))
        my = float(np.median([s[1] for s in samples]))
        ok = abs(mx - ex) <= tolerance and abs(my - ey) <= tolerance
        failures += 0 if ok else 1
        print("frame %3d %-12s at (%7.1f, %7.1f): motion %+6.2f, %+6.2f  expected %+6.2f, %+6.2f  %s"
              % (frame, names[probe], x, y, mx, my, ex, ey, "ok" if ok else "FAIL"))

    print("%d of %d probes within %.2f px" % (checked - failures, checked, tolerance))
    return 1 if failures or not checked else 0


if __name__ == "__main__":
    sys.exit(main())
