"""32-bit float path vs 8-bit path (59.94p synthetic clip -> 25p, exact timing).

The float path should match the 8-bit path except for rounding (a float run
avoids 8-bit quantisation, so tiny differences are expected), score the same
against ground truth, and pass values above 1.0 through without clipping.
"""
import os
import subprocess
from pathlib import Path

import numpy as np

from oneeighty import read_frames

HERE = Path(__file__).parent
BUILD = HERE.parent / "plugin" / "build"
SRC, GT = HERE / "test_src_5994p.mkv", HERE / "test_gt_25p.mkv"
W, H, FPS = 1280, 720, 60000 / 1001


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def run(exe, cadence, flt, extra=()):
    env = dict(os.environ, OE_CADENCE=cadence, OE_FLOAT=flt)
    r = subprocess.run([str(BUILD / exe), str(raw_in), str(W), str(H), f"{FPS:.8f}", "25", str(raw_out), *extra],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    frames = np.fromfile(raw_out, np.uint8).reshape(-1, H, W, 4)[..., :3]
    stats = [l for l in r.stdout.splitlines() if l.startswith("float output")]
    return frames, (stats[0] if stats else "")


raw_in, raw_out = HERE / "_src.bgra", HERE / "_out.bgra"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SRC), "-f", "rawvideo", "-pix_fmt", "bgra", str(raw_in)],
               check=True)
gt = list(read_frames(str(GT)))

cases = [
    ("GPU hybrid, Normal (DIS motion)", "oe_gpu_offline.exe", "4", ["640", "2", "5"]),
    ("GPU hybrid, Fast (NVIDIA motion)", "oe_gpu_offline.exe", "3", ["640", "2", "5"]),
    ("CPU only (two-frame)", "oe_offline.exe", "2", []),
]
print(f"{'path':34s} {'8-bit vs GT':>11s} {'float vs GT':>11s} {'float vs 8-bit':>15s} {'max |diff|':>10s}")
for name, exe, cad, extra in cases:
    a, _ = run(exe, cad, "0", extra)
    b, _ = run(exe, cad, "1", extra)
    n = min(len(a), len(b), len(gt))
    pa = np.mean([psnr(a[i], gt[i]) for i in range(1, n - 1)])
    pb = np.mean([psnr(b[i], gt[i]) for i in range(1, n - 1)])
    pab = np.mean([psnr(a[i], b[i]) for i in range(n)])
    mx = int(np.abs(a[:n].astype(int) - b[:n]).max())
    print(f"{name:34s} {pa:11.2f} {pb:11.2f} {pab:15.2f} {mx:10d}")

_, stats = run("oe_gpu_offline.exe", "4", "2", ["640", "2", "5"])
print("\nHDR check (input x1.5, GPU hybrid Normal):", stats)
raw_in.unlink(missing_ok=True)
raw_out.unlink(missing_ok=True)
