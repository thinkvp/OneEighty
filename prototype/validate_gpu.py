"""Score the GPU path (plugin/build/oe_gpu_offline.exe) against ground truth.

Tries several optical-flow settings (analysis width, NVOF grid size) on the
synthetic 720p clip, alongside the CPU plugin for reference.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from oneeighty import read_frames

HERE = Path(__file__).parent
BUILD = HERE.parent / "plugin" / "build"
SRC, GT = HERE / "test_src_50p.mkv", HERE / "test_gt_25p.mkv"
W, H = 1280, 720

# (label, exe, extra args)
CONFIGS = [
    ("CPU (OpenCV DIS @960)", "oe_offline.exe", []),
    ("GPU NVOF @1280 grid1", "oe_gpu_offline.exe", ["1280", "1", "5"]),
    ("GPU NVOF @1280 grid2", "oe_gpu_offline.exe", ["1280", "2", "5"]),
    ("GPU NVOF @1280 grid4", "oe_gpu_offline.exe", ["1280", "4", "5"]),
    ("GPU NVOF @640 grid1", "oe_gpu_offline.exe", ["640", "1", "5"]),
    ("GPU NVOF @640 grid2", "oe_gpu_offline.exe", ["640", "2", "5"]),
]


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


raw_in, raw_out = HERE / "_src.bgra", HERE / "_out.bgra"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SRC), "-f", "rawvideo", "-pix_fmt", "bgra",
                str(raw_in)], check=True)
gt = list(read_frames(str(GT)))
n = len(gt)
drop = list(read_frames(str(SRC)))[::2]
print(f"{'Premiere frame dropping':26s} {np.mean([psnr(drop[i], gt[i]) for i in range(1, n - 1)]):6.2f} dB")

for label, exe, extra in CONFIGS:
    r = subprocess.run([str(BUILD / exe), str(raw_in), str(W), str(H), "50", "25", str(raw_out), *extra],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"{label:26s} FAILED: {r.stdout.strip()} {r.stderr.strip()}")
        continue
    out = np.fromfile(raw_out, np.uint8).reshape(-1, H, W, 4)[..., :3]
    score = np.mean([psnr(out[i], gt[i]) for i in range(1, n - 1)])
    timing = r.stdout.strip().splitlines()[-1].split(";", 1)[-1].strip()
    print(f"{label:26s} {score:6.2f} dB   ({timing})")

raw_in.unlink(missing_ok=True)
raw_out.unlink(missing_ok=True)
