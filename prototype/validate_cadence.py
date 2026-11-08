"""59.94p -> 25p: does the frame-timing fix remove the uneven cadence?

59.94/25 = 2.4 source frames per output frame, so Premiere shows source frames
0, 2, 5, 7, 10, 12, ... : motion advances in uneven 2- and 3-frame steps
(33 ms / 50 ms) = judder. Exact frame timing (the default) builds each output
frame at its exact time from the source frames either side instead.

Reports, per method:
  PSNR vs ground truth (real 25p camera, 180 deg)    higher = better
  pan step mean / std: background shift between consecutive output frames,
  measured by phase correlation on the background band. A steady 25p pan has
  std ~0; judder shows up as a large std.
"""
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from oneeighty import read_frames

HERE = Path(__file__).parent
BUILD = HERE.parent / "plugin" / "build"
SRC, GT = HERE / "test_src_5994p.mkv", HERE / "test_gt_25p.mkv"
W, H = 1280, 720
SRC_FPS = 60000 / 1001
BAND = (slice(0, 200), slice(0, W))  # background only (subject lives in 250..470)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def pan_steps(frames):
    win = cv2.createHanningWindow((W, 200), cv2.CV_32F)
    g = [cv2.cvtColor(f[BAND], cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    return np.array([cv2.phaseCorrelate(g[i], g[i + 1], win)[0][0] for i in range(len(g) - 1)])


def run(exe, exact, quality="normal"):
    env = dict(os.environ, OE_EXACT="1" if exact else "0", OE_QUALITY=quality)
    r = subprocess.run([str(BUILD / exe), str(raw_in), str(W), str(H), f"{SRC_FPS:.8f}", "25", str(raw_out)],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return list(np.fromfile(raw_out, np.uint8).reshape(-1, H, W, 4)[..., :3])


raw_in, raw_out = HERE / "_src.bgra", HERE / "_out.bgra"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SRC), "-f", "rawvideo", "-pix_fmt", "bgra", str(raw_in)],
               check=True)
src = list(read_frames(str(SRC)))
gt = list(read_frames(str(GT)))
r = SRC_FPS / 25
drop = [src[int(round(k * r))] for k in range(len(gt)) if round(k * r) < len(src)]

methods = {
    "ground truth (25p camera)": gt,
    "Premiere (nearest frame)": drop,
    "CPU, exact timing off": run("oe_offline.exe", False),
    "CPU, exact timing": run("oe_offline.exe", True),
    "GPU, exact timing off": run("oe_gpu_offline.exe", False),
    "GPU, exact timing, Normal": run("oe_gpu_offline.exe", True),
    "GPU, exact timing, High": run("oe_gpu_offline.exe", True, "high"),
    "GPU, exact timing, Extra High": run("oe_gpu_offline.exe", True, "extra"),
}

def tpsnr(frames, i):
    d = (frames[i + 1].astype(np.float32) - frames[i]) - (gt[i + 1].astype(np.float32) - gt[i])
    return 10 * np.log10(255 ** 2 / max(float(np.mean(d ** 2)), 1e-9))


n = min(len(v) for v in methods.values())
print(f"{'method':28s} {'PSNR':>7s} {'tPSNR':>7s} {'pan step mean':>14s} {'std':>6s}  (frames 1..{n - 2})")
for name, frames in methods.items():
    score = np.mean([psnr(frames[i], gt[i]) for i in range(1, n - 1)])
    tscore = np.mean([tpsnr(frames, i) for i in range(1, n - 2)])
    steps = pan_steps(frames[1:n - 1])
    print(f"{name:28s} {score:7.2f} {tscore:7.2f} {steps.mean():14.2f} {steps.std():6.2f}")

raw_in.unlink(missing_ok=True)
raw_out.unlink(missing_ok=True)
