"""Score the C++ plugin algorithm (plugin/build/oe_offline.exe) against ground truth.

Runs the same synthetic clip as validate.py through the C++ code and prints PSNR
next to the Python prototype's result, so we know the port is faithful.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from oneeighty import process, read_frames

HERE = Path(__file__).parent
EXE = HERE.parent / "plugin" / "build" / "oe_offline.exe"
SRC, GT = HERE / "test_src_50p.mkv", HERE / "test_gt_25p.mkv"
W, H = 1280, 720


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def to_raw(src, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "rawvideo",
                    "-pix_fmt", "bgra", str(dst)], check=True)


raw_in, raw_out = HERE / "_src.bgra", HERE / "_out.bgra"
to_raw(SRC, raw_in)
# exact timing off: Premiere's frame plus neighbours, as the Python "flow" mode
r = subprocess.run([str(EXE), str(raw_in), str(W), str(H), "50", "25", str(raw_out)],
                   capture_output=True, text=True, env=dict(os.environ, OE_EXACT="0"))
print(r.stdout.strip(), r.stderr.strip())
if r.returncode:
    sys.exit(r.returncode)

out = np.fromfile(raw_out, np.uint8).reshape(-1, H, W, 4)[..., :3]
gt = list(read_frames(str(GT)))
py = [np.clip(f + 0.5, 0, 255).astype(np.uint8)
      for f in process(read_frames(str(SRC)), 50, 25, "flow")]
n = len(gt)
cpp_score = np.mean([psnr(out[i], gt[i]) for i in range(1, n - 1)])
py_score = np.mean([psnr(py[i], gt[i]) for i in range(1, n - 1)])
cpp_vs_py = np.mean([psnr(out[i], py[i]) for i in range(1, n - 1)])
print(f"C++ plugin vs ground truth : {cpp_score:.2f} dB")
print(f"Python proto vs ground truth: {py_score:.2f} dB")
print(f"C++ vs Python (agreement)  : {cpp_vs_py:.2f} dB")
raw_in.unlink()
raw_out.unlink()
