"""Pass/fail check of the built offline tools against ground truth, for CI.

Runs the CPU path (oe_offline) and the GPU path (oe_gpu_offline: CUDA on
Windows, Metal on macOS) on the synthetic clips and checks:
  - GPU accuracy vs a real 25p camera at 180 deg (PSNR) is close to the
    figures measured on Windows (README: 59.94p 38.6 dB, 50p 39.2 dB, Normal;
    High (RIFE) 40.2 / 40.5 dB, needs the RIFE model and Vulkan / MoltenVK)
  - GPU and CPU paths agree (their PSNRs within 0.6 dB)
  - motion estimated on the GPU (OE_GPUFLOW=1, the effect's default) scores
    within 0.3 dB of the CPU motion (OpenCV DIS), exact timing on and off
  - 32-bit float and ARGB-byte-order runs match the plain 8-bit BGRA run
  - the motion-vector view runs

Needs ffmpeg plus test_src_5994p.mkv, test_src_50p.mkv and test_gt_25p.mkv
(python make_test_clip.py 5994p 50p gt). Build folder: plugin/build
(Windows) or plugin/build-mac, or OE_BUILD. Exit code 1 if a check fails.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from oneeighty import read_frames

HERE = Path(__file__).parent
BUILD = Path(os.environ.get("OE_BUILD") or HERE.parent / "plugin" / ("build-mac" if sys.platform == "darwin" else "build"))
EXE = ".exe" if sys.platform == "win32" else ""
W, H = 1280, 720
GT = HERE / "test_gt_25p.mkv"
# clip, source fps, Windows accuracy for Normal and High (RIFE) (README)
CLIPS = {"59.94p": (HERE / "test_src_5994p.mkv", 60000 / 1001, 38.6, 40.2),
         "50p": (HERE / "test_src_50p.mkv", 50.0, 39.2, 40.5)}
MARGIN = 0.6  # dB allowed below the Windows figure / between CPU and GPU
FLOW_MARGIN = 0.3  # dB between GPU motion and CPU motion (Windows: within 0.05)

raw_in, raw_out = HERE / "_ci_src.bgra", HERE / "_ci_out.bgra"
failures = []
skipped = []


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def run(tool, fps, **env):
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    r = subprocess.run([str(BUILD / (tool + EXE)), str(raw_in), str(W), str(H), f"{fps:.8f}", "25", str(raw_out)],
                       capture_output=True, text=True, env=e)
    if r.returncode:
        raise RuntimeError(f"{tool} {env}: exit {r.returncode}\n{r.stdout}{r.stderr}")
    for line in r.stdout.splitlines():
        if line.startswith(("GPU engine", "RIFE ready", "wrote", "float output")):
            print(f"    {tool} {env or ''}: {line}")
    return np.fromfile(raw_out, np.uint8).reshape(-1, H, W, 4)[..., :3]


def score(frames, gt):
    n = min(len(frames), len(gt))
    return float(np.mean([psnr(frames[i], gt[i]) for i in range(1, n - 1)]))


def agree(a, b):
    n = min(len(a), len(b))
    return float(np.mean([psnr(a[i], b[i]) for i in range(n)]))


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


gt = np.array(list(read_frames(str(GT))))
for name, (clip, fps, windows_db, rife_db) in CLIPS.items():
    print(f"{name} -> 25p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "bgra", str(raw_in)],
                   check=True)
    cpu = run("oe_offline", fps)
    gpu = run("oe_gpu_offline", fps)
    cpu_db, gpu_db = score(cpu, gt), score(gpu, gt)
    check(gpu_db >= windows_db - MARGIN, f"GPU Normal {gpu_db:.2f} dB (Windows: {windows_db} dB)")
    check(abs(gpu_db - cpu_db) <= MARGIN, f"CPU Normal {cpu_db:.2f} dB, GPU within {MARGIN} dB")

    cpu_off, gpu_off = score(run("oe_offline", fps, OE_EXACT=0), gt), score(run("oe_gpu_offline", fps, OE_EXACT=0), gt)
    check(abs(gpu_off - cpu_off) <= MARGIN, f"exact timing off: CPU {cpu_off:.2f} dB, GPU {gpu_off:.2f} dB")

    gm_frames = run("oe_gpu_offline", fps, OE_GPUFLOW=1)
    gm = score(gm_frames, gt)
    check(abs(gm - gpu_db) <= FLOW_MARGIN, f"GPU motion Normal {gm:.2f} dB (CPU motion {gpu_db:.2f} dB)")
    gmo = score(run("oe_gpu_offline", fps, OE_EXACT=0, OE_GPUFLOW=1), gt)
    check(abs(gmo - gpu_off) <= FLOW_MARGIN, f"GPU motion, exact timing off {gmo:.2f} dB (CPU motion {gpu_off:.2f} dB)")
    gmf = agree(run("oe_gpu_offline", fps, OE_FLOAT=1, OE_GPUFLOW=1), gm_frames)
    check(gmf >= 50, f"GPU motion: 32-bit float matches 8-bit: {gmf:.1f} dB")

    try:
        rife = score(run("oe_gpu_offline", fps, OE_QUALITY="high"), gt)
        check(rife >= rife_db - MARGIN, f"GPU High (RIFE) {rife:.2f} dB (Windows: {rife_db} dB)")
    except RuntimeError as e:
        # RIFE refuses Apple's paravirtual GPU (GitHub's Mac runners): its
        # driver crashes under MoltenVK. Anything else is a failure.
        if "virtual GPU not supported" not in str(e):
            raise
        print("  skip  GPU High (RIFE): virtual GPU (macOS VM), RIFE not supported")
        skipped.append("High (RIFE)")

    fl = agree(run("oe_gpu_offline", fps, OE_FLOAT=1), gpu)
    check(fl >= 50, f"32-bit float matches 8-bit: {fl:.1f} dB")
    argb = agree(run("oe_gpu_offline", fps, OE_ARGB=1), gpu)
    check(argb >= 50, f"ARGB byte order matches BGRA: {argb:.1f} dB")
    if name == "50p":
        run("oe_gpu_offline", fps, OE_VIEW=1)
        run("oe_gpu_offline", fps, OE_VIEW=1, OE_GPUFLOW=1)
        check(True, "motion-vector view runs")

raw_in.unlink(missing_ok=True)
raw_out.unlink(missing_ok=True)
print(f"\n{len(failures)} check(s) failed" if failures else "\nall checks passed")
if skipped:
    print("skipped: " + ", ".join(sorted(set(skipped))))
sys.exit(1 if failures else 0)
