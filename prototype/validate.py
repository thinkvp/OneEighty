"""Compare each mode against the ground-truth 25p/180deg clip.

Prints PSNR (higher = closer to a real 180deg shutter), overall and for the
moving-subject region, and writes compare.png with crops from one frame.
"""
import numpy as np
import cv2

from oneeighty import process, read_frames

SRC, GT = "test_src_50p.mkv", "test_gt_25p.mkv"
SUBJECT = (slice(200, 520), slice(0, 1280))  # band the swinging square moves in
SHOW = 10                                   # frame used for the picture


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def run(mode, **kw):
    return [np.clip(f, 0, 255).astype(np.uint8)
            for f in process(read_frames(SRC), 50, 25, mode, **kw)]


gt = list(read_frames(GT))
variants = {
    "drop (Premiere)": run("drop"),
    "blend2 (Echo)": run("blend2"),
    "flow support": run("flow", rule="support"),
    "flow variance": run("flow", rule="variance"),
}

print(f"{'mode':18s} {'PSNR all':>9s} {'PSNR subject':>13s} {'PSNR bg':>8s}")
n = len(gt)
for name, out in variants.items():
    assert len(out) >= n, (name, len(out), n)
    # skip first/last frame (one-sided flow)
    a = np.mean([psnr(out[i], gt[i]) for i in range(1, n - 1)])
    s = np.mean([psnr(out[i][SUBJECT], gt[i][SUBJECT]) for i in range(1, n - 1)])
    b = np.mean([psnr(out[i][:200], gt[i][:200]) for i in range(1, n - 1)])
    print(f"{name:18s} {a:9.2f} {s:13.2f} {b:8.2f}")

# picture: crop around the subject for each variant + ground truth
f = SHOW
crops = []
for name, img in [*((k, v[f]) for k, v in variants.items()), ("ground truth", gt[f])]:
    c = img[180:540, 200:1080].copy()
    cv2.putText(c, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(c, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    crops.append(c)
grid = np.vstack([np.hstack(crops[0:2]), np.hstack(crops[2:4]),
                  np.hstack([crops[4], np.zeros_like(crops[4])])])
cv2.imwrite("compare.png", grid)
print("wrote compare.png")
