"""Edge handling: plain gather vs reconstruction-filter variants.

The synthetic scene has a fast pan behind a subject swinging the other way, so
the subject's edges are where two motions meet. Scores PSNR vs ground truth
overall, in the subject band and in a background-only band, plus a tighter
"edge" score: pixels within 24 px of the subject's outline.
"""
import numpy as np
import cv2

from oneeighty import process, read_frames

GT = "test_gt_25p.mkv"
CLIPS = [("50p", "test_src_50p.mkv", 50.0), ("59.94p", "test_src_5994p.mkv", 60000 / 1001)]
MODES = ["flow", "recon_max", "recon_mcguire", "recon_gated"]
SUBJECT = (slice(200, 520), slice(0, 1280))
BG = (slice(0, 200), slice(0, 1280))


def psnr_masked(a, b, m=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    mse = d.mean() if m is None else d[m].mean()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def edge_mask(gt_frame, prev_gt):
    # moving-subject outline from the ground truth: where the subject region
    # differs strongly between consecutive GT frames, dilated
    diff = cv2.absdiff(gt_frame, prev_gt).max(axis=2)
    m = np.zeros(diff.shape, np.uint8)
    m[SUBJECT] = (diff[SUBJECT] > 40).astype(np.uint8)
    return cv2.dilate(m, np.ones((49, 49), np.uint8)).astype(bool)


gt = list(read_frames(GT))
for label, src, fps in CLIPS:
    print(f"\n{label} source -> 25p (exact timing)")
    print(f"  {'mode':15s} {'all':>7s} {'subject':>8s} {'edges':>7s} {'bg':>7s}")
    for mode in MODES:
        out = [np.clip(f + 0.5, 0, 255).astype(np.uint8) for f in process(read_frames(src), fps, 25, mode)]
        n = min(len(out), len(gt))
        rng = range(2, n - 1)
        a = np.mean([psnr_masked(out[i], gt[i]) for i in rng])
        s = np.mean([psnr_masked(out[i][SUBJECT], gt[i][SUBJECT]) for i in rng])
        e = np.mean([psnr_masked(out[i], gt[i], edge_mask(gt[i], gt[i - 1])) for i in rng])
        b = np.mean([psnr_masked(out[i][BG], gt[i][BG]) for i in rng])
        print(f"  {mode:15s} {a:7.2f} {s:8.2f} {e:7.2f} {b:7.2f}")
