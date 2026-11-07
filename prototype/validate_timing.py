"""59.94p -> 25p frame-timing strategies, including frame-to-frame stability.

  flow_nearest : blur, nearest source frame's moment (v0.3; Premiere's timing)
  flow         : blur shifted to the exact output time from ONE frame (v0.4)
  flow2        : two-frame interpolation (frames either side, weighted)

Metrics vs ground truth (real 25p camera, 180 deg), higher = better except std:
  PSNR      per-frame accuracy
  tPSNR     accuracy of the change between consecutive frames: catches
            flicker / edge artefacts that jump around from frame to frame
  edge      PSNR within 24 px of the moving subject's outline
  step std  unevenness of the background pan (judder), px
"""
import cv2
import numpy as np

from oneeighty import process, read_frames

SRC, GT = "test_src_5994p.mkv", "test_gt_25p.mkv"
FPS = 60000 / 1001
W = 1280
SUBJECT = (slice(200, 520), slice(0, 1280))


def psnr(err_sq):
    return 10 * np.log10(255 ** 2 / max(float(err_sq), 1e-9))


def edge_mask(g1, g0):
    diff = cv2.absdiff(g1, g0).max(axis=2)
    m = np.zeros(diff.shape, np.uint8)
    m[SUBJECT] = (diff[SUBJECT] > 40).astype(np.uint8)
    return cv2.dilate(m, np.ones((49, 49), np.uint8)).astype(bool)


def pan_steps(frames):
    win = cv2.createHanningWindow((W, 200), cv2.CV_32F)
    g = [cv2.cvtColor(f[:200], cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    return np.array([cv2.phaseCorrelate(g[i], g[i + 1], win)[0][0] for i in range(len(g) - 1)])


gt = [f.astype(np.float32) for f in read_frames(GT)]
print(f"{'method':14s} {'PSNR':>7s} {'tPSNR':>7s} {'edge':>7s} {'step std':>9s}")
for mode in ["drop", "flow_nearest", "flow", "flow2"]:
    out = [np.clip(f, 0, 255).astype(np.float32) for f in process(read_frames(SRC), FPS, 25, mode)]
    n = min(len(out), len(gt))
    ks = range(2, n - 2)
    p = np.mean([psnr(np.mean((out[k] - gt[k]) ** 2)) for k in ks])
    t = np.mean([psnr(np.mean(((out[k + 1] - out[k]) - (gt[k + 1] - gt[k])) ** 2)) for k in ks])
    e = np.mean([psnr(np.mean(((out[k] - gt[k]) ** 2)[edge_mask(gt[k], gt[k - 1])])) for k in ks])
    s = pan_steps([o.astype(np.uint8) for o in out[2:n - 2]]).std()
    print(f"{mode:14s} {p:7.2f} {t:7.2f} {e:7.2f} {s:9.2f}")
