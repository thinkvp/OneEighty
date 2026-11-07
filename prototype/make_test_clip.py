"""Render a synthetic test clip with a known-correct answer.

Scene: textured background panning right-to-left (camera pan) plus a textured
square swinging left/right at varying speed (moving subject, occlusion edges).

Outputs (lossless FFV1):
  test_src_50p.mkv   - what the camera records: 50p, 180 deg shutter (10 ms)
  test_gt_25p.mkv    - ground truth: 25p, 180 deg shutter (20 ms)
Motion blur is rendered by averaging many sub-frame renders.
"""
import numpy as np
import cv2

from oneeighty import Writer

W, H = 1280, 720
DUR = 2.0
rng = np.random.default_rng(1)


def texture(w, h, seed):
    r = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for scale in (4, 16, 64):
        n = r.random((h // scale + 2, w // scale + 2, 3)).astype(np.float32)
        img += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)[:h, :w] / 3
    img = (img - img.min()) / (img.max() - img.min()) * 200 + 20
    for x in range(0, w, 80):
        cv2.line(img, (x, 0), (x, h), (240, 240, 240), 2)
    for y in range(0, h, 80):
        cv2.line(img, (0, y), (w, y), (30, 30, 30), 2)
    for _ in range(w * h // 20000):
        c = tuple(float(v) for v in r.integers(0, 255, 3))
        cv2.circle(img, (int(r.integers(0, w)), int(r.integers(0, h))),
                   int(r.integers(5, 30)), c, -1)
    return img


BG = texture(W + 3000, H + 200, 1)
OBJ = texture(220, 220, 2)
cv2.rectangle(OBJ, (0, 0), (219, 219), (0, 0, 255), 6)


def render(t):
    px, py = 100 + 1200 * t, 50 + 40 * t                 # camera pan (px/s)
    M = np.float32([[1, 0, -px], [0, 1, -py]])
    frame = cv2.warpAffine(BG, M, (W, H), flags=cv2.INTER_LINEAR)
    ox = 530 + 420 * np.sin(2 * np.pi * t / 1.6)          # subject, up to ~1650 px/s
    oy = 250
    M2 = np.float32([[1, 0, ox], [0, 1, oy]])
    obj = cv2.warpAffine(OBJ, M2, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones((220, 220), np.float32), M2, (W, H),
                          flags=cv2.INTER_LINEAR)[..., None]
    return frame * (1 - mask) + obj * mask


def exposure(t_center, shutter_s, samples):
    acc = np.zeros((H, W, 3), np.float32)
    for j in range(samples):
        acc += render(t_center + shutter_s * ((j + 0.5) / samples - 0.5))
    return acc / samples


def write(path, fps, shutter_deg, samples):
    wr = Writer(path, W, H, fps)
    n = int(DUR * fps)
    for i in range(n):
        f = exposure(i / fps, shutter_deg / 360 / fps, samples)
        wr.write(np.clip(f + 0.5, 0, 255))
    wr.close()
    print(f"wrote {path} ({n} frames)")


if __name__ == "__main__":
    write("test_src_50p.mkv", 50, 180, 16)
    write("test_gt_25p.mkv", 25, 180, 32)
