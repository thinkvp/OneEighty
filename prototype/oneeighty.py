"""OneEighty prototype.

Conform high-frame-rate footage (e.g. 50p shot at 180 deg) to a lower timeline
rate (e.g. 25p) at 100% speed, adding the motion blur a proper shutter would
have recorded, instead of just dropping frames (what Premiere does).

Modes
  drop   - nearest source frame, no blur (Premiere's behaviour)
  blend2 - average of two neighbouring source frames (the old Echo trick)
  flow   - optical-flow motion blur: estimate per-pixel motion, then smear each
           pixel along its motion path to fill in the missing shutter time

Usage
  python oneeighty.py in.mov out.mov --out-fps 25 --mode flow
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import deque

import cv2
import numpy as np


# --------------------------------------------------------------------------- I/O
def probe(path: str) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den


def read_frames(path: str):
    w, h, _ = probe(path)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path,
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


class Writer:
    """Pipe BGR frames to ffmpeg. Codec chosen from the output extension."""

    CODECS = {
        ".mov": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        ".mkv": ["-c:v", "ffv1", "-pix_fmt", "bgr0"],
        ".mp4": ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"],
    }

    def __init__(self, path: str, w: int, h: int, fps: float):
        ext = path[path.rfind("."):].lower()
        codec = self.CODECS.get(ext, self.CODECS[".mp4"])
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *codec, path],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ----------------------------------------------------------------- optical flow
class Flow:
    """Dense optical flow (OpenCV DIS) computed at reduced resolution."""

    def __init__(self, max_width: int = 960):
        self.max_width = max_width
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def _small_gray(self, img):
        h, w = img.shape[:2]
        s = min(1.0, self.max_width / w)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            g = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return g, s

    def __call__(self, a, b):
        """Flow f such that a(x) ~= b(x + f(x)), in full-res pixels."""
        ga, s = self._small_gray(a)
        gb, _ = self._small_gray(b)
        f = self.dis.calc(ga, gb, None)
        if s < 1.0:
            h, w = a.shape[:2]
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / s
        return f


# ------------------------------------------------------------------ blur kernel
def blur_half_width(src_fps, out_fps, src_shutter, target_shutter, rule="variance"):
    """Half-width (in source-frame intervals) of the synthetic blur to add.

    The source frame already contains `e` of exposure; we want `T` in total.
      support : add a box of width T - e  (total smear length matches T)
      variance: add a box of width sqrt(T^2 - e^2)  (blur 'spread' matches T)
    """
    T = target_shutter / 360.0 * src_fps / out_fps
    e = src_shutter / 360.0
    if T <= e:
        return 0.0
    w = (T - e) if rule == "support" else math.sqrt(T * T - e * e)
    return w / 2.0


def flow_blur(cur, v, half_width, center=0.0, max_samples=64):
    """Average `cur` sampled along per-pixel motion v over offsets
    center +/- half_width (offsets in source-frame intervals)."""
    if half_width <= 0 and center == 0:
        return cur.copy()
    h, w = cur.shape[:2]
    speed = np.linalg.norm(v, axis=2)
    smear_px = 2 * half_width * float(np.percentile(speed, 99.5))
    n = int(np.clip(math.ceil(smear_px), 4, max_samples))
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    curf = cur.astype(np.float32)
    acc = np.zeros_like(curf)
    for j in range(n):
        o = center + half_width * (((j + 0.5) / n) * 2.0 - 1.0)
        acc += cv2.remap(curf, gx - o * v[..., 0], gy - o * v[..., 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return acc / n


# --------------------------------------------------------------------- pipeline
def windowed(frames):
    """Yield (index, prev, cur, next) with None at the ends."""
    win = deque(maxlen=3)
    i = -1
    for f in frames:
        win.append(f)
        if len(win) == 2 and i == -1:
            i = 0
            yield 0, None, win[0], win[1]
        elif len(win) == 3:
            i += 1
            yield i, win[0], win[1], win[2]
    if len(win) >= 2:
        yield i + 1, win[-2], win[-1], None
    elif len(win) == 1:
        yield 0, None, win[0], None


def process(frames, src_fps, out_fps, mode="flow", src_shutter=180.0,
            target_shutter=180.0, rule="variance", flow_width=960):
    """Generator of output frames (float32 or uint8 BGR)."""
    r = src_fps / out_fps
    hw = blur_half_width(src_fps, out_fps, src_shutter, target_shutter, rule)
    flow = Flow(flow_width) if mode == "flow" else None
    k = 0
    for i, prev, cur, nxt in windowed(frames):
        # emit every output frame whose (rounded) centre lands on source frame i
        while round(k * r) == i:
            center = k * r - i  # sub-frame timing offset, 0 for integer ratios
            if mode == "drop":
                out = cur
            elif mode == "blend2":
                out = (cur.astype(np.float32) + (prev if prev is not None else cur)) / 2
            else:
                vs = []
                if nxt is not None:
                    vs.append(flow(cur, nxt))
                if prev is not None:
                    vs.append(-flow(cur, prev))
                v = sum(vs) / len(vs)
                out = flow_blur(cur, v, hw, center)
            yield out
            k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--out-fps", type=float, default=25.0)
    ap.add_argument("--mode", choices=["drop", "blend2", "flow"], default="flow")
    ap.add_argument("--src-shutter", type=float, default=180.0,
                    help="shutter angle the footage was shot at (default 180)")
    ap.add_argument("--target-shutter", type=float, default=180.0,
                    help="shutter angle to simulate at the output rate (default 180)")
    ap.add_argument("--rule", choices=["support", "variance"], default="variance")
    ap.add_argument("--flow-width", type=int, default=960,
                    help="resolution (width) at which motion is estimated")
    a = ap.parse_args()

    w, h, fps = probe(a.input)
    wr = Writer(a.output, w, h, a.out_fps)
    n = 0
    for out in process(read_frames(a.input), fps, a.out_fps, a.mode,
                       a.src_shutter, a.target_shutter, a.rule, a.flow_width):
        wr.write(np.clip(out + 0.5, 0, 255) if out.dtype != np.uint8 else out)
        n += 1
        if n % 25 == 0:
            print(f"  {n} frames", flush=True)
    wr.close()
    print(f"wrote {n} frames @ {a.out_fps} fps -> {a.output}")


if __name__ == "__main__":
    main()
