"""Prototype of an occlusion-aware Quality Normal (Python, not in the plugin yet).

Today's Normal blurs each pixel only along its own motion, so (1) a fast
object's motion leaks onto the background it is about to cover and smears it
(the court line under the ball), and (2) the object can't streak past its own
outline. This prototype:
  - two_sided: per pixel, takes A's motion from A->next or A->prev, whichever
    passes the forward/backward check (a region about to be covered was still
    visible a frame earlier), then refine_edges (numpy copy of
    RefineFlowEdges) tidies the narrow band left at the object's outline
  - recon_blur: the pixel's own blur along its own motion, plus the share of
    the shutter during which a clearly faster layer passing over it covers it
    (speed stands in for depth; tile/neighbourhood max of the motion finds
    the faster layers)

Usage (the input is raw 1080x1920 BGRA, read as 60p played on a 30p timeline:
output k uses A = 2k, B = 2k+1, phase 0, i.e. 200% speed):
    layered_blur.py dir outname k0 k1 [halfWidth] --twosided --refine
reads dir/src.bgra, writes dir/<outname>.bgra (40 frames, k0..k1 rendered).
Other flags (experiments): --gather (old blur), --noleak, --occ (show the
occlusion mask). Compare with crop_compare.py.
"""
import sys
import numpy as np
import cv2

d, outname, k0, k1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
HW = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4330
W, H = 1080, 1920
FB = W * H * 4
FW = 960
src = np.memmap(f"{d}/src.bgra", np.uint8, "r")
out = np.memmap(f"{d}/{outname}.bgra", np.uint8, "w+", shape=(40, H, W, 4))


def frame(i):
    return np.asarray(src[i * FB:(i + 1) * FB]).reshape(H, W, 4)


def gray(small):
    return cv2.cvtColor(small, cv2.COLOR_BGRA2GRAY)


def warp(img, fx, fy, interp=cv2.INTER_LINEAR):
    h, w = fx.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fx, gy + fy, interp, borderMode=cv2.BORDER_REPLICATE)


def fix_leak(fab, fba, col, colB):
    """Pixels of A that B covers (forward/backward mismatch) get the motion of
    nearby consistent pixels that look like them and move differently from the
    leaked motion (i.e. the layer behind)."""
    h, w = fab.shape[:2]
    back = warp(fba, fab[..., 0], fab[..., 1])
    err = np.abs(fab + back).sum(-1)
    occ = err > 1.0
    # ...and whose own motion lands on something that doesn't match them in B
    # (court moved with the ball's motion lands on the ball; a non-rigid arm
    # still lands on arm)
    ca = cv2.blur(col, (3, 3)).astype(np.float32)
    cb = warp(cv2.blur(colB, (3, 3)).astype(np.float32), fab[..., 0], fab[..., 1])
    occ &= np.abs(ca - cb).sum(-1) > 60
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    good = ~occ
    colf = cv2.blur(col, (3, 3)).astype(np.float32)
    best = fab.copy()
    acc = np.zeros((h, w, 2), np.float32)
    wsum = np.zeros((h, w), np.float32)
    same = np.zeros((h, w), np.float32)  # support for keeping the own motion
    # candidates in 8 directions out to 120 flow px; median would be nicer, a
    # weighted mean of "background-like" candidates is enough for a prototype
    cands = []
    for r in (4, 10, 20, 32, 48, 66, 90, 120):
        for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            M = np.float32([[1, 0, -dx * r], [0, 1, -dy * r]])
            sh = lambda a, i=cv2.INTER_NEAREST: cv2.warpAffine(a, M, (w, h), flags=i, borderMode=cv2.BORDER_REPLICATE)
            v = sh(fab)
            g = sh(good.astype(np.uint8)) > 0
            c = sh(colf)
            dcol = np.abs(c - colf).sum(-1)
            differs = np.abs(v - fab).sum(-1) > 2.0
            like = np.exp(-(dcol / 30.0) ** 2) * g / (1.0 + r / 30.0)
            wt = like * differs
            acc += wt[..., None] * v
            wsum += wt
            same += like * ~differs
    fill = (wsum > 1e-3) & (wsum > same)
    m = occ & fill
    best[m] = acc[m] / wsum[m, None]
    return best, occ


def tile_max(v, T, R=1, smooth=False):
    h, w = v.shape[:2]
    th, tw = (h + T - 1) // T, (w + T - 1) // T
    pad = np.zeros((th * T, tw * T, 2), np.float32)
    pad[:h, :w] = v
    t = pad.reshape(th, T, tw, T, 2).transpose(0, 2, 1, 3, 4).reshape(th, tw, T * T, 2)
    mag = (t ** 2).sum(-1)
    i = mag.argmax(-1)
    tm = np.take_along_axis(t, i[..., None, None], 2)[:, :, 0]
    # neighbour max (3x3 tiles)
    nm = tm.copy()
    nmag = (tm ** 2).sum(-1)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            s = np.roll(np.roll(tm, dy, 0), dx, 1)
            smag = (s ** 2).sum(-1)
            upd = smag > nmag
            nm[upd] = s[upd]
            nmag[upd] = smag[upd]
    if smooth:
        # interpolate tile centres (the field is a max, so stay conservative:
        # keep the longer of the interpolated and own-tile vectors)
        lin = cv2.resize(nm, (tw * T, th * T), interpolation=cv2.INTER_LINEAR)[:h, :w]
        near = cv2.resize(nm, (tw * T, th * T), interpolation=cv2.INTER_NEAREST)[:h, :w]
        return lin
    full = cv2.resize(nm, (tw * T, th * T), interpolation=cv2.INTER_NEAREST)[:h, :w]
    return full


def recon_blur(img, v, hw, S=32, T=16):
    """Two-layer reconstruction: what the pixel's own motion averages over
    (back), plus the share of the shutter during which a faster layer passing
    over it covers it (front). Speed stands in for depth (faster = in front)."""
    h, w = v.shape[:2]
    imgf = img.astype(np.float32)
    vn = tile_max(v, T, R=2, smooth=True)
    L = 2 * hw * np.sqrt((vn ** 2).sum(-1))       # dominant path length (px)
    sp = np.sqrt((v ** 2).sum(-1))
    ep = np.maximum(hw * sp, 0.5)                 # own blur half-extent
    half = S // 2
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # own blur: plain average along the pixel's own path, except samples that
    # belong to a clearly faster layer (those are counted in front instead)
    bacc = np.zeros_like(imgf)
    bw = np.zeros((h, w), np.float32)
    facc = np.zeros_like(imgf)
    fw = np.zeros((h, w), np.float32)
    SZ = 1.5
    jit = np.mod(52.9829189 * np.mod(0.06711056 * gx + 0.00583715 * gy, 1.0), 1.0) - 0.5
    for j in range(half):
        t = ((j + 0.5 + jit) / half) * 2 - 1
        for own in (False, True):
            dirv = v if own else vn
            ox, oy = t * hw * dirv[..., 0], t * hw * dirv[..., 1]
            qx, qy = gx + ox, gy + oy
            c = cv2.remap(imgf, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vq = cv2.remap(v, qx, qy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            sq = np.sqrt((vq ** 2).sum(-1))
            front = np.clip(0.5 + (sq - sp - SZ) / (2 * SZ), 0, 1)  # q clearly faster than p
            if own:
                wb = 1 - front
                bacc += wb[..., None] * c
                bw += wb
            else:
                eq = np.maximum(hw * sq, 0.5)
                dist = np.sqrt(ox * ox + oy * oy)
                cover = np.clip((eq - dist) + 0.5, 0, 1)            # q's smear reaches p
                wf = front * cover * (L / half) / np.maximum(L, 1.0)  # share of the shutter
                facc += wf[..., None] * c
                fw += wf
    bcol_fallback = imgf
    bcol = np.where((bw > 1e-3)[..., None], bacc / np.maximum(bw, 1e-6)[..., None], bcol_fallback)
    alpha = np.clip(fw, 0, 1)[..., None]
    fcol = facc / np.maximum(fw, 1e-6)[..., None]
    return np.clip(alpha * fcol + (1 - alpha) * bcol + 0.5, 0, 255).astype(np.uint8)


def gather(img, v, hw, S=32):
    h, w = v.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    acc = np.zeros(img.shape, np.float32)
    for j in range(S):
        t = ((j + 0.5) / S) * 2 - 1
        acc += cv2.remap(img.astype(np.float32), gx - t * hw * v[..., 0], gy - t * hw * v[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return np.clip(acc / S + 0.5, 0, 255).astype(np.uint8)


def consistency(f, g):
    return np.abs(f + warp(g, f[..., 0], f[..., 1])).sum(-1)


def two_sided(fn, fnb, fp, fpb):
    """A's velocity per frame from A->next (fn) and A->prev (fp), each with
    its reverse: per pixel, trust the direction whose forward/backward check
    passes (a region about to be covered was still visible a frame earlier)."""
    en = cv2.blur(consistency(fn, fnb), (5, 5))
    ep = cv2.blur(consistency(fp, fpb), (5, 5))
    wn = 1.0 / (0.25 + en ** 2)
    wp = 1.0 / (0.25 + ep ** 2)
    v = (wn[..., None] * fn - wp[..., None] * fp) / (wn + wp)[..., None]
    return v.astype(np.float32), en, ep


def refine_edges(col, flow):
    """numpy copy of RefineFlowEdges (MotionBlur.cpp)."""
    H_, W_ = flow.shape[:2]
    colb = cv2.blur(col, (3, 3)).astype(np.int32)
    src = flow.copy()
    q = cv2.resize(src, ((W_ + 3) // 4, (H_ + 3) // 4), interpolation=cv2.INTER_NEAREST)
    k = np.ones((3, 3), np.uint8)
    g = sum(cv2.morphologyEx(q[..., i], cv2.MORPH_GRADIENT, k) for i in range(2))
    mask = (g > 3.0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((2 * (42 // 4) + 3,) * 2, np.uint8))
    mask = cv2.resize(mask, (W_, H_), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return flow
    vx = [src[ys, xs, 0]]; vy = [src[ys, xs, 1]]; ww = [np.ones(len(ys), np.float32)]
    c0 = colb[ys, xs]
    spread = np.zeros(len(ys), np.float32)
    for r in (3, 8, 16, 28, 42):
        for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            cx, cy = xs + dx * r, ys + dy * r
            ok = (cx >= 0) & (cy >= 0) & (cx < W_) & (cy < H_)
            cxc, cyc = np.clip(cx, 0, W_ - 1), np.clip(cy, 0, H_ - 1)
            v = src[cyc, cxc]
            dcol = np.abs(colb[cyc, cxc] - c0).sum(-1)
            spread = np.maximum(spread, np.where(ok, np.abs(v[:, 0] - vx[0]) + np.abs(v[:, 1] - vy[0]), 0))
            vx.append(v[:, 0]); vy.append(v[:, 1])
            ww.append(np.where(ok, np.exp(-(dcol / 36.0) ** 2), 0).astype(np.float32))
    VX, VY, WW = np.stack(vx, 1), np.stack(vy, 1), np.stack(ww, 1)
    def wmed(V):
        o = np.argsort(V, 1)
        Vs = np.take_along_axis(V, o, 1); Ws = np.take_along_axis(WW, o, 1)
        cw = np.cumsum(Ws, 1)
        i = (cw >= 0.5 * cw[:, -1:]).argmax(1)
        return Vs[np.arange(len(V)), i]
    out = flow.copy()
    upd = spread >= 1.0
    out[ys[upd], xs[upd], 0] = wmed(VX)[upd]
    out[ys[upd], xs[upd], 1] = wmed(VY)[upd]
    return out


dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
for k in range(k0, k1 + 1):
    A, B = frame(2 * k), frame(2 * k + 1)
    sc = FW / W
    sa = cv2.resize(A, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    sb = cv2.resize(B, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    fab = dis.calc(gray(sa), gray(sb), None)
    fba = dis.calc(gray(sb), gray(sa), None)
    if "--twosided" in sys.argv:
        P_ = frame(2 * k - 1)
        sp_ = cv2.resize(P_, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        fap = dis.calc(gray(sa), gray(sp_), None)
        fpa = dis.calc(gray(sp_), gray(sa), None)
        fixed, en, ep_ = two_sided(fab, fba, fap, fpa)
        if "--refine" in sys.argv:
            fixed = refine_edges(sa[..., :3], fixed)
        occ = (en > 1) & (ep_ > 1)
    else:
      fixed, occ = fix_leak(fab, fba, sa[..., :3], sb[..., :3]) if "--noleak" not in sys.argv else (fab, fab[..., 0] > 1e9)
    v = cv2.resize(fixed, (W, H), interpolation=cv2.INTER_LINEAR) / sc
    o = recon_blur(A, v.astype(np.float32), HW) if "--gather" not in sys.argv else gather(A, v.astype(np.float32), HW)
    out[k] = o
    if "--occ" in sys.argv:
        dbg = A.copy()
        om = cv2.resize(occ.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        dbg[om, 2] = 255
        out[k] = dbg
    print("frame", k, "occluded %.1f%%" % (100 * occ.mean()), flush=True)
out.flush()
