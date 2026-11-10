// GPU motion estimation for Quality Normal / High (CUDA). Included by
// GpuMotionBlur.cu (one translation unit, so build.bat is unchanged).
//
// Same pipeline as the CPU path (EstimateFlow in MotionBlur.cpp), on frames
// already on the GPU:
//   1. area downscale to the analysis size (960 px wide), 8-bit luma and a
//      3x3-smoothed 8-bit colour copy (GrayOf / ColorOf)
//   2. OpenCV DIS (Kroeger et al. 2016) as in OpenCV 4.x modules/video
//      (dis_flow.cpp, Apache 2.0): luma pyramid, per level
//        - patch inverse search (8x8 patches, mean-normalised SSD, inverse-
//          compositional Gauss-Newton) in the CPU version's order: 8 stripes
//          swept forward and back, each patch starting from the best of its
//          own and its finished neighbours' motion; run as a wavefront
//          (KPatchSearch). OpenCV's OpenCL kernels (row scans, then
//          Gauss-Newton in parallel) were tried first: 0.6-0.8 dB worse on
//          the synthetic tests, mostly at moving objects' outlines
//        - densification (patches blended by 1 / photometric error)
//        - variational refinement (variational_refinement.cpp: colour and
//          gradient constancy + smoothness, fixed-point iterations with
//          red-black SOR); OpenCV runs this part on the CPU even in its
//          OpenCL path
//   3. motion-boundary clean-up (RefineFlowEdges in MotionBlur.cpp)

#pragma once

namespace oe::gpu::dis
{

constexpr int kPsz = 8; // patch size (every OpenCV preset uses 8)
constexpr int kMaxLevels = 10;

struct Params
{
    int finest, stride, gdIter, varIter;
};

// cv::DISOpticalFlow::create(preset): 0 ULTRAFAST, 1 FAST, 2 MEDIUM
inline Params Preset(int preset)
{
    if (preset <= 0)
        return {2, 4, 12, 0};
    if (preset == 1)
        return {2, 4, 16, 5};
    return {1, 3, 25, 5};
}

// variational refinement (DIS settings: alpha 20, gamma 10, delta 5, epsilon
// 0.01; VariationalRefinement defaults: zeta 0.1, omega 1.6; 5 SOR iterations)
constexpr float kVrAlpha = 20.0f, kVrGamma = 10.0f, kVrDelta = 5.0f, kVrEps = 0.01f, kVrZeta = 0.1f,
                kVrOmega = 1.6f;
constexpr int kVrSor = 5;

// ---------------------------------------------------------------- helpers
__device__ __forceinline__ float At(const float* img, int w, int h, int x, int y)
{
    x = min(max(x, 0), w - 1);
    y = min(max(y, 0), h - 1);
    return img[y * w + x];
}

__device__ __forceinline__ int Reflect101(int i, int n)
{
    if (n == 1)
        return 0;
    if (i < 0)
        i = -i;
    if (i >= n)
        i = 2 * n - 2 - i;
    return i;
}

__device__ __forceinline__ float Round255(float v01) { return floorf(fminf(fmaxf(v01, 0.0f), 1.0f) * 255.0f + 0.5f); }

__device__ __forceinline__ float WarpSum(float v)
{
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Bilinear with replicated borders (remap BORDER_REPLICATE).
__device__ __forceinline__ float BilinearClamp(const float* img, int w, int h, float x, float y)
{
    x = fminf(fmaxf(x, 0.0f), (float)(w - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(h - 1));
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = x - x0, fy = y - y0;
    return (1 - fy) * ((1 - fx) * img[y0 * w + x0] + fx * img[y0 * w + x1]) +
           fy * ((1 - fx) * img[y1 * w + x0] + fx * img[y1 * w + x1]);
}

// ------------------------------------------------- analysis-size images
// Area average of the float BGRA frame over each analysis pixel's footprint,
// rounded to 8 bit per channel as cv::resize(INTER_AREA) on 8-bit frames
// gives; then luma as GrayOf (77/150/29 of 256) and the colour copy.
__global__ void KAreaFrame(const char* frame, size_t pitch, int W, int H, int fw, int fh, float* gray,
                           uchar4* colour)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fw || y >= fh)
        return;
    const float sx = (float)W / fw, sy = (float)H / fh;
    const float x0 = x * sx, x1 = fminf((x + 1) * sx, (float)W), y0 = y * sy, y1 = fminf((y + 1) * sy, (float)H);
    float b = 0, g = 0, r = 0, wsum = 0;
    for (int iy = (int)y0; iy < (int)ceilf(y1); ++iy)
    {
        const float wy = fminf(iy + 1.0f, y1) - fmaxf((float)iy, y0);
        const float4* row = reinterpret_cast<const float4*>(frame + (size_t)iy * pitch);
        for (int ix = (int)x0; ix < (int)ceilf(x1); ++ix)
        {
            const float wgt = wy * (fminf(ix + 1.0f, x1) - fmaxf((float)ix, x0));
            const float4 p = row[ix]; // x=B y=G z=R
            b += wgt * fminf(fmaxf(p.x, 0.0f), 1.0f);
            g += wgt * fminf(fmaxf(p.y, 0.0f), 1.0f);
            r += wgt * fminf(fmaxf(p.z, 0.0f), 1.0f);
            wsum += wgt;
        }
    }
    const float inv = 1.0f / fmaxf(wsum, 1e-12f);
    const int B8 = (int)Round255(b * inv), G8 = (int)Round255(g * inv), R8 = (int)Round255(r * inv);
    gray[y * fw + x] = (float)((77 * R8 + 150 * G8 + 29 * B8 + 128) >> 8);
    colour[y * fw + x] = make_uchar4((unsigned char)B8, (unsigned char)G8, (unsigned char)R8, 0);
}

// cv::blur 3x3 (BORDER_REFLECT_101), rounded to 8 bit.
__global__ void KBox3(const uchar4* src, int w, int h, uchar4* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    int b = 0, g = 0, r = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const uchar4 p = src[Reflect101(y + dy, h) * w + Reflect101(x + dx, w)];
            b += p.x;
            g += p.y;
            r += p.z;
        }
    dst[y * w + x] = make_uchar4((unsigned char)((b * 2 + 9) / 18), (unsigned char)((g * 2 + 9) / 18),
                                 (unsigned char)((r * 2 + 9) / 18), 0);
}

// Area downscale of an 8-bit luma image (values 0..255 in float), rounded.
__global__ void KAreaGray(const float* src, int sw, int sh, float* dst, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;
    const float sx = (float)sw / dw, sy = (float)sh / dh;
    const float x0 = x * sx, x1 = fminf((x + 1) * sx, (float)sw), y0 = y * sy, y1 = fminf((y + 1) * sy, (float)sh);
    float acc = 0, wsum = 0;
    for (int iy = (int)y0; iy < (int)ceilf(y1); ++iy)
    {
        const float wy = fminf(iy + 1.0f, y1) - fmaxf((float)iy, y0);
        for (int ix = (int)x0; ix < (int)ceilf(x1); ++ix)
        {
            const float wgt = wy * (fminf(ix + 1.0f, x1) - fmaxf((float)ix, x0));
            acc += wgt * src[iy * sw + ix];
            wsum += wgt;
        }
    }
    dst[y * dw + x] = floorf(acc / fmaxf(wsum, 1e-12f) + 0.5f);
}

// cv::spatialGradient: 3x3 Sobel, BORDER_REFLECT_101 (unnormalised, as DIS uses it).
__global__ void KSobel(const float* I, int w, int h, float* gx, float* gy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const int xm = Reflect101(x - 1, w), xp = Reflect101(x + 1, w), ym = Reflect101(y - 1, h), yp = Reflect101(y + 1, h);
    const float a = I[ym * w + xm], b = I[ym * w + x], c = I[ym * w + xp];
    const float d = I[y * w + xm], f = I[y * w + xp];
    const float g = I[yp * w + xm], hh = I[yp * w + x], k = I[yp * w + xp];
    gx[y * w + x] = (c - a) + 2.0f * (f - d) + (k - g);
    gy[y * w + x] = (g - a) + 2.0f * (hh - b) + (k - c);
}

// ------------------------------------------------- patch inverse search
// Sums over each patch of Ix^2, Iy^2, IxIy, Ix, Iy.
__global__ void KStructure(const float* gx, const float* gy, int w, int ws, int hs, int stride, float* xx, float* yy,
                           float* xy, float* sx, float* sy)
{
    const int js = blockIdx.x * blockDim.x + threadIdx.x, is = blockIdx.y * blockDim.y + threadIdx.y;
    if (js >= ws || is >= hs)
        return;
    const int i = is * stride, j = js * stride;
    float a = 0, b = 0, c = 0, d = 0, e = 0;
    for (int r = 0; r < kPsz; ++r)
        for (int q = 0; q < kPsz; ++q)
        {
            const float X = gx[(i + r) * w + j + q], Y = gy[(i + r) * w + j + q];
            a += X * X;
            b += Y * Y;
            c += X * Y;
            d += X;
            e += Y;
        }
    const int k = is * ws + js;
    xx[k] = a;
    yy[k] = b;
    xy[k] = c;
    sx[k] = d;
    sy[k] = e;
}

struct Bilin
{
    int iy, ix;
    float w00, w01, w10, w11;
};

// Where patch (i, j) of I0 lands in I1 for motion U (OpenCV clamps the patch
// to at most psz-1 pixels outside I1; samples there are replicated borders).
__device__ __forceinline__ Bilin PatchPos(int i, int j, float2 U, int w, int h)
{
    const float y = fminf(fmaxf(i + U.y, -(kPsz - 1.0f)), h - 1.0f);
    const float x = fminf(fmaxf(j + U.x, -(kPsz - 1.0f)), w - 1.0f);
    const float y0 = floorf(y), x0 = floorf(x);
    const float di = y - y0, dj = x - x0;
    return {(int)y0, (int)x0, (1 - di) * (1 - dj), (1 - di) * dj, di * (1 - dj), di * dj};
}

__device__ __forceinline__ float PatchDiff(const float* I0, const float* I1, int w, int h, int i, int j, const Bilin& p,
                                           int r, int c)
{
    const float v = p.w00 * At(I1, w, h, p.ix + c, p.iy + r) + p.w01 * At(I1, w, h, p.ix + c + 1, p.iy + r) +
                    p.w10 * At(I1, w, h, p.ix + c, p.iy + r + 1) + p.w11 * At(I1, w, h, p.ix + c + 1, p.iy + r + 1);
    return v - I0[(i + r) * w + j + c];
}

// Mean-normalised SSD of a patch, computed by a whole warp (2 pixels per lane).
__device__ float WarpPatchSSD(const float* I0, const float* I1, int w, int h, int i, int j, float2 U, int lane)
{
    const Bilin p = PatchPos(i, j, U, w, h);
    float sd = 0, sq = 0;
    for (int k = lane; k < kPsz * kPsz; k += 32)
    {
        const float v = PatchDiff(I0, I1, w, h, i, j, p, k / kPsz, k % kPsz);
        sd += v;
        sq += v * v;
    }
    sd = WarpSum(sd);
    sq = WarpSum(sq);
    return sq - sd * sd * (1.0f / (kPsz * kPsz));
}

// Patch inverse search exactly as the CPU DIS orders it: 8 fixed stripes of
// patch rows (one block each), each swept top-left to bottom-right and back;
// every patch picks the best of its current motion and its already-finished
// left/upper (right/lower on the way back) neighbours' and refines it at once
// by Gauss-Newton. A patch only waits for those two neighbours, so the
// patches on each anti-diagonal of a stripe run in parallel (one warp each).
constexpr int kStripes = 8;

__device__ void WarpPatchGN(const float* I0, const float* I1, const float* gx, const float* gy, int w, int h, int i,
                            int j, float i11, float i12, float i22, float gsx, float gsy, int iters, float2& u, int lane)
{
    const float invN = 1.0f / (kPsz * kPsz);
    float prevSSD = 1e10f;
    for (int t = 0; t < iters; ++t)
    {
        const Bilin p = PatchPos(i, j, u, w, h);
        float sd = 0, sq = 0, mx = 0, my = 0;
        for (int k = lane; k < kPsz * kPsz; k += 32)
        {
            const int r = k / kPsz, c = k % kPsz;
            const float v = PatchDiff(I0, I1, w, h, i, j, p, r, c);
            sd += v;
            sq += v * v;
            mx += v * gx[(i + r) * w + j + c];
            my += v * gy[(i + r) * w + j + c];
        }
        sd = WarpSum(sd);
        sq = WarpSum(sq);
        mx = WarpSum(mx);
        my = WarpSum(my);
        const float dUx = mx - sd * gsx * invN, dUy = my - sd * gsy * invN, ssd = sq - sd * sd * invN;
        u.x -= i11 * dUx + i12 * dUy;
        u.y -= i12 * dUx + i22 * dUy;
        if (ssd >= prevSSD)
            break;
        prevSSD = ssd;
    }
}

__global__ void KPatchSearch(const float2* U, const float* I0, const float* I1, const float* gx, const float* gy,
                             const float* xx, const float* yy, const float* xy, const float* sx, const float* sy, int w,
                             int h, int ws, int hs, int stride, int iters, float2* S)
{
    const int stripeRows = (hs + kStripes - 1) / kStripes;
    const int rs = min(blockIdx.x * stripeRows, hs), re = min(rs + stripeRows, hs);
    const int rows = re - rs;
    if (rows <= 0)
        return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nWarps = blockDim.x / 32;
    const int c = kPsz / 2;
    for (int pass = 0; pass < 2; ++pass)
    {
        const int dir = pass == 0 ? 1 : -1;
        for (int d = 0; d < rows + ws - 1; ++d)
        {
            for (int r = warp; r < rows; r += nWarps)
            {
                const int q = d - r; // position along the row in sweep order
                if (q < 0 || q >= ws)
                    continue;
                const int is = dir > 0 ? rs + r : re - 1 - r;
                const int js = dir > 0 ? q : ws - 1 - q;
                const int i = is * stride, j = js * stride, k = is * ws + js;
                float2 best = pass == 0 ? U[(i + c) * w + j + c] : S[k];
                float minSSD = WarpPatchSSD(I0, I1, w, h, i, j, best, lane);
                if (q > 0) // previous patch in this row
                {
                    const float2 cand = S[k - dir];
                    const float ssd = WarpPatchSSD(I0, I1, w, h, i, j, cand, lane);
                    if (ssd < minSSD)
                    {
                        minSSD = ssd;
                        best = cand;
                    }
                }
                if (r > 0) // previous row of the stripe
                {
                    const float2 cand = S[k - dir * ws];
                    const float ssd = WarpPatchSSD(I0, I1, w, h, i, j, cand, lane);
                    if (ssd < minSSD)
                        best = cand;
                }
                const float hxx = xx[k], hyy = yy[k], hxy = xy[k];
                float det = hxx * hyy - hxy * hxy;
                if (fabsf(det) < 0.001f)
                    det = 0.001f;
                float2 u = best;
                WarpPatchGN(I0, I1, gx, gy, w, h, i, j, hyy / det, -hxy / det, hxx / det, sx[k], sy[k], iters, u, lane);
                const float dx = u.x - best.x, dy = u.y - best.y;
                if (lane == 0)
                    S[k] = dx * dx + dy * dy <= (float)(kPsz * kPsz) ? u : best;
            }
            __syncthreads();
        }
    }
}

// Dense motion: the patches covering each pixel, weighted by 1 / max(1, |error|).
__global__ void KDensify(const float2* S, const float* I0, const float* I1, int w, int h, int ws, int stride, float2* U)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const int endI = min(y / stride, (h - kPsz) / stride);
    const int startI = min(max(0, y - kPsz + stride) / stride, endI);
    const int endJ = min(x / stride, (w - kPsz) / stride);
    const int startJ = min(max(0, x - kPsz + stride) / stride, endJ);
    float su = 0, sv = 0, sc = 0;
    const float i0 = I0[y * w + x];
    for (int is = startI; is <= endI; ++is)
        for (int js = startJ; js <= endJ; ++js)
        {
            const float2 s = S[is * ws + js];
            const float jm = fminf(fmaxf(x + s.x, 0.0f), w - 1.0f - 0.001f);
            const float im = fminf(fmaxf(y + s.y, 0.0f), h - 1.0f - 0.001f);
            const int jl = (int)jm, il = (int)im, ju = jl + 1, iu = il + 1;
            const float diff = (jm - jl) * (im - il) * I1[iu * w + ju] + (ju - jm) * (im - il) * I1[iu * w + jl] +
                               (jm - jl) * (iu - im) * I1[il * w + ju] + (ju - jm) * (iu - im) * I1[il * w + jl] - i0;
            const float coef = 1.0f / fmaxf(1.0f, fabsf(diff));
            su += coef * s.x;
            sv += coef * s.y;
            sc += coef;
        }
    U[y * w + x] = make_float2(su / sc, sv / sc);
}

// ------------------------------------------------- variational refinement
struct VarBufs
{
    float *Ix, *Iy, *Iz, *Ixx, *Ixy, *Iyy, *Ixz, *Iyz; // derivatives (linearised around the initial motion W)
    float *avg;                                     // (I0 + warped I1) / 2, scratch
    float *a11, *a12, *a22, *b1, *b2;               // linear system
    float *sw;                                      // smoothness weight of each pixel's right / down edges
    float2 *W, *dW, *tW;                            // initial motion, increment, current (W + dW)
};

// Warp I1 by W; average with I0 and temporal difference.
__global__ void KVarWarp(const float* I0, const float* I1, const float2* W, int w, int h, float* avg, float* Iz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float2 u = W[y * w + x];
    const float wi = BilinearClamp(I1, w, h, x + u.x, y + u.y);
    const float i0 = I0[y * w + x];
    avg[y * w + x] = 0.5f * (i0 + wi);
    Iz[y * w + x] = wi - i0;
}

// Central differences [-1 0 1] (Sobel ksize 1), replicated borders.
__global__ void KVarDeriv1(const float* avg, const float* Iz, int w, int h, float* Ix, float* Iy, float* Ixz, float* Iyz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1), ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    const int k = y * w + x;
    Ix[k] = avg[y * w + xp] - avg[y * w + xm];
    Iy[k] = avg[yp * w + x] - avg[ym * w + x];
    Ixz[k] = Iz[y * w + xp] - Iz[y * w + xm];
    Iyz[k] = Iz[yp * w + x] - Iz[ym * w + x];
}

__global__ void KVarDeriv2(const float* Ix, const float* Iy, int w, int h, float* Ixx, float* Ixy, float* Iyy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1), ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    const int k = y * w + x;
    Ixx[k] = Ix[y * w + xp] - Ix[y * w + xm];
    Ixy[k] = Ix[yp * w + x] - Ix[ym * w + x];
    Iyy[k] = Iy[yp * w + x] - Iy[ym * w + x];
}

// Smoothness weight from the current motion (forward differences; 0 across the far borders).
__global__ void KVarSmoothWeight(const float2* tW, int w, int h, float* sw)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float2 c = tW[y * w + x];
    const float2 r = x + 1 < w ? tW[y * w + x + 1] : c;
    const float2 d = y + 1 < h ? tW[(y + 1) * w + x] : c;
    const float ux = r.x - c.x, vx = r.y - c.y, uy = d.x - c.x, vy = d.y - c.y;
    sw[y * w + x] = (kVrAlpha / 2) / sqrtf(ux * ux + vx * vx + uy * uy + vy * vy + kVrEps * kVrEps);
}

// Linear system of the current fixed-point iteration: data terms (colour and
// gradient constancy, robust weights at the current increment) + smoothness.
__global__ void KVarSystem(VarBufs v, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const int k = y * w + x;
    const float zeta2 = kVrZeta * kVrZeta, eps2 = kVrEps * kVrEps;
    const float Ix = v.Ix[k], Iy = v.Iy[k], Iz = v.Iz[k];
    const float Ixx = v.Ixx[k], Ixy = v.Ixy[k], Iyy = v.Iyy[k], Ixz = v.Ixz[k], Iyz = v.Iyz[k];
    const float2 d = v.dW[k];
    // colour constancy
    float norm = Ix * Ix + Iy * Iy + zeta2;
    const float e = Iz + Ix * d.x + Iy * d.y;
    float wt = ((kVrDelta / 2) / sqrtf(e * e / norm + eps2)) / norm;
    float a11 = wt * Ix * Ix + zeta2, a12 = wt * Ix * Iy, a22 = wt * Iy * Iy + zeta2;
    float b1 = -wt * Iz * Ix, b2 = -wt * Iz * Iy;
    // gradient constancy
    norm = Ixx * Ixx + Ixy * Ixy + zeta2;
    const float norm2 = Iyy * Iyy + Ixy * Ixy + zeta2;
    const float ex = Ixz + Ixx * d.x + Ixy * d.y, ey = Iyz + Ixy * d.x + Iyy * d.y;
    wt = (kVrGamma / 2) / sqrtf(ex * ex / norm + ey * ey / norm2 + eps2);
    a11 += wt * (Ixx * Ixx / norm + Ixy * Ixy / norm2);
    a12 += wt * (Ixx * Ixy / norm + Ixy * Iyy / norm2);
    a22 += wt * (Ixy * Ixy / norm + Iyy * Iyy / norm2);
    b1 -= wt * (Ixx * Ixz / norm + Ixy * Iyz / norm2);
    b2 -= wt * (Ixy * Ixz / norm + Iyy * Iyz / norm2);
    // smoothness: each edge to a neighbour inside the frame
    const float2 W0 = v.W[k];
    float sA = 0, su = 0, sv = 0;
    if (x + 1 < w)
    {
        const float s = v.sw[k];
        const float2 n = v.W[k + 1];
        sA += s;
        su += s * (n.x - W0.x);
        sv += s * (n.y - W0.y);
    }
    if (x > 0)
    {
        const float s = v.sw[k - 1];
        const float2 n = v.W[k - 1];
        sA += s;
        su -= s * (W0.x - n.x);
        sv -= s * (W0.y - n.y);
    }
    if (y + 1 < h)
    {
        const float s = v.sw[k];
        const float2 n = v.W[k + w];
        sA += s;
        su += s * (n.x - W0.x);
        sv += s * (n.y - W0.y);
    }
    if (y > 0)
    {
        const float s = v.sw[k - w];
        const float2 n = v.W[k - w];
        sA += s;
        su -= s * (W0.x - n.x);
        sv -= s * (W0.y - n.y);
    }
    v.a11[k] = a11 + sA;
    v.a12[k] = a12;
    v.a22[k] = a22 + sA;
    v.b1[k] = b1 + su;
    v.b2[k] = b2 + sv;
}

// One colour of a red-black SOR sweep (red: x + y even).
__global__ void KVarSor(VarBufs v, int w, int h, int colour)
{
    const int xh = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = 2 * xh + ((y + colour) & 1);
    if (x >= w || y >= h)
        return;
    const int k = y * w + x;
    float su = 0, sv = 0;
    if (x > 0)
    {
        const float s = v.sw[k - 1];
        su += s * v.dW[k - 1].x;
        sv += s * v.dW[k - 1].y;
    }
    if (x + 1 < w)
    {
        const float s = v.sw[k];
        su += s * v.dW[k + 1].x;
        sv += s * v.dW[k + 1].y;
    }
    if (y > 0)
    {
        const float s = v.sw[k - w];
        su += s * v.dW[k - w].x;
        sv += s * v.dW[k - w].y;
    }
    if (y + 1 < h)
    {
        const float s = v.sw[k];
        su += s * v.dW[k + w].x;
        sv += s * v.dW[k + w].y;
    }
    float2 d = v.dW[k];
    d.x += kVrOmega * ((su + v.b1[k] - d.y * v.a12[k]) / v.a11[k] - d.x);
    d.y += kVrOmega * ((sv + v.b2[k] - d.x * v.a12[k]) / v.a22[k] - d.y);
    v.dW[k] = d;
}

__global__ void KVarUpdate(const float2* W, const float2* dW, int n, float2* tW)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    tW[k] = make_float2(W[k].x + dW[k].x, W[k].y + dW[k].y);
}

// ------------------------------------------------- resizing the motion
// cv::resize INTER_LINEAR of a motion field, values times `mul`.
__global__ void KResizeFlow(const float2* src, int sw, int sh, float2* dst, int dw, int dh, float mul)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;
    float fx = (x + 0.5f) * ((float)sw / dw) - 0.5f, fy = (y + 0.5f) * ((float)sh / dh) - 0.5f;
    fx = fminf(fmaxf(fx, 0.0f), (float)(sw - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(sh - 1));
    const int x0 = (int)fx, y0 = (int)fy, x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
    const float tx = fx - x0, ty = fy - y0;
    const float2 a = src[y0 * sw + x0], b = src[y0 * sw + x1], c = src[y1 * sw + x0], d = src[y1 * sw + x1];
    dst[y * dw + x] = make_float2(mul * ((1 - ty) * ((1 - tx) * a.x + tx * b.x) + ty * ((1 - tx) * c.x + tx * d.x)),
                                  mul * ((1 - ty) * ((1 - tx) * a.y + tx * b.y) + ty * ((1 - tx) * c.y + tx * d.y)));
}

// ------------------------------------------------- motion-boundary clean-up
// (RefineFlowEdges in MotionBlur.cpp.) Where the motion changes fast: on a
// grid of every 4th pixel, morphological gradient of each component.
__global__ void KEdgeFlagQ(const float2* f, int w, int h, int qw, int qh, unsigned char* flag)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= qw || y >= qh)
        return;
    float mnx = 1e30f, mxx = -1e30f, mny = 1e30f, mxy = -1e30f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int qx = x + dx, qy = y + dy;
            if (qx < 0 || qy < 0 || qx >= qw || qy >= qh)
                continue;
            // INTER_NEAREST from the full grid
            const int sx = min((int)floorf(qx * ((float)w / qw)), w - 1), sy = min((int)floorf(qy * ((float)h / qh)), h - 1);
            const float2 v = f[sy * w + sx];
            mnx = fminf(mnx, v.x);
            mxx = fmaxf(mxx, v.x);
            mny = fminf(mny, v.y);
            mxy = fmaxf(mxy, v.y);
        }
    flag[y * qw + x] = (mxx - mnx) + (mxy - mny) > 3.0f ? 1 : 0;
}

// Grown by the candidate reach (square, radius r on the quarter grid).
__global__ void KDilateQ(const unsigned char* src, int qw, int qh, int r, unsigned char* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= qw || y >= qh)
        return;
    unsigned char m = 0;
    for (int dy = -r; dy <= r && !m; ++dy)
    {
        const int qy = y + dy;
        if (qy < 0 || qy >= qh)
            continue;
        for (int dx = -r; dx <= r; ++dx)
        {
            const int qx = x + dx;
            if (qx >= 0 && qx < qw && src[qy * qw + qx])
            {
                m = 1;
                break;
            }
        }
    }
    dst[y * qw + x] = m;
}

// Near a motion boundary each pixel takes the weighted median motion of 40
// candidates (5 radii x 8 directions) and itself, weighted by colour likeness.
__global__ void KEdgeRefine(const float2* src, const uchar4* col, const unsigned char* maskQ, int w, int h, int qw,
                            int qh, float2* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float2 own = src[y * w + x];
    const int mx = min((int)floorf(x * ((float)qw / w)), qw - 1), my = min((int)floorf(y * ((float)qh / h)), qh - 1);
    if (!maskQ[my * qw + mx])
    {
        dst[y * w + x] = own;
        return;
    }
    constexpr int kN = 41;
    const int radii[5] = {3, 8, 16, 28, 42};
    const int dirs[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    float vx[kN], vy[kN], wt[kN];
    const uchar4 c0 = col[y * w + x];
    int n = 0;
    float spread = 0;
    vx[0] = own.x;
    vy[0] = own.y;
    wt[0] = 1.0f;
    n = 1;
    for (int ri = 0; ri < 5; ++ri)
        for (int di = 0; di < 8; ++di)
        {
            const int cx = x + dirs[di][0] * radii[ri], cy = y + dirs[di][1] * radii[ri];
            if (cx < 0 || cy < 0 || cx >= w || cy >= h)
                continue;
            const float2 v = src[cy * w + cx];
            const uchar4 c = col[cy * w + cx];
            spread = fmaxf(spread, fabsf(v.x - own.x) + fabsf(v.y - own.y));
            const float dcol = (float)(abs((int)c.x - (int)c0.x) + abs((int)c.y - (int)c0.y) + abs((int)c.z - (int)c0.z));
            vx[n] = v.x;
            vy[n] = v.y;
            wt[n] = expf(-(dcol / 36.0f) * (dcol / 36.0f));
            ++n;
        }
    if (spread < 1.0f)
    {
        dst[y * w + x] = own;
        return;
    }
    float total = 0;
    for (int i = 0; i < n; ++i)
        total += wt[i];
    const float half = 0.5f * total;
    // weighted median: the smallest value whose cumulative weight (values <= it) reaches half
    auto median = [&](const float* v) {
        float best = 3.4e38f;
        for (int i = 0; i < n; ++i)
        {
            if (v[i] >= best)
                continue;
            float acc = 0;
            for (int j = 0; j < n; ++j)
                if (v[j] <= v[i])
                    acc += wt[j];
            if (acc >= half)
                best = v[i];
        }
        return best;
    };
    dst[y * w + x] = make_float2(median(vx), median(vy));
}

// Final motion in full-res pixels into the engine's flow buffer (pitched),
// `mul` = sign / scale; accumulate adds to what is there (single-frame path).
__global__ void KStoreFlow(const float2* f, int w, int h, float mul, int accumulate, char* dst, size_t dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float2* d = reinterpret_cast<float2*>(dst + (size_t)y * dstPitch) + x;
    const float2 v = f[y * w + x];
    if (accumulate)
        *d = make_float2(d->x + mul * v.x, d->y + mul * v.y);
    else
        *d = make_float2(mul * v.x, mul * v.y);
}

// ---------------------------------------------------------------- host side
template <typename T> struct Buf
{
    T* p = nullptr;
    size_t n = 0;
    bool Ensure(size_t count)
    {
        if (p && n >= count)
            return true;
        Release();
        if (cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)) != cudaSuccess)
        {
            p = nullptr;
            return false;
        }
        n = count;
        return true;
    }
    void Release()
    {
        if (p)
            cudaFree(p);
        p = nullptr;
        n = 0;
    }
};

class DisFlow
{
public:
    static constexpr int kSlots = 3;

    // Call with the engine's context current.
    void Release()
    {
        for (Slot& s : slot_)
        {
            for (auto& b : s.gray)
                b.Release();
            for (auto& b : s.gx)
                b.Release();
            for (auto& b : s.gy)
                b.Release();
            s.colourRaw.Release();
            s.colour.Release();
        }
        for (auto* b : {&xx_, &yy_, &xy_, &sx_, &sy_, &vIx_, &vIy_, &vIz_, &vIxx_, &vIxy_, &vIyy_, &vIxz_, &vIyz_,
                        &vAvg_, &a11_, &a12_, &a22_, &b1_, &b2_, &sw_})
            b->Release();
        for (auto* b : {&S_, &U_[0], &U_[1], &W_, &dW_, &tW_, &f0_, &f1_})
            b->Release();
        flagQ_[0].Release();
        flagQ_[1].Release();
    }

    // Analysis size and pyramid for a frame of W x H (returns false when the
    // frame is too small for DIS: the caller then uses the CPU path).
    bool Setup(int W, int H, int flowWidth, int preset)
    {
        p_ = Preset(preset);
        const double scale = std::min(1.0, (double)flowWidth / W);
        scale_ = scale;
        fw_ = std::max(1, (int)std::lround(W * scale));
        fh_ = std::max(1, (int)std::lround(H * scale));
        const int coarsest = std::min((int)(std::log(std::max(fw_, fh_) / (4.0 * kPsz)) / std::log(2.0) + 0.5),
                                      (int)(std::log(std::min(fw_, fh_) / (double)kPsz) / std::log(2.0)));
        if (coarsest < p_.finest || coarsest >= kMaxLevels)
            return false;
        coarsest_ = coarsest;
        lw_[0] = fw_;
        lh_[0] = fh_;
        // the finest level is resized straight from the analysis image; above it, halved
        lw_[p_.finest] = fw_ >> p_.finest;
        lh_[p_.finest] = fh_ >> p_.finest;
        for (int i = p_.finest + 1; i <= coarsest_; ++i)
        {
            lw_[i] = lw_[i - 1] / 2;
            lh_[i] = lh_[i - 1] / 2;
        }
        for (int i = p_.finest; i <= coarsest_; ++i)
            if (lw_[i] < kPsz || lh_[i] < kPsz)
                return false;
        return true;
    }

    int Fw() const { return fw_; }
    int Fh() const { return fh_; }
    double Scale() const { return scale_; }

    // Frame (float BGRA on the GPU) -> slot: analysis image, colour, pyramid, gradients.
    bool Prepare(int slot, const char* frame, size_t pitch, int W, int H, std::string* err)
    {
        Slot& s = slot_[slot];
        const size_t n0 = (size_t)fw_ * fh_;
        if (!s.gray[0].Ensure(n0) || !s.colourRaw.Ensure(n0) || !s.colour.Ensure(n0))
            return Fail(err, "out of GPU memory (motion)");
        const dim3 b(16, 16);
        KAreaFrame<<<Grid2(fw_, fh_, b), b>>>(frame, pitch, W, H, fw_, fh_, s.gray[0].p, s.colourRaw.p);
        KBox3<<<Grid2(fw_, fh_, b), b>>>(s.colourRaw.p, fw_, fh_, s.colour.p);
        for (int i = p_.finest; i <= coarsest_; ++i)
        {
            const size_t n = (size_t)lw_[i] * lh_[i];
            if (i > 0 && !s.gray[i].Ensure(n))
                return Fail(err, "out of GPU memory (motion)");
            if (!s.gx[i].Ensure(n) || !s.gy[i].Ensure(n))
                return Fail(err, "out of GPU memory (motion)");
            if (i > 0)
            {
                const int src = i == p_.finest ? 0 : i - 1;
                KAreaGray<<<Grid2(lw_[i], lh_[i], b), b>>>(s.gray[src].p, lw_[src], lh_[src], s.gray[i].p, lw_[i], lh_[i]);
            }
            KSobel<<<Grid2(lw_[i], lh_[i], b), b>>>(s.gray[i].p, lw_[i], lh_[i], s.gx[i].p, s.gy[i].p);
        }
        return true;
    }

    // Motion from slot `from` to slot `to` (from(x) ~ to(x + f)), cleaned up
    // at motion boundaries if asked, stored into dst as full-res pixels times
    // sign (accumulate: added to dst, times `weight`).
    bool Compute(int from, int to, bool refineEdges, float weight, bool accumulate, char* dst, size_t dstPitch,
                 std::string* err)
    {
        const Slot& A = slot_[from];
        const Slot& B = slot_[to];
        const int fin = p_.finest;
        const size_t nf = (size_t)lw_[fin] * lh_[fin];
        const size_t np = (size_t)(1 + (lw_[fin] - kPsz) / p_.stride) * (1 + (lh_[fin] - kPsz) / p_.stride);
        bool ok = S_.Ensure(np) && U_[0].Ensure(nf) && U_[1].Ensure(nf);
        for (auto* q : {&xx_, &yy_, &xy_, &sx_, &sy_})
            ok = ok && q->Ensure(np);
        if (p_.varIter > 0)
        {
            for (auto* q : {&vIx_, &vIy_, &vIz_, &vIxx_, &vIxy_, &vIyy_, &vIxz_, &vIyz_, &vAvg_, &a11_, &a12_, &a22_,
                            &b1_, &b2_, &sw_})
                ok = ok && q->Ensure(nf);
            ok = ok && W_.Ensure(nf) && dW_.Ensure(nf) && tW_.Ensure(nf);
        }
        const size_t n0 = (size_t)fw_ * fh_;
        ok = ok && f0_.Ensure(n0) && f1_.Ensure(n0);
        if (!ok)
            return Fail(err, "out of GPU memory (motion)");

        const dim3 b(16, 16);
        const int iters = (int)std::floor(p_.gdIter / 2.0f);
        for (int i = coarsest_; i >= fin; --i)
        {
            const int w = lw_[i], h = lh_[i];
            const int ws = 1 + (w - kPsz) / p_.stride, hs = 1 + (h - kPsz) / p_.stride;
            float2* U = U_[i & 1].p;
            if (i == coarsest_)
                cudaMemsetAsync(U, 0, (size_t)w * h * sizeof(float2));
            const float* I0 = A.gray[i].p;
            const float* I1 = B.gray[i].p;
            KStructure<<<Grid2(ws, hs, b), b>>>(A.gx[i].p, A.gy[i].p, w, ws, hs, p_.stride, xx_.p, yy_.p, xy_.p, sx_.p,
                                                 sy_.p);
            const int stripeRows = (hs + kStripes - 1) / kStripes;
            KPatchSearch<<<kStripes, 32 * std::min(stripeRows, 32)>>>(U, I0, I1, A.gx[i].p, A.gy[i].p, xx_.p, yy_.p,
                                                                      xy_.p, sx_.p, sy_.p, w, h, ws, hs, p_.stride,
                                                                      iters, S_.p);
            KDensify<<<Grid2(w, h, b), b>>>(S_.p, I0, I1, w, h, ws, p_.stride, U);
            if (p_.varIter > 0)
                Refine(I0, I1, U, w, h);
            if (i > fin)
                KResizeFlow<<<Grid2(lw_[i - 1], lh_[i - 1], b), b>>>(U, w, h, U_[(i - 1) & 1].p, lw_[i - 1], lh_[i - 1],
                                                                     2.0f);
        }
        // to the analysis size, in analysis pixels
        KResizeFlow<<<Grid2(fw_, fh_, b), b>>>(U_[fin & 1].p, lw_[fin], lh_[fin], f0_.p, fw_, fh_, (float)(1 << fin));
        const float2* result = f0_.p;
        if (refineEdges)
        {
            constexpr int kReach = 42;
            const int qw = (fw_ + 3) / 4, qh = (fh_ + 3) / 4;
            if (!flagQ_[0].Ensure((size_t)qw * qh) || !flagQ_[1].Ensure((size_t)qw * qh))
                return Fail(err, "out of GPU memory (motion)");
            KEdgeFlagQ<<<Grid2(qw, qh, b), b>>>(f0_.p, fw_, fh_, qw, qh, flagQ_[0].p);
            KDilateQ<<<Grid2(qw, qh, b), b>>>(flagQ_[0].p, qw, qh, kReach / 4 + 1, flagQ_[1].p);
            KEdgeRefine<<<Grid2(fw_, fh_, b), b>>>(f0_.p, A.colour.p, flagQ_[1].p, fw_, fh_, qw, qh, f1_.p);
            result = f1_.p;
        }
        KStoreFlow<<<Grid2(fw_, fh_, b), b>>>(result, fw_, fh_, (float)(weight / scale_), accumulate ? 1 : 0, dst,
                                              dstPitch);
        return true;
    }

private:
    struct Slot
    {
        Buf<float> gray[kMaxLevels], gx[kMaxLevels], gy[kMaxLevels];
        Buf<uchar4> colourRaw, colour;
    };

    static dim3 Grid2(int w, int h, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y); }

    static bool Fail(std::string* err, const char* msg)
    {
        if (err)
            *err = msg;
        return false;
    }

    VarBufs Vars()
    {
        return {vIx_.p, vIy_.p, vIz_.p, vIxx_.p, vIxy_.p, vIyy_.p, vIxz_.p, vIyz_.p, vAvg_.p, a11_.p, a12_.p,
                a22_.p, b1_.p,  b2_.p,  sw_.p,   W_.p,    dW_.p,   tW_.p};
    }

    // Variational refinement of U (in place) at one level.
    void Refine(const float* I0, const float* I1, float2* U, int w, int h)
    {
        const dim3 b(16, 16), g = Grid2(w, h, b), gh = Grid2((w + 1) / 2, h, b);
        const int n = w * h;
        VarBufs v = Vars();
        cudaMemcpyAsync(v.W, U, n * sizeof(float2), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(v.tW, U, n * sizeof(float2), cudaMemcpyDeviceToDevice);
        cudaMemsetAsync(v.dW, 0, n * sizeof(float2));
        KVarWarp<<<g, b>>>(I0, I1, v.W, w, h, v.avg, v.Iz);
        KVarDeriv1<<<g, b>>>(v.avg, v.Iz, w, h, v.Ix, v.Iy, v.Ixz, v.Iyz);
        KVarDeriv2<<<g, b>>>(v.Ix, v.Iy, w, h, v.Ixx, v.Ixy, v.Iyy);
        for (int it = 0; it < p_.varIter; ++it)
        {
            KVarSmoothWeight<<<g, b>>>(v.tW, w, h, v.sw);
            KVarSystem<<<g, b>>>(v, w, h);
            for (int s = 0; s < kVrSor; ++s)
            {
                KVarSor<<<gh, b>>>(v, w, h, 0);
                KVarSor<<<gh, b>>>(v, w, h, 1);
            }
            KVarUpdate<<<(n + 255) / 256, 256>>>(v.W, v.dW, n, v.tW);
        }
        cudaMemcpyAsync(U, v.tW, n * sizeof(float2), cudaMemcpyDeviceToDevice);
    }

    Params p_ = Preset(1);
    double scale_ = 1.0;
    int fw_ = 0, fh_ = 0, coarsest_ = 0;
    int lw_[kMaxLevels] = {}, lh_[kMaxLevels] = {};
    Slot slot_[kSlots];
    Buf<float> xx_, yy_, xy_, sx_, sy_;
    Buf<float> vIx_, vIy_, vIz_, vIxx_, vIxy_, vIyy_, vIxz_, vIyz_, vAvg_, a11_, a12_, a22_, b1_, b2_, sw_;
    Buf<float2> S_, U_[2], W_, dW_, tW_, f0_, f1_;
    Buf<unsigned char> flagQ_[2];
};

} // namespace oe::gpu::dis
