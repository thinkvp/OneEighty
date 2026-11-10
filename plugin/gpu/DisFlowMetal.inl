// GPU motion estimation for Quality Normal / High (Metal). Included by
// MetalMotionBlur.mm. A port of DisFlowCuda.cuh (see there for the
// algorithm): same kernels, same maths, same order of work; warp reductions
// become SIMD-group reductions (simd_sum).

namespace oe::gpu::dis
{

// One argument struct for every DIS kernel (each uses the fields it needs);
// plain ints / floats only so C++ and MSL lay it out the same.
struct DisArgs
{
    int W, H;          // full frame
    int fw, fh;        // analysis size
    int w, h;          // pyramid level
    int ws, hs, stride, iters;
    int sw, sh, dw, dh; // resize source / destination
    int colour;         // SOR: 0 red, 1 black
    int qw, qh, r;      // edge mask grid, dilation radius
    int acc;            // store: add to the destination
    float mul;          // resize / store factor
};

const char* kDisSource = R"MSL(
struct DisArgs { int W, H, fw, fh, w, h, ws, hs, stride, iters, sw, sh, dw, dh, colour, qw, qh, r, acc; float mul; };

constant int kPsz = 8;
constant int kStripes = 8;
constant float kVrAlpha = 20.0f, kVrGamma = 10.0f, kVrDelta = 5.0f, kVrEps = 0.01f, kVrZeta = 0.1f, kVrOmega = 1.6f;

inline float At(device const float* img, int w, int h, int x, int y)
{
    x = min(max(x, 0), w - 1);
    y = min(max(y, 0), h - 1);
    return img[y * w + x];
}

inline int Reflect101(int i, int n)
{
    if (n == 1)
        return 0;
    if (i < 0)
        i = -i;
    if (i >= n)
        i = 2 * n - 2 - i;
    return i;
}

inline float Round255(float v01) { return floor(min(max(v01, 0.0f), 1.0f) * 255.0f + 0.5f); }

inline float BilinearClamp(device const float* img, int w, int h, float x, float y)
{
    x = min(max(x, 0.0f), float(w - 1));
    y = min(max(y, 0.0f), float(h - 1));
    const int x0 = int(x), y0 = int(y);
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = x - x0, fy = y - y0;
    return (1 - fy) * ((1 - fx) * img[y0 * w + x0] + fx * img[y0 * w + x1]) +
           fy * ((1 - fx) * img[y1 * w + x0] + fx * img[y1 * w + x1]);
}

kernel void d_area_frame(device const float4* frame [[buffer(0)]], device float* gray [[buffer(1)]],
                         device uchar4* colour [[buffer(2)]], constant DisArgs& a [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= a.fw || y >= a.fh)
        return;
    const float sx = float(a.W) / a.fw, sy = float(a.H) / a.fh;
    const float x0 = x * sx, x1 = min((x + 1) * sx, float(a.W)), y0 = y * sy, y1 = min((y + 1) * sy, float(a.H));
    float b = 0, g = 0, r = 0, wsum = 0;
    for (int iy = int(y0); iy < int(ceil(y1)); ++iy)
    {
        const float wy = min(iy + 1.0f, y1) - max(float(iy), y0);
        for (int ix = int(x0); ix < int(ceil(x1)); ++ix)
        {
            const float wgt = wy * (min(ix + 1.0f, x1) - max(float(ix), x0));
            const float4 p = frame[iy * a.W + ix]; // x=B y=G z=R
            b += wgt * min(max(p.x, 0.0f), 1.0f);
            g += wgt * min(max(p.y, 0.0f), 1.0f);
            r += wgt * min(max(p.z, 0.0f), 1.0f);
            wsum += wgt;
        }
    }
    const float inv = 1.0f / max(wsum, 1e-12f);
    const int B8 = int(Round255(b * inv)), G8 = int(Round255(g * inv)), R8 = int(Round255(r * inv));
    gray[y * a.fw + x] = float((77 * R8 + 150 * G8 + 29 * B8 + 128) >> 8);
    colour[y * a.fw + x] = uchar4(uchar(B8), uchar(G8), uchar(R8), 0);
}

kernel void d_box3(device const uchar4* src [[buffer(0)]], device uchar4* dst [[buffer(1)]],
                   constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.fw, h = a.fh;
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
    dst[y * w + x] = uchar4(uchar((b * 2 + 9) / 18), uchar((g * 2 + 9) / 18), uchar((r * 2 + 9) / 18), 0);
}

kernel void d_area_gray(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                        constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= a.dw || y >= a.dh)
        return;
    const float sx = float(a.sw) / a.dw, sy = float(a.sh) / a.dh;
    const float x0 = x * sx, x1 = min((x + 1) * sx, float(a.sw)), y0 = y * sy, y1 = min((y + 1) * sy, float(a.sh));
    float acc = 0, wsum = 0;
    for (int iy = int(y0); iy < int(ceil(y1)); ++iy)
    {
        const float wy = min(iy + 1.0f, y1) - max(float(iy), y0);
        for (int ix = int(x0); ix < int(ceil(x1)); ++ix)
        {
            const float wgt = wy * (min(ix + 1.0f, x1) - max(float(ix), x0));
            acc += wgt * src[iy * a.sw + ix];
            wsum += wgt;
        }
    }
    dst[y * a.dw + x] = floor(acc / max(wsum, 1e-12f) + 0.5f);
}

kernel void d_sobel(device const float* I [[buffer(0)]], device float* gx [[buffer(1)]], device float* gy [[buffer(2)]],
                    constant DisArgs& a [[buffer(3)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h;
    if (x >= w || y >= h)
        return;
    const int xm = Reflect101(x - 1, w), xp = Reflect101(x + 1, w), ym = Reflect101(y - 1, h), yp = Reflect101(y + 1, h);
    const float p00 = I[ym * w + xm], p01 = I[ym * w + x], p02 = I[ym * w + xp];
    const float p10 = I[y * w + xm], p12 = I[y * w + xp];
    const float p20 = I[yp * w + xm], p21 = I[yp * w + x], p22 = I[yp * w + xp];
    gx[y * w + x] = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
    gy[y * w + x] = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);
}

kernel void d_structure(device const float* gx [[buffer(0)]], device const float* gy [[buffer(1)]],
                        device float* xx [[buffer(2)]], device float* yy [[buffer(3)]], device float* xy [[buffer(4)]],
                        device float* sx [[buffer(5)]], device float* sy [[buffer(6)]], constant DisArgs& a [[buffer(7)]],
                        uint2 gid [[thread_position_in_grid]])
{
    const int js = gid.x, is = gid.y;
    if (js >= a.ws || is >= a.hs)
        return;
    const int i = is * a.stride, j = js * a.stride, w = a.w;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    for (int r = 0; r < kPsz; ++r)
        for (int q = 0; q < kPsz; ++q)
        {
            const float X = gx[(i + r) * w + j + q], Y = gy[(i + r) * w + j + q];
            s0 += X * X;
            s1 += Y * Y;
            s2 += X * Y;
            s3 += X;
            s4 += Y;
        }
    const int k = is * a.ws + js;
    xx[k] = s0;
    yy[k] = s1;
    xy[k] = s2;
    sx[k] = s3;
    sy[k] = s4;
}

struct Bilin
{
    int iy, ix;
    float w00, w01, w10, w11;
};

inline Bilin PatchPos(int i, int j, float2 U, int w, int h)
{
    const float y = min(max(i + U.y, -(kPsz - 1.0f)), h - 1.0f);
    const float x = min(max(j + U.x, -(kPsz - 1.0f)), w - 1.0f);
    const float y0 = floor(y), x0 = floor(x);
    const float di = y - y0, dj = x - x0;
    Bilin p;
    p.iy = int(y0);
    p.ix = int(x0);
    p.w00 = (1 - di) * (1 - dj);
    p.w01 = (1 - di) * dj;
    p.w10 = di * (1 - dj);
    p.w11 = di * dj;
    return p;
}

inline float PatchDiff(device const float* I0, device const float* I1, int w, int h, int i, int j, Bilin p, int r, int c)
{
    const float v = p.w00 * At(I1, w, h, p.ix + c, p.iy + r) + p.w01 * At(I1, w, h, p.ix + c + 1, p.iy + r) +
                    p.w10 * At(I1, w, h, p.ix + c, p.iy + r + 1) + p.w11 * At(I1, w, h, p.ix + c + 1, p.iy + r + 1);
    return v - I0[(i + r) * w + j + c];
}

inline float SimdPatchSSD(device const float* I0, device const float* I1, int w, int h, int i, int j, float2 U,
                          int lane, int simdW)
{
    const Bilin p = PatchPos(i, j, U, w, h);
    float sd = 0, sq = 0;
    for (int k = lane; k < kPsz * kPsz; k += simdW)
    {
        const float v = PatchDiff(I0, I1, w, h, i, j, p, k / kPsz, k % kPsz);
        sd += v;
        sq += v * v;
    }
    sd = simd_sum(sd);
    sq = simd_sum(sq);
    return sq - sd * sd * (1.0f / (kPsz * kPsz));
}

inline float2 SimdPatchGN(device const float* I0, device const float* I1, device const float* gx,
                          device const float* gy, int w, int h, int i, int j, float i11, float i12, float i22, float gsx,
                          float gsy, int iters, float2 u, int lane, int simdW)
{
    const float invN = 1.0f / (kPsz * kPsz);
    float prevSSD = 1e10f;
    for (int t = 0; t < iters; ++t)
    {
        const Bilin p = PatchPos(i, j, u, w, h);
        float sd = 0, sq = 0, mx = 0, my = 0;
        for (int k = lane; k < kPsz * kPsz; k += simdW)
        {
            const int r = k / kPsz, c = k % kPsz;
            const float v = PatchDiff(I0, I1, w, h, i, j, p, r, c);
            sd += v;
            sq += v * v;
            mx += v * gx[(i + r) * w + j + c];
            my += v * gy[(i + r) * w + j + c];
        }
        sd = simd_sum(sd);
        sq = simd_sum(sq);
        mx = simd_sum(mx);
        my = simd_sum(my);
        const float dUx = mx - sd * gsx * invN, dUy = my - sd * gsy * invN, ssd = sq - sd * sd * invN;
        u.x -= i11 * dUx + i12 * dUy;
        u.y -= i12 * dUx + i22 * dUy;
        if (ssd >= prevSSD)
            break;
        prevSSD = ssd;
    }
    return u;
}

// Patch inverse search in the CPU DIS order (see KPatchSearch in DisFlowCuda.cuh):
// one threadgroup per stripe, one SIMD-group per patch on each anti-diagonal.
kernel void d_patch_search(device const float2* U [[buffer(0)]], device const float* I0 [[buffer(1)]],
                           device const float* I1 [[buffer(2)]], device const float* gx [[buffer(3)]],
                           device const float* gy [[buffer(4)]], device const float* xx [[buffer(5)]],
                           device const float* yy [[buffer(6)]], device const float* xy [[buffer(7)]],
                           device const float* sx [[buffer(8)]], device const float* sy [[buffer(9)]],
                           device float2* S [[buffer(10)]], constant DisArgs& a [[buffer(11)]],
                           uint stripe [[threadgroup_position_in_grid]], uint warp [[simdgroup_index_in_threadgroup]],
                           uint nWarps [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint simdW [[threads_per_simdgroup]])
{
    const int w = a.w, h = a.h, ws = a.ws, hs = a.hs, stride = a.stride;
    const int stripeRows = (hs + kStripes - 1) / kStripes;
    const int rs = min(int(stripe) * stripeRows, hs), re = min(rs + stripeRows, hs);
    const int rows = re - rs;
    const int c = kPsz / 2;
    for (int pass = 0; pass < 2; ++pass)
    {
        const int dir = pass == 0 ? 1 : -1;
        for (int d = 0; d < rows + ws - 1; ++d)
        {
            for (int r = int(warp); r < rows; r += int(nWarps))
            {
                const int q = d - r;
                if (q < 0 || q >= ws)
                    continue;
                const int is = dir > 0 ? rs + r : re - 1 - r;
                const int js = dir > 0 ? q : ws - 1 - q;
                const int i = is * stride, j = js * stride, k = is * ws + js;
                float2 best = pass == 0 ? U[(i + c) * w + j + c] : S[k];
                float minSSD = SimdPatchSSD(I0, I1, w, h, i, j, best, int(lane), int(simdW));
                if (q > 0)
                {
                    const float2 cand = S[k - dir];
                    const float ssd = SimdPatchSSD(I0, I1, w, h, i, j, cand, int(lane), int(simdW));
                    if (ssd < minSSD)
                    {
                        minSSD = ssd;
                        best = cand;
                    }
                }
                if (r > 0)
                {
                    const float2 cand = S[k - dir * ws];
                    const float ssd = SimdPatchSSD(I0, I1, w, h, i, j, cand, int(lane), int(simdW));
                    if (ssd < minSSD)
                        best = cand;
                }
                const float hxx = xx[k], hyy = yy[k], hxy = xy[k];
                float det = hxx * hyy - hxy * hxy;
                if (abs(det) < 0.001f)
                    det = 0.001f;
                const float2 u = SimdPatchGN(I0, I1, gx, gy, w, h, i, j, hyy / det, -hxy / det, hxx / det, sx[k], sy[k],
                                             a.iters, best, int(lane), int(simdW));
                const float dx = u.x - best.x, dy = u.y - best.y;
                if (lane == 0)
                    S[k] = dx * dx + dy * dy <= float(kPsz * kPsz) ? u : best;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
}

kernel void d_densify(device const float2* S [[buffer(0)]], device const float* I0 [[buffer(1)]],
                      device const float* I1 [[buffer(2)]], device float2* U [[buffer(3)]],
                      constant DisArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h, stride = a.stride;
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
            const float2 s = S[is * a.ws + js];
            const float jm = min(max(x + s.x, 0.0f), w - 1.0f - 0.001f);
            const float im = min(max(y + s.y, 0.0f), h - 1.0f - 0.001f);
            const int jl = int(jm), il = int(im), ju = jl + 1, iu = il + 1;
            const float diff = (jm - jl) * (im - il) * I1[iu * w + ju] + (ju - jm) * (im - il) * I1[iu * w + jl] +
                               (jm - jl) * (iu - im) * I1[il * w + ju] + (ju - jm) * (iu - im) * I1[il * w + jl] - i0;
            const float coef = 1.0f / max(1.0f, abs(diff));
            su += coef * s.x;
            sv += coef * s.y;
            sc += coef;
        }
    U[y * w + x] = float2(su / sc, sv / sc);
}

kernel void d_var_warp(device const float* I0 [[buffer(0)]], device const float* I1 [[buffer(1)]],
                       device const float2* W [[buffer(2)]], device float* avg [[buffer(3)]], device float* Iz [[buffer(4)]],
                       constant DisArgs& a [[buffer(5)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h;
    if (x >= w || y >= h)
        return;
    const float2 u = W[y * w + x];
    const float wi = BilinearClamp(I1, w, h, x + u.x, y + u.y);
    const float i0 = I0[y * w + x];
    avg[y * w + x] = 0.5f * (i0 + wi);
    Iz[y * w + x] = wi - i0;
}

kernel void d_var_deriv1(device const float* avg [[buffer(0)]], device const float* Iz [[buffer(1)]],
                         device float* Ix [[buffer(2)]], device float* Iy [[buffer(3)]], device float* Ixz [[buffer(4)]],
                         device float* Iyz [[buffer(5)]], constant DisArgs& a [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h;
    if (x >= w || y >= h)
        return;
    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1), ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    const int k = y * w + x;
    Ix[k] = avg[y * w + xp] - avg[y * w + xm];
    Iy[k] = avg[yp * w + x] - avg[ym * w + x];
    Ixz[k] = Iz[y * w + xp] - Iz[y * w + xm];
    Iyz[k] = Iz[yp * w + x] - Iz[ym * w + x];
}

kernel void d_var_deriv2(device const float* Ix [[buffer(0)]], device const float* Iy [[buffer(1)]],
                         device float* Ixx [[buffer(2)]], device float* Ixy [[buffer(3)]], device float* Iyy [[buffer(4)]],
                         constant DisArgs& a [[buffer(5)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h;
    if (x >= w || y >= h)
        return;
    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1), ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    const int k = y * w + x;
    Ixx[k] = Ix[y * w + xp] - Ix[y * w + xm];
    Ixy[k] = Ix[yp * w + x] - Ix[ym * w + x];
    Iyy[k] = Iy[yp * w + x] - Iy[ym * w + x];
}

kernel void d_var_smooth(device const float2* tW [[buffer(0)]], device float* sw [[buffer(1)]],
                         constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h;
    if (x >= w || y >= h)
        return;
    const float2 c = tW[y * w + x];
    const float2 r = x + 1 < w ? tW[y * w + x + 1] : c;
    const float2 d = y + 1 < h ? tW[(y + 1) * w + x] : c;
    const float ux = r.x - c.x, vx = r.y - c.y, uy = d.x - c.x, vy = d.y - c.y;
    sw[y * w + x] = (kVrAlpha / 2) / sqrt(ux * ux + vx * vx + uy * uy + vy * vy + kVrEps * kVrEps);
}

kernel void d_var_system(device const float* vIx [[buffer(0)]], device const float* vIy [[buffer(1)]],
                         device const float* vIz [[buffer(2)]], device const float* vIxx [[buffer(3)]],
                         device const float* vIxy [[buffer(4)]], device const float* vIyy [[buffer(5)]],
                         device const float* vIxz [[buffer(6)]], device const float* vIyz [[buffer(7)]],
                         device const float2* dW [[buffer(8)]], device const float2* W [[buffer(9)]],
                         device const float* sw [[buffer(10)]], device float* A11 [[buffer(11)]],
                         device float* A12 [[buffer(12)]], device float* A22 [[buffer(13)]], device float* B1 [[buffer(14)]],
                         device float* B2 [[buffer(15)]], constant DisArgs& a [[buffer(16)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.w, h = a.h;
    if (x >= w || y >= h)
        return;
    const int k = y * w + x;
    const float zeta2 = kVrZeta * kVrZeta, eps2 = kVrEps * kVrEps;
    const float Ix = vIx[k], Iy = vIy[k], Iz = vIz[k];
    const float Ixx = vIxx[k], Ixy = vIxy[k], Iyy = vIyy[k], Ixz = vIxz[k], Iyz = vIyz[k];
    const float2 d = dW[k];
    float norm = Ix * Ix + Iy * Iy + zeta2;
    const float e = Iz + Ix * d.x + Iy * d.y;
    float wt = ((kVrDelta / 2) / sqrt(e * e / norm + eps2)) / norm;
    float a11 = wt * Ix * Ix + zeta2, a12 = wt * Ix * Iy, a22 = wt * Iy * Iy + zeta2;
    float b1 = -wt * Iz * Ix, b2 = -wt * Iz * Iy;
    norm = Ixx * Ixx + Ixy * Ixy + zeta2;
    const float norm2 = Iyy * Iyy + Ixy * Ixy + zeta2;
    const float ex = Ixz + Ixx * d.x + Ixy * d.y, ey = Iyz + Ixy * d.x + Iyy * d.y;
    wt = (kVrGamma / 2) / sqrt(ex * ex / norm + ey * ey / norm2 + eps2);
    a11 += wt * (Ixx * Ixx / norm + Ixy * Ixy / norm2);
    a12 += wt * (Ixx * Ixy / norm + Ixy * Iyy / norm2);
    a22 += wt * (Ixy * Ixy / norm + Iyy * Iyy / norm2);
    b1 -= wt * (Ixx * Ixz / norm + Ixy * Iyz / norm2);
    b2 -= wt * (Ixy * Ixz / norm + Iyy * Iyz / norm2);
    const float2 W0 = W[k];
    float sA = 0, su = 0, sv = 0;
    if (x + 1 < w)
    {
        const float s = sw[k];
        const float2 n = W[k + 1];
        sA += s;
        su += s * (n.x - W0.x);
        sv += s * (n.y - W0.y);
    }
    if (x > 0)
    {
        const float s = sw[k - 1];
        const float2 n = W[k - 1];
        sA += s;
        su -= s * (W0.x - n.x);
        sv -= s * (W0.y - n.y);
    }
    if (y + 1 < h)
    {
        const float s = sw[k];
        const float2 n = W[k + w];
        sA += s;
        su += s * (n.x - W0.x);
        sv += s * (n.y - W0.y);
    }
    if (y > 0)
    {
        const float s = sw[k - w];
        const float2 n = W[k - w];
        sA += s;
        su -= s * (W0.x - n.x);
        sv -= s * (W0.y - n.y);
    }
    A11[k] = a11 + sA;
    A12[k] = a12;
    A22[k] = a22 + sA;
    B1[k] = b1 + su;
    B2[k] = b2 + sv;
}

kernel void d_var_sor(device const float* sw [[buffer(0)]], device float2* dW [[buffer(1)]],
                      device const float* A11 [[buffer(2)]], device const float* A12 [[buffer(3)]],
                      device const float* A22 [[buffer(4)]], device const float* B1 [[buffer(5)]],
                      device const float* B2 [[buffer(6)]], constant DisArgs& a [[buffer(7)]],
                      uint2 gid [[thread_position_in_grid]])
{
    const int y = gid.y, w = a.w, h = a.h;
    const int x = 2 * int(gid.x) + ((y + a.colour) & 1);
    if (x >= w || y >= h)
        return;
    const int k = y * w + x;
    float su = 0, sv = 0;
    if (x > 0)
    {
        const float s = sw[k - 1];
        su += s * dW[k - 1].x;
        sv += s * dW[k - 1].y;
    }
    if (x + 1 < w)
    {
        const float s = sw[k];
        su += s * dW[k + 1].x;
        sv += s * dW[k + 1].y;
    }
    if (y > 0)
    {
        const float s = sw[k - w];
        su += s * dW[k - w].x;
        sv += s * dW[k - w].y;
    }
    if (y + 1 < h)
    {
        const float s = sw[k];
        su += s * dW[k + w].x;
        sv += s * dW[k + w].y;
    }
    float2 d = dW[k];
    d.x += kVrOmega * ((su + B1[k] - d.y * A12[k]) / A11[k] - d.x);
    d.y += kVrOmega * ((sv + B2[k] - d.x * A12[k]) / A22[k] - d.y);
    dW[k] = d;
}

// dst = src0 + src1 (src1 may be the zero increment); also used as a copy with src1 = zeros
kernel void d_flow_add(device const float2* src0 [[buffer(0)]], device const float2* src1 [[buffer(1)]],
                       device float2* dst [[buffer(2)]], constant DisArgs& a [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= a.w || y >= a.h)
        return;
    const int k = y * a.w + x;
    dst[k] = src0[k] + src1[k];
}

kernel void d_flow_zero(device float2* dst [[buffer(0)]], constant DisArgs& a [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= a.w || y >= a.h)
        return;
    dst[y * a.w + x] = float2(0.0f);
}

kernel void d_resize_flow(device const float2* src [[buffer(0)]], device float2* dst [[buffer(1)]],
                          constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, sw = a.sw, sh = a.sh;
    if (x >= a.dw || y >= a.dh)
        return;
    float fx = (x + 0.5f) * (float(sw) / a.dw) - 0.5f, fy = (y + 0.5f) * (float(sh) / a.dh) - 0.5f;
    fx = min(max(fx, 0.0f), float(sw - 1));
    fy = min(max(fy, 0.0f), float(sh - 1));
    const int x0 = int(fx), y0 = int(fy), x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
    const float tx = fx - x0, ty = fy - y0;
    const float2 p = src[y0 * sw + x0], q = src[y0 * sw + x1], r = src[y1 * sw + x0], s = src[y1 * sw + x1];
    dst[y * a.dw + x] = a.mul * ((1 - ty) * ((1 - tx) * p + tx * q) + ty * ((1 - tx) * r + tx * s));
}

kernel void d_edge_flag(device const float2* f [[buffer(0)]], device uchar* flag [[buffer(1)]],
                        constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.fw, h = a.fh, qw = a.qw, qh = a.qh;
    if (x >= qw || y >= qh)
        return;
    float mnx = 1e30f, mxx = -1e30f, mny = 1e30f, mxy = -1e30f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int qx = x + dx, qy = y + dy;
            if (qx < 0 || qy < 0 || qx >= qw || qy >= qh)
                continue;
            const int sx = min(int(floor(qx * (float(w) / qw))), w - 1), sy = min(int(floor(qy * (float(h) / qh))), h - 1);
            const float2 v = f[sy * w + sx];
            mnx = min(mnx, v.x);
            mxx = max(mxx, v.x);
            mny = min(mny, v.y);
            mxy = max(mxy, v.y);
        }
    flag[y * qw + x] = (mxx - mnx) + (mxy - mny) > 3.0f ? 1 : 0;
}

kernel void d_dilate(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                     constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, qw = a.qw, qh = a.qh, r = a.r;
    if (x >= qw || y >= qh)
        return;
    uchar m = 0;
    for (int dy = -r; dy <= r && m == 0; ++dy)
    {
        const int qy = y + dy;
        if (qy < 0 || qy >= qh)
            continue;
        for (int dx = -r; dx <= r; ++dx)
        {
            const int qx = x + dx;
            if (qx >= 0 && qx < qw && src[qy * qw + qx] != 0)
            {
                m = 1;
                break;
            }
        }
    }
    dst[y * qw + x] = m;
}

kernel void d_edge_refine(device const float2* src [[buffer(0)]], device const uchar4* col [[buffer(1)]],
                          device const uchar* maskQ [[buffer(2)]], device float2* dst [[buffer(3)]],
                          constant DisArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, w = a.fw, h = a.fh, qw = a.qw, qh = a.qh;
    if (x >= w || y >= h)
        return;
    const float2 own = src[y * w + x];
    const int mx = min(int(floor(x * (float(qw) / w))), qw - 1), my = min(int(floor(y * (float(qh) / h))), qh - 1);
    if (maskQ[my * qw + mx] == 0)
    {
        dst[y * w + x] = own;
        return;
    }
    const int radii[5] = {3, 8, 16, 28, 42};
    const int dirs[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    float vx[41], vy[41], wt[41];
    const uchar4 c0 = col[y * w + x];
    float spread = 0;
    vx[0] = own.x;
    vy[0] = own.y;
    wt[0] = 1.0f;
    int n = 1;
    for (int ri = 0; ri < 5; ++ri)
        for (int di = 0; di < 8; ++di)
        {
            const int cx = x + dirs[di][0] * radii[ri], cy = y + dirs[di][1] * radii[ri];
            if (cx < 0 || cy < 0 || cx >= w || cy >= h)
                continue;
            const float2 v = src[cy * w + cx];
            const uchar4 c = col[cy * w + cx];
            spread = max(spread, abs(v.x - own.x) + abs(v.y - own.y));
            const float dcol = float(abs(int(c.x) - int(c0.x)) + abs(int(c.y) - int(c0.y)) + abs(int(c.z) - int(c0.z)));
            vx[n] = v.x;
            vy[n] = v.y;
            wt[n] = exp(-(dcol / 36.0f) * (dcol / 36.0f));
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
    float bestX = 3.4e38f, bestY = 3.4e38f;
    for (int i = 0; i < n; ++i)
    {
        float accX = 0, accY = 0;
        for (int j = 0; j < n; ++j)
        {
            if (vx[j] <= vx[i])
                accX += wt[j];
            if (vy[j] <= vy[i])
                accY += wt[j];
        }
        if (accX >= half && vx[i] < bestX)
            bestX = vx[i];
        if (accY >= half && vy[i] < bestY)
            bestY = vy[i];
    }
    dst[y * w + x] = float2(bestX, bestY);
}

kernel void d_store_flow(device const float2* f [[buffer(0)]], device float2* dst [[buffer(1)]],
                         constant DisArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= a.fw || y >= a.fh)
        return;
    const int k = y * a.fw + x;
    dst[k] = a.acc != 0 ? dst[k] + a.mul * f[k] : a.mul * f[k];
}
)MSL";

const char* const kDisKernels[] = {"d_area_frame", "d_box3",       "d_area_gray",  "d_sobel",       "d_structure",
                                   "d_patch_search", "d_densify",  "d_var_warp",   "d_var_deriv1",  "d_var_deriv2",
                                   "d_var_smooth", "d_var_system", "d_var_sor",    "d_flow_add",    "d_flow_zero",
                                   "d_resize_flow", "d_edge_flag", "d_dilate",     "d_edge_refine", "d_store_flow"};
enum DisKernel
{
    kAreaFrame,
    kBox3,
    kAreaGray,
    kSobel,
    kStructure,
    kPatchSearch,
    kDensify,
    kVarWarp,
    kVarDeriv1,
    kVarDeriv2,
    kVarSmooth,
    kVarSystem,
    kVarSor,
    kFlowAdd,
    kFlowZero,
    kResizeFlow,
    kEdgeFlag,
    kDilate,
    kEdgeRefine,
    kStoreFlow,
    kDisKernelCount
};

constexpr int kPsz = 8;
constexpr int kMaxLevels = 10;
constexpr int kStripes = 8;
constexpr int kVrSor = 5;

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

// Host side: encodes the DIS passes into the engine's compute encoder.
// Buffers are shared MTLBuffers, tightly packed (pitch = width).
class MetalDis
{
public:
    static constexpr int kSlots = 3;

    bool Init(id<MTLDevice> dev, id<MTLLibrary> lib, std::string* err)
    {
        dev_ = dev;
        for (int k = 0; k < kDisKernelCount; ++k)
        {
            NSError* e = nil;
            id<MTLFunction> fn = [lib newFunctionWithName:@(kDisKernels[k])];
            ps_[k] = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
            if (!ps_[k])
                return Fail(err, (std::string("Metal pipeline ") + kDisKernels[k] + " failed").c_str());
        }
        return true;
    }

    // Same as DisFlow::Setup in DisFlowCuda.cuh.
    bool Setup(int W, int H, int flowWidth, int preset)
    {
        p_ = Preset(preset);
        const double scale = std::min(1.0, (double)flowWidth / W);
        scale_ = scale;
        frameW_ = W;
        frameH_ = H;
        fw_ = std::max(1, (int)std::lround(W * scale));
        fh_ = std::max(1, (int)std::lround(H * scale));
        const int coarsest = std::min((int)(std::log(std::max(fw_, fh_) / (4.0 * kPsz)) / std::log(2.0) + 0.5),
                                      (int)(std::log(std::min(fw_, fh_) / (double)kPsz) / std::log(2.0)));
        if (coarsest < p_.finest || coarsest >= kMaxLevels)
            return false;
        coarsest_ = coarsest;
        lw_[0] = fw_;
        lh_[0] = fh_;
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

    bool Prepare(id<MTLComputeCommandEncoder> enc, int slot, id<MTLBuffer> frame, std::string* err)
    {
        Slot& s = slot_[slot];
        const size_t n0 = (size_t)fw_ * fh_;
        if (!Ensure(s.gray[0], n0 * 4) || !Ensure(s.colourRaw, n0 * 4) || !Ensure(s.colour, n0 * 4))
            return Fail(err, "out of GPU memory (motion)");
        DisArgs a = Args();
        Run(enc, kAreaFrame, {frame, s.gray[0], s.colourRaw}, a, fw_, fh_);
        Run(enc, kBox3, {s.colourRaw, s.colour}, a, fw_, fh_);
        for (int i = p_.finest; i <= coarsest_; ++i)
        {
            const size_t n = (size_t)lw_[i] * lh_[i];
            if ((i > 0 && !Ensure(s.gray[i], n * 4)) || !Ensure(s.gx[i], n * 4) || !Ensure(s.gy[i], n * 4))
                return Fail(err, "out of GPU memory (motion)");
            if (i > 0)
            {
                const int src = i == p_.finest ? 0 : i - 1;
                DisArgs r = Args();
                r.sw = lw_[src];
                r.sh = lh_[src];
                r.dw = lw_[i];
                r.dh = lh_[i];
                Run(enc, kAreaGray, {s.gray[src], s.gray[i]}, r, lw_[i], lh_[i]);
            }
            DisArgs g = Args();
            g.w = lw_[i];
            g.h = lh_[i];
            Run(enc, kSobel, {s.gray[i], s.gx[i], s.gy[i]}, g, lw_[i], lh_[i]);
        }
        return true;
    }

    // Same as DisFlow::Compute in DisFlowCuda.cuh; dst: fw x fh float2, tight.
    bool Compute(id<MTLComputeCommandEncoder> enc, int from, int to, bool refineEdges, float weight, bool accumulate,
                 id<MTLBuffer> dst, std::string* err)
    {
        const Slot& A = slot_[from];
        const Slot& B = slot_[to];
        const int fin = p_.finest;
        const size_t nf = (size_t)lw_[fin] * lh_[fin];
        const size_t np = (size_t)(1 + (lw_[fin] - kPsz) / p_.stride) * (1 + (lh_[fin] - kPsz) / p_.stride);
        bool ok = Ensure(S_, np * 8) && Ensure(U_[0], nf * 8) && Ensure(U_[1], nf * 8);
        for (id<MTLBuffer> __strong* q : {&xx_, &yy_, &xy_, &sx_, &sy_})
            ok = ok && Ensure(*q, np * 4);
        if (p_.varIter > 0)
        {
            for (id<MTLBuffer> __strong* q : {&vIx_, &vIy_, &vIz_, &vIxx_, &vIxy_, &vIyy_, &vIxz_, &vIyz_, &vAvg_, &a11_,
                                              &a12_, &a22_, &b1_, &b2_, &sw_})
                ok = ok && Ensure(*q, nf * 4);
            ok = ok && Ensure(W_, nf * 8) && Ensure(dW_, nf * 8) && Ensure(tW_, nf * 8);
        }
        const size_t n0 = (size_t)fw_ * fh_;
        ok = ok && Ensure(f0_, n0 * 8) && Ensure(f1_, n0 * 8);
        if (!ok)
            return Fail(err, "out of GPU memory (motion)");

        const int iters = (int)std::floor(p_.gdIter / 2.0f);
        for (int i = coarsest_; i >= fin; --i)
        {
            const int w = lw_[i], h = lh_[i];
            const int ws = 1 + (w - kPsz) / p_.stride, hs = 1 + (h - kPsz) / p_.stride;
            id<MTLBuffer> U = U_[i & 1];
            DisArgs a = Args();
            a.w = w;
            a.h = h;
            a.ws = ws;
            a.hs = hs;
            a.stride = p_.stride;
            a.iters = iters;
            if (i == coarsest_)
                Run(enc, kFlowZero, {U}, a, w, h);
            Run(enc, kStructure, {A.gx[i], A.gy[i], xx_, yy_, xy_, sx_, sy_}, a, ws, hs);
            {
                const int stripeRows = (hs + kStripes - 1) / kStripes;
                id<MTLComputePipelineState> ps = ps_[kPatchSearch];
                const NSUInteger maxT = std::max<NSUInteger>(32, ps.maxTotalThreadsPerThreadgroup / 32 * 32);
                const NSUInteger t = std::min<NSUInteger>(32 * std::min(stripeRows, 32), maxT);
                Bind(enc, ps, {U, A.gray[i], B.gray[i], A.gx[i], A.gy[i], xx_, yy_, xy_, sx_, sy_, S_}, a);
                [enc dispatchThreadgroups:MTLSizeMake(kStripes, 1, 1) threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
            }
            Run(enc, kDensify, {S_, A.gray[i], B.gray[i], U}, a, w, h);
            if (p_.varIter > 0)
                Refine(enc, A.gray[i], B.gray[i], U, w, h);
            if (i > fin)
            {
                DisArgs r = Args();
                r.sw = w;
                r.sh = h;
                r.dw = lw_[i - 1];
                r.dh = lh_[i - 1];
                r.mul = 2.0f;
                Run(enc, kResizeFlow, {U, U_[(i - 1) & 1]}, r, r.dw, r.dh);
            }
        }
        DisArgs r = Args();
        r.sw = lw_[fin];
        r.sh = lh_[fin];
        r.dw = fw_;
        r.dh = fh_;
        r.mul = (float)(1 << fin);
        Run(enc, kResizeFlow, {U_[fin & 1], f0_}, r, fw_, fh_);
        id<MTLBuffer> result = f0_;
        if (refineEdges)
        {
            constexpr int kReach = 42;
            DisArgs e = Args();
            e.qw = (fw_ + 3) / 4;
            e.qh = (fh_ + 3) / 4;
            e.r = kReach / 4 + 1;
            if (!Ensure(flagQ_[0], (size_t)e.qw * e.qh) || !Ensure(flagQ_[1], (size_t)e.qw * e.qh))
                return Fail(err, "out of GPU memory (motion)");
            Run(enc, kEdgeFlag, {f0_, flagQ_[0]}, e, e.qw, e.qh);
            Run(enc, kDilate, {flagQ_[0], flagQ_[1]}, e, e.qw, e.qh);
            Run(enc, kEdgeRefine, {f0_, A.colour, flagQ_[1], f1_}, e, fw_, fh_);
            result = f1_;
        }
        DisArgs st = Args();
        st.mul = (float)(weight / scale_);
        st.acc = accumulate ? 1 : 0;
        Run(enc, kStoreFlow, {result, dst}, st, fw_, fh_);
        return true;
    }

    // Zero motion into dst (fw x fh), for a single frame without neighbours.
    void Zero(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> dst)
    {
        DisArgs a = Args();
        a.w = fw_;
        a.h = fh_;
        Run(enc, kFlowZero, {dst}, a, fw_, fh_);
    }

private:
    struct Slot
    {
        id<MTLBuffer> gray[kMaxLevels], gx[kMaxLevels], gy[kMaxLevels];
        id<MTLBuffer> colourRaw, colour;
    };

    static bool Fail(std::string* err, const char* msg)
    {
        if (err)
            *err = msg;
        return false;
    }

    bool Ensure(id<MTLBuffer> __strong& b, size_t bytes)
    {
        if (b && b.length >= bytes)
            return true;
        b = [dev_ newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModePrivate];
        return b != nil;
    }

    DisArgs Args() const
    {
        DisArgs a = {};
        a.W = frameW_;
        a.H = frameH_;
        a.fw = fw_;
        a.fh = fh_;
        a.mul = 1.0f;
        return a;
    }

    void Bind(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, std::initializer_list<id<MTLBuffer>> bufs,
              const DisArgs& a)
    {
        [enc setComputePipelineState:ps];
        NSUInteger i = 0;
        for (id<MTLBuffer> b : bufs)
            [enc setBuffer:b offset:0 atIndex:i++];
        [enc setBytes:&a length:sizeof(a) atIndex:i];
    }

    void Run(id<MTLComputeCommandEncoder> enc, DisKernel k, std::initializer_list<id<MTLBuffer>> bufs, const DisArgs& a,
             int w, int h)
    {
        id<MTLComputePipelineState> ps = ps_[k];
        Bind(enc, ps, bufs, a);
        const NSUInteger t = ps.maxTotalThreadsPerThreadgroup >= 256 ? 16 : 8;
        [enc dispatchThreadgroups:MTLSizeMake((w + t - 1) / t, (h + t - 1) / t, 1)
            threadsPerThreadgroup:MTLSizeMake(t, t, 1)];
    }

    // Variational refinement of U (in place) at one level.
    void Refine(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> I0, id<MTLBuffer> I1, id<MTLBuffer> U, int w, int h)
    {
        DisArgs a = Args();
        a.w = w;
        a.h = h;
        Run(enc, kFlowZero, {dW_}, a, w, h);
        Run(enc, kFlowAdd, {U, dW_, W_}, a, w, h);  // W = U
        Run(enc, kFlowAdd, {U, dW_, tW_}, a, w, h); // tW = U
        Run(enc, kVarWarp, {I0, I1, W_, vAvg_, vIz_}, a, w, h);
        Run(enc, kVarDeriv1, {vAvg_, vIz_, vIx_, vIy_, vIxz_, vIyz_}, a, w, h);
        Run(enc, kVarDeriv2, {vIx_, vIy_, vIxx_, vIxy_, vIyy_}, a, w, h);
        for (int it = 0; it < p_.varIter; ++it)
        {
            Run(enc, kVarSmooth, {tW_, sw_}, a, w, h);
            Run(enc, kVarSystem,
                {vIx_, vIy_, vIz_, vIxx_, vIxy_, vIyy_, vIxz_, vIyz_, dW_, W_, sw_, a11_, a12_, a22_, b1_, b2_}, a, w, h);
            for (int s = 0; s < kVrSor; ++s)
                for (int colour = 0; colour < 2; ++colour)
                {
                    DisArgs c = a;
                    c.colour = colour;
                    Run(enc, kVarSor, {sw_, dW_, a11_, a12_, a22_, b1_, b2_}, c, (w + 1) / 2, h);
                }
            Run(enc, kFlowAdd, {W_, dW_, tW_}, a, w, h);
        }
        Run(enc, kFlowAdd, {W_, dW_, U}, a, w, h); // U = W + dW (= tW)
    }

    id<MTLDevice> dev_ = nil;
    id<MTLComputePipelineState> ps_[kDisKernelCount] = {};
    Params p_ = Preset(1);
    double scale_ = 1.0;
    int frameW_ = 0, frameH_ = 0, fw_ = 0, fh_ = 0, coarsest_ = 0;
    int lw_[kMaxLevels] = {}, lh_[kMaxLevels] = {};
    Slot slot_[kSlots];
    id<MTLBuffer> xx_, yy_, xy_, sx_, sy_;
    id<MTLBuffer> vIx_, vIy_, vIz_, vIxx_, vIxy_, vIyy_, vIxz_, vIyz_, vAvg_, a11_, a12_, a22_, b1_, b2_, sw_;
    id<MTLBuffer> S_, U_[2], W_, dW_, tW_, f0_, f1_;
    id<MTLBuffer> flagQ_[2];
};

} // namespace oe::gpu::dis
