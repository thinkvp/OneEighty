// GPU motion blur. See GpuMotionBlur.h.
//
// Per output frame:
//   1. upload the frame(s) to float BGRA (8-bit converted, float copied as is)
//      and the caller's motion field(s)
//   2. mip chain of each frame (1/2, 1/4, 1/8, box filtered)
//   3. each pixel: average the frame along +/- halfWidth * v; long smears are
//      sampled from a mip level so the cost stays bounded
//   4. download in the input's layout
// The *GpuFlow calls also estimate the motion here, from the uploaded frames
// (DisFlowCuda.cuh), instead of taking it from the caller.

#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#include "GpuMotionBlur.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>

namespace oe::gpu
{
namespace
{

// ------------------------------------------------------------ pixel access
// Device images are 4 x float32, B,G,R,A, 0..1 (values above 1 allowed).
__device__ __forceinline__ float4 Load(const char* base, size_t pitch, int x, int y)
{
    return reinterpret_cast<const float4*>(base + (size_t)y * pitch)[x];
}

__device__ __forceinline__ void Store(char* base, size_t pitch, int x, int y, float4 v)
{
    reinterpret_cast<float4*>(base + (size_t)y * pitch)[x] = v;
}

__device__ __forceinline__ float4 Bilinear(const char* base, size_t pitch, int w, int h, float sx, float sy)
{
    sx = fminf(fmaxf(sx, 0.0f), (float)(w - 1));
    sy = fminf(fmaxf(sy, 0.0f), (float)(h - 1));
    const int x0 = min((int)sx, max(0, w - 2)), y0 = min((int)sy, max(0, h - 2));
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float4 a = Load(base, pitch, x0, y0), b = Load(base, pitch, x1, y0);
    const float4 c = Load(base, pitch, x0, y1), d = Load(base, pitch, x1, y1);
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    return make_float4(w00 * a.x + w10 * b.x + w01 * c.x + w11 * d.x, w00 * a.y + w10 * b.y + w01 * c.y + w11 * d.y,
                       w00 * a.z + w10 * b.z + w01 * c.z + w11 * d.z, w00 * a.w + w10 * b.w + w01 * c.w + w11 * d.w);
}

// ------------------------------------------------------------------ kernels
// 2x2 box downsample into the next mip level.
__global__ void KDown(const char* src, size_t sp, int sw, int sh, char* dst, size_t dp, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;
    const int x0 = min(2 * x, sw - 1), x1 = min(2 * x + 1, sw - 1), y0 = min(2 * y, sh - 1), y1 = min(2 * y + 1, sh - 1);
    const float4 a = Load(src, sp, x0, y0), b = Load(src, sp, x1, y0);
    const float4 c = Load(src, sp, x0, y1), d = Load(src, sp, x1, y1);
    Store(dst, dp, x, y,
          make_float4((a.x + b.x + c.x + d.x) * 0.25f, (a.y + b.y + c.y + d.y) * 0.25f,
                      (a.z + b.z + c.z + d.z) * 0.25f, (a.w + b.w + c.w + d.w) * 0.25f));
}

struct Mip
{
    const char* ptr;
    size_t pitch;
    int w, h;
};
struct MipSet
{
    Mip level[4]; // level[0] = the frame itself
    int top;
};

// Motion-field colour: hue = direction, saturation = speed / norm (RGB 0..1).
__device__ float3 VectorColour(float vx, float vy, float norm)
{
    const float sat = fminf(1.0f, sqrtf(vx * vx + vy * vy) / norm);
    float h = atan2f(vy, vx) / 6.2831853f;
    if (h < 0)
        h += 1.0f;
    const float h6 = h * 6.0f;
    const int i = ((int)h6) % 6;
    const float f = h6 - floorf(h6);
    const float p = 1 - sat, q = 1 - sat * f, t = 1 - sat * (1 - f);
    switch (i)
    {
    case 0: return make_float3(1, t, p);
    case 1: return make_float3(q, 1, p);
    case 2: return make_float3(p, 1, t);
    case 3: return make_float3(p, q, 1);
    case 4: return make_float3(t, p, 1);
    default: return make_float3(1, p, q);
    }
}

struct FlowField
{
    const float2* ptr;
    size_t pitch;
    int w, h;
};

// Velocity at full-res pixel (x, y): bilinear from the (coarser) flow grid.
__device__ __forceinline__ float2 SampleFlow(const FlowField& f, int W, int H, int x, int y)
{
    float gx = (x + 0.5f) * ((float)f.w / W) - 0.5f, gy = (y + 0.5f) * ((float)f.h / H) - 0.5f;
    gx = fminf(fmaxf(gx, 0.0f), (float)(f.w - 1));
    gy = fminf(fmaxf(gy, 0.0f), (float)(f.h - 1));
    const int ix = min((int)gx, max(0, f.w - 2)), iy = min((int)gy, max(0, f.h - 2));
    const int ix1 = min(ix + 1, f.w - 1), iy1 = min(iy + 1, f.h - 1);
    const float tx = gx - ix, ty = gy - iy;
    auto F = [&](int a, int b) { return reinterpret_cast<const float2*>(reinterpret_cast<const char*>(f.ptr) + b * f.pitch)[a]; };
    const float2 f00 = F(ix, iy), f10 = F(ix1, iy), f01 = F(ix, iy1), f11 = F(ix1, iy1);
    return make_float2((1 - tx) * (1 - ty) * f00.x + tx * (1 - ty) * f10.x + (1 - tx) * ty * f01.x + tx * ty * f11.x,
                       (1 - tx) * (1 - ty) * f00.y + tx * (1 - ty) * f10.y + (1 - tx) * ty * f01.y + tx * ty * f11.y);
}

// Average of the image along -(c0 +/- hw) * v around (x, y): the shutter
// window [c0-hw, c0+hw] (source-frame units) of a pixel moving with v.
__device__ float4 Gather(const MipSet& mips, int x, int y, float vx, float vy, float hw, float c0, int maxS, float mipT)
{
    const Mip& L0 = mips.level[0];
    const float speed = sqrtf(vx * vx + vy * vy);
    const float smear = 2.0f * hw * speed;
    if (smear <= 1.0f && fabsf(c0) * speed < 0.5f)
        return Load(L0.ptr, L0.pitch, x, y);
    int lvl = 0;
    while (lvl < mips.top && smear / (float)(1 << lvl) > mipT)
        ++lvl;
    const float inv2l = 1.0f / (float)(1 << lvl);
    const int n = min(maxS, max(2, (int)ceilf(smear * inv2l)));
    const Mip& M = mips.level[lvl];
    float4 acc = make_float4(0, 0, 0, 0);
    for (int j = 0; j < n; ++j)
    {
        const float off = c0 + hw * (((j + 0.5f) / n) * 2.0f - 1.0f);
        const float sx = (x - off * vx + 0.5f) * inv2l - 0.5f;
        const float sy = (y - off * vy + 0.5f) * inv2l - 0.5f;
        const float4 p = Bilinear(M.ptr, M.pitch, M.w, M.h, sx, sy);
        acc.x += p.x;
        acc.y += p.y;
        acc.z += p.z;
        acc.w += p.w;
    }
    const float inv = 1.0f / n;
    return make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

__device__ float4 VectorPixel(const Mip& L0, int x, int y, float vx, float vy)
{
    const float4 p = Load(L0.ptr, L0.pitch, x, y); // x=B y=G z=R
    const float g = fminf(fmaxf(0.114f * p.x + 0.587f * p.y + 0.299f * p.z, 0.0f), 1.0f);
    const float3 c = VectorColour(vx, vy, fmaxf(4.0f, 0.01f * L0.w));
    const float k = 0.65f * (0.4f + 0.6f * g);
    return make_float4(0.35f * g + k * c.z, 0.35f * g + k * c.y, 0.35f * g + k * c.x, p.w);
}

// Single frame. mode 0: blur, 1: motion-vector view. c0 = output time offset (source frames).
__global__ void KBlur(MipSet mips, FlowField flow, char* out, size_t outP, float hw, float c0, int mode, int maxS,
                      float mipT)
{
    const Mip& L0 = mips.level[0];
    const int W = L0.w, H = L0.h;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const float2 v = SampleFlow(flow, W, H, x, y);
    Store(out, outP, x, y,
          mode == 1 ? VectorPixel(L0, x, y, v.x, v.y) : Gather(mips, x, y, v.x, v.y, hw, c0, maxS, mipT));
}

// Two-frame interpolation: output time lies a (0..1) of the way from frame A
// to frame B. Blur A at +a and B at -(1-a), each with its own motion, and
// blend by closeness.
__global__ void KBlurPair(MipSet ma, MipSet mb, FlowField fa, FlowField fb, char* out, size_t outP, float hw, float a,
                          int mode, int maxS, float mipT)
{
    const Mip& L0 = ma.level[0];
    const int W = L0.w, H = L0.h;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const float2 va = SampleFlow(fa, W, H, x, y);
    if (mode == 1)
    {
        Store(out, outP, x, y, VectorPixel(L0, x, y, va.x, va.y));
        return;
    }
    float4 r = Gather(ma, x, y, va.x, va.y, hw, a, maxS, mipT);
    if (a > 1e-3f)
    {
        const float2 vb = SampleFlow(fb, W, H, x, y);
        const float4 q = Gather(mb, x, y, vb.x, vb.y, hw, a - 1.0f, maxS, mipT);
        r = make_float4((1 - a) * r.x + a * q.x, (1 - a) * r.y + a * q.y, (1 - a) * r.z + a * q.z,
                        (1 - a) * r.w + a * q.w);
    }
    Store(out, outP, x, y, r);
}

// One time sample of the temporal supersampling (Quality High): warp both frames
// by the sample's field (bilinear from the reduced field grid), blend by its
// mask and add weight * result to acc (first sample: overwrite).
__global__ void KRifeAccum(const char* f0, size_t p0, const char* f1, size_t p1, int W, int H, const float4* flow,
                           size_t flowP, const float* mask, size_t maskP, int aw, int ah, float weight, char* acc,
                           size_t accP, int first, int vectors)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    float gx = (x + 0.5f) * ((float)aw / W) - 0.5f, gy = (y + 0.5f) * ((float)ah / H) - 0.5f;
    gx = fminf(fmaxf(gx, 0.0f), (float)(aw - 1));
    gy = fminf(fmaxf(gy, 0.0f), (float)(ah - 1));
    const int ix = min((int)gx, max(0, aw - 2)), iy = min((int)gy, max(0, ah - 2));
    const int ix1 = min(ix + 1, aw - 1), iy1 = min(iy + 1, ah - 1);
    const float tx = gx - ix, ty = gy - iy;
    const float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    auto F = [&](int a, int b) { return reinterpret_cast<const float4*>(reinterpret_cast<const char*>(flow) + b * flowP)[a]; };
    auto M = [&](int a, int b) { return reinterpret_cast<const float*>(reinterpret_cast<const char*>(mask) + b * maskP)[a]; };
    const float4 q00 = F(ix, iy), q10 = F(ix1, iy), q01 = F(ix, iy1), q11 = F(ix1, iy1);
    const float4 f = make_float4(w00 * q00.x + w10 * q10.x + w01 * q01.x + w11 * q11.x,
                                 w00 * q00.y + w10 * q10.y + w01 * q01.y + w11 * q11.y,
                                 w00 * q00.z + w10 * q10.z + w01 * q01.z + w11 * q11.z,
                                 w00 * q00.w + w10 * q10.w + w01 * q01.w + w11 * q11.w);
    const float m = fminf(fmaxf(w00 * M(ix, iy) + w10 * M(ix1, iy) + w01 * M(ix, iy1) + w11 * M(ix1, iy1), 0.0f), 1.0f);
    const float sx = (float)W / aw, sy = (float)H / ah;
    float4 r;
    if (vectors)
    {
        const Mip L0 = {f0, p0, W, H};
        r = VectorPixel(L0, x, y, -f.x * sx, -f.y * sy);
    }
    else
    {
        const float4 v0 = Bilinear(f0, p0, W, H, x + f.x * sx, y + f.y * sy);
        const float4 v1 = Bilinear(f1, p1, W, H, x + f.z * sx, y + f.w * sy);
        r = make_float4(m * v0.x + (1 - m) * v1.x, m * v0.y + (1 - m) * v1.y, m * v0.z + (1 - m) * v1.z,
                        m * v0.w + (1 - m) * v1.w);
    }
    float4 a = first ? make_float4(0, 0, 0, 0) : Load(acc, accP, x, y);
    Store(acc, accP, x, y,
          make_float4(a.x + weight * r.x, a.y + weight * r.y, a.z + weight * r.z, a.w + weight * r.w));
}

// 8-bit -> float4 (B,G,R,A order internally). argb: source bytes are A,R,G,B.
__global__ void KU8ToF4(const unsigned char* src, size_t sp, char* dst, size_t dp, int w, int h, int argb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const unsigned char* p = src + (size_t)y * sp + 4 * x;
    const float k = 1.0f / 255.0f;
    Store(dst, dp, x, y,
          argb ? make_float4(p[3] * k, p[2] * k, p[1] * k, p[0] * k) : make_float4(p[0] * k, p[1] * k, p[2] * k, p[3] * k));
}

__device__ __forceinline__ unsigned char ToU8(float v) { return (unsigned char)(fminf(fmaxf(v, 0.f), 1.f) * 255.f + 0.5f); }

__global__ void KF4ToU8(const char* src, size_t sp, unsigned char* dst, size_t dp, int w, int h, int argb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float4 v = Load(src, sp, x, y); // x=B y=G z=R w=A
    unsigned char* p = dst + (size_t)y * dp + 4 * x;
    if (argb)
    {
        p[0] = ToU8(v.w);
        p[1] = ToU8(v.z);
        p[2] = ToU8(v.y);
        p[3] = ToU8(v.x);
    }
    else
    {
        p[0] = ToU8(v.x);
        p[1] = ToU8(v.y);
        p[2] = ToU8(v.z);
        p[3] = ToU8(v.w);
    }
}

inline dim3 Grid(int w, int h, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y); }

} // namespace
} // namespace oe::gpu

#include "DisFlowCuda.cuh"

namespace oe::gpu
{
namespace
{

// ------------------------------------------------------------------ engine
struct DevBuf
{
    void* ptr = nullptr;
    size_t pitch = 0;
    int w = 0, h = 0, elem = 0;
    bool Ensure(int W, int Hh, int bytesPerPixel)
    {
        if (ptr && w == W && h == Hh && elem == bytesPerPixel)
            return true;
        Release();
        if (cudaMallocPitch(&ptr, &pitch, (size_t)W * bytesPerPixel, Hh) != cudaSuccess)
        {
            ptr = nullptr;
            return false;
        }
        w = W;
        h = Hh;
        elem = bytesPerPixel;
        return true;
    }
    void Release()
    {
        if (ptr)
            cudaFree(ptr);
        ptr = nullptr;
        w = h = 0;
    }
};

class EngineImpl final : public Engine
{
public:
    ~EngineImpl() override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (ctx_ && cuCtxPushCurrent(ctx_) == CUDA_SUCCESS)
        {
            for (DevBuf* b : {&flowA_, &flowB_, &out_, &stageOut_, &fieldFlow_, &fieldMask_})
                b->Release();
            for (int i = 0; i < kMaxFrames; ++i)
            {
                frame_[i].Release();
                stage_[i].Release();
            }
            for (auto& m : mipA_)
                m.Release();
            for (auto& m : mipB_)
                m.Release();
            for (auto& e : ev_)
                if (e)
                    cudaEventDestroy(e);
            dis_.Release();
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
        }
    }

    bool Init(std::string* err)
    {
        // nvcuda.dll is delay-loaded by the plugin so it still loads on machines
        // without an NVIDIA driver: check it exists before the first cu* call.
        if (!GetModuleHandleA("nvcuda.dll") && !LoadLibraryA("nvcuda.dll"))
            return Fail(err, "nvcuda.dll not found (no NVIDIA driver)");
        if (cuInit(0) != CUDA_SUCCESS)
            return Fail(err, "cuInit failed (no NVIDIA driver?)");
        if (cudaFree(nullptr) != cudaSuccess) // creates/attaches the primary context
            return Fail(err, "CUDA runtime init failed (no supported NVIDIA GPU?)");
        if (cuCtxGetCurrent(&ctx_) != CUDA_SUCCESS || !ctx_)
            return Fail(err, "no current CUDA context");
        for (auto& e : ev_)
            if (cudaEventCreate(&e) != cudaSuccess)
                return Fail(err, "cudaEventCreate failed");
        return true;
    }

    bool ProcessPair(const HostImage& A, const HostImage& B, double a, const HostFlow& fa, const HostFlow& fb,
                     void* out, size_t outRowBytes, const Settings& s, Stats* stats, std::string* err) override
    {
        if (B.width != A.width || B.height != A.height || B.isFloat != A.isFloat)
            return Fail(err, "frame size/format mismatch");
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = Upload(A, 0, err) && Upload(B, 1, err) && EnsureOut(A.width, A.height, err) &&
                  UploadFlow(fa, flowA_, err) && UploadFlow(fb, flowB_, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            cudaEventRecord(ev_[2], nullptr);
            ok = BlurPair(a, s, err) && Finish(stats, err);
        }
        ok = ok && Download(out, outRowBytes, A.argb, A.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool Process(const HostImage& cur, const HostFlow& v, void* out, size_t outRowBytes, const Settings& s,
                 Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = Upload(cur, 0, err) && EnsureOut(cur.width, cur.height, err) && UploadFlow(v, flowA_, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            cudaEventRecord(ev_[2], nullptr);
            ok = BlurSingle(s, err) && Finish(stats, err);
        }
        ok = ok && Download(out, outRowBytes, cur.argb, cur.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessSamples(const HostImage* frames, int nFrames, const TimeSample* samples, int nSamples, void* out,
                        size_t outRowBytes, bool showVectors, Stats* stats, std::string* err) override
    {
        if (nFrames < 1 || nFrames > kMaxFrames || nSamples < 1)
            return Fail(err, "bad sample setup");
        for (int i = 1; i < nFrames; ++i)
            if (frames[i].width != frames[0].width || frames[i].height != frames[0].height ||
                frames[i].isFloat != frames[0].isFloat)
                return Fail(err, "frame size/format mismatch");
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        const int W = frames[0].width, H = frames[0].height;
        bool ok = EnsureOut(W, H, err);
        for (int i = 0; i < nFrames && ok; ++i)
            ok = Upload(frames[i], i, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            cudaEventRecord(ev_[2], nullptr);
        }
        const dim3 b(16, 16), g = Grid(W, H, b);
        for (int i = 0; i < nSamples && ok; ++i)
        {
            const TimeSample& s = samples[i];
            const HostField& f = s.field;
            if (s.frame0 < 0 || s.frame0 >= nFrames || s.frame1 < 0 || s.frame1 >= nFrames || !f.flow || !f.mask ||
                f.width < 2 || f.height < 2)
            {
                ok = Fail(err, "bad time sample");
                break;
            }
            ok = fieldFlow_.Ensure(f.width, f.height, 16) && fieldMask_.Ensure(f.width, f.height, 4);
            if (!ok)
            {
                Fail(err, "out of GPU memory (field)");
                break;
            }
            ok = cudaMemcpy2D(fieldFlow_.ptr, fieldFlow_.pitch, f.flow, f.flowRowBytes, (size_t)f.width * 16, f.height,
                              cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy2D(fieldMask_.ptr, fieldMask_.pitch, f.mask, f.maskRowBytes, (size_t)f.width * 4, f.height,
                              cudaMemcpyHostToDevice) == cudaSuccess;
            if (!ok)
            {
                Fail(err, "field upload failed");
                break;
            }
            const DevBuf& a = frame_[s.frame0];
            const DevBuf& c = frame_[s.frame1];
            KRifeAccum<<<g, b>>>((const char*)a.ptr, a.pitch, (const char*)c.ptr, c.pitch, W, H,
                                 (const float4*)fieldFlow_.ptr, fieldFlow_.pitch, (const float*)fieldMask_.ptr,
                                 fieldMask_.pitch, f.width, f.height, showVectors ? 1.0f : s.weight, (char*)out_.ptr,
                                 out_.pitch, i == 0 ? 1 : 0, showVectors ? 1 : 0);
            if (showVectors)
                break;
        }
        ok = ok && Finish(stats, err);
        ok = ok && Download(out, outRowBytes, frames[0].argb, frames[0].isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessPairGpuFlow(const HostImage& A, const HostImage& B, double a, const FlowSettings& fs, void* out,
                            size_t outRowBytes, const Settings& s, Stats* stats, std::string* err) override
    {
        if (B.width != A.width || B.height != A.height || B.isFloat != A.isFloat)
            return Fail(err, "frame size/format mismatch");
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = Upload(A, 0, err) && Upload(B, 1, err) && EnsureOut(A.width, A.height, err) &&
                  SetupFlow(A.width, A.height, fs, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            ok = dis_.Prepare(0, (const char*)frame_[0].ptr, frame_[0].pitch, A.width, A.height, err) &&
                 dis_.Prepare(1, (const char*)frame_[1].ptr, frame_[1].pitch, A.width, A.height, err) &&
                 // A's motion forward (A->B) and B's motion forward (minus B->A)
                 dis_.Compute(0, 1, fs.refineEdges, 1.0f, false, (char*)flowA_.ptr, flowA_.pitch, err) &&
                 dis_.Compute(1, 0, fs.refineEdges, -1.0f, false, (char*)flowB_.ptr, flowB_.pitch, err);
            cudaEventRecord(ev_[2], nullptr);
            ok = ok && BlurPair(a, s, err) && Finish(stats, err);
        }
        ok = ok && Download(out, outRowBytes, A.argb, A.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessGpuFlow(const HostImage* prev, const HostImage& cur, const HostImage* next, const FlowSettings& fs,
                        void* out, size_t outRowBytes, const Settings& s, Stats* stats, std::string* err) override
    {
        for (const HostImage* o : {prev, next})
            if (o && (o->width != cur.width || o->height != cur.height || o->isFloat != cur.isFloat))
                return Fail(err, "frame size/format mismatch");
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        // slots: 0 = cur, 1 = next, 2 = prev
        bool ok = Upload(cur, 0, err) && (!next || Upload(*next, 1, err)) && (!prev || Upload(*prev, 2, err)) &&
                  EnsureOut(cur.width, cur.height, err) && SetupFlow(cur.width, cur.height, fs, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            ok = dis_.Prepare(0, (const char*)frame_[0].ptr, frame_[0].pitch, cur.width, cur.height, err) &&
                 (!next || dis_.Prepare(1, (const char*)frame_[1].ptr, frame_[1].pitch, cur.width, cur.height, err)) &&
                 (!prev || dis_.Prepare(2, (const char*)frame_[2].ptr, frame_[2].pitch, cur.width, cur.height, err));
            // velocity: mean of (cur->next) and -(cur->prev), whichever exist
            const float k = next && prev ? 0.5f : 1.0f;
            if (ok && !next && !prev)
                cudaMemset2DAsync(flowA_.ptr, flowA_.pitch, 0, (size_t)flowA_.w * sizeof(float2), flowA_.h);
            if (ok && next)
                ok = dis_.Compute(0, 1, fs.refineEdges, k, false, (char*)flowA_.ptr, flowA_.pitch, err);
            if (ok && prev)
                ok = dis_.Compute(0, 2, fs.refineEdges, -k, next != nullptr, (char*)flowA_.ptr, flowA_.pitch, err);
            cudaEventRecord(ev_[2], nullptr);
            ok = ok && BlurSingle(s, err) && Finish(stats, err);
        }
        ok = ok && Download(out, outRowBytes, cur.argb, cur.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool EstimateFlowPairGpu(const HostImage& A, const HostImage& B, const FlowSettings& fs, float* fab, float* fba,
                             int* fw, int* fh, Stats* stats, std::string* err) override
    {
        if (B.width != A.width || B.height != A.height || B.isFloat != A.isFloat)
            return Fail(err, "frame size/format mismatch");
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = Upload(A, 0, err) && Upload(B, 1, err) && SetupFlow(A.width, A.height, fs, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            ok = dis_.Prepare(0, (const char*)frame_[0].ptr, frame_[0].pitch, A.width, A.height, err) &&
                 dis_.Prepare(1, (const char*)frame_[1].ptr, frame_[1].pitch, A.width, A.height, err) &&
                 dis_.Compute(0, 1, fs.refineEdges, 1.0f, false, (char*)flowA_.ptr, flowA_.pitch, err) &&
                 dis_.Compute(1, 0, fs.refineEdges, 1.0f, false, (char*)flowB_.ptr, flowB_.pitch, err);
            cudaEventRecord(ev_[2], nullptr);
            ok = ok && Finish(stats, err);
        }
        if (ok)
        {
            const size_t row = (size_t)flowA_.w * sizeof(float2);
            ok = (cudaMemcpy2D(fab, row, flowA_.ptr, flowA_.pitch, row, flowA_.h, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy2D(fba, row, flowB_.ptr, flowB_.pitch, row, flowB_.h, cudaMemcpyDeviceToHost) == cudaSuccess) ||
                 Fail(err, "flow download failed");
            *fw = flowA_.w;
            *fh = flowA_.h;
        }
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

private:
    static constexpr int kMaxFrames = 4;

    // Analysis size / pyramid for this frame size, and flow buffers to match.
    bool SetupFlow(int W, int H, const FlowSettings& fs, std::string* err)
    {
        if (!dis_.Setup(W, H, fs.flowWidth, fs.preset))
            return Fail(err, "frame too small for GPU motion");
        if (!flowA_.Ensure(dis_.Fw(), dis_.Fh(), sizeof(float2)) || !flowB_.Ensure(dis_.Fw(), dis_.Fh(), sizeof(float2)))
            return Fail(err, "out of GPU memory (flow)");
        return true;
    }

    // Two-frame blur of frame_[0] / frame_[1] with flowA_ / flowB_ into out_.
    bool BlurPair(double a, const Settings& s, std::string* err)
    {
        MipSet ma = {}, mb = {};
        if (!BuildMips(frame_[0], mipA_, s, &ma, err) || !BuildMips(frame_[1], mipB_, s, &mb, err))
            return false;
        const dim3 b(16, 16), g = Grid(out_.w, out_.h, b);
        KBlurPair<<<g, b>>>(ma, mb, Field(flowA_), Field(flowB_), (char*)out_.ptr, out_.pitch,
                            (float)std::max(0.0, s.halfWidth), (float)std::clamp(a, 0.0, 1.0), s.showVectors ? 1 : 0,
                            std::max(2, s.maxSamples), (float)std::max(1, s.mipThreshold));
        return true;
    }

    // Single-frame blur of frame_[0] with flowA_ into out_.
    bool BlurSingle(const Settings& s, std::string* err)
    {
        MipSet ms = {};
        if (!BuildMips(frame_[0], mipA_, s, &ms, err))
            return false;
        const dim3 b(16, 16), g = Grid(out_.w, out_.h, b);
        KBlur<<<g, b>>>(ms, Field(flowA_), (char*)out_.ptr, out_.pitch, (float)std::max(0.0, s.halfWidth),
                        (float)s.centerOffset, s.showVectors ? 1 : 0, std::max(2, s.maxSamples),
                        (float)std::max(1, s.mipThreshold));
        return true;
    }

    static bool Fail(std::string* err, const char* msg)
    {
        if (err)
            *err = msg;
        return false;
    }

    static FlowField Field(const DevBuf& f) { return {(const float2*)f.ptr, f.pitch, f.w, f.h}; }

    // Host frame -> float device frame in frame_[slot].
    bool Upload(const HostImage& img, int slot, std::string* err)
    {
        if (!img.data || img.width <= 0 || img.height <= 0)
            return Fail(err, "empty host image");
        DevBuf& f = frame_[slot];
        if (!f.Ensure(img.width, img.height, 16))
            return Fail(err, "out of GPU memory (upload)");
        if (img.isFloat) // already float BGRA: straight copy, nothing clamped or quantised
            return cudaMemcpy2D(f.ptr, f.pitch, img.data, img.rowBytes, (size_t)img.width * 16, img.height,
                                cudaMemcpyHostToDevice) == cudaSuccess ||
                   Fail(err, "upload failed");
        if (!stage_[slot].Ensure(img.width, img.height, 4))
            return Fail(err, "out of GPU memory (upload)");
        if (cudaMemcpy2D(stage_[slot].ptr, stage_[slot].pitch, img.data, img.rowBytes, (size_t)img.width * 4,
                         img.height, cudaMemcpyHostToDevice) != cudaSuccess)
            return Fail(err, "upload failed");
        const dim3 b(16, 16);
        KU8ToF4<<<Grid(img.width, img.height, b), b>>>((const unsigned char*)stage_[slot].ptr, stage_[slot].pitch,
                                                        (char*)f.ptr, f.pitch, img.width, img.height,
                                                        img.argb ? 1 : 0);
        return true;
    }

    bool UploadFlow(const HostFlow& f, DevBuf& dst, std::string* err)
    {
        if (!f.data || f.width <= 0 || f.height <= 0)
            return Fail(err, "empty flow");
        if (!dst.Ensure(f.width, f.height, sizeof(float2)))
            return Fail(err, "out of GPU memory (flow)");
        if (cudaMemcpy2D(dst.ptr, dst.pitch, f.data, f.rowBytes, (size_t)f.width * sizeof(float2), f.height,
                         cudaMemcpyHostToDevice) != cudaSuccess)
            return Fail(err, "flow upload failed");
        return true;
    }

    bool EnsureOut(int w, int h, std::string* err)
    {
        if (!out_.Ensure(w, h, 16) || !stageOut_.Ensure(w, h, 4))
            return Fail(err, "out of GPU memory (output)");
        return true;
    }

    bool Download(void* out, size_t outRowBytes, bool argb, bool isFloat, std::string* err)
    {
        if (isFloat)
            return cudaMemcpy2D(out, outRowBytes, out_.ptr, out_.pitch, (size_t)out_.w * 16, out_.h,
                                cudaMemcpyDeviceToHost) == cudaSuccess ||
                   Fail(err, "download failed");
        const dim3 b(16, 16);
        KF4ToU8<<<Grid(out_.w, out_.h, b), b>>>((const char*)out_.ptr, out_.pitch, (unsigned char*)stageOut_.ptr,
                                                stageOut_.pitch, out_.w, out_.h, argb ? 1 : 0);
        if (cudaMemcpy2D(out, outRowBytes, stageOut_.ptr, stageOut_.pitch, (size_t)out_.w * 4, out_.h,
                         cudaMemcpyDeviceToHost) != cudaSuccess)
            return Fail(err, "download failed");
        return true;
    }

    // Mip chain of f into bufs (level 0 = f itself).
    bool BuildMips(const DevBuf& f, DevBuf* bufs, const Settings& s, MipSet* ms, std::string* err)
    {
        const dim3 b(16, 16);
        *ms = {};
        ms->level[0] = {(const char*)f.ptr, f.pitch, f.w, f.h};
        ms->top = 0;
        if (s.mipThreshold <= 0)
            return true;
        for (int l = 1; l <= std::min(3, s.maxMipLevel); ++l)
        {
            const Mip& src = ms->level[l - 1];
            if (src.w < 4 || src.h < 4)
                break;
            const int dw = (src.w + 1) / 2, dh = (src.h + 1) / 2;
            if (!bufs[l - 1].Ensure(dw, dh, 16))
                return Fail(err, "out of GPU memory (mips)");
            KDown<<<Grid(dw, dh, b), b>>>(src.ptr, src.pitch, src.w, src.h, (char*)bufs[l - 1].ptr, bufs[l - 1].pitch,
                                          dw, dh);
            ms->level[l] = {(const char*)bufs[l - 1].ptr, bufs[l - 1].pitch, dw, dh};
            ms->top = l;
        }
        return true;
    }

    bool Finish(Stats* stats, std::string* err)
    {
        cudaEventRecord(ev_[1], nullptr);
        const cudaError_t e = cudaEventSynchronize(ev_[1]);
        const cudaError_t le = cudaGetLastError();
        if (e != cudaSuccess || le != cudaSuccess)
        {
            if (err)
                *err = std::string("CUDA error: ") + cudaGetErrorString(e != cudaSuccess ? e : le);
            return false;
        }
        if (stats)
        {
            float t = 0, tf = 0;
            cudaEventElapsedTime(&t, ev_[0], ev_[1]);
            cudaEventElapsedTime(&tf, ev_[0], ev_[2]);
            stats->gpuMs = t;
            stats->flowMs = tf;
        }
        return true;
    }

    std::mutex mu_;
    CUcontext ctx_ = nullptr;
    DevBuf frame_[kMaxFrames], out_, stage_[kMaxFrames], stageOut_; // float frames, output, 8-bit staging
    DevBuf fieldFlow_, fieldMask_;                                   // Quality High: one sample's field
    DevBuf flowA_, flowB_, mipA_[3], mipB_[3];
    cudaEvent_t ev_[3] = {}; // start, end, end of motion estimation
    dis::DisFlow dis_;       // GPU motion (the *GpuFlow calls)
};

} // namespace

std::unique_ptr<Engine> Engine::Create(std::string* err)
{
    auto e = std::make_unique<EngineImpl>();
    if (!e->Init(err))
        return nullptr;
    return e;
}

} // namespace oe::gpu
