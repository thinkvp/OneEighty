// GPU optical-flow motion blur. See GpuMotionBlur.h.
//
// Pipeline per output frame:
//   1. cur/prev/next (BGRA float or half) -> 8-bit luma at flowWidth   (CUDA)
//   2. motion cur->next and cur->prev                                 (NVOF hardware)
//   3. v = (fwd - bwd) / 2, in full-res pixels per source frame        (CUDA)
//   4. mip chain of cur (1/2, 1/4, 1/8, area filtered)                 (CUDA)
//   5. each pixel: average cur along +/- halfWidth * v, long smears     (CUDA)
//      sampled from a mip level so cost stays bounded

#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#include "GpuMotionBlur.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>

namespace oe::gpu
{
namespace
{

// ------------------------------------------------------------ pixel access
struct alignas(8) Half4
{
    __half b, g, r, a;
};

template <bool HALF>
__device__ __forceinline__ float4 Load(const char* base, size_t pitch, int x, int y)
{
    const char* row = base + (size_t)y * pitch;
    if constexpr (HALF)
    {
        const Half4 p = reinterpret_cast<const Half4*>(row)[x];
        return make_float4(__half2float(p.b), __half2float(p.g), __half2float(p.r), __half2float(p.a));
    }
    else
        return reinterpret_cast<const float4*>(row)[x];
}

template <bool HALF>
__device__ __forceinline__ void Store(char* base, size_t pitch, int x, int y, float4 v)
{
    char* row = base + (size_t)y * pitch;
    if constexpr (HALF)
    {
        Half4 p;
        p.b = __float2half(v.x);
        p.g = __float2half(v.y);
        p.r = __float2half(v.z);
        p.a = __float2half(v.w);
        reinterpret_cast<Half4*>(row)[x] = p;
    }
    else
        reinterpret_cast<float4*>(row)[x] = v;
}

template <bool HALF>
__device__ __forceinline__ float4 Bilinear(const char* base, size_t pitch, int w, int h, float sx, float sy)
{
    sx = fminf(fmaxf(sx, 0.0f), (float)(w - 1));
    sy = fminf(fmaxf(sy, 0.0f), (float)(h - 1));
    const int x0 = min((int)sx, max(0, w - 2)), y0 = min((int)sy, max(0, h - 2));
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float4 a = Load<HALF>(base, pitch, x0, y0), b = Load<HALF>(base, pitch, x1, y0);
    const float4 c = Load<HALF>(base, pitch, x0, y1), d = Load<HALF>(base, pitch, x1, y1);
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    return make_float4(w00 * a.x + w10 * b.x + w01 * c.x + w11 * d.x, w00 * a.y + w10 * b.y + w01 * c.y + w11 * d.y,
                       w00 * a.z + w10 * b.z + w01 * c.z + w11 * d.z, w00 * a.w + w10 * b.w + w01 * c.w + w11 * d.w);
}

// ------------------------------------------------------------------ kernels
// Area-average BGRA (0..1) down to an 8-bit luma image for the flow engine.
template <bool HALF>
__global__ void KGray(const char* src, size_t sp, int W, int H, unsigned char* dst, size_t dp, int w, int h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h)
        return;
    const int x0 = (int)((long long)i * W / w), x1 = max(x0 + 1, (int)((long long)(i + 1) * W / w));
    const int y0 = (int)((long long)j * H / h), y1 = max(y0 + 1, (int)((long long)(j + 1) * H / h));
    float acc = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
        {
            const float4 p = Load<HALF>(src, sp, x, y); // x=B y=G z=R
            acc += 0.114f * p.x + 0.587f * p.y + 0.299f * p.z;
        }
    acc /= (float)((x1 - x0) * (y1 - y0));
    dst[(size_t)j * dp + i] = (unsigned char)(fminf(fmaxf(acc, 0.0f), 1.0f) * 255.0f + 0.5f);
}

// NVOF S10.5 vectors -> velocity in full-res pixels per source frame.
__global__ void KCombineFlow(const short2* fwd, size_t fp, const short2* bwd, size_t bp, float2* out, size_t op, int fw,
                             int fh, float toFull)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fw || y >= fh)
        return;
    float vx = 0, vy = 0;
    int n = 0;
    if (fwd)
    {
        const short2 f = reinterpret_cast<const short2*>(reinterpret_cast<const char*>(fwd) + y * fp)[x];
        vx += f.x / 32.0f;
        vy += f.y / 32.0f;
        ++n;
    }
    if (bwd)
    {
        const short2 b = reinterpret_cast<const short2*>(reinterpret_cast<const char*>(bwd) + y * bp)[x];
        vx -= b.x / 32.0f;
        vy -= b.y / 32.0f;
        ++n;
    }
    const float k = n ? toFull / n : 0.0f;
    reinterpret_cast<float2*>(reinterpret_cast<char*>(out) + y * op)[x] = make_float2(vx * k, vy * k);
}

// 2x2 box downsample into a float4 mip level.
template <bool HALF>
__global__ void KDown(const char* src, size_t sp, int sw, int sh, char* dst, size_t dp, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;
    const int x0 = min(2 * x, sw - 1), x1 = min(2 * x + 1, sw - 1), y0 = min(2 * y, sh - 1), y1 = min(2 * y + 1, sh - 1);
    const float4 a = Load<HALF>(src, sp, x0, y0), b = Load<HALF>(src, sp, x1, y0);
    const float4 c = Load<HALF>(src, sp, x0, y1), d = Load<HALF>(src, sp, x1, y1);
    Store<false>(dst, dp, x, y,
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
    Mip level[4]; // level[0] = cur itself (may be half), 1..3 float4
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
template <bool HALF_IN>
__device__ float4 Gather(const MipSet& mips, int x, int y, float vx, float vy, float hw, float c0, int maxS, float mipT)
{
    const Mip& L0 = mips.level[0];
    const float speed = sqrtf(vx * vx + vy * vy);
    const float smear = 2.0f * hw * speed;
    if (smear <= 1.0f && fabsf(c0) * speed < 0.5f)
        return Load<HALF_IN>(L0.ptr, L0.pitch, x, y);
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
        const float4 p = lvl == 0 ? Bilinear<HALF_IN>(M.ptr, M.pitch, M.w, M.h, sx, sy)
                                  : Bilinear<false>(M.ptr, M.pitch, M.w, M.h, sx, sy);
        acc.x += p.x;
        acc.y += p.y;
        acc.z += p.z;
        acc.w += p.w;
    }
    const float inv = 1.0f / n;
    return make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

template <bool HALF_IN>
__device__ float4 VectorPixel(const Mip& L0, int x, int y, float vx, float vy)
{
    const float4 p = Load<HALF_IN>(L0.ptr, L0.pitch, x, y); // x=B y=G z=R
    const float g = fminf(fmaxf(0.114f * p.x + 0.587f * p.y + 0.299f * p.z, 0.0f), 1.0f);
    const float3 c = VectorColour(vx, vy, fmaxf(4.0f, 0.01f * L0.w));
    const float k = 0.65f * (0.4f + 0.6f * g);
    return make_float4(0.35f * g + k * c.z, 0.35f * g + k * c.y, 0.35f * g + k * c.x, p.w);
}

// Single frame. mode 0: blur, 1: motion-vector view. c0 = output time offset (source frames).
template <bool HALF_IN, bool HALF_OUT>
__global__ void KBlur(MipSet mips, FlowField flow, char* out, size_t outP, float hw, float c0, int mode, int maxS,
                      float mipT)
{
    const Mip& L0 = mips.level[0];
    const int W = L0.w, H = L0.h;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    if (hw <= 0.0f && c0 == 0.0f && mode == 0) // pure copy / format conversion; flow may not exist
    {
        Store<HALF_OUT>(out, outP, x, y, Load<HALF_IN>(L0.ptr, L0.pitch, x, y));
        return;
    }
    const float2 v = SampleFlow(flow, W, H, x, y);
    Store<HALF_OUT>(out, outP, x, y,
                    mode == 1 ? VectorPixel<HALF_IN>(L0, x, y, v.x, v.y)
                              : Gather<HALF_IN>(mips, x, y, v.x, v.y, hw, c0, maxS, mipT));
}

// Two-frame interpolation: output time lies a (0..1) of the way from frame A
// to frame B. Blur A at +a and B at -(1-a), each with its own motion, and
// blend by closeness.
template <bool HALF_IN, bool HALF_OUT>
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
        Store<HALF_OUT>(out, outP, x, y, VectorPixel<HALF_IN>(L0, x, y, va.x, va.y));
        return;
    }
    float4 r = Gather<HALF_IN>(ma, x, y, va.x, va.y, hw, a, maxS, mipT);
    if (a > 1e-3f)
    {
        const float2 vb = SampleFlow(fb, W, H, x, y);
        const float4 q = Gather<HALF_IN>(mb, x, y, vb.x, vb.y, hw, a - 1.0f, maxS, mipT);
        r = make_float4((1 - a) * r.x + a * q.x, (1 - a) * r.y + a * q.y, (1 - a) * r.z + a * q.z,
                        (1 - a) * r.w + a * q.w);
    }
    Store<HALF_OUT>(out, outP, x, y, r);
}

// NVOF S10.5 vectors * sign -> velocity in full-res pixels per source frame.
__global__ void KScaleFlow(const short2* src, size_t sp, float2* out, size_t op, int fw, int fh, float k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fw || y >= fh)
        return;
    const short2 f = reinterpret_cast<const short2*>(reinterpret_cast<const char*>(src) + y * sp)[x];
    reinterpret_cast<float2*>(reinterpret_cast<char*>(out) + y * op)[x] = make_float2(f.x * k / 32.0f, f.y * k / 32.0f);
}

// Sum of |a - b| over a 64x64 sample grid (diagnostics).
template <bool HALF_IN>
__global__ void KSampledDiff(const char* a, size_t ap, const char* b, size_t bp, int W, int H, float* sum)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 64 || j >= 64)
        return;
    const int x = (int)((i + 0.5f) * W / 64), y = (int)((j + 0.5f) * H / 64);
    const float4 p = Load<HALF_IN>(a, ap, x, y), q = Load<HALF_IN>(b, bp, x, y);
    atomicAdd(sum, fabsf(p.x - q.x) + fabsf(p.y - q.y) + fabsf(p.z - q.z));
}

// 8-bit -> float4 (B,G,R,A order internally). argb: source bytes are A,R,G,B.
__global__ void KU8ToF4(const unsigned char* src, size_t sp, char* dst, size_t dp, int w, int h, int argb = 0)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const unsigned char* p = src + (size_t)y * sp + 4 * x;
    const float k = 1.0f / 255.0f;
    Store<false>(dst, dp, x, y,
                 argb ? make_float4(p[3] * k, p[2] * k, p[1] * k, p[0] * k)
                      : make_float4(p[0] * k, p[1] * k, p[2] * k, p[3] * k));
}

__device__ __forceinline__ unsigned char ToU8(float v) { return (unsigned char)(fminf(fmaxf(v, 0.f), 1.f) * 255.f + 0.5f); }

__global__ void KF4ToU8(const char* src, size_t sp, unsigned char* dst, size_t dp, int w, int h, int argb = 0)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float4 v = Load<false>(src, sp, x, y); // x=B y=G z=R w=A
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
            DestroyOF();
            flow_.Release();
            flowB_.Release();
            for (auto& m : mip_)
                m.Release();
            for (auto& m : mipB_)
                m.Release();
            if (diffSum_)
                cudaFree(diffSum_);
            for (auto& b : stage_)
                b.Release();
            stageOut_.Release();
            for (auto& f : hostF_)
                if (f.ptr)
                    cudaFree(f.ptr);
            if (hostOut_.ptr)
                cudaFree(hostOut_.ptr);
            for (auto& e : ev_)
                if (e)
                    cudaEventDestroy(e);
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
        }
        if (lib_)
            FreeLibrary(lib_);
    }

    bool Init(void* cuContext, std::string* err)
    {
        // nvcuda.dll is delay-loaded by the plugin so it still loads on machines
        // without an NVIDIA driver: check it exists before the first cu* call.
        if (!GetModuleHandleA("nvcuda.dll") && !LoadLibraryA("nvcuda.dll"))
            return Fail(err, "nvcuda.dll not found (no NVIDIA driver)");
        ofs_.reserve(kMaxOFStates + 1);
        if (cuInit(0) != CUDA_SUCCESS)
            return Fail(err, "cuInit failed (no NVIDIA driver?)");
        if (cuContext)
            ctx_ = static_cast<CUcontext>(cuContext);
        else
        {
            if (cudaFree(nullptr) != cudaSuccess) // creates/attaches the primary context
                return Fail(err, "CUDA runtime init failed");
            if (cuCtxGetCurrent(&ctx_) != CUDA_SUCCESS || !ctx_)
                return Fail(err, "no current CUDA context");
        }
        lib_ = LoadLibraryA("nvofapi64.dll");
        if (!lib_)
            return Fail(err, "nvofapi64.dll not found (NVIDIA driver without optical flow)");
        using CreateFn = NV_OF_STATUS(NVOFAPI*)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);
        auto create = reinterpret_cast<CreateFn>(GetProcAddress(lib_, "NvOFAPICreateInstanceCuda"));
        if (!create)
            return Fail(err, "NvOFAPICreateInstanceCuda missing");
        if (create(NV_OF_API_VERSION, &of_) != NV_OF_SUCCESS)
            return Fail(err, "NvOFAPICreateInstanceCuda failed (API version)");

        // Probe the hardware once so Create() fails early on unsupported GPUs.
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        NvOFHandle probe = nullptr;
        const NV_OF_STATUS st = of_.nvCreateOpticalFlowCuda(ctx_, &probe);
        if (probe)
            of_.nvOFDestroy(probe);
        for (auto& e : ev_)
            cudaEventCreate(&e);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        if (st != NV_OF_SUCCESS)
            return Fail(err, "optical-flow hardware not available on this GPU");
        return true;
    }

    bool Process(const Frame* prev, const Frame& cur, const Frame* next, const Frame& out, const Settings& s,
                 void* stream, Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        const bool ok = ProcessLocked(prev, cur, next, out, s, static_cast<cudaStream_t>(stream), stats, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessPair(const Frame& A, const Frame& B, double a, const Frame& out, const Settings& s, void* stream,
                     Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        const bool ok = ProcessPairLocked(A, B, a, out, s, static_cast<cudaStream_t>(stream), stats, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    double SampledDiff(const Frame& x, const Frame& y, void* stream) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (x.half != y.half || x.width != y.width || x.height != y.height || cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return -1.0;
        const cudaStream_t st = static_cast<cudaStream_t>(stream);
        float sum = -1.0f;
        if (diffSum_ || cudaMalloc(&diffSum_, sizeof(float)) == cudaSuccess)
        {
            cudaMemsetAsync(diffSum_, 0, sizeof(float), st);
            const dim3 b(16, 16), g(4, 4);
            if (x.half)
                KSampledDiff<true><<<g, b, 0, st>>>((const char*)x.ptr, x.pitch, (const char*)y.ptr, y.pitch, x.width,
                                                    x.height, diffSum_);
            else
                KSampledDiff<false><<<g, b, 0, st>>>((const char*)x.ptr, x.pitch, (const char*)y.ptr, y.pitch,
                                                     x.width, x.height, diffSum_);
            cudaMemcpyAsync(&sum, diffSum_, sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
        }
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return sum < 0 ? -1.0 : sum / (64.0 * 64.0 * 3.0) * 255.0;
    }

    bool ProcessHostPair(const HostImage& A, const HostImage& B, double a, void* out, size_t outRowBytes,
                         const Settings& s, Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = UploadHost(A, 0, err) && UploadHost(B, 1, err) && EnsureHostOut(A.width, A.height, err);
        ok = ok && ProcessPairLocked(hostF_[0], hostF_[1], a, hostOut_, s, nullptr, stats, err);
        ok = ok && DownloadHost(out, outRowBytes, A.argb, A.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessHost(const HostImage* prev, const HostImage& cur, const HostImage* next, void* out,
                     size_t outRowBytes, const Settings& s, Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = UploadHost(cur, 1, err) && EnsureHostOut(cur.width, cur.height, err);
        const bool hp = ok && prev && UploadHost(*prev, 0, err);
        const bool hn = ok && next && UploadHost(*next, 2, err);
        ok = ok && ProcessLocked(hp ? &hostF_[0] : nullptr, hostF_[1], hn ? &hostF_[2] : nullptr, hostOut_, s,
                                 nullptr, stats, err);
        ok = ok && DownloadHost(out, outRowBytes, cur.argb, cur.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessHostPairFlow(const HostImage& A, const HostImage& B, double a, const HostFlow& fa, const HostFlow& fb,
                             void* out, size_t outRowBytes, const Settings& s, Stats* stats,
                             std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = UploadHost(A, 0, err) && UploadHost(B, 1, err) && EnsureHostOut(A.width, A.height, err) &&
                  UploadFlow(fa, flow_, err) && UploadFlow(fb, flowB_, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            cudaEventRecord(ev_[1], nullptr);
            MipSet ma = {}, mb = {};
            ok = BuildMips(hostF_[0], mip_, s, nullptr, &ma, err) && BuildMips(hostF_[1], mipB_, s, nullptr, &mb, err);
            if (ok)
            {
                const dim3 b(16, 16), g = Grid(A.width, A.height, b);
                const FlowField ffa = {(const float2*)flow_.ptr, flow_.pitch, flow_.w, flow_.h};
                const FlowField ffb = {(const float2*)flowB_.ptr, flowB_.pitch, flowB_.w, flowB_.h};
                KBlurPair<false, false><<<g, b>>>(ma, mb, ffa, ffb, (char*)hostOut_.ptr, hostOut_.pitch,
                                                  (float)std::max(0.0, s.halfWidth), (float)std::clamp(a, 0.0, 1.0),
                                                  s.showVectors ? 1 : 0, std::max(2, s.maxSamples),
                                                  (float)std::max(1, s.mipThreshold));
                ok = Finish(nullptr, stats, err, true);
            }
        }
        ok = ok && DownloadHost(out, outRowBytes, A.argb, A.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool ProcessHostFlow(const HostImage& cur, const HostFlow& v, void* out, size_t outRowBytes, const Settings& s,
                         Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cuCtxPushCurrent(ctx_) != CUDA_SUCCESS)
            return Fail(err, "cuCtxPushCurrent failed");
        bool ok = UploadHost(cur, 1, err) && EnsureHostOut(cur.width, cur.height, err) && UploadFlow(v, flow_, err);
        if (ok)
        {
            cudaEventRecord(ev_[0], nullptr);
            cudaEventRecord(ev_[1], nullptr);
            MipSet ms = {};
            ok = BuildMips(hostF_[1], mip_, s, nullptr, &ms, err);
            if (ok)
            {
                LaunchBlur(ms, false, hostOut_, (float)s.halfWidth, (float)s.centerOffset, s.showVectors ? 1 : 0, s,
                           nullptr);
                ok = Finish(nullptr, stats, err, true);
            }
        }
        ok = ok && DownloadHost(out, outRowBytes, cur.argb, cur.isFloat, err);
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok;
    }

    bool Upload8(const void* host, int w, int h, Frame* dev, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        cuCtxPushCurrent(ctx_);
        DevBuf tmp;
        bool ok = tmp.Ensure(w, h, 4) && cudaMallocPitch(&dev->ptr, &dev->pitch, (size_t)w * 16, h) == cudaSuccess;
        if (ok)
        {
            dev->width = w;
            dev->height = h;
            dev->half = false;
            ok = cudaMemcpy2D(tmp.ptr, tmp.pitch, host, (size_t)w * 4, (size_t)w * 4, h, cudaMemcpyHostToDevice) ==
                 cudaSuccess;
            const dim3 b(16, 16);
            KU8ToF4<<<Grid(w, h, b), b>>>((const unsigned char*)tmp.ptr, tmp.pitch, (char*)dev->ptr, dev->pitch, w, h);
            ok = ok && cudaDeviceSynchronize() == cudaSuccess;
        }
        tmp.Release();
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok || Fail(err, "upload failed");
    }

    bool Download8(const Frame& dev, void* host, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        cuCtxPushCurrent(ctx_);
        DevBuf tmp;
        bool ok = tmp.Ensure(dev.width, dev.height, 4);
        if (ok)
        {
            const dim3 b(16, 16);
            KF4ToU8<<<Grid(dev.width, dev.height, b), b>>>((const char*)dev.ptr, dev.pitch, (unsigned char*)tmp.ptr,
                                                            tmp.pitch, dev.width, dev.height);
            ok = cudaMemcpy2D(host, (size_t)dev.width * 4, tmp.ptr, tmp.pitch, (size_t)dev.width * 4, dev.height,
                              cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        tmp.Release();
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        return ok || Fail(err, "download failed");
    }

    void Free(Frame* dev) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        cuCtxPushCurrent(ctx_);
        if (dev->ptr)
            cudaFree(dev->ptr);
        dev->ptr = nullptr;
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
    }

private:
    static bool Fail(std::string* err, const char* msg)
    {
        if (err)
            *err = msg;
        return false;
    }

    // One NVOF session per analysis size. Premiere renders at several sizes
    // (full / 1/2 / 1/4 playback resolution, thumbnails), so keep a few alive
    // instead of re-initialising the hardware on every size change.
    struct OFState
    {
        NvOFHandle h = nullptr;
        int w = 0, ht = 0, grid = 0, perf = 0, outW = 0, outH = 0;
        cudaStream_t stream = nullptr;
        NvOFGPUBufferHandle in[3] = {}, out[2] = {};
        CUdeviceptr inPtr[3] = {}, outPtr[2] = {};
        uint32_t inPitch[3] = {}, outPitch[2] = {};
        unsigned long long lastUse = 0;
    };
    static constexpr size_t kMaxOFStates = 4;

    void DestroyState(OFState& a)
    {
        for (auto& b : a.in)
            if (b)
                of_.nvOFDestroyGPUBufferCuda(b), b = nullptr;
        for (auto& b : a.out)
            if (b)
                of_.nvOFDestroyGPUBufferCuda(b), b = nullptr;
        if (a.h)
            of_.nvOFDestroy(a.h), a.h = nullptr;
    }

    void DestroyOF()
    {
        for (auto& a : ofs_)
            DestroyState(a);
        ofs_.clear();
        active_ = nullptr;
    }

    bool EnsureOF(int w, int h, int grid, int perf, cudaStream_t stream, std::string* err)
    {
        ++useClock_;
        for (auto& a : ofs_)
            if (a.w == w && a.ht == h && a.grid == grid && a.perf == perf)
            {
                if (a.stream != stream)
                {
                    of_.nvOFSetIOCudaStreams(a.h, (CUstream)stream, (CUstream)stream);
                    a.stream = stream;
                }
                a.lastUse = useClock_;
                active_ = &a;
                return true;
            }
        if (ofs_.size() >= kMaxOFStates) // evict least recently used
        {
            auto lru = std::min_element(ofs_.begin(), ofs_.end(),
                                        [](const OFState& x, const OFState& y) { return x.lastUse < y.lastUse; });
            DestroyState(*lru);
            ofs_.erase(lru);
        }
        active_ = nullptr;
        OFState a;
        if (of_.nvCreateOpticalFlowCuda(ctx_, &a.h) != NV_OF_SUCCESS)
            return Fail(err, "nvCreateOpticalFlowCuda failed");
        NV_OF_INIT_PARAMS ip = {};
        ip.width = w;
        ip.height = h;
        ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)grid;
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = (NV_OF_PERF_LEVEL)perf;
        if (of_.nvOFInit(a.h, &ip) != NV_OF_SUCCESS)
        {
            LastError(a.h, err, "nvOFInit failed");
            DestroyState(a);
            return false;
        }
        a.w = w;
        a.ht = h;
        a.grid = grid;
        a.perf = perf;
        a.outW = (w + grid - 1) / grid;
        a.outH = (h + grid - 1) / grid;
        NV_OF_BUFFER_DESCRIPTOR in = {(uint32_t)w, (uint32_t)h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8};
        NV_OF_BUFFER_DESCRIPTOR od = {(uint32_t)a.outW, (uint32_t)a.outH, NV_OF_BUFFER_USAGE_OUTPUT,
                                      NV_OF_BUFFER_FORMAT_SHORT2};
        for (int i = 0; i < 3; ++i)
        {
            NV_OF_CUDA_BUFFER_STRIDE_INFO si = {};
            if (of_.nvOFCreateGPUBufferCuda(a.h, &in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &a.in[i]) != NV_OF_SUCCESS ||
                of_.nvOFGPUBufferGetStrideInfo(a.in[i], &si) != NV_OF_SUCCESS)
            {
                DestroyState(a);
                return Fail(err, "NVOF input buffer creation failed");
            }
            a.inPtr[i] = of_.nvOFGPUBufferGetCUdeviceptr(a.in[i]);
            a.inPitch[i] = si.strideInfo[0].strideXInBytes;
        }
        for (int i = 0; i < 2; ++i)
        {
            NV_OF_CUDA_BUFFER_STRIDE_INFO si = {};
            if (of_.nvOFCreateGPUBufferCuda(a.h, &od, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &a.out[i]) != NV_OF_SUCCESS ||
                of_.nvOFGPUBufferGetStrideInfo(a.out[i], &si) != NV_OF_SUCCESS)
            {
                DestroyState(a);
                return Fail(err, "NVOF output buffer creation failed");
            }
            a.outPtr[i] = of_.nvOFGPUBufferGetCUdeviceptr(a.out[i]);
            a.outPitch[i] = si.strideInfo[0].strideXInBytes;
        }
        of_.nvOFSetIOCudaStreams(a.h, (CUstream)stream, (CUstream)stream);
        a.stream = stream;
        a.lastUse = useClock_;
        ofs_.push_back(a);
        active_ = &ofs_.back();
        return true;
    }

    void LastError(NvOFHandle h, std::string* err, const char* prefix)
    {
        char buf[512] = {0};
        uint32_t n = sizeof(buf);
        if (h && of_.nvOFGetLastError)
            of_.nvOFGetLastError(h, buf, &n);
        if (err)
            *err = std::string(prefix) + ": " + buf;
    }

    template <bool HALF_IN>
    void Gray(const Frame& f, int i, int w, int h, cudaStream_t st)
    {
        const dim3 b(16, 16);
        KGray<HALF_IN><<<Grid(w, h, b), b, 0, st>>>((const char*)f.ptr, f.pitch, f.width, f.height,
                                                    (unsigned char*)active_->inPtr[i], active_->inPitch[i], w, h);
    }

    bool Execute(int inIdx, int refIdx, int outIdx, std::string* err)
    {
        NV_OF_EXECUTE_INPUT_PARAMS ip = {};
        ip.inputFrame = active_->in[inIdx];
        ip.referenceFrame = active_->in[refIdx];
        ip.disableTemporalHints = NV_OF_TRUE; // frame pairs are independent
        NV_OF_EXECUTE_OUTPUT_PARAMS op = {};
        op.outputBuffer = active_->out[outIdx];
        if (of_.nvOFExecute(active_->h, &ip, &op) != NV_OF_SUCCESS)
        {
            LastError(active_->h, err, "nvOFExecute failed");
            return false;
        }
        return true;
    }

    bool ProcessLocked(const Frame* prev, const Frame& cur, const Frame* next, const Frame& out, const Settings& s,
                       cudaStream_t st, Stats* stats, std::string* err)
    {
        const int W = cur.width, H = cur.height;
        if (prev && (prev->width != W || prev->height != H || prev->half != cur.half))
            prev = nullptr;
        if (next && (next->width != W || next->height != H || next->half != cur.half))
            next = nullptr;
        if (out.width != W || out.height != H)
            return Fail(err, "output size mismatch");

        cudaEventRecord(ev_[0], st);
        const bool hIn = cur.half;
        const dim3 b(16, 16);
        const float hw = (float)s.halfWidth;

        if ((hw <= 0 && s.centerOffset == 0 && !s.showVectors) || (!prev && !next))
        {
            if (hIn == out.half)
                cudaMemcpy2DAsync(out.ptr, out.pitch, cur.ptr, cur.pitch, (size_t)W * (hIn ? 8 : 16), H,
                                  cudaMemcpyDeviceToDevice, st);
            else
            {
                MipSet ms = {};
                ms.level[0] = {(const char*)cur.ptr, cur.pitch, W, H};
                LaunchBlur(ms, hIn, out, 0.0f, 0.0f, 0, s, st); // copies with format conversion
            }
            return Finish(st, stats, err, false);
        }

        // 1-3: motion
        const double scale = std::min(1.0, (double)std::max(64, s.flowWidth) / W);
        const int w = std::max(32, (int)std::lround(W * scale) & ~1), h = std::max(32, (int)std::lround(H * scale) & ~1);
        const int grid = (s.flowGrid == 1 || s.flowGrid == 2) ? s.flowGrid : 4;
        if (!EnsureOF(w, h, grid, s.perfLevel, st, err))
            return false;
        if (hIn)
        {
            Gray<true>(cur, 1, w, h, st);
            if (prev) Gray<true>(*prev, 0, w, h, st);
            if (next) Gray<true>(*next, 2, w, h, st);
        }
        else
        {
            Gray<false>(cur, 1, w, h, st);
            if (prev) Gray<false>(*prev, 0, w, h, st);
            if (next) Gray<false>(*next, 2, w, h, st);
        }
        if (next && !Execute(1, 2, 0, err))
            return false;
        if (prev && !Execute(1, 0, 1, err))
            return false;
        const OFState& a = *active_;
        if (!flow_.Ensure(a.outW, a.outH, sizeof(float2)))
            return Fail(err, "out of GPU memory (flow)");
        KCombineFlow<<<Grid(a.outW, a.outH, b), b, 0, st>>>(
            next ? (const short2*)a.outPtr[0] : nullptr, a.outPitch[0], prev ? (const short2*)a.outPtr[1] : nullptr,
            a.outPitch[1], (float2*)flow_.ptr, flow_.pitch, a.outW, a.outH, (float)W / w);
        cudaEventRecord(ev_[1], st);

        // 4: mip chain
        MipSet ms = {};
        if (!BuildMips(cur, mip_, s, st, &ms, err))
            return false;

        // 5: blur
        LaunchBlur(ms, hIn, out, hw, (float)s.centerOffset, s.showVectors ? 1 : 0, s, st);
        return Finish(st, stats, err, true);
    }

    bool ProcessPairLocked(const Frame& A, const Frame& B, double a, const Frame& out, const Settings& s,
                           cudaStream_t st, Stats* stats, std::string* err)
    {
        const int W = A.width, H = A.height;
        if (B.width != W || B.height != H || B.half != A.half || out.width != W || out.height != H)
            return Fail(err, "frame size/format mismatch");
        cudaEventRecord(ev_[0], st);
        const bool hIn = A.half;
        const dim3 b(16, 16);

        // motion both ways between A and B
        const double scale = std::min(1.0, (double)std::max(64, s.flowWidth) / W);
        const int w = std::max(32, (int)std::lround(W * scale) & ~1), h = std::max(32, (int)std::lround(H * scale) & ~1);
        const int grid = (s.flowGrid == 1 || s.flowGrid == 2) ? s.flowGrid : 4;
        if (!EnsureOF(w, h, grid, s.perfLevel, st, err))
            return false;
        if (hIn)
        {
            Gray<true>(A, 0, w, h, st);
            Gray<true>(B, 1, w, h, st);
        }
        else
        {
            Gray<false>(A, 0, w, h, st);
            Gray<false>(B, 1, w, h, st);
        }
        if (!Execute(0, 1, 0, err) || !Execute(1, 0, 1, err)) // A->B, B->A
            return false;
        const OFState& o = *active_;
        if (!flow_.Ensure(o.outW, o.outH, sizeof(float2)) || !flowB_.Ensure(o.outW, o.outH, sizeof(float2)))
            return Fail(err, "out of GPU memory (flow)");
        const float toFull = (float)W / w;
        KScaleFlow<<<Grid(o.outW, o.outH, b), b, 0, st>>>((const short2*)o.outPtr[0], o.outPitch[0],
                                                          (float2*)flow_.ptr, flow_.pitch, o.outW, o.outH, toFull);
        // B->A points backwards in time: negate for B's forward motion
        KScaleFlow<<<Grid(o.outW, o.outH, b), b, 0, st>>>((const short2*)o.outPtr[1], o.outPitch[1],
                                                          (float2*)flowB_.ptr, flowB_.pitch, o.outW, o.outH, -toFull);
        cudaEventRecord(ev_[1], st);

        MipSet ma = {}, mb = {};
        if (!BuildMips(A, mip_, s, st, &ma, err) || !BuildMips(B, mipB_, s, st, &mb, err))
            return false;

        const dim3 g = Grid(W, H, b);
        const FlowField fa = {(const float2*)flow_.ptr, flow_.pitch, flow_.w, flow_.h};
        const FlowField fb = {(const float2*)flowB_.ptr, flowB_.pitch, flowB_.w, flowB_.h};
        const float hw = (float)std::max(0.0, s.halfWidth), af = (float)std::clamp(a, 0.0, 1.0);
        const int mode = s.showVectors ? 1 : 0, maxS = std::max(2, s.maxSamples);
        const float mipT = (float)std::max(1, s.mipThreshold);
        char* op = (char*)out.ptr;
        if (hIn && out.half)
            KBlurPair<true, true><<<g, b, 0, st>>>(ma, mb, fa, fb, op, out.pitch, hw, af, mode, maxS, mipT);
        else if (hIn)
            KBlurPair<true, false><<<g, b, 0, st>>>(ma, mb, fa, fb, op, out.pitch, hw, af, mode, maxS, mipT);
        else if (out.half)
            KBlurPair<false, true><<<g, b, 0, st>>>(ma, mb, fa, fb, op, out.pitch, hw, af, mode, maxS, mipT);
        else
            KBlurPair<false, false><<<g, b, 0, st>>>(ma, mb, fa, fb, op, out.pitch, hw, af, mode, maxS, mipT);
        return Finish(st, stats, err, true);
    }

    // ---- hybrid path: persistent device copies of host frames
    bool EnsureFloatFrame(Frame& f, int w, int h)
    {
        if (f.ptr && f.width == w && f.height == h)
            return true;
        if (f.ptr)
            cudaFree(f.ptr);
        f = Frame();
        if (cudaMallocPitch(&f.ptr, &f.pitch, (size_t)w * 16, h) != cudaSuccess)
        {
            f.ptr = nullptr;
            return false;
        }
        f.width = w;
        f.height = h;
        f.half = false;
        return true;
    }

    bool UploadHost(const HostImage& img, int slot, std::string* err)
    {
        if (!img.data || img.width <= 0 || img.height <= 0)
            return Fail(err, "empty host image");
        if (img.isFloat) // already float BGRA: straight copy, nothing clamped or quantised
        {
            if (!EnsureFloatFrame(hostF_[slot], img.width, img.height))
                return Fail(err, "out of GPU memory (upload)");
            if (cudaMemcpy2D(hostF_[slot].ptr, hostF_[slot].pitch, img.data, img.rowBytes, (size_t)img.width * 16,
                             img.height, cudaMemcpyHostToDevice) != cudaSuccess)
                return Fail(err, "upload failed");
            return true;
        }
        if (!stage_[slot].Ensure(img.width, img.height, 4) || !EnsureFloatFrame(hostF_[slot], img.width, img.height))
            return Fail(err, "out of GPU memory (upload)");
        if (cudaMemcpy2D(stage_[slot].ptr, stage_[slot].pitch, img.data, img.rowBytes, (size_t)img.width * 4,
                         img.height, cudaMemcpyHostToDevice) != cudaSuccess)
            return Fail(err, "upload failed");
        const dim3 b(16, 16);
        KU8ToF4<<<Grid(img.width, img.height, b), b>>>((const unsigned char*)stage_[slot].ptr, stage_[slot].pitch,
                                                        (char*)hostF_[slot].ptr, hostF_[slot].pitch, img.width,
                                                        img.height, img.argb ? 1 : 0);
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

    bool EnsureHostOut(int w, int h, std::string* err)
    {
        if (!EnsureFloatFrame(hostOut_, w, h) || !stageOut_.Ensure(w, h, 4))
            return Fail(err, "out of GPU memory (output)");
        return true;
    }

    bool DownloadHost(void* out, size_t outRowBytes, bool argb, bool isFloat, std::string* err)
    {
        if (isFloat)
        {
            if (cudaMemcpy2D(out, outRowBytes, hostOut_.ptr, hostOut_.pitch, (size_t)hostOut_.width * 16,
                             hostOut_.height, cudaMemcpyDeviceToHost) != cudaSuccess)
                return Fail(err, "download failed");
            return true;
        }
        const dim3 b(16, 16);
        KF4ToU8<<<Grid(hostOut_.width, hostOut_.height, b), b>>>((const char*)hostOut_.ptr, hostOut_.pitch,
                                                                  (unsigned char*)stageOut_.ptr, stageOut_.pitch,
                                                                  hostOut_.width, hostOut_.height, argb ? 1 : 0);
        if (cudaMemcpy2D(out, outRowBytes, stageOut_.ptr, stageOut_.pitch, (size_t)hostOut_.width * 4,
                         hostOut_.height, cudaMemcpyDeviceToHost) != cudaSuccess)
            return Fail(err, "download failed");
        return true;
    }

    // Mip chain of f into bufs (level 0 = f itself).
    bool BuildMips(const Frame& f, DevBuf* bufs, const Settings& s, cudaStream_t st, MipSet* ms, std::string* err)
    {
        const dim3 b(16, 16);
        *ms = {};
        ms->level[0] = {(const char*)f.ptr, f.pitch, f.width, f.height};
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
            if (l == 1 && f.half)
                KDown<true><<<Grid(dw, dh, b), b, 0, st>>>(src.ptr, src.pitch, src.w, src.h, (char*)bufs[0].ptr,
                                                          bufs[0].pitch, dw, dh);
            else
                KDown<false><<<Grid(dw, dh, b), b, 0, st>>>(src.ptr, src.pitch, src.w, src.h, (char*)bufs[l - 1].ptr,
                                                           bufs[l - 1].pitch, dw, dh);
            ms->level[l] = {(const char*)bufs[l - 1].ptr, bufs[l - 1].pitch, dw, dh};
            ms->top = l;
        }
        return true;
    }

    void LaunchBlur(const MipSet& ms, bool hIn, const Frame& out, float hw, float c0, int mode, const Settings& s,
                    cudaStream_t st)
    {
        const dim3 b(16, 16);
        const dim3 g = Grid(ms.level[0].w, ms.level[0].h, b);
        // With hw == 0 every pixel takes the copy path, so the flow buffer is never read.
        const FlowField fl = {(const float2*)flow_.ptr, flow_.pitch, std::max(1, flow_.w), std::max(1, flow_.h)};
        const int maxS = std::max(2, s.maxSamples);
        const float mipT = (float)std::max(1, s.mipThreshold);
        char* o = (char*)out.ptr;
        if (hIn && out.half)
            KBlur<true, true><<<g, b, 0, st>>>(ms, fl, o, out.pitch, hw, c0, mode, maxS, mipT);
        else if (hIn)
            KBlur<true, false><<<g, b, 0, st>>>(ms, fl, o, out.pitch, hw, c0, mode, maxS, mipT);
        else if (out.half)
            KBlur<false, true><<<g, b, 0, st>>>(ms, fl, o, out.pitch, hw, c0, mode, maxS, mipT);
        else
            KBlur<false, false><<<g, b, 0, st>>>(ms, fl, o, out.pitch, hw, c0, mode, maxS, mipT);
    }

    bool Finish(cudaStream_t st, Stats* stats, std::string* err, bool flowRan)
    {
        cudaEventRecord(ev_[2], st);
        const cudaError_t e = cudaEventSynchronize(ev_[2]);
        const cudaError_t le = cudaGetLastError();
        if (e != cudaSuccess || le != cudaSuccess)
        {
            if (err)
                *err = std::string("CUDA error: ") + cudaGetErrorString(e != cudaSuccess ? e : le);
            return false;
        }
        if (stats)
        {
            float a = 0, b2 = 0, t = 0;
            if (flowRan)
            {
                cudaEventElapsedTime(&a, ev_[0], ev_[1]);
                cudaEventElapsedTime(&b2, ev_[1], ev_[2]);
            }
            cudaEventElapsedTime(&t, ev_[0], ev_[2]);
            stats->flowMs = a;
            stats->blurMs = b2;
            stats->totalMs = t;
        }
        return true;
    }

    std::mutex mu_;
    CUcontext ctx_ = nullptr;
    HMODULE lib_ = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST of_ = {};
    std::vector<OFState> ofs_ = std::vector<OFState>(); // capacity reserved in Init (active_ points into it)
    OFState* active_ = nullptr;
    unsigned long long useClock_ = 0;
    DevBuf flow_, flowB_, mip_[3], mipB_[3];
    float* diffSum_ = nullptr;
    // hybrid path
    DevBuf stage_[3], stageOut_;
    Frame hostF_[3], hostOut_;
    cudaEvent_t ev_[3] = {};
};

} // namespace

std::unique_ptr<Engine> Engine::Create(void* cuContext, std::string* err)
{
    auto e = std::make_unique<EngineImpl>();
    if (!e->Init(cuContext, err))
        return nullptr;
    return e;
}

} // namespace oe::gpu
