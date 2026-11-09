// GPU motion blur on macOS (Metal). See GpuMotionBlur.h; a port of
// GpuMotionBlur.cu with the same kernels and the same maths.
//
// Apple Silicon shares memory between CPU and GPU, so frames go into shared
// buffers with a plain copy (no upload / download transfers). Buffers are
// tightly packed (pitch = width). The kernels are compiled from source when
// the engine is created, with fast math off so results match the CUDA path.
//
// Per output frame, one command buffer (serial compute encoder, so each
// dispatch sees the previous one's writes):
//   1. 8-bit frames converted to float BGRA (float frames copied as is)
//   2. mip chain of each frame (1/2, 1/4, 1/8, box filtered)
//   3. each pixel: average the frame along +/- halfWidth * v
//   4. back to 8-bit when the input was 8-bit
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "GpuMotionBlur.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace oe::gpu
{
namespace
{

// Argument structs: plain ints / floats only, so C++ and MSL lay them out the same.
struct MipDims
{
    int w[4], h[4];
    int top;
};
struct BlurArgs
{
    MipDims a, b;
    int fw, fh, fbw, fbh; // flow A / flow B grid sizes
    float hw, c0;         // half-width; centre offset (single frame) or a (pair)
    int mode, maxS;       // 0 blur, 1 motion vectors; sample cap
    float mipT;
};
struct AccumArgs
{
    int W, H, aw, ah;
    float weight;
    int first, vectors;
};
struct ConvArgs
{
    int w, h, argb;
};
struct DownArgs
{
    int sw, sh, dw, dh;
};

const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct MipDims { int w[4]; int h[4]; int top; };
struct BlurArgs { MipDims a; MipDims b; int fw, fh, fbw, fbh; float hw, c0; int mode, maxS; float mipT; };
struct AccumArgs { int W, H, aw, ah; float weight; int first, vectors; };
struct ConvArgs { int w, h, argb; };
struct DownArgs { int sw, sh, dw, dh; };

// Images are float4 B,G,R,A, 0..1 (values above 1 allowed), pitch = width.
inline float4 Ld(device const float4* img, int w, int x, int y) { return img[y * w + x]; }

inline float4 Bilinear(device const float4* img, int w, int h, float sx, float sy)
{
    sx = min(max(sx, 0.0f), float(w - 1));
    sy = min(max(sy, 0.0f), float(h - 1));
    const int x0 = min(int(sx), max(0, w - 2)), y0 = min(int(sy), max(0, h - 2));
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    return w00 * Ld(img, w, x0, y0) + w10 * Ld(img, w, x1, y0) + w01 * Ld(img, w, x0, y1) + w11 * Ld(img, w, x1, y1);
}

struct Mips
{
    device const float4* p[4];
    int w[4], h[4];
    int top;
};

inline Mips MakeMips(device const float4* l0, device const float4* l1, device const float4* l2,
                     device const float4* l3, constant MipDims& d)
{
    Mips m;
    m.p[0] = l0; m.p[1] = l1; m.p[2] = l2; m.p[3] = l3;
    for (int i = 0; i < 4; ++i) { m.w[i] = d.w[i]; m.h[i] = d.h[i]; }
    m.top = d.top;
    return m;
}

// Motion-field colour: hue = direction, saturation = speed / norm (RGB 0..1).
inline float3 VectorColour(float vx, float vy, float norm)
{
    const float sat = min(1.0f, sqrt(vx * vx + vy * vy) / norm);
    float h = atan2(vy, vx) / 6.2831853f;
    if (h < 0)
        h += 1.0f;
    const float h6 = h * 6.0f;
    const int i = int(h6) % 6;
    const float f = h6 - floor(h6);
    const float p = 1 - sat, q = 1 - sat * f, t = 1 - sat * (1 - f);
    switch (i)
    {
    case 0: return float3(1, t, p);
    case 1: return float3(q, 1, p);
    case 2: return float3(p, 1, t);
    case 3: return float3(p, q, 1);
    case 4: return float3(t, p, 1);
    default: return float3(1, p, q);
    }
}

// Velocity at full-res pixel (x, y): bilinear from the (coarser) flow grid.
inline float2 SampleFlow(device const float2* f, int fw, int fh, int W, int H, int x, int y)
{
    float gx = (x + 0.5f) * (float(fw) / W) - 0.5f, gy = (y + 0.5f) * (float(fh) / H) - 0.5f;
    gx = min(max(gx, 0.0f), float(fw - 1));
    gy = min(max(gy, 0.0f), float(fh - 1));
    const int ix = min(int(gx), max(0, fw - 2)), iy = min(int(gy), max(0, fh - 2));
    const int ix1 = min(ix + 1, fw - 1), iy1 = min(iy + 1, fh - 1);
    const float tx = gx - ix, ty = gy - iy;
    return (1 - tx) * (1 - ty) * f[iy * fw + ix] + tx * (1 - ty) * f[iy * fw + ix1] +
           (1 - tx) * ty * f[iy1 * fw + ix] + tx * ty * f[iy1 * fw + ix1];
}

// Average of the image along -(c0 +/- hw) * v around (x, y): the shutter
// window [c0-hw, c0+hw] (source-frame units) of a pixel moving with v.
float4 Gather(thread const Mips& m, int x, int y, float vx, float vy, float hw, float c0, int maxS, float mipT)
{
    const float speed = sqrt(vx * vx + vy * vy);
    const float smear = 2.0f * hw * speed;
    if (smear <= 1.0f && fabs(c0) * speed < 0.5f)
        return Ld(m.p[0], m.w[0], x, y);
    int lvl = 0;
    while (lvl < m.top && smear / float(1 << lvl) > mipT)
        ++lvl;
    const float inv2l = 1.0f / float(1 << lvl);
    const int n = min(maxS, max(2, int(ceil(smear * inv2l))));
    device const float4* P = m.p[lvl];
    const int w = m.w[lvl], h = m.h[lvl];
    float4 acc = float4(0);
    for (int j = 0; j < n; ++j)
    {
        const float off = c0 + hw * (((j + 0.5f) / n) * 2.0f - 1.0f);
        const float sx = (x - off * vx + 0.5f) * inv2l - 0.5f;
        const float sy = (y - off * vy + 0.5f) * inv2l - 0.5f;
        acc += Bilinear(P, w, h, sx, sy);
    }
    return acc * (1.0f / n);
}

inline float4 VectorPixel(device const float4* L0, int W, int x, int y, float vx, float vy)
{
    const float4 p = Ld(L0, W, x, y); // x=B y=G z=R
    const float g = min(max(0.114f * p.x + 0.587f * p.y + 0.299f * p.z, 0.0f), 1.0f);
    const float3 c = VectorColour(vx, vy, max(4.0f, 0.01f * W));
    const float k = 0.65f * (0.4f + 0.6f * g);
    return float4(0.35f * g + k * c.z, 0.35f * g + k * c.y, 0.35f * g + k * c.x, p.w);
}

// Single frame. mode 0: blur, 1: motion-vector view. c0 = output time offset (source frames).
kernel void k_blur(device const float4* a0 [[buffer(0)]], device const float4* a1 [[buffer(1)]],
                   device const float4* a2 [[buffer(2)]], device const float4* a3 [[buffer(3)]],
                   device const float2* flow [[buffer(8)]], device float4* out [[buffer(10)]],
                   constant BlurArgs& p [[buffer(11)]], uint2 gid [[thread_position_in_grid]])
{
    const int W = p.a.w[0], H = p.a.h[0];
    const int x = gid.x, y = gid.y;
    if (x >= W || y >= H)
        return;
    const Mips m = MakeMips(a0, a1, a2, a3, p.a);
    const float2 v = SampleFlow(flow, p.fw, p.fh, W, H, x, y);
    out[y * W + x] = p.mode == 1 ? VectorPixel(a0, W, x, y, v.x, v.y)
                                 : Gather(m, x, y, v.x, v.y, p.hw, p.c0, p.maxS, p.mipT);
}

// Two-frame interpolation: output time lies a (0..1) of the way from frame A
// to frame B. Blur A at +a and B at -(1-a), each with its own motion, and
// blend by closeness.
kernel void k_blur_pair(device const float4* a0 [[buffer(0)]], device const float4* a1 [[buffer(1)]],
                        device const float4* a2 [[buffer(2)]], device const float4* a3 [[buffer(3)]],
                        device const float4* b0 [[buffer(4)]], device const float4* b1 [[buffer(5)]],
                        device const float4* b2 [[buffer(6)]], device const float4* b3 [[buffer(7)]],
                        device const float2* fa [[buffer(8)]], device const float2* fb [[buffer(9)]],
                        device float4* out [[buffer(10)]], constant BlurArgs& p [[buffer(11)]],
                        uint2 gid [[thread_position_in_grid]])
{
    const int W = p.a.w[0], H = p.a.h[0];
    const int x = gid.x, y = gid.y;
    if (x >= W || y >= H)
        return;
    const float a = p.c0;
    const float2 va = SampleFlow(fa, p.fw, p.fh, W, H, x, y);
    if (p.mode == 1)
    {
        out[y * W + x] = VectorPixel(a0, W, x, y, va.x, va.y);
        return;
    }
    const Mips ma = MakeMips(a0, a1, a2, a3, p.a);
    float4 r = Gather(ma, x, y, va.x, va.y, p.hw, a, p.maxS, p.mipT);
    if (a > 1e-3f)
    {
        const Mips mb = MakeMips(b0, b1, b2, b3, p.b);
        const float2 vb = SampleFlow(fb, p.fbw, p.fbh, W, H, x, y);
        const float4 q = Gather(mb, x, y, vb.x, vb.y, p.hw, a - 1.0f, p.maxS, p.mipT);
        r = (1 - a) * r + a * q;
    }
    out[y * W + x] = r;
}

// One time sample of the temporal supersampling (Extra High): warp both frames
// by the sample's field (bilinear from the reduced field grid), blend by its
// mask and add weight * result to acc (first sample: overwrite).
kernel void k_rife_accum(device const float4* f0 [[buffer(0)]], device const float4* f1 [[buffer(1)]],
                         device const float4* flow [[buffer(2)]], device const float* mask [[buffer(3)]],
                         device float4* acc [[buffer(4)]], constant AccumArgs& p [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const int W = p.W, H = p.H, aw = p.aw, ah = p.ah;
    const int x = gid.x, y = gid.y;
    if (x >= W || y >= H)
        return;
    float gx = (x + 0.5f) * (float(aw) / W) - 0.5f, gy = (y + 0.5f) * (float(ah) / H) - 0.5f;
    gx = min(max(gx, 0.0f), float(aw - 1));
    gy = min(max(gy, 0.0f), float(ah - 1));
    const int ix = min(int(gx), max(0, aw - 2)), iy = min(int(gy), max(0, ah - 2));
    const int ix1 = min(ix + 1, aw - 1), iy1 = min(iy + 1, ah - 1);
    const float tx = gx - ix, ty = gy - iy;
    const float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    const float4 f = w00 * flow[iy * aw + ix] + w10 * flow[iy * aw + ix1] + w01 * flow[iy1 * aw + ix] +
                     w11 * flow[iy1 * aw + ix1];
    const float m = min(max(w00 * mask[iy * aw + ix] + w10 * mask[iy * aw + ix1] + w01 * mask[iy1 * aw + ix] +
                            w11 * mask[iy1 * aw + ix1], 0.0f), 1.0f);
    const float sx = float(W) / aw, sy = float(H) / ah;
    float4 r;
    if (p.vectors)
        r = VectorPixel(f0, W, x, y, -f.x * sx, -f.y * sy);
    else
    {
        const float4 v0 = Bilinear(f0, W, H, x + f.x * sx, y + f.y * sy);
        const float4 v1 = Bilinear(f1, W, H, x + f.z * sx, y + f.w * sy);
        r = m * v0 + (1 - m) * v1;
    }
    const float4 a = p.first ? float4(0) : acc[y * W + x];
    acc[y * W + x] = a + p.weight * r;
}

// 8-bit -> float4 (B,G,R,A order internally). argb: source bytes are A,R,G,B.
kernel void k_u8_to_f4(device const uchar4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                       constant ConvArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h)
        return;
    const float4 v = float4(src[y * p.w + x]) * (1.0f / 255.0f);
    dst[y * p.w + x] = p.argb ? float4(v.w, v.z, v.y, v.x) : v;
}

inline uchar ToU8(float v) { return uchar(min(max(v, 0.0f), 1.0f) * 255.0f + 0.5f); }

kernel void k_f4_to_u8(device const float4* src [[buffer(0)]], device uchar4* dst [[buffer(1)]],
                       constant ConvArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h)
        return;
    const float4 v = src[y * p.w + x]; // x=B y=G z=R w=A
    dst[y * p.w + x] = p.argb ? uchar4(ToU8(v.w), ToU8(v.z), ToU8(v.y), ToU8(v.x))
                              : uchar4(ToU8(v.x), ToU8(v.y), ToU8(v.z), ToU8(v.w));
}

// 2x2 box downsample into the next mip level.
kernel void k_down(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant DownArgs& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.dw || y >= p.dh)
        return;
    const int x0 = min(2 * x, p.sw - 1), x1 = min(2 * x + 1, p.sw - 1);
    const int y0 = min(2 * y, p.sh - 1), y1 = min(2 * y + 1, p.sh - 1);
    dst[y * p.dw + x] = (src[y0 * p.sw + x0] + src[y0 * p.sw + x1] + src[y1 * p.sw + x0] + src[y1 * p.sw + x1]) * 0.25f;
}
)MSL";

// Copies `rows` rows of `rowBytes` between buffers with different pitches.
void CopyRows(void* dst, size_t dstPitch, const void* src, size_t srcPitch, size_t rowBytes, int rows)
{
    if (dstPitch == rowBytes && srcPitch == rowBytes)
    {
        std::memcpy(dst, src, rowBytes * rows);
        return;
    }
    for (int y = 0; y < rows; ++y)
        std::memcpy(static_cast<char*>(dst) + y * dstPitch, static_cast<const char*>(src) + y * srcPitch, rowBytes);
}

struct Buf
{
    id<MTLBuffer> b = nil;
    size_t len = 0;
    bool Ensure(id<MTLDevice> dev, size_t bytes)
    {
        if (b && len >= bytes)
            return true;
        b = [dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
        len = b ? bytes : 0;
        return b != nil;
    }
    void* Ptr() const { return b.contents; }
};

class EngineImpl final : public Engine
{
public:
    bool Init(std::string* err)
    {
        @autoreleasepool
        {
            dev_ = MTLCreateSystemDefaultDevice();
            if (!dev_)
                return Fail(err, "no Metal device");
            queue_ = [dev_ newCommandQueue];
            if (!queue_)
                return Fail(err, "no Metal command queue");
            MTLCompileOptions* opt = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opt.fastMathEnabled = NO; // match CUDA's IEEE maths (mathMode on macOS 15+ says the same)
#pragma clang diagnostic pop
            NSError* e = nil;
            id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(kSource) options:opt error:&e];
            if (!lib)
                return Fail(err, ("Metal kernels didn't compile: " +
                                  std::string(e ? e.localizedDescription.UTF8String : "?")).c_str());
            struct
            {
                const char* name;
                id<MTLComputePipelineState> __strong* ps;
            } kernels[] = {{"k_blur", &blur_},       {"k_blur_pair", &pair_}, {"k_rife_accum", &accum_},
                           {"k_u8_to_f4", &toF4_},   {"k_f4_to_u8", &toU8_},  {"k_down", &down_}};
            for (auto& k : kernels)
            {
                id<MTLFunction> fn = [lib newFunctionWithName:@(k.name)];
                *k.ps = fn ? [dev_ newComputePipelineStateWithFunction:fn error:&e] : nil;
                if (!*k.ps)
                    return Fail(err, (std::string("Metal pipeline ") + k.name + " failed").c_str());
            }
            name_ = std::string("Metal: ") + dev_.name.UTF8String;
            return true;
        }
    }

    std::string Describe() const override { return name_; }

    bool ProcessPair(const HostImage& A, const HostImage& B, double a, const HostFlow& fa, const HostFlow& fb,
                     void* out, size_t outRowBytes, const Settings& s, Stats* stats, std::string* err) override
    {
        if (B.width != A.width || B.height != A.height || B.isFloat != A.isFloat)
            return Fail(err, "frame size/format mismatch");
        std::lock_guard<std::mutex> lock(mu_);
        @autoreleasepool
        {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            BlurArgs args = {};
            id<MTLBuffer> ma[4], mb[4];
            bool ok = EnsureOut(A.width, A.height, err) && Upload(enc, A, 0, err) && Upload(enc, B, 1, err) &&
                      UploadFlow(fa, flowA_, err) && UploadFlow(fb, flowB_, err) &&
                      BuildMips(enc, frame_[0], A.width, A.height, mipA_, s, &args.a, ma, err) &&
                      BuildMips(enc, frame_[1], A.width, A.height, mipB_, s, &args.b, mb, err);
            if (ok)
            {
                args.fw = fa.width;
                args.fh = fa.height;
                args.fbw = fb.width;
                args.fbh = fb.height;
                args.hw = (float)std::max(0.0, s.halfWidth);
                args.c0 = (float)std::clamp(a, 0.0, 1.0);
                args.mode = s.showVectors ? 1 : 0;
                args.maxS = std::max(2, s.maxSamples);
                args.mipT = (float)std::max(1, s.mipThreshold);
                [enc setComputePipelineState:pair_];
                for (int i = 0; i < 4; ++i)
                {
                    [enc setBuffer:ma[i] offset:0 atIndex:i];
                    [enc setBuffer:mb[i] offset:0 atIndex:4 + i];
                }
                [enc setBuffer:flowA_.b offset:0 atIndex:8];
                [enc setBuffer:flowB_.b offset:0 atIndex:9];
                [enc setBuffer:out_.b offset:0 atIndex:10];
                [enc setBytes:&args length:sizeof(args) atIndex:11];
                Dispatch(enc, pair_, A.width, A.height);
                EncodeToU8(enc, A);
            }
            [enc endEncoding];
            ok = ok && Finish(cb, stats, err);
            return ok && Download(out, outRowBytes, A, err);
        }
    }

    bool Process(const HostImage& cur, const HostFlow& v, void* out, size_t outRowBytes, const Settings& s,
                 Stats* stats, std::string* err) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        @autoreleasepool
        {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            BlurArgs args = {};
            id<MTLBuffer> ms[4];
            bool ok = EnsureOut(cur.width, cur.height, err) && Upload(enc, cur, 0, err) &&
                      UploadFlow(v, flowA_, err) &&
                      BuildMips(enc, frame_[0], cur.width, cur.height, mipA_, s, &args.a, ms, err);
            if (ok)
            {
                args.fw = v.width;
                args.fh = v.height;
                args.hw = (float)std::max(0.0, s.halfWidth);
                args.c0 = (float)s.centerOffset;
                args.mode = s.showVectors ? 1 : 0;
                args.maxS = std::max(2, s.maxSamples);
                args.mipT = (float)std::max(1, s.mipThreshold);
                [enc setComputePipelineState:blur_];
                for (int i = 0; i < 4; ++i)
                    [enc setBuffer:ms[i] offset:0 atIndex:i];
                [enc setBuffer:flowA_.b offset:0 atIndex:8];
                [enc setBuffer:out_.b offset:0 atIndex:10];
                [enc setBytes:&args length:sizeof(args) atIndex:11];
                Dispatch(enc, blur_, cur.width, cur.height);
                EncodeToU8(enc, cur);
            }
            [enc endEncoding];
            ok = ok && Finish(cb, stats, err);
            return ok && Download(out, outRowBytes, cur, err);
        }
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
        @autoreleasepool
        {
            const int W = frames[0].width, H = frames[0].height;
            double gpuMs = 0;
            bool ok = EnsureOut(W, H, err);
            // The field buffer is refilled from the CPU for every sample, so each
            // sample gets its own command buffer (frames converted with the first).
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            bool encoding = true; // enc still open (must be ended exactly once)
            for (int i = 0; i < nFrames && ok; ++i)
                ok = Upload(enc, frames[i], i, err);
            for (int i = 0; i < nSamples && ok; ++i)
            {
                const TimeSample& smp = samples[i];
                const HostField& f = smp.field;
                if (smp.frame0 < 0 || smp.frame0 >= nFrames || smp.frame1 < 0 || smp.frame1 >= nFrames || !f.flow ||
                    !f.mask || f.width < 2 || f.height < 2)
                {
                    ok = Fail(err, "bad time sample");
                    break;
                }
                if (!fieldFlow_.Ensure(dev_, (size_t)f.width * f.height * 16) ||
                    !fieldMask_.Ensure(dev_, (size_t)f.width * f.height * 4))
                {
                    ok = Fail(err, "out of GPU memory (field)");
                    break;
                }
                CopyRows(fieldFlow_.Ptr(), (size_t)f.width * 16, f.flow, f.flowRowBytes, (size_t)f.width * 16,
                         f.height);
                CopyRows(fieldMask_.Ptr(), (size_t)f.width * 4, f.mask, f.maskRowBytes, (size_t)f.width * 4,
                         f.height);
                const AccumArgs args = {W, H, f.width, f.height, showVectors ? 1.0f : smp.weight, i == 0 ? 1 : 0,
                                        showVectors ? 1 : 0};
                [enc setComputePipelineState:accum_];
                [enc setBuffer:frame_[smp.frame0].b offset:0 atIndex:0];
                [enc setBuffer:frame_[smp.frame1].b offset:0 atIndex:1];
                [enc setBuffer:fieldFlow_.b offset:0 atIndex:2];
                [enc setBuffer:fieldMask_.b offset:0 atIndex:3];
                [enc setBuffer:out_.b offset:0 atIndex:4];
                [enc setBytes:&args length:sizeof(args) atIndex:5];
                Dispatch(enc, accum_, W, H);
                const bool last = showVectors || i == nSamples - 1;
                if (last)
                    EncodeToU8(enc, frames[0]);
                [enc endEncoding];
                encoding = false;
                Stats st;
                ok = Finish(cb, &st, err);
                gpuMs += st.gpuMs;
                if (last || !ok)
                    break;
                cb = [queue_ commandBuffer];
                enc = [cb computeCommandEncoder];
                encoding = true;
            }
            if (encoding)
                [enc endEncoding]; // failed before this command buffer was committed: drop it
            if (stats)
                stats->gpuMs = gpuMs;
            return ok && Download(out, outRowBytes, frames[0], err);
        }
    }

private:
    static constexpr int kMaxFrames = 4;

    static bool Fail(std::string* err, const char* msg)
    {
        if (err)
            *err = msg;
        return false;
    }

    void Dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int w, int h)
    {
        // 16x16 as in CUDA; smaller if the kernel's register use doesn't allow 256 threads.
        const NSUInteger t = ps.maxTotalThreadsPerThreadgroup >= 256 ? 16 : 8;
        [enc dispatchThreadgroups:MTLSizeMake((w + t - 1) / t, (h + t - 1) / t, 1)
            threadsPerThreadgroup:MTLSizeMake(t, t, 1)];
    }

    // Host frame -> float frame in frame_[slot] (8-bit: via staging + conversion kernel).
    bool Upload(id<MTLComputeCommandEncoder> enc, const HostImage& img, int slot, std::string* err)
    {
        if (!img.data || img.width <= 0 || img.height <= 0)
            return Fail(err, "empty host image");
        const size_t px = (size_t)img.width * img.height;
        Buf& f = frame_[slot];
        if (!f.Ensure(dev_, px * 16))
            return Fail(err, "out of GPU memory (upload)");
        if (img.isFloat) // already float BGRA: straight copy, nothing clamped or quantised
        {
            CopyRows(f.Ptr(), (size_t)img.width * 16, img.data, img.rowBytes, (size_t)img.width * 16, img.height);
            return true;
        }
        Buf& st = stage_[slot];
        if (!st.Ensure(dev_, px * 4))
            return Fail(err, "out of GPU memory (upload)");
        CopyRows(st.Ptr(), (size_t)img.width * 4, img.data, img.rowBytes, (size_t)img.width * 4, img.height);
        const ConvArgs args = {img.width, img.height, img.argb ? 1 : 0};
        [enc setComputePipelineState:toF4_];
        [enc setBuffer:st.b offset:0 atIndex:0];
        [enc setBuffer:f.b offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        Dispatch(enc, toF4_, img.width, img.height);
        return true;
    }

    bool UploadFlow(const HostFlow& f, Buf& dst, std::string* err)
    {
        if (!f.data || f.width <= 0 || f.height <= 0)
            return Fail(err, "empty flow");
        if (!dst.Ensure(dev_, (size_t)f.width * f.height * 8))
            return Fail(err, "out of GPU memory (flow)");
        CopyRows(dst.Ptr(), (size_t)f.width * 8, f.data, f.rowBytes, (size_t)f.width * 8, f.height);
        return true;
    }

    bool EnsureOut(int w, int h, std::string* err)
    {
        outW_ = w;
        outH_ = h;
        if (!out_.Ensure(dev_, (size_t)w * h * 16) || !stageOut_.Ensure(dev_, (size_t)w * h * 4))
            return Fail(err, "out of GPU memory (output)");
        return true;
    }

    // 8-bit output: convert out_ into stageOut_ in the same command buffer.
    void EncodeToU8(id<MTLComputeCommandEncoder> enc, const HostImage& like)
    {
        if (like.isFloat)
            return;
        const ConvArgs args = {outW_, outH_, like.argb ? 1 : 0};
        [enc setComputePipelineState:toU8_];
        [enc setBuffer:out_.b offset:0 atIndex:0];
        [enc setBuffer:stageOut_.b offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        Dispatch(enc, toU8_, outW_, outH_);
    }

    bool Download(void* out, size_t outRowBytes, const HostImage& like, std::string*)
    {
        if (like.isFloat)
            CopyRows(out, outRowBytes, out_.Ptr(), (size_t)outW_ * 16, (size_t)outW_ * 16, outH_);
        else
            CopyRows(out, outRowBytes, stageOut_.Ptr(), (size_t)outW_ * 4, (size_t)outW_ * 4, outH_);
        return true;
    }

    // Mip chain of f (W x H) into bufs; level buffers in lv (level 0 = f itself;
    // unused levels point at f so every binding is valid).
    bool BuildMips(id<MTLComputeCommandEncoder> enc, const Buf& f, int W, int H, Buf* bufs, const Settings& s,
                   MipDims* d, __strong id<MTLBuffer> lv[4], std::string* err)
    {
        for (int l = 0; l < 4; ++l)
        {
            d->w[l] = W;
            d->h[l] = H;
            lv[l] = f.b;
        }
        d->top = 0;
        if (s.mipThreshold <= 0)
            return true;
        for (int l = 1; l <= std::min(3, s.maxMipLevel); ++l)
        {
            const int sw = d->w[l - 1], sh = d->h[l - 1];
            if (sw < 4 || sh < 4)
                break;
            const int dw = (sw + 1) / 2, dh = (sh + 1) / 2;
            if (!bufs[l - 1].Ensure(dev_, (size_t)dw * dh * 16))
                return Fail(err, "out of GPU memory (mips)");
            const DownArgs args = {sw, sh, dw, dh};
            [enc setComputePipelineState:down_];
            [enc setBuffer:lv[l - 1] offset:0 atIndex:0];
            [enc setBuffer:bufs[l - 1].b offset:0 atIndex:1];
            [enc setBytes:&args length:sizeof(args) atIndex:2];
            Dispatch(enc, down_, dw, dh);
            d->w[l] = dw;
            d->h[l] = dh;
            lv[l] = bufs[l - 1].b;
            d->top = l;
        }
        return true;
    }

    bool Finish(id<MTLCommandBuffer> cb, Stats* stats, std::string* err)
    {
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            return Fail(err, ("Metal error: " +
                              std::string(cb.error ? cb.error.localizedDescription.UTF8String : "command failed"))
                                 .c_str());
        if (stats)
            stats->gpuMs = std::max(0.0, (cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
        return true;
    }

    std::mutex mu_;
    id<MTLDevice> dev_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> blur_ = nil, pair_ = nil, accum_ = nil, toF4_ = nil, toU8_ = nil, down_ = nil;
    std::string name_;
    Buf frame_[kMaxFrames], stage_[kMaxFrames], out_, stageOut_; // float frames, 8-bit staging, output
    Buf fieldFlow_, fieldMask_;                                   // Extra High: one sample's field
    Buf flowA_, flowB_, mipA_[3], mipB_[3];
    int outW_ = 0, outH_ = 0;
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
