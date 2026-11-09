#pragma once
// GPU motion blur (CUDA on Windows: GpuMotionBlur.cu; Metal on macOS:
// MetalMotionBlur.mm). Motion is supplied by the caller (OpenCV DIS on the
// CPU, see MotionBlur.h); the GPU does the expensive part at UHD: upload, mip
// chain, the per-pixel gather along the motion path, and download. Same
// algorithm as FlowMotionBlur in MotionBlur.cpp (the CPU fallback).

#include <memory>
#include <string>

namespace oe::gpu
{

struct Settings
{
    double halfWidth = 0.25;  // synthetic blur half-width, source-frame intervals
    int maxSamples = 64;
    int mipThreshold = 48;    // samples before switching to the next mip level (0 = off)
    int maxMipLevel = 3;
    double centerOffset = 0;  // single-frame path: output time relative to the frame, source-frame intervals
    bool showVectors = false; // draw the motion field instead of blurring
};

struct Stats
{
    double gpuMs = 0; // mip chain + blur (after upload, before download)
};

// A frame in host memory, as checked out by the effect.
struct HostImage
{
    const void* data = nullptr;
    size_t rowBytes = 0;
    int width = 0, height = 0;
    bool argb = false;    // 8-bit only: byte order A,R,G,B instead of B,G,R,A
    bool isFloat = false; // 4 x float32 B,G,R,A (Premiere 32-bit), copied as is: no clamping
};

// Motion field: 2 x float32 per vector, velocity in full-res pixels per source
// frame, any resolution (sampled bilinearly).
struct HostFlow
{
    const void* data = nullptr;
    size_t rowBytes = 0;
    int width = 0, height = 0;
};

// Interpolation field for one time sample (from RIFE, rife/RifeFlow.h), at a
// reduced size: flow = 4 x float32 (f0x, f0y, f1x, f1y) in field pixels,
// mask = 1 x float32 (weight of frame0).
struct HostField
{
    const void* flow = nullptr;
    size_t flowRowBytes = 0;
    const void* mask = nullptr;
    size_t maskRowBytes = 0;
    int width = 0, height = 0;
};

struct TimeSample
{
    int frame0 = 0, frame1 = 1; // indices into the frames passed to ProcessSamples
    float weight = 1.0f;
    HostField field;
};

class Engine
{
public:
    // CUDA: primary context of device 0; Metal: the system default GPU.
    // Returns nullptr and sets *err when there is no usable GPU / driver.
    static std::unique_ptr<Engine> Create(std::string* err);
    virtual ~Engine() = default;

    // For the log, e.g. "Metal: Apple M1" (empty when the engine doesn't say).
    virtual std::string Describe() const { return {}; }

    // Calls are thread-safe (serialised per engine). `out` has the layout of
    // the inputs (8-bit with the same byte order, or float).

    // Two-frame interpolation: the output time lies `a` (0..1) of the way from
    // source frame A to the next source frame B. fa / fb: motion at A and at B,
    // both forward in time. A is blurred at +a and B at -(1-a), blended by
    // closeness (exact frame timing).
    virtual bool ProcessPair(const HostImage& A, const HostImage& B, double a, const HostFlow& fa,
                             const HostFlow& fb, void* out, size_t outRowBytes, const Settings& s, Stats* stats,
                             std::string* err) = 0;

    // Single frame with its motion v (exact frame timing off).
    virtual bool Process(const HostImage& cur, const HostFlow& v, void* out, size_t outRowBytes, const Settings& s,
                         Stats* stats, std::string* err) = 0;

    // Temporal supersampling (Quality Extra High): the output is the weighted
    // sum over samples of  mask * frame0(x + f0) + (1 - mask) * frame1(x + f1),
    // each sample's field upsampled to the frame size, i.e. the shutter
    // integrated from interpolated in-between frames. Works on the frames as
    // given (8-bit or float, no clamping). nFrames <= 4, all the same size.
    // showVectors: draw sample 0's frame0 motion instead.
    virtual bool ProcessSamples(const HostImage* frames, int nFrames, const TimeSample* samples, int nSamples,
                                void* out, size_t outRowBytes, bool showVectors, Stats* stats, std::string* err) = 0;
};

} // namespace oe::gpu
