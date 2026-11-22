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
    // Layered blur (Quality Normal): a clearly faster layer passing over a
    // pixel covers it for its share of the shutter, so fast objects streak
    // past their own outline (see Layered in GpuMotionBlur.cu). Off: each
    // pixel is only averaged along its own motion.
    bool layered = false;
};

struct Stats
{
    double gpuMs = 0;  // GPU work after upload, before download (motion + mip chain + blur)
    double flowMs = 0; // of which motion estimation (the *GpuFlow calls)
    // wall-clock parts of a call (CUDA engine, ProcessGpuFlow): waiting for
    // the engine (other calls), for the host's exclusive access to the device,
    // uploads, download
    double lockMs = 0, acquireMs = 0, uploadMs = 0, downloadMs = 0;
};

// Motion estimated on the GPU (Quality Normal / High): the CPU path
// (EstimateFlow in MotionBlur.cpp: OpenCV DIS + motion-boundary clean-up)
// ported to the GPU and run on the uploaded frames, so nothing comes back to
// the CPU between motion and blur.
struct FlowSettings
{
    int flowWidth = 960;     // analysis width
    int preset = 1;          // cv::DISOpticalFlow preset: 0 ultrafast, 1 fast (Normal), 2 medium (High)
    bool refineEdges = true; // motion-boundary clean-up (RefineFlowEdges)
    // Motion of a frame from both neighbours (the previous and the next
    // frame, each checked forward and backward): where an object is about
    // to cover the background, the next frame has no match and the motion
    // estimate leaks the object's motion onto it; the previous frame still
    // saw it. Needs the frame before A / after B (ProcessPairGpuFlow's prev /
    // next); without them, one side as before.
    bool twoSided = false;
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

#ifdef _WIN32
// A CUDA device owned by the host (Premiere's MediaCore GPU Device Suite).
// An engine on it works in the host's CUDA context instead of creating its
// own, holds the host's exclusive access to the device around every call and
// allocates device memory through the host. (Premiere 26 renders third-party
// GPU effects on the CPU for the whole session when another CUDA context
// shows up in the process before it has set up its GPU effects.)
struct HostDevice
{
    virtual ~HostDevice() = default;
    virtual void* Context() = 0; // CUcontext
    virtual bool Acquire() = 0;  // exclusive access to the device
    virtual void Release() = 0;
    virtual void* Allocate(size_t bytes) = 0; // device memory, nullptr on failure
    virtual void Free(void* p) = 0;
};
#endif

class Engine
{
public:
    // CUDA: device 0, in a context of its own; Metal: the system default GPU.
    // Returns nullptr and sets *err when there is no usable GPU / driver.
    static std::unique_ptr<Engine> Create(std::string* err);
#ifdef _WIN32
    // CUDA in the host's context (see HostDevice). `dev` must outlive the engine.
    static std::unique_ptr<Engine> CreateOnHostDevice(HostDevice* dev, std::string* err);
#endif
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

    // Temporal supersampling (Quality High, RIFE): the output is the weighted
    // sum over samples of  mask * frame0(x + f0) + (1 - mask) * frame1(x + f1),
    // each sample's field upsampled to the frame size, i.e. the shutter
    // integrated from interpolated in-between frames. Works on the frames as
    // given (8-bit or float, no clamping). nFrames <= 4, all the same size.
    // showVectors: draw sample 0's frame0 motion instead.
    virtual bool ProcessSamples(const HostImage* frames, int nFrames, const TimeSample* samples, int nSamples,
                                void* out, size_t outRowBytes, bool showVectors, Stats* stats, std::string* err) = 0;

    // ProcessPair / Process with the motion estimated on the GPU from the
    // frames themselves (see FlowSettings). prev / next may be null (clip
    // edges; for the pair: the frame before A and the one after B, used by
    // FlowSettings::twoSided only). Engines without it return false and the
    // caller estimates the motion on the CPU instead.
    virtual bool ProcessPairGpuFlow(const HostImage* /*prev*/, const HostImage& /*A*/, const HostImage& /*B*/,
                                    const HostImage* /*next*/, double /*a*/, const FlowSettings& /*fs*/,
                                    void* /*out*/, size_t /*outRowBytes*/, const Settings& /*s*/, Stats* /*stats*/,
                                    std::string* err)
    {
        if (err)
            *err = "GPU motion not available";
        return false;
    }
    virtual bool ProcessGpuFlow(const HostImage* /*prev*/, const HostImage& /*cur*/, const HostImage* /*next*/,
                                const FlowSettings& /*fs*/, void* /*out*/, size_t /*outRowBytes*/,
                                const Settings& /*s*/, Stats* /*stats*/, std::string* err)
    {
        if (err)
            *err = "GPU motion not available";
        return false;
    }

    // Test hook (offline tools): motion a->b and b->a only, downloaded at the
    // analysis size in full-res pixels (2 x float32), as EstimateFlowPair.
    virtual bool EstimateFlowPairGpu(const HostImage& /*A*/, const HostImage& /*B*/, const FlowSettings& /*fs*/,
                                     float* /*fab*/, float* /*fba*/, int* /*fw*/, int* /*fh*/, Stats* /*stats*/,
                                     std::string* err)
    {
        if (err)
            *err = "GPU motion not available";
        return false;
    }
};

} // namespace oe::gpu
