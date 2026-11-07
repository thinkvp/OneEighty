#pragma once
// GPU optical-flow motion blur: NVIDIA Optical Flow hardware for motion, CUDA
// for the blur. Same algorithm as MotionBlur.cpp (CPU), working on device
// frames in Premiere's GPU layout (BGRA, 32-bit float or 16-bit half, 0..1).

#include <memory>
#include <string>

namespace oe::gpu
{

struct Frame
{
    void* ptr = nullptr;   // device pointer, pixels are BGRA
    size_t pitch = 0;      // bytes per row
    int width = 0, height = 0;
    bool half = false;     // false: 4 x float32, true: 4 x float16
};

struct Settings
{
    double halfWidth = 0.25; // synthetic blur half-width, source-frame intervals
    int flowWidth = 1920;    // width of the image fed to the optical-flow engine
    int flowGrid = 4;        // NVOF output grid: 1, 2 or 4 pixels per vector
    int perfLevel = 5;       // NVOF: 5 slow/best, 10 medium, 20 fast
    int maxSamples = 64;
    int mipThreshold = 48;   // samples before switching to the next mip level (0 = off)
    int maxMipLevel = 3;
    double centerOffset = 0; // wanted output time relative to cur, source-frame intervals (-0.5..0.5)
    bool showVectors = false; // draw the motion field instead of blurring
};

struct Stats
{
    double flowMs = 0, blurMs = 0, totalMs = 0;
};

class Engine
{
public:
    // cuContext: a CUcontext to work in (Premiere's), or nullptr for the
    // primary context of device 0. Returns nullptr and sets *err on failure
    // (e.g. no NVIDIA optical-flow hardware/driver).
    static std::unique_ptr<Engine> Create(void* cuContext, std::string* err);
    virtual ~Engine() = default;

    // prev/next may be null (clip edges). out must not alias the inputs.
    // stream: CUstream to run on (nullptr = default stream). Thread-safe
    // (calls are serialised per engine).
    virtual bool Process(const Frame* prev, const Frame& cur, const Frame* next, const Frame& out,
                         const Settings& s, void* stream, Stats* stats, std::string* err) = 0;

    // Two-frame interpolation: the output time lies `a` (0..1) of the way from
    // source frame A to the next source frame B. Motion is estimated both ways,
    // A is blurred at +a and B at -(1-a), blended by closeness. Validated as
    // better than the single-frame shift on both 50p and 59.94p.
    virtual bool ProcessPair(const Frame& A, const Frame& B, double a, const Frame& out, const Settings& s,
                             void* stream, Stats* stats, std::string* err) = 0;

    // Mean |x - y| per channel (0..255 scale) over a 64x64 sample grid; 0 = same frame.
    virtual double SampledDiff(const Frame& x, const Frame& y, void* stream) = 0;

    // Hybrid path: frames that live in host memory (8-bit BGRA, e.g. checked out
    // by the CPU effect) are uploaded, processed on the GPU and the result
    // written back to `out`. Device buffers are kept between calls.
    struct HostImage
    {
        const void* data = nullptr;
        size_t rowBytes = 0;
        int width = 0, height = 0;
        bool argb = false;    // 8-bit only: byte order A,R,G,B instead of B,G,R,A
        bool isFloat = false; // 4 x float32 B,G,R,A (Premiere 32-bit), copied as is: no clamping
    };
    // Two-frame interpolation (see ProcessPair).
    virtual bool ProcessHostPair(const HostImage& A, const HostImage& B, double a, void* out, size_t outRowBytes,
                                 const Settings& s, Stats* stats, std::string* err) = 0;
    // Single frame with optional neighbours (see Process).
    virtual bool ProcessHost(const HostImage* prev, const HostImage& cur, const HostImage* next, void* out,
                             size_t outRowBytes, const Settings& s, Stats* stats, std::string* err) = 0;

    // Same, but with motion supplied by the caller (e.g. OpenCV DIS on the CPU,
    // which copes better with fine repeating patterns than the NVOF hardware's
    // block matching). Flow: 2 x float32 per vector, velocity in full-res pixels
    // per source frame, any resolution (sampled bilinearly).
    struct HostFlow
    {
        const void* data = nullptr;
        size_t rowBytes = 0;
        int width = 0, height = 0;
    };
    // Two-frame interpolation; fa = motion at A, fb = motion at B (both forward in time).
    virtual bool ProcessHostPairFlow(const HostImage& A, const HostImage& B, double a, const HostFlow& fa,
                                     const HostFlow& fb, void* out, size_t outRowBytes, const Settings& s,
                                     Stats* stats, std::string* err) = 0;
    // Single frame with its motion v.
    virtual bool ProcessHostFlow(const HostImage& cur, const HostFlow& v, void* out, size_t outRowBytes,
                                 const Settings& s, Stats* stats, std::string* err) = 0;

    // Test helpers (offline harness): 8-bit BGRA host image <-> float device frame.
    virtual bool Upload8(const void* host, int width, int height, Frame* dev, std::string* err) = 0;
    virtual bool Download8(const Frame& dev, void* host, std::string* err) = 0;
    virtual void Free(Frame* dev) = 0;
};

} // namespace oe::gpu
