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

    // Test helpers (offline harness): 8-bit BGRA host image <-> float device frame.
    virtual bool Upload8(const void* host, int width, int height, Frame* dev, std::string* err) = 0;
    virtual bool Download8(const Frame& dev, void* host, std::string* err) = 0;
    virtual void Free(Frame* dev) = 0;
};

} // namespace oe::gpu
