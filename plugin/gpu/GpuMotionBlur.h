#pragma once
// GPU motion blur (CUDA). Motion is supplied by the caller (OpenCV DIS on the
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

class Engine
{
public:
    // Primary CUDA context of device 0. Returns nullptr and sets *err when
    // there is no usable NVIDIA GPU / driver.
    static std::unique_ptr<Engine> Create(std::string* err);
    virtual ~Engine() = default;

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
};

} // namespace oe::gpu
