#pragma once
// Quality "Extra High": the shutter integrated from RIFE in-between frames.
// For an output time inside the source frames, N samples are spread evenly
// over the shutter window [t - halfWidth, t + halfWidth] (source-frame units);
// each sample is an interpolated frame between the two source frames around
// its time (RIFE motion and blend mask, rife/RifeFlow), warped and blended on
// the GPU from the original frames (gpu/GpuMotionBlur::ProcessSamples) and the
// samples are averaged. Validated offline: 40.5 dB on the synthetic 50p -> 25p
// test (High: 39.2 dB), and clean backgrounds next to a fast ball.

#include "RifeFlow.h"
#include "../gpu/GpuMotionBlur.h"

namespace oe::rife
{

struct ShutterFrames
{
    // Source frames at integer times -1, 0, 1, 2 relative to frame A (BGRA,
    // 8-bit or float, all the same size; same byte order). A and B are
    // required; P (before A) and N (after B) only when the window reaches them,
    // otherwise samples there are clamped to A / B.
    const cv::Mat* P = nullptr;
    const cv::Mat* A = nullptr;
    const cv::Mat* B = nullptr;
    const cv::Mat* N = nullptr;
};

struct ShutterStats
{
    int samples = 0;
    int pairs = 0;     // RIFE frame pairs used (1..3)
    double rifeMs = 0; // network (incl. upload/download)
    double gpuMs = 0;  // warping + averaging
};

// t: output time relative to A (0..1 = between A and B). Writes the result in
// the frames' layout to out (8-bit: same byte order as given; argb8 says
// whether that is A,R,G,B). maxPixels: RIFE analysis size limit.
bool RenderShutter(Engine& rife, gpu::Engine& gpu, const ShutterFrames& f, const ChannelOrder& order, bool argb8,
                   double t, double halfWidth, int nSamples, int maxPixels, bool showVectors, void* out,
                   size_t outRowBytes, ShutterStats* stats, std::string* err);

// Whether the window around t needs P / N (so the caller can fetch them).
inline bool NeedsPrev(double t, double halfWidth) { return t - halfWidth < 0.0; }
inline bool NeedsNext(double t, double halfWidth) { return t + halfWidth > 1.0; }

} // namespace oe::rife
