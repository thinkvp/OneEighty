#pragma once
// RIFE (Real-time Intermediate Flow Estimation, v4.6 network, MIT) run with
// ncnn on the GPU through Vulkan, used by Quality "Extra High".
//
// RIFE builds a frame at any time t between two frames as
//     mask * warp(frame0, flow0) + (1 - mask) * warp(frame1, flow1)
// OneEighty takes only the motion (flow0 / flow1) and the blend mask from the
// network and does the warping and blending itself on the original frames
// (gpu/GpuMotionBlur), so 10-bit / log / 32-bit float footage keeps its
// precision and values above 1.0 survive. RIFE itself sees an RGB copy
// clamped to 0..1 at a reduced "analysis" size.

#include "../MotionBlur.h" // ChannelOrder

#include <memory>
#include <string>
#include <vector>

namespace oe::rife
{

struct Field
{
    cv::Mat flow; // CV_32FC4 at the analysis size: (f0x, f0y, f1x, f1y) in analysis pixels; sample frame0 at x + f0
    cv::Mat mask; // CV_32FC1 at the analysis size: weight of frame0 (frame1 gets 1 - mask)
};

class Engine
{
public:
    // modelDir: folder holding flownet.param / flownet.bin. Returns nullptr and
    // sets *err when Vulkan, a GPU or the model is missing.
    static std::unique_ptr<Engine> Create(const std::wstring& modelDir, std::string* err);
    virtual ~Engine() = default;

    // Interpolation fields between frames a and b (BGRA, 8-bit or float, same
    // size) for each time t in ts (0..1, 0 = a). maxPixels: the analysis size is
    // the frame scaled down (never up) to at most this many pixels. Thread-safe
    // (serialised).
    virtual bool Estimate(const cv::Mat& a, const cv::Mat& b, const ChannelOrder& order, int maxPixels,
                          const std::vector<float>& ts, std::vector<Field>* out, std::string* err) = 0;
};

// Folder of the model next to the plugin binary (<plugin dir>\rife-v4.6).
std::wstring DefaultModelDir();

} // namespace oe::rife
