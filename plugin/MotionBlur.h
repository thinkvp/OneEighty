#pragma once
// Optical-flow motion blur (host independent; port of prototype/oneeighty.py "flow" mode).

#include <opencv2/core.hpp>

namespace oe
{

struct ChannelOrder
{
    int r, g, b, a; // byte index of each channel within a 4-byte pixel
};
constexpr ChannelOrder kBGRA{2, 1, 0, 3};
constexpr ChannelOrder kARGB{1, 2, 3, 0};

struct BlurSettings
{
    double halfWidth = 0.25; // synthetic blur half-width, in source-frame intervals
    int flowWidth = 960;     // width at which motion is estimated
    int maxSamples = 64;     // cap on samples per pixel
    // Long smears are sampled from a downscaled copy (mip level) so cost stays
    // bounded: a pixel switches to the next level once its smear exceeds
    // mipThreshold samples. 0 disables (always full resolution).
    int mipThreshold = 32; // UHD test: 2x faster than exact, 49.6 dB vs exact (24 was 3.6x but 40.9 dB)
    int maxMipLevel = 3;
};

struct BlurStats
{
    double flowMs = 0, blurMs = 0, avgSamples = 0, maxSpeedPx = 0;
};

// Half-width of the blur to add (source-frame units). The source frame already
// holds srcShutter/360 of a source interval; we want targetShutter/360 of an
// output interval. Widths combine like standard deviations ("variance" rule,
// which matched a real shutter best in prototype/validate.py).
double BlurHalfWidth(double srcFps, double outFps, double srcShutterDeg, double targetShutterDeg);

// prev/next may be empty (clip edges). All images CV_8UC4, same size; out must
// be allocated and must not alias the inputs.
void FlowMotionBlur(const cv::Mat& prev, const cv::Mat& cur, const cv::Mat& next, cv::Mat& out,
                    const ChannelOrder& order, const BlurSettings& s, BlurStats* stats = nullptr);

} // namespace oe
