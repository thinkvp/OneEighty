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
    // Time of the wanted output frame relative to `cur`, in source-frame
    // intervals (-0.5..0.5). Non-zero when source and timeline rates don't
    // divide evenly (59.94p on 25p): the blur window is shifted to that time.
    double centerOffset = 0.0;
    bool showVectors = false; // draw the motion field instead of blurring
    int disPreset = 2;        // cv::DISOpticalFlow preset (0 ultrafast, 1 fast, 2 medium)
    bool refineEdges = true;  // clean up motion spilling past moving objects' outlines
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

// Dense motion from `from` to `to` (OpenCV DIS) estimated at
// `flowWidth` and returned at that reduced size (CV_32FC2), in full-res pixels:
// from(x) ~ to(x + f(x)). Used to feed the GPU blur in the hybrid path.
// disPreset: cv::DISOpticalFlow::PRESET_ULTRAFAST / PRESET_FAST / PRESET_MEDIUM
// refineEdges: clean up motion spilling past moving objects' outlines (see
// RefineFlowEdges in MotionBlur.cpp).
cv::Mat EstimateFlow(const cv::Mat& from, const cv::Mat& to, const ChannelOrder& order, int flowWidth,
                     int disPreset = 2 /* PRESET_MEDIUM */, bool refineEdges = true);

// Both directions between a and b (a->b and b->a, same convention as
// EstimateFlow), each frame downscaled once.
void EstimateFlowPair(const cv::Mat& a, const cv::Mat& b, const ChannelOrder& order, int flowWidth, cv::Mat& fab,
                      cv::Mat& fba, int disPreset = 2 /* PRESET_MEDIUM */, bool refineEdges = true);

// Colour for a motion vector (hue = direction, saturation = speed / norm), RGB 0..1.
void VectorColour(float vx, float vy, float norm, float rgb[3]);

// prev/next may be empty (clip edges). All images CV_8UC4, same size; out must
// be allocated and must not alias the inputs.
void FlowMotionBlur(const cv::Mat& prev, const cv::Mat& cur, const cv::Mat& next, cv::Mat& out,
                    const ChannelOrder& order, const BlurSettings& s, BlurStats* stats = nullptr);

} // namespace oe
