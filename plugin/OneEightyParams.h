#pragma once
// Effect parameters and shared helpers. No SDK includes here (kept free of
// Adobe headers so any translation unit can use it).

#define OE_NAME        "OneEighty Motion Blur"
#define OE_MATCH_NAME  "OneEighty MotionBlur"
#define OE_CATEGORY    "OneEighty"
#define OE_DESCRIPTION "Optical-flow shutter blur for high-frame-rate footage on a lower-rate timeline."

// Keep in sync with AE_Effect_Version in OneEightyPiPL.r
#define MAJOR_VERSION 0
#define MINOR_VERSION 9
#define BUG_VERSION   0
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 1

// Parameter indices in UI order (0 = input layer; group start/end markers
// count as parameters in the AE API).
enum
{
    OE_INPUT = 0,
    OE_TARGET_SHUTTER, // "Shutter angle": the look to simulate at the timeline rate
    OE_QUALITY,        // Normal / High
    OE_EXACT_TIMING,
    OE_ADV_START,      // ---- "Advanced" (collapsed)
    OE_RIFE_SAMPLES,   // High: RIFE in-between frames per output frame
    OE_RIFE_SIZE,      // High: RIFE analysis size
    OE_SRC_SHUTTER,
    OE_SRC_FPS,
    OE_VIEW,
    OE_FORCE_CPU,
    OE_LOG,
    OE_UPDATE_BUTTON,  // "Check for updates" button (label shows when one is available)
    OE_ADV_END,        // ----
    OE_NUM_PARAMS
};

// Disk IDs identify parameters in saved projects: never reuse a retired ID
// (1 = v0.1-v0.3 "Mode" popup, 6 = v0.4-v0.7 "Quality" popup with Fast,
// 9 = v0.4 "Render" popup, 13 = test-build "Check for updates automatically"
// checkbox). Projects saved with a retired ID open with that control's default.
enum
{
    OE_SRC_FPS_DISK_ID = 2,
    OE_LOG_DISK_ID = 3,
    OE_SRC_SHUTTER_DISK_ID = 4,
    OE_TARGET_SHUTTER_DISK_ID = 5,
    OE_VIEW_DISK_ID = 7,
    OE_EXACT_TIMING_DISK_ID = 8,
    OE_FORCE_CPU_DISK_ID = 10,
    OE_ADV_START_DISK_ID = 11,
    OE_ADV_END_DISK_ID = 12,
    OE_UPDATE_BUTTON_DISK_ID = 14,
    OE_QUALITY_DISK_ID = 15,
    OE_RIFE_SAMPLES_DISK_ID = 16,
    OE_RIFE_SIZE_DISK_ID = 17,
};

// Rendering: Premiere's GPU filter API only delivers whole timeline frames
// (requests are floored to the timeline grid), so it can't see the in-between
// source frames. The effect therefore fetches frames through the AE API
// (sub-frame layer checkout works), estimates motion on the CPU and blurs on
// the GPU when an NVIDIA GPU is available ("hybrid"), otherwise on the CPU.
// "Force CPU" is for troubleshooting.

// Popup entries are 1-based
enum
{
    OE_VIEW_RESULT = 1,
    OE_VIEW_VECTORS,
    OE_VIEW_COUNT = OE_VIEW_VECTORS
};
#define OE_VIEW_CHOICES "Result|Motion vectors"

enum
{
    OE_QUALITY_NORMAL = 1,
    OE_QUALITY_HIGH,
    OE_QUALITY_COUNT = OE_QUALITY_HIGH
};
// Normal: DIS motion at 960 px (MEDIUM preset), on the GPU with the blur (CPU
// OpenCV DIS without a GPU engine). High: the shutter integrated from RIFE
// in-between frames (GPU via Vulkan + CUDA / Metal); falls back to Normal
// without them. Before v0.10 this was Normal (DIS FAST) / High (DIS MEDIUM) /
// Extra High (RIFE): with motion on the GPU, FAST saved only ~4 ms per UHD
// frame for 0.4 dB, so it went. Same disk ID: saved v0.9 clips aren't migrated.
#define OE_QUALITY_CHOICES "Normal|High"

// High samples: RIFE in-between frames averaged per output frame (one
// network pass each). 6: same look as 8 on a fast ball at 2x zoom, -0.2 dB on
// the 59.94p test, +0.0 on 50p, ~20% faster (8: 40.4 / 40.5 dB, 4: 39.8 /
// 40.4 dB). Fewer = faster but the blur can break into steps on fast motion.
#define OE_RIFE_SAMPLES_DEFAULT 6
#define OE_RIFE_SAMPLES_MIN 2
#define OE_RIFE_SAMPLES_MAX 32
#define OE_RIFE_SAMPLES_SLIDER_MAX 16

// High analysis size: the frame is scaled down (never up) to at most
// this many pixels before RIFE sees it; RIFE time scales roughly with pixel
// count. Below ~2 MP at UHD cost 0.9-1.5 dB on the synthetic tests.
enum
{
    OE_RIFE_SIZE_540 = 1,
    OE_RIFE_SIZE_720,
    OE_RIFE_SIZE_1080,
    OE_RIFE_SIZE_1440,
    OE_RIFE_SIZE_COUNT = OE_RIFE_SIZE_1440
};
#define OE_RIFE_SIZE_CHOICES "540p (fastest)|720p|1080p|1440p (slowest)"

namespace oe
{
inline int RifeMaxPixels(int sizeChoice)
{
    switch (sizeChoice)
    {
        case OE_RIFE_SIZE_540: return 960 * 540;
        case OE_RIFE_SIZE_720: return 1280 * 720;
        case OE_RIFE_SIZE_1440: return 2560 * 1440;
        default: return 2100000; // 1080p (about 1920 x 1080, the pre-0.10 fixed size)
    }
}
} // namespace oe

namespace oe
{
constexpr double kPrTicksPerSecond = 254016000000.0;

// Defined in MotionBlur.cpp (see MotionBlur.h).
double BlurHalfWidth(double srcFps, double outFps, double srcShutterDeg, double targetShutterDeg);

// Appends a line to the debug log (Log.cpp; path in Platform.h).
void Log(const char* fmt, ...);
} // namespace oe
