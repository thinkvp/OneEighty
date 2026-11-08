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
    OE_QUALITY,        // Normal / High / Extra High
    OE_EXACT_TIMING,
    OE_ADV_START,      // ---- "Advanced" (collapsed)
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
    OE_QUALITY_EXTRA,
    OE_QUALITY_COUNT = OE_QUALITY_EXTRA
};
// Normal / High: motion estimated on the CPU (OpenCV DIS at 960 px, FAST /
// MEDIUM preset), blur on the GPU. Extra High: the shutter integrated from RIFE
// in-between frames (GPU via Vulkan + CUDA); falls back to High without them.
// Appended last so projects saved with Normal / High keep their setting.
#define OE_QUALITY_CHOICES "Normal|High|Extra High"

namespace oe
{
constexpr double kPrTicksPerSecond = 254016000000.0;

// Defined in MotionBlur.cpp (see MotionBlur.h).
double BlurHalfWidth(double srcFps, double outFps, double srcShutterDeg, double targetShutterDeg);

// Appends a line to %TEMP%\OneEighty.log (Log.cpp).
void Log(const char* fmt, ...);
} // namespace oe
