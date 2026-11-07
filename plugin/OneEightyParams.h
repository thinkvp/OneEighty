#pragma once
// Shared by the CPU effect (After Effects API, OneEighty.cpp) and the Premiere
// GPU filter (OneEightyGPU.cpp). No SDK includes here: the two sides compile
// against different (partly conflicting) Adobe headers.

#define OE_NAME        "OneEighty Motion Blur"
#define OE_MATCH_NAME  "OneEighty MotionBlur"
#define OE_CATEGORY    "OneEighty"
#define OE_DESCRIPTION "Optical-flow shutter blur for high-frame-rate footage on a lower-rate timeline."

// Keep in sync with AE_Effect_Version in OneEightyPiPL.r
#define MAJOR_VERSION 0
#define MINOR_VERSION 3
#define BUG_VERSION   0
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 1

// Parameter indices (0 = input layer, as in the AE API)
enum
{
    OE_INPUT = 0,
    OE_MODE,
    OE_SRC_FPS,
    OE_SRC_SHUTTER,
    OE_TARGET_SHUTTER,
    OE_QUALITY,
    OE_LOG,
    OE_NUM_PARAMS
};

enum
{
    OE_MODE_DISK_ID = 1,
    OE_SRC_FPS_DISK_ID,
    OE_LOG_DISK_ID,
    OE_SRC_SHUTTER_DISK_ID,
    OE_TARGET_SHUTTER_DISK_ID,
    OE_QUALITY_DISK_ID,
};

// Popup entries are 1-based
enum
{
    OE_MODE_FLOW = 1,
    OE_MODE_BLEND_CENTRED,
    OE_MODE_SHOW_PREVIOUS,
    OE_MODE_SHOW_NEXT,
    OE_MODE_OFF,
    OE_MODE_COUNT = OE_MODE_OFF
};
#define OE_MODE_CHOICES "Optical flow blur|Blend 3 frames (test)|Show previous source frame|Show next source frame|Off"

enum
{
    OE_QUALITY_FAST = 1,
    OE_QUALITY_NORMAL,
    OE_QUALITY_HIGH,
    OE_QUALITY_COUNT = OE_QUALITY_HIGH
};
#define OE_QUALITY_CHOICES "Fast|Normal|High"

namespace oe
{
constexpr double kPrTicksPerSecond = 254016000000.0;

// Defined in MotionBlur.cpp (see MotionBlur.h).
double BlurHalfWidth(double srcFps, double outFps, double srcShutterDeg, double targetShutterDeg);

// Appends a line to %TEMP%\OneEighty.log (Log.cpp).
void Log(const char* fmt, ...);
} // namespace oe
