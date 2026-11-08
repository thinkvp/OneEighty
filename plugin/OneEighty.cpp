/*
    OneEighty Motion Blur: main effect (After Effects API).

    Premiere lets this effect check out its input at sub-frame times, so it can
    read the in-between source frames (Premiere's GPU filter API can't: it only
    delivers whole timeline frames). Frames are fetched here, motion is
    estimated with OpenCV DIS on the CPU (Quality Normal / High = FAST / MEDIUM
    preset) and the blur runs on the CUDA engine when an NVIDIA GPU is
    available ("hybrid"), otherwise with OpenCV on the CPU.

    When high-frame-rate footage (50p, 59.94p) plays at 100% speed (or a
    slowed-down file is sped back up) on a lower rate timeline (25p), Premiere
    just drops frames, leaving each frame with far
    less motion blur than a normal 180 deg shutter. This effect reads the
    neighbouring source frames (Premiere supports checking out the layer at
    sub-frame times), estimates per-pixel motion with optical flow and smears
    each pixel along its path to add the missing shutter time.

    Debug log: %TEMP%\OneEighty.log
*/

#include "OneEighty.h"
#include "MotionBlur.h"
#include "gpu/GpuMotionBlur.h"
#include "UpdateCheck.h"

#include <opencv2/video/tracking.hpp> // DIS presets

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace
{

using oe::kPrTicksPerSecond;
using oe::Log;

// ----------------------------------------------------- source frame rate info
struct SourceInfo
{
    double fps = 0;        // source frame rate used for offsets/blur
    double mediaFps = 0;   // Premiere GetMediaFrameRate (= rate conformed to the timeline)
    double origFps = 0;    // Premiere GetOriginalClipFrameRate (= real source rate)
    double speed = 1.0;    // Premiere GetClipSpeed
    const char* source = "";
};

SourceInfo GetSourceInfo(PF_InData* in_data, PF_ParamDef* params[])
{
    SourceInfo si;
    if (in_data->appl_id == kAppID_Premiere && in_data->pica_basicP)
    {
        const void* p = nullptr;
        if (in_data->pica_basicP->AcquireSuite(kPFUtilitySuite, kPFUtilitySuiteVersion, &p) == kSPNoError && p)
        {
            const PF_UtilitySuite* u = static_cast<const PF_UtilitySuite*>(p);
            PrTime ticks = 0;
            if (u->GetMediaFrameRate && u->GetMediaFrameRate(in_data->effect_ref, &ticks) == PF_Err_NONE && ticks > 0)
                si.mediaFps = kPrTicksPerSecond / (double)ticks;
            ticks = 0;
            if (u->GetOriginalClipFrameRate &&
                u->GetOriginalClipFrameRate(in_data->effect_ref, &ticks) == PF_Err_NONE && ticks > 0)
                si.origFps = kPrTicksPerSecond / (double)ticks;
            double sp = 0;
            if (u->GetClipSpeed && u->GetClipSpeed(in_data->effect_ref, &sp) == PF_Err_NONE && sp != 0.0)
                si.speed = std::fabs(sp);
            in_data->pica_basicP->ReleaseSuite(kPFUtilitySuite, kPFUtilitySuiteVersion);
        }
    }

    const double user = params[OE_SRC_FPS]->u.fs_d.value;
    if (user > 0)
    {
        si.fps = user;
        si.source = "user";
    }
    else if (si.origFps > 0)
    {
        si.fps = si.origFps;
        si.source = "host original clip rate";
    }
    else
    {
        si.fps = 2.0 * in_data->time_scale / std::max<double>(1.0, in_data->time_step);
        si.source = "guess 2x timeline";
    }
    return si;
}

// Byte order of 8-bit pixels: Premiere hands AE-API effects BGRA, AE uses ARGB.
oe::ChannelOrder PixelOrder(PF_InData* in_data, PF_LayerDef* world, unsigned* fourccOut)
{
    *fourccOut = 0;
    if (in_data->appl_id != kAppID_Premiere || !in_data->pica_basicP)
        return oe::kARGB;
    oe::ChannelOrder order = oe::kBGRA;
    const void* p = nullptr;
    if (in_data->pica_basicP->AcquireSuite(kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, &p) == kSPNoError && p)
    {
        const PF_PixelFormatSuite1* pf = static_cast<const PF_PixelFormatSuite1*>(p);
        PrPixelFormat fmt = PrPixelFormat_BGRA_4444_8u;
        if (pf->GetPixelFormat && pf->GetPixelFormat(world, &fmt) == PF_Err_NONE)
        {
            *fourccOut = (unsigned)fmt;
            if (fmt == PrPixelFormat_ARGB_4444_8u)
                order = oe::kARGB;
        }
        in_data->pica_basicP->ReleaseSuite(kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
    }
    return order;
}

// Premiere 32-bit float BGRA (declared in GlobalSetup); everything else here is 8-bit.
inline bool IsFloatFormat(unsigned fourcc) { return fourcc == (unsigned)PrPixelFormat_BGRA_4444_32f; }

inline cv::Mat Wrap(const PF_LayerDef* w, bool isFloat)
{
    return cv::Mat(w->height, w->width, isFloat ? CV_32FC4 : CV_8UC4, w->data, (size_t)w->rowbytes);
}

bool Valid(const PF_LayerDef* w) { return w && w->data && w->width > 0 && w->height > 0; }
bool SameSize(const PF_LayerDef* a, const PF_LayerDef* b) { return a->width == b->width && a->height == b->height; }

// ------------------------------------------------------------- PF commands
PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[], PF_LayerDef*)
{
    const oe::update::Status u = oe::update::Current();
    const char* upd = u.state == oe::update::State::Available ? "Update available: v"
                      : u.state == oe::update::State::UpToDate ? "Up to date (latest v"
                                                                : "Updates: not checked yet";
    const std::string tail = u.state == oe::update::State::Available ? u.latest
                             : u.state == oe::update::State::UpToDate ? u.latest + ")"
                                                                       : "";
    PF_SPRINTF(out_data->return_msg, "%s v%s\r%s\r\r%s%s\rgithub.com/thinkvp/OneEighty", OE_NAME,
               oe::update::CurrentVersion().c_str(), OE_DESCRIPTION, upd, tail.c_str());
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[], PF_LayerDef*)
{
    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);
    out_data->out_flags = PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_SEND_UPDATE_PARAMS_UI;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
    // Premiere: formats in order of preference. 32-bit float BGRA keeps the
    // clip in Premiere's 32-bit pipeline (no 8-bit banding on log/graded
    // footage, HDR values above 1.0 preserved); 8-bit BGRA is Premiere's native
    // 8-bit layout; 8-bit ARGB stays available as the implicit default.
    if (in_data->appl_id == kAppID_Premiere && in_data->pica_basicP)
    {
        const void* p = nullptr;
        if (in_data->pica_basicP->AcquireSuite(kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, &p) == kSPNoError && p)
        {
            const PF_PixelFormatSuite1* pf = static_cast<const PF_PixelFormatSuite1*>(p);
            if (pf->AddSupportedPixelFormat)
            {
                pf->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
                pf->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
            }
            in_data->pica_basicP->ReleaseSuite(kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
        }
        // Checked-out frames otherwise arrive as ARGB while input/output are
        // BGRA; ask for the render format so no per-frame reorder is needed
        // (Render still reorders defensively if they differ).
        p = nullptr;
        if (in_data->pica_basicP->AcquireSuite(kPFUtilitySuite, kPFUtilitySuiteVersion, &p) == kSPNoError && p)
        {
            const PF_UtilitySuite* u = static_cast<const PF_UtilitySuite*>(p);
            if (u->EffectWantsCheckedOutFramesToMatchRenderPixelFormat)
                u->EffectWantsCheckedOutFramesToMatchRenderPixelFormat(in_data->effect_ref);
            in_data->pica_basicP->ReleaseSuite(kPFUtilitySuite, kPFUtilitySuiteVersion);
        }
    }
    Log("GlobalSetup appl_id=%c%c%c%c version=%d.%d", (char)(in_data->appl_id >> 24), (char)(in_data->appl_id >> 16),
        (char)(in_data->appl_id >> 8), (char)in_data->appl_id, MAJOR_VERSION, MINOR_VERSION);
    return PF_Err_NONE;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[], PF_LayerDef*)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    // ---- main controls (order must match the OE_* indices)
    AEFX_CLR_STRUCT(def);
    // Slider 90-360: at or below the footage's own exposure (90 deg for 50p shot at 180,
    // 75 for 59.94p) there is no blur to add. Typed values 0-720 are still accepted.
    PF_ADD_FLOAT_SLIDERX("Shutter angle", 0, 720, 90, 360, 180, PF_Precision_INTEGER, 0, 0, OE_TARGET_SHUTTER_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Quality", OE_QUALITY_COUNT, OE_QUALITY_NORMAL, OE_QUALITY_CHOICES, OE_QUALITY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Exact frame timing", TRUE, 0, OE_EXACT_TIMING_DISK_ID);

    // ---- Advanced (collapsed)
    PF_ADD_TOPICX("Advanced", PF_ParamFlag_START_COLLAPSED, OE_ADV_START_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Source shutter angle", 1, 360, 1, 360, 180, PF_Precision_INTEGER, 0, 0,
                         OE_SRC_SHUTTER_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Source fps (0 = auto)", 0, 1000, 0, 120, 0, PF_Precision_HUNDREDTHS, 0, 0,
                         OE_SRC_FPS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("View", OE_VIEW_COUNT, OE_VIEW_RESULT, OE_VIEW_CHOICES, OE_VIEW_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Force CPU", FALSE, 0, OE_FORCE_CPU_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Write debug log", FALSE, 0, OE_LOG_DISK_ID);

    PF_ADD_BUTTON("Updates", "Check for updates", 0, PF_ParamFlag_SUPERVISE, OE_UPDATE_BUTTON_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(OE_ADV_END_DISK_ID);

    out_data->num_params = OE_NUM_PARAMS;
    return err;
}

// ---------------------------------------------------------------- GPU engine
// Hybrid path: one CUDA engine for the CPU effect (primary context of device
// 0). Created on first use; a failure is remembered (CPU maths then).
std::mutex gEngineMutex;
std::unique_ptr<oe::gpu::Engine> gEngine;
bool gEngineTried = false;

oe::gpu::Engine* HybridEngine()
{
    std::lock_guard<std::mutex> lock(gEngineMutex);
    if (!gEngineTried)
    {
        gEngineTried = true;
        std::string err;
        gEngine = oe::gpu::Engine::Create(&err);
        if (gEngine)
            Log("[hybrid] CUDA engine ready");
        else
            Log("[hybrid] GPU engine unavailable (%s) -> CPU maths", err.c_str());
    }
    return gEngine.get();
}

bool SameOrder(const oe::ChannelOrder& a, const oe::ChannelOrder& b)
{
    return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

// src (byte order `from`) returned in the output's byte order and bit depth
// (a copy only when something differs; float is 0..1 = 8-bit 0..255).
cv::Mat Conform(const cv::Mat& src, const oe::ChannelOrder& from, const oe::ChannelOrder& to, bool toFloat)
{
    cv::Mat m = src;
    if (toFloat && m.depth() != CV_32F)
        m.convertTo(m, CV_32FC4, 1.0 / 255.0);
    else if (!toFloat && m.depth() == CV_32F)
        m.convertTo(m, CV_8UC4, 255.0);
    if (SameOrder(from, to))
        return m;
    cv::Mat dst(m.size(), m.type());
    const int pairs[] = {from.r, to.r, from.g, to.g, from.b, to.b, from.a, to.a};
    cv::mixChannels(&m, 1, &dst, 1, pairs, 4);
    return dst;
}

// Mean |a-b| per channel (0..255) on a 64x64 sample grid; -1 if not comparable.
double SampledDiff(const cv::Mat& a, const cv::Mat& b)
{
    if (a.empty() || b.empty() || a.size() != b.size())
        return -1.0;
    double acc = 0;
    for (int j = 0; j < 64; ++j)
        for (int i = 0; i < 64; ++i)
        {
            const int x = (int)((i + 0.5) * a.cols / 64), y = (int)((j + 0.5) * a.rows / 64);
            if (a.depth() == CV_32F)
            {
                const float* p = a.ptr<float>(y) + 4 * x;
                const float* q = b.ptr<float>(y) + 4 * x;
                acc += 255.0 * (std::fabs(p[0] - q[0]) + std::fabs(p[1] - q[1]) + std::fabs(p[2] - q[2]));
            }
            else
            {
                const uchar* p = a.ptr<uchar>(y) + 4 * x;
                const uchar* q = b.ptr<uchar>(y) + 4 * x;
                acc += std::abs(p[0] - q[0]) + std::abs(p[1] - q[1]) + std::abs(p[2] - q[2]);
            }
        }
    return acc / (64.0 * 64.0 * 3.0);
}

std::atomic<int> gDiagRenders{0};

PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();

    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const int view = params[OE_VIEW]->u.pd.value;
    const bool forceCpu = params[OE_FORCE_CPU]->u.bd.value != 0;
    const bool exactTiming = params[OE_EXACT_TIMING]->u.bd.value != 0;
    const bool log = params[OE_LOG]->u.bd.value != 0;
    PF_LayerDef* curW = &params[OE_INPUT]->u.ld;
    oe::update::MaybeCheckInBackground(); // daily, background, never blocks (opt-out: see UpdateCheck.cpp)

    if (!Valid(curW) || !SameSize(curW, output))
        return PF_COPY(curW, output, NULL, NULL);

    const SourceInfo si = GetSourceInfo(in_data, params);
    const double timelineFps = (double)in_data->time_scale / std::max<A_long>(1, in_data->time_step);
    // Premiere gives an effect on a speed-changed clip its times in the clip's
    // own (media) time: at 200% each timeline frame advances the effect's clock
    // by two source frames (measured: S&Q 25p file at 200% -> current_time
    // steps of 2 frames while time_step stays one frame). So:
    //  - frame requests are spaced by the source rate alone (clipRate), and
    //  - the blur uses source frames per timeline second (srcRate = rate x speed).
    const double clipRate = si.fps;           // source frames per second of effect time
    const double srcRate = si.fps * si.speed; // source frames per second of timeline time

    // Times on a finer scale so sub-frame offsets are representable
    // (timeline scale 25: half a frame is 0.5 units).
    long long mult = 1000;
    while (mult > 1 && ((long long)std::llabs((long long)in_data->current_time) * mult > INT_MAX / 2 ||
                        (long long)in_data->time_scale * mult > INT_MAX / 2))
        mult /= 10;
    const long long S = (long long)in_data->time_scale * mult;
    const long long T = (long long)in_data->current_time * mult;
    const long long dt = std::max<long long>(1, std::llround((double)S / clipRate));

    // Frame plan. Exact: interpolate between source frames A = floor(p) and
    // B = A+1 around the output time p (source frames of clip time; assumes the
    // clip's in-point sits on a source frame - otherwise the whole clip shifts
    // by a constant < 1 frame, which keeps the cadence even).
    // Otherwise: Premiere's frame plus neighbours one source interval away.
    const double p = (double)in_data->current_time / in_data->time_scale * clipRate;
    const double phase = p - std::floor(p + 1e-6);
    const bool pairPlan = exactTiming;
    long long t1, t2; // checkout times (units of S)
    if (pairPlan)
    {
        // a quarter frame past each frame's timestamp: robust to round/floor
        t1 = T + std::llround((-phase + 0.25) * (double)S / clipRate);
        t2 = T + std::llround((1.0 - phase + 0.25) * (double)S / clipRate);
    }
    else
    {
        t1 = T - dt;
        t2 = T + dt;
    }

    PF_ParamDef d1, d2;
    AEFX_CLR_STRUCT(d1);
    AEFX_CLR_STRUCT(d2);
    const auto tFetch0 = Clock::now();
    const PF_Err e1 = PF_CHECKOUT_PARAM(in_data, OE_INPUT, (A_long)t1, (A_long)dt, (A_u_long)S, &d1);
    const auto tFetch1 = Clock::now();
    const PF_Err e2 = PF_CHECKOUT_PARAM(in_data, OE_INPUT, (A_long)t2, (A_long)dt, (A_u_long)S, &d2);
    const auto tFetch2 = Clock::now();
    const double fetch1Ms = std::chrono::duration<double, std::milli>(tFetch1 - tFetch0).count();
    const double fetch2Ms = std::chrono::duration<double, std::milli>(tFetch2 - tFetch1).count();
    const double fetchMs = fetch1Ms + fetch2Ms;
    const bool have1 = !e1 && Valid(&d1.u.ld) && SameSize(&d1.u.ld, curW);
    const bool have2 = !e2 && Valid(&d2.u.ld) && SameSize(&d2.u.ld, curW);

    // Premiere can hand the input, the checked-out frames and the output in
    // different formats (seen: input BGRA, checkouts ARGB). Bring every frame
    // into the output's byte order and bit depth (8-bit or 32-bit float).
    unsigned fourcc = 0, fcIn = 0, fc1 = 0, fc2 = 0;
    const oe::ChannelOrder order = PixelOrder(in_data, output, &fourcc);
    const oe::ChannelOrder ordIn = PixelOrder(in_data, curW, &fcIn);
    const oe::ChannelOrder ord1 = have1 ? PixelOrder(in_data, &d1.u.ld, &fc1) : order;
    const oe::ChannelOrder ord2 = have2 ? PixelOrder(in_data, &d2.u.ld, &fc2) : order;
    const bool outFloat = IsFloatFormat(fourcc);
    const cv::Mat cur = Conform(Wrap(curW, IsFloatFormat(fcIn)), ordIn, order, outFloat);
    const cv::Mat m1 = have1 ? Conform(Wrap(&d1.u.ld, IsFloatFormat(fc1)), ord1, order, outFloat) : cv::Mat();
    const cv::Mat m2 = have2 ? Conform(Wrap(&d2.u.ld, IsFloatFormat(fc2)), ord2, order, outFloat) : cv::Mat();
    cv::Mat out = Wrap(output, outFloat);
    char fmts[48];
    auto cc = [](unsigned f, int i) { return f ? (char)(f >> (8 * i)) : '?'; };
    snprintf(fmts, sizeof(fmts), "out=%c%c%c%c in=%c%c%c%c d1=%c%c%c%c", cc(fourcc, 0), cc(fourcc, 1), cc(fourcc, 2),
             cc(fourcc, 3), cc(fcIn, 0), cc(fcIn, 1), cc(fcIn, 2), cc(fcIn, 3), cc(fc1, 0), cc(fc1, 1), cc(fc1, 2),
             cc(fc1, 3));

    const double srcShutter = params[OE_SRC_SHUTTER]->u.fs_d.value;
    const double tgtShutter = params[OE_TARGET_SHUTTER]->u.fs_d.value;
    const double hw = oe::BlurHalfWidth(srcRate, timelineFps, srcShutter, tgtShutter);
    const int q = params[OE_QUALITY]->u.pd.value;
    const bool vectors = view == OE_VIEW_VECTORS;

    // Motion from OpenCV DIS on the CPU (copes with fine repeating patterns);
    // the blur on the GPU when there is one (the expensive part at UHD), else CPU.
    // Normal: DIS FAST preset (17 ms/pair at UHD, -0.4 dB on the synthetic test);
    // High: DIS MEDIUM (41 ms/pair; cleaner edges where people cross the background).
    const int disPreset = q == OE_QUALITY_HIGH ? cv::DISOpticalFlow::PRESET_MEDIUM : cv::DISOpticalFlow::PRESET_FAST;
    constexpr int kFlowWidth = 960; // motion analysis width
    oe::gpu::Engine* eng = nullptr;
    const bool bgra = order.b == 0 && order.g == 1 && order.r == 2;
    const bool argb = order.a == 0 && order.r == 1 && order.g == 2 && order.b == 3;
    const char* why = "";
    if (forceCpu)
        why = "Force CPU";
    else if (!bgra && !argb)
        why = "unsupported pixel order";
    else if (!(eng = HybridEngine()))
        why = "GPU engine unavailable";

    const auto tCompute0 = Clock::now();
    double disMs = 0; // CPU motion estimation time (hybrid path)
    const char* path = eng ? "hybrid" : "cpu";
    std::string gerr;
    bool done = false;
    const bool pair = pairPlan && have1 && have2;
    try
    {
        if (eng)
        {
            oe::gpu::Settings gs;
            gs.halfWidth = hw;
            gs.showVectors = vectors;
            const bool a8 = argb && !outFloat;
            const oe::gpu::HostImage hc{cur.data, cur.step, cur.cols, cur.rows, a8, outFloat};
            const oe::gpu::HostImage h1{m1.data, m1.step, m1.cols, m1.rows, a8, outFloat};
            const oe::gpu::HostImage h2{m2.data, m2.step, m2.cols, m2.rows, a8, outFloat};
            oe::gpu::Stats gst;
            const auto tFlow0 = Clock::now();
            if (pair)
            {
                cv::Mat fa, fbBack;
                oe::EstimateFlowPair(m1, m2, order, kFlowWidth, fa, fbBack, disPreset); // A->B and B->A
                const cv::Mat fb = -fbBack;                                               // B's forward motion
                disMs = std::chrono::duration<double, std::milli>(Clock::now() - tFlow0).count();
                const oe::gpu::HostFlow hfa{fa.data, fa.step, fa.cols, fa.rows};
                const oe::gpu::HostFlow hfb{fb.data, fb.step, fb.cols, fb.rows};
                done = eng->ProcessPair(h1, h2, phase, hfa, hfb, out.data, out.step, gs, &gst, &gerr);
            }
            else
            {
                cv::Mat v;
                int nv = 0;
                if (have2)
                {
                    v = oe::EstimateFlow(cur, m2, order, kFlowWidth, disPreset);
                    ++nv;
                }
                if (have1)
                {
                    const cv::Mat b = oe::EstimateFlow(cur, m1, order, kFlowWidth, disPreset);
                    v = v.empty() ? cv::Mat(-b) : cv::Mat(v - b);
                    ++nv;
                }
                disMs = std::chrono::duration<double, std::milli>(Clock::now() - tFlow0).count();
                if (nv == 2)
                    v *= 0.5;
                if (nv == 0)
                {
                    cur.copyTo(out);
                    done = true;
                }
                else
                {
                    const oe::gpu::HostFlow hv{v.data, v.step, v.cols, v.rows};
                    done = eng->Process(hc, hv, out.data, out.step, gs, &gst, &gerr);
                }
            }
            if (!done)
                path = "cpu (gpu error)";
        }
        if (!done)
        {
            oe::BlurSettings bs;
            bs.halfWidth = hw;
            bs.flowWidth = kFlowWidth;
            bs.disPreset = disPreset;
            bs.showVectors = vectors;
            if (pair && !vectors && phase > 1e-3)
            {
                // two-frame interpolation on the CPU: A at +a, B at -(1-a), blended
                cv::Mat outA(cur.size(), cur.type()), outB(cur.size(), cur.type());
                bs.centerOffset = phase;
                oe::FlowMotionBlur(cv::Mat(), m1, m2, outA, order, bs);
                bs.centerOffset = phase - 1.0;
                oe::FlowMotionBlur(m1, m2, cv::Mat(), outB, order, bs);
                cv::addWeighted(outA, 1.0 - phase, outB, phase, 0.0, out);
            }
            else if (pair)
                oe::FlowMotionBlur(cv::Mat(), m1, m2, out, order, bs); // on a source frame: A with A->B motion
            else
                oe::FlowMotionBlur(m1, cur, m2, out, order, bs);
        }
    }
    catch (const cv::Exception& e)
    {
        Log("OpenCV error: %s", e.what());
        cur.copyTo(out);
    }
    catch (const std::exception& e)
    {
        Log("error: %s", e.what());
        cur.copyTo(out);
    }
    const double computeMs = std::chrono::duration<double, std::milli>(Clock::now() - tCompute0).count();

    // Diagnostics for the first renders: which fetched frame equals Premiere's
    // own frame at this time (expect cur == A for a pair plan).
    char diag[128] = "";
    if (log && have1 && have2 && gDiagRenders.fetch_add(1) < 30)
        snprintf(diag, sizeof(diag), " | diff cur-d1=%.2f cur-d2=%.2f d1-d2=%.2f", SampledDiff(cur, m1),
                 SampledDiff(cur, m2), SampledDiff(m1, m2));

    ERR2(PF_CHECKIN_PARAM(in_data, &d1));
    ERR2(PF_CHECKIN_PARAM(in_data, &d2));

    if (log)
    {
        const double totalMs = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count();
        Log("[%s] render%s%s%s forceCpu=%d t=%d/%u view=%d %dx%d %s | srcFps=%.3f (%s) speed=%.2f timeline=%.3f "
            "(src frames/out %.2f; host orig=%.3f media=%.3f) | "
            "timing=%s p=%.3f a=%.3f d1:%s d2:%s | halfWidth=%.3f q=%d | fetch=%.1fms (d1 %.1fms, d2 %.1fms) compute=%.1fms "
            "(cpu motion %.1fms) total=%.1fms%s%s%s",
            path, *why ? " (" : "", why, *why ? ")" : "", forceCpu ? 1 : 0, in_data->current_time, in_data->time_scale, view,
            output->width, output->height, fmts, si.fps, si.source, si.speed, timelineFps, srcRate / timelineFps,
            si.origFps, si.mediaFps, pair ? "exact-pair" : "nearest", p, pair ? phase : 0.0,
            have1 ? "ok" : "MISSING", have2 ? "ok" : "MISSING", hw, q, fetchMs, fetch1Ms, fetch2Ms,
            computeMs, disMs, totalMs, diag, gerr.empty() ? "" : " gpu error: ", gerr.c_str());
    }
    return err ? err : err2;
}

} // namespace

namespace
{

// Shows update status in Effect Controls by renaming the "Advanced" group and
// the Updates button (PF_UpdateParamUI may change a parameter's name; allowed
// during PF_Cmd_UPDATE_PARAMS_UI and PF_Cmd_USER_CHANGED_PARAM).
std::atomic<int> gLabelLogCount{0};

void ApplyUpdateLabels(PF_InData* in_data, PF_ParamDef* params[])
{
    const oe::update::Status u = oe::update::Current();
    const bool avail = u.state == oe::update::State::Available;
    // Param names are capped at 31 characters (PF_MAX_EFFECT_PARAM_NAME_LEN): keep the version on the button label.
    const std::string group = avail ? "Advanced - update available" : "Advanced";
    const std::string button = avail ? "Update v" + u.latest + " available" : "Updates";

    const void* p = nullptr;
    if (!in_data->pica_basicP ||
        in_data->pica_basicP->AcquireSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, &p) != kSPNoError || !p)
    {
        if (gLabelLogCount.fetch_add(1) < 20)
            Log("[update] PF Param Utils Suite unavailable: can't rename controls");
        return;
    }
    const PF_ParamUtilsSuite3* pu = static_cast<const PF_ParamUtilsSuite3*>(p);
    const struct
    {
        int index;
        const std::string* name;
    } items[] = {{OE_ADV_START, &group}, {OE_UPDATE_BUTTON, &button}};
    for (const auto& it : items)
    {
        if (!params[it.index] || *it.name == params[it.index]->PF_DEF_NAME)
            continue;
        PF_ParamDef def = *params[it.index];
        PF_STRNNCPY(def.PF_DEF_NAME, it.name->c_str(), sizeof(def.PF_DEF_NAME));
        const PF_Err e = pu->PF_UpdateParamUI(in_data->effect_ref, it.index, &def);
        if (gLabelLogCount.fetch_add(1) < 20)
            Log("[update] rename param %d to '%s': %s (err %d)", it.index, it.name->c_str(), e ? "FAILED" : "ok", (int)e);
    }
    in_data->pica_basicP->ReleaseSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
}

std::atomic<int> gUpdateUiCount{0};

PF_Err UpdateParamsUI(PF_InData* in_data, PF_OutData*, PF_ParamDef* params[])
{
    if (gUpdateUiCount.fetch_add(1) < 10)
        Log("[update] PF_Cmd_UPDATE_PARAMS_UI received");
    oe::update::MaybeCheckInBackground();
    ApplyUpdateLabels(in_data, params);
    return PF_Err_NONE;
}

PF_Err UserChangedParam(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], void* extra)
{
    const PF_UserChangedParamExtra* ex = static_cast<const PF_UserChangedParamExtra*>(extra);
    if (!ex || ex->param_index != OE_UPDATE_BUTTON)
        return PF_Err_NONE;
    const oe::update::Status u = oe::update::CheckNow(6000);
    const std::string cur = oe::update::CurrentVersion();
    // A normal Windows message box: Premiere's own effect message (PF_OutFlag_DISPLAY_ERROR_MESSAGE)
    // is styled as a red error, which reads wrong for "you're up to date".
    HWND owner = GetActiveWindow();
    char msg[512];
    if (u.state == oe::update::State::Available)
    {
        snprintf(msg, sizeof(msg),
                 "OneEighty Motion Blur v%s is available (you have v%s).\n\nOpen the download page?", u.latest.c_str(),
                 cur.c_str());
        if (MessageBoxA(owner, msg, "OneEighty Motion Blur", MB_YESNO | MB_ICONINFORMATION) == IDYES)
            ShellExecuteA(nullptr, "open", u.pageUrl.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
    else if (u.state == oe::update::State::UpToDate)
    {
        snprintf(msg, sizeof(msg), "OneEighty Motion Blur v%s is up to date.", cur.c_str());
        MessageBoxA(owner, msg, "OneEighty Motion Blur", MB_OK | MB_ICONINFORMATION);
    }
    else
    {
        snprintf(msg, sizeof(msg), "Couldn't check for updates (%s).\n\nReleases: github.com/thinkvp/OneEighty/releases",
                 u.error.c_str());
        MessageBoxA(owner, msg, "OneEighty Motion Blur", MB_OK | MB_ICONWARNING);
    }
    ApplyUpdateLabels(in_data, params);
    return PF_Err_NONE;
}

} // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                                     const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK; // the macro below assigns to `result`
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, OE_NAME, OE_MATCH_NAME, OE_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", "https://example.invalid/oneeighty");
    return result;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output,
                  void* extra)
{
    PF_Err err = PF_Err_NONE;
    try
    {
        switch (cmd)
        {
        case PF_Cmd_ABOUT: err = About(in_data, out_data, params, output); break;
        case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(in_data, out_data, params, output); break;
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data, params, output); break;
        case PF_Cmd_RENDER: err = Render(in_data, out_data, params, output); break;
        case PF_Cmd_UPDATE_PARAMS_UI: err = UpdateParamsUI(in_data, out_data, params); break;
        case PF_Cmd_USER_CHANGED_PARAM: err = UserChangedParam(in_data, out_data, params, extra); break;
        default: break;
        }
    }
    catch (PF_Err& thrown)
    {
        err = thrown; // never throw into the host
    }
    catch (...)
    {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
