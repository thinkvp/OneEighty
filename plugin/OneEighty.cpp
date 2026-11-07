/*
    OneEighty Motion Blur: CPU effect (After Effects API). In Premiere with the
    CUDA renderer, OneEightyGPU.cpp renders instead; this path is the fallback.

    When high-frame-rate footage (50p, 59.94p) plays at 100% speed on a lower
    rate timeline (25p), Premiere just drops frames, leaving each frame with far
    less motion blur than a normal 180 deg shutter. This effect reads the
    neighbouring source frames (Premiere supports checking out the layer at
    sub-frame times), estimates per-pixel motion with optical flow and smears
    each pixel along its path to add the missing shutter time.

    Debug log: %TEMP%\OneEighty.log
*/

#include "OneEighty.h"
#include "MotionBlur.h"

#include <algorithm>
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

inline cv::Mat Wrap(const PF_LayerDef* w)
{
    return cv::Mat(w->height, w->width, CV_8UC4, w->data, (size_t)w->rowbytes);
}

bool Valid(const PF_LayerDef* w) { return w && w->data && w->width > 0 && w->height > 0; }
bool SameSize(const PF_LayerDef* a, const PF_LayerDef* b) { return a->width == b->width && a->height == b->height; }

// out = sum(w[i] * src[i]); channel order agnostic. (Test mode only.)
void Blend(const cv::Mat src[3], const float wts[3], cv::Mat& out)
{
    cv::Mat acc;
    src[1].convertTo(acc, CV_32FC4, wts[1]);
    for (int i : {0, 2})
        if (wts[i] != 0.0f)
            cv::scaleAdd(cv::Mat_<cv::Vec4f>(src[i]), wts[i], acc, acc);
    acc.convertTo(out, CV_8UC4);
}

// ------------------------------------------------------------- PF commands
PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[], PF_LayerDef*)
{
    PF_SPRINTF(out_data->return_msg, "%s v%d.%d\r%s", OE_NAME, MAJOR_VERSION, MINOR_VERSION, OE_DESCRIPTION);
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[], PF_LayerDef*)
{
    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);
    out_data->out_flags = PF_OutFlag_WIDE_TIME_INPUT;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
    Log("GlobalSetup appl_id=%c%c%c%c version=%d.%d", (char)(in_data->appl_id >> 24), (char)(in_data->appl_id >> 16),
        (char)(in_data->appl_id >> 8), (char)in_data->appl_id, MAJOR_VERSION, MINOR_VERSION);
    return PF_Err_NONE;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[], PF_LayerDef*)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Mode", OE_MODE_COUNT, OE_MODE_FLOW, OE_MODE_CHOICES, OE_MODE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Source fps (0 = auto)", 0, 1000, 0, 120, 0, PF_Precision_HUNDREDTHS, 0, 0,
                         OE_SRC_FPS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Source shutter angle", 1, 360, 1, 360, 180, PF_Precision_INTEGER, 0, 0,
                         OE_SRC_SHUTTER_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Target shutter angle", 0, 720, 0, 360, 180, PF_Precision_INTEGER, 0, 0,
                         OE_TARGET_SHUTTER_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Motion analysis", OE_QUALITY_COUNT, OE_QUALITY_NORMAL, OE_QUALITY_CHOICES, OE_QUALITY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Write debug log", TRUE, 0, OE_LOG_DISK_ID);

    out_data->num_params = OE_NUM_PARAMS;
    return err;
}

PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();

    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const int mode = params[OE_MODE]->u.pd.value;
    const bool log = params[OE_LOG]->u.bd.value != 0;
    PF_LayerDef* curW = &params[OE_INPUT]->u.ld;

    if (mode == OE_MODE_OFF || !Valid(curW) || !SameSize(curW, output))
        return PF_COPY(curW, output, NULL, NULL);

    const SourceInfo si = GetSourceInfo(in_data, params);
    const double timelineFps = (double)in_data->time_scale / std::max<A_long>(1, in_data->time_step);

    // One source frame expressed in timeline time, on a finer time scale so the
    // sub-frame offset is representable (timeline scale 25 => half frame = 0.5).
    long long mult = 1000;
    while (mult > 1 && ((long long)std::llabs((long long)in_data->current_time) * mult > INT_MAX / 2 ||
                        (long long)in_data->time_scale * mult > INT_MAX / 2))
        mult /= 10;
    const long long S = (long long)in_data->time_scale * mult;
    const long long T = (long long)in_data->current_time * mult;
    const long long dt = std::max<long long>(1, std::llround((double)S / (si.fps * si.speed)));

    PF_ParamDef prev, next;
    AEFX_CLR_STRUCT(prev);
    AEFX_CLR_STRUCT(next);
    const bool needPrev = mode != OE_MODE_SHOW_NEXT;
    const bool needNext = mode != OE_MODE_SHOW_PREVIOUS;
    PF_Err ePrev = PF_Err_NONE, eNext = PF_Err_NONE;
    if (needPrev)
        ePrev = PF_CHECKOUT_PARAM(in_data, OE_INPUT, (A_long)(T - dt), (A_long)dt, (A_u_long)S, &prev);
    if (needNext)
        eNext = PF_CHECKOUT_PARAM(in_data, OE_INPUT, (A_long)(T + dt), (A_long)dt, (A_u_long)S, &next);

    const bool havePrev = needPrev && !ePrev && Valid(&prev.u.ld) && SameSize(&prev.u.ld, curW);
    const bool haveNext = needNext && !eNext && Valid(&next.u.ld) && SameSize(&next.u.ld, curW);

    unsigned fourcc = 0;
    const oe::ChannelOrder order = PixelOrder(in_data, curW, &fourcc);

    const cv::Mat cur = Wrap(curW);
    const cv::Mat mPrev = havePrev ? Wrap(&prev.u.ld) : cv::Mat();
    const cv::Mat mNext = haveNext ? Wrap(&next.u.ld) : cv::Mat();
    cv::Mat out = Wrap(output);

    oe::BlurSettings bs;
    bs.halfWidth = oe::BlurHalfWidth(si.fps * si.speed, timelineFps, params[OE_SRC_SHUTTER]->u.fs_d.value,
                                     params[OE_TARGET_SHUTTER]->u.fs_d.value);
    const int q = params[OE_QUALITY]->u.pd.value;
    bs.flowWidth = q == OE_QUALITY_FAST ? 480 : q == OE_QUALITY_HIGH ? 1920 : 960;
    oe::BlurStats st;

    try
    {
        switch (mode)
        {
        case OE_MODE_FLOW:
            oe::FlowMotionBlur(mPrev, cur, mNext, out, order, bs, &st);
            break;
        case OE_MODE_BLEND_CENTRED:
        {
            const cv::Mat src[3] = {havePrev ? mPrev : cur, cur, haveNext ? mNext : cur};
            const float w[3] = {0.25f, 0.5f, 0.25f};
            Blend(src, w, out);
            break;
        }
        case OE_MODE_SHOW_PREVIOUS: (havePrev ? mPrev : cur).copyTo(out); break;
        case OE_MODE_SHOW_NEXT: (haveNext ? mNext : cur).copyTo(out); break;
        default: cur.copyTo(out); break;
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

    ERR2(PF_CHECKIN_PARAM(in_data, &prev));
    ERR2(PF_CHECKIN_PARAM(in_data, &next));

    if (log)
    {
        const double totalMs = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count();
        Log("render t=%d/%u mode=%d %dx%d fmt=%c%c%c%c | srcFps=%.3f (%s) speed=%.2f timeline=%.3f dt=%lld/%lld | "
            "prev:%s next:%s | halfWidth=%.3f flowW=%d | flow=%.1fms blur=%.1fms total=%.1fms samples=%.2f maxSpeed=%.1fpx",
            in_data->current_time, in_data->time_scale, mode, output->width, output->height, (char)(fourcc >> 24),
            (char)(fourcc >> 16), (char)(fourcc >> 8), (char)fourcc, si.fps, si.source, si.speed, timelineFps, dt, S,
            havePrev ? "ok" : (needPrev ? "MISSING" : "-"), haveNext ? "ok" : (needNext ? "MISSING" : "-"),
            bs.halfWidth, bs.flowWidth, st.flowMs, st.blurMs, totalMs, st.avgSamples, st.maxSpeedPx);
    }
    return err ? err : err2;
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

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
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
