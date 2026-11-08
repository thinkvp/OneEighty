// Offline harness for the plugin's GPU path: raw BGRA8 frames in, raw BGRA8
// frames out. Motion from OpenCV DIS on the CPU, blur on the GPU, exactly as
// the effect does it inside Premiere.
//   oe_gpu_offline in.bgra W H srcFps outFps out.bgra [mipThreshold]
// Environment:
//   OE_EXACT=0     exact frame timing off: Premiere's frame round(k*r) plus its
//                  neighbours (default: on, two-frame interpolation)
//   OE_QUALITY=high  DIS MEDIUM preset (default: Normal, DIS FAST)
//   OE_QUALITY=extra Extra High: shutter from RIFE in-between frames
//                  (model: OE_RIFE_MODEL or rife-v4.6 next to this exe)
//   OE_SAMPLES=n   Extra High samples per output frame (default 6, as the effect)
//   OE_RIFE_PIXELS=n  Extra High RIFE analysis size limit in pixels (default 2100000)
//   OE_FLOAT=1     run as 32-bit float (Premiere 32-bit); 2 = input x1.5 to
//                  check values above 1.0 pass through unclipped
//   OE_ARGB=1      feed ARGB bytes instead of BGRA (After Effects order)
//   OE_VIEW=1      motion-vector view
//   OE_EDGEFIX=0   skip the motion-boundary clean-up (RefineFlowEdges)
#include "../MotionBlur.h"
#include "../gpu/GpuMotionBlur.h"
#include "../rife/RifeShutter.h"

#include <opencv2/video/tracking.hpp> // DIS presets

#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
char Env(const char* name)
{
    char v[8] = {0};
    size_t len = 0;
    getenv_s(&len, v, sizeof(v), name);
    return v[0];
}

// B,G,R,A <-> A,R,G,B (reversing the channel order works both ways)
void ReverseChannels(cv::Mat& m)
{
    const int pairs[] = {0, 3, 1, 2, 2, 1, 3, 0};
    cv::Mat d(m.size(), m.type());
    cv::mixChannels(&m, 1, &d, 1, pairs, 4);
    m = d;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::fprintf(stderr, "usage: oe_gpu_offline in.bgra W H srcFps outFps out.bgra [mipThreshold]\n");
        return 2;
    }
    const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
    const double srcFps = std::atof(argv[4]), outFps = std::atof(argv[5]);
    oe::gpu::Settings s;
    s.halfWidth = oe::BlurHalfWidth(srcFps, outFps, 180.0, 180.0);
    if (argc > 7)
        s.mipThreshold = std::atoi(argv[7]);
    s.showVectors = Env("OE_VIEW") == '1';
    const bool exact = Env("OE_EXACT") != '0';
    const int preset = Env("OE_QUALITY") == 'h' ? cv::DISOpticalFlow::PRESET_MEDIUM : cv::DISOpticalFlow::PRESET_FAST;
    const bool extra = Env("OE_QUALITY") == 'e';
    int nSamples = 6;
    {
        char v[8] = {0};
        size_t len = 0;
        getenv_s(&len, v, sizeof(v), "OE_SAMPLES");
        if (v[0])
            nSamples = std::max(1, std::atoi(v));
    }
    int rifePixels = 2100000;
    {
        char v[16] = {0};
        size_t len = 0;
        getenv_s(&len, v, sizeof(v), "OE_RIFE_PIXELS");
        if (v[0])
            rifePixels = std::max(4096, std::atoi(v));
    }
    const bool argb = Env("OE_ARGB") == '1';
    const bool edgeFix = Env("OE_EDGEFIX") != '0';
    const char fl = Env("OE_FLOAT");
    const bool useFloat = fl == '1' || fl == '2';
    const float gain = fl == '2' ? 1.5f : 1.0f;
    const oe::ChannelOrder order = argb && !useFloat ? oe::kARGB : oe::kBGRA;

    std::string err;
    auto eng = oe::gpu::Engine::Create(&err);
    if (!eng)
    {
        std::fprintf(stderr, "GPU engine: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<oe::rife::Engine> rife;
    if (extra)
    {
        wchar_t model[260] = {0};
        size_t len = 0;
        _wgetenv_s(&len, model, 260, L"OE_RIFE_MODEL");
        const auto tr = std::chrono::steady_clock::now();
        rife = oe::rife::Engine::Create(model[0] ? std::wstring(model) : oe::rife::DefaultModelDir(), &err);
        if (!rife)
        {
            std::fprintf(stderr, "RIFE: %s\n", err.c_str());
            return 1;
        }
        std::printf("RIFE ready in %.0f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count());
    }

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<cv::Mat> src;
    FILE* fi = nullptr;
    if (fopen_s(&fi, argv[1], "rb") || !fi)
        return 1;
    for (;;)
    {
        cv::Mat m(H, W, CV_8UC4);
        if (std::fread(m.data, 1, frameBytes, fi) != frameBytes)
            break;
        if (useFloat)
            m.convertTo(m, CV_32FC4, gain / 255.0);
        else if (argb)
            ReverseChannels(m);
        src.push_back(m);
    }
    std::fclose(fi);
    std::printf("%zu source frames, halfWidth=%.4f, %s timing, %s, %s\n", src.size(), s.halfWidth,
                exact ? "exact" : "nearest",
                extra ? "Extra High" : preset == cv::DISOpticalFlow::PRESET_MEDIUM ? "High" : "Normal",
                useFloat ? "float" : argb ? "8-bit ARGB" : "8-bit BGRA");

    FILE* fo = nullptr;
    if (fopen_s(&fo, argv[6], "wb") || !fo)
        return 1;
    cv::Mat out(H, W, useFloat ? CV_32FC4 : CV_8UC4), out8(H, W, CV_8UC4);
    auto host = [&](const cv::Mat& m) {
        return oe::gpu::HostImage{m.data, m.step, m.cols, m.rows, argb && !useFloat, useFloat};
    };
    const double r = srcFps / outFps;
    double disMs = 0, gpuMs = 0, outMax = 0, above1 = 0, comps = 0;
    int k = 0, timed = 0;
    for (;; ++k)
    {
        oe::gpu::Stats st;
        bool ok;
        const auto t0 = std::chrono::steady_clock::now();
        double tFlow = 0;
        if (extra)
        {
            // frame A and the output time t relative to it, as the effect does
            const long A = exact ? (long)std::floor(k * r + 1e-9) : std::lround(k * r);
            const double t = exact ? k * r - (double)A : 0.0;
            if (A + 1 >= (long)src.size())
                break;
            oe::rife::ShutterFrames f;
            f.A = &src[A];
            f.B = &src[A + 1];
            if (A > 0 && oe::rife::NeedsPrev(t, s.halfWidth))
                f.P = &src[A - 1];
            if (A + 2 < (long)src.size() && oe::rife::NeedsNext(t, s.halfWidth))
                f.N = &src[A + 2];
            oe::rife::ShutterStats rs;
            ok = oe::rife::RenderShutter(*rife, *eng, f, order, argb && !useFloat, t, s.halfWidth, nSamples, rifePixels,
                                         s.showVectors, out.data, out.step, &rs, &err);
            tFlow = rs.rifeMs;
            st.gpuMs = rs.gpuMs;
        }
        else if (exact)
        {
            const long A = (long)std::floor(k * r + 1e-9);
            if (A + 1 >= (long)src.size())
                break;
            cv::Mat fa, fbBack;
            oe::EstimateFlowPair(src[A], src[A + 1], order, 960, fa, fbBack, preset, edgeFix);
            const cv::Mat fb = -fbBack;
            tFlow = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ok = eng->ProcessPair(host(src[A]), host(src[A + 1]), k * r - (double)A,
                                  {fa.data, fa.step, fa.cols, fa.rows}, {fb.data, fb.step, fb.cols, fb.rows}, out.data,
                                  out.step, s, &st, &err);
        }
        else
        {
            const long i = std::lround(k * r);
            if (i >= (long)src.size())
                break;
            cv::Mat v;
            int nv = 0;
            if (i + 1 < (long)src.size())
            {
                v = oe::EstimateFlow(src[i], src[i + 1], order, 960, preset, edgeFix);
                ++nv;
            }
            if (i > 0)
            {
                const cv::Mat b = oe::EstimateFlow(src[i], src[i - 1], order, 960, preset, edgeFix);
                v = v.empty() ? cv::Mat(-b) : cv::Mat(v - b);
                ++nv;
            }
            if (nv == 2)
                v *= 0.5;
            tFlow = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ok = eng->Process(host(src[i]), {v.data, v.step, v.cols, v.rows}, out.data, out.step, s, &st, &err);
        }
        if (!ok)
        {
            std::fprintf(stderr, "frame %d: %s\n", k, err.c_str());
            return 1;
        }
        if (k > 0) // the first frame includes one-off setup
        {
            disMs += tFlow;
            gpuMs += st.gpuMs;
            ++timed;
        }
        if (useFloat)
        {
            double mn, mx;
            cv::minMaxLoc(out.reshape(1), &mn, &mx);
            outMax = std::max(outMax, mx);
            above1 += cv::countNonZero(out.reshape(1) > 1.0);
            comps += (double)W * H * 4;
            out.convertTo(out8, CV_8UC4, 255.0 / gain);
        }
        else
        {
            out.copyTo(out8);
            if (argb)
                ReverseChannels(out8);
        }
        std::fwrite(out8.data, 1, frameBytes, fo);
    }
    std::fclose(fo);
    if (useFloat)
        std::printf("float output: max value %.3f, %.2f%% of components above 1.0 (input gain %.1f)\n", outMax,
                    100.0 * above1 / std::max(1.0, comps), gain);
    if (timed)
        std::printf("wrote %d frames; avg DIS motion %.1f ms, GPU blur %.2f ms per frame\n", k, disMs / timed,
                    gpuMs / timed);
    return 0;
}
