// Offline harness for the GPU path: raw BGRA8 frames in, raw BGRA8 frames out.
//   oe_gpu_offline in.bgra W H srcFps outFps out.bgra [flowWidth grid perf mipThreshold]
// Frames are uploaded as 32-bit float BGRA (Premiere's GPU format) and processed
// exactly as the plugin will: frame round(k*srcFps/outFps) plus its neighbours.
#include "../MotionBlur.h" // BlurHalfWidth
#include "../gpu/GpuMotionBlur.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::fprintf(stderr, "usage: oe_gpu_offline in.bgra W H srcFps outFps out.bgra [flowWidth grid perf mipThr]\n");
        return 2;
    }
    const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
    const double srcFps = std::atof(argv[4]), outFps = std::atof(argv[5]);
    oe::gpu::Settings s;
    s.halfWidth = oe::BlurHalfWidth(srcFps, outFps, 180.0, 180.0);
    if (argc > 7) s.flowWidth = std::atoi(argv[7]);
    if (argc > 8) s.flowGrid = std::atoi(argv[8]);
    if (argc > 9) s.perfLevel = std::atoi(argv[9]);
    if (argc > 10) s.mipThreshold = std::atoi(argv[10]);

    std::string err;
    auto eng = oe::gpu::Engine::Create(nullptr, &err);
    if (!eng)
    {
        std::fprintf(stderr, "GPU engine: %s\n", err.c_str());
        return 1;
    }

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<unsigned char> buf(frameBytes);
    std::vector<oe::gpu::Frame> src;
    std::vector<std::vector<unsigned char>> hostSrc; // for OE_CADENCE=3
    FILE* fi = nullptr;
    if (fopen_s(&fi, argv[1], "rb") || !fi)
        return 1;
    while (std::fread(buf.data(), 1, frameBytes, fi) == frameBytes)
    {
        oe::gpu::Frame f;
        if (!eng->Upload8(buf.data(), W, H, &f, &err))
        {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        src.push_back(f);
        hostSrc.push_back(buf);
    }
    std::fclose(fi);

    oe::gpu::Frame out;
    eng->Upload8(buf.data(), W, H, &out, &err); // allocate an output frame
    std::printf("%zu source frames, halfWidth=%.4f flowWidth=%d grid=%d perf=%d mip=%d\n", src.size(), s.halfWidth,
                s.flowWidth, s.flowGrid, s.perfLevel, s.mipThreshold);

    FILE* fo = nullptr;
    if (fopen_s(&fo, argv[6], "wb") || !fo)
        return 1;
    const double r = srcFps / outFps;
    // OE_CADENCE=1: shift the nearest frame to the exact output time (k/outFps)
    // OE_CADENCE=2: two-frame interpolation between the frames either side (plugin default)
    // otherwise: nearest source frame's moment (what Premiere does)
    char cad[8] = {0};
    size_t cadLen = 0;
    getenv_s(&cadLen, cad, sizeof(cad), "OE_CADENCE");
    const bool cadence = cad[0] == '1';
    const bool pair = cad[0] == '2';
    const bool hostPair = cad[0] == '3' || cad[0] == '4'; // hybrid: frames in host memory (CPU effect path)
    const bool disMotion = cad[0] == '4';                   // ...with CPU (OpenCV DIS) motion, GPU blur
    char seqEnv[8] = {0};
    size_t seqLen = 0;
    getenv_s(&seqLen, seqEnv, sizeof(seqEnv), "OE_SEQFLOW");
    const bool seqFlow = seqEnv[0] == '1';
    double disMsTotal = 0;
    int disCount = 0;
    char argbEnv[8] = {0};
    size_t argbLen = 0;
    getenv_s(&argbLen, argbEnv, sizeof(argbEnv), "OE_ARGB"); // OE_ARGB=1: feed the hybrid path ARGB bytes
    const bool argb = argbEnv[0] == '1';
    auto swapBGRA_ARGB = [](std::vector<unsigned char>& v, bool toArgb) {
        for (size_t i = 0; i + 3 < v.size(); i += 4)
        {
            const unsigned char b = v[i], g = v[i + 1], r = v[i + 2], a = v[i + 3];
            if (toArgb) { v[i] = a; v[i + 1] = r; v[i + 2] = g; v[i + 3] = b; }
            else { v[i] = v[i + 3]; v[i + 3] = b; v[i + 1] = v[i + 2]; v[i + 2] = g; (void)r; (void)a; }
        }
    };
    if (hostPair && argb)
        for (auto& f : hostSrc)
            swapBGRA_ARGB(f, true);
    // OE_FLOAT=1: hybrid path with 32-bit float frames (Premiere 32-bit);
    // OE_FLOAT=2: same, input values x1.5 (HDR-style >1.0) to check nothing clips.
    char floatEnv[8] = {0};
    size_t floatLen = 0;
    getenv_s(&floatLen, floatEnv, sizeof(floatEnv), "OE_FLOAT");
    const bool useFloat = hostPair && (floatEnv[0] == '1' || floatEnv[0] == '2');
    const float gain = floatEnv[0] == '2' ? 1.5f : 1.0f;
    std::vector<cv::Mat> hostF;
    if (useFloat)
        for (auto& f : hostSrc)
        {
            cv::Mat m;
            cv::Mat(H, W, CV_8UC4, f.data()).convertTo(m, CV_32FC4, gain / 255.0);
            hostF.push_back(m);
        }
    cv::Mat outF(H, W, CV_32FC4);
    double outMax = 0, outAbove1 = 0, outPixels = 0;
    char view[8] = {0};
    size_t viewLen = 0;
    getenv_s(&viewLen, view, sizeof(view), "OE_VIEW"); // OE_VIEW=1: motion-vector view
    s.showVectors = view[0] == '1';
    double flowMs = 0, blurMs = 0, totMs = 0;
    int k = 0, timed = 0;
    for (;; ++k)
    {
        const long i = std::lround(k * r);
        if (i >= (long)src.size())
            break;
        const oe::gpu::Frame* prev = i > 0 ? &src[i - 1] : nullptr;
        const oe::gpu::Frame* next = i + 1 < (long)src.size() ? &src[i + 1] : nullptr;
        oe::gpu::Stats st;
        s.centerOffset = cadence ? k * r - (double)i : 0.0;
        bool ok;
        if (hostPair)
        {
            const long A = (long)std::floor(k * r + 1e-9);
            if (A + 1 >= (long)src.size())
                break;
            using HI = oe::gpu::Engine::HostImage;
            HI ha{hostSrc[A].data(), (size_t)W * 4, W, H, argb}, hb{hostSrc[A + 1].data(), (size_t)W * 4, W, H, argb};
            void* outPtr = buf.data();
            size_t outRB = (size_t)W * 4;
            if (useFloat)
            {
                ha = HI{hostF[A].data, hostF[A].step, W, H, false, true};
                hb = HI{hostF[A + 1].data, hostF[A + 1].step, W, H, false, true};
                outPtr = outF.data;
                outRB = outF.step;
            }
            if (disMotion)
            {
                const cv::Mat ma = useFloat ? hostF[A] : cv::Mat(H, W, CV_8UC4, hostSrc[A].data());
                const cv::Mat mb = useFloat ? hostF[A + 1] : cv::Mat(H, W, CV_8UC4, hostSrc[A + 1].data());
                const oe::ChannelOrder ord = argb ? oe::kARGB : oe::kBGRA;
                cv::Mat fa, fbBack;
                const auto td0 = std::chrono::steady_clock::now();
                if (seqFlow) // OE_SEQFLOW=1: the two estimates one after the other (old behaviour, for timing)
                {
                    fa = oe::EstimateFlow(ma, mb, ord, 960);
                    fbBack = oe::EstimateFlow(mb, ma, ord, 960);
                }
                else
                    oe::EstimateFlowPair(ma, mb, ord, 960, fa, fbBack);
                disMsTotal += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                ++disCount;
                const cv::Mat fb = -fbBack;
                const oe::gpu::Engine::HostFlow hfa{fa.data, fa.step, fa.cols, fa.rows};
                const oe::gpu::Engine::HostFlow hfb{fb.data, fb.step, fb.cols, fb.rows};
                ok = eng->ProcessHostPairFlow(ha, hb, k * r - (double)A, hfa, hfb, outPtr, outRB, s, &st, &err);
            }
            else
                ok = eng->ProcessHostPair(ha, hb, k * r - (double)A, outPtr, outRB, s, &st, &err);
            if (ok && useFloat)
            {
                double mn, mx;
                cv::minMaxLoc(outF.reshape(1), &mn, &mx);
                outMax = std::max(outMax, mx);
                outAbove1 += cv::countNonZero(outF.reshape(1) > 1.0);
                outPixels += (double)W * H * 4;
                cv::Mat(H, W, CV_8UC4, buf.data()) = cv::Scalar::all(0);
                outF.convertTo(cv::Mat(H, W, CV_8UC4, buf.data()), CV_8UC4, 255.0 / gain);
            }
            if (ok)
            {
                if (argb)
                    swapBGRA_ARGB(buf, false);
                std::fwrite(buf.data(), 1, frameBytes, fo);
                continue;
            }
        }
        else if (pair)
        {
            const long A = (long)std::floor(k * r + 1e-9);
            const double a = k * r - (double)A;
            ok = A + 1 < (long)src.size() ? eng->ProcessPair(src[A], src[A + 1], a, out, s, nullptr, &st, &err)
                                          : eng->Process(&src[A - 1], src[A], nullptr, out, s, nullptr, &st, &err);
        }
        else
            ok = eng->Process(prev, src[i], next, out, s, nullptr, &st, &err);
        if (!ok)
        {
            std::fprintf(stderr, "frame %d: %s\n", k, err.c_str());
            return 1;
        }
        if (k > 0) // first frame includes one-off setup
        {
            flowMs += st.flowMs;
            blurMs += st.blurMs;
            totMs += st.totalMs;
            ++timed;
        }
        eng->Download8(out, buf.data(), &err);
        std::fwrite(buf.data(), 1, frameBytes, fo);
    }
    std::fclose(fo);
    if (useFloat)
        std::printf("float output: max value %.3f, %.2f%% of components above 1.0 (input gain %.1f)\n", outMax,
                    100.0 * outAbove1 / std::max(1.0, outPixels), gain);
    if (disCount)
        std::printf("DIS motion (%s): avg %.1f ms per frame pair\n", seqFlow ? "sequential" : "parallel pair",
                    disMsTotal / disCount);
    if (timed)
        std::printf("wrote %d frames; GPU avg flow %.2f ms, blur %.2f ms, total %.2f ms per frame\n", k,
                    flowMs / timed, blurMs / timed, totMs / timed);
    for (auto& f : src)
        eng->Free(&f);
    eng->Free(&out);
    return 0;
}
