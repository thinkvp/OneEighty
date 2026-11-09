// Offline harness for the plugin's CPU path (MotionBlur.cpp): raw BGRA frames
// in, raw BGRA frames out, processed exactly as the effect does without a GPU.
//   oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter [mipThreshold]]
// Environment:
//   OE_EXACT=0       exact frame timing off: Premiere's frame round(k*r) plus its
//                    neighbours (default: on, two-frame interpolation)
//   OE_QUALITY=high  DIS MEDIUM preset (default: Normal, DIS FAST)
//   OE_FLOAT=1       run as 32-bit float (Premiere 32-bit), write 8-bit
//   OE_EDGEFIX=0     skip the motion-boundary clean-up (RefineFlowEdges)
#include "../MotionBlur.h"

#include <opencv2/video/tracking.hpp> // DIS presets

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
char Env(const char* name)
{
#ifdef _WIN32
    char v[8] = {0};
    size_t len = 0;
    getenv_s(&len, v, sizeof(v), name);
    return v[0];
#else
    const char* v = std::getenv(name);
    return v ? v[0] : 0;
#endif
}

FILE* Open(const char* path, const char* mode)
{
#ifdef _WIN32
    FILE* f = nullptr;
    return fopen_s(&f, path, mode) == 0 ? f : nullptr;
#else
    return std::fopen(path, mode);
#endif
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::fprintf(stderr, "usage: oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter [mipThr]]\n");
        return 2;
    }
    const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
    const double srcFps = std::atof(argv[4]), outFps = std::atof(argv[5]);
    const double srcSh = argc > 7 ? std::atof(argv[7]) : 180.0, tgtSh = argc > 8 ? std::atof(argv[8]) : 180.0;
    const bool exact = Env("OE_EXACT") != '0';
    const bool useFloat = Env("OE_FLOAT") == '1';
    const size_t frameBytes = (size_t)W * H * 4;

    FILE* fi = Open(argv[1], "rb");
    if (!fi)
        return 1;
    std::vector<cv::Mat> src;
    for (;;)
    {
        cv::Mat m(H, W, CV_8UC4);
        if (std::fread(m.data, 1, frameBytes, fi) != frameBytes)
            break;
        if (useFloat)
            m.convertTo(m, CV_32FC4, 1.0 / 255.0);
        src.push_back(m);
    }
    std::fclose(fi);

    oe::BlurSettings s;
    s.halfWidth = oe::BlurHalfWidth(srcFps, outFps, srcSh, tgtSh);
    s.disPreset = Env("OE_QUALITY") == 'h' ? cv::DISOpticalFlow::PRESET_MEDIUM : cv::DISOpticalFlow::PRESET_FAST;
    s.refineEdges = Env("OE_EDGEFIX") != '0';
    if (argc > 9)
        s.mipThreshold = std::atoi(argv[9]); // 0 = exact (no mip levels)
    std::printf("%zu source frames, halfWidth=%.4f, %s timing\n", src.size(), s.halfWidth, exact ? "exact" : "nearest");

    FILE* fo = nullptr;
    if (!(fo = Open(argv[6], "wb")))
        return 1;
    const int type = useFloat ? CV_32FC4 : CV_8UC4;
    cv::Mat out(H, W, type), outA(H, W, type), outB(H, W, type), out8(H, W, CV_8UC4);
    const double r = srcFps / outFps;
    double flowMs = 0, blurMs = 0;
    int k = 0;
    for (;; ++k)
    {
        oe::BlurStats st;
        if (exact)
        {
            // two-frame interpolation: A at +a, B at -(1-a), blended (as the effect's CPU path)
            const long A = (long)std::floor(k * r + 1e-9);
            if (A + 1 >= (long)src.size())
                break;
            const double a = k * r - (double)A;
            s.centerOffset = a;
            oe::FlowMotionBlur(cv::Mat(), src[A], src[A + 1], a > 1e-3 ? outA : out, oe::kBGRA, s, &st);
            if (a > 1e-3)
            {
                s.centerOffset = a - 1.0;
                oe::FlowMotionBlur(src[A], src[A + 1], cv::Mat(), outB, oe::kBGRA, s, &st);
                cv::addWeighted(outA, 1.0 - a, outB, a, 0.0, out);
            }
        }
        else
        {
            const long i = std::lround(k * r);
            if (i >= (long)src.size())
                break;
            const cv::Mat empty;
            s.centerOffset = 0.0;
            oe::FlowMotionBlur(i > 0 ? src[i - 1] : empty, src[i], i + 1 < (long)src.size() ? src[i + 1] : empty, out,
                               oe::kBGRA, s, &st);
        }
        flowMs += st.flowMs;
        blurMs += st.blurMs;
        if (useFloat)
            out.convertTo(out8, CV_8UC4, 255.0);
        std::fwrite((useFloat ? out8 : out).data, 1, frameBytes, fo);
    }
    std::fclose(fo);
    if (k)
        std::printf("wrote %d frames; avg flow %.1f ms, blur %.1f ms per frame\n", k, flowMs / k, blurMs / k);
    return 0;
}
