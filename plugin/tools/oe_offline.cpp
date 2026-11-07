// Offline harness for MotionBlur.cpp: raw BGRA frames in, raw BGRA frames out.
//   oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter [mipThreshold]]
// Output frame k uses source frame round(k*srcFps/outFps) and its neighbours,
// exactly like the plugin does inside Premiere.
#include "../MotionBlur.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::fprintf(stderr, "usage: oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter]\n");
        return 2;
    }
    const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
    const double srcFps = std::atof(argv[4]), outFps = std::atof(argv[5]);
    const double srcSh = argc > 7 ? std::atof(argv[7]) : 180.0, tgtSh = argc > 8 ? std::atof(argv[8]) : 180.0;
    const size_t frameBytes = (size_t)W * H * 4;

    FILE* fi = nullptr;
    if (fopen_s(&fi, argv[1], "rb") || !fi)
        return 1;
    std::vector<cv::Mat> src;
    for (;;)
    {
        cv::Mat m(H, W, CV_8UC4);
        if (std::fread(m.data, 1, frameBytes, fi) != frameBytes)
            break;
        src.push_back(m);
    }
    std::fclose(fi);

    oe::BlurSettings s;
    s.halfWidth = oe::BlurHalfWidth(srcFps, outFps, srcSh, tgtSh);
    if (argc > 9)
        s.mipThreshold = std::atoi(argv[9]); // 0 = exact (no mip levels)
    std::printf("%zu source frames, halfWidth=%.4f\n", src.size(), s.halfWidth);

    FILE* fo = nullptr;
    if (fopen_s(&fo, argv[6], "wb") || !fo)
        return 1;
    const double r = srcFps / outFps;
    cv::Mat out(H, W, CV_8UC4);
    double flowMs = 0, blurMs = 0;
    int k = 0;
    for (;; ++k)
    {
        const long i = std::lround(k * r);
        if (i >= (long)src.size())
            break;
        const cv::Mat empty;
        const cv::Mat& prev = i > 0 ? src[i - 1] : empty;
        const cv::Mat& next = i + 1 < (long)src.size() ? src[i + 1] : empty;
        oe::BlurStats st;
        oe::FlowMotionBlur(prev, src[i], next, out, oe::kBGRA, s, &st);
        flowMs += st.flowMs;
        blurMs += st.blurMs;
        std::fwrite(out.data, 1, frameBytes, fo);
    }
    std::fclose(fo);
    std::printf("wrote %d frames; avg flow %.1f ms, blur %.1f ms per frame\n", k, flowMs / k, blurMs / k);
    return 0;
}
