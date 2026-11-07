// Offline harness for the GPU path: raw BGRA8 frames in, raw BGRA8 frames out.
//   oe_gpu_offline in.bgra W H srcFps outFps out.bgra [flowWidth grid perf mipThreshold]
// Frames are uploaded as 32-bit float BGRA (Premiere's GPU format) and processed
// exactly as the plugin will: frame round(k*srcFps/outFps) plus its neighbours.
#include "../MotionBlur.h" // BlurHalfWidth
#include "../gpu/GpuMotionBlur.h"

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
        if (!eng->Process(prev, src[i], next, out, s, nullptr, &st, &err))
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
    if (timed)
        std::printf("wrote %d frames; GPU avg flow %.2f ms, blur %.2f ms, total %.2f ms per frame\n", k,
                    flowMs / timed, blurMs / timed, totMs / timed);
    for (auto& f : src)
        eng->Free(&f);
    eng->Free(&out);
    return 0;
}
