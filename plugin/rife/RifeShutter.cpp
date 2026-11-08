#include "RifeShutter.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace oe::rife
{

bool RenderShutter(Engine& rife, gpu::Engine& gpu, const ShutterFrames& f, const ChannelOrder& order, bool argb8,
                   double t, double halfWidth, int nSamples, int maxPixels, bool showVectors, void* out,
                   size_t outRowBytes, ShutterStats* stats, std::string* err)
{
    using Clock = std::chrono::steady_clock;
    if (!f.A || !f.B)
    {
        if (err)
            *err = "Extra High: frames A and B are required";
        return false;
    }
    const int n = halfWidth > 1e-4 ? std::max(1, nSamples) : 1;
    // segment s covers times [s, s+1): -1 = (P, A), 0 = (A, B), 1 = (B, N)
    const cv::Mat* src[4] = {f.P, f.A, f.B, f.N};
    std::vector<float> ts[3];
    for (int i = 0; i < n; ++i)
    {
        double ti = n == 1 ? t : t - halfWidth + (i + 0.5) / n * 2.0 * halfWidth;
        if (ti < 0.0 && !f.P)
            ti = 0.0; // window reaches before A but P isn't available: hold A
        if (ti > 1.0 && !f.N)
            ti = 1.0; // ... after B without N: hold B
        int seg = (int)std::floor(ti);
        seg = std::clamp(seg, -1, 1);
        if (seg == 1 && ti == 1.0)
            seg = 0; // exactly B: use (A, B) at 1 rather than needing N
        if (seg == -1 && !f.P)
            seg = 0;
        if (seg == 1 && !f.N)
            seg = 0;
        ts[seg + 1].push_back((float)(ti - seg));
    }

    // frames used, compacted for the GPU call
    std::vector<gpu::HostImage> frames;
    int index[4] = {-1, -1, -1, -1};
    auto use = [&](int k) {
        if (index[k] < 0)
        {
            const cv::Mat& m = *src[k];
            index[k] = (int)frames.size();
            frames.push_back({m.data, m.step, m.cols, m.rows, argb8 && m.depth() != CV_32F, m.depth() == CV_32F});
        }
        return index[k];
    };

    std::vector<Field> fields[3];
    std::vector<gpu::TimeSample> samples;
    ShutterStats st;
    const auto t0 = Clock::now();
    for (int seg = -1; seg <= 1; ++seg)
    {
        std::vector<float>& tl = ts[seg + 1];
        if (tl.empty())
            continue;
        const int k0 = seg + 1, k1 = seg + 2; // indices into src
        if (!rife.Estimate(*src[k0], *src[k1], order, maxPixels, tl, &fields[seg + 1], err))
            return false;
        ++st.pairs;
        const int i0 = use(k0), i1 = use(k1);
        for (const Field& fl : fields[seg + 1])
        {
            gpu::TimeSample s;
            s.frame0 = i0;
            s.frame1 = i1;
            s.weight = 1.0f / n;
            s.field = {fl.flow.data, fl.flow.step, fl.mask.data, fl.mask.step, fl.flow.cols, fl.flow.rows};
            samples.push_back(s);
        }
    }
    st.rifeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    gpu::Stats gs;
    const bool ok = gpu.ProcessSamples(frames.data(), (int)frames.size(), samples.data(), (int)samples.size(), out,
                                       outRowBytes, showVectors, &gs, err);
    st.samples = (int)samples.size();
    st.gpuMs = gs.gpuMs;
    if (stats)
        *stats = st;
    return ok;
}

} // namespace oe::rife
