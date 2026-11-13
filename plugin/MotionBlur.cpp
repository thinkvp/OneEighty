#include "MotionBlur.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

namespace oe
{
namespace
{

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// Frame downscaled (area) to the motion-analysis size.
cv::Mat Downscale(const cv::Mat& bgra, double scale)
{
    if (scale >= 1.0)
        return bgra;
    cv::Mat small;
    cv::resize(bgra, small, cv::Size(), scale, scale, cv::INTER_AREA);
    return small;
}

// 8-bit luma of a (downscaled) frame, for motion estimation (float input is
// clamped to 0..1, which is fine for estimating motion).
cv::Mat GrayOf(const cv::Mat& small, const ChannelOrder& o)
{
    cv::Mat gray(small.size(), CV_8UC1);
    if (small.depth() == CV_32F)
    {
        for (int y = 0; y < small.rows; ++y)
        {
            const float* s = small.ptr<float>(y);
            uchar* d = gray.ptr<uchar>(y);
            for (int x = 0; x < small.cols; ++x, s += 4)
            {
                // same weights and rounding as the 8-bit branch (77/150/29 of 256) so an
                // 8-bit preview and a float export of the same frame get the same motion
                const int r = (int)(std::clamp(s[o.r], 0.0f, 1.0f) * 255.0f + 0.5f);
                const int g = (int)(std::clamp(s[o.g], 0.0f, 1.0f) * 255.0f + 0.5f);
                const int b = (int)(std::clamp(s[o.b], 0.0f, 1.0f) * 255.0f + 0.5f);
                d[x] = (uchar)((77 * r + 150 * g + 29 * b + 128) >> 8);
            }
        }
        return gray;
    }
    for (int y = 0; y < small.rows; ++y)
    {
        const uchar* s = small.ptr<uchar>(y);
        uchar* d = gray.ptr<uchar>(y);
        for (int x = 0; x < small.cols; ++x, s += 4)
            d[x] = (uchar)((77 * s[o.r] + 150 * s[o.g] + 29 * s[o.b] + 128) >> 8);
    }
    return gray;
}

cv::Mat SmallGray(const cv::Mat& bgra, double scale, const ChannelOrder& o) { return GrayOf(Downscale(bgra, scale), o); }

// 8-bit colour copy of a (downscaled) frame (B,G,R; 3x3 box-smoothed against
// noise) for the motion-boundary clean-up below.
cv::Mat ColorOf(const cv::Mat& small, const ChannelOrder& o)
{
    cv::Mat s8 = small;
    if (s8.depth() == CV_32F)
        small.convertTo(s8, CV_8UC4, 255.0); // saturates: fine for comparing colours
    cv::Mat col(s8.size(), CV_8UC3);
    const int pairs[] = {o.b, 0, o.g, 1, o.r, 2};
    cv::mixChannels(&s8, 1, &col, 1, pairs, 3);
    cv::blur(col, col, cv::Size(3, 3));
    return col;
}

// Motion-boundary clean-up. DIS motion is smooth, so a fast object's motion
// spills 20-40 px past its outline onto the background around it, and the
// blur then smears that background (e.g. a court line next to a bouncing
// ball). Near a motion boundary each pixel takes the motion of nearby pixels
// that look like it: candidates are sampled in 8 directions out to beyond the
// spill, weighted by colour similarity to the pixel, and the weighted median
// motion wins. Court and line pixels next to the ball side with the court
// around them, ball pixels with the ball. col: the flow's source frame
// (ColorOf), same size as flow.
void RefineFlowEdges(const cv::Mat& col, cv::Mat& flow)
{
    CV_Assert(col.type() == CV_8UC3 && flow.type() == CV_32FC2 && flow.size() == col.size());
    const int W = col.cols, H = col.rows;
    const int radii[] = {3, 8, 16, 28, 42};    // candidate distances (flow-res pixels)
    const int dirs[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    constexpr int kReach = 42;
    constexpr int kN = 5 * 8 + 1;
    float colourWeight[766];                    // by L1 colour distance (0..765)
    for (int d = 0; d < 766; ++d)
        colourWeight[d] = std::exp(-(d / 36.0f) * (d / 36.0f));
    const cv::Mat src = flow.clone();

    // Only near motion boundaries: where the motion changes by more than
    // ~1.5 px per pixel, grown by the candidate reach. Worked out on
    // every 4th pixel: it only needs to be roughly right.
    cv::Mat q, ch[2], g0, g1, mask;
    cv::resize(src, q, cv::Size((W + 3) / 4, (H + 3) / 4), 0, 0, cv::INTER_NEAREST);
    cv::split(q, ch);
    cv::morphologyEx(ch[0], g0, cv::MORPH_GRADIENT, cv::Mat());
    cv::morphologyEx(ch[1], g1, cv::MORPH_GRADIENT, cv::Mat());
    mask = (g0 + g1) > 3.0f; // neighbours here are 4 px apart
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * (kReach / 4) + 3, 2 * (kReach / 4) + 3)));
    cv::resize(mask, mask, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);

    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& rows) {
        float vx[kN], vy[kN], w[kN];
        int idx[kN];
        auto weightedMedian = [&](const float* v, int n) {
            for (int i = 0; i < n; ++i)
                idx[i] = i;
            std::sort(idx, idx + n, [&](int p, int q) { return v[p] < v[q]; });
            float total = 0;
            for (int i = 0; i < n; ++i)
                total += w[i];
            float acc = 0;
            for (int i = 0; i < n; ++i)
            {
                acc += w[idx[i]];
                if (acc >= 0.5f * total)
                    return v[idx[i]];
            }
            return v[idx[n - 1]];
        };
        for (int y = rows.start; y < rows.end; ++y)
        {
            const uchar* mrow = mask.ptr<uchar>(y);
            const cv::Vec3b* crow = col.ptr<cv::Vec3b>(y);
            cv::Vec2f* drow = flow.ptr<cv::Vec2f>(y);
            for (int x = 0; x < W; ++x)
            {
                if (!mrow[x])
                    continue;
                const cv::Vec2f own = src.ptr<cv::Vec2f>(y)[x];
                const cv::Vec3b c0 = crow[x];
                int n = 0;
                float spread = 0;
                vx[n] = own[0];
                vy[n] = own[1];
                w[n++] = 1.0f;
                for (int r : radii)
                    for (const auto& d : dirs)
                    {
                        const int cx = x + d[0] * r, cy = y + d[1] * r;
                        if (cx < 0 || cy < 0 || cx >= W || cy >= H)
                            continue;
                        const cv::Vec2f v = src.ptr<cv::Vec2f>(cy)[cx];
                        const cv::Vec3b c = col.ptr<cv::Vec3b>(cy)[cx];
                        spread = std::max(spread, std::abs(v[0] - own[0]) + std::abs(v[1] - own[1]));
                        vx[n] = v[0];
                        vy[n] = v[1];
                        w[n++] = colourWeight[std::abs(c[0] - c0[0]) + std::abs(c[1] - c0[1]) + std::abs(c[2] - c0[2])];
                    }
                if (spread < 1.0f)
                    continue; // every candidate agrees: nothing to fix
                // weighted median per component (robust: never averages two motions)
                drow[x] = cv::Vec2f(weightedMedian(vx, n), weightedMedian(vy, n));
            }
        }
    });
}

// Pixel component traits: 8-bit (0..255) or 32-bit float (0..1, may exceed).
template <typename T> struct Px;
template <> struct Px<uchar>
{
    static constexpr float kOne = 255.0f;
    static uchar From(float v) { return (uchar)std::clamp(v + 0.5f, 0.0f, 255.0f); }
};
template <> struct Px<float>
{
    static constexpr float kOne = 1.0f;
    static float From(float v) { return v; } // no clamping: keeps HDR / super-whites
};

// Motion field view: hue = direction, colour strength = speed (full at 1% of
// the frame width per source frame), over a grey version of the picture.
template <typename T>
void DrawVectors(const cv::Mat& cur, const cv::Mat& vFull, const ChannelOrder& o, cv::Mat& out)
{
    const float norm = std::max(4.0f, 0.01f * cur.cols), one = Px<T>::kOne;
    cv::parallel_for_(cv::Range(0, cur.rows), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y)
        {
            const cv::Vec2f* vr = vFull.ptr<cv::Vec2f>(y);
            const T* p = cur.ptr<T>(y);
            T* d = out.ptr<T>(y);
            for (int x = 0; x < cur.cols; ++x, p += 4, d += 4)
            {
                float rgb[3];
                VectorColour(vr[x][0], vr[x][1], norm, rgb);
                const float g = std::clamp((0.299f * p[o.r] + 0.587f * p[o.g] + 0.114f * p[o.b]) / one, 0.0f, 1.0f);
                const float k = 0.65f * (0.4f + 0.6f * g);
                d[o.r] = Px<T>::From((0.35f * g + k * rgb[0]) * one);
                d[o.g] = Px<T>::From((0.35f * g + k * rgb[1]) * one);
                d[o.b] = Px<T>::From((0.35f * g + k * rgb[2]) * one);
                d[o.a] = p[o.a];
            }
        }
    });
}

// Average of cur along each pixel's motion path (see FlowMotionBlur).
template <typename T>
void GatherBlur(const cv::Mat& cur, const cv::Mat& vFull, const BlurSettings& s, cv::Mat& out, long long* samplesOut,
                int* maxSpeedX100Out)
{
    const int W = cur.cols, H = cur.rows;
    // mip chain for long smears (level l = 1/2^l resolution, area filtered)
    std::vector<cv::Mat> mip{cur};
    if (s.mipThreshold > 0)
        for (int l = 1; l <= s.maxMipLevel && mip.back().cols >= 4 && mip.back().rows >= 4; ++l)
        {
            cv::Mat half;
            cv::resize(mip.back(), half, cv::Size((mip.back().cols + 1) / 2, (mip.back().rows + 1) / 2), 0, 0,
                       cv::INTER_AREA);
            mip.push_back(half);
        }
    const int topLevel = (int)mip.size() - 1;
    const float hw = (float)s.halfWidth;
    const float c0 = (float)s.centerOffset;
    const int maxS = std::max(1, s.maxSamples);
    const float mipT = (float)std::max(1, s.mipThreshold);
    std::atomic<long long> totalSamples{0};
    std::atomic<int> maxSpeedX100{0};

    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& rows) {
        long long localSamples = 0;
        int localMax = 0;
        for (int y = rows.start; y < rows.end; ++y)
        {
            const cv::Vec2f* vr = vFull.ptr<cv::Vec2f>(y);
            T* o = out.ptr<T>(y);
            for (int x = 0; x < W; ++x, o += 4)
            {
                const float vx = vr[x][0], vy = vr[x][1];
                const float speed = std::sqrt(vx * vx + vy * vy);
                localMax = std::max(localMax, (int)(speed * 100));
                const float smear = 2.0f * hw * speed; // blur length in full-res pixels
                if (smear <= 1.0f && std::fabs(c0) * speed < 0.5f)
                {
                    const T* p = cur.ptr<T>(y) + 4 * x;
                    o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; o[3] = p[3];
                    ++localSamples;
                    continue;
                }
                int lvl = 0;
                while (lvl < topLevel && smear / (float)(1 << lvl) > mipT)
                    ++lvl;
                const cv::Mat& img = mip[lvl];
                const float inv2l = 1.0f / (float)(1 << lvl);
                const int w = img.cols, h = img.rows;
                const int n = std::min(maxS, std::max(2, (int)std::ceil(smear * inv2l)));
                localSamples += n;

                float acc[4] = {0, 0, 0, 0};
                for (int j = 0; j < n; ++j)
                {
                    const float off = c0 + hw * (((j + 0.5f) / n) * 2.0f - 1.0f);
                    // full-res sample position -> this level's pixel grid
                    float sx = (x - off * vx + 0.5f) * inv2l - 0.5f;
                    float sy = (y - off * vy + 0.5f) * inv2l - 0.5f;
                    sx = std::clamp(sx, 0.0f, (float)(w - 1));
                    sy = std::clamp(sy, 0.0f, (float)(h - 1));
                    const int x0 = std::min((int)sx, std::max(0, w - 2)), y0 = std::min((int)sy, std::max(0, h - 2));
                    const float fx = sx - x0, fy = sy - y0;
                    const T* p00 = img.ptr<T>(y0) + 4 * x0;
                    const T* p10 = p00 + 4;
                    const T* p01 = img.ptr<T>(y0 + 1) + 4 * x0;
                    const T* p11 = p01 + 4;
                    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                    for (int c = 0; c < 4; ++c)
                        acc[c] += w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
                }
                const float inv = 1.0f / n;
                for (int c = 0; c < 4; ++c)
                    o[c] = Px<T>::From(acc[c] * inv);
            }
        }
        totalSamples += localSamples;
        int prevMax = maxSpeedX100.load();
        while (localMax > prevMax && !maxSpeedX100.compare_exchange_weak(prevMax, localMax)) {}
    });
    *samplesOut = totalSamples;
    *maxSpeedX100Out = maxSpeedX100;
}

// Layered blur (see Layered in gpu/GpuMotionBlur.cu, same constants): the
// pixel's own blur along its own motion, plus for the share of the shutter
// during which a clearly faster layer passing over it covers it, that
// layer's colour. Faster layers are found through the fastest motion per 16 px
// tile, then over +/-2 tiles, interpolated between tile centres.
template <typename T>
void LayeredBlur(const cv::Mat& cur, const cv::Mat& vFull, const BlurSettings& s, cv::Mat& out, long long* samplesOut,
                 int* maxSpeedX100Out)
{
    constexpr int kTile = 16, kReach = 2;
    constexpr float kSz = 1.5f;
    const int W = cur.cols, H = cur.rows, tw = (W + kTile - 1) / kTile, th = (H + kTile - 1) / kTile;
    std::vector<cv::Mat> mip{cur};
    if (s.mipThreshold > 0)
        for (int l = 1; l <= s.maxMipLevel && mip.back().cols >= 4 && mip.back().rows >= 4; ++l)
        {
            cv::Mat half;
            cv::resize(mip.back(), half, cv::Size((mip.back().cols + 1) / 2, (mip.back().rows + 1) / 2), 0, 0,
                       cv::INTER_AREA);
            mip.push_back(half);
        }
    const int topLevel = (int)mip.size() - 1;
    // tile max, then over the neighbourhood
    cv::Mat tiles(th, tw, CV_32FC2), nb(th, tw, CV_32FC2);
    cv::parallel_for_(cv::Range(0, th), [&](const cv::Range& r) {
        for (int ty = r.start; ty < r.end; ++ty)
            for (int tx = 0; tx < tw; ++tx)
            {
                cv::Vec2f best(0, 0);
                float bm = -1;
                for (int y = ty * kTile; y < std::min(H, (ty + 1) * kTile); ++y)
                {
                    const cv::Vec2f* vr = vFull.ptr<cv::Vec2f>(y);
                    for (int x = tx * kTile; x < std::min(W, (tx + 1) * kTile); ++x)
                    {
                        const float m = vr[x][0] * vr[x][0] + vr[x][1] * vr[x][1];
                        if (m > bm)
                        {
                            bm = m;
                            best = vr[x];
                        }
                    }
                }
                tiles.at<cv::Vec2f>(ty, tx) = best;
            }
    });
    for (int ty = 0; ty < th; ++ty)
        for (int tx = 0; tx < tw; ++tx)
        {
            cv::Vec2f best = tiles.at<cv::Vec2f>(ty, tx);
            float bm = best.dot(best);
            for (int dy = -kReach; dy <= kReach; ++dy)
                for (int dx = -kReach; dx <= kReach; ++dx)
                {
                    const cv::Vec2f v = tiles.at<cv::Vec2f>(std::clamp(ty + dy, 0, th - 1), std::clamp(tx + dx, 0, tw - 1));
                    if (v.dot(v) > bm)
                    {
                        bm = v.dot(v);
                        best = v;
                    }
                }
            nb.at<cv::Vec2f>(ty, tx) = best;
        }
    auto sampleTiles = [&](int x, int y) {
        const float gx = std::clamp((x + 0.5f) / kTile - 0.5f, 0.0f, (float)(tw - 1));
        const float gy = std::clamp((y + 0.5f) / kTile - 0.5f, 0.0f, (float)(th - 1));
        const int x0 = std::min((int)gx, std::max(0, tw - 2)), y0 = std::min((int)gy, std::max(0, th - 2));
        const int x1 = std::min(x0 + 1, tw - 1), y1 = std::min(y0 + 1, th - 1);
        const float fx = gx - x0, fy = gy - y0;
        const cv::Vec2f a = nb.at<cv::Vec2f>(y0, x0), b = nb.at<cv::Vec2f>(y0, x1), c = nb.at<cv::Vec2f>(y1, x0),
                        d = nb.at<cv::Vec2f>(y1, x1);
        return cv::Vec2f((1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d));
    };
    auto bilinear = [](const cv::Mat& img, float sx, float sy, float acc[4], float w) {
        const int iw = img.cols, ih = img.rows;
        sx = std::clamp(sx, 0.0f, (float)(iw - 1));
        sy = std::clamp(sy, 0.0f, (float)(ih - 1));
        const int x0 = std::min((int)sx, std::max(0, iw - 2)), y0 = std::min((int)sy, std::max(0, ih - 2));
        const int x1 = std::min(x0 + 1, iw - 1), y1 = std::min(y0 + 1, ih - 1);
        const float fx = sx - x0, fy = sy - y0;
        const T* p00 = img.ptr<T>(y0) + 4 * x0;
        const T* p10 = img.ptr<T>(y0) + 4 * x1;
        const T* p01 = img.ptr<T>(y1) + 4 * x0;
        const T* p11 = img.ptr<T>(y1) + 4 * x1;
        const float w00 = (1 - fx) * (1 - fy) * w, w10 = fx * (1 - fy) * w, w01 = (1 - fx) * fy * w, w11 = fx * fy * w;
        for (int c = 0; c < 4; ++c)
            acc[c] += w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
    };
    auto velAt = [&](float qx, float qy) {
        const int ix = std::clamp((int)std::lround(qx), 0, W - 1), iy = std::clamp((int)std::lround(qy), 0, H - 1);
        return vFull.ptr<cv::Vec2f>(iy)[ix];
    };
    const float hw = (float)s.halfWidth, c0 = (float)s.centerOffset;
    const float t0 = c0 - hw, t1 = c0 + hw;
    const int maxS = std::max(2, s.maxSamples);
    const float mipT = (float)std::max(1, s.mipThreshold);
    auto level = [&](float smear) {
        int l = 0;
        while (l < topLevel && smear / (float)(1 << l) > mipT)
            ++l;
        return l;
    };
    std::atomic<long long> totalSamples{0};
    std::atomic<int> maxSpeedX100{0};
    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& rows) {
        long long localSamples = 0;
        int localMax = 0;
        for (int y = rows.start; y < rows.end; ++y)
        {
            const cv::Vec2f* vr = vFull.ptr<cv::Vec2f>(y);
            T* o = out.ptr<T>(y);
            for (int x = 0; x < W; ++x, o += 4)
            {
                const cv::Vec2f v = vr[x], vn = sampleTiles(x, y);
                const float sp = std::sqrt(v.dot(v)), sn = std::sqrt(vn.dot(vn));
                localMax = std::max(localMax, (int)(sp * 100));
                const T* pc = cur.ptr<T>(y) + 4 * x;
                if (2.0f * hw * std::max(sp, sn) <= 1.0f && std::fabs(c0) * std::max(sp, sn) < 0.5f)
                {
                    o[0] = pc[0];
                    o[1] = pc[1];
                    o[2] = pc[2];
                    o[3] = pc[3];
                    ++localSamples;
                    continue;
                }
                const float jit = std::fmod(52.9829189f * std::fmod(0.06711056f * x + 0.00583715f * y, 1.0f), 1.0f);
                float bacc[4] = {0, 0, 0, 0}, facc[4] = {0, 0, 0, 0}, bw = 0, fw = 0;
                {
                    // the pixel's own blur, minus samples of a faster layer
                    const float smear = 2.0f * hw * sp;
                    const int l = level(smear);
                    const float inv2l = 1.0f / (float)(1 << l);
                    const int n = std::min(maxS, std::max(4, (int)std::ceil(smear * inv2l)));
                    localSamples += n;
                    for (int j = 0; j < n; ++j)
                    {
                        const float t = t0 + (t1 - t0) * (j + jit) / n;
                        const float qx = x - t * v[0], qy = y - t * v[1];
                        const cv::Vec2f vq = velAt(qx, qy);
                        const float front = std::clamp(0.5f + (std::sqrt(vq.dot(vq)) - sp - kSz) / (2 * kSz), 0.0f, 1.0f);
                        const float wb = 1.0f - front;
                        bilinear(mip[l], (qx + 0.5f) * inv2l - 0.5f, (qy + 0.5f) * inv2l - 0.5f, bacc, wb);
                        bw += wb;
                    }
                }
                if (sn > sp)
                {
                    // faster layers passing over the pixel
                    const float smear = 2.0f * hw * sn;
                    const int l = level(smear);
                    const float inv2l = 1.0f / (float)(1 << l);
                    const int n = std::min(maxS, std::max(4, (int)std::ceil(smear * inv2l)));
                    const float share = std::min(smear, 1.0f) / n;
                    localSamples += n;
                    for (int j = 0; j < n; ++j)
                    {
                        const float t = t0 + (t1 - t0) * (j + jit) / n;
                        const float qx = x - t * vn[0], qy = y - t * vn[1];
                        const cv::Vec2f vq = velAt(qx, qy);
                        const float sq = std::sqrt(vq.dot(vq));
                        const float front = std::clamp(0.5f + (sq - sp - kSz) / (2 * kSz), 0.0f, 1.0f);
                        if (front <= 0.0f)
                            continue;
                        const float tq = t * sn / std::max(sq, 1e-3f);
                        const float cover = std::clamp(std::min(tq - t0, t1 - tq) * sq + 0.5f, 0.0f, 1.0f);
                        const float wf = front * cover * share;
                        bilinear(mip[l], (qx + 0.5f) * inv2l - 0.5f, (qy + 0.5f) * inv2l - 0.5f, facc, wf);
                        fw += wf;
                    }
                }
                float bcol[4];
                for (int c = 0; c < 4; ++c)
                    bcol[c] = bw > 1e-3f ? bacc[c] / bw : (float)pc[c];
                const float alpha = std::min(fw, 1.0f), k = fw > 1e-6f ? alpha / fw : 0.0f;
                for (int c = 0; c < 4; ++c)
                    o[c] = Px<T>::From(facc[c] * k + (1 - alpha) * bcol[c]);
            }
        }
        totalSamples += localSamples;
        int prevMax = maxSpeedX100.load();
        while (localMax > prevMax && !maxSpeedX100.compare_exchange_weak(prevMax, localMax)) {}
    });
    *samplesOut = totalSamples;
    *maxSpeedX100Out = maxSpeedX100;
}

// |f(x) + g(x + f(x))| (g bilinear, edges clamped), 5x5 box mean: how badly
// the motion f fails its forward/backward check.
cv::Mat ConsistencyError(const cv::Mat& f, const cv::Mat& g)
{
    cv::Mat map(f.size(), CV_32FC2), back, e(f.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
    {
        const cv::Vec2f* fr = f.ptr<cv::Vec2f>(y);
        cv::Vec2f* mr = map.ptr<cv::Vec2f>(y);
        for (int x = 0; x < f.cols; ++x)
            mr[x] = cv::Vec2f(x + fr[x][0], y + fr[x][1]);
    }
    cv::remap(g, back, map, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    for (int y = 0; y < f.rows; ++y)
    {
        const cv::Vec2f* fr = f.ptr<cv::Vec2f>(y);
        const cv::Vec2f* br = back.ptr<cv::Vec2f>(y);
        float* er = e.ptr<float>(y);
        for (int x = 0; x < f.cols; ++x)
            er[x] = std::abs(fr[x][0] + br[x][0]) + std::abs(fr[x][1] + br[x][1]);
    }
    cv::blur(e, e, cv::Size(5, 5));
    return e;
}

// cur's velocity (analysis size and pixels) from both sides: cur->next and
// cur->prev, each weighted by its forward/backward check (a region about to
// be covered was still visible a frame earlier).
cv::Mat TwoSidedFlow(cv::DISOpticalFlow& dis, const cv::Mat& gCur, const cv::Mat& gPrev, const cv::Mat& gNext)
{
    cv::Mat fn, fnb, fp, fpb;
    dis.calc(gCur, gNext, fn);
    dis.calc(gNext, gCur, fnb);
    dis.calc(gCur, gPrev, fp);
    dis.calc(gPrev, gCur, fpb);
    const cv::Mat en = ConsistencyError(fn, fnb), ep = ConsistencyError(fp, fpb);
    cv::Mat v(fn.size(), CV_32FC2);
    for (int y = 0; y < v.rows; ++y)
    {
        const cv::Vec2f* a = fn.ptr<cv::Vec2f>(y);
        const cv::Vec2f* b = fp.ptr<cv::Vec2f>(y);
        const float* ea = en.ptr<float>(y);
        const float* eb = ep.ptr<float>(y);
        cv::Vec2f* d = v.ptr<cv::Vec2f>(y);
        for (int x = 0; x < v.cols; ++x)
        {
            const float wn = 1.0f / (0.25f + ea[x] * ea[x]), wp = 1.0f / (0.25f + eb[x] * eb[x]);
            d[x] = (wn * a[x] - wp * b[x]) * (1.0f / (wn + wp));
        }
    }
    return v;
}

} // namespace

cv::Mat EstimateFlow(const cv::Mat& from, const cv::Mat& to, const ChannelOrder& order, int flowWidth, int disPreset,
                     bool refineEdges)
{
    const double scale = std::min(1.0, (double)flowWidth / from.cols);
    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(disPreset);
    cv::Mat f;
    const cv::Mat s0 = Downscale(from, scale);
    dis->calc(GrayOf(s0, order), SmallGray(to, scale, order), f);
    if (refineEdges)
        RefineFlowEdges(ColorOf(s0, order), f);
    if (scale < 1.0)
        f *= 1.0 / scale;
    return f;
}

void EstimateFlowPair(const cv::Mat& a, const cv::Mat& b, const ChannelOrder& order, int flowWidth, cv::Mat& fab,
                      cv::Mat& fba, int disPreset, bool refineEdges)
{
    // Sequential on purpose: DIS already uses all cores internally (running the
    // two directions concurrently measured no faster at UHD: 64.2 vs 64.3 ms).
    const double scale = std::min(1.0, (double)flowWidth / a.cols);
    const cv::Mat sa = Downscale(a, scale), sb = Downscale(b, scale);
    const cv::Mat ga = GrayOf(sa, order), gb = GrayOf(sb, order);
    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(disPreset);
    dis->calc(ga, gb, fab);
    dis->calc(gb, ga, fba);
    if (refineEdges)
    {
        RefineFlowEdges(ColorOf(sa, order), fab);
        RefineFlowEdges(ColorOf(sb, order), fba);
    }
    if (scale < 1.0)
    {
        fab *= 1.0 / scale;
        fba *= 1.0 / scale;
    }
}

void VectorColour(float vx, float vy, float norm, float rgb[3])
{
    // HSV with H = direction, S = speed / norm, V = 1
    const float sat = std::min(1.0f, std::sqrt(vx * vx + vy * vy) / norm);
    float h = std::atan2(vy, vx) / 6.2831853f; // -0.5..0.5
    if (h < 0)
        h += 1.0f;
    const float h6 = h * 6.0f;
    const int i = (int)h6 % 6;
    const float f = h6 - std::floor(h6);
    const float p = 1 - sat, q = 1 - sat * f, t = 1 - sat * (1 - f);
    const float table[6][3] = {{1, t, p}, {q, 1, p}, {p, 1, t}, {p, q, 1}, {t, p, 1}, {1, p, q}};
    rgb[0] = table[i][0];
    rgb[1] = table[i][1];
    rgb[2] = table[i][2];
}

double BlurHalfWidth(double srcFps, double outFps, double srcShutterDeg, double targetShutterDeg)
{
    if (srcFps <= 0 || outFps <= 0)
        return 0.0;
    const double T = targetShutterDeg / 360.0 * srcFps / outFps; // wanted exposure, source intervals
    const double e = srcShutterDeg / 360.0;                      // exposure already in the frame
    if (T <= e)
        return 0.0;
    return std::sqrt(T * T - e * e) / 2.0;
}

void FlowMotionBlur(const cv::Mat& prev, const cv::Mat& cur, const cv::Mat& next, cv::Mat& out,
                    const ChannelOrder& order, const BlurSettings& s, BlurStats* stats)
{
    CV_Assert((cur.type() == CV_8UC4 || cur.type() == CV_32FC4) && out.type() == cur.type() &&
              out.size() == cur.size());
    const bool isFloat = cur.depth() == CV_32F;
    const auto t0 = Clock::now();
    const int W = cur.cols, H = cur.rows;

    if ((s.halfWidth <= 0 && s.centerOffset == 0.0 && !s.showVectors) || (prev.empty() && next.empty()))
    {
        cur.copyTo(out);
        return;
    }

    // ---- motion estimation (reduced resolution)
    const double scale = std::min(1.0, (double)s.flowWidth / W);
    const cv::Mat sCur = Downscale(cur, scale);
    const cv::Mat gCur = GrayOf(sCur, order);
    cv::Mat cCur; // colour copy for the boundary clean-up, made once
    if (s.refineEdges)
        cCur = ColorOf(sCur, order);
    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(s.disPreset);

    cv::Mat v; // velocity per source frame, small res, CV_32FC2
    int nFlows = 0;
    if (s.twoSided && !prev.empty() && !next.empty())
    {
        v = TwoSidedFlow(*dis, gCur, SmallGray(prev, scale, order), SmallGray(next, scale, order));
        if (s.refineEdges)
            RefineFlowEdges(cCur, v);
    }
    else if (!next.empty())
    {
        cv::Mat f;
        const cv::Mat gNext = SmallGray(next, scale, order);
        dis->calc(gCur, gNext, f);
        if (s.refineEdges)
            RefineFlowEdges(cCur, f);
        v = f;
        ++nFlows;
    }
    if (!prev.empty() && !(s.twoSided && !next.empty()))
    {
        cv::Mat f;
        const cv::Mat gPrev = SmallGray(prev, scale, order);
        dis->calc(gCur, gPrev, f);
        if (s.refineEdges)
            RefineFlowEdges(cCur, f);
        if (v.empty())
            v = -f;
        else
            v -= f;
        ++nFlows;
    }
    if (nFlows == 2)
        v *= 0.5;

    // to full resolution, in full-res pixels
    cv::Mat vFull;
    if (scale < 1.0)
    {
        cv::resize(v, vFull, cur.size(), 0, 0, cv::INTER_LINEAR);
        vFull *= 1.0 / scale;
    }
    else
        vFull = v;
    const auto t1 = Clock::now();

    if (s.showVectors)
    {
        if (isFloat)
            DrawVectors<float>(cur, vFull, order, out);
        else
            DrawVectors<uchar>(cur, vFull, order, out);
        if (stats)
            stats->flowMs = Ms(t0, t1);
        return;
    }

    // ---- gather blur: average cur along each pixel's motion path
    long long totalSamples = 0;
    int maxSpeedX100 = 0;
    if (s.layered && isFloat)
        LayeredBlur<float>(cur, vFull, s, out, &totalSamples, &maxSpeedX100);
    else if (s.layered)
        LayeredBlur<uchar>(cur, vFull, s, out, &totalSamples, &maxSpeedX100);
    else if (isFloat)
        GatherBlur<float>(cur, vFull, s, out, &totalSamples, &maxSpeedX100);
    else
        GatherBlur<uchar>(cur, vFull, s, out, &totalSamples, &maxSpeedX100);
    const auto t2 = Clock::now();

    if (stats)
    {
        stats->flowMs = Ms(t0, t1);
        stats->blurMs = Ms(t1, t2);
        stats->avgSamples = (double)totalSamples / ((double)W * H);
        stats->maxSpeedPx = maxSpeedX100 / 100.0;
    }
}

} // namespace oe
