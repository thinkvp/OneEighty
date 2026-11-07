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

// Downscale (area) then convert to 8-bit luma.
cv::Mat SmallGray(const cv::Mat& bgra, double scale, const ChannelOrder& o)
{
    cv::Mat small;
    if (scale < 1.0)
        cv::resize(bgra, small, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        small = bgra;
    cv::Mat gray(small.size(), CV_8UC1);
    for (int y = 0; y < small.rows; ++y)
    {
        const uchar* s = small.ptr<uchar>(y);
        uchar* d = gray.ptr<uchar>(y);
        for (int x = 0; x < small.cols; ++x, s += 4)
            d[x] = (uchar)((77 * s[o.r] + 150 * s[o.g] + 29 * s[o.b] + 128) >> 8);
    }
    return gray;
}

} // namespace

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
    CV_Assert(cur.type() == CV_8UC4 && out.type() == CV_8UC4 && out.size() == cur.size());
    const auto t0 = Clock::now();
    const int W = cur.cols, H = cur.rows;

    if (s.halfWidth <= 0 || (prev.empty() && next.empty()))
    {
        cur.copyTo(out);
        return;
    }

    // ---- motion estimation (reduced resolution)
    const double scale = std::min(1.0, (double)s.flowWidth / W);
    cv::Mat gCur = SmallGray(cur, scale, order);
    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

    cv::Mat v; // velocity per source frame, small res, CV_32FC2
    int nFlows = 0;
    if (!next.empty())
    {
        cv::Mat f;
        dis->calc(gCur, SmallGray(next, scale, order), f);
        v = f;
        ++nFlows;
    }
    if (!prev.empty())
    {
        cv::Mat f;
        dis->calc(gCur, SmallGray(prev, scale, order), f);
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

    // ---- mip chain for long smears (level l = 1/2^l resolution, area filtered)
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

    // ---- gather blur: average cur along each pixel's motion path
    const float hw = (float)s.halfWidth;
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
            uchar* o = out.ptr<uchar>(y);
            for (int x = 0; x < W; ++x, o += 4)
            {
                const float vx = vr[x][0], vy = vr[x][1];
                const float speed = std::sqrt(vx * vx + vy * vy);
                localMax = std::max(localMax, (int)(speed * 100));
                const float smear = 2.0f * hw * speed; // blur length in full-res pixels
                if (smear <= 1.0f)
                {
                    const uchar* p = cur.ptr<uchar>(y) + 4 * x;
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
                    const float off = hw * (((j + 0.5f) / n) * 2.0f - 1.0f);
                    // full-res sample position -> this level's pixel grid
                    float sx = (x - off * vx + 0.5f) * inv2l - 0.5f;
                    float sy = (y - off * vy + 0.5f) * inv2l - 0.5f;
                    sx = std::clamp(sx, 0.0f, (float)(w - 1));
                    sy = std::clamp(sy, 0.0f, (float)(h - 1));
                    const int x0 = std::min((int)sx, std::max(0, w - 2)), y0 = std::min((int)sy, std::max(0, h - 2));
                    const float fx = sx - x0, fy = sy - y0;
                    const uchar* p00 = img.ptr<uchar>(y0) + 4 * x0;
                    const uchar* p10 = p00 + 4;
                    const uchar* p01 = img.ptr<uchar>(y0 + 1) + 4 * x0;
                    const uchar* p11 = p01 + 4;
                    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                    for (int c = 0; c < 4; ++c)
                        acc[c] += w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
                }
                const float inv = 1.0f / n;
                for (int c = 0; c < 4; ++c)
                    o[c] = (uchar)std::min(255.0f, acc[c] * inv + 0.5f);
            }
        }
        totalSamples += localSamples;
        int prevMax = maxSpeedX100.load();
        while (localMax > prevMax && !maxSpeedX100.compare_exchange_weak(prevMax, localMax)) {}
    });
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
