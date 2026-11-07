/*
    NOT BUILT (kept for reference, removed from the plugin in v0.5).

    Finding: Premiere's GPU filter API floors GetFrameDependencies requests to
    whole timeline frames, so on a 25p timeline a GPU filter can never see the
    in-between 50p/59.94p source frames this effect needs (logged in v0.4.1:
    requests at -0.23/+0.18 timeline frames both returned the current frame;
    -0.42 returned the previous timeline frame). The shipped effect fetches
    frames through the AE API instead and processes them on the GPU itself.
    To build it again it would need the parameter list it was written for
    (v0.4, OE_MODE/OE_RENDER) and the PRINC compile step in build.bat.

    OneEighty Motion Blur: Premiere Pro GPU filter.

    Paired with the CPU effect (same match name, same plugin file). Premiere
    asks for a GPU instance per parameter set; we accept when the project
    renders with CUDA and the GPU has NVIDIA optical-flow hardware, otherwise
    Premiere falls back to the CPU effect.

    GetFrameDependencies asks Premiere for the source frames one source-frame
    interval before and after the current time; Premiere decodes them and
    hands them to Render in GPU memory (BGRA float32/half), where the CUDA/NVOF
    engine (gpu/GpuMotionBlur.cu) does the work.

    Compiled against the Premiere Pro SDK headers (kept apart from
    OneEighty.cpp, which uses the After Effects SDK's versions).
*/

#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "A.h"
#include "AE_Macros.h"
#include "PrSDKAESupport.h"
#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#include "OneEightyParams.h"
#include "gpu/GpuMotionBlur.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <memory>
#include <mutex>
#include <string>

namespace
{

// One engine per GPU device, shared by all instances (an engine serialises
// its own calls). Created on first use; a failure is remembered so we don't
// retry (and re-log) for every instance.
struct DeviceEngines
{
    std::mutex mu;
    std::map<csSDK_uint32, std::shared_ptr<oe::gpu::Engine>> engines;
    std::map<csSDK_uint32, std::string> failures;
};
DeviceEngines& Engines()
{
    static DeviceEngines e;
    return e;
}

// Popup values reach the GPU filter 0-based (seen in Premiere 26: the default
// first item reads 0); the AE API and our enums are 1-based.
int PopupValue(const PrParam& p);

double ParamNumber(const PrParam& p)
{
    switch (p.mType)
    {
    case kPrParamType_Int8: return p.mInt8;
    case kPrParamType_Int16: return p.mInt16;
    case kPrParamType_Int32: return p.mInt32;
    case kPrParamType_Int64: return (double)p.mInt64;
    case kPrParamType_Float32: return p.mFloat32;
    case kPrParamType_Float64: return p.mFloat64;
    case kPrParamType_Bool: return p.mBool ? 1.0 : 0.0;
    default: return 0.0;
    }
}

int PopupValue(const PrParam& p) { return (int)ParamNumber(p) + 1; }

} // namespace

class OneEightyGPU : public PrGPUFilterBase
{
public:
    prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData) override
    {
        PrGPUFilterBase::Initialize(ioInstanceData);

        mLog = ParamNumber(GetParam(OE_LOG, 0)) != 0.0;

        // A new instance is created whenever parameters change, so this
        // decision follows the Render popup.
        if (PopupValue(GetParam(OE_RENDER, 0)) != OE_RENDER_PREMIERE_GPU)
            return suiteError_Fail; // Auto / CPU only render through the CPU effect (see OneEightyParams.h)

        if (mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_CUDA)
        {
            if (mLog)
                oe::Log("[gpu] device %u framework %d is not CUDA -> CPU fallback (set the project renderer to "
                        "Mercury Playback Engine GPU Acceleration (CUDA))",
                        mDeviceIndex, (int)mDeviceInfo.outDeviceFramework);
            return suiteError_Fail;
        }

        DeviceEngines& de = Engines();
        std::lock_guard<std::mutex> lock(de.mu);
        auto it = de.engines.find(mDeviceIndex);
        if (it != de.engines.end())
            mEngine = it->second;
        else if (!de.failures.count(mDeviceIndex))
        {
            std::string err;
            std::shared_ptr<oe::gpu::Engine> eng(oe::gpu::Engine::Create(mDeviceInfo.outContextHandle, &err).release());
            if (eng)
            {
                de.engines[mDeviceIndex] = eng;
                mEngine = eng;
                oe::Log("[gpu] device %u: CUDA + NVIDIA optical flow engine ready", mDeviceIndex);
            }
            else
            {
                de.failures[mDeviceIndex] = err;
                oe::Log("[gpu] device %u: engine unavailable (%s) -> CPU fallback", mDeviceIndex, err.c_str());
            }
        }
        if (!mEngine)
            return suiteError_Fail;

        DetectSource();
        return suiteError_NoError;
    }

    // Where this output frame sits in the source. With the clip's media in-point
    // known, `centre` is the exact fractional source frame of the output time;
    // we interpolate between frame A = floor(centre) and B = A+1 (a = centre - A).
    // Without it (or with "Exact frame timing" off) we use the frame Premiere
    // picked plus neighbours one source interval away.
    struct Plan
    {
        bool exact = false;
        long long frameA = 0;
        double centre = 0, a = 0;
        PrTime seqTime[2] = {}; // dependency times: exact = A, B; fallback = prev, next
    };

    Plan MakePlan(const PrGPUFilterRenderParams* rp)
    {
        Plan p;
        const double fps = SourceFps();
        const double tpf = oe::kPrTicksPerSecond / fps; // media ticks per source frame
        const bool wantExact = ParamNumber(GetParam(OE_EXACT_TIMING, rp->inClipTime)) != 0.0;
        if (wantExact && mHaveInPoint && mSpeed > 0 && fps > 0)
        {
            const PrTime trackStart = rp->inSequenceTime - rp->inClipTime;
            const double mediaTime = (double)mInPoint + (double)rp->inClipTime * mSpeed;
            p.centre = mediaTime / tpf;
            p.frameA = (long long)std::floor(p.centre + 1e-6);
            p.a = std::clamp(p.centre - (double)p.frameA, 0.0, 1.0);
            p.exact = true;
            for (int k = 0; k < 2; ++k)
            {
                // a quarter frame past the frame's own timestamp: picks that
                // frame whether Premiere rounds to nearest or floors
                const double media = ((double)(p.frameA + k) + 0.25) * tpf;
                p.seqTime[k] = trackStart + (PrTime)std::llround((media - (double)mInPoint) / mSpeed);
            }
        }
        else
        {
            const PrTime dt = SourceFrameTicks(rp);
            p.seqTime[0] = rp->inSequenceTime - dt;
            p.seqTime[1] = rp->inSequenceTime + dt;
        }
        return p;
    }

    prSuiteError GetFrameDependencies(const PrGPUFilterRenderParams* inRenderParams, csSDK_int32* ioQueryIndex,
                                      PrGPUFilterFrameDependency* outFrameRequirements) override
    {
        // One dependency per call; incrementing ioQueryIndex asks Premiere to
        // call again for the next one. Two either way (A, B or prev, next).
        const int q = *ioQueryIndex;
        if (q >= 2)
            return suiteError_NotImplemented;
        const Plan plan = MakePlan(inRenderParams);
        outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
        outFrameRequirements->outTrackID = 0;
        outFrameRequirements->outSequenceTime = plan.seqTime[q];
        if (q == 0)
            *ioQueryIndex += 1;
        return suiteError_NoError;
    }

    prSuiteError Render(const PrGPUFilterRenderParams* inRenderParams, const PPixHand* inFrames,
                        csSDK_size_t inFrameCount, PPixHand* outFrame) override
    {
        if (inFrameCount < 1 || !inFrames[0])
            return suiteError_Fail;
        const PrTime t = inRenderParams->inClipTime;
        const int view = PopupValue(GetParam(OE_VIEW, t));
        const Plan plan = MakePlan(inRenderParams);

        // Frame 0 = Premiere's current frame; then our two dependencies in order.
        oe::gpu::Frame f0, d1, d2, out;
        if (!Describe(inFrames[0], &f0))
            return suiteError_Fail;
        const bool have1 = inFrameCount > 1 && inFrames[1] && Describe(inFrames[1], &d1);
        const bool have2 = inFrameCount > 2 && inFrames[2] && Describe(inFrames[2], &d2);

        PrPixelFormat pf = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(inFrames[0], &pf);
        csSDK_uint32 parN = 1, parD = 1;
        mPPixSuite->GetPixelAspectRatio(inFrames[0], &parN, &parD);
        prFieldType field = 0;
        mPPix2Suite->GetFieldOrder(inFrames[0], &field);
        csSDK_uint32 dev = mDeviceIndex;
        mGPUDeviceSuite->GetGPUPPixDeviceIndex(inFrames[0], &dev);
        if (mGPUDeviceSuite->CreateGPUPPix(dev, pf, f0.width, f0.height, parN, parD, field, outFrame) !=
                suiteError_NoError ||
            !*outFrame || !Describe(*outFrame, &out))
            return suiteError_Fail;

        const double timelineFps = oe::kPrTicksPerSecond / (double)std::max<PrTime>(1, inRenderParams->inRenderTicksPerFrame);
        const double srcRate = SourceFps() * mSpeed;
        oe::gpu::Settings s;
        s.halfWidth = oe::BlurHalfWidth(srcRate, timelineFps, ParamNumber(GetParam(OE_SRC_SHUTTER, t)),
                                        ParamNumber(GetParam(OE_TARGET_SHUTTER, t)));
        const int q = PopupValue(GetParam(OE_QUALITY, t));
        // Analysis resolution relative to the frame (validated: half-res, grid 2 scored best)
        if (q == OE_QUALITY_FAST)
        {
            s.flowWidth = std::max(256, f0.width / 4);
            s.flowGrid = 4;
        }
        else if (q == OE_QUALITY_HIGH)
        {
            s.flowWidth = std::max(512, f0.width / 2);
            s.flowGrid = 1;
        }
        else
        {
            s.flowWidth = std::max(512, f0.width / 2);
            s.flowGrid = 2;
        }
        s.showVectors = view == OE_VIEW_VECTORS;

        // Diagnostics for the first renders of each instance: which of our
        // requested frames is identical to Premiere's own current frame.
        // Logged in both timing modes: requested times (relative to this render,
        // in timeline frames) and pairwise differences (0.00 = same picture).
        char match[192] = "";
        if (mLog && mDiagRenders < 20 && have1 && have2)
        {
            ++mDiagRenders;
            void* st0 = mDeviceInfo.outCommandQueueHandle;
            const double tlFrame = (double)std::max<PrTime>(1, inRenderParams->inRenderTicksPerFrame);
            snprintf(match, sizeof(match),
                     " | req d1=%+.3f d2=%+.3f tl-frames | diff f0-d1=%.2f f0-d2=%.2f d1-d2=%.2f",
                     (double)(plan.seqTime[0] - inRenderParams->inSequenceTime) / tlFrame,
                     (double)(plan.seqTime[1] - inRenderParams->inSequenceTime) / tlFrame,
                     mEngine->SampledDiff(f0, d1, st0), mEngine->SampledDiff(f0, d2, st0),
                     mEngine->SampledDiff(d1, d2, st0));
        }

        oe::gpu::Stats st;
        std::string err;
        bool ok;
        const bool pair = plan.exact && have1 && have2;
        if (pair)
            ok = mEngine->ProcessPair(d1, d2, plan.a, out, s, mDeviceInfo.outCommandQueueHandle, &st, &err);
        else
            ok = mEngine->Process(have1 ? &d1 : nullptr, f0, have2 ? &d2 : nullptr, out, s,
                                  mDeviceInfo.outCommandQueueHandle, &st, &err);
        if (mLog)
            oe::Log("[gpu] render seq=%.4fs clip=%.4fs view=%d %dx%d %s | srcFps=%.3f (%s) speed=%.2f timeline=%.3f "
                    "frames=%zu | timing=%s src=%.3f A=%lld a=%.3f | halfWidth=%.3f flow=%d/g%d | flow=%.2fms "
                    "blur=%.2fms total=%.2fms%s%s%s",
                    inRenderParams->inSequenceTime / oe::kPrTicksPerSecond, t / oe::kPrTicksPerSecond, view, f0.width,
                    f0.height, f0.half ? "16f" : "32f", SourceFps(), mFpsSource.c_str(), mSpeed, timelineFps,
                    (size_t)inFrameCount, pair ? "exact-pair" : "nearest", plan.centre, plan.frameA, plan.a,
                    s.halfWidth, s.flowWidth, s.flowGrid, st.flowMs, st.blurMs, st.totalMs, match, ok ? "" : " ERROR: ",
                    ok ? "" : err.c_str());
        return ok ? suiteError_NoError : suiteError_Fail;
    }

private:
    bool Describe(PPixHand h, oe::gpu::Frame* f)
    {
        void* data = nullptr;
        if (mGPUDeviceSuite->GetGPUPPixData(h, &data) != suiteError_NoError || !data)
            return false;
        csSDK_int32 rowBytes = 0;
        mPPixSuite->GetRowBytes(h, &rowBytes);
        prRect b = {};
        mPPixSuite->GetBounds(h, &b);
        PrPixelFormat pf = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(h, &pf);
        f->ptr = data;
        f->pitch = (size_t)rowBytes;
        f->width = b.right - b.left;
        f->height = b.bottom - b.top;
        f->half = pf != PrPixelFormat_GPU_BGRA_4444_32f;
        return f->width > 0 && f->height > 0;
    }

    template <typename T>
    bool NodeProperty(csSDK_int32 node, const char* key, T& value)
    {
        PrMemoryPtr buf = nullptr;
        if (mVideoSegmentSuite->GetNodeProperty(node, key, &buf) != suiteError_NoError || !buf)
            return false;
        std::istringstream ss((const char*)buf);
        ss >> value;
        mMemoryManagerSuite->PrDisposePtr(buf);
        return !ss.fail();
    }

    std::string NodeString(csSDK_int32 node, const char* key)
    {
        PrMemoryPtr buf = nullptr;
        if (mVideoSegmentSuite->GetNodeProperty(node, key, &buf) != suiteError_NoError || !buf)
            return "?";
        std::string s((const char*)buf);
        mMemoryManagerSuite->PrDisposePtr(buf);
        return s;
    }

    // Find the clip's media frame rate and speed: effect node -> owning clip
    // node -> (inputs) -> media node.
    void DetectSource()
    {
        csSDK_int32 clip = 0;
        if (mVideoSegmentSuite->AcquireOperatorOwnerNodeID(mNodeID, &clip) != suiteError_NoError || !clip)
        {
            mFpsSource = "no clip node";
            return;
        }
        double speed = 0;
        if (NodeProperty(clip, kVideoSegmentProperty_Clip_ClipSpeed, speed) && speed != 0)
            mSpeed = std::fabs(speed);

        // breadth-first through the clip's inputs, a few levels deep
        csSDK_int32 frontier[16] = {clip};
        int count = 1;
        for (int depth = 0; depth < 4 && count > 0 && mMediaTicks == 0; ++depth)
        {
            csSDK_int32 nextLevel[16];
            int nextCount = 0;
            for (int i = 0; i < count && mMediaTicks == 0; ++i)
            {
                csSDK_int32 n = 0;
                mVideoSegmentSuite->GetNodeInputCount(frontier[i], &n);
                for (csSDK_int32 k = 0; k < n && mMediaTicks == 0; ++k)
                {
                    PrTime offset = 0;
                    csSDK_int32 in = 0;
                    if (mVideoSegmentSuite->AcquireInputNodeID(frontier[i], k, &offset, &in) != suiteError_NoError ||
                        !in)
                        continue;
                    char type[kMaxNodeTypeStringSize] = {0};
                    prPluginID hash = {};
                    csSDK_int32 flags = 0;
                    mVideoSegmentSuite->GetNodeInfo(in, type, &hash, &flags);
                    if (strcmp(type, kVideoSegment_NodeType_Media) == 0)
                    {
                        // Some nodes report INT64_MAX ("no rate"): accept 1..1000 fps only.
                        PrTime ticks = 0;
                        if (NodeProperty(in, kVideoSegmentProperty_Media_StreamFrameRate, ticks) &&
                            ticks >= (PrTime)(oe::kPrTicksPerSecond / 1000.0) && ticks <= (PrTime)oe::kPrTicksPerSecond)
                            mMediaTicks = ticks;
                        double ms = 0;
                        if (mSpeed == 1.0 && NodeProperty(in, kVideoSegmentProperty_Media_ClipSpeed, ms) && ms != 0)
                            mSpeed = std::fabs(ms);
                        long long inPt = 0;
                        if (NodeProperty(in, kVideoSegmentProperty_Media_InPointMediaTimeAsTicks, inPt))
                        {
                            mInPoint = (PrTime)inPt;
                            mHaveInPoint = true;
                        }
                        bool backwards = false;
                        if (NodeProperty(in, kVideoSegmentProperty_Media_ClipBackwards, backwards) && backwards)
                            mHaveInPoint = false; // reversed clips: keep the simple nearest-frame path
                        mVideoSegmentSuite->ReleaseVideoNodeID(in);
                    }
                    else if (nextCount < 16)
                        nextLevel[nextCount++] = in; // released after the walk
                    else
                        mVideoSegmentSuite->ReleaseVideoNodeID(in);
                }
            }
            for (int i = (depth == 0 ? 1 : 0); i < count; ++i) // frontier[0] at depth 0 is the clip, released below
                mVideoSegmentSuite->ReleaseVideoNodeID(frontier[i]);
            for (int i = 0; i < nextCount; ++i)
                frontier[i] = nextLevel[i];
            count = nextCount;
        }
        for (int i = 0; i < count; ++i)
            mVideoSegmentSuite->ReleaseVideoNodeID(frontier[i]);
        mVideoSegmentSuite->ReleaseVideoNodeID(clip);
        mFpsSource = mMediaTicks ? "media stream rate" : "media node not found";
        if (mLog)
            oe::Log("[gpu] instance node=%d clip='%s' file='%s' streamTicksPerFrame=%lld (%.3f fps) speed=%.3f "
                    "inPoint=%s%lld ticks (%.4f source frames)",
                    mNodeID, NodeString(mNodeID, kVideoSegmentProperty_Effect_ClipName).c_str(),
                    NodeString(mNodeID, kVideoSegmentProperty_Effect_FileName).c_str(), (long long)mMediaTicks,
                    mMediaTicks ? oe::kPrTicksPerSecond / (double)mMediaTicks : 0.0, mSpeed,
                    mHaveInPoint ? "" : "(none) ", (long long)mInPoint,
                    mMediaTicks ? (double)mInPoint / (double)mMediaTicks : 0.0);
    }

    double SourceFps()
    {
        const double user = ParamNumber(GetParam(OE_SRC_FPS, 0));
        if (user > 0)
        {
            mFpsSource = "user";
            return user;
        }
        if (mMediaTicks > 0)
            return oe::kPrTicksPerSecond / (double)mMediaTicks;
        return 50.0; // last resort; logged as "media node not found"
    }

    PrTime SourceFrameTicks(const PrGPUFilterRenderParams* rp)
    {
        const double fps = SourceFps() * mSpeed;
        if (fps <= 0)
            return rp->inRenderTicksPerFrame / 2;
        return (PrTime)std::llround(oe::kPrTicksPerSecond / fps);
    }

    std::shared_ptr<oe::gpu::Engine> mEngine;
    PrTime mMediaTicks = 0;
    PrTime mInPoint = 0;
    bool mHaveInPoint = false;
    int mDiagRenders = 0;
    double mSpeed = 1.0;
    std::string mFpsSource = "not detected";
    bool mLog = false;
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<OneEightyGPU>)
