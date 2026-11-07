/*
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

#include <cmath>
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

        const int mode = PopupValue(GetParam(OE_MODE, 0));
        mLog = ParamNumber(GetParam(OE_LOG, 0)) != 0.0;

        if (mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_CUDA)
        {
            if (mLog)
                oe::Log("[gpu] device %u framework %d is not CUDA -> CPU fallback (set the project renderer to "
                        "Mercury Playback Engine GPU Acceleration (CUDA))",
                        mDeviceIndex, (int)mDeviceInfo.outDeviceFramework);
            return suiteError_Fail;
        }
        if (mode == OE_MODE_BLEND_CENTRED) // test mode only exists on the CPU path
            return suiteError_Fail;

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

    prSuiteError GetFrameDependencies(const PrGPUFilterRenderParams* inRenderParams, csSDK_int32* ioQueryIndex,
                                      PrGPUFilterFrameDependency* outFrameRequirements) override
    {
        const int mode = PopupValue(GetParam(OE_MODE, inRenderParams->inClipTime));
        if (mode == OE_MODE_OFF)
            return suiteError_NotImplemented; // only the current frame

        const PrTime dt = SourceFrameTicks(inRenderParams);
        const bool wantPrev = mode != OE_MODE_SHOW_NEXT;
        const bool wantNext = mode != OE_MODE_SHOW_PREVIOUS;

        // Dependencies are returned one per call; incrementing ioQueryIndex asks
        // Premiere to call again for the next one.
        int slot = *ioQueryIndex; // 0 = first dependency
        if (!wantPrev)
            ++slot; // skip "prev"
        if (slot > 1 || (slot == 1 && !wantNext))
            return suiteError_NotImplemented;

        outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
        outFrameRequirements->outTrackID = 0;
        outFrameRequirements->outSequenceTime = inRenderParams->inSequenceTime + (slot == 0 ? -dt : dt);
        const bool more = slot == 0 && wantNext && wantPrev;
        if (more)
            *ioQueryIndex += 1;
        return suiteError_NoError;
    }

    prSuiteError Render(const PrGPUFilterRenderParams* inRenderParams, const PPixHand* inFrames,
                        csSDK_size_t inFrameCount, PPixHand* outFrame) override
    {
        if (inFrameCount < 1 || !inFrames[0])
            return suiteError_Fail;
        const PrTime t = inRenderParams->inClipTime;
        const int mode = PopupValue(GetParam(OE_MODE, t));
        const bool wantPrev = mode != OE_MODE_SHOW_NEXT && mode != OE_MODE_OFF;
        const bool wantNext = mode != OE_MODE_SHOW_PREVIOUS && mode != OE_MODE_OFF;

        // Frame 0 = current; then dependencies in the order we returned them.
        size_t idx = 1;
        PPixHand prevH = nullptr, nextH = nullptr;
        if (wantPrev && idx < inFrameCount)
            prevH = inFrames[idx++];
        if (wantNext && idx < inFrameCount)
            nextH = inFrames[idx++];

        oe::gpu::Frame cur, prev, next, out;
        if (!Describe(inFrames[0], &cur))
            return suiteError_Fail;
        const bool havePrev = prevH && Describe(prevH, &prev);
        const bool haveNext = nextH && Describe(nextH, &next);

        PrPixelFormat pf = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(inFrames[0], &pf);
        csSDK_uint32 parN = 1, parD = 1;
        mPPixSuite->GetPixelAspectRatio(inFrames[0], &parN, &parD);
        prFieldType field = 0;
        mPPix2Suite->GetFieldOrder(inFrames[0], &field);
        csSDK_uint32 dev = mDeviceIndex;
        mGPUDeviceSuite->GetGPUPPixDeviceIndex(inFrames[0], &dev);
        if (mGPUDeviceSuite->CreateGPUPPix(dev, pf, cur.width, cur.height, parN, parD, field, outFrame) !=
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
            s.flowWidth = std::max(256, cur.width / 4);
            s.flowGrid = 4;
        }
        else if (q == OE_QUALITY_HIGH)
        {
            s.flowWidth = std::max(512, cur.width / 2);
            s.flowGrid = 1;
        }
        else
        {
            s.flowWidth = std::max(512, cur.width / 2);
            s.flowGrid = 2;
        }

        const oe::gpu::Frame* pPrev = havePrev ? &prev : nullptr;
        const oe::gpu::Frame* pNext = haveNext ? &next : nullptr;
        const oe::gpu::Frame* src = &cur;
        if (mode == OE_MODE_SHOW_PREVIOUS || mode == OE_MODE_SHOW_NEXT || mode == OE_MODE_OFF)
        {
            // plain copy of the requested frame (halfWidth 0 = copy path)
            if (mode == OE_MODE_SHOW_PREVIOUS && pPrev)
                src = pPrev;
            if (mode == OE_MODE_SHOW_NEXT && pNext)
                src = pNext;
            s.halfWidth = 0;
            pPrev = pNext = nullptr;
        }

        oe::gpu::Stats st;
        std::string err;
        const bool ok = mEngine->Process(pPrev, *src, pNext, out, s, mDeviceInfo.outCommandQueueHandle, &st, &err);
        if (mLog)
            oe::Log("[gpu] render seq=%.4fs clip=%.4fs mode=%d %dx%d %s | srcFps=%.3f (%s) speed=%.2f timeline=%.3f "
                    "frames=%zu prev:%s next:%s | halfWidth=%.3f flow=%d/g%d | flow=%.2fms blur=%.2fms total=%.2fms%s%s",
                    inRenderParams->inSequenceTime / oe::kPrTicksPerSecond, t / oe::kPrTicksPerSecond, mode, cur.width,
                    cur.height, cur.half ? "16f" : "32f", SourceFps(), mFpsSource.c_str(), mSpeed, timelineFps,
                    (size_t)inFrameCount, havePrev ? "ok" : "-", haveNext ? "ok" : "-", s.halfWidth, s.flowWidth,
                    s.flowGrid, st.flowMs, st.blurMs, st.totalMs, ok ? "" : " ERROR: ", ok ? "" : err.c_str());
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
                        PrTime ticks = 0;
                        if (NodeProperty(in, kVideoSegmentProperty_Media_StreamFrameRate, ticks) && ticks > 0)
                            mMediaTicks = ticks;
                        double ms = 0;
                        if (mSpeed == 1.0 && NodeProperty(in, kVideoSegmentProperty_Media_ClipSpeed, ms) && ms != 0)
                            mSpeed = std::fabs(ms);
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
            oe::Log("[gpu] instance node=%d clip='%s' file='%s' streamTicksPerFrame=%lld (%.3f fps) speed=%.3f",
                    mNodeID, NodeString(mNodeID, kVideoSegmentProperty_Effect_ClipName).c_str(),
                    NodeString(mNodeID, kVideoSegmentProperty_Effect_FileName).c_str(), (long long)mMediaTicks,
                    mMediaTicks ? oe::kPrTicksPerSecond / (double)mMediaTicks : 0.0, mSpeed);
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
    double mSpeed = 1.0;
    std::string mFpsSource = "not detected";
    bool mLog = false;
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<OneEightyGPU>)
