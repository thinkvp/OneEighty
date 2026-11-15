// See PremiereGpu.h. Premiere SDK headers only (sdk/premiere/Examples/Headers).

#include "PremiereGpu.h"

#ifdef NOMINMAX
    #undef NOMINMAX // the build defines it; PrSDKTypes.h defines it again
#endif
#include "PrSDKGPUDeviceSuite.h"

namespace oe::premiere
{
namespace
{
constexpr csSDK_uint32 kDevice = 0;

class Device final : public gpu::HostDevice
{
public:
    Device(const PrSDKGPUDeviceSuite* s, void* ctx) : s_(s), ctx_(ctx) {}
    void* Context() override { return ctx_; }
    bool Acquire() override { return s_->AcquireExclusiveDeviceAccess(kDevice) == suiteError_NoError; }
    void Release() override { s_->ReleaseExclusiveDeviceAccess(kDevice); }
    void* Allocate(size_t bytes) override
    {
        void* p = nullptr;
        return s_->AllocateDeviceMemory(kDevice, bytes, &p) == suiteError_NoError ? p : nullptr;
    }
    void Free(void* p) override { s_->FreeDeviceMemory(kDevice, p); }

private:
    const PrSDKGPUDeviceSuite* s_;
    void* ctx_;
};
} // namespace

const char* GpuDeviceSuiteName() { return kPrSDKGPUDeviceSuite; }
int GpuDeviceSuiteVersion() { return kPrSDKGPUDeviceSuiteVersion; }

GpuDeviceInfo DeviceInfo(const void* suite)
{
    GpuDeviceInfo r;
    PrGPUDeviceInfo info = {};
    r.err = (int)static_cast<const PrSDKGPUDeviceSuite*>(suite)->GetDeviceInfo(kPrSDKGPUDeviceSuiteVersion, kDevice,
                                                                                &info);
    if (r.err != suiteError_NoError)
        return r;
    r.framework = (int)info.outDeviceFramework;
    r.cuda = info.outDeviceFramework == PrGPUDeviceFramework_CUDA;
    r.meetsMinimum = info.outMeetsMinimumRequirementsForAcceleration != 0;
    r.context = info.outContextHandle;
    return r;
}

std::unique_ptr<gpu::HostDevice> MakeHostDevice(const void* suite, void* context)
{
    return std::make_unique<Device>(static_cast<const PrSDKGPUDeviceSuite*>(suite), context);
}

} // namespace oe::premiere
