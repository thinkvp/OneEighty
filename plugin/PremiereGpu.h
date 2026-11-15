#pragma once
// Premiere's GPU through its MediaCore GPU Device Suite (Windows). Kept apart
// from OneEighty.cpp: the Premiere SDK's headers clash with the After Effects
// SDK's (PrSDKString), so only this file includes them.

#include "gpu/GpuMotionBlur.h"

#include <memory>

namespace oe::premiere
{

// For pica_basicP->AcquireSuite(GpuDeviceSuiteName(), GpuDeviceSuiteVersion(), &suite).
const char* GpuDeviceSuiteName();
int GpuDeviceSuiteVersion();

struct GpuDeviceInfo
{
    int err = 0;          // suite error (0 = none)
    int framework = -1;   // PrGPUDeviceFramework (0 = CUDA)
    bool cuda = false;    // framework is CUDA
    bool meetsMinimum = false;
    void* context = nullptr; // CUcontext (null until Premiere has set up the device)
};

// Premiere's device 0 (the one it renders on).
GpuDeviceInfo DeviceInfo(const void* suite);

// The device as an engine's host device (oe::gpu::HostDevice): Premiere's
// context, exclusive access, device memory from Premiere.
std::unique_ptr<gpu::HostDevice> MakeHostDevice(const void* suite, void* context);

} // namespace oe::premiere
