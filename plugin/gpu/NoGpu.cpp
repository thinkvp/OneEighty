// Builds without a GPU engine (macOS until the Metal port lands): Engine
// factories report "unavailable", so the effect renders on the CPU and Quality
// Extra High falls back to High, exactly as on a Windows PC without NVIDIA.
#include "GpuMotionBlur.h"
#include "../rife/RifeFlow.h"

namespace oe::gpu
{
std::unique_ptr<Engine> Engine::Create(std::string* err)
{
    if (err)
        *err = "no GPU engine in this build";
    return nullptr;
}
} // namespace oe::gpu

namespace oe::rife
{
std::unique_ptr<Engine> Engine::Create(const std::wstring&, std::string* err)
{
    if (err)
        *err = "no RIFE engine in this build";
    return nullptr;
}

std::wstring DefaultModelDir() { return {}; }
} // namespace oe::rife
