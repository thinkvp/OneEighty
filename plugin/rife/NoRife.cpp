// Builds without RIFE (macOS until ncnn is ported there): the engine factory
// reports "unavailable", so Quality Extra High renders as High, exactly as on
// a Windows PC without Vulkan or the model.
#include "RifeFlow.h"

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
