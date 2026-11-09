#pragma once
// Runs a function on a thread with a large stack and waits for it.
//
// Premiere on macOS renders on threads with 512 KB stacks (measured: 73 of
// Premiere 2026's threads). ncnn's Vulkan device setup and its glslang shader
// compiles need more than that, so Quality Extra High crashed Premiere's render
// thread ("A low-level exception occurred in: OneEighty Motion Blur
// (AEVideoFilter:11)") on a real Apple GPU. Windows threads get 1 MB or more,
// and command-line tools run on the 8 MB main thread, so neither showed it.
//
// On macOS fn runs on a fresh thread with kLargeStackBytes of stack (falling
// back to the calling thread if that thread can't be created); exceptions are
// carried back to the caller. Elsewhere fn simply runs on the calling thread.

#include <cstddef>
#include <exception>
#include <type_traits>

#ifdef __APPLE__
    #include <pthread.h>
#endif

namespace oe
{

constexpr std::size_t kLargeStackBytes = std::size_t(16) << 20; // 16 MB

template <class F>
void RunOnLargeStack(F&& fn, std::size_t stackBytes = kLargeStackBytes)
{
#ifdef __APPLE__
    using Fn = std::remove_reference_t<F>;
    struct Ctx
    {
        Fn* fn;
        std::exception_ptr error;
    } ctx{&fn, nullptr};
    auto entry = [](void* p) -> void* {
        Ctx* c = static_cast<Ctx*>(p);
        try
        {
            (*c->fn)();
        }
        catch (...)
        {
            c->error = std::current_exception();
        }
        return nullptr;
    };
    pthread_attr_t attr;
    bool started = false;
    pthread_t th;
    if (pthread_attr_init(&attr) == 0)
    {
        if (pthread_attr_setstacksize(&attr, stackBytes) == 0)
            started = pthread_create(&th, &attr, entry, &ctx) == 0;
        pthread_attr_destroy(&attr);
    }
    if (!started)
    {
        fn();
        return;
    }
    pthread_join(th, nullptr);
    if (ctx.error)
        std::rethrow_exception(ctx.error);
#else
    (void)stackBytes;
    fn();
#endif
}

} // namespace oe
