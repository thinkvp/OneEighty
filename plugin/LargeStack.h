#pragma once
// Runs a function on a thread with a chosen stack size and waits for it.
//
// Used by oe_gpu_offline (OE_STACK_KB=512) to run the effect's maths on a
// stack the size of Premiere's macOS render threads (512 KB, measured on
// Premiere 2026), which command-line tools on the 8 MB main thread never see.
// The effect itself runs on Premiere's own thread, where Premiere catches a
// crash and disables the effect instead of quitting.
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
