#include "OneEightyParams.h"
#include "Platform.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace oe
{
namespace
{
std::mutex gLogMutex;
int gLogLines = 0;
constexpr int kMaxLogLines = 5000;

FILE* OpenAppend(const std::filesystem::path& p)
{
#ifdef _WIN32
    FILE* f = nullptr;
    return _wfopen_s(&f, p.c_str(), L"a") == 0 ? f : nullptr;
#else
    return std::fopen(p.c_str(), "a");
#endif
}
} // namespace

void Log(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    if (gLogLines >= kMaxLogLines)
        return;
    static const std::filesystem::path path = platform::LogPath();
    if (path.empty())
        return;
    FILE* f = OpenAppend(path);
    if (!f)
        return;
    int h, m, s, ms;
    platform::LocalTime(&h, &m, &s, &ms);
    fprintf(f, "%02d:%02d:%02d.%03d [t%05lu] ", h, m, s, ms, platform::ThreadId() % 100000);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
    ++gLogLines;
}

} // namespace oe
