#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#include "OneEightyParams.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace oe
{
namespace
{
std::mutex gLogMutex;
int gLogLines = 0;
constexpr int kMaxLogLines = 5000;
} // namespace

void Log(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    if (gLogLines >= kMaxLogLines)
        return;
    char path[MAX_PATH] = {0};
    DWORD n = GetTempPathA(MAX_PATH, path);
    if (n == 0 || n > MAX_PATH - 20)
        return;
    strcat_s(path, "OneEighty.log");
    FILE* f = nullptr;
    if (fopen_s(&f, path, "a") != 0 || !f)
        return;
    SYSTEMTIME st;
    GetLocalTime(&st);
    fprintf(f, "%02d:%02d:%02d.%03d [t%05lu] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
            GetCurrentThreadId() % 100000);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
    ++gLogLines;
}

} // namespace oe
