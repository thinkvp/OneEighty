#pragma once
// The few things the effect needs from the operating system, so the rest of
// the code builds on Windows and macOS alike. Platform_win.cpp / Platform_mac.mm.

#include <filesystem>
#include <string>

namespace oe::platform
{

// Debug log file: %TEMP%\OneEighty.log (Windows), ~/Library/Logs/OneEighty.log (macOS).
std::filesystem::path LogPath();

// Per-user folder for the update cache, created if needed (empty on failure):
// %LOCALAPPDATA%\OneEighty (Windows), ~/Library/Application Support/OneEighty (macOS).
std::filesystem::path DataDir();

// Short id of the calling thread, for the log.
unsigned long ThreadId();

// Local wall-clock time of day, for the log.
void LocalTime(int* h, int* m, int* s, int* ms);

// One HTTPS GET. Returns false (and sets *err) on network failure; *status is
// the HTTP status code when a response arrived. Uses the system proxy settings.
bool HttpsGet(const std::string& host, const std::string& path, const std::string& userAgent,
              const std::string& accept, int timeoutMs, std::string* body, int* status, std::string* err);

// IT policy switching off the automatic update check: DWORD DisableUpdateCheck=1
// under HKCU / HKLM\Software\thinkvp\OneEighty (Windows); preference
// DisableUpdateCheck in the domain com.thinkvp.OneEighty, user or managed
// (macOS). The environment variable is checked by the caller.
bool PolicyDisablesUpdateCheck();

// Modal message boxes from the host's UI thread.
bool AskYesNo(const std::string& title, const std::string& message);
void ShowMessage(const std::string& title, const std::string& message, bool warning);

// Opens a web page in the default browser.
void OpenUrl(const std::string& url);

} // namespace oe::platform
