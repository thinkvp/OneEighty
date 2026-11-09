// Windows side of Platform.h.
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")  // update request
#pragma comment(lib, "shell32.lib")  // SHGetKnownFolderPath, ShellExecute
#pragma comment(lib, "ole32.lib")    // CoTaskMemFree
#pragma comment(lib, "advapi32.lib") // RegGetValueW (opt-out)
#pragma comment(lib, "user32.lib")   // MessageBox

#include "Platform.h"

#include <vector>

namespace oe::platform
{
namespace
{
std::wstring Widen(const std::string& s)
{
    if (s.empty())
        return L"";
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
} // namespace

std::filesystem::path LogPath()
{
    wchar_t dir[MAX_PATH + 1] = {0};
    const DWORD n = GetTempPathW(MAX_PATH + 1, dir);
    if (n == 0 || n > MAX_PATH)
        return {};
    return std::filesystem::path(dir) / L"OneEighty.log";
}

// Via the Known Folders API: the LOCALAPPDATA environment variable isn't
// reliable inside host processes.
std::filesystem::path DataDir()
{
    PWSTR base = nullptr;
    std::filesystem::path dir;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &base)) && base)
    {
        dir = std::filesystem::path(base) / L"OneEighty";
        CreateDirectoryW(dir.c_str(), nullptr);
    }
    CoTaskMemFree(base);
    return dir;
}

unsigned long ThreadId() { return GetCurrentThreadId(); }

void LocalTime(int* h, int* m, int* s, int* ms)
{
    SYSTEMTIME st;
    GetLocalTime(&st);
    *h = st.wHour;
    *m = st.wMinute;
    *s = st.wSecond;
    *ms = st.wMilliseconds;
}

bool HttpsGet(const std::string& host, const std::string& path, const std::string& userAgent,
              const std::string& accept, int timeoutMs, std::string* body, int* status, std::string* err)
{
    *status = 0;
    const std::wstring ua = Widen(userAgent);
    HINTERNET s = WinHttpOpen(ua.c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                              WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s)
        s = WinHttpOpen(ua.c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                        WINHTTP_NO_PROXY_BYPASS, 0); // pre-8.1 fallback
    if (!s)
    {
        *err = "WinHttpOpen failed";
        return false;
    }
    WinHttpSetTimeouts(s, timeoutMs, timeoutMs, timeoutMs, timeoutMs);
    bool ok = false;
    HINTERNET c = WinHttpConnect(s, Widen(host).c_str(), INTERNET_DEFAULT_HTTPS_PORT, 0);
    HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", Widen(path).c_str(), nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE)
                    : nullptr;
    const std::wstring headers = L"Accept: " + Widen(accept) + L"\r\n";
    if (r && WinHttpSendRequest(r, headers.c_str(), (DWORD)-1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
        WinHttpReceiveResponse(r, nullptr))
    {
        DWORD code = 0, len = sizeof(code);
        WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                            &code, &len, WINHTTP_NO_HEADER_INDEX);
        DWORD avail = 0;
        while (WinHttpQueryDataAvailable(r, &avail) && avail > 0 && body->size() < 1 << 20)
        {
            std::vector<char> buf(avail);
            DWORD got = 0;
            if (!WinHttpReadData(r, buf.data(), avail, &got) || got == 0)
                break;
            body->append(buf.data(), got);
        }
        *status = (int)code;
        ok = true;
    }
    else
        *err = "request failed (" + std::to_string(GetLastError()) + ")";
    if (r)
        WinHttpCloseHandle(r);
    if (c)
        WinHttpCloseHandle(c);
    WinHttpCloseHandle(s);
    return ok;
}

bool PolicyDisablesUpdateCheck()
{
    for (HKEY root : {HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE})
    {
        DWORD v = 0, size = sizeof(v);
        if (RegGetValueW(root, L"Software\\thinkvp\\OneEighty", L"DisableUpdateCheck", RRF_RT_REG_DWORD, nullptr,
                         &v, &size) == ERROR_SUCCESS &&
            v != 0)
            return true;
    }
    return false;
}

bool AskYesNo(const std::string& title, const std::string& message)
{
    return MessageBoxW(GetActiveWindow(), Widen(message).c_str(), Widen(title).c_str(),
                       MB_YESNO | MB_ICONINFORMATION) == IDYES;
}

void ShowMessage(const std::string& title, const std::string& message, bool warning)
{
    MessageBoxW(GetActiveWindow(), Widen(message).c_str(), Widen(title).c_str(),
                MB_OK | (warning ? MB_ICONWARNING : MB_ICONINFORMATION));
}

void OpenUrl(const std::string& url)
{
    ShellExecuteW(nullptr, L"open", Widen(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

} // namespace oe::platform
