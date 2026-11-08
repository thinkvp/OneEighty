#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")  // update request
#pragma comment(lib, "shell32.lib")  // SHGetKnownFolderPath
#pragma comment(lib, "ole32.lib")    // CoTaskMemFree
#pragma comment(lib, "advapi32.lib") // RegGetValueW (opt-out)

#include "UpdateCheck.h"
#include "OneEightyParams.h"

#include <atomic>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace oe::update
{
namespace
{

constexpr wchar_t kHost[] = L"api.github.com";
constexpr wchar_t kPath[] = L"/repos/thinkvp/OneEighty/releases/latest";
constexpr char kReleasesPage[] = "https://github.com/thinkvp/OneEighty/releases/latest";
constexpr long long kCheckIntervalSec = 24 * 3600;

std::mutex gMu;
Status gStatus;
bool gLoadedCache = false;
std::atomic<bool> gBackgroundStarted{false};
long long gLastCheck = 0; // unix seconds

// %LOCALAPPDATA%\OneEighty\update.txt, via the Known Folders API (the
// LOCALAPPDATA environment variable isn't reliable inside host processes).
std::wstring CachePath()
{
    PWSTR base = nullptr;
    std::wstring path;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &base)) && base)
    {
        const std::wstring dir = std::wstring(base) + L"\\OneEighty";
        CreateDirectoryW(dir.c_str(), nullptr);
        path = dir + L"\\update.txt";
    }
    CoTaskMemFree(base);
    return path;
}

std::string Narrow(const std::wstring& w)
{
    if (w.empty())
        return "";
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

std::string Trim(std::string s)
{
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' '))
        s.pop_back();
    return s;
}

// cache: line 1 = unix time of last successful check, line 2 = latest version, line 3 = page url
void LoadCacheLocked()
{
    if (gLoadedCache)
        return;
    gLoadedCache = true;
    const std::wstring path = CachePath();
    std::ifstream f(path);
    long long t = 0;
    std::string latest, url;
    bool loaded = false;
    if (f >> t)
    {
        loaded = true;
        std::getline(f, latest); // rest of line 1
        std::getline(f, latest);
        std::getline(f, url);
        latest = Trim(latest);
        url = Trim(url);
        gLastCheck = t;
        if (!latest.empty())
        {
            gStatus.latest = latest;
            gStatus.pageUrl = url.empty() ? kReleasesPage : url;
            gStatus.state = CompareVersions(latest, CurrentVersion()) > 0 ? State::Available : State::UpToDate;
        }
    }
    Log("[update] cache %s: %s (last check %lld, latest '%s')", Narrow(path).c_str(), loaded ? "loaded" : "not found",
        gLastCheck, gStatus.latest.c_str());
}

void SaveCacheLocked()
{
    const std::wstring path = CachePath();
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    f << gLastCheck << "\n" << gStatus.latest << "\n" << gStatus.pageUrl << "\n";
    f.close();
    Log("[update] cache saved to %s: %s", Narrow(path).c_str(), f.fail() ? "FAILED" : "ok");
}

// Value of a top-level JSON string field ("key":"value"); enough for this API.
std::string JsonString(const std::string& json, const char* key)
{
    const std::string k = std::string("\"") + key + "\"";
    size_t p = json.find(k);
    if (p == std::string::npos)
        return "";
    p = json.find(':', p + k.size());
    if (p == std::string::npos)
        return "";
    p = json.find('"', p);
    if (p == std::string::npos)
        return "";
    const size_t e = json.find('"', p + 1);
    return e == std::string::npos ? "" : json.substr(p + 1, e - p - 1);
}

bool HttpGetLatest(int timeoutMs, std::string* body, std::string* err)
{
    const std::string ver = CurrentVersion(); // one string: begin() and end() must come from the same object
    const std::wstring ua = L"OneEighty-UpdateCheck/" + std::wstring(ver.begin(), ver.end());
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
    HINTERNET c = WinHttpConnect(s, kHost, INTERNET_DEFAULT_HTTPS_PORT, 0);
    HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", kPath, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         WINHTTP_FLAG_SECURE)
                    : nullptr;
    if (r && WinHttpSendRequest(r, L"Accept: application/vnd.github+json\r\n", (DWORD)-1L, WINHTTP_NO_REQUEST_DATA,
                                0, 0, 0) &&
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
        ok = code == 200;
        if (!ok)
            *err = "HTTP " + std::to_string(code);
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

Status DoCheck(int timeoutMs)
{
    std::string body, err;
    Status st;
    if (HttpGetLatest(timeoutMs, &body, &err))
    {
        std::string tag = JsonString(body, "tag_name");
        if (!tag.empty() && (tag[0] == 'v' || tag[0] == 'V'))
            tag.erase(0, 1);
        if (tag.empty())
        {
            st.state = State::Failed;
            st.error = "no tag_name in response";
        }
        else
        {
            st.latest = tag;
            const std::string page = JsonString(body, "html_url");
            st.pageUrl = page.empty() ? kReleasesPage : page;
            st.state = CompareVersions(tag, CurrentVersion()) > 0 ? State::Available : State::UpToDate;
        }
    }
    else
    {
        st.state = State::Failed;
        st.error = err;
    }
    {
        std::lock_guard<std::mutex> lock(gMu);
        if (st.state == State::Failed)
        {
            // keep an earlier known result (e.g. from the cache); only report
            // "failed" when we have nothing better
            if (gStatus.latest.empty())
                gStatus.state = State::Failed;
            gStatus.error = st.error;
        }
        else
        {
            gStatus = st;
            gLastCheck = (long long)std::time(nullptr);
            SaveCacheLocked();
        }
    }
    Log("[update] check: %s latest=%s current=%s%s%s",
        st.state == State::Available ? "update available" : st.state == State::UpToDate ? "up to date" : "failed",
        st.latest.c_str(), CurrentVersion().c_str(), st.error.empty() ? "" : " error=", st.error.c_str());
    return st;
}

std::vector<int> Parts(const std::string& v)
{
    std::vector<int> out;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, '.'))
        out.push_back(std::atoi(item.c_str()));
    while (out.size() < 3)
        out.push_back(0);
    return out;
}

// Studio / IT opt-out for the automatic check (the "Check for updates" button
// always works): environment variable ONEEIGHTY_DISABLE_UPDATE_CHECK=1, or
// DWORD DisableUpdateCheck=1 under HKCU or HKLM \Software\thinkvp\OneEighty.
bool AutoCheckDisabled()
{
    char env[8] = {0};
    if (GetEnvironmentVariableA("ONEEIGHTY_DISABLE_UPDATE_CHECK", env, sizeof(env)) && env[0] && env[0] != '0')
        return true;
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

} // namespace

std::string CurrentVersion()
{
    return std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) + "." + std::to_string(BUG_VERSION);
}

int CompareVersions(const std::string& a, const std::string& b)
{
    const std::vector<int> x = Parts(a), y = Parts(b);
    for (size_t i = 0; i < std::max(x.size(), y.size()); ++i)
    {
        const int p = i < x.size() ? x[i] : 0, q = i < y.size() ? y[i] : 0;
        if (p != q)
            return p < q ? -1 : 1;
    }
    return 0;
}

Status Current()
{
    std::lock_guard<std::mutex> lock(gMu);
    LoadCacheLocked();
    return gStatus;
}

void MaybeCheckInBackground()
{
    static const bool disabled = [] {
        const bool d = AutoCheckDisabled();
        if (d)
            Log("[update] automatic check disabled (ONEEIGHTY_DISABLE_UPDATE_CHECK / DisableUpdateCheck)");
        return d;
    }();
    if (disabled)
        return;
    {
        std::lock_guard<std::mutex> lock(gMu);
        LoadCacheLocked();
        if ((long long)std::time(nullptr) - gLastCheck < kCheckIntervalSec)
            return; // checked recently (cache)
    }
    bool expected = false;
    if (!gBackgroundStarted.compare_exchange_strong(expected, true))
        return; // once per session
    {
        std::lock_guard<std::mutex> lock(gMu);
        if (gStatus.state == State::Unknown)
            gStatus.state = State::Checking;
    }
    std::thread([] { DoCheck(8000); }).detach();
}

Status CheckNow(int timeoutMs)
{
    {
        std::lock_guard<std::mutex> lock(gMu);
        LoadCacheLocked();
    }
    return DoCheck(timeoutMs);
}

} // namespace oe::update
