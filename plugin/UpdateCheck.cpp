#include "UpdateCheck.h"
#include "OneEightyParams.h"
#include "Platform.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
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

constexpr char kHost[] = "api.github.com";
constexpr char kPath[] = "/repos/thinkvp/OneEighty/releases/latest";
constexpr char kReleasesPage[] = "https://github.com/thinkvp/OneEighty/releases/latest";
constexpr long long kCheckIntervalSec = 24 * 3600;

std::mutex gMu;
Status gStatus;
bool gLoadedCache = false;
std::atomic<bool> gBackgroundStarted{false};
long long gLastCheck = 0; // unix seconds

// update.txt in the per-user data folder (Platform.h).
std::filesystem::path CachePath()
{
    const std::filesystem::path dir = platform::DataDir();
    return dir.empty() ? dir : dir / "update.txt";
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
    const std::filesystem::path path = CachePath();
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
    Log("[update] cache %s: %s (last check %lld, latest '%s')", path.u8string().c_str(), loaded ? "loaded" : "not found",
        gLastCheck, gStatus.latest.c_str());
}

void SaveCacheLocked()
{
    const std::filesystem::path path = CachePath();
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    f << gLastCheck << "\n" << gStatus.latest << "\n" << gStatus.pageUrl << "\n";
    f.close();
    Log("[update] cache saved to %s: %s", path.u8string().c_str(), f.fail() ? "FAILED" : "ok");
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
    int code = 0;
    if (!platform::HttpsGet(kHost, kPath, "OneEighty-UpdateCheck/" + CurrentVersion(), "application/vnd.github+json",
                            timeoutMs, body, &code, err))
        return false;
    if (code != 200)
    {
        *err = "HTTP " + std::to_string(code);
        return false;
    }
    return true;
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
// always works): environment variable ONEEIGHTY_DISABLE_UPDATE_CHECK=1, or the
// DisableUpdateCheck policy (registry on Windows, defaults on macOS; Platform.h).
bool AutoCheckDisabled()
{
#ifdef _WIN32
    char env[8] = {0};
    size_t len = 0;
    getenv_s(&len, env, sizeof(env), "ONEEIGHTY_DISABLE_UPDATE_CHECK");
#else
    const char* env = std::getenv("ONEEIGHTY_DISABLE_UPDATE_CHECK");
    if (!env)
        return platform::PolicyDisablesUpdateCheck();
#endif
    if (env[0] && env[0] != '0')
        return true;
    return platform::PolicyDisablesUpdateCheck();
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
