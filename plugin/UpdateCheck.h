#pragma once
// Checks GitHub for a newer OneEighty release.
//
// One anonymous HTTPS GET of the public "latest release" API
// (https://api.github.com/repos/thinkvp/OneEighty/releases/latest); nothing
// about the user or machine is sent beyond a generic User-Agent. Automatic
// checks run at most once a day on a background thread and are cached in
// %LOCALAPPDATA%\OneEighty\update.txt; editors can switch them off in the
// effect (Advanced > Check for updates automatically).

#include <string>

namespace oe::update
{

enum class State
{
    Unknown,   // never checked (or cache empty)
    Checking,  // background check running
    UpToDate,
    Available, // newer release on GitHub
    Failed     // network / parse error
};

struct Status
{
    State state = State::Unknown;
    std::string latest;  // e.g. "0.7.0" (no leading v)
    std::string pageUrl; // release page to send people to
    std::string error;
};

// This build's version, e.g. "0.6.0".
std::string CurrentVersion();

// Last known result (from this session or the on-disk cache).
Status Current();

// Starts a background check if none ran in the last 24 h (per the cache).
// Cheap to call often; never blocks.
void MaybeCheckInBackground();

// Checks now, blocking up to ~timeoutMs. Updates Current() and the cache.
Status CheckNow(int timeoutMs = 6000);

// <0 if a < b, 0 if equal, >0 if a > b ("0.10.0" > "0.9.3").
int CompareVersions(const std::string& a, const std::string& b);

} // namespace oe::update
