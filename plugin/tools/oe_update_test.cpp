// Tests UpdateCheck.cpp against the live GitHub release and checks the version comparison.
//   oe_update_test
#include "../UpdateCheck.h"

#include <cstdio>

int main()
{
    using namespace oe::update;
    int bad = 0;
    auto expect = [&](const char* a, const char* b, int want) {
        const int got = CompareVersions(a, b);
        const int sgn = (got > 0) - (got < 0);
        if (sgn != want)
        {
            std::printf("FAIL compare %s vs %s: %d (want %d)\n", a, b, sgn, want);
            ++bad;
        }
    };
    expect("0.6.0", "0.6.0", 0);
    expect("0.7.0", "0.6.0", 1);
    expect("0.6.1", "0.6.0", 1);
    expect("0.10.0", "0.9.3", 1);
    expect("1.0", "0.99.99", 1);
    expect("0.5.9", "0.6", -1);
    std::printf("version comparison: %s\n", bad ? "FAILED" : "ok");

    std::printf("this build: v%s\n", CurrentVersion().c_str());
    const Status s = CheckNow(8000);
    const char* st = s.state == State::Available ? "update available" : s.state == State::UpToDate ? "up to date"
                     : s.state == State::Failed ? "failed" : "unknown";
    std::printf("GitHub check: %s, latest v%s, page %s%s%s\n", st, s.latest.c_str(), s.pageUrl.c_str(),
                s.error.empty() ? "" : ", error: ", s.error.c_str());
    const Status c = Current();
    std::printf("cached status after check: latest v%s\n", c.latest.c_str());
    return bad;
}
