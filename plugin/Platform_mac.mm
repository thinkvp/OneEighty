// macOS side of Platform.h (Objective-C++, ARC).
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#include "Platform.h"

#include <algorithm>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

namespace oe::platform
{
namespace
{
std::filesystem::path LibraryDir(NSSearchPathDirectory which)
{
    @autoreleasepool
    {
        NSArray<NSString*>* dirs = NSSearchPathForDirectoriesInDomains(which, NSUserDomainMask, YES);
        if (dirs.count == 0)
            return {};
        return std::filesystem::path(dirs.firstObject.fileSystemRepresentation);
    }
}

NSString* NS(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()] ?: @""; }

// Runs block on the main thread (AppKit UI), waiting for it.
void OnMain(void (^block)(void))
{
    if ([NSThread isMainThread])
        block();
    else
        dispatch_sync(dispatch_get_main_queue(), block);
}
} // namespace

std::filesystem::path LogPath()
{
    std::filesystem::path dir = LibraryDir(NSLibraryDirectory);
    if (dir.empty())
        return {};
    dir /= "Logs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir / "OneEighty.log";
}

std::filesystem::path DataDir()
{
    std::filesystem::path dir = LibraryDir(NSApplicationSupportDirectory);
    if (dir.empty())
        return {};
    dir /= "OneEighty";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return ec ? std::filesystem::path() : dir;
}

unsigned long ThreadId()
{
    uint64_t id = 0;
    pthread_threadid_np(nullptr, &id);
    return (unsigned long)id;
}

void LocalTime(int* h, int* m, int* s, int* ms)
{
    timeval tv;
    gettimeofday(&tv, nullptr);
    tm lt;
    localtime_r(&tv.tv_sec, &lt);
    *h = lt.tm_hour;
    *m = lt.tm_min;
    *s = lt.tm_sec;
    *ms = (int)(tv.tv_usec / 1000);
}

bool HttpsGet(const std::string& host, const std::string& path, const std::string& userAgent,
              const std::string& accept, int timeoutMs, std::string* body, int* status, std::string* err)
{
    *status = 0;
    @autoreleasepool
    {
        NSURL* url = [NSURL URLWithString:NS("https://" + host + path)];
        if (!url)
        {
            *err = "bad URL";
            return false;
        }
        NSMutableURLRequest* req = [NSMutableURLRequest requestWithURL:url
                                                           cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                       timeoutInterval:timeoutMs / 1000.0];
        [req setValue:NS(userAgent) forHTTPHeaderField:@"User-Agent"];
        [req setValue:NS(accept) forHTTPHeaderField:@"Accept"];

        // Ephemeral: no cookies or cache written to disk.
        NSURLSession* session =
            [NSURLSession sessionWithConfiguration:[NSURLSessionConfiguration ephemeralSessionConfiguration]];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block NSData* data = nil;
        __block NSInteger code = 0;
        __block NSString* failure = nil;
        NSURLSessionDataTask* task =
            [session dataTaskWithRequest:req
                       completionHandler:^(NSData* d, NSURLResponse* response, NSError* error) {
                         if (error)
                             failure = error.localizedDescription;
                         else
                         {
                             data = d;
                             if ([response isKindOfClass:[NSHTTPURLResponse class]])
                                 code = ((NSHTTPURLResponse*)response).statusCode;
                         }
                         dispatch_semaphore_signal(done);
                       }];
        [task resume];
        // The request has its own timeout; this is a safety net.
        const long timedOut =
            dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeoutMs + 2000) * 1000000));
        [session finishTasksAndInvalidate];
        if (timedOut)
        {
            [task cancel];
            *err = "timed out";
            return false;
        }
        if (failure)
        {
            *err = failure.UTF8String ?: "request failed";
            return false;
        }
        *status = (int)code;
        if (data)
            body->assign((const char*)data.bytes, std::min<size_t>(data.length, 1 << 20));
        return true;
    }
}

bool PolicyDisablesUpdateCheck()
{
    // Reads user defaults and managed (MDM / configuration profile) preferences:
    //   defaults write com.thinkvp.OneEighty DisableUpdateCheck -bool true
    Boolean valid = false;
    const Boolean v =
        CFPreferencesGetAppBooleanValue(CFSTR("DisableUpdateCheck"), CFSTR("com.thinkvp.OneEighty"), &valid);
    return valid && v;
}

bool AskYesNo(const std::string& title, const std::string& message)
{
    __block bool yes = false;
    NSString* t = NS(title);
    NSString* m = NS(message);
    OnMain(^{
      NSAlert* alert = [[NSAlert alloc] init];
      alert.messageText = t;
      alert.informativeText = m;
      alert.alertStyle = NSAlertStyleInformational;
      [alert addButtonWithTitle:@"Yes"];
      [alert addButtonWithTitle:@"No"];
      yes = [alert runModal] == NSAlertFirstButtonReturn;
    });
    return yes;
}

void ShowMessage(const std::string& title, const std::string& message, bool warning)
{
    NSString* t = NS(title);
    NSString* m = NS(message);
    OnMain(^{
      NSAlert* alert = [[NSAlert alloc] init];
      alert.messageText = t;
      alert.informativeText = m;
      alert.alertStyle = warning ? NSAlertStyleWarning : NSAlertStyleInformational;
      [alert addButtonWithTitle:@"OK"];
      [alert runModal];
    });
}

void OpenUrl(const std::string& url)
{
    NSString* u = NS(url);
    OnMain(^{
      if (NSURL* nsurl = [NSURL URLWithString:u])
          [[NSWorkspace sharedWorkspace] openURL:nsurl];
    });
}

} // namespace oe::platform
