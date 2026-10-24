#pragma once

// Log lines for the server and the libraries it runs (serve/, runtime/): slog(level, printf-style), one line each.
// Levels (most to least severe): error, warn, info, debug. The log-level setting filters at the source: a line
// less severe than it isn't formatted or printed. The default is info; debug adds the per-request detail rows (MTP,
// PLE, cache saves) and the background writer's lines. serve/log.hpp has the request-block helpers on top.
//
// Under systemd (stderr connected to the journal: $JOURNAL_STREAM names
// stderr's device:inode - a terminal started from a desktop session inherits the variable, so it alone isn't enough)
// each line starts with a syslog priority prefix "<N>", which journald strips and stores as the entry's priority -
// journalctl then colours errors red and warnings yellow, and `journalctl -p info` hides the debug detail lines.
// Run by hand, lines are plain text.

#include <sys/stat.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "common/check.hpp"

namespace strix {

enum class LogLevel { Error = 3, Warning = 4, Info = 6, Debug = 7 };  // the values are the syslog priorities

// The least severe level printed (log-level; default info). Set once at startup, read by every thread.
inline std::atomic<int> &log_threshold() {
    static std::atomic<int> t{(int)LogLevel::Info};
    return t;
}
inline bool log_enabled(LogLevel level) { return (int)level <= log_threshold().load(std::memory_order_relaxed); }
inline void set_log_level(LogLevel level) {
    STRIX_CHECK(level == LogLevel::Error || level == LogLevel::Warning || level == LogLevel::Info || level == LogLevel::Debug,
                "set_log_level: level ", (int)level, " is not one of 3 / 4 / 6 / 7");
    log_threshold().store((int)level, std::memory_order_relaxed);
}

// "error" / "warn" / "info" / "debug" (log-level's values) and back.
inline LogLevel parse_log_level(const std::string &name) {
    if (name == "error") return LogLevel::Error;
    if (name == "warn") return LogLevel::Warning;
    if (name == "info") return LogLevel::Info;
    if (name == "debug") return LogLevel::Debug;
    STRIX_FAIL("parse_log_level: log-level '", name, "', expected error, warn, info or debug");
}
inline const char *log_level_name(LogLevel level) {
    switch (level) {
    case LogLevel::Error: return "error";
    case LogLevel::Warning: return "warn";
    case LogLevel::Info: return "info";
    case LogLevel::Debug: return "debug";
    }
    STRIX_FAIL("log_level_name: level ", (int)level, " is not one of 3 / 4 / 6 / 7");
}

inline bool log_to_journal() {
    static const bool on = [] {
        const char *js = std::getenv("JOURNAL_STREAM");
        unsigned long long dev = 0, ino = 0;
        struct stat st {};
        return js && std::sscanf(js, "%llu:%llu", &dev, &ino) == 2 && ::fstat(2, &st) == 0 &&
               (unsigned long long)st.st_dev == dev && (unsigned long long)st.st_ino == ino;
    }();
    return on;
}

// The calling thread's serving slot (-1: none). With several slots the engine's workers set theirs, and every line
// they log starts "[slot N] ", so two requests' blocks that interleave in the journal still read apart. One slot
// (and every other thread): no tag - the lines are exactly as before.
inline int &log_slot_tag() {
    thread_local int tag = -1;
    return tag;
}

// printf-style; one line (a trailing newline is added).
#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
inline void slog(LogLevel level, const char *fmt, ...) {
    if (!log_enabled(level)) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    char tag[16] = "";
    if (log_slot_tag() >= 0) std::snprintf(tag, sizeof tag, "[slot %d] ", log_slot_tag());
    if (log_to_journal()) std::fprintf(stderr, "<%d>%s%s\n", (int)level, tag, buf);
    else std::fprintf(stderr, "%s%s\n", tag, buf);
}

}  // namespace strix
