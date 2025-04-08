#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#define RESET "\033[0m"
#define BLACK "\033[30m"                 // Black
#define RED "\033[31m"                   // Red
#define GREEN "\033[32m"                 // Green
#define YELLOW "\033[33m"                // Yellow
#define BLUE "\033[34m"                  // Blue
#define MAGENTA "\033[35m"               // Magenta
#define CYAN "\033[36m"                  // Cyan
#define WHITE "\033[37m"                 // White
#define REDPURPLE "\033[95m"             // Red Purple
#define BOLDBLACK "\033[1m\033[30m"      // Bold Black
#define BOLDRED "\033[1m\033[31m"        // Bold Red
#define BOLDGREEN "\033[1m\033[32m"      // Bold Green
#define BOLDYELLOW "\033[1m\033[33m"     // Bold Yellow
#define BOLDBLUE "\033[1m\033[34m"       // Bold Blue
#define BOLDMAGENTA "\033[1m\033[35m"    // Bold Magenta
#define BOLDCYAN "\033[1m\033[36m"       // Bold Cyan
#define BOLDWHITE "\033[1m\033[37m"      // Bold White
#define BOLDREDPURPLE "\033[1m\033[95m"  // Bold Red Purple

namespace sk4slam {

enum class Verbose : uint32_t {
  ALL = 0,
  DEBUG = 1,
  INFO = 2,
  WARNING = 3,
  ERROR = 4,
  SILENT = 5
};

// operator std::string(Verbose v);
// operator Verbose(const std::string& v_str);

class Logging {
 public:
  // Set the verbose level
  static void setVerbose(Verbose verbose);
  static void setVerbose(const std::string& verbose);

  static void enableLogLock();  // disabled by default.
  static void disableLogLock();

  static void enableForceFlush();  // disabled by default.
  static void disableForceFlush();

  // The print function
  static void log(
      const char location[], const char line[], const char* format, ...);

  static Verbose current_verbose_level;
};
}  // namespace sk4slam

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CHECK_LOGGING_LEVEL(level)              \
  (static_cast<int>(sk4slam::Verbose::level) >= \
   static_cast<int>(sk4slam::Logging::current_verbose_level))  // NOLINT

// Different Types of log

// LOGA() is mainly used in unit tests.
#define LOGA_ENABLED CHECK_LOGGING_LEVEL(ALL)
#define LOGA(fmt, ...)                                                     \
  if (LOGA_ENABLED) {                                                      \
    sk4slam::Logging::log(                                                 \
        __FILE__, TOSTRING(__LINE__), "A " fmt RESET "\n", ##__VA_ARGS__); \
  }  // NOLINT

#define LOGD_ENABLED CHECK_LOGGING_LEVEL(DEBUG)
#define LOGD(fmt, ...)                                                     \
  if (LOGD_ENABLED) {                                                      \
    sk4slam::Logging::log(                                                 \
        __FILE__, TOSTRING(__LINE__), "D " fmt RESET "\n", ##__VA_ARGS__); \
  }  // NOLINT

#define LOGI_ENABLED CHECK_LOGGING_LEVEL(INFO)
#define LOGI(fmt, ...)                                                     \
  if (LOGI_ENABLED) {                                                      \
    sk4slam::Logging::log(                                                 \
        __FILE__, TOSTRING(__LINE__), "I " fmt RESET "\n", ##__VA_ARGS__); \
  }  // NOLINT

#define LOGW_ENABLED CHECK_LOGGING_LEVEL(WARNING)
#define LOGW(fmt, ...)                                            \
  if (LOGW_ENABLED) {                                             \
    sk4slam::Logging::log(                                        \
        __FILE__, TOSTRING(__LINE__), "W " YELLOW fmt RESET "\n", \
        ##__VA_ARGS__);                                           \
  }  // NOLINT

#define LOGE_ENABLED CHECK_LOGGING_LEVEL(ERROR)
#define LOGE(fmt, ...)                                                         \
  if (LOGE_ENABLED) {                                                          \
    sk4slam::Logging::log(                                                     \
        __FILE__, TOSTRING(__LINE__), "E " RED fmt RESET "\n", ##__VA_ARGS__); \
  }  // NOLINT

// weak assert
#define ASSERT_FAILED_ACTION throw std::runtime_error("Assertion failed")

// // strong assert
// #define ASSERT_FAILED_ACTION std::exit(EXIT_FAILURE)

#define LOGE_ASSERT(condition, fmt, ...)                                      \
  do {                                                                        \
    if (!(condition)) {                                                       \
      sk4slam::Logging::log(                                                  \
          __FILE__, TOSTRING(__LINE__),                                       \
          "X " RED "ASSERT_FAILED: " TOSTRING(condition) ". " fmt RESET "\n", \
          ##__VA_ARGS__);                                                     \
      fflush(stdout);                                                         \
      ASSERT_FAILED_ACTION;                                                   \
    }                                                                         \
  } while (false)

#ifndef ASSERT
#define ASSERT(condition) LOGE_ASSERT(condition, "")
#endif  // ASSERT

// Note that there is a little of overhead when using
//     TRACE_EXCEPTION or LOGE_EXCEPTION
// since we need to instantiate a lambda object.
#define LOGE_EXCEPTION(expression, fmt, ...)                                \
  [&]() {                                                                   \
    try {                                                                   \
      return expression;                                                    \
    } catch (const std::exception& e) {                                     \
      sk4slam::Logging::log(                                                \
          __FILE__, TOSTRING(__LINE__),                                     \
          "X " RED "EXCEPTION: " TOSTRING(expression) " --- %s. " fmt RESET \
                                                      "\n",                 \
          e.what(), ##__VA_ARGS__);                                         \
      fflush(stdout);                                                       \
      throw;                                                                \
    }                                                                       \
  }()

#define TRACE_EXCEPTION(expression) LOGE_EXCEPTION(expression, "")
