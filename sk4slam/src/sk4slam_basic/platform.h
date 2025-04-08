#pragma once

#if defined(__clang__)
#define COMPILER "Clang"
#elif defined(__GNUC__)
#define COMPILER "GCC"
#elif defined(_MSC_VER)
#define COMPILER "MSVC"
#else
#define COMPILER "Unknown"
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define CPU_ARCH "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
#define CPU_ARCH "x86"
#elif defined(__arm__) || defined(_M_ARM)
#if defined(__aarch64__) || defined(_M_ARM64)
#define CPU_ARCH "ARM64"
#else
#define CPU_ARCH "ARM"
#endif
#else
#define CPU_ARCH "Unknown"
#endif

#if defined(_WIN32)
#define OS "Windows"
#elif defined(__linux__)
#include <cstring>
#include <sys/utsname.h>
inline bool isAndroid() {
  struct utsname info;
  if (uname(&info) == 0) {
    return std::strstr(info.sysname, "Android") != nullptr;
  }
  return false;
}
#define OS isAndroid() ? "Android" : "Linux"
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IOS
#define OS "iOS"
#else
#define OS "MacOS"
#endif
#else
#define OS "Unknown"
#endif

#if __cplusplus >= 202002L
#define CPP_STD "C++20"
#elif __cplusplus >= 201703L
#define CPP_STD "C++17"
#elif __cplusplus >= 201402L
#define CPP_STD "C++14"
#elif __cplusplus >= 201103L
#define CPP_STD "C++11"
#else
#define CPP_STD "Unknown"
#endif
