#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#elif defined(_MSC_VER)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif
