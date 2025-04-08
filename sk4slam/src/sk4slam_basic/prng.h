#pragma once

#include <random>

namespace sk4slam {

using PRNG = std::mt19937;
// using PRNG = std::mt19937_64;

/// @brief   Get a Pseudo-Random Number Generator.
/// @return  A reference to the PRNG.
/// @note    This function is thread-safe and but return different PRNGs
///          in different threads.
PRNG& getPRNG();

uint64_t random64();

uint32_t random32();

}  // namespace sk4slam
