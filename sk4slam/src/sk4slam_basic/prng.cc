
#include "sk4slam_basic/prng.h"

#include <memory>

#include "sk4slam_basic/likely.h"

namespace sk4slam {

namespace {
thread_local std::unique_ptr<PRNG> prng;
thread_local std::uniform_int_distribution<uint64_t> dist64;
thread_local std::uniform_int_distribution<uint32_t> dist32;
}  // namespace

PRNG& getPRNG() {
  if (UNLIKELY(prng == nullptr)) {
    std::random_device rd;
    prng = std::make_unique<PRNG>(rd());
    // prng = std::make_unique<PRNG>(0);
  }
  return *prng;
}

uint64_t random64() {
  return dist64(getPRNG());  // random low 64 bits
}

uint32_t random32() {
  return dist32(getPRNG());  // random low 32 bits
}

}  // namespace sk4slam
