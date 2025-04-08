#include "sk4slam_basic/unique_id.h"

#include <cstring>  // std::memcmp

#if defined(__GNUC__) || defined(__clang__)
#define BSWAP64(x) __builtin_bswap64(x)
#define IS_BIG_ENDIAN (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#elif defined(_MSC_VER)
#include <cstdlib>  // _byteswap_uint64
#define BSWAP64(x) _byteswap_uint64(x)
#define IS_BIG_ENDIAN 0  // Windows is little endian
#else
namespace {
static inline uint64_t bswap64(uint64_t x) {
  return ((x & 0xFF00000000000000ULL) >> 56) |
         ((x & 0x00FF000000000000ULL) >> 40) |
         ((x & 0x0000FF0000000000ULL) >> 24) |
         ((x & 0x000000FF00000000ULL) >> 8) |
         ((x & 0x00000000FF000000ULL) << 8) |
         ((x & 0x0000000000FF0000ULL) << 24) |
         ((x & 0x000000000000FF00ULL) << 40) |
         ((x & 0x00000000000000FFULL) << 56);
}
static inline bool isBigEndian() {
  const uint32_t x = 1;
  return *((const uint8_t*)&x) == 0;
}
}  // namespace
#define BSWAP64(x) bswap64(x)
#define IS_BIG_ENDIAN isBigEndian()
#endif

namespace sk4slam {

UniqueId::UniqueId(bool generate) {
  if (!generate) {
    val_.u64[0] = 0;
    val_.u64[1] = 0;
  } else {
    val_.u64[0] = random64();
    val_.u64[1] = random64();
  }
}

const UniqueId UniqueId::null = UniqueId(false);

void UniqueId::setNull() {
  val_.u64[0] = 0;
  val_.u64[1] = 0;
}

bool UniqueId::isNull() const {
  return val_.u64[0] == 0 && val_.u64[1] == 0;
}

const std::string UniqueId::hexString() const {
  static const char* kHexConversion = "0123456789abcdef";
  char buffer[2 * sizeof(val_) + 1];  // 1 for the \0 character
  buffer[2 * sizeof(val_)] = '\0';
  for (size_t i = 0; i < sizeof(val_); ++i) {
    buffer[2 * i + 1] = kHexConversion[val_.c[i] & 0xf];
    buffer[2 * i] = kHexConversion[val_.c[i] >> 4];
  }
  return std::string(buffer);
}

bool UniqueId::fromHexString(const std::string& hexString) {
  // hexadecimal string takes 2 characters per byte
  if (hexString.size() != 2 * sizeof(val_)) {
    return false;
  }
  for (size_t i = 0; i < sizeof(val_); ++i) {
    val_.c[i] = static_cast<unsigned char>(
        stoul(std::string(hexString, 2 * i, 2), 0, 16));
  }
  return true;
}

void UniqueId::setVal(const char bytes[16]) {
  std::memcpy(&val_, bytes, sizeof(val_));
}

void UniqueId::setVal(uint64_t high64, uint64_t low64) {
  if (IS_BIG_ENDIAN) {
    val_.u64[0] = BSWAP64(high64);
    val_.u64[1] = BSWAP64(low64);
  } else {
    val_.u64[0] = low64;
    val_.u64[1] = high64;
  }
}

uint64_t UniqueId::high64() const {
  if (IS_BIG_ENDIAN) {
    return BSWAP64(val_.u64[0]);
  } else {
    return val_.u64[1];
  }
}

uint64_t UniqueId::low64() const {
  if (IS_BIG_ENDIAN) {
    return BSWAP64(val_.u64[1]);
  } else {
    return val_.u64[0];
  }
}

bool UniqueId::operator<(const UniqueId& other) const {
  return std::memcmp(&val_, &other.val_, sizeof(val_)) < 0;
}

bool UniqueId::operator>(const UniqueId& other) const {
  return other < *this;
}

bool UniqueId::operator==(const UniqueId& other) const {
  return val_.u64[0] == other.val_.u64[0] && val_.u64[1] == other.val_.u64[1];
}

bool UniqueId::operator!=(const UniqueId& other) const {
  return !(*this == other);
}

std::size_t UniqueId::hash() const {
  // Note the hash value differs between big and little endian platforms.
  return std::hash<std::uint64_t>()(val_.u64[0]) ^
         std::hash<std::uint64_t>()(val_.u64[1]);
}
}  // namespace sk4slam
