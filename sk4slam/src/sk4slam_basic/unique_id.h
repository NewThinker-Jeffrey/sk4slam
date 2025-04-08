#pragma once

#include <string>

#include "sk4slam_basic/prng.h"

namespace sk4slam {

/// @class UniqueId
/// @brief A class representing a unique identifier, with utilities for
///        creation, comparison, and conversion to/from string formats.
class UniqueId {
 public:
  /// @brief A constant representing a null unique ID.
  static const UniqueId null;

  /// @brief Constructor to generate a new unique ID or create a null ID.
  /// @param generate If true, a new unique ID is generated; if false, a null ID
  /// is created (default is true).
  explicit UniqueId(bool generate = true);

  /// @brief Sets the unique ID to null.
  void setNull();

  /// @brief Checks if the unique ID is null.
  bool isNull() const;

  /// @brief Initializes the unique ID from a hexadecimal string.
  /// @param hex_string The input hexadecimal string.
  /// @return True if parsing succeeds, false otherwise.
  bool fromHexString(const std::string& hex_string);

  /// @brief Converts the unique ID to its hexadecimal string representation.
  /// @return A string representing the unique ID in hexadecimal format.
  const std::string hexString() const;

 public:  // Discouraged APIs (Not recommended due to dependency on underlying
          // implementation)
  /// @brief Sets the unique ID's value directly using two 64-bit integers.
  /// @param high64 The high 64 bits of the ID.
  /// @param low64 The low 64 bits of the ID.
  void setVal(uint64_t high64, uint64_t low64);

  /// @brief Sets the unique ID's value directly using a 16-byte array.
  /// @param bytes A pointer to a 16-byte array representing the ID.
  void setVal(const char bytes[16]);

  /// @brief Retrieves the high 64 bits of the unique ID.
  /// @return The high 64 bits of the ID as a uint64_t.
  uint64_t high64() const;

  /// @brief Retrieves the low 64 bits of the unique ID.
  /// @return The low 64 bits of the ID as a uint64_t.
  uint64_t low64() const;

 public:
  /// @brief Comparison operators for UniqueId.
  bool operator<(const UniqueId& other) const;
  bool operator>(const UniqueId& other) const;
  bool operator==(const UniqueId& other) const;
  bool operator!=(const UniqueId& other) const;

  /// @brief Computes the hash value for the unique ID.
  /// @note The hash value may differ between platforms with different
  /// endianness (e.g., big-endian vs. little-endian).
  std::size_t hash() const;

 protected:
  /// @union Val
  /// @brief Internal representation of the unique ID as either 16 bytes or two
  /// 64-bit integers.
  union Val {
    unsigned char c[16];  ///< 16-byte representation of the ID.
    uint64_t u64[2];      ///< Two 64-bit integers representation of the ID.
  };

 protected:
  Val val_;  ///< The stored unique ID value.
};

}  // namespace sk4slam

namespace std {

/// @brief Outputs the unique ID as a hexadecimal string to an output stream.
inline ostream& operator<<(ostream& out, const sk4slam::UniqueId& uid) {
  out << uid.hexString();
  return out;
}

/// @brief Specialization of std::hash for sk4slam::UniqueId.
template <>
struct hash<sk4slam::UniqueId> {
  typedef sk4slam::UniqueId argument_type;
  typedef std::size_t value_type;

  /// @brief Computes the hash value for a UniqueId.
  value_type operator()(const argument_type& uid) const {
    return uid.hash();
  }
};

}  // namespace std
