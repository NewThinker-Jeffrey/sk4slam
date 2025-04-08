#pragma once

#include <cstdarg>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

int replace(
    std::string* p_str, const std::string& old_pattern,
    const std::string& new_pattern, int max_replace_n = -1,
    std::string::size_type start_pos = 0);

int replaceOnce(
    std::string* p_str, const std::string& old_pattern,
    const std::string& new_pattern, std::string::size_type start_pos = 0);

int replaceAll(
    std::string* p_str, const std::string& old_pattern,
    const std::string& new_pattern, std::string::size_type start_pos = 0);

bool contains(const std::string& str, const std::string& substr);

std::string strip(const std::string& str);

std::vector<std::string> split(const std::string& s);

std::vector<std::string> split(const std::string& s, char delimiter);

std::vector<std::string> split(
    const std::string& s, const std::string& delimiter);

std::string join(const std::vector<std::string>& strings, char delimiter);

std::string join(
    const std::vector<std::string>& strings, const std::string& delimiter);

// For formatting float numbers
struct Precision {
  enum class Type : uint32_t { DECIMAL_PLACES = 0, SIGNIFICANT_DIGITS = 1 };

  explicit Precision(int _digits = 4, Type _type = Type::DECIMAL_PLACES)
      : type(_type), digits(_digits) {}

  Precision(int _digits, uint32_t _type_0_or_1)
      : type(static_cast<Type>(_type_0_or_1)), digits(_digits) {}

  Type type;
  int digits;
};

// For formatting those objects that support "<<".
struct Oss : public std::ostringstream {
  explicit Oss(Precision default_precision = Precision());

  template <size_t BUF_SIZE = 8192>
  bool format(const char* format, ...) {
    char buf[BUF_SIZE];
    va_list ap;
    va_start(ap, format);
    int len = vsprintf(buf, format, ap);
    va_end(ap);
    if (len > 0) {
      (*this) << std::string(buf);
      return true;
    } else {
      return false;
    }
  }
};

inline std::string __toOneLineStr(const std::string& str) {
  std::string one_line = str;
  replace(&one_line, "\n", " ");
  return one_line;
}

template <typename T>
std::string toStr(const T& t, Precision precision = Precision()) {
  Oss oss;
  oss.precision(precision.digits);
  oss << t;
  return oss.str();
}

template <typename T>
std::string toOneLineStr(const T& t, Precision precision = Precision()) {
  return __toOneLineStr(toStr<T>(t, precision));
}

template <typename Container>
std::string __containerToStr(
    const Container& container, Precision precision = Precision()) {
  Oss oss;
  oss.precision(precision.digits);
  oss << "{";
  bool is_first = true;
  for (const auto& t : container) {
    if (!is_first) {
      oss << ", ";
    }
    oss << t;
    is_first = false;
  }
  oss << "}";
  return oss.str();
}

template <typename Map>
std::string __mapToStr(const Map& map, Precision precision = Precision()) {
  Oss oss;
  oss.precision(precision.digits);
  oss << "{";
  bool is_first = true;
  for (const auto& t : map) {
    if (!is_first) {
      oss << ", ";
    }
    oss << t.first << " : " << t.second;
    is_first = false;
  }
  oss << "}";
  return oss.str();
}

#define DEFINE_TO_STR_FOR_CONTAINER(Container)                            \
  template <typename T>                                                   \
  std::string toStr(                                                      \
      const Container<T>& container, Precision precision = Precision()) { \
    return __containerToStr(container, precision);                        \
  }                                                                       \
  template <typename T>                                                   \
  std::string toOneLineStr(                                               \
      const Container<T>& container, Precision precision = Precision()) { \
    return __toOneLineStr(__containerToStr(container, precision));        \
  }

#define DEFINE_TO_STR_FOR_MAP(Map)                                       \
  template <typename Tkey, typename Tvalue>                              \
  std::string toStr(                                                     \
      const Map<Tkey, Tvalue>& map, Precision precision = Precision()) { \
    return __mapToStr(map, precision);                                   \
  }                                                                      \
  template <typename Tkey, typename Tvalue>                              \
  std::string toOneLineStr(                                              \
      const Map<Tkey, Tvalue>& map, Precision precision = Precision()) { \
    return __toOneLineStr(__mapToStr(map, precision));                   \
  }

DEFINE_TO_STR_FOR_CONTAINER(std::vector)
DEFINE_TO_STR_FOR_CONTAINER(std::set)
DEFINE_TO_STR_FOR_CONTAINER(std::unordered_set)
DEFINE_TO_STR_FOR_MAP(std::map)
DEFINE_TO_STR_FOR_MAP(std::unordered_map)

// Example:
//     toStr(data, sqrt);  -> output: sqrt(data[i]) ...
template <typename Container, typename Op>
std::string toStr(
    const Container& container, const Op& op,
    Precision precision = Precision()) {
  Oss oss;
  oss.precision(precision.digits);
  oss << "{";
  bool is_first = true;
  for (const auto& t : container) {
    if (!is_first) {
      oss << ", ";
    }
    oss << op(t);
    is_first = false;
  }
  oss << "}";
  return oss.str();
}
template <typename Container, typename Op>
std::string toOneLineStr(
    const Container& container, const Op& op,
    Precision precision = Precision()) {
  return __toOneLineStr(toStr<Container, Op>(container, op, precision));
}

// Example:
//     toStr<int>(data);  -> output: int(data[i]) ...
// template <typename CastType, typename Container>
template <
    typename CastType, typename Container,
    ENABLE_IF(!(
        std::is_same_v<CastType, Container>))>  // to de-ambiguate with toStr<T>
                                                // and <Container, Op>
std::string toStr(
    const Container& container, Precision precision = Precision()) {
  Oss oss;
  oss.precision(precision.digits);
  oss << "{";
  bool is_first = true;
  for (const auto& t : container) {
    if (!is_first) {
      oss << ", ";
    }
    oss << static_cast<CastType>(t);
    is_first = false;
  }
  oss << "}";
  return oss.str();
}
// template <typename CastType, typename Container>
template <
    typename CastType, typename Container,
    ENABLE_IF(!(std::is_same_v<
                CastType, Container>))>  // to de-ambiguate with toOneLineStr<T>
                                         // and <Container, Op>
std::string toOneLineStr(
    const Container& container, Precision precision = Precision()) {
  return __toOneLineStr(toStr<CastType, Container>(container, precision));
}

template <size_t BUF_SIZE = 8192>
std::string formatStr(const char* format, ...) {
  std::string ret;
  {
    char buf[BUF_SIZE];
    va_list ap;
    va_start(ap, format);
    int len = vsprintf(buf, format, ap);
    va_end(ap);
    if (len > 0) {
      ret = std::string(buf);
    }
  }
  return ret;
}
#define formatStr8K formatStr
#define formatStr16K formatStr<16384>
#define formatStr32K formatStr<32768>
#define formatStr64K formatStr<65536>
#define formatStr128K formatStr<131072>
#define formatStr256K formatStr<262144>
#define formatStr512K formatStr<524288>
#define formatStr1M formatStr<1048576>

}  // namespace sk4slam
