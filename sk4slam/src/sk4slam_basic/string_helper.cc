#include "sk4slam_basic/string_helper.h"

#include <regex>  // split

namespace sk4slam {

Oss::Oss(Precision precision) {
  if (precision.type == Precision::Type::SIGNIFICANT_DIGITS) {
    this->precision(precision.digits);
  } else {  // precision.type == Precision::Type::DECIMAL_PLACES
    this->setf(std::ios::fixed);
    this->precision(precision.digits);
  }
}

int replace(
    std::string* p_str, const std::string& old_pattern,
    const std::string& new_pattern, int max_replace_n,
    std::string::size_type start_pos) {
  std::string& str = *p_str;
  int n_replaced = 0;
  for (std::string::size_type pos(start_pos);
       pos != std::string::npos &&
       (max_replace_n < 0 || n_replaced < max_replace_n);
       pos += new_pattern.length()) {
    if ((pos = str.find(old_pattern, pos)) != std::string::npos) {
      str.replace(pos, old_pattern.length(), new_pattern);
      n_replaced++;
    } else {
      break;
    }
  }
  return n_replaced;
}

int replaceOnce(
    std::string* p_str, const std::string& old_pattern,
    const std::string& new_pattern, std::string::size_type start_pos) {
  return replace(p_str, old_pattern, new_pattern, 1, start_pos);
}

int replaceAll(
    std::string* p_str, const std::string& old_pattern,
    const std::string& new_pattern, std::string::size_type start_pos) {
  return replace(p_str, old_pattern, new_pattern, -1, start_pos);
}

bool contains(const std::string& str, const std::string& substr) {
  return str.find(substr) != std::string::npos;
}

std::string strip(const std::string& str) {
  return std::regex_replace(str, std::regex("^\\s+|\\s+$"), "");
}  // namespace sk4slam

std::vector<std::string> split(const std::string& s) {
  // Match one or more whitespace characters, including spaces, tabs, and
  // newlines.
  std::regex re("\\s+");
  std::sregex_token_iterator it(s.begin(), s.end(), re, -1);
  std::vector<std::string> tokens{it, {}};
  return tokens;
}

std::vector<std::string> split(const std::string& s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> split(
    const std::string& s, const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0, end = 0;

  while ((end = s.find(delimiter, start)) != std::string::npos) {
    tokens.push_back(s.substr(start, end - start));
    start = end + delimiter.length();
  }

  tokens.push_back(s.substr(start));
  return tokens;
}

std::string join(const std::vector<std::string>& strings, char delimiter) {
  if (strings.empty()) {
    return "";
  }
  std::string result = strings[0];

  for (size_t i = 1; i < strings.size(); ++i) {
    result += delimiter + strings[i];
  }
  return result;
}

std::string join(
    const std::vector<std::string>& strings, const std::string& delimiter) {
  if (strings.empty()) {
    return "";
  }
  std::string result = strings[0];

  for (size_t i = 1; i < strings.size(); ++i) {
    result += delimiter + strings[i];
  }
  return result;
}

}  // namespace sk4slam
