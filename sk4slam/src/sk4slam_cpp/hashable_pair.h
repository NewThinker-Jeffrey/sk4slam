#pragma once

#include <map>

namespace sk4slam {
template <typename T1, typename T2>
struct hashable_pair : public std::pair<T1, T2> {
  /// @brief Construct a new hashable pair object
  using std::pair<T1, T2>::pair;

  size_t hash() const {
    return std::hash<T1>{}(this->first) ^ std::hash<T2>{}(this->second);
  }

  bool operator<(const hashable_pair& other) const {
    return this->first < other.first ||
           (this->first == other.first && this->second < other.second);
  }

  bool operator==(const hashable_pair& other) const {
    return this->first == other.first && this->second == other.second;
  }
};

}  // namespace sk4slam

namespace std {
template <typename T1, typename T2>
struct hash<sk4slam::hashable_pair<T1, T2>> {
  size_t operator()(const sk4slam::hashable_pair<T1, T2>& p) const {
    return p.hash();
  }
};
}  // namespace std
