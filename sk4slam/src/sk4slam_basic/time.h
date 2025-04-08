#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sk4slam {

static constexpr int32_t int_1e9 = 1000000000;
static constexpr int32_t int_1e6 = 1000000;
static constexpr int32_t int_1e3 = 1000;

class Duration {
 public:
  explicit Duration(int64_t nanos = 0) : nanos_(nanos) {}

  static Duration Nanos(double nanos) {
    return Duration(nanos);
  }

  static Duration Micros(double micros) {
    return Duration(micros * 1e3);
  }

  static Duration Millis(double millis) {
    return Duration(millis * 1e6);
  }

  static Duration Seconds(double seconds) {
    return Duration(seconds * 1e9);
  }

  inline double nanos() const {
    return static_cast<double>(nanos_);
  }

  inline double micros() const {
    return static_cast<double>(nanos_ * 1e-3);
  }

  inline double millis() const {
    return static_cast<double>(nanos_ * 1e-6);
  }

  inline double seconds() const {
    return static_cast<double>(nanos_ * 1e-9);
  }

 public:
  inline Duration operator+(const Duration& rhs) const {
    return Duration(nanos_ + rhs.nanos_);
  }

  inline Duration operator-(const Duration& rhs) const {
    return Duration(nanos_ - rhs.nanos_);
  }

  inline Duration operator*(double scalar) const {
    return Duration(static_cast<int64_t>(nanos_ * scalar));
  }

  inline Duration operator/(double scalar) const {
    return Duration(static_cast<int64_t>(nanos_ / scalar));
  }

  inline bool operator<(const Duration& rhs) const {
    return nanos_ < rhs.nanos_;
  }

  inline bool operator>(const Duration& rhs) const {
    return nanos_ > rhs.nanos_;
  }

  inline bool operator==(const Duration& rhs) const {
    return nanos_ == rhs.nanos_;
  }

  inline bool operator!=(const Duration& rhs) const {
    return nanos_ != rhs.nanos_;
  }

 protected:
  int64_t nanos_;
  friend class Time;
};

class Time {
 public:
  static Time now();

  explicit Time(int64_t timestamp_ns = 0);

  explicit Time(const std::chrono::system_clock::time_point& tp);

  explicit Time(const struct tm& t, int32_t plus_nanos = 0);

  inline int64_t timestamp_ns() const {
    return timestamp_ns_;
  }

  inline double nanosSinceEpoch() const {
    return timestamp_ns_;
  }

  inline double microsSinceEpoch() const {
    return timestamp_ns_ * 1e-6;
  }

  inline double millisSinceEpoch() const {
    return timestamp_ns_ * 1e-3;
  }

  inline double secondsSinceEpoch() const {
    return timestamp_ns_ * 1e-9;
  }

  std::string dateStr(
      const char* format = "%Z %z %w %Y-%m-%d %H:%M:%S $3",
      // more format specifiers: https://cplusplus.com/reference/ctime/strftime/
      bool use_utc_time = false) const;  // false for local time.

 public:
  // convert to standard time types.

  inline std::chrono::system_clock::time_point time_point() const {
    return std::chrono::system_clock::time_point(
        std::chrono::nanoseconds(timestamp_ns_));
  }

  // std::time_t = int64_t ?
  // And it's in seconds (not nanoseconds).
  inline std::time_t time_t() const {
    // return std::chrono::system_clock::to_time_t(time_point());
    return std::time_t(timestamp_ns_ / int_1e9);
  }

  inline struct tm datetime(bool use_utc_time = false) const {
    struct tm date_time;
    std::time_t t = time_t();
    if (use_utc_time) {
      gmtime_r(&t, &date_time);
    } else {
      localtime_r(&t, &date_time);
    }
    return date_time;
    /////////////////////////////////////////////////////////
    // Member    | Type  | Meaning                   | Range
    // --------- | ----- | ------------------------- | -----
    // tm_sec    | int   | seconds after the minute  | 0-60*
    // tm_min    | int   | minutes after the hour    | 0-59
    // tm_hour   | int   | hours since midnight      | 0-23
    // tm_mday   | int   | day of the month          | 1-31
    // tm_mon    | int   | months since January      | 0-11
    // tm_year   | int   | years since 1900          |
    // tm_wday   | int   | days since Sunday         | 0-6
    // tm_yday   | int   | days since January 1      | 0-365
    // tm_isdst  | int   | Daylight Saving Time flag |
    /////////////////////////////////////////////////////////
  }

 public:
  inline Duration operator-(const Time& other) const {
    return Duration(timestamp_ns_ - other.timestamp_ns_);
  }

  inline Time operator+(const Duration& duration) const {
    return Time(timestamp_ns_ + duration.nanos_);
  }

  inline bool operator<(const Time& other) const {
    return timestamp_ns_ < other.timestamp_ns_;
  }

  inline bool operator>(const Time& other) const {
    return timestamp_ns_ > other.timestamp_ns_;
  }

  inline bool operator==(const Time& other) const {
    return timestamp_ns_ == other.timestamp_ns_;
  }

  inline bool operator!=(const Time& other) const {
    return timestamp_ns_ != other.timestamp_ns_;
  }

  // For unordered key.
  inline std::size_t hash() const {
    return std::hash<int64_t>()(timestamp_ns_);
  }

 protected:
  // timestamp starts from: January 1, 1970 00:00:00 (GMT time, not local time).
  int64_t timestamp_ns_;

  // Note:
  //     GMT time ≈ UTC time.
  //     They differ just less than 1 sencond.
  //     By the original definitions the difference is that GMT (also officially
  //     known as Universal Time (UT), which may be confusing) is based on
  //     astronomical observations while UTC is based on atomic clocks. Later
  //     GMT has become to be used at least unofficially to refer to UTC, which
  //     blurs the distinction somewhat. See:
  //         https://stackoverflow.com/a/48960297
  //         https://www.timeanddate.com/time/gmt-utc-time.html
};

class TimeCounter {
 public:
  explicit TimeCounter(const std::string default_prefix = "")
      : default_prefix_(default_prefix) {
    reset();
  }

  inline void reset() {
    tagged_times_.clear();
    start_ = Time::now();
  }

  inline Duration elapsed(const std::string& since_tag = "") const {
    if (since_tag.empty()) {
      return Time::now() - start_;
    } else {
      return Time::now() - tagged_times_.at(since_tag);
    }
  }

  inline Duration between(
      const std::string& from_tag, const std::string& to_tag) const {
    if (from_tag.empty()) {
      return tagged_times_.at(to_tag) - start_;
    } else {
      return tagged_times_.at(to_tag) - tagged_times_.at(from_tag);
    }
  }

  inline void tag(const std::string& tag, bool reset = false) {
    auto now = Time::now();
    if (reset) {
      tagged_times_.clear();
      start_ = now;
    }
    if (!tag.empty()) {
      tagged_times_[tag] = now;
    }
  }

  std::string report(
      const std::string& prefix, bool print_to_stdout = false) const;

  std::string report(bool print_to_stdout = false) const {
    return report(default_prefix_, print_to_stdout);
  }

  using TagPair = std::pair<std::string, std::string>;
  using Threshold = std::pair<TagPair, Duration>;
  using Thresholds = std::vector<Threshold>;
  bool checkThresholds(const Thresholds& thresholds) const;

 protected:
  Time start_;
  std::unordered_map<std::string, Time> tagged_times_;
  std::string default_prefix_;
};

}  // namespace sk4slam

namespace std {
inline ostream& operator<<(ostream& out, const sk4slam::Time& time) {
  out << time.dateStr();
  return out;
}

inline ostream& operator<<(ostream& out, const sk4slam::Duration& duration) {
  out << duration.millis() << " ms";
  return out;
}

template <>
struct hash<sk4slam::Time> {
  typedef sk4slam::Time argument_type;
  typedef std::size_t value_type;
  value_type operator()(const argument_type& time) const {
    return time.hash();
  }
};
}  // namespace std
