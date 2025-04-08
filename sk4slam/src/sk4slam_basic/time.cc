#include "sk4slam_basic/time.h"

#include <cstring>
#include <map>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"

namespace sk4slam {

namespace {

inline bool isNumberBetween_1_and_9(char c) {
  return (c >= '1' && c <= '9');
}

void formatNanoseconds(std::string* p_str, int nano) {
  char nano_str[10];
  sprintf(nano_str, "%09d", nano);  // NOLINT

  std::string format("$");
  std::string& str = *p_str;
  for (std::string::size_type pos(0); pos != std::string::npos;) {
    if ((pos = str.find(format, pos)) != std::string::npos) {
      if (pos == 0 || str.at(pos - 1) != '\\') {
        int decimal_len;
        if (pos + 1 < str.length() &&
            isNumberBetween_1_and_9(str.at(pos + 1))) {
          decimal_len = str.at(pos + 1) - '0';
          str.replace(
              pos, format.length() + 1, std::string(nano_str, decimal_len));
        } else {
          decimal_len = 9;
          str.replace(pos, format.length(), std::string(nano_str, decimal_len));
        }
        pos += decimal_len;
      } else {
        pos += format.length();
      }
    } else {
      break;
    }
  }
}
}  // namespace

Time::Time(int64_t timestamp_ns) : timestamp_ns_(timestamp_ns) {}

Time::Time(const std::chrono::system_clock::time_point& tp) {
  std::chrono::nanoseconds time_since_epoch = tp.time_since_epoch();
  timestamp_ns_ = time_since_epoch.count();
}

Time::Time(const struct tm& datetime, int32_t plus_nanos) {
  struct tm datetime_copy = datetime;

  // std::time_t = int64_t ?
  // And it's in seconds (not nanoseconds).
  std::time_t time = mktime(&datetime_copy);
  timestamp_ns_ = static_cast<int64_t>(time) * int_1e9;
  timestamp_ns_ += plus_nanos;
}

Time Time::now() {
  // Though unspecified by the standard, every implementation of
  // std::chrono::system_clock::now() is tracking Unix Time (which
  // is a very close approximation to UTC)
  //     https://stackoverflow.com/a/39274464
  //
  // Unix time is a date and time representation widely used in computing.
  // It measures time by the number of seconds that have elapsed since 00:00:00
  // UTC on 1 January 1970, the Unix epoch,
  //    **without adjustments made because of leap seconds**.
  // In modern computing, values are sometimes stored with higher granularity,
  // such as microseconds or nanoseconds.
  //     https://en.wikipedia.org/wiki/Unix_time

  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
      now = std::chrono::system_clock::now();

  std::chrono::nanoseconds time_since_epoch = now.time_since_epoch();

  return Time(time_since_epoch.count());
}

std::string Time::dateStr(const char* format, bool use_utc_time) const {
  auto date_time = datetime(use_utc_time);
  std::string date_str;
  if (format) {
    int len = std::strlen(format) + 100;
    char* date = new char[len + 1];
    strftime(date, len, format, &date_time);
    date_str = std::string(date);
    delete[] date;
    formatNanoseconds(&date_str, timestamp_ns_ % int_1e9);
  }
  return date_str;
}

std::string TimeCounter::report(
    const std::string& prefix, bool print_to_stdout) const {
  Oss oss(Precision(1));
  oss << prefix;

  std::map<Time, std::string> inverse_map;
  for (const auto& tt : tagged_times_) {
    if (inverse_map.count(tt.second)) {
      LOGW(
          "TimeCounter::print():  Skip tag '%s' since its timestamp is "
          "identical with some other tags",
          tt.first.c_str());
      continue;
    }
    inverse_map[tt.second] = tt.first;
  }

  // Print tags in oder (also print the duration between adjacent tags)
  auto prev_time = start_;
  auto last_time = start_;
  oss << "start";
  for (auto it = inverse_map.begin(); it != inverse_map.end(); it++) {
    oss << " -- " << (it->first - prev_time).millis() << " -- " << it->second;
    prev_time = it->first;
    last_time = it->first;
  }
  oss << ", total elapsed " << (last_time - start_).millis();
  oss << ", starting time is " << start_.dateStr()
      << " (starting ts = " << start_.timestamp_ns() << ")";

  std::string ret = oss.str();
  if (print_to_stdout) {
    std::cout << ret << std::endl;
  }
  return ret;
}

bool TimeCounter::checkThresholds(const Thresholds& thresholds) const {
  for (const auto& threshold : thresholds) {
    const auto& tag_pair = threshold.first;
    const auto& duration_thr = threshold.second;
    const auto& tag_start = tag_pair.first;
    const auto& tag_end = tag_pair.second;
    if (tagged_times_.count(tag_end) == 0) {
      // LOGW("TimeCounter::checkThresholds(): Unknown tag '%s'",
      // tag_end.c_str());
      continue;
    }
    if (tagged_times_.count(tag_start) == 0) {
      // LOGW("TimeCounter::checkThresholds(): Unknown tag '%s'",
      // tag_start.c_str());
      continue;
    }
    const auto& start_time = tagged_times_.at(tag_start);
    const auto& end_time = tagged_times_.at(tag_end);
    if (end_time - start_time > duration_thr) {
      LOGW(
          "TimeCounter::checkThresholds(): duration from '%s' to '%s' "
          "exceeded its threshold ((%.1f > %.1f ms)!",
          tag_start.c_str(), tag_end.c_str(), (end_time - start_time).millis(),
          duration_thr.millis());
      return false;
    }
  }

  return true;
}

}  // namespace sk4slam
