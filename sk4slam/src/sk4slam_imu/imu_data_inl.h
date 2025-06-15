#pragma once

#include "sk4slam_cpp/binary_search.h"

namespace sk4slam {
namespace imu_internal {
template <typename ImuDataContainer>
std::vector<ImuData> binarySearchBetween(
    const ImuDataContainer& imu_data, double time0, double time1, bool warn) {
  std::vector<ImuData> prop_data;
  if (imu_data.empty()) {
    if (warn) {
      LOGW("ImuData::selectBetween(): No IMU measurements!!!");
    }
    return prop_data;
  }

  if (time0 > time1) {
    if (warn) {
      LOGW(
          "ImuData::selectBetween(): bad arguments! time0(%f) > "
          "time1(%f)!",
          time0, time1);
    }
    return prop_data;
  }

  if (time1 < imu_data.front().timestamp || time0 > imu_data.back().timestamp) {
    if (warn) {
      LOGW(
          "ImuData::selectBetween(): No IMU measurements!!! query (%f to %f) "
          "totally out of range (%f to %f)!",
          time0, time1, imu_data.front().timestamp, imu_data.back().timestamp);
    }
    return prop_data;
  }

  int idx1;
  if (time0 < imu_data.front().timestamp) {
    if (warn) {
      LOGW(
          "ImuData::selectBetween(): query (%f to %f) partially out of range "
          "(%f to %f)!",
          time0, time1, imu_data.front().timestamp, imu_data.back().timestamp);
    }
    return {};  // Return empty?
    // TODO(jeffrey): Whether to allow extrapolation at the begining?

    // // If time0 is even older than the first imu measurement,
    // // then we insert at the begining of prop_data a copy of
    // // the first measurement with timestamp changed to time0.
    // ImuData data = imu_data.front();
    // data.timestamp = time0;
    // prop_data.push_back(data);
    // idx1 = 0;
  } else {
    auto search_res = binarySearchLowerBound(
        imu_data, time0, std::less<double>(),
        [](const ImuData& imu_data) { return imu_data.timestamp; });
    ASSERT(search_res);
    auto idx0 = *search_res;
    ASSERT(idx0 > 0 || imu_data.at(idx0).timestamp == time0);
    ASSERT(idx0 < imu_data.size());
    if (imu_data.at(idx0).timestamp > time0) {
      ASSERT(idx0 > 0);
      // interpolate for time0 with (imu_data.at(idx0-1), imu_data.at(idx0))
      prop_data.push_back(ImuData::interpolate(
          imu_data.at(idx0 - 1), imu_data.at(idx0), time0));
    } else {
      ASSERT(imu_data.at(idx0).timestamp == time0);
      // no interpolation needed
    }
    idx1 = idx0;
  }

  while (idx1 < imu_data.size() && imu_data.at(idx1).timestamp <= time1) {
    if (!prop_data.empty() &&
        imu_data.at(idx1).timestamp <= prop_data.back().timestamp) {
      if (warn) {
        LOGW(
            "ImuData::selectBetween(): Zero or Negative dt(%f)!!",
            prop_data.back().timestamp - imu_data.at(idx1).timestamp);
      }
      continue;
    }

    prop_data.push_back(imu_data.at(idx1));
    ++idx1;
  }

  if (imu_data.back().timestamp < time1) {
    if (warn) {
      LOGW(
          "ImuData::selectBetween(): query (%f to %f) partially out of range "
          "(%f to %f)!",
          time0, time1, imu_data.front().timestamp, imu_data.back().timestamp);
    }
    return {};  // Return empty?
    // TODO(jeffrey): Whether to allow extrapolation at the end?

    // // If time1 is even newer than the last imu measurement,
    // // then we insert at the end of prop_data a copy of
    // // the last measurement with timestamp changed to time1.
    // ImuData data = imu_data.back();
    // data.timestamp = time1;
    // prop_data.push_back(data);
  }

  ASSERT(!prop_data.empty());
  if (prop_data.back().timestamp < time1) {
    ASSERT(idx1 < imu_data.size());
    // interpolate for time1 with (imu_data.at(idx1-1), imu_data.at(idx1))
    prop_data.push_back(
        ImuData::interpolate(imu_data.at(idx1 - 1), imu_data.at(idx1), time1));
  }

  ASSERT(prop_data.size() >= 2);
  ASSERT(time0 == prop_data.front().timestamp);
  ASSERT(time1 == prop_data.back().timestamp);

  // Success
  return prop_data;
}

}  // namespace imu_internal

template <typename ImuDataContainer /*vector or deque*/>
inline std::vector<ImuData> ImuData::selectBetween(
    const ImuDataContainer& imu_data, double time0, double time1, bool warn) {
  return imu_internal::binarySearchBetween(imu_data, time0, time1, warn);
}

template <typename ImuDataContainer /*vector or deque*/>
inline std::vector<ImuData> ImuData::selectRecent(
    const ImuDataContainer& imu_data, double recent_seconds) {
  if (imu_data.empty()) {
    return {};
  }
  double back_time = imu_data.back().timestamp;
  double front_time = back_time - recent_seconds;
  if (imu_data.front().timestamp >= front_time) {
    // If the front time is already within the range, then return all data
    return {imu_data.begin(), imu_data.end()};
  }

  auto search_res = binarySearchLowerBound(
      imu_data, front_time, std::less<double>(),
      [](const ImuData& imu_data) { return imu_data.timestamp; });
  ASSERT(search_res);
  auto idx0 = *search_res;
  if (imu_data.at(idx0).timestamp < front_time) {
    ++idx0;
  }
  ASSERT(imu_data.at(idx0).timestamp >= front_time);
  return {imu_data.begin() + idx0, imu_data.end()};
}

}  // namespace sk4slam
