#pragma once

#include "sk4slam_cpp/binary_search.h"

namespace sk4slam {
namespace imu_internal {

// TODO(jeffrey): use binary search (i.e. use selectImuDataBetween_New())

template <typename ImuDataContainer>
std::vector<ImuData> selectImuDataBetween_Old(
    const ImuDataContainer& imu_data, double time0, double time1, bool warn) {
  // Our vector imu readings
  std::vector<ImuData> prop_data;

  // Ensure we have some measurements in the first place!
  if (imu_data.empty()) {
    if (warn)
      LOGW(YELLOW
           "ImuData::selectBetween(): No IMU measurements. "
           "IMU-CAMERA are likely messed up!!!\n" RESET);
    return prop_data;
  }

  // Loop through and find all the needed measurements to propagate with
  // Note we split measurements based on the given state time, and the update
  // timestamp
  for (size_t i = 0; i < imu_data.size() - 1; i++) {
    // START OF THE INTEGRATION PERIOD
    // If the next timestamp is greater then our current state time
    // And the current is not greater then it yet...
    // Then we should "split" our current IMU measurement
    if (imu_data.at(i + 1).timestamp > time0 &&
        imu_data.at(i).timestamp < time0) {
      ImuData data =
          ImuData::interpolate(imu_data.at(i), imu_data.at(i + 1), time0);
      prop_data.push_back(data);
      // LOGD("propagation #%d = CASE 1 = %.3f => %.3f\n",
      //      (int)i, data.timestamp - prop_data.at(0).timestamp,
      //      time0 - prop_data.at(0).timestamp);
      continue;
    }

    // MIDDLE OF INTEGRATION PERIOD
    // If our imu measurement is right in the middle of our propagation period
    // Then we should just append the whole measurement time to our propagation
    // vector
    if (imu_data.at(i).timestamp >= time0 &&
        imu_data.at(i + 1).timestamp <= time1) {
      prop_data.push_back(imu_data.at(i));
      // LOGD("propagation #%d = CASE 2 = %.3f\n",
      //      (int)i, imu_data.at(i).timestamp - prop_data.at(0).timestamp);
      continue;
    }

    // END OF THE INTEGRATION PERIOD
    // If the current timestamp is greater then our update time
    // We should just "split" the NEXT IMU measurement to the update time,
    // NOTE: we add the current time, and then the time at the end of the
    // interval (so we can get a dt) NOTE: we also break out of this loop, as
    // this is the last IMU measurement we need!
    if (imu_data.at(i + 1).timestamp > time1) {
      // If we have a very low frequency IMU then, we could have only recorded
      // the first integration (i.e. case 1) and nothing else In this case, both
      // the current IMU measurement and the next is greater than the desired
      // intepolation, thus we should just cut the current at the desired time
      // Else, we have hit CASE2 and this IMU measurement is not past the
      // desired propagation time, thus add the whole IMU reading
      if (imu_data.at(i).timestamp > time1 && i == 0) {
        // This case can happen if we don't have any imu data that has occured
        // before the startup time This means that either we have dropped IMU
        // data, or we have not gotten enough. In this case we can't propgate
        // forward in time, so there is not that much we can do.
        break;
      } else if (imu_data.at(i).timestamp > time1) {
        ImuData data =
            ImuData::interpolate(imu_data.at(i - 1), imu_data.at(i), time1);
        prop_data.push_back(data);
        // LOGD("propagation #%d = CASE 3.1 = %.3f => %.3f\n",
        // (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
      } else {
        prop_data.push_back(imu_data.at(i));
        // LOGD("propagation #%d = CASE 3.2 = %.3f => %.3f\n",
        // (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
      }
      // If the added IMU message doesn't end exactly at the camera time
      // Then we need to add another one that is right at the ending time
      if (prop_data.at(prop_data.size() - 1).timestamp != time1) {
        ImuData data =
            ImuData::interpolate(imu_data.at(i), imu_data.at(i + 1), time1);
        prop_data.push_back(data);
        // LOGD("propagation #%d = CASE 3.3 = %.3f => %.3f\n",
        // (int)i,data.timestamp-prop_data.at(0).timestamp,data.timestamp-time0);
      }
      break;
    }
  }

  // Check that we have at least one measurement to propagate with
  if (prop_data.empty()) {
    if (warn)
      LOGW(
          YELLOW
          "ImuData::selectBetween(): No IMU measurements to propagate "
          "with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET,
          (int)prop_data.size());
    return prop_data;
  }

  // If we did not reach the whole integration period (i.e., the last inertial
  // measurement we have is smaller then the time we want to reach) Then we
  // should just "stretch" the last measurement to be the whole period (case 3
  // in the above loop)
  //
  // if(time1-imu_data.at(imu_data.size()-1).timestamp > 1e-3) {
  //    LOGD(YELLOW
  //         "ImuData::selectBetween(): Missing inertial measurements to "
  //         "propagate with (%.6f sec missing). IMU-CAMERA are likely "
  //         "messed up!!!\n" RESET,
  //         (time1-imu_data.at(imu_data.size()-1).timestamp));
  //    return prop_data;
  // }

  // Loop through and ensure we do not have an zero dt values
  // This would cause the noise covariance to be Infinity
  for (size_t i = 0; i < prop_data.size() - 1; i++) {
    if (std::abs(prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp) <
        1e-12) {
      if (warn)
        LOGW(
            YELLOW
            "ImuData::selectBetween(): Zero DT between "
            "IMU reading %d and %d, removing it!\n" RESET,
            (int)i, (int)(i + 1));
      prop_data.erase(prop_data.begin() + i);
      i--;
    }
  }

  // Check that we have at least one measurement to propagate with
  if (prop_data.size() < 2) {
    if (warn)
      LOGW(
          YELLOW
          "ImuData::selectBetween(): No IMU measurements to propagate "
          "with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET,
          (int)prop_data.size());
    return prop_data;
  }

  // Success :D
  return prop_data;
}

template <typename ImuDataContainer>
std::vector<ImuData> selectImuDataBetween_New(
    const ImuDataContainer& imu_data, double time0, double time1, bool warn) {
  // Our vector imu readings
  std::vector<ImuData> prop_data;
  // Ensure we have some measurements in the first place!
  if (imu_data.empty()) {
    if (warn) {
      LOGW(YELLOW "ImuData::selectBetween(): No IMU measurements!!!" RESET);
    }
    return prop_data;
  }

  if (time0 > time1) {
    if (warn) {
      LOGW(
          YELLOW
          "ImuData::selectBetween(): bad arguments! time0(%f) > "
          "time1(%f)!" RESET,
          time0, time1);
    }
    return prop_data;
  }

  if (time1 < imu_data.front().timestamp || time0 > imu_data.back().timestamp) {
    if (warn) {
      LOGW(
          YELLOW
          "ImuData::selectBetween(): No IMU measurements!!! query (%f to %f) "
          "totally out of range (%f to %f)!" RESET,
          time0, time1, imu_data.front().timestamp, imu_data.back().timestamp);
    }
    return prop_data;
  }

  int idx1;
  if (time0 < imu_data.front().timestamp) {
    if (warn) {
      LOGW(
          YELLOW
          "ImuData::selectBetween(): query (%f to %f) partially out of range "
          "(%f "
          "to %f)!" RESET,
          time0, time1, imu_data.front().timestamp, imu_data.back().timestamp);
    }
    // return prop_data;  // Return empty?
    // TODO(jeffrey): Whether to allow extrapolation at the begining?

    // If time0 is even older than the first imu measurement,
    // then we insert at the begining of prop_data a copy of
    // the first measurement with timestamp changed to time0.
    ImuData data = imu_data.front();
    data.timestamp = time0;
    prop_data.push_back(data);
    idx1 = 0;
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
            YELLOW "ImuData::selectBetween(): Zero or Negative dt(%f)!!" RESET,
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
          YELLOW
          "ImuData::selectBetween(): query (%f to %f) partially out of range "
          "(%f "
          "to %f)!" RESET,
          time0, time1, imu_data.front().timestamp, imu_data.back().timestamp);
    }
    // return prop_data;  // Return empty?
    // TODO(jeffrey): Whether to allow extrapolation at the end?

    // If time1 is even newer than the last imu measurement,
    // then we insert at the end of prop_data a copy of
    // the last measurement with timestamp changed to time1.
    ImuData data = imu_data.back();
    data.timestamp = time1;
    prop_data.push_back(data);
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
  return imu_internal::selectImuDataBetween_Old(
      // return imu_internal::selectImuDataBetween_New(
      imu_data, time0, time1, warn);
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
