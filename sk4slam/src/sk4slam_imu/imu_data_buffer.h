#pragma once

#include <Eigen/Core>
#include <map>

#include "sk4slam_basic/unique_id.h"
#include "sk4slam_cpp/deque.h"  // for Deque
#include "sk4slam_cpp/mutex.h"
#include "sk4slam_imu/imu_model.h"

namespace sk4slam {

/// Thread-safe access to imu_data. This class can manage data from one or
/// multiple IMUs.
class ImuDataBuffer {
 public:
  ImuDataBuffer() = default;

  /// @brief Add imu data from a specific imu
  void addImuData(const UniqueId& imu_uid, const ImuData& imu_data)
      EXCLUDES(imu_data_mutex_);

  /// @brief Remove imu data older than timestamp from a specific imu
  void removeOldImuData(const UniqueId& imu_uid, double timestamp /*exclusive*/)
      EXCLUDES(imu_data_mutex_);

  /// @brief Remove imu data older than timestamp from all imus.
  void removeOldImuData(double timestamp /*exclusive*/)
      EXCLUDES(imu_data_mutex_);

  /// @brief Get imu data from a specific imu between start_timestamp and
  /// end_timestamp.
  std::vector<ImuData> getImuDataBetween(
      const UniqueId& imu_uid, double start_timestamp,
      double end_timestamp) const EXCLUDES(imu_data_mutex_);

  /// @brief Get imu data from a specific imu in the last seconds.
  std::vector<ImuData> getRecentImuData(
      const UniqueId& imu_uid, double seconds) const EXCLUDES(imu_data_mutex_);

  /// @brief Add imu data from the default imu.
  /// @note This function can be only used when the internal imu_data_ is empty
  /// or only contains one imu. Otherwise, it will throw an exception.
  void addImuData(const ImuData& imu_data) EXCLUDES(imu_data_mutex_);

  /// @brief Get imu data from the default imu between start_timestamp and
  /// end_timestamp.
  /// @note This function can be only used when the internal imu_data_ is empty
  /// or only contains one imu. Otherwise, it will throw an exception.
  std::vector<ImuData> getImuDataBetween(
      double start_timestamp, double end_timestamp) const
      EXCLUDES(imu_data_mutex_);

  /// @brief Get imu data from the default imu in the last seconds.
  std::vector<ImuData> getRecentImuData(double seconds) const
      EXCLUDES(imu_data_mutex_);

  double getLatestImuTime(const UniqueId& imu_uid) const
      EXCLUDES(imu_data_mutex_);

  double getLatestImuTime() const EXCLUDES(imu_data_mutex_);

 protected:
#define USE_SK4SLAM_DEQUE_FOR_IMU_DATA_SEQ_IN_VIO_IMU_DATA_BUFFER \
  1  // Set to 0 if sk4slam::Deque is not stable.
#if USE_SK4SLAM_DEQUE_FOR_IMU_DATA_SEQ_IN_VIO_IMU_DATA_BUFFER
  using ImuDataSeq = Deque<ImuData, 1024>;
#else
  using ImuDataSeq =
      std::deque<ImuData>;  // TODO(jeffrey): use our Deque for more control
                            // over memory usage and better performance
#endif
  /// @brief Add imu data from a specific imu
  void addImuData(ImuDataSeq& imu_data_seq, const ImuData& imu_data);

  /// @brief Remove imu data older than timestamp from a specific imu
  void removeOldImuData(
      ImuDataSeq& imu_data_seq, double timestamp /*exclusive*/)
      REQUIRES(imu_data_mutex_);

  /// @brief Get imu data from a specific imu between start_timestamp and
  /// end_timestamp.
  std::vector<ImuData> getImuDataBetween(
      const ImuDataSeq& imu_data_seq, double start_timestamp,
      double end_timestamp) const REQUIRES_SHARED(imu_data_mutex_);

  std::vector<ImuData> getRecentImuData(
      const ImuDataSeq& imu_data_seq, double seconds) const
      REQUIRES_SHARED(imu_data_mutex_);

 protected:  // Thread-safe access to imu_data_
  std::map<UniqueId /*Imu UniqueId*/, ImuDataSeq> imu_data_;
  mutable Mutex imu_data_mutex_;
};

}  // namespace sk4slam
