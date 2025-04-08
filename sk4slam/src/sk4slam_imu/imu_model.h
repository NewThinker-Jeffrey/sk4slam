#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/configurable.h"
#include "sk4slam_basic/serializable.h"

namespace sk4slam {

struct ImuData {
  double timestamp;    ///< Timestamp in seconds.
  Eigen::Vector3d wm;  ///< Measured angular ratio (rad/s)
  Eigen::Vector3d am;  ///< Measured acceleration (m/s^2)

  static ImuData interpolate(
      const ImuData& imu_1, const ImuData& imu_2, double timestamp);

  static std::vector<ImuData> fillDataGaps(
      const std::vector<ImuData>& data, double max_gap = 0.011);

  template <typename ImuDataContainer /*vector or deque*/>
  static inline std::vector<ImuData> selectBetween(
      const ImuDataContainer& imu_data, double time0, double time1,
      bool warn = true);

  template <typename ImuDataContainer /*vector or deque*/>
  static inline std::vector<ImuData> selectRecent(
      const ImuDataContainer& imu_data, double recent_seconds);
};

struct ImuSigmas : public Serializable<ImuSigmas> {
  double ct_gyro_sigma =
      1e-3;  ///< The standard deviation (sigma) of continuous-time gyro
             ///< measurement noise, representing the Amplitude Spectral
             ///< Density (ASD = √PSD, where PSD is the Power Spectral Density),
             ///< in units of [(rad/s)/√Hz].

  double ct_accel_sigma =
      1e-2;  ///< The standard deviation (sigma) of continuous-time
             ///< accelerometer measurement noise, representing the
             ///< Amplitude Spectral Density (ASD = √PSD), in units of
             ///< [(m/s²)/√Hz].

  double ct_gyro_bias_sigma =
      1e-4;  ///< The standard deviation (sigma) of continuous-time gyro
             ///< bias random-walk noise, representing the ASD of the
             ///< random-walk process, in units of [(rad/s²)/√Hz], which is
             ///< equivalent to [(rad/s)/√s].

  double ct_accel_bias_sigma =
      1e-3;  ///< The standard deviation (sigma) of continuous-time
             ///< accelerometer bias random-walk noise, representing the ASD
             ///< of the random-walk process, in units of [(m/s³)/√Hz],
             ///< which is equivalent to [(m/s²)/√s].

  double gyro_scale_err_sigma =
      1e-2;  ///< The standard deviation (sigma) of gyro scale factor error
  double accel_scale_err_sigma =
      1e-2;  ///< The standard deviation (sigma) of accelerometer scale factor

  ImuSigmas() {}
  CONFIG_MEMBERS(ImuSigmas) {
    CONFIG_OPTIONAL_MEM(ct_gyro_sigma);
    CONFIG_OPTIONAL_MEM(ct_accel_sigma);
    CONFIG_OPTIONAL_MEM(ct_gyro_bias_sigma);
    CONFIG_OPTIONAL_MEM(ct_accel_bias_sigma);
    CONFIG_OPTIONAL_MEM(gyro_scale_err_sigma);
    CONFIG_OPTIONAL_MEM(accel_scale_err_sigma);
  }
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    SERIALIZE(ar, ct_gyro_sigma);
    SERIALIZE(ar, ct_accel_sigma);
    SERIALIZE(ar, ct_gyro_bias_sigma);
    SERIALIZE(ar, ct_accel_bias_sigma);
    SERIALIZE(ar, gyro_scale_err_sigma);
    SERIALIZE(ar, accel_scale_err_sigma);
  }
};

}  // namespace sk4slam

#include "sk4slam_imu/imu_data_inl.h"
