#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_basic/configurable.h"
#include "sk4slam_basic/unique_id.h"
#include "sk4slam_imu/imu_data_buffer.h"
#include "sk4slam_imu/imu_integration.h"
#include "sk4slam_imu/imu_model.h"

namespace sk4slam {

class ImuHandler {
 public:
  /// @brief Configuration options for ImuHandler.
  struct Options {
    std::string motion_filter_method{
        "polynomial"};  ///< The motion filter method to use.  Values:
                        ///< "polynomial", "lowpass" ...

    int polynomial_order = 1;
    ///< The order of the polynomial used to fit the acceleration data. Only
    ///< valid if motion_filter_method is "polynomial". Default is 1 (linear
    ///< fit). Supports higher orders (up to 3) for improved fitting on larger
    ///< data windows.

    double lowpass_cutoff_freq =
        5.0;  ///< The cutoff frequency for the lowpass filter used in motion
              ///< filtering. Only valid if motion_filter_method is "lowpass".
              ///< Default is 5.0 Hz.

    double required_future_data_time =
        0.0;  ///< The minimum amount of future data time required for motion
              ///< filtering. Future data is used to improve the accuracy of
              ///< motion filtering. Default is 0.0 seconds.

    double vibration_rmse_thr = 3.0;
    ///< Threshold for the Root Mean Square Error (RMSE) of normalized
    ///< acceleration data in vibration detection. The errors are normalized
    ///< by the accelerometer noise sigma before computing the RMSE. Unitless
    ///< (normalized).

    double vibration_max_err_thr = 10.0;
    ///< Maximum normalized error threshold for acceleration data in vibration
    ///< detection. The unit is also normalized to 1, as the errors are divided
    ///< by the sensor noise sigma.

    double zero_rotation_chi2_thr{
        7.814727903251179};  ///< Chi2 threshold for zero rotation motion
                             ///< detection. The default is Chi2(0.95, 3).
    double const_velocity_chi2_thr{
        7.814727903251179};  ///< Chi2 threshold for const velocity motion
                             ///< detection. The default is Chi2(0.95, 3).

    double acc_range = 36.0;    ///< Accelerometer range in m/s^2.
    double gyro_range = 999.0;  ///< Gyroscope range in deg/s.

    bool auto_clear_old_cache = true;  ///< Automatically clear old cache when
                                       ///< new data segment is added.
    double cache_duration = 2.0;       ///< Duration of the cache.

    Options() {}

    CONFIG_MEMBERS(Options) {
      CONFIG_OPTIONAL_MEM(motion_filter_method);
      CONFIG_OPTIONAL_MEM(polynomial_order);
      CONFIG_OPTIONAL_MEM(lowpass_cutoff_freq);
      CONFIG_OPTIONAL_MEM(vibration_rmse_thr);
      CONFIG_OPTIONAL_MEM(vibration_max_err_thr);
      CONFIG_OPTIONAL_MEM(zero_rotation_chi2_thr);
      CONFIG_OPTIONAL_MEM(const_velocity_chi2_thr);
      CONFIG_OPTIONAL_MEM(acc_range);
      CONFIG_OPTIONAL_MEM(gyro_range);
      CONFIG_OPTIONAL_MEM(auto_clear_old_cache);
      CONFIG_OPTIONAL_MEM(cache_duration);
    }
  };

  // using VibrationStatus = ImuVibrationStatus;
  enum class VibrationStatus {
    kUnknown,           ///< Unknown vibration status.
    kNoVibration,       ///< No vibration detected.
    kVibrationDetected  ///< Vibration detected.
  };

  /// @brief Represents a segment of processed IMU data.
  struct Segment {
    /// @name Input data
    /// @{
    double start_time{-1.0};
    double end_time{-1.0};
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d accel_bias;
    std::vector<ImuData> data;
    /// @}

    /// @name Motion filter results
    /// @{
    std::vector<ImuData> filtered_data;
    ImuSigmas corrected_sigmas;
    VibrationStatus vibration_status{VibrationStatus::kUnknown};
    bool gyro_exceeds_range{false};
    bool accel_exceeds_range{false};
    /// @}

    /// @name Integration and motion analysis results
    /// @{
    std::shared_ptr<ImuIntegration> pre_integration;
    bool is_zero_rotation{false};
    bool is_const_velocity{false};
    /// @}

    /// @brief  Default constructor
    Segment() {}

    bool isAccReliable() const {
      return vibration_status == VibrationStatus::kNoVibration &&
             !accel_exceeds_range;
    }

    bool isGyroReliable() const {
      return !gyro_exceeds_range;
    }

    /// @name Functions for states prediction
    /// @{
    bool predictRotation(
        double time, Rot3d* predicted_rot,
        const Rot3d& start_rot = Rot3d::Identity()) const;

    bool predictRotation(
        Rot3d* end_rot, const Rot3d& start_rot = Rot3d::Identity()) const {
      return predictRotation(end_time, end_rot, start_rot);
    }

    std::unordered_map<double, Rot3d> predictRotations(
        const std::vector<double> times,
        const Rot3d& start_rot = Rot3d::Identity()) const;

    bool predictState(
        double time, Pose3d* predicted_pose,
        Eigen::Vector3d* predicted_vel = nullptr,
        const Pose3d& start_pose = Pose3d::Identity(),
        const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
        bool apply_gravity = true) const;

    bool predictState(
        Pose3d* end_pose, Eigen::Vector3d* end_vel = nullptr,
        const Pose3d& start_pose = Pose3d::Identity(),
        const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
        bool apply_gravity = true) const {
      return predictState(
          end_time, end_pose, end_vel, start_pose, start_vel, apply_gravity);
    }

    std::unordered_map<double, ImuIntegration::State> predictStates(
        const std::vector<double> times,
        const Pose3d& start_pose = Pose3d::Identity(),
        const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
        bool apply_gravity = true, bool trasform_velocity = true) const;

    /// @}
  };

 public:
  /// @brief Constructs an ImuHandler with given IMU sigmas and
  /// options.
  /// @param sigmas IMU noise characteristics (e.g., accelerometer noise sigma).
  /// @param options Vibration detection configuration options.
  ImuHandler(
      const UniqueId& imu_uid, const Options& options = Options(),
      const ImuSigmas& sigmas = ImuSigmas())
      : imu_uid_(imu_uid), options_(options), sigmas_(sigmas) {}

  bool isImuDataReady(
      const ImuDataBuffer& imu_data_buf, double required_imu_time) const;

  /// @brief Processes a segment of IMU data and returns a processed Segment
  /// object.
  /// @param imu_data_buf Buffer of IMU data to process.
  /// @param start_time Start time of the segment.
  /// @param end_time End time of the segment.
  /// @param gyro_bias Gyroscope bias.
  /// @param accel_bias Accelerometer bias.
  /// @param gravity_in_start_frame Gravity direction in the start frame.
  /// @param bias_cov_6x6 Covariance matrix of the IMU biases. If empty, the
  /// flags `is_const_velocity` and `is_zero_rotation` in the returned Segment
  /// will be set to false.
  std::shared_ptr<const Segment> processNewSegment(
      const ImuDataBuffer& imu_data_buf, double start_time, double end_time,
      const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias,
      const Vector3d& gravity_in_start_frame,
      const Eigen::MatrixXd& bias_cov_6x6 = Eigen::MatrixXd()) const;

  /// @brief Returns a cached Segment object if it exists.
  std::shared_ptr<const Segment> getCachedSegment(
      double start_time, double end_time) const {
    auto it = segments_cache_.find(start_time);
    if (it != segments_cache_.end() && it->second->end_time == end_time) {
      return it->second;
    }
    return nullptr;
  }

  /// @brief Clears cached Segment objects that start before the given time.
  void clearOldSegmentsCache(double time) const {
    while (!segments_cache_.empty() && segments_cache_.begin()->first < time) {
      segments_cache_.erase(segments_cache_.begin());
    }
  }

 public:
  static double estimateSamplingRate(const std::vector<ImuData>& data);

 protected:
  void runMotionFilter(Segment* segment, double sampling_rate = -1) const;

  void runPolynomialMotionFilter(
      Segment* segment, double sampling_rate = -1) const;

  void runLowPassMotionFilter(
      Segment* segment, double sampling_rate = -1) const {
    throw std::runtime_error(
        "ImuHandler: LowPassMotionFilter has not beed implemented yet!");
  }

 private:
  void autoClear() const {
    if (options_.auto_clear_old_cache && !segments_cache_.empty()) {
      clearOldSegmentsCache(
          segments_cache_.rbegin()->first - options_.cache_duration);
    }
  }

  void cacheNew(const std::shared_ptr<Segment>& segment) const {
    segments_cache_[segment->start_time] = segment;
    autoClear();
  }

 private:
  UniqueId imu_uid_;
  Options options_;   ///< Configuration options.
  ImuSigmas sigmas_;  ///< IMU noise characteristics.

  mutable std::map<double, std::shared_ptr<Segment>> segments_cache_;
};

}  // namespace sk4slam
