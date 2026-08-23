#pragma once

#include <Eigen/Core>
#include <map>
#include <memory>
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
        "none"};  ///< The motion filter method to use.  Values: "none",
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

    double acc_range = 36.0;  ///< Accelerometer range in m/s^2. Negative value
                              ///< means infinite range.
    double gyro_range = 999.0;  ///< Gyroscope range in deg/s. Negative value
                                ///< means infinite range.
    double ex_range_acc_sigma =
        -1.;  ///< Sigma (discret-time) for accel measurements exceeding range.
              ///< Negative value means no correction.
    double ex_range_gyro_sigma =
        -1.;  ///< Sigma (discret-time) for gyro measurements exceeding range.
              ///< Negative value means no correction.
    double sampling_rate =
        -1.;  ///< Sampling rate in Hz. Only used to compute the discretized
              ///< noise sigmas. If negative, the sampling rate will be
              ///< inferred from the data.

    bool correct_sigmas = true;  ///< Whether to correct the noise sigmas based
                                 ///< on the motion filter results.
    bool integrate_filtered_data =
        true;  ///< Whether to integrate the filtered data.

    double large_data_gap_threshold{
        0.011};  ///< IMU intervals above this threshold are reported as a
                 ///< large gap and filled before integration, in seconds.

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
      CONFIG_OPTIONAL_MEM(ex_range_acc_sigma);
      CONFIG_OPTIONAL_MEM(ex_range_gyro_sigma);
      CONFIG_OPTIONAL_MEM(sampling_rate);
      CONFIG_OPTIONAL_MEM(correct_sigmas);
      CONFIG_OPTIONAL_MEM(integrate_filtered_data);
      CONFIG_OPTIONAL_MEM(large_data_gap_threshold);
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
    /// Largest interval between adjacent original IMU samples in this segment.
    double largest_data_gap{0.0};
    /// @}

    /// @name Integration result
    /// @{
    std::shared_ptr<ImuIntegration> pre_integration;
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
        const Rot3d& start_rot = Rot3d::Identity(),
        const Vector3d& delta_gyro_bias = Vector3d::Zero(),
        const Vector3d& delta_acc_bias = Vector3d::Zero()) const;

    bool predictRotation(
        Rot3d* end_rot, const Rot3d& start_rot = Rot3d::Identity(),
        const Vector3d& delta_gyro_bias = Vector3d::Zero(),
        const Vector3d& delta_acc_bias = Vector3d::Zero()) const {
      return predictRotation(
          end_time, end_rot, start_rot, delta_gyro_bias, delta_acc_bias);
    }

    std::map<double, Rot3d> predictRotations(
        const std::vector<double> times,
        const Rot3d& start_rot = Rot3d::Identity(),
        const Vector3d& delta_gyro_bias = Vector3d::Zero(),
        const Vector3d& delta_acc_bias = Vector3d::Zero()) const;

    bool predictState(
        double time, Pose3d* predicted_pose,
        Eigen::Vector3d* predicted_vel = nullptr,
        const Pose3d& start_pose = Pose3d::Identity(),
        const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
        bool apply_gravity = true,
        const Vector3d& delta_gyro_bias = Vector3d::Zero(),
        const Vector3d& delta_acc_bias = Vector3d::Zero()) const;

    bool predictState(
        Pose3d* end_pose, Eigen::Vector3d* end_vel = nullptr,
        const Pose3d& start_pose = Pose3d::Identity(),
        const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
        bool apply_gravity = true,
        const Vector3d& delta_gyro_bias = Vector3d::Zero(),
        const Vector3d& delta_acc_bias = Vector3d::Zero()) const {
      return predictState(
          end_time, end_pose, end_vel, start_pose, start_vel, apply_gravity,
          delta_gyro_bias, delta_acc_bias);
    }

    std::map<double, ImuIntegration::State> predictStates(
        const std::vector<double> times,
        const Pose3d& start_pose = Pose3d::Identity(),
        const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
        bool apply_gravity = true, bool trasform_velocity = true,
        const Vector3d& delta_gyro_bias = Vector3d::Zero(),
        const Vector3d& delta_acc_bias = Vector3d::Zero()) const;

    /// @}

   private:
    friend class ImuHandler;
    /// Data actually fed to pre_integration, including samples inserted to
    /// fill large gaps. Kept for subsequent analyses without exposing an
    /// implementation-specific propagation sequence.
    std::vector<ImuData> propagation_data_;
  };

  struct SpecialMotionResult {
    bool is_zero_rotation{false};
    bool is_const_velocity{false};
  };

 public:
  static ImuIntegration::Options defaultIntegrationOptions();

  /// @brief Constructs an ImuHandler with given IMU sigmas and
  /// options.
  /// @param sigmas IMU noise characteristics (e.g., accelerometer noise sigma).
  /// @param options Vibration detection configuration options.
  ImuHandler(
      const UniqueId& imu_uid, const Options& options = Options(),
      const ImuSigmas& sigmas = ImuSigmas())
      : imu_uid_(imu_uid), options_(options), sigmas_(sigmas) {}
  virtual ~ImuHandler() = default;

  bool isImuDataReady(
      const ImuDataBuffer& imu_data_buf, double required_imu_time) const;

  /// @brief Processes a segment of IMU data and returns a processed Segment
  /// object.
  /// @param imu_data_buf Buffer of IMU data to process.
  /// @param start_time Start time of the segment.
  /// @param end_time End time of the segment.
  /// @param gyro_bias Gyroscope bias.
  /// @param accel_bias Accelerometer bias.
  /// @param integration_options Options used to construct the segment's IMU
  /// pre-integration.
  std::shared_ptr<const Segment> processNewSegment(
      const ImuDataBuffer& imu_data_buf, double start_time, double end_time,
      const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias,
      const ImuIntegration::Options& integration_options =
          defaultIntegrationOptions()) const;

  /// Detects zero-rotation and constant-velocity motion from an already
  /// processed segment. Detection requires cached intermediate results, bias
  /// Jacobians, process-noise covariances, and a 6x6 bias covariance.
  /// Constant-velocity detection additionally requires full-state integration.
  SpecialMotionResult detectSpecialMotion(
      const Segment& segment, const Vector3d& gravity_in_start_frame,
      const Eigen::MatrixXd& bias_cov_6x6) const;

 public:
  static double estimateSamplingRate(const std::vector<ImuData>& data);

 protected:
  /// Creates the integration implementation used by processNewSegment().
  /// Derived handlers may override this factory to provide a specialized
  /// ImuIntegration subclass.
  virtual std::shared_ptr<ImuIntegration> createImuIntegration(
      const ImuIntegration::Options& options, const ImuSigmas& sigmas,
      const Vector3d& gyro_bias, const Vector3d& accel_bias) const;

  void runMotionFilter(Segment* segment, double sampling_rate = -1) const;

  void runPolynomialMotionFilter(
      Segment* segment, double sampling_rate = -1) const;

  void runLowPassMotionFilter(
      Segment* segment, double sampling_rate = -1) const {
    throw std::runtime_error(
        "ImuHandler: LowPassMotionFilter has not beed implemented yet!");
  }

 private:
  UniqueId imu_uid_;
  Options options_;   ///< Configuration options.
  ImuSigmas sigmas_;  ///< IMU noise characteristics.
};

}  // namespace sk4slam
