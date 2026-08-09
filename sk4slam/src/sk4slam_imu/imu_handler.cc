#include "sk4slam_imu/imu_handler.h"

#include <Eigen/Dense>

#include "sk4slam_basic/logging.h"

namespace sk4slam {

bool ImuHandler::isImuDataReady(
    const ImuDataBuffer& imu_data_buf, double required_imu_time) const {
  const UniqueId& imu_uid = imu_uid_;
  if (imu_data_buf.getLatestImuTime(imu_uid) <
      required_imu_time + options_.required_future_data_time) {
    return false;  // IMU data is not available yet
  } else {
    return true;  // IMU data is available
  }
}

double ImuHandler::estimateSamplingRate(const std::vector<ImuData>& data) {
  size_t n = data.size();
  ASSERT(n >= 2);

  if (n < 4) {
    LOGD(
        YELLOW
        "ImuHandler::estimateSamplingRate(): At least 4 data points are "
        "required to calculate sampling rate! while only %d points "
        "provided." RESET,
        n);
    return (n - 1) / (data.back().timestamp - data.front().timestamp);
    // return -1;
  }

  // Calculate sampling intervals (excluding the first and last points, so
  // that we can get n-3 intervals from n data points)
  std::vector<double> intervals;
  for (size_t i = 1; i < n - 2; ++i) {
    intervals.push_back(data[i + 1].timestamp - data[i].timestamp);
  }

  // Calculate the median of the intervals
  size_t n_intervals = intervals.size();
  double median_interval;
  if (n_intervals % 2 == 1) {
    std::nth_element(
        intervals.begin(), intervals.begin() + n_intervals / 2,
        intervals.end());
    median_interval = intervals[n_intervals / 2];
  } else {
    auto mid1_it = intervals.begin() + n_intervals / 2;
    std::nth_element(intervals.begin(), mid1_it, intervals.end());
    double mid1 = *mid1_it;
    double mid2 = *std::max_element(intervals.begin(), mid1_it);
    median_interval = (mid1 + mid2) / 2.0;
  }

  // Calculate the sampling rate
  return 1.0 / median_interval;
}

void ImuHandler::runMotionFilter(Segment* segment, double sampling_rate) const {
  // Get the sampling rate
  if (sampling_rate <= 0) {
    sampling_rate = estimateSamplingRate(segment->data);
    ASSERT(sampling_rate > 0);
  }

  // Check whether the measurements are within the specified ranges
  segment->gyro_exceeds_range = false;
  segment->accel_exceeds_range = false;
  const double w_range = options_.gyro_range * M_PI / 180.0;
  const double a_range = options_.acc_range;
  for (const auto& imu_data : segment->data) {
    const auto& am = imu_data.am;
    const auto& wm = imu_data.wm;
    if (w_range > 0 && !segment->gyro_exceeds_range) {
      segment->gyro_exceeds_range = std::abs(wm.x()) > w_range ||
                                    std::abs(wm.y()) > w_range ||
                                    std::abs(wm.z()) > w_range;
    }
    if (a_range > 0 && !segment->accel_exceeds_range) {
      segment->accel_exceeds_range = std::abs(am.x()) > a_range ||
                                     std::abs(am.y()) > a_range ||
                                     std::abs(am.z()) > a_range;
    }
  }

  // Run specific motion filter
  segment->filtered_data = segment->data;
  segment->corrected_sigmas = sigmas_;
  segment->vibration_status = VibrationStatus::kUnknown;
  if (options_.motion_filter_method == "polynomial") {
    runPolynomialMotionFilter(segment, sampling_rate);
  } else if (options_.motion_filter_method == "lowpass") {
    runLowPassMotionFilter(segment, sampling_rate);
  } else if (options_.motion_filter_method != "none") {
    LOGE(
        "ImuHandler::runMotionFilter(): Unknown motion filter method: %s",
        options_.motion_filter_method.c_str());
  }

  // Logging for debugging
  if (segment->gyro_exceeds_range) {
    LOGD(
        YELLOW
        "ImuHandler::runMotionFilter(): Gyro measurements exceed the "
        "specified range (%f deg/s)." RESET,
        options_.gyro_range);
  }
  if (segment->accel_exceeds_range) {
    LOGD(
        YELLOW
        "ImuHandler::runMotionFilter(): Accelerometer measurements exceed "
        "the specified range (%f g)." RESET,
        options_.acc_range);
  }
  if (segment->vibration_status == VibrationStatus::kNoVibration) {
    LOGD(BLUE
         "ImuHandler::runMotionFilter(): No vibration detected in the IMU "
         "data." RESET);
  } else if (segment->vibration_status == VibrationStatus::kVibrationDetected) {
    LOGD(YELLOW
         "ImuHandler::runMotionFilter(): Vibration detected in the IMU "
         "data!" RESET);
  } else {
    LOGD(
        RED "ImuHandler::runMotionFilter(): Unknown vibration status!" RESET,
        static_cast<int>(segment->vibration_status));
  }
}

void ImuHandler::runPolynomialMotionFilter(
    Segment* segment, double sampling_rate) const {
  size_t n = segment->data.size();
  segment->filtered_data = segment->data;
  segment->corrected_sigmas = sigmas_;
  segment->vibration_status = VibrationStatus::kNoVibration;

  int polynomial_order = options_.polynomial_order;
  int errors_dof = static_cast<int>(n) - (polynomial_order + 1);
  if (errors_dof <= 0) {
    LOGD(
        YELLOW
        "ImuHandler::runPolynomialMotionFilter(): A minimum of %d data "
        "points is required to fit a polynomial of order %d, plus at least 1 "
        "additional point to compute residual errors, while only %d data "
        "points are provided." RESET,
        polynomial_order + 2, polynomial_order, n);
    segment->vibration_status = VibrationStatus::kUnknown;
    return;

    // polynomial_order = 0;
    // errors_dof = static_cast<int>(n) - (polynomial_order + 1);
  }

  // Convert continuous-time noise standard deviation to discrete-time
  double discrete_accel_sigma =
      sigmas_.ct_accel_sigma * std::sqrt(sampling_rate);
  LOGD(
      "ImuHandler::runPolynomialMotionFilter(): "
      "DEBUG ct_accel_sigma = %f, discrete_accel_sigma = %f",
      sigmas_.ct_accel_sigma, discrete_accel_sigma);

  // Adjust time vector to improve numerical stability
  Eigen::VectorXd t(n);
  for (size_t i = 0; i < n; ++i) {
    t(i) = segment->data[i].timestamp;
  }
  t = t.array() - t(0);  // Subtract the first timestamp from all timestamps

  // Construct the design matrix X for polynomial fitting
  Eigen::MatrixXd X(t.size(), polynomial_order + 1);
  X.col(0) = Eigen::VectorXd::Ones(t.size());  // The first column is all ones

  // Compute higher order terms iteratively
  for (int i = 1; i <= polynomial_order; ++i) {
    X.col(i) = X.col(i - 1).cwiseProduct(
        t);  // Each column is the previous column multiplied by t
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
  ASSERT(qr.rank() >= polynomial_order + 1);

  double max_normalized_rmse = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::VectorXd a(segment->data.size());
    for (size_t i = 0; i < segment->data.size(); ++i) {
      a(i) = segment->data[i].am(axis);
    }

    // Perform least squares fitting to find polynomial coefficients using QR
    // decomposition
    Eigen::VectorXd coeffs = qr.solve(a);

    // Compute the fitted values and residuals
    Eigen::VectorXd fitted = X * coeffs;
    for (size_t i = 0; i < segment->data.size(); ++i) {
      segment->filtered_data[i].am(axis) = fitted(i);
    }
    Eigen::VectorXd residuals = a - fitted;

    // Update max_normalized_rmse and the vibration status.

    // Normalize residuals using discrete-time noise standard deviation
    Eigen::VectorXd normalized_residuals = residuals / discrete_accel_sigma;

    // Calculate RMSE and maximum absolute error of the normalized residuals
    double normalized_rmse =
        std::sqrt(normalized_residuals.squaredNorm() / errors_dof);
    double max_normalized_error = normalized_residuals.cwiseAbs().maxCoeff();
    LOGD(
        "ImuHandler::runPolynomialMotionFilter(): DEBUG Axis %d, "
        "normalized_rmse = %f, max_normalized_error = %f",
        axis, normalized_rmse, max_normalized_error);

    // Check if either RMSE or maximum error exceeds the predefined thresholds
    if (segment->vibration_status == VibrationStatus::kNoVibration) {
      if (normalized_rmse > options_.vibration_rmse_thr ||
          max_normalized_error > options_.vibration_max_err_thr) {
        segment->vibration_status =
            VibrationStatus::kVibrationDetected;  // Vibration detected
      }
    }
    max_normalized_rmse = std::max(max_normalized_rmse, normalized_rmse);
  }

  // Correct the acc sigma
  if (options_.correct_sigmas) {
    segment->corrected_sigmas.ct_accel_sigma *= max_normalized_rmse;
  }
}

std::shared_ptr<const ImuHandler::Segment> ImuHandler::processNewSegment(
    const ImuDataBuffer& imu_data_buf, double start_time, double end_time,
    const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias,
    const Vector3d& gravity_in_start_frame, const Eigen::MatrixXd& bias_cov_6x6,
    bool state_only) const {
  auto segment = std::make_shared<Segment>();
  segment->start_time = start_time;
  segment->end_time = end_time;

  // Get IMU data between the two time points
  const UniqueId& imu_uid = imu_uid_;
  // ASSERT(imu_data_buf.getLatestImuTime(imu_uid) >= end_time);
  if (imu_data_buf.getLatestImuTime(imu_uid) < end_time) {
    LOGD(
        YELLOW
        "ImuHandler::processNewSegment(): IMU data after end_time %f (> %f) "
        "not ready!" RESET,
        end_time, imu_data_buf.getLatestImuTime(imu_uid));
    return nullptr;
  }
  segment->data = imu_data_buf.getImuDataBetween(imu_uid, start_time, end_time);
  // ASSERT(segment->data.size() >= 2);
  if (segment->data.size() < 2) {
    LOGD(
        YELLOW
        "ImuHandler::processNewSegment(): IMU data between %f and %f not "
        "available!" RESET,
        start_time, end_time);
    return nullptr;
  }

  // First run the motion filter to correct the IMU data and sigmas
  runMotionFilter(segment.get(), options_.sampling_rate);

  // If the time gap between two consecutive IMU data is too large, the
  // integration will be inaccurate. We fill the gaps by interpolating the
  // IMU data to avoid this.
  std::vector<ImuData> prop_data;
  if (options_.integrate_filtered_data) {
    prop_data = ImuData::fillDataGaps(segment->filtered_data, 0.011);
  } else {
    prop_data = ImuData::fillDataGaps(segment->data, 0.011);
  }
  ASSERT(prop_data.size() >= 2);

  // Feed data to the imu preintegrator

  // Prepare the IMU preintegrator
  const ImuSigmas& imu_sigmas = segment->corrected_sigmas;
  auto imu_preint_options = ImuIntegration::Options::PreIntegration();
  imu_preint_options.cache_intermediate_results = true;
  if (state_only) {
    imu_preint_options = ImuIntegration::Options::StateBuffer();
  }
  auto imu_preint = std::make_shared<ImuIntegration>(
      imu_preint_options, imu_sigmas, gyro_bias, accel_bias);

  // Prepare flags and values needed to analyze the motion (whether it's
  // zero-rotation, or constant-velocity).
  ASSERT(bias_cov_6x6.rows() == bias_cov_6x6.cols());
  ASSERT(bias_cov_6x6.rows() == 0 || bias_cov_6x6.rows() == 6);
  bool is_zero_rotation =
      (!state_only && bias_cov_6x6.rows() == 6 && segment->isGyroReliable());
  bool is_const_velocity =
      (!state_only && bias_cov_6x6.rows() == 6 && segment->isGyroReliable() &&
       segment->isAccReliable());

  double ex_range_gyro_var = -1.;
  double ex_range_acc_var = -1.;
  const double gyro_range_rad = options_.gyro_range * M_PI / 180.0;
  if (options_.gyro_range > 0. && options_.ex_range_gyro_sigma > 0.) {
    ex_range_gyro_var =
        (options_.ex_range_gyro_sigma * options_.ex_range_gyro_sigma);
  }
  if (options_.acc_range > 0. && options_.ex_range_acc_sigma > 0.) {
    ex_range_acc_var =
        (options_.ex_range_acc_sigma * options_.ex_range_acc_sigma);
  }

  // PreIntegrate the IMU data and analyze the motion
  size_t data_i = 0;
  std::vector<double> const_velocity_chi2s,
      zero_rotation_chi2s;  // For debugging
  double prev_time = -1;
  for (const auto& imu_data : prop_data) {
    const auto& t = imu_data.timestamp;
    if (t <= prev_time) {
      LOGD(
          RED
          "ImuHandler::processNewSegment(): IMU prop data is not in order! %f "
          "<= %f" RESET,
          t, prev_time);
      continue;
    }

    double dt = 1e6;  // Defaults to a large value
    if (prev_time >= 0) {
      dt = t - prev_time;
    }
    prev_time = t;

    const auto& wm = imu_data.wm;
    const auto& am = imu_data.am;
    auto wm_vars = imu_preint->getGyroDiscretVars(wm, dt);
    auto am_vars = imu_preint->getAccelDiscretVars(am, dt);

    // Set entries of wm_vars to ex_range_gyro_var if the gyro measurement
    // exceeds the specified range.
    if (ex_range_gyro_var > 0.) {
      wm_vars = (wm.cwiseAbs().array() > gyro_range_rad)
                    .select(ex_range_gyro_var, wm_vars);
    }

    // Set entries of am_vars to ex_range_acc_var if the accel measurement
    // exceeds the specified range.
    if (ex_range_acc_var > 0.) {
      am_vars = (am.cwiseAbs().array() > options_.acc_range)
                    .select(ex_range_acc_var, am_vars);
    }

    imu_preint->update(t, wm, am, wm_vars, am_vars);

    if (data_i > 0 && is_const_velocity) {
      Eigen::MatrixXd vel_cov =
          imu_preint->getLatestResult()
              .process_noise_cov.bottomRightCorner<3, 3>();
      const auto& J_bias = imu_preint->getLatestResult().J_bias.bottomRows<3>();
      vel_cov += J_bias * bias_cov_6x6 * J_bias.transpose();
      Eigen::Vector3d vel =
          imu_preint->retrieveLatestState(true, 9.81, gravity_in_start_frame)
              .v();
      double chi2 = vel.dot(vel_cov.llt().solve(vel));
      const_velocity_chi2s.push_back(chi2);
      if (chi2 > options_.const_velocity_chi2_thr) {
        is_const_velocity = false;
      }
    }

    if (data_i > 0 && is_zero_rotation) {
      Eigen::MatrixXd rot_cov =
          imu_preint->getLatestResult().process_noise_cov.topLeftCorner<3, 3>();
      const auto& J_bias =
          imu_preint->getLatestResult().J_bias.topLeftCorner<3, 3>();
      rot_cov +=
          J_bias * bias_cov_6x6.topLeftCorner<3, 3>() * J_bias.transpose();

      Rot3d rot = imu_preint->retrieveLatestState(false).R();
      Eigen::Vector3d rot_vec = SO3d::Log(rot);
      double chi2 = rot_vec.dot(rot_cov.llt().solve(rot_vec));
      zero_rotation_chi2s.push_back(chi2);
      if (chi2 > options_.zero_rotation_chi2_thr) {
        is_zero_rotation = false;
      }
    }

    ++data_i;
  }
  LOGD(
      "ImuHandler::processNewSegment:  const_velocity_chi2s = %s",
      toStr(const_velocity_chi2s).c_str());
  LOGD(
      "ImuHandler::processNewSegment:  zero_rotation_chi2s = %s",
      toStr(zero_rotation_chi2s).c_str());

  // Save the integration and motion analysis results
  segment->pre_integration = std::move(imu_preint);
  segment->is_zero_rotation = is_zero_rotation;
  segment->is_const_velocity = is_const_velocity;

  // Cache and return the processed segment
  cacheNew(segment);
  return segment;
}

bool ImuHandler::Segment::predictRotation(
    double time, Rot3d* predicted_rot, const Rot3d& start_rot) const {
  ImuIntegration::State state;
  bool success = pre_integration->retrieveState(time, &state, false);
  if (!success) {
    return false;
  }

  if (predicted_rot) {
    *predicted_rot = start_rot * state.R();
  }

  return true;
}

std::unordered_map<double, Rot3d> ImuHandler::Segment::predictRotations(
    const std::vector<double> times, const Rot3d& start_rot) const {
  std::unordered_map<double, ImuIntegration::State> states =
      pre_integration->retrieveStates(times, false);
  std::unordered_map<double, Rot3d> rotations;
  for (const auto& [time, state] : states) {
    rotations[time] = start_rot * state.R();
  }
  return rotations;
}

bool ImuHandler::Segment::predictState(
    double time, Pose3d* predicted_pose, Eigen::Vector3d* predicted_vel,
    const Pose3d& start_pose, const Eigen::Vector3d& start_vel,
    bool apply_gravity) const {
  const Rot3d& start_rot = start_pose.rotation();
  Eigen::Vector3d gravity_in_start_frame =
      start_rot.matrix().transpose().col(2);
  ImuIntegration::State state;
  bool success = pre_integration->retrieveState(
      time, &state, apply_gravity, 9.81, gravity_in_start_frame);
  if (!success) {
    return false;
  }

  double dt = time - start_time;
  if (predicted_pose) {
    predicted_pose->rotation() = start_rot * state.R();
    predicted_pose->translation() =
        start_pose.translation() + start_vel * dt + start_rot * state.p();
  }

  if (predicted_vel) {
    *predicted_vel = start_vel + start_rot * state.v();
  }

  return true;
}

std::unordered_map<double, ImuIntegration::State>
ImuHandler::Segment::predictStates(
    const std::vector<double> times, const Pose3d& start_pose,
    const Eigen::Vector3d& start_vel, bool apply_gravity,
    bool trasform_velocity) const {
  const Rot3d& start_rot = start_pose.rotation();
  Eigen::Vector3d gravity_in_start_frame =
      start_rot.matrix().transpose().col(2);
  std::unordered_map<double, ImuIntegration::State> states =
      pre_integration->retrieveStates(
          times, apply_gravity, 9.81, gravity_in_start_frame);

  // Transform the states to the start frame
  for (auto& [time, state] : states) {
    double dt = time - start_time;
    state.R() = start_rot * state.R();
    state.p() =
        start_pose.translation() + start_vel * dt + start_rot * state.p();
    if (trasform_velocity) {
      state.v() = start_vel + start_rot * state.v();
    }
  }

  return states;
}

}  // namespace sk4slam
