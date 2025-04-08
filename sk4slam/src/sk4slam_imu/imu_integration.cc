
#include "sk4slam_imu/imu_integration.h"

#include "sk4slam_basic/string_helper.h"
#include "sk4slam_cpp/binary_search.h"

namespace sk4slam {

/// @brief  Creates options suitable for IMU Pre-integration.
ImuIntegration::Options ImuIntegration::Options::PreIntegration() {
  Options options;
  options.cache_measurements = true;  // We may need to re-propagate when
                                      // biases change significantly.
  options.cache_intermediate_results = false;
  options.compute_jacobian_wrt_bias = true;
  options.compute_jacobian_wrt_state = false;
  options.compute_process_noise_cov = true;
  return options;
}

/// @brief  Creates options suitable for EKF propagation.
ImuIntegration::Options ImuIntegration::Options::CovPropagation() {
  Options options;
  options.cache_measurements = false;
  options.cache_intermediate_results = false;
  options.compute_jacobian_wrt_bias = true;
  options.compute_jacobian_wrt_state = true;
  options.compute_process_noise_cov = true;
  return options;
}

/// @brief  Creates options for only computing the final state.
ImuIntegration::Options ImuIntegration::Options::StateOnly() {
  Options options;
  options.cache_measurements = false;
  options.cache_intermediate_results = false;
  options.compute_jacobian_wrt_bias = false;
  options.compute_jacobian_wrt_state = false;
  options.compute_process_noise_cov = false;
  return options;
}

/// @brief  Creates options for only computing the states and saving the
/// state buffer for later retrieval.
ImuIntegration::Options ImuIntegration::Options::StateBuffer() {
  Options options;
  options.cache_measurements = false;
  options.cache_intermediate_results = true;
  options.compute_jacobian_wrt_bias = false;
  options.compute_jacobian_wrt_state = false;
  options.compute_process_noise_cov = false;
  return options;
}

const ImuIntegration::Result& ImuIntegration::update(
    const Timestamp& timestamp, const Vector3d& gyro, const Vector3d& accel) {
  if (timestamps_.empty()) {
    ASSERT(initial_time_ < 0);
    initial_time_ = timestamp;
    timestamps_.push_back(timestamp);
    accel_measurements_.push_back(accel);
    gyro_measurements_.push_back(gyro);
    results_.emplace_back(Result(initial_state_));
    return results_.back();
  } else if (timestamp <= timestamps_.back()) {
    LOGW(
        "IMU integration: timestamp is not strictly increasing! The new "
        "timestamp is %f, but the last timestamp is %f (new - last = %f)."
        "We'll ignore the new measurement!",
        timestamp, timestamps_.back(), timestamp - timestamps_.back());
    return results_.back();
  } else {
    const Timestamp& last_time = timestamps_.back();
    const Vector3d& last_gyro = gyro_measurements_.back();
    const Vector3d& last_accel = accel_measurements_.back();
    double dt = timestamp - last_time;
    const Result& last_result = results_.back();
    Result new_result =
        integrate(last_result, dt, last_gyro, gyro, last_accel, accel);
    cache(timestamp, gyro, accel, std::move(new_result));
    return results_.back();
  }
}

const ImuIntegration::Result& ImuIntegration::repropagate(
    const Vector3d& new_gyro_bias, const Vector3d& new_accel_bias,
    const State& new_initial_state) {
  ASSERT(options_.cache_measurements);
  ASSERT(options_.cache_intermediate_results || results_.size() == 1);
  ASSERT(
      !options_.cache_intermediate_results ||
      results_.size() == timestamps_.size());
  ASSERT(gyro_measurements_.size() == timestamps_.size());
  ASSERT(
      options_.rotation_only ||
      accel_measurements_.size() == timestamps_.size());
  ASSERT(!options_.rotation_only || accel_measurements_.size() == 1);

  gyro_bias_ = new_gyro_bias;
  accel_bias_ = new_accel_bias;
  initial_state_ = new_initial_state;
  results_[0] = Result(initial_state_);

  for (size_t i = 1; i < timestamps_.size(); ++i) {
    const Timestamp& last_time = timestamps_[i - 1];
    const Vector3d& last_gyro = gyro_measurements_[i - 1];
    int last_accel_idx = options_.rotation_only ? 0 : i - 1;
    const Vector3d& last_accel = accel_measurements_[last_accel_idx];
    int last_result_idx = options_.cache_intermediate_results ? i - 1 : 0;
    const Result& last_result = results_[last_result_idx];

    const Timestamp& timestamp = timestamps_[i];
    const Vector3d& gyro = gyro_measurements_[i];
    int accel_idx = options_.rotation_only ? 0 : i;
    const Vector3d& accel = accel_measurements_[accel_idx];

    double dt = timestamp - last_time;
    int result_idx = options_.cache_intermediate_results ? i : 0;
    results_[result_idx] =
        integrate(last_result, dt, last_gyro, gyro, last_accel, accel);
  }
  return results_.back();
}

Eigen::MatrixXd ImuIntegration::propagateCovariance(
    const Eigen::MatrixXd& initial_covariance, bool include_bias_cov) const {
  ASSERT(options_.compute_jacobian_wrt_bias);
  ASSERT(options_.compute_jacobian_wrt_state);
  ASSERT(options_.compute_process_noise_cov);

  // Check matrix dimensions
  int bias_dim = 0;
  if (include_bias_cov) {
    if (options_.rotation_only) {
      bias_dim = 3;
      ASSERT(initial_covariance.rows() == 6 && initial_covariance.cols() == 6);
    } else {
      bias_dim = 6;
      ASSERT(
          initial_covariance.rows() == 15 && initial_covariance.cols() == 15);
    }
  } else {
    if (options_.rotation_only) {
      ASSERT(initial_covariance.rows() == 3 && initial_covariance.cols() == 3);
    } else {
      ASSERT(initial_covariance.rows() == 9 && initial_covariance.cols() == 9);
    }
  }

  // Compute the covariance matrix for the propagated state
  int total_dim = initial_covariance.rows();
  int state_dim = total_dim - bias_dim;
  Eigen::MatrixXd covariance = Eigen::MatrixXd(total_dim, total_dim);

  const Result& last_result = getLatestResult();
  const Timestamp& last_time = getLatestTime();
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(state_dim, total_dim);
  if (include_bias_cov) {
    J.block(0, 0, state_dim, state_dim) =
        last_result.J_state;  // Rotation, Position, Velocity
    J.block(0, state_dim, state_dim, bias_dim) =
        last_result.J_bias;  // GyroBias, AccelBias
  } else {
    J = last_result.J_state;  // Rotation, Position, Velocity
  }
  Eigen::MatrixXd JC = J * initial_covariance;
  covariance.block(0, 0, state_dim, state_dim) =
      JC * J.transpose();  // Propagate the state covariance
  if (last_result.process_noise_cov.rows() > 0) {
    ASSERT(last_result.process_noise_cov.rows() == state_dim);
    covariance.block(0, 0, state_dim, state_dim) +=
        last_result.process_noise_cov;  // Add process noise for the state
  }
  if (bias_dim > 0) {
    // Propagate the bias covariance
    covariance.block(state_dim, state_dim, bias_dim, bias_dim) =
        initial_covariance.block(state_dim, state_dim, bias_dim, bias_dim);
    covariance.block(0, state_dim, state_dim, bias_dim) =
        JC.block(0, state_dim, state_dim, bias_dim);
    covariance.block(state_dim, 0, bias_dim, state_dim) =
        JC.block(0, state_dim, state_dim, bias_dim).transpose();

    // Add process noise (random walk noise) to the bias terms
    double DT = last_time - timestamps_[0];
    double gyro_bias_sigma2 =
        sigmas_.ct_gyro_bias_sigma * sigmas_.ct_gyro_bias_sigma * DT;
    covariance.block(state_dim, state_dim, 3, 3) +=
        gyro_bias_sigma2 * Eigen::MatrixXd::Identity(3, 3);
    if (!options_.rotation_only) {
      double accel_bias_sigma2 =
          sigmas_.ct_accel_bias_sigma * sigmas_.ct_accel_bias_sigma * DT;
      covariance.block(state_dim + 3, state_dim + 3, 3, 3) +=
          accel_bias_sigma2 * Eigen::MatrixXd::Identity(3, 3);
    }
  }
  return covariance;
}

ImuIntegration::State ImuIntegration::retrieveLatestState(
    bool apply_gravity, double gravity_magnitude,
    const Vector3d& gravity_direction_in_ref) const {
  if (!options_.rotation_only && apply_gravity) {
    return applyGravity(
        getLatestTime(), gravity_magnitude * gravity_direction_in_ref,
        getLatestResult().state);
  } else {
    return getLatestResult().state;
  }
}

void ImuIntegration::interpolate(
    double alpha, const State& state0, const State& state1,
    State* interpolated) const {
#define ALLOW_ROTATION_ONLY_INTERPOLATION
#ifndef ALLOW_ROTATION_ONLY_INTERPOLATION
  OptimizableState from_state(state0);
  OptimizableState to_state(state1);
  *interpolated = from_state + alpha * (to_state - from_state);
#else
  const State& from_state = state0;
  const State& to_state = state1;
  using OptimizableRot3d = OptimizableManifold<Rot3d, Rot3d::LeftPerturbation>;
  OptimizableRot3d from_R(from_state.R());
  OptimizableRot3d to_R(to_state.R());
  interpolated->R() = from_R + alpha * (to_R - from_R);
  if (!options_.rotation_only) {
    interpolated->p() =
        from_state.p() + alpha * (to_state.p() - from_state.p());
    interpolated->v() =
        from_state.v() + alpha * (to_state.v() - from_state.v());
    {
      // debug
      OptimizableState from_state(state0);
      OptimizableState to_state(state1);
      State state2 = from_state + alpha * (to_state - from_state);
      ASSERT(interpolated->isApprox(state2));
    }
  }
#endif
}

bool ImuIntegration::retrieveState(
    const Timestamp& timestamp, State* state, bool apply_gravity,
    double gravity_magnitude, const Vector3d& gravity_direction_in_ref) const {
  auto search_res = binarySearchLowerBound(timestamps_, timestamp);
  if (search_res) {
    if (timestamps_[*search_res] == timestamp) {
      *state = results_[*search_res].state;
      if (apply_gravity) {
        applyGravity(
            timestamp, gravity_magnitude * gravity_direction_in_ref, state);
      }
      return true;
    } else if (*search_res == 0) {
      // The timestamp is before the first cached timestamp
      return false;
    } else if (*search_res >= timestamps_.size()) {
      // The timestamp is after the last cached timestamp
      return false;
    } else {
      ASSERT(*search_res > 0 && *search_res < timestamps_.size());
      double alpha = (timestamp - timestamps_[*search_res - 1]) /
                     (timestamps_[*search_res] - timestamps_[*search_res - 1]);
      State state0 = results_[*search_res - 1].state;
      State state1 = results_[*search_res].state;
      if (!options_.rotation_only && apply_gravity) {
        Vector3d gravity = gravity_magnitude * gravity_direction_in_ref;
        applyGravity(timestamps_[*search_res - 1], gravity, &state0);
        applyGravity(timestamps_[*search_res], gravity, &state1);
      }

      // Interpolate between the two closest timestamps
      interpolate(alpha, state0, state1, state);
      return true;
    }
  } else {
    return false;
  }
}

const ImuIntegration::Result* ImuIntegration::findResult(
    const Timestamp& timestamp) const {
  auto search_res = binarySearchFirst(timestamps_, timestamp);
  if (search_res) {
    return nullptr;
  } else {
    return &results_[*search_res];
  }
}

ImuIntegration::Result ImuIntegration::integrate(
    const Result& prev_result, double dt, const Vector3d& prev_gyro,
    const Vector3d& new_gyro, const Vector3d& prev_accel,
    const Vector3d& new_accel) {
  Vector3d mean_gyro, mean_accel;
  mean_gyro = (new_gyro + prev_gyro) / 2.0;
  if (!options_.rotation_only) {
    mean_accel = (new_accel + prev_accel) / 2.0;
  }

  Result r;
  const Vector3d& bg = gyro_bias_;
  const Vector3d& ba = accel_bias_;
  Vector3d unbiased_gyro, unbiased_accel;
  unbiased_gyro = mean_gyro - bg;
  if (!options_.rotation_only) {
    unbiased_accel = mean_accel - ba;
  }
  Rot3d deltaR = Rot3d::Exp(unbiased_gyro * dt);
  r.state.R() = prev_result.state.R() * deltaR;

  const Matrix3d& Rk = prev_result.state.R();
  const Vector3d& pk = prev_result.state.p();
  const Vector3d& vk = prev_result.state.v();

  Vector3d acc_in_ref;
  if (!options_.rotation_only) {
    acc_in_ref = prev_result.state.R() * unbiased_accel;
    r.state.v() = prev_result.state.v() + acc_in_ref * dt;
    r.state.p() = prev_result.state.p() + prev_result.state.v() * dt +
                  0.5 * acc_in_ref * dt * dt;
  }

  const Matrix3d& Rkp1 = r.state.R();
  const Vector3d& pkp1 = r.state.p();
  const Vector3d& vkp1 = r.state.v();

  // Compute the jacobian wrt the prev state if:
  MatrixXd J_prev_state;
  bool compute_jacobian_wrt_prev_state =
      // 1. compute_jacobian_wrt_state is requested,
      options_.compute_jacobian_wrt_state ||
      // 2. or compute_jacobian_wrt_bias is requested,
      options_.compute_jacobian_wrt_bias ||
      // 3. or it is needed in the subsequent computation of the covariance of
      //   the accumulated process noise.
      options_.compute_process_noise_cov &&
          prev_result.process_noise_cov.rows() > 0;
  if (compute_jacobian_wrt_prev_state) {
    if (options_.rotation_only) {
      J_prev_state = MatrixXd::Identity(3, 3);  // (D R_kp1) / (D R_k)
    } else {
      Matrix3d hat_acc_in_ref = SO3d::hat(acc_in_ref);
      J_prev_state = MatrixXd::Zero(9, 9);
      J_prev_state.block<3, 3>(0, 0) =
          MatrixXd::Identity(3, 3);  // (D R_kp1) / (D R_k)
      J_prev_state.block<3, 3>(3, 0) =
          (-0.5 * dt * dt) * hat_acc_in_ref;  // (D p_kp1) / (D R_k)
      J_prev_state.block<3, 3>(3, 3) =
          MatrixXd::Identity(3, 3);  // (D p_kp1) / (D p_k)
      J_prev_state.block<3, 3>(3, 6) =
          dt * MatrixXd::Identity(3, 3);  // (D p_kp1) / (D v_k)
      J_prev_state.block<3, 3>(6, 0) =
          (-dt) * hat_acc_in_ref;  // (D v_kp1) / (D R_k)
      J_prev_state.block<3, 3>(6, 6) =
          MatrixXd::Identity(3, 3);  // (D v_kp1) / (D v_k)
    }
  }

  if (options_.compute_jacobian_wrt_state) {
    if (prev_result.J_state.rows() > 0) {
      ASSERT(prev_result.J_state.rows() == J_prev_state.rows());
      ASSERT(prev_result.J_state.cols() == J_prev_state.cols());
      r.J_state = J_prev_state * prev_result.J_state;  // Apply the chain rule
    } else {
      // prev_result corresponds to the first measurement.
      r.J_state = J_prev_state;
    }
  }

  Matrix3d Rk_Jl_dt;
  if (options_.compute_process_noise_cov ||
      options_.compute_jacobian_wrt_bias) {
    Rk_Jl_dt = Rk * SO3d::Jl(unbiased_gyro * dt) * dt;
  }

  if (options_.compute_jacobian_wrt_bias) {
    if (options_.rotation_only) {
      r.J_bias = -Rk_Jl_dt;  // (D R_kp1) / (D bg)
    } else {
      r.J_bias = MatrixXd::Zero(9, 6);
      r.J_bias.block<3, 3>(0, 0) = -Rk_Jl_dt;              // (D R_kp1) / (D bg)
      r.J_bias.block<3, 3>(3, 3) = -(0.5 * dt * dt) * Rk;  // (D p_kp1) / (D ba)
      r.J_bias.block<3, 3>(6, 3) = -dt * Rk;               // (D v_kp1) / (D ba)
    }

    // Add the jacobian propagated from the previous state.
    if (prev_result.J_bias.rows() > 0) {
      ASSERT(prev_result.J_bias.rows() == r.J_bias.rows());
      ASSERT(prev_result.J_bias.cols() == r.J_bias.cols());
      r.J_bias += J_prev_state * prev_result.J_bias;  // Apply the chain rule
    } else {
      // prev_result corresponds to the first measurement.
      // Nothing to add.
    }
  }

  if (options_.compute_process_noise_cov) {
    const double gyro_simga2 =
        sigmas_.ct_gyro_sigma * sigmas_.ct_gyro_sigma / dt;
    const double accel_sigma2 =
        sigmas_.ct_accel_sigma * sigmas_.ct_accel_sigma / dt;
    const double gyro_scale_sigma2 =
        sigmas_.gyro_scale_err_sigma * sigmas_.gyro_scale_err_sigma;
    const double accel_scale_sigma2 =
        sigmas_.accel_scale_err_sigma * sigmas_.accel_scale_err_sigma;
    const Eigen::Vector3d gyro_vars(
        gyro_simga2 + gyro_scale_sigma2 * mean_gyro.x() * mean_gyro.x(),
        gyro_simga2 + gyro_scale_sigma2 * mean_gyro.y() * mean_gyro.y(),
        gyro_simga2 + gyro_scale_sigma2 * mean_gyro.z() * mean_gyro.z());
    const Eigen::Vector3d accel_vars(
        accel_sigma2 + accel_scale_sigma2 * mean_accel.x() * mean_accel.x(),
        accel_sigma2 + accel_scale_sigma2 * mean_accel.y() * mean_accel.y(),
        accel_sigma2 + accel_scale_sigma2 * mean_accel.z() * mean_accel.z());
    const Eigen::Matrix3d gyro_cov = gyro_vars.asDiagonal();
    const Eigen::Matrix3d accel_cov = accel_vars.asDiagonal();

    // Compute the covariance of process noise in the current step
    if (options_.rotation_only) {
      r.process_noise_cov =
          Rk_Jl_dt * gyro_cov * Rk_Jl_dt.transpose();  // Cov(R_kp1, R_kp1)
    } else {
      r.process_noise_cov = MatrixXd::Zero(9, 9);
      r.process_noise_cov.block<3, 3>(0, 0) =
          Rk_Jl_dt * gyro_cov * Rk_Jl_dt.transpose();  // Cov(R_kp1, R_kp1)

      double dt2_0p5 = 0.5 * dt * dt;
      r.process_noise_cov.block<3, 3>(3, 3) =
          dt2_0p5 * dt2_0p5 * accel_cov;  // Cov(p_kp1, p_kp1)
      r.process_noise_cov.block<3, 3>(6, 6) =
          dt * dt * accel_cov;  // Cov(v_kp1, v_kp1)
      r.process_noise_cov.block<3, 3>(3, 6) =
          dt2_0p5 * dt * accel_cov;  // Cov(p_kp1, v_kp1)
      r.process_noise_cov.block<3, 3>(6, 3) =
          dt2_0p5 * dt * accel_cov;  // Cov(v_kp1, p_kp1)
    }

    // Add the process noise from the previous state
    if (prev_result.process_noise_cov.rows() > 0) {
      ASSERT(
          prev_result.process_noise_cov.rows() == r.process_noise_cov.rows());
      ASSERT(
          prev_result.process_noise_cov.cols() == r.process_noise_cov.cols());
      r.process_noise_cov += J_prev_state * prev_result.process_noise_cov *
                             J_prev_state.transpose();
    } else {
      // prev_result corresponds to the first measurement.
      // Nothing to add.
    }
  }

  return r;
}

void ImuIntegration::applyGravity(
    const Timestamp& timestamp, const Vector3d& gravity, State* state) const {
  ASSERT(initial_time_ >= 0);
  double DT = timestamp - initial_time_;
  state->p() -= (0.5 * DT * DT) * gravity;
  state->v() -= DT * gravity;
}

void ImuIntegration::cache(
    const Timestamp& timestamp, const Vector3d& gyro, const Vector3d& accel,
    Result result) {
  if (options_.cache_intermediate_results || options_.cache_measurements) {
    timestamps_.push_back(timestamp);
  } else {
    ASSERT(timestamps_.size() == 1);
    timestamps_[0] = timestamp;
  }

  if (options_.cache_intermediate_results) {
    results_.push_back(std::move(result));
    ASSERT(results_.size() == timestamps_.size());
  } else {
    ASSERT(results_.size() == 1);
    results_[0] = std::move(result);
  }

  if (options_.cache_measurements) {
    gyro_measurements_.push_back(gyro);
    ASSERT(gyro_measurements_.size() == timestamps_.size());

    if (!options_.rotation_only) {
      accel_measurements_.push_back(accel);
      ASSERT(accel_measurements_.size() == timestamps_.size());
    } else {
      ASSERT(accel_measurements_.size() == 1);
    }
  } else {
    ASSERT(gyro_measurements_.size() == 1);
    ASSERT(accel_measurements_.size() == 1);
    gyro_measurements_[0] = gyro;
    if (!options_.rotation_only) {
      accel_measurements_[0] = accel;
    }
  }
}

void ImuEKFPropagation::init(
    const Cov& prior_cov, const State& initial_state, const Vector3d& gyro_bias,
    const Vector3d& accel_bias) {
  ASSERT(CheckCovSize(prior_cov));
  reset();
  cov_ = prior_cov;
  gyro_bias_ = gyro_bias;
  accel_bias_ = accel_bias;
  initial_state_ = initial_state;
}

void ImuEKFPropagation::propagate(
    const Timestamp& timestamp, const Vector3d& gyro, const Vector3d& accel) {
  bool first_measurement = results_.empty() ? true : false;
  ImuIntegration::update(timestamp, gyro, accel);
  Result& last_result = results_.back();
  if (!first_measurement) {
    cov_ = ImuIntegration::propagateCovariance(cov_, true);
    ASSERT(results_.size() == 1);
    last_result.J_state = MatrixXd();
    last_result.J_bias = MatrixXd();
    last_result.process_noise_cov = MatrixXd();
  }
  ASSERT(last_result.J_state.rows() == 0);
  ASSERT(last_result.J_bias.rows() == 0);
  ASSERT(last_result.process_noise_cov.rows() == 0);
}

void ImuEKFPropagation::applyEKFUpdate(
    const Cov& updated_cov, const State& updated_state,
    const Vector3d& updated_gyro_bias, const Vector3d& updated_accel_bias) {
  if (results_.empty()) {
    ASSERT(timestamps_.empty());
    ASSERT(initial_time_ < 0);
    init(updated_cov, updated_state, updated_gyro_bias, updated_accel_bias);
  } else {
    ASSERT(timestamps_.size() == 1);
    ASSERT(results_.size() == 1);
    ASSERT(initial_time_ >= 0);
    ASSERT(CheckCovSize(updated_cov));
    cov_ = updated_cov;
    gyro_bias_ = updated_gyro_bias;
    accel_bias_ = updated_accel_bias;

    Result& last_result = results_.back();
    last_result.state = updated_state;

    Timestamp& last_timestamp = timestamps_.back();
    initial_time_ = last_timestamp;

    ASSERT(last_result.J_state.rows() == 0);
    ASSERT(last_result.J_bias.rows() == 0);
    ASSERT(last_result.process_noise_cov.rows() == 0);
  }
}

}  // namespace sk4slam
