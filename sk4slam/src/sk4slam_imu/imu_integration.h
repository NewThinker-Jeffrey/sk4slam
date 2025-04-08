#pragma once

#include "sk4slam_imu/imu_model.h"
#include "sk4slam_pose/pose.h"

namespace sk4slam {

/// @brief Integrates IMU measurements to compute the rotation and translation
/// of the IMU relative to a predefined inertial reference frame.
/// @details This class computes the rotation and translation of the IMU
/// relative to a pre-defined inertial reference frame by integrating IMU
/// measurements. The integration is based on a first-order discrete-time
/// approximation of continuous-time integration.
/// @note Gravity is not considered during the integration. But you can get the
/// gravity-corrected velocity and translation by calling
/// @ref retrieveLatestState() or @ref retrieveState() with setting the flag
/// @c apply_gravity to true. Note that whether gravity is considered or not
/// doesn't affect the computation of jacobians and covariances.
class ImuIntegration {
 public:
  using Timestamp = double;
  using Rot3d = SO3d;
  using MatrixXd = Eigen::MatrixXd;
  using Matrix3d = Eigen::Matrix3d;

  struct Options {
    Options() {}

    /// @brief  Creates options suitable for IMU Pre-integration.
    /// @details The IMU pre-integration is used to compute the IMU state change
    /// and the covariance of the accumulated process noise during a short time
    /// interval. In addtion, jacobians w.r.t. the biases will also be computed
    /// to allow for a quick first-order approximation of the re-propagation
    /// process when biases have been changed a bit.
    static Options PreIntegration();

    /// @brief  Creates options suitable for covariance propagation.
    /// @details Covariance propagation is used to propagate the IMU state and
    /// covariance forward in time. The options are suitable for the covariance
    /// propagation, which includes computing the jacobians and covariances.
    static Options CovPropagation();

    /// @brief  Creates options for only computing the final state.
    static Options StateOnly();

    /// @brief  Creates options for only computing the states and saving the
    /// state buffer for later retrieval.
    static Options StateBuffer();

    bool cache_measurements =
        true;  ///< Whether to cache the measurements. This is required when
               ///< re-propagation is needed.
    bool cache_intermediate_results =
        true;                    ///< Whether to cache intermediate results
    bool rotation_only = false;  ///< Whether to integrate only the rotation
    bool compute_jacobian_wrt_bias =
        true;  ///< Whether to compute the jacobian w.r.t. the biases
    bool compute_jacobian_wrt_state =
        true;  ///< Whether to compute the jacobian w.r.t. the initial state
    bool compute_process_noise_cov =
        true;  ///< Whether to compute the covariance of the accumulated process
               ///< noise
  };

  /// @brief  The inertial state
  class State : public ProductManifold<Pose3d, Vector3d> {
    using Base = ProductManifold<Pose3d, Vector3d>;

   public:
    using Base::Base;
    State() : Base(Pose3d::Identity(), Vector3d::Zero()) {}

    ///< The pose w.r.t. the inertial frame of reference
    Pose3d& pose() {
      return part<0>();
    }
    const Pose3d& pose() const {
      return part<0>();
    }
    Rot3d& R() {
      return pose().rotation();
    }
    const Rot3d& R() const {
      return pose().rotation();
    }
    Vector3d& p() {
      return pose().translation();
    }
    const Vector3d& p() const {
      return pose().translation();
    }

    /// The velocity w.r.t. the inertial frame of reference
    Vector3d& v() {
      return part<1>();
    }
    const Vector3d& v() const {
      return part<1>();
    }
  };

  /// @brief  We use separate-left perturbation for the
  /// pose part, and vector perturbation for the velocity part.
  using StateRetraction = ProductRetraction<
      State, Pose3d::AffineLeftPerturbation, VectorSpaceRetraction<3>>;

  using OptimizableState = OptimizableManifold<State, StateRetraction>;

  /// @brief  Defines the IMU integration result
  struct Result {
    State state;  ///< The integrated state
    MatrixXd
        J_state;      ///< The jacobian of the integrated state w.r.t. the
                      ///< initial state. If the jacobian is computed, it size
                      ///< is 3x3 when rotation_only is true, and 9x9 otherwise.
    MatrixXd J_bias;  ///< The jacobian of the integrated state w.r.t. the
                      ///< biases. If the jacobian is computed, it size
                      ///< is 3x3 when rotation_only is true, and 9x6 otherwise.
    MatrixXd process_noise_cov;  ///< The covariance of the accumulated process
                                 ///< noise. If the covariance is computed, it
                                 ///< size is 3x3 when rotation_only is true,
                                 ///< and 9x9 otherwise.
    explicit Result(const State& set_state = State())
        : J_state(MatrixXd()),
          J_bias(MatrixXd()),
          process_noise_cov(MatrixXd()),
          state(set_state) {}
  };

 public:
  /// @brief Constructor for the IMU integration class.
  /// @param options Configuration options for IMU integration (e.g., noise
  /// parameters, caching behavior).
  /// @param gyro_bias Initial gyroscope bias in [rad/s].
  /// @param accel_bias Initial accelerometer bias in [m/s^2].
  /// @param initial_state Initial inertial state of the system.
  /// Specifying this implicitly defines an inertial reference frame.
  /// By default, the reference frame is aligned with the IMU frame at the
  /// initial timestamp.
  ImuIntegration(
      const Options& options = Options(), const ImuSigmas& sigmas = ImuSigmas(),
      const Vector3d& gyro_bias = Vector3d::Zero(),
      const Vector3d& accel_bias = Vector3d::Zero(),
      const State& initial_state = State())
      : options_(options),
        sigmas_(sigmas),
        gyro_bias_(gyro_bias),
        accel_bias_(accel_bias),
        initial_state_(initial_state),
        initial_time_(-1) {}

  virtual ~ImuIntegration() {}

  bool isRotationOnly() const {
    return options_.rotation_only;
  }

  /// @brief Add new IMU measurements and update the integrated result.
  /// @note @c accel will be ignored if the options_.rotation_only flag is set.
  /// @note Jacobians and covariances will not be computed for the first
  /// measurement.
  const Result& update(
      const Timestamp& timestamp, const Vector3d& gyro,
      const Vector3d& accel = Vector3d::Zero());

  /// @brief Repropagate the integrated result using new biases and new initial
  /// state.
  /// @param new_gyro_bias New gyroscope bias in [rad/s].
  /// @param new_accel_bias New accelerometer bias in [m/s^2].
  /// @param new_initial_state New initial inertial state of the system.
  /// Specifying this implicitly defines an inertial reference frame.
  const Result& repropagate(
      const Vector3d& new_gyro_bias, const Vector3d& new_accel_bias,
      const State& new_initial_state);

  /// @brief Repropagate the integrated result using new biases while keeping
  /// the reference frame unchanged.
  /// @param new_gyro_bias New gyroscope bias in [rad/s].
  /// @param new_accel_bias New accelerometer bias in [m/s^2].
  const Result& repropagate(
      const Vector3d& new_gyro_bias, const Vector3d& new_accel_bias) {
    return repropagate(new_gyro_bias, new_accel_bias, initial_state_);
  }

  /// @brief Propagate the covariance from the initial state to the final state.
  /// @param initial_covariance The initial covariance matrix.
  /// @param include_bias_cov Whether to include the bias covariance in the
  /// result.
  ///
  /// Variable order in the covariance matrix:
  ///    1.  Rot, [dim = 3]
  ///    2.  Pos, [dim = 3] (present only if @c !options_.rotation_only)
  ///    3.  Vel, [dim = 3] (present only if @c !options_.rotation_only)
  ///    4.  GyroBias, [dim = 3]   (present only if @c include_bias_cov)
  ///    5.  AccelBias, [dim = 3]  (present only if @c include_bias_cov AND
  ///    @c !options_.rotation_only)
  ///
  /// The dimension of the covariance matrix depends on the
  /// @c options_.rotation_only flag and the @c include_bias_cov flag, which
  /// decide which variables are present in the final state vector (augmented
  /// with the biases if include_bias_cov is true)
  Eigen::MatrixXd propagateCovariance(
      const Eigen::MatrixXd& initial_covariance,
      bool include_bias_cov = true) const;

  const Vector3d& getGyroBias() const {
    return gyro_bias_;
  }

  const Vector3d& getAccBias() const {
    return accel_bias_;
  }

  const Result& getLatestResult() const {
    return results_.back();
  }

  const Timestamp& getLatestTime() const {
    return timestamps_.back();
  }

  const Timestamp& getInitialTime() const {
    return initial_time_;
  }

  double timeWindow() const {
    return timestamps_.back() - initial_time_;
  }

  /// @brief  Retrieve the latest integrated state.
  /// @param apply_gravity Whether to apply gravity to the original inertial
  /// reference frame (which makes it a non-inertial reference frame, i.e. an
  /// accelerated reference frame). If true, the output state will be expressed
  /// in the non-inertial reference frame.
  /// @param gravity_magnitude The magnitude of gravity to apply (default: 9.81)
  /// @param gravity_direction_in_ref The direction of gravity in the
  /// inertial reference frame (default: (0, 0, 1))
  State retrieveLatestState(
      bool apply_gravity = true, double gravity_magnitude = 9.81,
      const Vector3d& gravity_direction_in_ref = Vector3d(0, 0, 1)) const;

  /// Retrieve the state at a given timestamp. This function is only valid
  /// when the IMU integration result is cached (see @ref
  /// Options::cache_intermediate_results). Interpolation will be performed
  /// if the timestamp is not exactly equal to one of the cached timestamps.
  /// @param timestamp The timestamp at which to retrieve the state
  /// @param state[out] The state at the given timestamp
  /// @param apply_gravity Whether to apply gravity to the original inertial
  /// reference frame (which makes it a non-inertial reference frame, i.e. an
  /// accelerated reference frame). If true, the output state will be expressed
  /// in the non-inertial reference frame.
  /// @param gravity_magnitude The magnitude of gravity to apply (default: 9.81)
  /// @param gravity_direction_in_ref The direction of gravity in the
  /// inertial reference frame (default: (0, 0, 1))
  /// @return true if the state was successfully retrieved, false otherwise
  /// (e.g. if the timestamp is before the first cached timestamp
  /// or after the last cached timestamp)
  bool retrieveState(
      const Timestamp& timestamp, State* state, bool apply_gravity = true,
      double gravity_magnitude = 9.81,
      const Vector3d& gravity_direction_in_ref = Vector3d(0, 0, 1)) const;

  /// Find the IMU integration result at a given timestamp. This function
  /// is only valid when the IMU integration result is cached (see @ref
  /// Options::cache_intermediate_results). The given timestamp must be
  /// exactly equal to one of the cached timestamps.
  /// @param timestamp The timestamp at which to retrieve the IMU integration
  /// result
  /// @return The IMU integration result at the given timestamp
  const Result* findResult(const Timestamp& timestamp) const;

 protected:
  /// @brief  Update the integrated result by adding new IMU measurements.
  /// @param prev_result The previous IMU integration result
  /// @param dt The time interval between the previous IMU integration result
  /// and the new IMU measurements
  /// @param prev_gyro The previous gyro measurements
  /// @param new_gyro The new gyro measurements
  /// @param prev_accel The previous accelerometer measurements
  /// @param new_accel The new accelerometer measurements
  /// @return The updated IMU integration result
  virtual Result integrate(
      const Result& prev_result, double dt, const Vector3d& prev_gyro,
      const Vector3d& new_gyro, const Vector3d& prev_accel,
      const Vector3d& new_accel);

  /// @brief  Interpolate between two inertial states
  /// @param alpha   The interpolation factor (between 0 and 1)
  /// @param state0   The first state
  /// @param state1   The second state
  /// @param interpolated[out]   The interpolated state
  virtual void interpolate(
      double alpha, const State& state0, const State& state1,
      State* interpolated) const;

  /// @brief  Apply gravity to the given state, transforming it from the
  /// original inertial reference frame to a non-inertial reference frame
  /// with the given gravity @c gravity.
  /// @param timestamp The timestamp for the given state
  /// @param gravity The gravity vector
  /// @param state The state to apply gravity to
  void applyGravity(
      const Timestamp& timestamp, const Vector3d& gravity, State* state) const;

  State applyGravity(
      const Timestamp& timestamp, const Vector3d& gravity,
      const State& state) const {
    State state_out = state;
    applyGravity(timestamp, gravity, &state_out);
    return state_out;
  }

  /// Cache the IMU integration result and the measurements if enabled by the
  /// options.
  /// @param timestamp The timestamp for the IMU integration result
  /// @param gyro The new gyro measurements
  /// @param accel The new accelerometer measurements
  /// @param result The new IMU integration result
  void cache(
      const Timestamp& timestamp, const Vector3d& gyro, const Vector3d& accel,
      Result result);

  void reset() {
    timestamps_.clear();
    accel_measurements_.clear();
    gyro_measurements_.clear();
    results_.clear();
    initial_time_ = -1;
  }

 protected:
  Options options_;      ///< The options for the IMU integration
  ImuSigmas sigmas_;     ///< The IMU noise sigmas
  State initial_state_;  ///< By specifying the initial state, we also have
                         ///< chosen an inertial frame of reference
  Vector3d gyro_bias_;   ///< The initial gyro bias
  Vector3d accel_bias_;  ///< The initial accelerometer bias

  Timestamp initial_time_;  ///< The initial time of the IMU integration. It
                            ///< will be initialized to the first timestamp of
                            ///< the IMU measurements.

  std::vector<Timestamp>
      timestamps_;  ///< The timestamps of the IMU measurements
  std::vector<Vector3d>
      accel_measurements_;                   ///< The accelerometer measurements
  std::vector<Vector3d> gyro_measurements_;  ///< The gyro measurements
  std::vector<Result> results_;              ///< The IMU integration results
};

/// @brief  A wrapper of ImuIntegration that provides a more convenient
/// interface for IMU propagation in EKF.
class ImuEKFPropagation : private ImuIntegration {
 public:
  using ImuIntegration::getAccBias;
  using ImuIntegration::getGyroBias;
  using ImuIntegration::getLatestTime;
  using ImuIntegration::retrieveLatestState;
  using ImuIntegration::State;
  // using Cov = Eigen::Matrix<double, 15, 15>;
  using Cov = MatrixXd;

  ImuEKFPropagation() : ImuIntegration(Options::CovPropagation()) {}

  void init(
      const Cov& prior_cov = Cov::Identity(15, 15),
      const State& initial_state = State(),
      const Vector3d& gyro_bias = Vector3d::Zero(),
      const Vector3d& accel_bias = Vector3d::Zero());

  void propagate(
      const Timestamp& timestamp, const Vector3d& gyro, const Vector3d& accel);

  void applyEKFUpdate(
      const Cov& updated_cov, const State& updated_state,
      const Vector3d& updated_gyro_bias, const Vector3d& updated_accel_bias);

  const Cov& getCov() const {
    return cov_;
  }

 protected:
  static bool CheckCovSize(const Cov& cov) {
    return cov.rows() == 15 && cov.cols() == 15;
  }

 private:
  Cov cov_;
};

}  // namespace sk4slam
