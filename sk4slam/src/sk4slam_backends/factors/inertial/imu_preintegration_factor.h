#pragma once

#include "sk4slam_backends/factor_base.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_imu/imu_integration.h"

namespace sk4slam {

class ImuPreIntegrationFactor : public FactorBase<ImuPreIntegrationFactor> {
  using Base = FactorBase<ImuPreIntegrationFactor>;

 public:
  using Base::JacobianMatrixXd;
  static constexpr int kResidualDim =
      -1;  // The residual dimension is dynamic, depending on whether the
           // preintegration is rotation-only or not.

  /// @brief The indices of the variables involved in the factor.
  enum VariableIdx {
    IMU_POSE0 = 0,   ///< Pose of the IMU at the start of the integration.
    IMU_VEL0,        ///< Velocity of the IMU at the start of the integration.
    IMU_GYRO_BIAS0,  ///< Bias of the gyroscope at the start of the integration.
    IMU_ACC_BIAS0,   ///< Bias of the accelerometer at the start of the
                     ///< integration.
    IMU_POSE1,       ///< Pose of the IMU at the end of the integration.
    IMU_VEL1,        ///< Velocity of the IMU at the end of the integration.

    // IMU_TIME_OFFSET,   ///< Offset of IMU timestamp TODO(jeffrey)

    NUM_VARIABLES  ///< Number of variables.
  };

  static constexpr int kNumVariables =
      static_cast<int>(VariableIdx::NUM_VARIABLES);

  /// @brief Define the types of the variables for each index.
  DECLARE_VARIABLE_TYPES(
      Pose3d,    // IMU_POSE0
      Vector3d,  // IMU_VEL0
      Vector3d,  // IMU_GYRO_BIAS0
      Vector3d,  // IMU_ACC_BIAS0
      Pose3d,    // IMU_POSE1
      Vector3d   // IMU_VEL1
                 // Vector1d   // IMU_TIME_OFFSET   TODO(jeffrey)
  )

 public:
  /// @brief Constructor for the pre-integration factor.
  /// @param imu_pre_integration  The IMU integration object.
  /// @param variable_keys Keys of the variables used in the factor.
  ImuPreIntegrationFactor(
      std::shared_ptr<ImuIntegration> imu_pre_integration,
      const std::vector<VariableKey>& variable_keys,
      double gyro_bias_repropagation_thr = 1e-3,
      double acc_bias_repropagation_thr = 1e-3, double regularization = 0.0,
      bool assume_zero_rotation = false, bool assume_zero_velocity = false,
      bool assume_constant_velocity = false)
      : imu_pre_integration_(std::move(imu_pre_integration)),
        gyro_bias_repropagation_thr_(gyro_bias_repropagation_thr),
        acc_bias_repropagation_thr_(acc_bias_repropagation_thr),
        regularization_(regularization),
        assume_constant_velocity_(assume_constant_velocity),
        assume_zero_rotation_(assume_zero_rotation),
        assume_zero_velocity_(assume_zero_velocity),
        Base(variable_keys) {}

  static ImuPreIntegrationFactor RotationOnly(
      std::shared_ptr<ImuIntegration> imu_pre_integration,
      const VariableKey& T_M_I0_key, const VariableKey& bg0_key,
      const VariableKey& T_M_I1_key, double gyro_bias_repropagation_thr = 1e-3,
      double regularization = 0.0, bool assume_zero_rotation = false) {
    return ImuPreIntegrationFactor(
        imu_pre_integration,
        {T_M_I0_key, null_variable /* v_M_I0_key */, bg0_key,
         null_variable /* ba0_key */, T_M_I1_key,
         null_variable /* v_M_I1_key */},
        gyro_bias_repropagation_thr, 1.0, regularization, assume_zero_rotation);
  }

  /// @brief Evaluate the reprojection error and optionally compute Jacobians.
  /// @param T_M_I0 Pose of the IMU at the start of the integration.
  /// @param v_M_I0 Velocity of the IMU at the start of the integration.
  /// @param bg0 Bias of the gyroscope at the start of the integration.
  /// @param ba0 Bias of the accelerometer at the start of the integration.
  /// @param T_M_I1 Pose of the IMU at the end of the integration.
  /// @param v_M_I1 Velocity of the IMU at the end of the integration.
  /// @param j_T_M_I0 Optional Jacobian for T_M_I0.
  /// @param j_v_M_I0 Optional Jacobian for v_M_I0.
  /// @param j_bg0 Optional Jacobian for bg0.
  /// @param j_ba0 Optional Jacobian for ba0.
  /// @param j_T_M_I1 Optional Jacobian for T_M_I1.
  /// @param j_v_M_I1 Optional Jacobian for v_M_I1.
  /// @return The inertial error vector.
  virtual VectorXd evaluateError(
      const Pose3d& T_M_I0, const Vector3d& v_M_I0, const Vector3d& bg0,
      const Vector3d& ba0, const Pose3d& T_M_I1, const Vector3d& v_M_I1,
      JacobianMatrixXd* j_T_M_I0 = nullptr,
      JacobianMatrixXd* j_v_M_I0 = nullptr, JacobianMatrixXd* j_bg0 = nullptr,
      JacobianMatrixXd* j_ba0 = nullptr, JacobianMatrixXd* j_T_M_I1 = nullptr,
      JacobianMatrixXd* j_v_M_I1 = nullptr) const;

  int getResidualDim() const override {
    if (isRotationOnly()) {
      return 3;
    } else {
      return 9;
    }
  }

  std::shared_ptr<ImuIntegration> getPreIntegration() const {
    return imu_pre_integration_;
  }

 protected:
  const VariableKey& key(VariableIdx input) const {
    return Base::getVariableKey(static_cast<int>(input));
  }

  void updateSqrtInfo() const;

  bool isRotationOnly() const;

 private:
  std::shared_ptr<ImuIntegration> imu_pre_integration_;
  mutable Eigen::MatrixXd sqrt_info_;  // Reproparation may change it

  double gyro_bias_repropagation_thr_;
  double acc_bias_repropagation_thr_;
  double regularization_;

  bool assume_constant_velocity_;
  bool assume_zero_rotation_;
  bool assume_zero_velocity_;
};

}  // namespace sk4slam
