
#include "sk4slam_backends/factors/inertial/imu_preintegration_factor.h"

#include "sk4slam_basic/string_helper.h"

namespace sk4slam {

bool ImuPreIntegrationFactor::isRotationOnly() const {
  // return imu_pre_integration_->isRotationOnly();
  bool is_rotation_only =
      (null_variable == getVariableKey(static_cast<int>(IMU_VEL0)));
  if (is_rotation_only) {
    // Invariants check
    ASSERT(null_variable == getVariableKey(static_cast<int>(IMU_ACC_BIAS0)));
    ASSERT(null_variable == getVariableKey(static_cast<int>(IMU_VEL1)));
  }

  return is_rotation_only;
}

void ImuPreIntegrationFactor::updateSqrtInfo() const {
  Eigen::MatrixXd process_noise_cov;
  if (isRotationOnly()) {
    process_noise_cov = imu_pre_integration_->getLatestResult()
                            .process_noise_cov.topLeftCorner<3, 3>();
  } else {
    process_noise_cov =
        imu_pre_integration_->getLatestResult().process_noise_cov;
    ASSERT(process_noise_cov.rows() == 9 && process_noise_cov.cols() == 9);
  }

  int n = process_noise_cov.rows();
  if (regularization_ > 0.0) {
    double scale = process_noise_cov.trace() / process_noise_cov.rows();
    scale = std::max(1.0, scale);
    process_noise_cov +=
        Eigen::MatrixXd::Identity(n, n) * (regularization_ * scale);
  }

  sqrt_info_ =
      process_noise_cov.llt().matrixL().solve(Eigen::MatrixXd::Identity(n, n));
}

VectorXd ImuPreIntegrationFactor::evaluateError(
    const Pose3d& T_M_I0, const Vector3d& v_M_I0, const Vector3d& bg0,
    const Vector3d& ba0, const Pose3d& T_M_I1, const Vector3d& v_M_I1,
    JacobianMatrixXd* j_T_M_I0, JacobianMatrixXd* j_v_M_I0,
    JacobianMatrixXd* j_bg0, JacobianMatrixXd* j_ba0,
    JacobianMatrixXd* j_T_M_I1, JacobianMatrixXd* j_v_M_I1) const {
  static const Vector3d gravity = Vector3d(0, 0, 9.81);
  Vector<6> bias_correction;
  bias_correction << bg0 - imu_pre_integration_->getGyroBias(),
      ba0 - imu_pre_integration_->getAccBias();
  if (isRotationOnly()) {
    bias_correction.tail<3>().setZero();
  }

  static constexpr bool never_repropagate = true;  // For debugging
  if (!never_repropagate &&
      (bias_correction.head<3>().lpNorm<Eigen::Infinity>() >
           gyro_bias_repropagation_thr_ ||
       bias_correction.tail<3>().lpNorm<Eigen::Infinity>() >
           acc_bias_repropagation_thr_)) {
    imu_pre_integration_->repropagate(bg0, ba0);

    updateSqrtInfo();
    bias_correction.setZero();
  }
  if (sqrt_info_.rows() == 0) {
    updateSqrtInfo();
  }
  if (isRotationOnly()) {
    ASSERT(sqrt_info_.rows() == 3);
  } else {
    ASSERT(sqrt_info_.rows() == 9);
  }

  const SO3d& R0 = T_M_I0.rotation();
  const Vector3d& p0 = T_M_I0.translation();
  const Vector3d& v0 = v_M_I0;
  const SO3d& R1 = T_M_I1.rotation();
  const Vector3d& p1 = T_M_I1.translation();
  const Vector3d& v1 = v_M_I1;

  const SO3d invR0 = R0.inverse();
  const Eigen::Matrix3d& invR0m = invR0.matrix();

  const ImuIntegration::Result& last_pre_integ =
      imu_pre_integration_->getLatestResult();
  const SO3d& pre_integ_R = last_pre_integ.state.R();
  const Vector3d& pre_integ_p = last_pre_integ.state.p();
  const Vector3d& pre_integ_v = last_pre_integ.state.v();
  double DT = imu_pre_integration_->timeWindow();

  Vector3d pre_integ_theta_correction =
      last_pre_integ.J_bias.block<3, 3>(0, 0) * bias_correction.head<3>();
  Vector3d error_theta;
  if (assume_zero_rotation_) {
    error_theta = SO3d::Log(R1.inverse() * R0);
  } else {
    SO3d corrected_pre_integ_R =
        SO3d::Exp(pre_integ_theta_correction) * pre_integ_R;
    error_theta = SO3d::Log(corrected_pre_integ_R * R1.inverse() * R0);
  }

  VectorXd error;

  Vector3d world_p_change, world_v_change;
  if (!isRotationOnly()) {
    world_p_change = p1 - p0 - v0 * DT + (0.5 * DT * DT) * gravity;
    world_v_change = v1 - v0 + DT * gravity;

    Vector3d error_p, error_v;
    if (assume_zero_velocity_) {
      error_p = invR0 * (-(p1 - p0));
      error_v = invR0 * (-v1);
      // error_v = invR0 * (0.5 * (v0 + v1));
    } else if (assume_constant_velocity_) {
      error_p = invR0 * (-(p1 - p0 - v0 * DT));
      error_v = invR0 * (-(v1 - v0));
    } else {
      Vector3d pre_integ_p_correction =
          last_pre_integ.J_bias.block<3, 6>(3, 0) * bias_correction;
      Vector3d pre_integ_v_correction =
          last_pre_integ.J_bias.block<3, 6>(6, 0) * bias_correction;
      Vector3d corrected_pre_integ_p = pre_integ_p + pre_integ_p_correction;
      Vector3d corrected_pre_integ_v = pre_integ_v + pre_integ_v_correction;

      error_p = corrected_pre_integ_p - invR0 * world_p_change;
      error_v = corrected_pre_integ_v - invR0 * world_v_change;
    }

    error.resize(9);
    error << error_theta, error_p, error_v;
  } else {
    error = error_theta;
  }

  VectorXd whitened_error = sqrt_info_ * error;
  LOGA(
      "ImuPreIntegration error (from %f to %f): %s, whitened error: %s",
      imu_pre_integration_->getInitialTime(),
      imu_pre_integration_->getLatestTime(), toStr(error.transpose()).c_str(),
      toStr(whitened_error.transpose()).c_str());

  JacobianMatrixXd invJr_invR0;
  if (j_T_M_I0 || j_T_M_I1) {
    invJr_invR0 = SO3d::invJr(error_theta) * invR0m;
  }

  if (j_T_M_I0) {
    ASSERT(j_T_M_I0->cols() == 6);
    ASSERT(j_T_M_I0->rows() >= 3);
    j_T_M_I0->block<3, 3>(0, 0) = invJr_invR0;
    j_T_M_I0->block<3, 3>(0, 3).setZero();

    if (!isRotationOnly()) {
      ASSERT(j_T_M_I0->rows() == 9);
      j_T_M_I0->block<3, 3>(3, 0) = invR0m * SO3d::hat(-world_p_change);
      j_T_M_I0->block<3, 3>(6, 0) = invR0m * SO3d::hat(-world_v_change);

      j_T_M_I0->block<3, 3>(3, 3) = invR0m;
      j_T_M_I0->block<3, 3>(6, 3).setZero();
    } else {
      ASSERT(j_T_M_I0->rows() == 3);
    }

    *j_T_M_I0 = sqrt_info_ * (*j_T_M_I0);
  }

  if (j_T_M_I1) {
    ASSERT(j_T_M_I1->cols() == 6);
    ASSERT(j_T_M_I1->rows() >= 3);
    j_T_M_I1->block<3, 3>(0, 0) = -invJr_invR0;
    j_T_M_I1->block<3, 3>(0, 3).setZero();

    if (!isRotationOnly()) {
      ASSERT(j_T_M_I1->rows() == 9);
      j_T_M_I1->block<3, 3>(3, 0).setZero();
      j_T_M_I1->block<3, 3>(6, 0).setZero();

      j_T_M_I1->block<3, 3>(3, 3) = -invR0m;
      j_T_M_I1->block<3, 3>(6, 3).setZero();
    } else {
      ASSERT(j_T_M_I1->rows() == 3);
    }
    *j_T_M_I1 = sqrt_info_ * (*j_T_M_I1);
  }

  if (j_bg0) {
    ASSERT(j_bg0->cols() == 3);
    ASSERT(j_bg0->rows() >= 3);
    j_bg0->block<3, 3>(0, 0) = SO3d::invJl(error_theta) *
                               SO3d::Jl(pre_integ_theta_correction) *
                               last_pre_integ.J_bias.block<3, 3>(0, 0);
    if (!isRotationOnly()) {
      ASSERT(j_bg0->rows() == 9);
      j_bg0->block<6, 3>(3, 0) = last_pre_integ.J_bias.block<6, 3>(3, 0);
    } else {
      ASSERT(j_bg0->rows() == 3);
    }
    *j_bg0 = sqrt_info_ * (*j_bg0);
  }

  if (!isRotationOnly()) {
    if (j_v_M_I0) {
      ASSERT(j_v_M_I0->cols() == 3 && j_v_M_I0->rows() == 9);
      j_v_M_I0->block<3, 3>(0, 0).setZero();
      j_v_M_I0->block<3, 3>(3, 0) = invR0m * DT;
      j_v_M_I0->block<3, 3>(6, 0) = invR0m;

      *j_v_M_I0 = sqrt_info_ * (*j_v_M_I0);
    }

    if (j_v_M_I1) {
      ASSERT(j_v_M_I1->cols() == 3 && j_v_M_I1->rows() == 9);
      j_v_M_I1->block<3, 3>(0, 0).setZero();
      j_v_M_I1->block<3, 3>(3, 0).setZero();
      j_v_M_I1->block<3, 3>(6, 0) = -invR0m;

      *j_v_M_I1 = sqrt_info_ * (*j_v_M_I1);
    }

    if (j_ba0) {
      ASSERT(j_ba0->cols() == 3 && j_ba0->rows() == 9);
      j_ba0->block<3, 3>(0, 0).setZero();
      j_ba0->block<6, 3>(3, 0) = last_pre_integ.J_bias.block<6, 3>(3, 3);

      *j_ba0 = sqrt_info_ * (*j_ba0);
    }
  } else {
    ASSERT(!j_v_M_I0);
    ASSERT(!j_v_M_I1);
    ASSERT(!j_ba0);
  }

  return whitened_error;
}

}  // namespace sk4slam
