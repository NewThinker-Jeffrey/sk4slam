#pragma once

#include <Eigen/Geometry>

#include "sk4slam_basic/logging.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_pose/pose.h"

namespace sk4slam {

template <
    typename Perturbation = Pose3d::AffineRightPerturbation,
    typename Scalar = double>
struct Pose3WithCov;

}  // namespace sk4slam

template <typename Perturbation, typename Scalar>
sk4slam::Pose3WithCov<Perturbation, Scalar> operator*(
    const typename sk4slam::Pose3WithCov<Perturbation, Scalar>::Pose& lhs,
    const sk4slam::Pose3WithCov<Perturbation, Scalar>& rhs);

namespace sk4slam {

template <typename _Perturbation, typename _Scalar>
struct Pose3WithCov {
  using Scalar = _Scalar;
  using Perturbation = _Perturbation;
  using Pose = Pose3<Scalar>;
  using Rotation = SO3<Scalar>;

  static_assert(
      std::is_same_v<Perturbation, typename Pose::AffineLeftPerturbation> ||
          std::is_same_v<
              Perturbation, typename Pose::AffineRightPerturbation> ||
          std::is_same_v<Perturbation, typename Pose::LeftPerturbation> ||
          std::is_same_v<Perturbation, typename Pose::RightPerturbation>,
      "Perturbation must be one of AffineLeftPerturbation / "
      "AffineRightPerturbation / LeftPerturbation / "
      "RightPerturbation");
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;
  using Matrix12 = Eigen::Matrix<Scalar, 12, 12>;
  using Matrix6x12 = Eigen::Matrix<Scalar, 6, 12>;
  using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

  explicit Pose3WithCov(
      const Matrix3& R_in = Matrix3::Identity(),
      const Vector3& p_in = Vector3::Zero(),
      const Matrix6& cov_in = Matrix6::Identity())
      : pose_(R_in, p_in), cov_(cov_in) {}

  explicit Pose3WithCov(
      const Isometry3& pose, const Matrix6& cov_in = Matrix6::Identity())
      : pose_(pose.linear(), pose.translation()), cov_(cov_in) {}

  explicit Pose3WithCov(
      const Pose& pose, const Matrix6& cov_in = Matrix6::Identity())
      : pose_(pose), cov_(cov_in) {}

  template <typename OtherPerturbation>
  Pose3WithCov(                                                // NOLINT
      const Pose3WithCov<OtherPerturbation, Scalar>& other) {  // NOLINT
    pose_ = other.pose();
    setCov<OtherPerturbation>(other.cov());
  }

  template <typename OtherPerturbation>
  Pose3WithCov& operator=(
      const Pose3WithCov<OtherPerturbation, Scalar>& other) {
    pose_ = other.pose();
    setCov<OtherPerturbation>(other.cov());
    return *this;
  }

  Pose& pose() {
    return pose_;
  }

  const Pose& pose() const {
    return pose_;
  }

  Rotation& rotation() {
    return pose_.rotation();
  }

  const Rotation& rotation() const {
    return pose_.rotation();
  }

  Vector3& translation() {
    return pose_.translation();
  }

  const Vector3& translation() const {
    return pose_.translation();
  }

  Matrix6& cov() {
    return cov_;
  }

  template <typename TargetPerturbation = Perturbation>
  decltype(auto) cov() const {
    if constexpr (std::is_same_v<TargetPerturbation, Perturbation>) {
      return cov_;
    } else {
      Matrix6 J =
          pose_
              .template convertPerturbation<Perturbation, TargetPerturbation>();
      return Matrix6(J * cov_ * J.transpose());
    }
  }

  /// Returns the covariance matrix in {p q} order instead of {q p}
  template <typename TargetPerturbation = Perturbation>
  Matrix6 pqOrderCov() const {
    const Matrix6& cov = this->cov<TargetPerturbation>();
    Matrix6 pq_order_cov;
    pq_order_cov.template block<3, 3>(0, 0) = cov.template block<3, 3>(3, 3);
    pq_order_cov.template block<3, 3>(0, 3) = cov.template block<3, 3>(3, 0);
    pq_order_cov.template block<3, 3>(3, 0) = cov.template block<3, 3>(0, 3);
    pq_order_cov.template block<3, 3>(3, 3) = cov.template block<3, 3>(0, 0);
    return pq_order_cov;
  }

  template <
      typename SourcePerturbation = Perturbation, typename MatrixXpr = Matrix6>
  void setCov(const MatrixXpr& cov) {
    if constexpr (std::is_same_v<SourcePerturbation, Perturbation>) {
      cov_ = cov;
    } else {
      auto J =
          pose_
              .template convertPerturbation<SourcePerturbation, Perturbation>();
      cov_ = J * cov * J.transpose();
    }
  }

  Isometry3 toIsometry() const {
    Isometry3 ret(rotation().matrix());
    ret.translation() = translation();
    return ret;
  }

  Pose3WithCov operator*(const Pose3WithCov& rhs) const {
    Pose3WithCov result;
    Matrix6 J_lhs, J_rhs;
    perturbation()->Multiply(pose_, rhs.pose_, &result.pose_, &J_lhs, &J_rhs);
    result.cov_ =
        J_lhs * cov_ * J_lhs.transpose() + J_rhs * rhs.cov_ * J_rhs.transpose();
    return result;
  }

  Pose3WithCov operator*(const Pose& rhs) const {
    Pose3WithCov result;
    Matrix6 J_lhs;
    perturbation()->Multiply(pose_, rhs, &result.pose_, &J_lhs);
    result.cov_ = J_lhs * cov_ * J_lhs.transpose();
    return result;
  }

  Pose3WithCov inverse() const {
    Pose3WithCov result;
    Matrix6 J_inv;
    perturbation()->Inverse(pose_, &result.pose_, &J_inv);
    result.cov_ = J_inv * cov_ * J_inv.transpose();
    return result;
  }

  template <class SourcePerturbation>
  static Matrix12 ConvertJointPoseCov(
      const Pose& pose0, const Pose& pose1,
      const Matrix12& src_joint_pose_cov) {
    Matrix6 J0 =
        pose0.template convertPerturbation<SourcePerturbation, Perturbation>();
    Matrix6 J1 =
        pose1.template convertPerturbation<SourcePerturbation, Perturbation>();
    Matrix12 joint_pose_cov;
    joint_pose_cov.template block<6, 6>(0, 0) =
        J0 * src_joint_pose_cov.template block<6, 6>(0, 0) * J0.transpose();
    joint_pose_cov.template block<6, 6>(0, 6) =
        J0 * src_joint_pose_cov.template block<6, 6>(0, 6) * J1.transpose();
    joint_pose_cov.template block<6, 6>(6, 0) =
        joint_pose_cov.template block<6, 6>(0, 6).transpose();
    joint_pose_cov.template block<6, 6>(6, 6) =
        J1 * src_joint_pose_cov.template block<6, 6>(6, 6) * J1.transpose();
    return joint_pose_cov;
  }

  static Pose3WithCov RightPoseDelta(
      const Pose& pose0, const Pose& pose1, const Matrix12& joint_pose_cov) {
    Pose3WithCov result;
    Matrix6x12 J_6x12;
    auto J0 = J_6x12.template leftCols<6>();
    auto J1 = J_6x12.template rightCols<6>();
    perturbation()->RightDelta(pose0, pose1, &result.pose_, &J0, &J1);
    result.cov_ = J_6x12 * joint_pose_cov * J_6x12.transpose();
    return result;
  }

  static Pose3WithCov LeftPoseDelta(
      const Pose& pose0, const Pose& pose1, const Matrix12& joint_pose_cov) {
    Pose3WithCov result;
    Matrix6x12 J_6x12;
    auto J0 = J_6x12.template leftCols<6>();
    auto J1 = J_6x12.template rightCols<6>();
    perturbation()->LeftDelta(pose0, pose1, &result.pose_, &J0, &J1);
    result.cov_ = J_6x12 * joint_pose_cov * J_6x12.transpose();
    return result;
  }

 private:
  template <typename __Perturbation, typename __Scalar>
  friend Pose3WithCov<__Perturbation, __Scalar>(::operator*)(
      const typename Pose3WithCov<__Perturbation, __Scalar>::Pose& lhs,
      const Pose3WithCov<__Perturbation, __Scalar>& rhs);

  Pose pose_;

  /// @brief The covariance matrix of the pose.
  /// @details The top-left 3x3 block is the covariance of the rotation,
  /// and the bottom-right 3x3 block is the covariance of the translation.
  /// The other two 3x3 blocks are the cross-covariance between the rotation
  /// and translation.
  Matrix6 cov_;

  static const Perturbation* perturbation() {
    return Perturbation::defaultInstance();
  }
};

}  // namespace sk4slam

template <typename Perturbation, typename Scalar>
sk4slam::Pose3WithCov<Perturbation, Scalar> operator*(
    const typename sk4slam::Pose3WithCov<Perturbation, Scalar>::Pose& lhs,
    const sk4slam::Pose3WithCov<Perturbation, Scalar>& rhs) {
  using Pose3WithCov = sk4slam::Pose3WithCov<Perturbation, Scalar>;
  using Matrix6 = typename Pose3WithCov::Matrix6;

  Pose3WithCov result;
  Matrix6 J_rhs, *null_j_lhs = nullptr;
  Pose3WithCov::perturbation()->Multiply(
      lhs, rhs.pose(), &result.pose(), null_j_lhs, &J_rhs);
  result.cov_ = J_rhs * rhs.cov_ * J_rhs.transpose();
  return result;
}
