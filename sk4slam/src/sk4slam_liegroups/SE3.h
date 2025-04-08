#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/SubGLn_rx_Rn.h"

namespace sk4slam {

namespace se3_internal {
struct SE3Impl;
}  // namespace se3_internal

template <typename Scalar>
class SE3
    : public SubGLn_rx_Rn<SO3<Scalar>, Rn<3, Scalar>, se3_internal::SE3Impl>::
          template Extension<SE3, Scalar> {
  using _Base =
      typename SubGLn_rx_Rn<SO3<Scalar>, Rn<3, Scalar>, se3_internal::SE3Impl>::
          template Extension<SE3, Scalar>;
  using _XYOnlyPerturbation = liegroup_internal::SubSpacePerturbation<
      typename Rn<3, Scalar>::LeftPerturbation, SubSpaceByAxes<0, 1>>;
  using _XYFixedPerturbation = liegroup_internal::SubSpacePerturbation<
      typename Rn<3, Scalar>::LeftPerturbation, SubSpaceByAxes<2>>;
  using _TranslationFixedPerturbation = liegroup_internal::SubSpacePerturbation<
      typename Rn<3, Scalar>::LeftPerturbation, SubSpaceByAxes<>>;
  template <int _heading_axis>
  using _YawXYFixedPerturbation = liegroup_internal::ProductPerturbation<
      SE3, typename SO3<Scalar>::template YawFixedPerturbation<_heading_axis>,
      _XYFixedPerturbation>;

 public:
  using _Base::_Base;
  SO3<Scalar>& rotation() {
    return _Base::linear();
  }
  const SO3<Scalar>& rotation() const {
    return _Base::linear();
  }

  // non-standard perturbations
  template <int _heading_axis>
  // using YawXYZFixedPerturbation = liegroup_internal::SubSpacePerturbation<
  //     _YawXYFixedPerturbation<_heading_axis>, SubSpaceByAxes<0, 1>>;
  using YawXYZFixedPerturbation = typename _Base::template AffinePerturbation<
      typename SO3<Scalar>::template YawFixedPerturbation<_heading_axis>,
      _TranslationFixedPerturbation>;
  using YawXYZOnlyPerturbation = typename _Base::template AffinePerturbation<
      typename SO3<Scalar>::YawOnlyPerturbation,
      typename Rn<3, Scalar>::LeftPerturbation>;
  using YawXYOnlyPerturbation = typename _Base::template AffinePerturbation<
      typename SO3<Scalar>::YawOnlyPerturbation, _XYOnlyPerturbation>;

  template <int _heading_axis>
  using YawXYZFixed =
      OptimizableManifold<SE3, YawXYZFixedPerturbation<_heading_axis>>;
  using YawXYZOnly = OptimizableManifold<SE3, YawXYZOnlyPerturbation>;

  using YawXYOnly = OptimizableManifold<SE3, YawXYOnlyPerturbation>;
};

using SE3d = SE3<double>;

/////////// Implementation /////////////

namespace se3_internal {
struct SE3Impl {
  template <typename SO3, typename R3>
  struct Impl;
};

template <typename SO3, typename R3>
struct SE3Impl::Impl {
  using DefaultSE3 = SubGLn_rx_Rn<SO3, R3, SE3Impl>;
  using LieAlgebraEndomorphism = typename DefaultSE3::LieAlgebraEndomorphism;
  using LieAlgebra = typename DefaultSE3::LieAlgebra;
  using Scalar = typename DefaultSE3::Scalar;

  template <typename SE3>
  static LieAlgebraEndomorphism Ad(const SE3& g) {
    LieAlgebraEndomorphism Adj;
    Adj.setZero();
    const Eigen::Matrix<Scalar, 3, 3>& R = g.linear().matrix();
    const Eigen::Matrix<Scalar, 3, 1>& t = g.translation();
    static const Eigen::Matrix<Scalar, 3, 3> Zero3x3 =
        Eigen::Matrix<Scalar, 3, 3>::Zero();
    // clang-format off
    // Adj <<             R,    Zero3x3,
    //         SO3::hat(t) * R,    R;
    // clang-format on
    Adj.template block<3, 3>(0, 0) = R;
    Adj.template block<3, 3>(3, 3) = R;
    Adj.template block<3, 3>(3, 0) = SO3::hat(t) * R;
    return Adj;
  }

  // Ad(g, X) = Ad(g) * X
  template <typename SE3>
  static LieAlgebra Ad(const SE3& g, const LieAlgebra& X) {
    const Eigen::Matrix<Scalar, 3, 3>& R = g.linear().matrix();
    const Eigen::Matrix<Scalar, 3, 1>& t = g.translation();
    LieAlgebra ret;
    ret.template head<3>() = R * X.template head<3>();
    ret.template tail<3>() =
        SO3::hat(t) * ret.template head<3>() + R * X.template tail<3>();
    return ret;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    LieAlgebraEndomorphism adj;
    adj.setZero();
    Eigen::Matrix<Scalar, 3, 1> theta = X.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> rho = X.template tail<3>();
    static const Eigen::Matrix<Scalar, 3, 3> Zero3x3 =
        Eigen::Matrix<Scalar, 3, 3>::Zero();
    // clang-format off
    // adj << SO3::hat(theta), Zero3x3,
    //          SO3::hat(rho), SO3::hat(theta);
    // clang-format on
    adj.template block<3, 3>(0, 0) = SO3::hat(theta);
    adj.template block<3, 3>(3, 3) = adj.template block<3, 3>(0, 0);
    adj.template block<3, 3>(3, 0) = SO3::hat(rho);
    return adj;
  }

  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    LieAlgebra X;
    X.template head<3>() = X1.template head<3>().cross(X2.template head<3>());
    X.template tail<3>() = X1.template head<3>().cross(X2.template tail<3>()) -
                           X2.template head<3>().cross(X1.template tail<3>());
    return X;
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    LieAlgebraEndomorphism J;
    J.setZero();
    // clang-format off
    // J <<  Jlxi,  Zero3x3,
    //         Q,  Jlxi;
    // clang-format on
    J.template block<3, 3>(0, 0) = SO3::Jl(X.template head<3>());
    J.template block<3, 3>(3, 3) = J.template block<3, 3>(0, 0);
    J.template block<3, 3>(3, 0) =
        calcQ(X.template head<3>(), X.template tail<3>());
    return J;
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Jl(-X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    Eigen::Matrix<Scalar, 3, 3> Q =
        calcQ(X.template head<3>(), X.template tail<3>());
    Eigen::Matrix<Scalar, 3, 3> invJlxi = SO3::invJl(X.template head<3>());
    static const Eigen::Matrix<Scalar, 3, 3> Zero3x3 =
        Eigen::Matrix<Scalar, 3, 3>::Zero();
    LieAlgebraEndomorphism invJ;
    // clang-format off
    invJ <<                   invJlxi,  Zero3x3,
              - invJlxi * Q * invJlxi,  invJlxi;
    // clang-format on
    return invJ;
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return invJl(-X);
  }

  static Eigen::Matrix<Scalar, 3, 3> calcV(const typename SO3::LieAlgebra& xi) {
    return SO3::Jl(xi);
  }

  static Eigen::Matrix<Scalar, 3, 3> calcVinv(
      const typename SO3::LieAlgebra& xi, const SO3* rotation) {
    return SO3::invJl(xi);
  }

 protected:
  static Eigen::Matrix<Scalar, 3, 3> calcQ(
      const Eigen::Matrix<Scalar, 3, 1>& xi,
      const Eigen::Matrix<Scalar, 3, 1>& eta) {
    using std::cos;
    using std::sin;
    using std::sqrt;

    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    const Scalar& kThree = Scalar(3.0);
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    Eigen::Matrix<Scalar, 3, 3> etax = SO3::hat(eta);
    Eigen::Matrix<Scalar, 3, 3> Q = kNum_0p5 * etax;

    Scalar theta2 = xi.squaredNorm();
    Scalar theta = sqrt(theta2);
    if (theta < eps) {
      return Q;
    } else {
      Scalar theta3 = theta2 * theta;
      Scalar theta4 = theta2 * theta2;
      Scalar theta5 = theta2 * theta3;
      Scalar sintheta = sin(theta);
      Scalar costheta = cos(theta);
      Eigen::Matrix<Scalar, 3, 3> hat_xi = SO3::hat(xi);
      Eigen::Matrix<Scalar, 3, 3> hat_xi_etax = hat_xi * etax;
      Eigen::Matrix<Scalar, 3, 3> etax_hat_xi = etax * hat_xi;
      Eigen::Matrix<Scalar, 3, 3> hat_xi_on_theta = hat_xi / theta;
      Eigen::Matrix<Scalar, 3, 3> hat_xi2_on_theta2 =
          hat_xi_on_theta * hat_xi_on_theta;
      hat_xi* hat_xi;
      Q += ((theta - sintheta) / theta3) *
           (hat_xi_etax + etax_hat_xi + hat_xi * etax_hat_xi);
      Q += ((theta2 + kNum_2 * costheta - kNum_2) / (kNum_2 * theta4)) *
           (hat_xi * hat_xi_etax + etax_hat_xi * hat_xi -
            kThree * hat_xi * etax_hat_xi);
      Q += ((kNum_2 * theta - kThree * sintheta + theta * costheta) /
            (kNum_2 * theta3)) *
           (hat_xi_etax * hat_xi2_on_theta2 + hat_xi2_on_theta2 * etax_hat_xi);
      return Q;
    }
  }
};

}  // namespace se3_internal

}  // namespace sk4slam
