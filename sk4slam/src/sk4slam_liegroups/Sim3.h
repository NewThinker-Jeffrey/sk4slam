#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SubGLn_rx_Rn.h"
#include "sk4slam_liegroups/sim_common.h"

namespace sk4slam {

namespace sim3_internal {
// Declared the default implementation template for SubGLn_rx_Rn
template <typename SubGLn, typename Rn, typename ApproximationOptions>
struct Sim3Impl;

struct Sim3Accurate {
  template <typename Scalar>
  struct ApproximationOptions {
    using Constants = liegroup::Constants<Scalar>;
    static inline const Scalar kEpsForJacobian =
        liegroup::Constants<Scalar>::kEps;
    static constexpr int kMaxOrderForJacobian = -1;
  };
  template <typename SubGLn, typename Rn>
  struct Impl : public Sim3Impl<
                    SubGLn, Rn, ApproximationOptions<typename SubGLn::Scalar>> {
  };
};

template <int _jacobian_order>
struct Sim3UpToOrder {
  template <typename Scalar>
  struct ApproximationOptions
      : public Sim3Accurate::ApproximationOptions<Scalar> {
    static constexpr int kMaxOrderForJacobian = _jacobian_order;
  };

  template <typename SubGLn, typename Rn>
  class Impl : public Sim3Impl<
                   SubGLn, Rn, ApproximationOptions<typename SubGLn::Scalar>> {
  };
};

template <int _jacobian_order>
struct Sim3JacobianOrder {
  static constexpr int value = _jacobian_order;
};
}  // namespace sim3_internal

template <typename Scalar, typename ImplType = sim3_internal::Sim3Accurate>
struct Sim3 : public SubGLn_rx_Rn<Rp_x_SO3<Scalar>, Rn<3, Scalar>, ImplType>::
                  template Extension<Sim3, Scalar, ImplType> {
  using Base =
      typename SubGLn_rx_Rn<Rp_x_SO3<Scalar>, Rn<3, Scalar>, ImplType>::
          template Extension<Sim3, Scalar, ImplType>;
  using Base::Base;
  Scalar& scale() {
    return Base::linear().scale();
  }
  const Scalar& scale() const {
    return Base::linear().scale();
  }
  SO3<Scalar>& rotation() {
    return Base::linear().rotation();
  }
  const SO3<Scalar>& rotation() const {
    return Base::linear().rotation();
  }
};

using Sim3d = Sim3<double>;

template <typename Scalar, typename Sim3JacobianOrder>
struct Sim3UpToOrder_
    : public SubGLn_rx_Rn<
          Rp_x_SO3<Scalar>, Rn<3, Scalar>,
          sim3_internal::Sim3UpToOrder<Sim3JacobianOrder::value>>::
          template Extension<Sim3UpToOrder_, Scalar, Sim3JacobianOrder> {
  using Base = typename SubGLn_rx_Rn<
      Rp_x_SO3<Scalar>, Rn<3, Scalar>,
      sim3_internal::Sim3UpToOrder<Sim3JacobianOrder::value>>::
      template Extension<Sim3UpToOrder_, Scalar, Sim3JacobianOrder>;
  using Base::Base;
  Scalar& scale() {
    return Base::linear().scale();
  }
  const Scalar& scale() const {
    return Base::linear().scale();
  }
  SO3<Scalar>& rotation() {
    return Base::linear().rotation();
  }
  const SO3<Scalar>& rotation() const {
    return Base::linear().rotation();
  }
};

template <typename Scalar, int _jacobian_order>
using Sim3UpToOrder =
    Sim3UpToOrder_<Scalar, sim3_internal::Sim3JacobianOrder<_jacobian_order>>;

template <int _jacobian_order>
using Sim3dUpToOrder = Sim3UpToOrder<double, _jacobian_order>;

/////////// Implementation /////////////

namespace sim3_internal {

template <typename Rp_x_SO3, typename R3, typename ApproximationOptions>
struct Sim3Impl {
  using DefaultSim3 = SubGLn_rx_Rn<Rp_x_SO3, R3>;
  using LieAlgebraEndomorphism = typename DefaultSim3::LieAlgebraEndomorphism;
  using LieAlgebra = typename DefaultSim3::LieAlgebra;
  using Scalar = typename DefaultSim3::Scalar;

  template <typename Sim3>
  static LieAlgebraEndomorphism Ad(const Sim3& g) {
    using Helper = MatrixGroupHelper<Rp_x_SO3>;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    static const Eigen::Matrix<Scalar, 4, 3> Zero_corner =
        Eigen::Matrix<Scalar, 4, 3>::Zero();
    const Eigen::Matrix<Scalar, 3, 3>& M = g.linear().matrix();
    const Eigen::Matrix<Scalar, 3, 3>& R = g.linear().rotation().matrix();
    const Eigen::Matrix<Scalar, 3, 1>& t = g.translation();
    LieAlgebraEndomorphism Adj = LieAlgebraEndomorphism::Zero();
    Adj(0, 0) = kNum_1;
    Adj.template block<3, 1>(4, 0) = -t;
    Adj.template block<3, 3>(1, 1) = R;
    Adj.template block<3, 3>(4, 1) = SO3<Scalar>::hat(t) * R;
    Adj.template block<3, 3>(4, 4) = M;
    return Adj;
  }

  // Ad(g, X) = Ad(g) * X
  template <typename Sim3>
  static LieAlgebra Ad(const Sim3& g, const LieAlgebra& X) {
    const Eigen::Matrix<Scalar, 3, 3>& M = g.linear().matrix();
    const Eigen::Matrix<Scalar, 3, 1>& t = g.translation();
    typename Rp_x_SO3::LieAlgebra ret_head =
        Rp_x_SO3::Ad(g.linear(), X.template head<4>());
    Eigen::Matrix<Scalar, 3, 3> hat_ret_head = Rp_x_SO3::hat(ret_head);
    LieAlgebra ret;
    ret.template head<4>() = ret_head;
    ret.template tail<3>() = hat_ret_head * (-t) + M * X.template tail<3>();
    return ret;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    using Helper = MatrixGroupHelper<Rp_x_SO3>;
    static const Eigen::Matrix<Scalar, 4, 3> Zero_corner =
        Eigen::Matrix<Scalar, 4, 3>::Zero();
    typename Rp_x_SO3::LieAlgebra xi = X.template head<4>();
    Eigen::Matrix<Scalar, 3, 1> eta = X.template tail<3>();
    LieAlgebraEndomorphism adj = LieAlgebraEndomorphism::Zero();
    adj.template block<3, 1>(4, 0) = -eta;
    adj.template block<3, 3>(1, 1) = SO3<Scalar>::hat(xi.template tail<3>());
    adj.template block<3, 3>(4, 1) = SO3<Scalar>::hat(eta);
    adj.template block<3, 3>(4, 4) = adj.template block<3, 3>(1, 1);
    adj(4, 4) = adj(5, 5) = adj(6, 6) = xi(0);
    return adj;
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    using Helper = MatrixGroupHelper<Rp_x_SO3>;
    LieAlgebra X;
    X.template head<4>() =
        Rp_x_SO3::bracket(X1.template head<4>(), X2.template head<4>());
    const Eigen::Matrix<Scalar, 3, 3>& hat_xi_1 =
        Rp_x_SO3::hat(X1.template head<4>());
    const Eigen::Matrix<Scalar, 3, 3>& hat_xi_2 =
        Rp_x_SO3::hat(X2.template head<4>());
    X.template tail<3>() =
        hat_xi_1 * X2.template tail<3>() - hat_xi_2 * X1.template tail<3>();
    return X;
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    using Helper = MatrixGroupHelper<Rp_x_SO3>;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    typename Rp_x_SO3::LieAlgebra xi = X.template head<4>();
    Eigen::Matrix<Scalar, 3, 1> eta = X.template tail<3>();
    LieAlgebraEndomorphism J = LieAlgebraEndomorphism::Zero();
    J(0, 0) = kNum_1;
    J.template block<3, 3>(1, 1) = SO3<Scalar>::Jl(xi.template tail<3>());
    J.template block<3, 4>(4, 0) = calcQ(xi, eta);
    J.template block<3, 3>(4, 4) = calcV(xi);
    return J;
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Jl(-X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    using Helper = MatrixGroupHelper<Rp_x_SO3>;
    typename Rp_x_SO3::LieAlgebra xi = X.template head<4>();
    Eigen::Matrix<Scalar, 3, 1> eta = X.template tail<3>();
    Eigen::Matrix<Scalar, 4, 4> invJl_xi = Rp_x_SO3::invJl(xi);
    Eigen::Matrix<Scalar, 3, 4> Q = calcQ(xi, eta);
    static const Eigen::Matrix<Scalar, 4, 3> Zero_corner =
        Eigen::Matrix<Scalar, 4, 3>::Zero();
    Eigen::Matrix<Scalar, 3, 3> invV_xi = calcVinv(xi, nullptr);
    LieAlgebraEndomorphism invJ;
    // clang-format off
    invJ <<                 invJl_xi,  Zero_corner,
            - invV_xi * Q * invJl_xi,  invV_xi;
    // clang-format on
    return invJ;
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return invJl(-X);
  }

  static Eigen::Matrix<Scalar, 3, 3> calcV(
      const typename Rp_x_SO3::LieAlgebra& xi) {
    const Eigen::Matrix<Scalar, 3, 1> omega = xi.template tail<3>();
    const Scalar theta = omega.norm();
    Scalar A, B, C;
    sim_common::calcABCForV<Scalar>(xi[0], theta, &A, &B, &C);
    Eigen::Matrix<Scalar, 3, 3> V =
        SO3<Scalar>::hat(A * omega) + (B * omega) * omega.transpose();
    Scalar* Vd = V.data();
    const Scalar diag = C - B * theta * theta;
    // clang-format off
    Vd[0] += diag; Vd[4] += diag; Vd[8] += diag;
    // clang-format on
    return V;
  }

  static Eigen::Matrix<Scalar, 3, 3> calcVinv(
      const typename Rp_x_SO3::LieAlgebra& xi, const Rp_x_SO3* linear) {
    const Scalar* known_scale = linear ? &linear->scale() : nullptr;
    const Eigen::Matrix<Scalar, 3, 1> omega = xi.template tail<3>();
    const Scalar theta = omega.norm();
    Scalar A, B, C;
    sim_common::calcABCForVinv<Scalar>(xi[0], theta, known_scale, &A, &B, &C);
    Eigen::Matrix<Scalar, 3, 3> invV =
        SO3<Scalar>::hat(A * omega) + (B * omega) * omega.transpose();
    Scalar* invVd = invV.data();
    const Scalar diag = C - B * theta * theta;
    // clang-format off
    invVd[0] += diag; invVd[4] += diag; invVd[8] += diag;
    // clang-format on
    return invV;
  }

 protected:
  static Eigen::Matrix<Scalar, 3, 4> calcQ(
      const typename Rp_x_SO3::LieAlgebra& xi,
      const Eigen::Matrix<Scalar, 3, 1>& eta) {
    const int& max_order = ApproximationOptions::kMaxOrderForJacobian;
    const Scalar& eps = ApproximationOptions::kEpsForJacobian;
    static const Eigen::Matrix<Scalar, 3, 1> ZeroVec3 =
        Eigen::Matrix<Scalar, 3, 1>::Zero();
    const Eigen::Matrix<Scalar, 3, 3>& hat_xi = Rp_x_SO3::hat(xi);

    Eigen::Matrix<Scalar, 3, 4> H1;
    H1.col(0) = -eta;
    H1.template block<3, 3>(0, 1) = SO3<Scalar>::hat(eta);
    Eigen::Matrix<Scalar, 3, 4> Hk = H1;

    // const Eigen::Matrix<Scalar, 4, 4>& ad_xi = Rp_x_SO3::ad(xi);
    // Eigen::Matrix<Scalar, 3, 4> H1_adxi_pow_km1 = H1;

    const Eigen::Matrix<Scalar, 3, 3>& ad_omega =
        SO3<Scalar>::ad(xi.template tail<3>());
    Eigen::Matrix<Scalar, 3, 3> H1_adxi_pow_km1_nonzeros =
        H1.template block<3, 3>(0, 1);

    Scalar denominator_k =
        liegroup::Constants<Scalar>::kNum_2;  // denominator_k = 1/(k+1)!
    Eigen::Matrix<Scalar, 3, 4> Q = Hk / denominator_k;
    int max_iter;
    if (max_order < 0) {
      max_iter = std::numeric_limits<int>::max();
    } else {
      max_iter = max_order;
    }

    int k;
    for (k = 2; k < max_iter; k++) {
      const auto& H_km1 = Hk;

      // H1_adxi_pow_km1 *= ad_xi;
      // Hk = hat_xi * H_km1 + H1_adxi_pow_km1;

      H1_adxi_pow_km1_nonzeros *= ad_omega;
      Hk = hat_xi * H_km1;
      Hk.template block<3, 3>(0, 1) += H1_adxi_pow_km1_nonzeros;

      denominator_k *= Scalar(k + 1);
      auto delta = Hk / denominator_k;
      Q += delta;
      // if (max_order < 0) {
      Scalar max_abs = delta.array().abs().maxCoeff();
      if (max_abs < eps) {
        break;
      }
      // }
    }
    LOGA("Sim3::calcQ order: %d", k);
    return Q;
  }
};

}  // namespace sim3_internal

}  // namespace sk4slam
