#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SubGLn_rx_Rn.h"
#include "sk4slam_liegroups/sim_common.h"

namespace sk4slam {

namespace sim2_internal {
struct Sim2Impl;
}  // namespace sim2_internal

template <typename Scalar>
struct Sim2 : public SubGLn_rx_Rn<
                  Rp_x_SO2<Scalar>, Rn<2, Scalar>,
                  sim2_internal::Sim2Impl>::template Extension<Sim2, Scalar> {
  using Base = typename SubGLn_rx_Rn<
      Rp_x_SO2<Scalar>, Rn<2, Scalar>,
      sim2_internal::Sim2Impl>::template Extension<Sim2, Scalar>;
  using Base::Base;
  Scalar& scale() {
    return Base::linear().scale();
  }
  const Scalar& scale() const {
    return Base::linear().scale();
  }
  SO2<Scalar>& rotation() {
    return Base::linear().rotation();
  }
  const SO2<Scalar>& rotation() const {
    return Base::linear().rotation();
  }
};

using Sim2d = Sim2<double>;

/////////// Implementation /////////////

namespace sim2_internal {
struct Sim2Impl {
  template <typename Rp_x_SO2, typename R2>
  struct Impl;
};

template <typename Rp_x_SO2, typename R2>
struct Sim2Impl::Impl {
  using DefaultSim2 = SubGLn_rx_Rn<Rp_x_SO2, R2, Sim2Impl>;
  using LieAlgebraEndomorphism = typename DefaultSim2::LieAlgebraEndomorphism;
  using LieAlgebra = typename DefaultSim2::LieAlgebra;
  using Scalar = typename DefaultSim2::Scalar;

  template <typename Sim2>
  static LieAlgebraEndomorphism Ad(const Sim2& g) {
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Eigen::Matrix<Scalar, 2, 2>& M = g.linear().matrix();
    const Eigen::Matrix<Scalar, 2, 2>& R = g.linear().rotation().matrix();
    const Eigen::Matrix<Scalar, 2, 1>& t = g.translation();
    const Scalar& x = t[0];
    const Scalar& y = t[1];
    LieAlgebraEndomorphism Adj = LieAlgebraEndomorphism::Zero();
    const Scalar* Md = M.data();
    Scalar* Adjd = Adj.data();
    // clang-format off
    Adjd[0] = kNum_1;    /*0*/          /*0*/              /*0*/
        /*0*/      Adjd[5] = kNum_1;    /*0*/              /*0*/
    Adjd[2] = -x;  Adjd[6] = y;   Adjd[10] = Md[0];  Adjd[14] = Md[2];
    Adjd[3] = -y;  Adjd[7] = -x;  Adjd[11] = Md[1];  Adjd[15] = Md[3];
    // clang-format on
    return Adj;
  }

  // Ad(g, X) = Ad(g) * X
  template <typename Sim2>
  static LieAlgebra Ad(const Sim2& g, const LieAlgebra& X) {
    const Eigen::Matrix<Scalar, 2, 2>& M = g.linear().matrix();
    const Eigen::Matrix<Scalar, 2, 1>& t = g.translation();
    const Scalar& x = t[0];
    const Scalar& y = t[1];
    LieAlgebra ret = X;
    ret.template tail<2>() = M * X.template tail<2>();
    ret[2] += (-x) * X[0] + y * X[1];
    ret[3] += (-y) * X[0] + (-x) * X[1];
    return ret;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    const Scalar& x = X[2];
    const Scalar& y = X[3];
    LieAlgebraEndomorphism adj = LieAlgebraEndomorphism::Zero();
    Scalar* adjd = adj.data();
    // clang-format off
    adjd[2] = -x;  adjd[6] = y;   adjd[10] = X[0];  adjd[14] = -X[1];
    adjd[3] = -y;  adjd[7] = -x;  adjd[11] = X[1];  adjd[15] = X[0];
    // clang-format on
    return adj;
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    LieAlgebra X;
    X[0] = X[1] = kNum_0;
    X[2] = (-X1[2]) * X2[0] + X1[3] * X2[1] + X1[0] * X2[2] - X1[1] * X2[3];
    X[3] = (-X1[3]) * X2[0] + (-X1[2]) * X2[1] + X1[1] * X2[2] + X1[0] * X2[3];
    return X;
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    using std::abs;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    static const Eigen::Matrix<Scalar, 2, 2> I =
        Eigen::Matrix<Scalar, 2, 2>::Identity();
    Eigen::Matrix<Scalar, 2, 1> xi = X.template head<2>();
    const Scalar& sigma = X[0];
    const Scalar& theta = X[1];
    const Scalar& x = X[2];
    const Scalar& y = X[3];
    Eigen::Matrix<Scalar, 2, 2> H;
    Scalar* Hd = H.data();
    Hd[0] = -x;
    Hd[2] = y;
    Hd[1] = -y;
    Hd[3] = -x;
    Eigen::Matrix<Scalar, 2, 2> V = calcV(xi);

    // Compute Q.
    // Note ad_xi = 0, so
    //        H_k = hat_xi^{k-1} * H
    // and we get a closed form Q for Sim2:
    //        Q = \sum_{k=1}^{\infty} [H_k / (k+1)!]
    //          = (V - I) * hat_xi^{-1} * H
    Eigen::Matrix<Scalar, 2, 2> Q;
    const Scalar sigma2 = sigma * sigma;
    const Scalar theta2 = theta * theta;
    const Scalar sigma2_plus_theta2 = sigma2 + theta2;
    if (abs(sigma) < eps) {
      if (abs(theta) < eps) {
        Q = kNum_0p5 * H;
      } else {
        const Eigen::Matrix<Scalar, 2, 2> hax_xi = Rp_x_SO2::hat(xi);
        Q = (I - V) * hax_xi * H / theta2;
      }
    } else {
      const Eigen::Matrix<Scalar, 2, 2> inv_hat_xi =
          // Rp_x_SO2::hat(xi).inverse();
          Rp_x_SO2::hat(
              Eigen::Matrix<Scalar, 2, 1>(xi[0], -xi[1]) / sigma2_plus_theta2);
      Q = (V - I) * inv_hat_xi * H;
    }
    LieAlgebraEndomorphism J = LieAlgebraEndomorphism::Identity();
    J.template block<2, 2>(2, 0) = Q;
    J.template block<2, 2>(2, 2) = V;
    return J;
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Jl(-X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    using std::abs;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    static const Eigen::Matrix<Scalar, 2, 2> I =
        Eigen::Matrix<Scalar, 2, 2>::Identity();
    Eigen::Matrix<Scalar, 2, 1> xi = X.template head<2>();
    const Scalar& sigma = X[0];
    const Scalar& theta = X[1];
    const Scalar& x = X[2];
    const Scalar& y = X[3];
    Eigen::Matrix<Scalar, 2, 2> H;
    Scalar* Hd = H.data();
    Hd[0] = -x;
    Hd[2] = y;
    Hd[1] = -y;
    Hd[3] = -x;
    Eigen::Matrix<Scalar, 2, 2> invV = calcVinv(xi, nullptr);

    // Compute invV * Q.
    // Note ad_xi = 0, that's why we can compute invV * Q analytically
    // for Sim2.
    Eigen::Matrix<Scalar, 2, 2> invVQ;
    const Scalar sigma2 = sigma * sigma;
    const Scalar theta2 = theta * theta;
    const Scalar sigma2_plus_theta2 = sigma2 + theta2;
    if (abs(sigma) < eps) {
      if (abs(theta) < eps) {
        invVQ = kNum_0p5 * H;
      } else {
        const Eigen::Matrix<Scalar, 2, 2> hax_xi = Rp_x_SO2::hat(xi);
        invVQ = (invV - I) * hax_xi * H / theta2;
      }
    } else {
      const Eigen::Matrix<Scalar, 2, 2> inv_hat_xi =
          // Rp_x_SO2::hat(xi).inverse();
          Rp_x_SO2::hat(
              Eigen::Matrix<Scalar, 2, 1>(xi[0], -xi[1]) / sigma2_plus_theta2);
      invVQ = (I - invV) * inv_hat_xi * H;
    }

    LieAlgebraEndomorphism invJ = LieAlgebraEndomorphism::Identity();
    invJ.template block<2, 2>(2, 0) = -invVQ;
    invJ.template block<2, 2>(2, 2) = invV;
    return invJ;
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return invJl(-X);
  }

  static Eigen::Matrix<Scalar, 2, 2> calcV(
      const typename Rp_x_SO2::LieAlgebra& xi) {
    const Scalar theta = xi[1];
    Scalar A, B, C;
    sim_common::calcABCForV<Scalar>(xi[0], theta, &A, &B, &C);
    Eigen::Matrix<Scalar, 2, 2> V;
    Scalar* Vd = V.data();
    Vd[0] = Vd[3] = C - B * theta * theta;
    Vd[1] = A * theta;
    Vd[2] = -Vd[1];
    return V;
  }

  static Eigen::Matrix<Scalar, 2, 2> calcVinv(
      const typename Rp_x_SO2::LieAlgebra& xi, const Rp_x_SO2* linear) {
    const Scalar* known_scale = linear ? &linear->scale() : nullptr;
    const Scalar theta = xi[1];
    Scalar A, B, C;
    sim_common::calcABCForVinv<Scalar>(xi[0], theta, known_scale, &A, &B, &C);
    Eigen::Matrix<Scalar, 2, 2> invV;
    Scalar* invVd = invV.data();
    invVd[0] = invVd[3] = C - B * theta * theta;
    invVd[1] = A * theta;
    invVd[2] = -invVd[1];
    return invV;
  }
};
}  // namespace sim2_internal

}  // namespace sk4slam
