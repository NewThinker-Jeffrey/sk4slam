#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/SO2.h"
#include "sk4slam_liegroups/SubGLn_rx_Rn.h"

namespace sk4slam {

namespace se2_internal {
struct SE2Impl;
}  // namespace se2_internal

template <typename Scalar>
struct SE2
    : public SubGLn_rx_Rn<SO2<Scalar>, Rn<2, Scalar>, se2_internal::SE2Impl>::
          template Extension<SE2, Scalar> {
  using Base =
      typename SubGLn_rx_Rn<SO2<Scalar>, Rn<2, Scalar>, se2_internal::SE2Impl>::
          template Extension<SE2, Scalar>;
  using Base::Base;

  SO2<Scalar>& rotation() {
    return Base::linear();
  }
  const SO2<Scalar>& rotation() const {
    return Base::linear();
  }
};

using SE2d = SE2<double>;

/////////// Implementation /////////////

namespace se2_internal {

struct SE2Impl {
  template <typename SO2, typename R2>
  struct Impl;
};

template <typename SO2, typename R2>
struct SE2Impl::Impl {
  using DefaultSE2 = SubGLn_rx_Rn<SO2, R2, SE2Impl>;
  using LieAlgebraEndomorphism = typename DefaultSE2::LieAlgebraEndomorphism;
  using LieAlgebra = typename DefaultSE2::LieAlgebra;
  using Scalar = typename DefaultSE2::Scalar;

  template <typename SE2>
  static LieAlgebraEndomorphism Ad(const SE2& g) {
    LieAlgebraEndomorphism Adj;
    Adj.setZero();
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Eigen::Matrix<Scalar, 2, 2>& R = g.linear().matrix();
    const Eigen::Matrix<Scalar, 2, 1>& t = g.translation();
    static const Eigen::Matrix<Scalar, 1, 2> Zero2x2 =
        Eigen::Matrix<Scalar, 1, 2>::Zero();
    // clang-format off
    // Adj <<             1,    Zero1x2,
    //                 t[1],    R.row(0);
    //                -t[0],    R.row(1);
    // clang-format on
    Scalar* Adjd = Adj.data();
    Adjd[0] = kNum_1;
    Adjd[1] = t[1];
    Adjd[2] = -t[0];
    Adj.template block<2, 2>(1, 1) = R;
    return Adj;
  }

  // Ad(g, X) = Ad(g) * X
  template <typename SE2>
  static LieAlgebra Ad(const SE2& g, const LieAlgebra& X) {
    const Eigen::Matrix<Scalar, 2, 2>& R = g.linear().matrix();
    const Eigen::Matrix<Scalar, 2, 1>& t = g.translation();
    LieAlgebra ret;
    ret[0] = X[0];
    ret.template tail<2>() = R * X.template tail<2>();
    ret[1] += t[1] * X[0];
    ret[2] -= t[0] * X[0];
    return ret;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    LieAlgebraEndomorphism adj;
    adj.setZero();
    const Scalar& theta = X[0];
    // clang-format off
    // adj <<            0,     0,   0,
    //                X[2],     0,   -theta,
    //               -X[1], theta,   0,
    // clang-format on

    // Note: adj^3 = - theta^2 * adj

    Scalar* adjd = adj.data();
    adjd[1] = X[2];
    adjd[2] = -X[1];
    adjd[5] = theta;
    adjd[7] = -theta;
    return adj;
  }

  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    LieAlgebra X;
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    Scalar* Xd = X.data();
    Xd[0] = kNum_0;
    Xd[1] = X1[2] * X2[0] - X1[0] * X2[2];
    Xd[2] = -X1[1] * X2[0] + X1[0] * X2[1];
    return X;
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    using std::abs;
    using std::cos;
    using std::sin;
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar& theta = X[0];

    // Note adX^3 = - theta^2 * adX, similar to SO3::Jl

    LieAlgebraEndomorphism J;
    Scalar* Jd = J.data();
    if (abs(theta) > eps) {
      const Scalar sintheta_on_theta = sin(theta) / theta;
      const Scalar one_minus_sintheta_on_theta = kNum_1 - sintheta_on_theta;
      const Scalar x = X[1] / theta;
      const Scalar y = X[2] / theta;
      Jd[4] = sintheta_on_theta;
      Jd[5] = (kNum_1 - cos(theta)) / theta;
      Jd[1] = Jd[5] * y + one_minus_sintheta_on_theta * x;
      Jd[2] = -Jd[5] * x + one_minus_sintheta_on_theta * y;
    } else {
      Jd[4] = kNum_1;
      Jd[5] = kNum_0p5 * theta;
      Jd[1] = kNum_0p5 * X[2];
      Jd[2] = -kNum_0p5 * X[1];
    }
    Jd[7] = -Jd[5];
    Jd[8] = Jd[4];

    Jd[0] = kNum_1;
    Jd[3] = kNum_0;
    Jd[6] = kNum_0;
    return J;
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Jl(-X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    using std::abs;
    using std::cos;
    using std::sin;
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar& theta = X[0];

    // Note adX^3 = - theta^2 * adX, similar to SO3::Jl

    LieAlgebraEndomorphism invJ;
    Scalar* invJd = invJ.data();

    if (abs(theta) > eps) {
      const Scalar theta_on_sintheta = theta / sin(theta);
      const Scalar x = X[1] / theta;
      const Scalar y = X[2] / theta;
      invJd[4] = kNum_0p5 * (cos(theta) + kNum_1) * theta_on_sintheta;
      invJd[5] = -kNum_0p5 * theta;

      const Scalar one_minus_invJ11 = kNum_1 - invJd[4];
      invJd[1] = invJd[5] * y + one_minus_invJ11 * x;
      invJd[2] = -invJd[5] * x + one_minus_invJ11 * y;
    } else {
      invJd[4] = kNum_1;
      invJd[5] = -kNum_0p5 * theta;
      invJd[1] = kNum_0p5 * X[2];
      invJd[2] = -kNum_0p5 * X[1];
    }
    invJd[7] = -invJd[5];
    invJd[8] = invJd[4];

    invJd[0] = kNum_1;
    invJd[3] = kNum_0;
    invJd[6] = kNum_0;
    return invJ;
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return invJl(-X);
  }

  static Eigen::Matrix<Scalar, 2, 2> calcV(const typename SO2::LieAlgebra& xi) {
    using std::abs;
    using std::cos;
    using std::sin;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar& theta = xi[0];
    Eigen::Matrix<Scalar, 2, 2> V;
    Scalar* Vd = V.data();
    if (abs(theta) > eps) {
      const Scalar sintheta_on_theta = sin(theta) / theta;
      Vd[0] = sintheta_on_theta;
      Vd[1] = (kNum_1 - cos(theta)) / theta;
    } else {
      Vd[0] = kNum_1;
      Vd[1] = kNum_0p5 * theta;
    }
    Vd[2] = -Vd[1];
    Vd[3] = Vd[0];
    return V;
  }

  static Eigen::Matrix<Scalar, 2, 2> calcVinv(
      const typename SO2::LieAlgebra& xi, const SO2* rotation) {
    using std::abs;
    using std::cos;
    using std::sin;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar& theta = xi[0];
    Eigen::Matrix<Scalar, 2, 2> invV;
    Scalar* invVd = invV.data();
    if (abs(theta) > eps) {
      const Scalar theta_on_sintheta = theta / sin(theta);
      invVd[0] = kNum_0p5 * (cos(theta) + kNum_1) * theta_on_sintheta;
    } else {
      invVd[0] = kNum_1;
    }
    invVd[1] = -kNum_0p5 * theta;
    invVd[2] = -invVd[1];
    invVd[3] = invVd[0];
    return invV;
  }
};
}  // namespace se2_internal

}  // namespace sk4slam
