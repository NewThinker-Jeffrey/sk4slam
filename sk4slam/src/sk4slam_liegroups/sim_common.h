#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/constants.h"

namespace sk4slam {

namespace sim_common {

template <class Scalar>
void calcABCForV(
    const Scalar& sigma, const Scalar& theta, Scalar* pA, Scalar* pB,
    Scalar* pC) {
  // Using math functions in the std namespace does make difference on
  // some platforms, at least on NVIDIA jetson. The unit tests might
  // fail on those platforms without the following using declarations.
  // It seems that the precision of the math functions under the global
  // namespace might be lower than those under the std namespace.
  using std::abs;
  using std::cos;
  using std::exp;
  using std::sin;
  const Scalar& eps = liegroup::Constants<Scalar>::kEps;
  const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
  const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
  const Scalar theta_sq = theta * theta;
  const Scalar scale = exp(sigma);
  Scalar &A = *pA, &B = *pB, &C = *pC;
  if (abs(sigma) < eps) {
    C = kNum_1;
    if (abs(theta) < eps) {
      A = kNum_0p5;
      B = Scalar(1. / 6.);
    } else {
      A = (kNum_1 - cos(theta)) / theta_sq;
      B = (theta - sin(theta)) / (theta_sq * theta);
    }
  } else {
    C = (scale - kNum_1) / sigma;
    if (abs(theta) < eps) {
      Scalar sigma_sq = sigma * sigma;
      A = ((sigma - kNum_1) * scale + kNum_1) / sigma_sq;
      B = (scale * kNum_0p5 * sigma_sq + scale - kNum_1 - sigma * scale) /
          (sigma_sq * sigma);
    } else {
      Scalar a = scale * sin(theta);
      Scalar b = scale * cos(theta);
      Scalar c = theta_sq + sigma * sigma;
      A = (a * sigma + (kNum_1 - b) * theta) / (theta * c);
      B = (C - ((b - kNum_1) * sigma + a * theta) / (c)) * kNum_1 / (theta_sq);
    }
  }
}

template <class Scalar>
void calcABCForVinv(
    const Scalar& sigma, const Scalar& theta, const Scalar* known_scale,
    Scalar* pA, Scalar* pB, Scalar* pC) {
  using std::abs;
  using std::cos;
  using std::exp;
  using std::sin;

  const Scalar& eps = liegroup::Constants<Scalar>::kEps;
  const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
  const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
  const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
  const Scalar theta_sq = theta * theta;
  const Scalar scale = known_scale ? *known_scale : exp(sigma);

  const Scalar scale_sq = scale * scale;
  const Scalar sin_theta = sin(theta);
  const Scalar cos_theta = cos(theta);
  Scalar &a = *pA, &b = *pB, &c = *pC;
  if (abs(sigma) < eps) {
    c = kNum_1 - kNum_0p5 * sigma;
    a = -kNum_0p5;
    if (abs(theta) < eps) {
      b = Scalar(1. / 12.);
    } else {
      b = (theta * sin_theta + kNum_2 * cos_theta - kNum_2) /
          (kNum_2 * theta_sq * (cos_theta - kNum_1));
    }
  } else {
    const Scalar scale_cu = scale_sq * scale;
    c = sigma / (scale - kNum_1);
    if (abs(theta) < eps) {
      a = (-sigma * scale + scale - kNum_1) /
          ((scale - kNum_1) * (scale - kNum_1));
      b = (scale_sq * sigma - kNum_2 * scale_sq + scale * sigma +
           kNum_2 * scale) /
          (kNum_2 * scale_cu - Scalar(6) * scale_sq + Scalar(6) * scale -
           kNum_2);
    } else {
      const Scalar s_sin_theta = scale * sin_theta;
      const Scalar s_cos_theta = scale * cos_theta;
      a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
          (theta * (scale_sq - kNum_2 * s_cos_theta + kNum_1));
      b = -scale *
          (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
           scale * sigma + sigma * cos_theta - sigma) /
          (theta_sq * (scale_cu - kNum_2 * scale * s_cos_theta - scale_sq +
                       kNum_2 * s_cos_theta + scale - kNum_1));
    }
  }
}

}  // namespace sim_common

}  // namespace sk4slam
