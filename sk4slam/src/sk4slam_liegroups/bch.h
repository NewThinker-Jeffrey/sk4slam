#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/bch_helper/bch_helper.h"
#include "sk4slam_liegroups/constants.h"

namespace sk4slam {

// BCH approximation can be used to compute the inverse left Jacobian.

template <typename LieGroup>
typename LieGroup::LieAlgebra BCHInvLeftLieJacobianOfXOnY(
    const typename LieGroup::LieAlgebra& X,
    const typename LieGroup::LieAlgebra& Y, int max_order = -1,
    typename LieGroup::Scalar eps = liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename LieGroup::Scalar;
  // clang-format off

  // BCH formula (up to order 2 of X and Y)
  //   for Z that:
  //          exp(Z) = exp(X) * exp(Y)
  //   Z = BCH(X,Y) = X + Y + [X,Y]/2 + ([X,[X,Y]] - [Y,[X,Y]])/12 + ...
  // For Y much smaller than X, the 2nd order of Y can be ignored:
  //   Z = BCH(X,Y) ≈ X + Y + [X,Y]/2 + [X,[X,Y]]/12
  //
  // So for Jr(X) * Y, (Y much smaller than X) we have the approximation:
  //   invJr(X) * Y = BCH(X,Y) - X
  //                ≈ Y + [X,Y]/2 + [X,[X,Y]]/ 12
  //
  // But note we're using invJl instead of invJr, so we need to replace X
  // with -X in the above formula:
  //   invJl(X) * Y ≈ Y - [X,Y]/2 + [X,[X,Y]]/ 12

  // clang-format on

  // const typename LieGroup::LieAlgebra X_Y = LieGroup::bracket(X, Y);
  // const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
  // const Scalar& kNum_12 = liegroup::Constants<Scalar>::kNum_12;

  // BCH approximation (up to the 2nd order of X)
  // return Y - X_Y / kNum_2 + LieGroup::bracket(X, X_Y) / kNum_12;

  typename LieGroup::LieAlgebra Xk_Y = Y;
  typename LieGroup::LieAlgebra result = Xk_Y;
  static const typename LieGroup::LieAlgebra Zero =
      LieGroup::LieAlgebra::Zero();

  int max_iter;
  if (max_order < 0 || max_order > _inv_left_jacobian_coeffs.size()) {
    max_iter = _inv_left_jacobian_coeffs.size();
  } else {
    max_iter = max_order + 1;
  }

  int k;
  for (k = 1; k < max_iter; ++k) {
    Xk_Y = LieGroup::bracket(X, Xk_Y);
    if (_inv_left_jacobian_coeffs[k] == 0) {
      continue;
    }
    typename LieGroup::LieAlgebra item =
        Scalar(_inv_left_jacobian_coeffs[k]) * Xk_Y;
    result += item;
    if (item.isZero(eps)) {
      break;
    }
  }

  if (k == max_iter) {
    LOGA("invLeftLieJacobianOfXOnY max iter reached");
    k -= 1;
  }
  LOGA("invLeftLieJacobianOfXOnY order: %d", k);
  return result;
}

template <typename LieGroup>
typename LieGroup::LieAlgebraEndomorphism BCHInvLeftLieJacobian(
    const typename LieGroup::LieAlgebra& X, int max_order = -1,
    typename LieGroup::Scalar eps = liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename LieGroup::Scalar;
  using Endomorphism = typename LieGroup::LieAlgebraEndomorphism;
  static const Endomorphism I = Endomorphism::Identity();
  static const Endomorphism Zero = Endomorphism::Zero();
  const Endomorphism adX = LieGroup::ad(X);

  // const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
  // const Scalar& kNum_12 = liegroup::Constants<Scalar>::kNum_12;
  // if (max_order == 1) {
  //   return I - adX / kNum_2;
  // } else {
  //   return I - adX / kNum_2 + adX * adX / kNum_12;
  // }

  Endomorphism adXk = I;
  Endomorphism result = I;
  int max_iter;
  if (max_order < 0 || max_order > _inv_left_jacobian_coeffs.size()) {
    max_iter = _inv_left_jacobian_coeffs.size();
  } else {
    max_iter = max_order + 1;
  }

  int k;
  for (k = 1; k < max_iter; ++k) {
    adXk *= adX;
    if (_inv_left_jacobian_coeffs[k] == 0) {
      continue;
    }
    Endomorphism item = adXk * Scalar(_inv_left_jacobian_coeffs[k]);
    result += item;
    if (item.isZero(eps)) {
      break;
    }
  }

  if (k == max_iter) {
    LOGA("BCHInvLeftLieJacobian max iter reached");
    k -= 1;
  }
  LOGA("BCHInvLeftLieJacobian order: %d", k);
  return result;
}

// Use BCH to compute Z such that exp(Z) = exp(X) * exp(Y).
//
// This function is only for testing purpose due to its lack of efficiency.
// BCH converges slowly and may even diverge when X or Y is not sufficiently
// small, and the number of terms to be computed grows nearly exponentially
// with the order.
template <typename LieGroup>
const typename LieGroup::LieAlgebra BCH(
    const typename LieGroup::LieAlgebra& X,
    const typename LieGroup::LieAlgebra& Y, int max_xy_order = 6) {
  using LieAlgebra = typename LieGroup::LieAlgebra;
  using Scalar = typename LieAlgebra::Scalar;
  auto b = [](const LieAlgebra& X, const LieAlgebra& Y) {
    return LieGroup::bracket(X, Y);
  };
  LieAlgebra Z = X + Y;

  int max_ad_order = max_xy_order - 1;
  for (int ad_order = 1; ad_order < max_ad_order + 1; ++ad_order) {
    const auto& seqs_and_weights = _adxy_order_seqs_and_coeffs.at(ad_order);
    for (const auto& seq_and_weight : seqs_and_weights) {
      const auto& seq = seq_and_weight.first;
      const Scalar coeff = Scalar(seq_and_weight.second);
      LieAlgebra tmp_Z = X;
      for (auto iter = seq.rbegin(); iter != seq.rend(); ++iter) {
        auto& adx_order = iter->first;
        auto& ady_order = iter->second;
        // apply Y first, since Y is on the right
        for (int i = 0; i < ady_order; ++i) {
          tmp_Z = b(Y, tmp_Z);
        }
        for (int i = 0; i < adx_order; ++i) {
          tmp_Z = b(X, tmp_Z);
        }
      }
      Z += tmp_Z * coeff;
    }
  }
  return Z;
}

template <typename LieGroup>
const typename LieGroup::LieAlgebra BCHwiki(
    const typename LieGroup::LieAlgebra& X,
    const typename LieGroup::LieAlgebra& Y, int max_xy_order = 6) {
  using LieAlgebra = typename LieGroup::LieAlgebra;
  using Scalar = typename LieAlgebra::Scalar;
  auto b = [](const LieAlgebra& X, const LieAlgebra& Y) {
    return LieGroup::bracket(X, Y);
  };
  LieAlgebra Z = X + Y;
  // https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
  // https://wikimedia.org/api/rest_v1/media/math/render/svg/936fe5484a33c662e0f0d7db978f00178cc5a2e9
  if (max_xy_order >= 2) {
    Z += b(X, Y) * Scalar(1. / 2);
  }
  if (max_xy_order >= 3) {
    Z += b(X, b(X, Y)) * Scalar(1. / 12);
    Z += b(Y, b(Y, X)) * Scalar(1. / 12);
  }
  if (max_xy_order >= 4) {
    Z -= b(Y, b(X, b(X, Y))) * Scalar(1. / 24);
  }

  if (max_xy_order >= 5) {
    Z -= b(Y, b(Y, b(Y, b(Y, X)))) * Scalar(1. / 720);
    Z -= b(X, b(X, b(X, b(X, Y)))) * Scalar(1. / 720);
    Z += b(X, b(Y, b(Y, b(Y, X)))) * Scalar(1. / 360);
    Z += b(Y, b(X, b(X, b(X, Y)))) * Scalar(1. / 360);
    Z += b(X, b(Y, b(X, b(Y, X)))) * Scalar(1. / 120);
    Z += b(Y, b(X, b(Y, b(X, Y)))) * Scalar(1. / 120);
  }
  if (max_xy_order >= 6) {
    Z += b(X, b(Y, b(X, b(Y, b(X, Y))))) * Scalar(1. / 240);
    Z += b(X, b(Y, b(X, b(X, b(X, Y))))) * Scalar(1. / 720);
    Z -= b(X, b(X, b(Y, b(Y, b(X, Y))))) * Scalar(1. / 720);
    Z += b(X, b(Y, b(Y, b(Y, b(X, Y))))) * Scalar(1. / 1440);
    Z -= b(X, b(X, b(Y, b(X, b(X, Y))))) * Scalar(1. / 1440);
  }
  return Z;
}

}  // namespace sk4slam
