#pragma once

#include <limits>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/constants.h"

namespace sk4slam {

//// Computation of some series ////

//    exp(X)
template <typename AlgebraType>
AlgebraType expOnAlgebra(
    const AlgebraType& X, int max_order = -1,
    typename AlgebraType::Scalar eps =
        liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename AlgebraType::Scalar;
  static const AlgebraType Zero = AlgebraType::Zero();
  AlgebraType item = AlgebraType::Identity();
  AlgebraType sum = item;
  int max_iter;

  if (max_order < 0) {
    max_iter = std::numeric_limits<int>::max();
  } else {
    max_iter = max_order + 1;
  }

  int i;
  for (i = 1; i < max_iter; i++) {
    item *= (X / Scalar(i));
    sum += item;
    // if (max_order < 0) {
    if (item.isZero(eps)) {
      break;
    }
    // }
  }

  if (i == max_iter) {
    LOGA("expOnAlgebra max iter reached");
    i -= 1;
  }
  LOGA("expOnAlgebra order: %d", i);
  return sum;
}

//    (exp(X) - I) / X
template <typename AlgebraType>
AlgebraType expm1OverXOnAlgebra(
    const AlgebraType& X, int max_order = -1,
    typename AlgebraType::Scalar eps =
        liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename AlgebraType::Scalar;
  static const AlgebraType Zero = AlgebraType::Zero();
  AlgebraType item = AlgebraType::Identity();
  AlgebraType sum = item;
  int max_iter;

  if (max_order < 0) {
    max_iter = std::numeric_limits<int>::max();
  } else {
    max_iter = max_order + 1;
  }

  int i;
  for (i = 1; i < max_iter; ++i) {
    item *= (X / Scalar(i + 1));
    sum += item;
    // if (max_order < 0) {
    if (item.isZero(eps)) {
      break;
    }
    // }
  }

  if (i == max_iter) {
    LOGA("expm1OverXOnAlgebra max iter reached");
    i -= 1;
  }
  LOGA("expm1OverXOnAlgebra order: %d", i);
  return sum;
}

// This function is used for testing purpose, since the convergence of log
// is much slower than exp.
//    log(X)
template <typename AlgebraType>
AlgebraType logOnAlgebra(
    const AlgebraType& X, int max_order = -1,
    typename AlgebraType::Scalar eps =
        liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename AlgebraType::Scalar;
  static const AlgebraType Zero = AlgebraType::Zero();
  static const AlgebraType I = AlgebraType::Identity();
  const AlgebraType X_minus_I = X - I;
  AlgebraType X_minus_I_pow_k = X_minus_I;  // start from k = 1
  AlgebraType sum = X_minus_I_pow_k;
  int max_iter;

  if (max_order < 0) {
    max_iter = std::numeric_limits<int>::max();
  } else {
    max_iter = max_order + 1;
  }

  int i;
  for (i = 2; i < max_iter; ++i) {
    X_minus_I_pow_k *= X_minus_I;
    AlgebraType item = X_minus_I_pow_k / Scalar(i);
    if (i % 2) {
      sum += item;
    } else {
      sum -= item;
    }
    // if (max_order < 0) {
    if (item.isZero(eps)) {
      break;
    }
    // }
  }

  if (i == max_iter) {
    LOGA("logOnAlgebra max iter reached");
    i -= 1;
  }
  LOGA("logOnAlgebra order: %d", i);
  return sum;
}

// This function is used for testing purpose, since the convergence of log
// is much slower than exp.
//      log(X) / (X-I)
template <typename AlgebraType>
AlgebraType logOverXm1OnAlgebra(
    const AlgebraType& X, int max_order = -1,
    typename AlgebraType::Scalar eps =
        liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename AlgebraType::Scalar;
  static const AlgebraType Zero = AlgebraType::Zero();
  static const AlgebraType I = AlgebraType::Identity();
  const AlgebraType X_minus_I = X - I;
  AlgebraType X_minus_I_pow_k = I;  // start from k = 0
  AlgebraType sum = X_minus_I_pow_k;
  int max_iter;

  if (max_order < 0) {
    max_iter = std::numeric_limits<int>::max();
  } else {
    max_iter = max_order + 1;
  }

  int i;
  for (i = 1; i < max_iter; ++i) {
    X_minus_I_pow_k *= X_minus_I;
    AlgebraType item = X_minus_I_pow_k / Scalar(i + 1);
    if (i % 2) {
      sum -= item;
    } else {
      sum += item;
    }
    // if (max_order < 0) {
    if (item.isZero(eps)) {
      break;
    }
    // }
  }

  if (i == max_iter) {
    LOGA("logOverXm1OnAlgebra max iter reached");
    i -= 1;
  }
  LOGA("logOverXm1OnAlgebra order: %d", i);
  return sum;
}

//// Computation of general Left-Jacobian (right logarithmic derivative) ////

template <typename LieGroup>
typename LieGroup::LieAlgebraEndomorphism leftLieJacobian(
    const typename LieGroup::LieAlgebra& X, int max_order = -1,
    typename LieGroup::Scalar eps = liegroup::Constants<decltype(eps)>::kEps) {
  return expm1OverXOnAlgebra(LieGroup::ad(X), max_order, eps);
}

template <typename LieGroup>
typename LieGroup::LieAlgebra leftLieJacobianOfXOnY(
    const typename LieGroup::LieAlgebra& X,
    const typename LieGroup::LieAlgebra& Y, int max_Xorder = -1,
    typename LieGroup::Scalar eps = liegroup::Constants<decltype(eps)>::kEps) {
  using Scalar = typename LieGroup::Scalar;
  using LieAlgebra = typename LieGroup::LieAlgebra;
  static const LieAlgebra Zero = LieAlgebra::Zero();
  LieAlgebra item = Y;
  LieAlgebra sum = item;
  int max_iter;

  if (max_Xorder < 0) {
    max_iter = std::numeric_limits<int>::max();
  } else {
    max_iter = max_Xorder + 1;
  }

  int i;
  for (i = 1; i < max_iter; ++i) {
    item = LieGroup::bracket(X, item) / Scalar(i + 1);
    sum += item;
    // if (max_order < 0) {
    if (item.isZero(eps)) {
      break;
    }
    // }
  }

  if (i == max_iter) {
    LOGA("leftLieJacobianOfXOnY max iter reached");
    i -= 1;
  }
  LOGA("leftLieJacobianOfXOnY order: %d", i);
  return sum;
}

// #define INV_JACOBIAN_BY_SHIFT_LOG

template <typename LieGroup>
typename LieGroup::LieAlgebraEndomorphism invLeftLieJacobian(
    const LieGroup& g, int max_order = -1,
    typename LieGroup::Scalar eps = liegroup::Constants<decltype(eps)>::kEps) {
#ifdef INV_JACOBIAN_BY_SHIFT_LOG
  return logOverXm1OnAlgebra(LieGroup::Ad(g), max_order, eps);
#else
  return invLeftLieJacobian(LieGroup::Log(g), max_order, eps);
#endif
}

template <typename LieGroup>
typename LieGroup::LieAlgebraEndomorphism invLeftLieJacobian(
    const typename LieGroup::LieAlgebra& X, int max_order = -1,
    typename LieGroup::Scalar eps = liegroup::Constants<decltype(eps)>::kEps) {
#ifdef INV_JACOBIAN_BY_SHIFT_LOG
  return invLeftLieJacobian(LieGroup::Exp(X), max_order, eps);
#else
  return leftLieJacobian<LieGroup>(X, max_order, eps).inverse();
#endif
}

}  // namespace sk4slam
