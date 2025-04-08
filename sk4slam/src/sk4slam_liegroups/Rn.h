#pragma once

#include <Eigen/Core>

#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/matrix_group_helper.h"
#include "sk4slam_math/optimizable_manifold.h"

namespace sk4slam {

template <size_t _dim>
struct RnDim {
  static constexpr int value = _dim;
};

template <typename ScalarType, typename RnDim>
class Rn_;

// Rn (R^n): The n-dimensional vector space over R as a trivial Lie group,
//           with the group operation being vector addition.
template <size_t _dim, typename ScalarType>
using Rn = Rn_<ScalarType, RnDim<_dim>>;

////////// Implementation //////////

template <typename ScalarType, typename RnDim>
class Rn_ : public LieGroupBase<Rn_<ScalarType, RnDim>> {
 public:
  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = RnDim::value;

  static constexpr int kAmbientDim = RnDim::value;

  // Ambient is the vector space that the Lie group is embedded in.
  using Ambient = Eigen::Matrix<Scalar, kAmbientDim, 1>;

  // LieAlgebra should be defined as Eigen::Matrix<Scalar, kDim, 1>
  // or its derived types.
  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static Rn_ Identity() {
    return Rn_(Ambient::Zero());
  }

  Rn_ operator*(const Rn_& other) const {
    return Rn_(v_ + other.v_);
  }

  Rn_ inverse() const {
    return Rn_(-v_);
  }

  bool isApprox(
      const Rn_& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return manifold_traits<Ambient>::isApprox(this->v_, other.v_, eps);
  }

  static Rn_ Exp(const LieAlgebra& X) {
    return Rn_(X);
  }

  static LieAlgebra Log(const Rn_& g) {
    return LieAlgebra(g.v_);
  }

  // Rn_ is an abelian group, so Ad and ad are trivial.
  static LieAlgebraEndomorphism Ad(const Rn_& g) {
    return LieAlgebraEndomorphism::Identity();
  }

  // Ad(g, X) = Ad(g) * X
  static LieAlgebra Ad(const Rn_& g, const LieAlgebra& X) {
    return X;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    return LieAlgebraEndomorphism::Zero();
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    return LieAlgebra::Zero();
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    return LieAlgebraEndomorphism::Identity();
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return LieAlgebraEndomorphism::Identity();
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    return LieAlgebraEndomorphism::Identity();
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return LieAlgebraEndomorphism::Identity();
  }

  static Ambient hat(const LieAlgebra& X) {
    const Ambient& X_hat = X;
    return X_hat;
  }

  static LieAlgebra vee(const Ambient& X_hat) {
    const LieAlgebra& X = X_hat;
    return X;
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    Ambient t = Ambient::Zero();
    t(i) = 1;
    return t;
  }

  // For use with CeresManifoldBlock
  ScalarType* data() {
    return v_.data();
  }

  const ScalarType* data() const {
    return v_.data();
  }

  template <typename _ScalarType>
  Rn_<_ScalarType, RnDim> cast() const {
    return Rn_<_ScalarType, RnDim>(v_.template cast<_ScalarType>());
  }

 public:
  // Rn_ specific interfaces

  Rn_() : v_(Ambient::Zero()) {}

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  Rn_(const MatrixXpr& m) {  // NOLINT
    static_assert(
        (MatrixXpr::ColsAtCompileTime == 1 ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == kDim ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == 1);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == kDim);
    }
    v_ = m;
  }

  Rn_(const Ambient& v) : v_(v) {}  // NOLINT

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  Rn_& operator=(const MatrixXpr& m) {
    static_assert(
        (MatrixXpr::ColsAtCompileTime == 1 ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == kDim ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == 1);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == kDim);
    }
    v_ = m;
    return *this;
  }

  const Ambient& vector() const {
    return v_;
  }

  Ambient& vector() {
    return v_;
  }

 private:
  Ambient v_;
};

template <size_t _dim, typename Scalar>
Rn<_dim, Scalar>* asRn(Eigen::Matrix<Scalar, _dim, 1>* vector) {
  return reinterpret_cast<Rn<_dim, Scalar>*>(vector);
}

template <size_t _dim, typename Scalar>
const Rn<_dim, Scalar>* asRn(const Eigen::Matrix<Scalar, _dim, 1>* vector) {
  return reinterpret_cast<const Rn<_dim, Scalar>*>(vector);
}

template <size_t _dim, typename Scalar>
Rn<_dim, Scalar>* asRn(Scalar* vector_data) {
  return reinterpret_cast<Rn<_dim, Scalar>*>(vector_data);
}

template <size_t _dim, typename Scalar>
const Rn<_dim, Scalar>* asRn(const Scalar* vector_data) {
  return reinterpret_cast<const Rn<_dim, Scalar>*>(vector_data);
}

template <size_t _dim>
using Rnd = Rn<_dim, double>;

using R2d = Rnd<2>;
using R3d = Rnd<3>;

}  // namespace sk4slam
