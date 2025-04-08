#pragma once

#include <Eigen/Core>

#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/matrix_group_helper.h"

namespace sk4slam {

// Rp (R^+), `p` for 'positive':
//     the Multiplicative Group of Positive Real Numbers
template <typename ScalarType>
class Rp : public LieGroupBase<Rp<ScalarType>>,
           public MatrixGroupCommonOps<Rp<ScalarType>> {
  using _MatrixGroupCommonOps = MatrixGroupCommonOps<Rp<ScalarType>>;

 public:
  using _MatrixGroupCommonOps::operator*;
  using _MatrixGroupCommonOps::operator=;

  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = 1;

  static constexpr int kAmbientDim = 1;

  // Ambient is the vector space that the Lie group is embedded in.
  using Ambient = Eigen::Matrix<Scalar, 1, 1>;

  // LieAlgebra should be defined as Eigen::Matrix<Scalar, kDim, 1>
  // or its derived types.
  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  // clang-format off
  // It's recommended to just define LieAlgebraEndomorphism as
  // Eigen::Matrix<Scalar, kDim, kDim> since it makes everything easier.
  //
  // Otherwise, you need to implement all the interfaces manually,
  // which is tedious and error-prone.
  // If you really need to use a different type, the interfaces of
  // LieAlgebraEndomorphism that are required to implement include:
  //
  //  - The action on LieAlgebra:
  //      LieAlgebra operator*(const LieAlgebra& X) const {}
  //
  //  - The structure of Algebra (a vector space that supports
  //    vector-multiplication), i.e. the operators:
  //
  //        E = E1 + E2,
  //        E = E1 - E2,
  //        E = E1 * s          (<- s is a scalar).
  //        Zero()
  //               ^
  //               These make LieAlgebraEndomorphism a vector space.
  //               Note the operation `E * s` is required,
  //               but `s * E` is not. You should avoid using
  //               `s * E` so that you code will not depend on the
  //               underlying implementation of LieAlgebraEndomorphism.
  //
  //        Identity()
  //        E = E1 * E2
  //               ^
  //               These make LieAlgebraEndomorphism an algebra.
  //
  //  - The inverse() function:
  //
  //        inverse()
  //              ^
  //               A inverse() function is required but not every endomorphism
  //               has an inverse. The behavior of inverse() is undefined
  //               when it is called on an endomorphism that is not invertible.
  //
  // - operator Eigen::Matrix<Scalar, kDim, kDim>()
  //
  //     LieAlgebraEndomorphism should be static castable to
  //     Eigen::Matrix<Scalar, kDim, kDim>.
  //
  // For more specific interfaces, just see the implementation of
  // ProductLieGroup::LieAlgebraEndomorphism.
  //
  // clang-format on

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static Rp Identity() {
    return Rp(liegroup::Constants<Scalar>::kNum_1);
  }

  Rp operator*(const Rp& other) const {
    return Rp(s_ * other.s_);
  }

  Rp inverse() const {
    return Rp(liegroup::Constants<Scalar>::kNum_1 / s_[0]);
  }

  bool isApprox(
      const Rp& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    // return std::abs(s_ - other.s_) < eps;
    return s_.isApprox(other.s_, eps);
  }

  static Rp Exp(const LieAlgebra& X) {
    using std::exp;
    return Rp(exp(X[0]));
  }

  static LieAlgebra Log(const Rp& g) {
    using std::log;
    return LieAlgebra(log(g.s_[0]));
  }

  // Rp is an abelian group, so Ad and ad are trivial.
  static LieAlgebraEndomorphism Ad(const Rp& g) {
    return LieAlgebraEndomorphism::Identity();
  }

  // Ad(g, X) = Ad(g) * X
  static LieAlgebra Ad(const Rp& g, const LieAlgebra& X) {
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
    return Ambient(liegroup::Constants<Scalar>::kNum_1);
  }

  // For use with CeresManifoldBlock
  ScalarType* data() {
    return s_.data();
  }

  const ScalarType* data() const {
    return s_.data();
  }

  template <typename _ScalarType>
  Rp<_ScalarType> cast() const {
    return Rp<_ScalarType>(s_.template cast<_ScalarType>());
  }

 public:
  // Rp specific interfaces

  Rp() : s_(liegroup::Constants<Scalar>::kNum_1) {}

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  Rp(const MatrixXpr& m) {  // NOLINT
    static_assert(
        (MatrixXpr::ColsAtCompileTime == 1 ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == 1 ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == 1);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == 1);
    }
    s_ = m;
  }

  Rp(const Ambient& s) : s_(s) {}  // NOLINT

  Rp(const Scalar& s) : s_(s) {}  // NOLINT

  const Scalar& value() const {
    return s_[0];
  }

  Scalar& value() {
    return s_[0];
  }

  const Eigen::Matrix<Scalar, 1, 1>& matrix() const {
    return s_;
  }

 private:
  Ambient s_;
};

template <typename Scalar>
Rp<Scalar>* asRp(Scalar* scalar) {
  return reinterpret_cast<Rp<Scalar>*>(scalar);
}

template <typename Scalar>
const Rp<Scalar>* asRp(const Scalar* scalar) {
  return reinterpret_cast<const Rp<Scalar>*>(scalar);
}

using Rpd = Rp<double>;

}  // namespace sk4slam
