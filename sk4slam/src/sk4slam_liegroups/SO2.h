#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/matrix_group_helper.h"

namespace sk4slam {

template <typename ScalarType>
class SO2 : public LieGroupBase<SO2<ScalarType>>,
            public MatrixGroupCommonOps<SO2<ScalarType>> {
  using _MatrixGroupCommonOps = MatrixGroupCommonOps<SO2<ScalarType>>;

 public:
  using _MatrixGroupCommonOps::operator*;
  using _MatrixGroupCommonOps::operator=;

  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = 1;

  static constexpr int kAmbientDim = 4;

  // Ambient is the vector space that the Lie group is embedded in.
  using Ambient = Eigen::Matrix<Scalar, 2, 2>;

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static SO2 Identity() {
    return SO2(Eigen::Matrix<Scalar, 2, 2>::Identity());
  }

  SO2 operator*(const SO2& other) const {
    return SO2(rotation_matrix_ * other.rotation_matrix_);
  }

  SO2 inverse() const {
    return SO2(rotation_matrix_.transpose());
  }

  bool isApprox(
      const SO2& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return rotation_matrix_.isApprox(other.rotation_matrix_, eps);
  }

  static SO2 Exp(const LieAlgebra& w) {
    return SO2(expM(w[0]));
  }

  static LieAlgebra Log(const SO2& g) {
    return LieAlgebra(logM(g.matrix()));
  }

  // SO2 is an abelian group, so Ad and ad are trivial.
  static LieAlgebraEndomorphism Ad(const SO2& g) {
    return LieAlgebraEndomorphism::Identity();
  }

  // Ad(g, w) = Ad(g) * w
  static LieAlgebra Ad(const SO2& g, const LieAlgebra& w) {
    return w;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Zero();
  }

  static LieAlgebra bracket(const LieAlgebra& w1, const LieAlgebra& w2) {
    return LieAlgebra::Zero();
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  static Ambient hat(const LieAlgebra& w) {
    Ambient w_hat = Ambient::Zero();
    Scalar* d = w_hat.data();
    const Scalar w0 = w(0);
    d[1] = w0;
    d[2] = -w0;
    return w_hat;
  }

  static LieAlgebra vee(const Ambient& w_hat) {
    return LieAlgebra(w_hat.data()[1]);
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    LieAlgebra w(kNum_1);
    return hat(w);
  }

  // For use with AutoLocalBlock
  ScalarType* data() {
    return rotation_matrix_.data();
  }

  const ScalarType* data() const {
    return rotation_matrix_.data();
  }

  template <typename _ScalarType>
  SO2<_ScalarType> cast() const {
    return SO2<_ScalarType>(rotation_matrix_.template cast<_ScalarType>());
  }

 public:  // MatrixGroupCommonOps overrides.
  // Override MatrixGroupCommonOps::JmultVector().
  // Return
  //    J = d [Exp(x) * v] / d (x)      (Evaluated at x=0, x ∈ LieAlg, v ∈ R^N)
  template <typename = SO2>
  static Eigen::Matrix<Scalar, 2, kDim> JmultVector(
      const Vector<2, Scalar>& v) {
    const Scalar& x = v[0];
    const Scalar& y = v[1];
    return Eigen::Matrix<Scalar, 2, 1>(-y, x);
  }

 public:
  // SO2 specific interfaces

  SO2() : rotation_matrix_(Eigen::Matrix<Scalar, 2, 2>::Identity()) {}

  explicit SO2(const Scalar& theta) {
    rotation_matrix_ = expM(theta);
  }

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  SO2(const MatrixXpr& m) {  // NOLINT
    static_assert(
        (MatrixXpr::ColsAtCompileTime == 2 ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == 2 ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == 2);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == 2);
    }
    rotation_matrix_ = m;
  }

  SO2(const Eigen::Matrix<Scalar, 2, 2>& rotation_matrix)  // NOLINT
  : rotation_matrix_(rotation_matrix) {}

  static SO2 Perfect(const Eigen::Matrix<Scalar, 2, 2>& R) {
    // Return a perfect rotation matrix.
    // return SO2(expM(logM(*R)));
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return SO2(svd.matrixU() * svd.matrixV().transpose());
  }

  static SO2 Perfect(const SO2& R) {
    return Perfect(R.matrix());
  }

  void normalize() {
    rotation_matrix_ = Perfect(rotation_matrix_).matrix();
  }

  SO2 normalized() const {
    return Perfect(*this);
  }

  const Eigen::Matrix<Scalar, 2, 2>& matrix() const {
    return rotation_matrix_;
  }

 public:
  static Eigen::Matrix<Scalar, 2, 2> expM(const Scalar& w) {
    using std::cos;
    using std::sin;
    Eigen::Matrix<Scalar, 2, 2> R;
    const Scalar s = sin(w);
    const Scalar c = cos(w);
    Scalar* Rd = R.data();
    Rd[0] = c;
    Rd[1] = s;
    Rd[2] = -s;
    Rd[3] = c;
    return R;
  }

  static Scalar logM(const Eigen::Matrix<Scalar, 2, 2>& R) {
    using std::atan2;
    const Scalar* Rd = R.data();
    return atan2(Rd[1], Rd[0]);
  }

 private:
  Ambient rotation_matrix_;
};

template <typename Scalar>
SO2<Scalar>* asSO2(Eigen::Matrix<Scalar, 2, 2>* rotation_matrix) {
  return reinterpret_cast<SO2<Scalar>*>(rotation_matrix);
}

template <typename Scalar>
const SO2<Scalar>* asSO2(const Eigen::Matrix<Scalar, 2, 2>* rotation_matrix) {
  return reinterpret_cast<const SO2<Scalar>*>(rotation_matrix);
}

template <typename Scalar>
SO2<Scalar>* asSO2(Scalar* rotation_matrix_data) {
  return reinterpret_cast<SO2<Scalar>*>(rotation_matrix_data);
}

template <typename Scalar>
const SO2<Scalar>* asSO2(const Scalar* rotation_matrix_data) {
  return reinterpret_cast<const SO2<Scalar>*>(rotation_matrix_data);
}

using SO2d = SO2<double>;

}  // namespace sk4slam
