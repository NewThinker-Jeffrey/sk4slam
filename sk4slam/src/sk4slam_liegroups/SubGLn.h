#pragma once

#include <Eigen/Core>
#include <set>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/bch.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/matrix_group_helper.h"

namespace sk4slam {

// Declared the default implementation template for SubGLn
template <typename Scalar>
struct SubGLnApproximationOptions {
  template <typename _Scalar>
  using Constants = liegroup::Constants<_Scalar>;

  static inline const Scalar kEpsForJacobian =
      liegroup::Constants<Scalar>::kEps;
  static inline const Scalar kEpsForLog = liegroup::Constants<Scalar>::kEps;
  static inline const Scalar kEpsForExp = liegroup::Constants<Scalar>::kEps;

  static constexpr int kMaxOrderForJacobian = -1;
  static constexpr int kMaxOrderForLog = -1;
  static constexpr int kMaxOrderForExp = -1;
};

template <int _jacobian_order, int _log_order = -1, int _exp_order = -1>
struct SubGLnApproximationOrder {
  template <typename Scalar>
  struct ApproximationOptions : public SubGLnApproximationOptions<Scalar> {
    static constexpr int kMaxOrderForJacobian = _jacobian_order;
    static constexpr int kMaxOrderForLog = _log_order;
    static constexpr int kMaxOrderForExp = _exp_order;
  };
};

// This class template defines the interface of a Sub group of GLUnits(n).
template <
    typename Scalar, template <typename> class SubStructure,
    template <typename> class ApproximationOptionsTemp =
        SubGLnApproximationOptions>
class SubGLn;

template <
    typename Scalar, template <typename> class SubStructure,
    int _jacobian_order, int _log_order = -1, int _exp_order = -1>
using SubGLnUpToOrder = SubGLn<
    Scalar, SubStructure,
    SubGLnApproximationOrder<_jacobian_order, _log_order, _exp_order>::
        template ApproximationOptions>;

// The structure for the group of units (invertible elements) of n x n matrices
//     (https://en.wikipedia.org/wiki/Unit_(ring_theory))
template <int n>
struct GLUnits {
  template <typename ScalarType>
  struct Structure {
    using Scalar = ScalarType;
    static constexpr int N = n;
    static constexpr int kDim = N * N;

    static Eigen::Matrix<Scalar, N, N> hat(
        const Eigen::Matrix<Scalar, kDim, 1>& X) {
      // return X.template cast<Scalar>().template reshaped<N, N>();
      // return X.template reshaped<N, N>();

      Eigen::Matrix<Scalar, N, N> hat_X;

      // memcpy(hat_X.data(), X.data(), kDim * sizeof(Scalar));
      //   ^ memcpy is dangerous if Scalar is not trivially copyable.

      Scalar* hat_Xd = hat_X.data();
      const Scalar* Xd = X.data();
      // Will the compiler flatten this loop? (kDim is known at compile time)
      for (size_t i = 0; i < kDim; i++) {
        hat_Xd[i] = Xd[i];
      }

      return hat_X;
    }

    static Eigen::Matrix<Scalar, kDim, 1> vee(
        const Eigen::Matrix<Scalar, N, N>& hat_X) {
      // return hat_X.template cast<Scalar>().template reshaped<kDim, 1>();
      // return hat_X.template reshaped<kDim, 1>();

      Eigen::Matrix<Scalar, kDim, 1> X;

      // memcpy(X.data(), hat_X.data(), kDim * sizeof(Scalar));
      //   ^ memcpy is dangerous if Scalar is not trivially copyable.

      Scalar* Xd = X.data();
      const Scalar* hat_Xd = hat_X.data();
      // Will the compiler flatten this loop? (kDim is known at compile time)
      for (size_t i = 0; i < kDim; i++) {
        Xd[i] = hat_Xd[i];
      }

      return X;
    }

    static Eigen::Matrix<Scalar, N, N> generator(int i) {
      Eigen::Matrix<Scalar, kDim, 1> X = Eigen::Matrix<Scalar, kDim, 1>::Zero();
      X(i) = 1;
      return hat(X);
    }
  };
};

// Declare the structure of the unit group of n x n matrices.
template <
    int n, typename Scalar,
    template <typename> class ApproximationOptionsTemp =
        SubGLnApproximationOptions>
using GLn =
    SubGLn<Scalar, GLUnits<n>::template Structure, ApproximationOptionsTemp>;

template <
    int n, typename Scalar, int _jacobian_order, int _log_order = -1,
    int _exp_order = -1>
using GLnUpToOrder = SubGLn<
    Scalar, GLUnits<n>::template Structure,
    SubGLnApproximationOrder<_jacobian_order, _log_order, _exp_order>::
        template ApproximationOptions>;

template <typename Scalar>
using GL2 = GLn<2, Scalar>;

template <typename Scalar>
using GL3 = GLn<3, Scalar>;

template <typename Scalar>
using GL4 = GLn<4, Scalar>;

using GL2d = GL2<double>;
using GL3d = GL3<double>;
using GL4d = GL4<double>;

///////// Implementations //////////

template <
    typename ScalarType, template <typename> class SubStructureTemp,
    template <typename> class ApproximationOptionsTemp>
class SubGLn
    : public LieGroupBase<
          SubGLn<ScalarType, SubStructureTemp, ApproximationOptionsTemp>>,
      public MatrixGroupCommonOps<
          SubGLn<ScalarType, SubStructureTemp, ApproximationOptionsTemp>> {
  using SubStructure = SubStructureTemp<ScalarType>;
  using ApproximationOptions = ApproximationOptionsTemp<ScalarType>;
  static constexpr int N = SubStructure::N;

  template <typename _ScalarType>
  using _CastTemplate =
      SubGLn<_ScalarType, SubStructureTemp, ApproximationOptionsTemp>;

  using _MatrixGroupCommonOps = MatrixGroupCommonOps<
      SubGLn<ScalarType, SubStructureTemp, ApproximationOptionsTemp>>;

 public:
  using _MatrixGroupCommonOps::operator*;
  using _MatrixGroupCommonOps::operator=;

  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = SubStructure::kDim;

  static constexpr int kAmbientDim = N * N;

  using Ambient = Eigen::Matrix<Scalar, N, N>;

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static SubGLn Identity() {
    return SubGLn(Eigen::Matrix<Scalar, N, N>::Identity());
  }

  SubGLn operator*(const SubGLn& other) const {
    return SubGLn(mat_ * other.mat_);
  }

  SubGLn inverse() const {
    return SubGLn(mat_.inverse());
  }

  bool isApprox(
      const SubGLn& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return mat_.isApprox(other.mat_, eps);
  }

  static SubGLn Exp(const LieAlgebra& X) {
    return SubGLn(expOnAlgebra(
        SubStructure::hat(X), ApproximationOptions::kMaxOrderForExp,
        ApproximationOptions::kEpsForExp));
  }

  static LieAlgebra Log(const SubGLn& g) {
    return SubStructure::vee(logOnAlgebra(
        g.mat_, ApproximationOptions::kMaxOrderForLog,
        ApproximationOptions::kEpsForLog));
  }

  static LieAlgebraEndomorphism Ad(const SubGLn& g) {
    LieAlgebraEndomorphism Ad_g;
    Eigen::Matrix<Scalar, N, N> mat_inverse = g.mat_.inverse();
    for (int i = 0; i < kDim; ++i) {
      Ad_g.col(i) =
          SubStructure::vee(g.mat_ * SubStructure::generator(i) * mat_inverse);
    }
    return Ad_g;
  }

  // Ad(g, X) = Ad(g) * X
  static LieAlgebra Ad(const SubGLn& g, const LieAlgebra& X) {
    return SubStructure::vee(g.mat_ * SubStructure::hat(X) * g.mat_.inverse());
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    LieAlgebraEndomorphism adX;
    Eigen::Matrix<Scalar, N, N> hat_X = SubStructure::hat(X);
    for (int i = 0; i < kDim; ++i) {
      Eigen::Matrix<Scalar, N, N> gen_i = SubStructure::generator(i);
      adX.col(i) = SubStructure::vee(hat_X * gen_i - gen_i * hat_X);
    }
    return adX;
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    Eigen::Matrix<Scalar, N, N> hat_X1 = SubStructure::hat(X1);
    Eigen::Matrix<Scalar, N, N> hat_X2 = SubStructure::hat(X2);
    return SubStructure::vee(hat_X1 * hat_X2 - hat_X2 * hat_X1);
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    return leftLieJacobian<SubGLn>(
        X, ApproximationOptions::kMaxOrderForJacobian,
        ApproximationOptions::kEpsForJacobian);
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Jl(-X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    return BCHInvLeftLieJacobian<SubGLn>(
        X, ApproximationOptions::kMaxOrderForJacobian,
        ApproximationOptions::kEpsForJacobian);
    // return Jl(X).inverse();
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return invJl(-X);
  }

  static Ambient hat(const LieAlgebra& X) {
    return SubStructure::hat(X);
  }

  static LieAlgebra vee(const Ambient& X_hat) {
    return SubStructure::vee(X_hat);
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    return SubStructure::generator(i);
  }

  // For use with CeresManifoldBlock
  Scalar* data() {
    return mat_.data();
  }

  const Scalar* data() const {
    return mat_.data();
  }

  template <typename _ScalarType>
  _CastTemplate<_ScalarType> cast() const {
    return _CastTemplate<_ScalarType>(mat_.template cast<_ScalarType>());
  }

 public:
  // SubGLn specific interfaces

  SubGLn() : mat_(Eigen::Matrix<Scalar, N, N>::Identity()) {}

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  SubGLn(const MatrixXpr& m) {  // NOLINT
    static_assert(
        (MatrixXpr::ColsAtCompileTime == N ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == N ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == N);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == N);
    }
    mat_ = m;
  }

  SubGLn(const Eigen::Matrix<Scalar, N, N>& mat) : mat_(mat) {}  // NOLINT

  SubGLn(const SubGLn& other) : mat_(other.mat_) {}

  const Eigen::Matrix<Scalar, N, N>& matrix() const {
    return mat_;
  }

 private:
  Eigen::Matrix<Scalar, N, N> mat_;
};

}  // namespace sk4slam
