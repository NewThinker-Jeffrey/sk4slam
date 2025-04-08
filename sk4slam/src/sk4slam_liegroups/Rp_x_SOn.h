#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/Rp.h"
#include "sk4slam_liegroups/SO2.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/direct_product.h"
#include "sk4slam_liegroups/matrix_group_helper.h"

namespace sk4slam {

// For all n > 1, the matrix representation for the Lie algebra
// of SO(n) is always n x n skew-symmetric matrix.
// So the scaled-SO(n), i.e. "R^+ x SO(n)", the direct product
// of R^+ and SO(n), can be uniformly addressed for all n.
template <typename Rp, typename SOn>
class Rp_x_SOn_Base;

// Specialization for "R^+ x SO(3)"
template <typename Scalar>
using Rp_x_SO3 = Rp_x_SOn_Base<Rp<Scalar>, SO3<Scalar>>;

using Rp_x_SO3d = Rp_x_SO3<double>;

// Specialization for "R^+ x SO(2)".
// Note, since R^+ and SO(2) are both abelian groups, the direct product
// keeps the abelian property (its Lie algebra is trivial).
template <typename Scalar>
using Rp_x_SO2 = Rp_x_SOn_Base<Rp<Scalar>, SO2<Scalar>>;

using Rp_x_SO2d = Rp_x_SO2<double>;

//////////// Implementation for Rp_x_SOn_Base ////////////////

template <typename RpAmbient, typename SOnAmbient>
class Rp_x_SOn_AmbientBase : public ProductVectorSpaceBase<
                                 Rp_x_SOn_AmbientBase, RpAmbient, SOnAmbient> {
  using _AmbientBase =
      ProductVectorSpaceBase<Rp_x_SOn_AmbientBase, RpAmbient, SOnAmbient>;

 public:
  // Import the constructors from the base class
  using _AmbientBase::_AmbientBase;

  using Scalar = typename _AmbientBase::Scalar;
  static constexpr int N = SOnAmbient::RowsAtCompileTime;
  static_assert(
      N > 1, "SO(1) is a single point set and (R^+ x SO(1)) is nonsensical.");
  using MatrixRepresentation = Eigen::Matrix<Scalar, N, N>;

  MatrixRepresentation matrix() const {
    Scalar sigma = this->template part<0>()[0];
    MatrixRepresentation M = this->template part<1>();
    // M.diagonal().array() = sigma;
    if constexpr (N == 3) {
      Scalar* d = M.data();
      d[0] /*M(0, 0)*/ = sigma;
      d[4] /*M(1, 1)*/ = sigma;
      d[8] /*M(2, 2)*/ = sigma;
    } else if constexpr (N == 2) {
      Scalar* d = M.data();
      d[0] /*M(0, 0)*/ = sigma;
      d[3] /*M(1, 1)*/ = sigma;
    } else {
      M.diagonal().array() = sigma;
    }
    return M;
  }

  operator MatrixRepresentation() const {
    return matrix();
  }
};

template <typename Rp, typename SOn>
class Rp_x_SOn_Base
    : public ProductLieGroupBase<Rp_x_SOn_Base, Rp_x_SOn_AmbientBase, Rp, SOn>,
      public MatrixGroupCommonOps<Rp_x_SOn_Base<Rp, SOn>> {
  using _ProductLieGroupBase =
      ProductLieGroupBase<Rp_x_SOn_Base, Rp_x_SOn_AmbientBase, Rp, SOn>;
  using _MatrixGroupCommonOps = MatrixGroupCommonOps<Rp_x_SOn_Base<Rp, SOn>>;

 public:
  using _MatrixGroupCommonOps::operator*;
  using _MatrixGroupCommonOps::operator=;
  using _ProductLieGroupBase::operator*;
  using _ProductLieGroupBase::kDim;
  static constexpr int N = SOn::Ambient::RowsAtCompileTime;

  // Import the constructors from the base class
  using _ProductLieGroupBase::_ProductLieGroupBase;

  using Scalar = typename _ProductLieGroupBase::Scalar;

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  Rp_x_SOn_Base(const MatrixXpr& m) {  // NOLINT
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
    using std::sqrt;
    Scalar scale = sqrt(m.squaredNorm() / Scalar(N));
    if constexpr (std::is_same_v<Scalar, double>) {
      LOGA("Rp_x_SOn_Base construct from MatrixXpr: scale:  %f", scale);
    }
    this->template part<0>() = scale;
    this->template part<1>() = SOn(m / scale);
  }

  const Scalar& scale() const {
    return this->template part<0>().value();
  }
  const SOn& rotation() const {
    return this->template part<1>();
  }
  Scalar& scale() {
    return this->template part<0>().value();
  }
  SOn& rotation() {
    return this->template part<1>();
  }
  Eigen::Matrix<Scalar, N, N> matrix() const {
    return this->template part<0>().value() * this->template part<1>().matrix();
  }

 public:  // MatrixGroupCommonOps overrides.
  // Override MatrixGroupCommonOps::JmultVector() for N = 2 and 3.
  // Return
  //    J = d [Exp(x) * v] / d (x)      (Evaluated at x=0, x ∈ LieAlg, v ∈ R^N)
  template <typename = Rp_x_SOn_Base>
  static Eigen::Matrix<Scalar, N, kDim> JmultVector(
      const Vector<N, Scalar>& v) {
    if constexpr (N == 2) {
      Eigen::Matrix<Scalar, N, kDim> J;
      Scalar* Jd = J.data();
      const Scalar& x = v[0];
      const Scalar& y = v[1];
      // clang-format off
      Jd[0] = x;   Jd[2] = -y;
      Jd[1] = y;   Jd[3] = x;
      // clang-format on
      return J;
    } else if constexpr (N == 3) {
      Eigen::Matrix<Scalar, N, kDim> J;
      J << v, SO3<Scalar>::hat(-v);
      return J;
    } else {
      // The default implementation
      return _MatrixGroupCommonOps::JmultVector(v);
    }
  }

 public:
#define Define_TransformVector_For_Rp_x_SOn                                \
  template <                                                               \
      typename _LieGroup, typename VectorXpr, typename ResultVector,       \
      typename JacobianWrtG =                                              \
          Eigen::Matrix<typename _LieGroup::Scalar, Eigen::Dynamic, kDim>> \
  void TransformVector(                                                    \
      const _LieGroup& g, const VectorXpr& v, ResultVector* result,        \
      JacobianWrtG* jacobian_wrt_g = nullptr) const {                      \
    static_assert(kDof == _LieGroup::kDim);                                \
    const auto& rotation_perturbation = this->template part<1>();          \
    /* Alloc memory for jacobian_wrt_g if needed */                        \
    if constexpr (                                                         \
        JacobianWrtG::RowsAtCompileTime == Eigen::Dynamic ||               \
        JacobianWrtG::ColsAtCompileTime == Eigen::Dynamic) {               \
      if (jacobian_wrt_g &&                                                \
          (jacobian_wrt_g->rows() == 0 || jacobian_wrt_g->cols() == 0)) {  \
        jacobian_wrt_g->resize(N, kDof);                                   \
      }                                                                    \
    }                                                                      \
    Vector<N, typename _LieGroup::Scalar> result_v;                        \
    if (jacobian_wrt_g) {                                                  \
      auto jacobian_block_wrt_rotation =                                   \
          jacobian_wrt_g->template rightCols<3>();                         \
      rotation_perturbation.TransformVector(                               \
          g.rotation(), v* g.scale(), &result_v,                           \
          &jacobian_block_wrt_rotation);                                   \
    } else {                                                               \
      rotation_perturbation.TransformVector(                               \
          g.rotation(), v* g.scale(), &result_v);                          \
    }                                                                      \
    if (result) {                                                          \
      *result = result_v;                                                  \
    }                                                                      \
    if (jacobian_wrt_g) {                                                  \
      jacobian_wrt_g->col(0) = result_v;                                   \
    }                                                                      \
  }

  /// Override _ProductLieGroupBase::LeftPerturbation::TransformVector()
  template <typename LieGroup>
  struct LeftPerturbationTemplate
      : public _ProductLieGroupBase::template LeftPerturbationTemplate<
            LieGroup> {
    using _Base =
        typename _ProductLieGroupBase::template LeftPerturbationTemplate<
            LieGroup>;
    using _Base::kDof;
    Define_TransformVector_For_Rp_x_SOn
  };

  /// Override _ProductLieGroupBase::RightPerturbation::TransformVector()
  template <typename LieGroup>
  struct RightPerturbationTemplate
      : public _ProductLieGroupBase::template RightPerturbationTemplate<
            LieGroup> {
    using _Base =
        typename _ProductLieGroupBase::template RightPerturbationTemplate<
            LieGroup>;
    using _Base::kDof;
    Define_TransformVector_For_Rp_x_SOn
  };

  DEFINE_LIE_PERTURBATIONS(Rp_x_SOn_Base);
  DEFINE_LIE_OPTIMIZABLES(Rp_x_SOn_Base);
};

}  // namespace sk4slam
