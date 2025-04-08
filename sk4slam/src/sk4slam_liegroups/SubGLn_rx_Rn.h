#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/liegroup_data_iterator.h"
#include "sk4slam_liegroups/matrix_group_helper.h"
#include "sk4slam_math/vector_space.h"

namespace sk4slam {

namespace affine_group_internal {

/// @brief The default implementation template for SubGLn_rx_Rn
template <typename SubGLn, typename Rn, typename ApproximationOptions>
struct SubGLn_rx_Rn_Impl;

struct SubGLn_rx_Rn_Accurate {
  template <typename Scalar>
  struct ApproximationOptions {
    using Constants = liegroup::Constants<Scalar>;
    static inline const Scalar kEpsForJacobian =
        liegroup::Constants<Scalar>::kEps;
    static inline const Scalar kEpsForLog = liegroup::Constants<Scalar>::kEps;
    static inline const Scalar kEpsForExp = liegroup::Constants<Scalar>::kEps;

    static constexpr int kMaxOrderForJacobian = -1;
    static constexpr int kMaxOrderForLog = -1;
    static constexpr int kMaxOrderForExp = -1;
  };

  template <typename SubGLn, typename Rn>
  struct Impl : public SubGLn_rx_Rn_Impl<
                    SubGLn, Rn, ApproximationOptions<typename SubGLn::Scalar>> {
  };
};

template <int _jacobian_order, int _log_order = -1, int _exp_order = -1>
struct SubGLn_rx_Rn_UpToOrder {
  template <typename Scalar>
  struct ApproximationOptions
      : public SubGLn_rx_Rn_Accurate::ApproximationOptions<Scalar> {
    static constexpr int kMaxOrderForJacobian = _jacobian_order;
    static constexpr int kMaxOrderForLog = _log_order;
    static constexpr int kMaxOrderForExp = _exp_order;
  };

  template <typename SubGLn, typename Rn>
  class Impl : public SubGLn_rx_Rn_Impl<
                   SubGLn, Rn, ApproximationOptions<typename SubGLn::Scalar>> {
  };
};

/// @brief Perturbation for Affine Groups
/// @details
/// Affine groups support the standard LieGroup perturbation types, including
/// LeftPerturbation and RightPerturbation, as defined in the base class @ref
/// LieGroupBase. However, affine groups have a special structure that enables
/// more convenient perturbations by separately perturbing the linear and
/// translation components of an affine transformation.
///
/// The AffinePerturbation class implements this separation strategy:
/// - @ref AffineLeftPerturbation:
///   When @c SubGLnPerturbation is set to LeftPerturbation for the linear
///   part, the linear component is perturbed using the standard left
///   perturbation, and the translation component is perturbed using ordinary
///   vector perturbation (with no distinction between left and right).
/// - @ref AffineRightPerturbation:
///   When @c SubGLnPerturbation is set to RightPerturbation for the linear
///   part, the linear component is perturbed using the standard right
///   perturbation, and the translation component is perturbed using ordinary
///   vector perturbation (with no distinction between left and right).
template <
    typename SubGLn_rx_Rn, typename SubGLnPerturbation, typename RnPerturbation>
class AffinePerturbation;

/// @brief Standard Lie Perturbation for Affine Groups. Extends the default
/// Perturbation class with affine group-specific functionalities.
template <typename SubGLn_rx_Rn, bool _left_perturbation>
class LiePerturbation;

}  // namespace affine_group_internal

// This class template defines the interface of the semi direct product
// of a subgroup of GL(n) and the vector space R^n.
template <
    typename SubGLn, typename Rn,
    typename ImplType = affine_group_internal::SubGLn_rx_Rn_Accurate>
class SubGLn_rx_Rn;

template <
    typename SubGLn, typename Rn, int _jacobian_order, int _log_order = -1,
    int _exp_order = -1>
using SubGLn_rx_Rn_UpToOrder = SubGLn_rx_Rn<
    SubGLn, Rn,
    affine_group_internal::SubGLn_rx_Rn_UpToOrder<
        _jacobian_order, _log_order, _exp_order>>;

////////////////////////////////////

template <typename>
inline constexpr bool Is_SubGLn_rx_Rn_NoExtension = false;

template <typename SubGLn, typename Rn, typename ImplType>
inline constexpr bool
    Is_SubGLn_rx_Rn_NoExtension<SubGLn_rx_Rn<SubGLn, Rn, ImplType>> = true;

template <typename LieGroup>
inline constexpr bool Is_SubGLn_rx_Rn =
    Is_SubGLn_rx_Rn_NoExtension<LieGroup> ||
    Is_SubGLn_rx_Rn<typename LieGroup::ExtensionBase>;

template <>
inline constexpr bool Is_SubGLn_rx_Rn<void> = false;

///////// Implementations //////////

template <typename _SubGLn, typename _Rn, typename ImplType>
class SubGLn_rx_Rn
    : public LieGroupBase<SubGLn_rx_Rn<_SubGLn, _Rn, ImplType>>,
      public MatrixGroupCommonOps<SubGLn_rx_Rn<_SubGLn, _Rn, ImplType>> {
  friend class ImplType::template Impl<_SubGLn, _Rn>;
  using Impl = typename ImplType::template Impl<_SubGLn, _Rn>;
  template <typename, bool>
  friend class affine_group_internal::LiePerturbation;
  template <typename, typename, typename>
  friend class affine_group_internal::AffinePerturbation;

  static_assert(
      std::is_same_v<typename _SubGLn::Scalar, typename _Rn::Scalar>,
      "SubGLn and Rn must have the same scalar type!");

  static_assert(
      MatrixGroupHelper<_SubGLn>::N == _Rn::kDim,
      "The dimension of SubGLn and Rn must match!");

  using _MatrixGroupCommonOps =
      MatrixGroupCommonOps<SubGLn_rx_Rn<_SubGLn, _Rn, ImplType>>;

  template <typename _ScalarType>
  using _CastTemplate = SubGLn_rx_Rn<
      decltype(std::declval<_SubGLn>().template cast<_ScalarType>()),
      decltype(std::declval<_Rn>().template cast<_ScalarType>()), ImplType>;

 public:
  using SubGLn = _SubGLn;
  using Rn = _Rn;
  using _MatrixGroupCommonOps::operator*;
  using _MatrixGroupCommonOps::operator=;

  // Standard LieGroup interfaces

  using Scalar = typename SubGLn::Scalar;

  static constexpr int N = MatrixGroupHelper<SubGLn>::N;

  static constexpr int kDim = SubGLn::kDim + N;

  static constexpr int kAmbientDim = SubGLn::kAmbientDim + Rn::kAmbientDim;

  template <typename SubGLnAmbient, typename RnAmbient>
  class AmbientBase
      : public ProductVectorSpaceBase<AmbientBase, SubGLnAmbient, RnAmbient> {
    using _ProductVectorSpaceBase =
        ProductVectorSpaceBase<AmbientBase, SubGLnAmbient, RnAmbient>;

   public:
    // Import the constructors from the base class
    using _ProductVectorSpaceBase::_ProductVectorSpaceBase;

    using MatrixRepresentation = Eigen::Matrix<Scalar, N + 1, N + 1>;

    MatrixRepresentation matrix() const {
      using Helper = MatrixGroupHelper<SubGLn>;
      Eigen::Matrix<Scalar, N + 1, N + 1> result =
          Eigen::Matrix<Scalar, N + 1, N + 1>::Zero();
      result.template block<N, N>(0, 0) =
          Helper::ambientToMatrix(this->template part<0>());
      result.template block<N, 1>(0, N) = this->template part<1>();
      return result;
    }

    operator MatrixRepresentation() const {
      return matrix();
    }
  };

  using Ambient = AmbientBase<typename SubGLn::Ambient, typename Rn::Ambient>;

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static SubGLn_rx_Rn Identity() {
    return SubGLn_rx_Rn(
        SubGLn::Identity(), Eigen::Matrix<Scalar, N, 1>::Zero());
  }

  SubGLn_rx_Rn operator*(const SubGLn_rx_Rn& other) const {
    return SubGLn_rx_Rn(
        linear_ * other.linear_,
        translation_.vector() + linear_.matrix() * other.translation_.vector());
  }

  SubGLn_rx_Rn inverse() const {
    auto inverse_linear = linear_.inverse();
    return SubGLn_rx_Rn(
        inverse_linear, -inverse_linear.matrix() * translation_.vector());
  }

  bool isApprox(
      const SubGLn_rx_Rn& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return linear_.isApprox(other.linear_, eps) &&
           translation_.isApprox(other.translation_, eps);
  }

  static SubGLn_rx_Rn Exp(const LieAlgebra& X) {
    typename SubGLn::LieAlgebra xi = X.template head<SubGLn::kDim>();
    Eigen::Matrix<Scalar, N, 1> eta = X.template tail<N>();
    return SubGLn_rx_Rn(SubGLn::Exp(xi), Impl::calcV(xi) * eta);
  }

  static LieAlgebra Log(const SubGLn_rx_Rn& g) {
    typename SubGLn::LieAlgebra xi = SubGLn::Log(g.linear_);
    Eigen::Matrix<Scalar, N, 1> eta =
        Impl::calcVinv(xi, &g.linear_) * g.translation_.vector();
    LieAlgebra X;
    X << xi, eta;
    return X;
  }

  static LieAlgebraEndomorphism Ad(const SubGLn_rx_Rn& g) {
    return Impl::Ad(g);
  }

  // Ad(g, X) = Ad(g) * X
  static LieAlgebra Ad(const SubGLn_rx_Rn& g, const LieAlgebra& X) {
    return Impl::Ad(g, X);
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    return Impl::ad(X);
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    return Impl::bracket(X1, X2);
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    return Impl::Jl(X);
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Impl::Jr(X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    return Impl::invJl(X);
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return Impl::invJr(X);
  }

  static Ambient hat(const LieAlgebra& X) {
    Ambient X_hat;
    X_hat.template part<0>() = SubGLn::hat(X.template head<SubGLn::kDim>());
    X_hat.template part<1>() = Rn::hat(X.template tail<N>());
    return X_hat;
  }

  static LieAlgebra vee(const Ambient& X_hat) {
    LieAlgebra X;
    X.template head<SubGLn::kDim>() = SubGLn::vee(X_hat.template part<0>());
    X.template tail<N>() = Rn::vee(X_hat.template part<1>());
    return X;
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    Ambient X_hat = Ambient::Zero();
    if (i < SubGLn::kDim) {
      X_hat.template part<0>() = SubGLn::generator(i);
    } else {
      X_hat.template part<1>() = Rn::generator(i - SubGLn::kDim);
    }
    return X_hat;
  }

  // For use with CeresManifoldBlock
  using DataIterator = LieGroupDataIterator<Scalar>;
  DataIterator data() {
    return DataIterator::ConcatenateDataForLieGroups(linear_, translation_);
  }

  using ConstDataIterator = LieGroupDataIterator<const Scalar>;
  ConstDataIterator data() const {
    return ConstDataIterator::ConcatenateDataForLieGroups(
        linear_, translation_);
  }

  template <typename _ScalarType>
  _CastTemplate<_ScalarType> cast() const {
    return _CastTemplate<_ScalarType>(
        linear_.template cast<_ScalarType>(),
        translation_.template cast<_ScalarType>());
  }

 public:
  // When multiplying with a matrix
  // - If the rhs m is has N rows, then each column of m is treated as a
  //   N-vector and will be transformed by the group element, i.e. the output
  //   will be 'linear() * m + translation()'.
  // - Otherwise, the group element is just treated as a (N+1) x (N+1) matrix.
  //
  template <
      typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
      int _MaxCols>
  decltype(auto) operator*(
      const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>&
          m) const {
    LOGA("operator*(const Matrix&) called.");
    return _mult_matrix_impl(m);
  }

  template <typename UnaryOp, typename ValueType>
  decltype(auto) operator*(
      const Eigen::CwiseUnaryOp<UnaryOp, ValueType>& m) const {
    LOGA("operator*(const CwiseUnaryOp&) called.");
    return _mult_matrix_impl(m);
  }

  template <typename BinaryOp, typename ValueType1, typename ValueType2>
  decltype(auto) operator*(
      const Eigen::CwiseBinaryOp<BinaryOp, ValueType1, ValueType2>& m) const {
    LOGA("operator*(const CwiseBinaryOp&) called.");
    return _mult_matrix_impl(m);
  }

  template <typename Lhs, typename Rhs, int Option>
  decltype(auto) operator*(const Eigen::Product<Lhs, Rhs, Option>& m) const {
    LOGA("operator*(const Product&) called.");
    return _mult_matrix_impl(m);
  }

  template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  decltype(auto) operator*(
      const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& m) const {
    LOGA("operator*(const Block&) called.");
    return _mult_matrix_impl(m);
  }

 public:
  // SubGLn_rx_Rn specific interfaces

  SubGLn_rx_Rn()
      : linear_(SubGLn::Identity()),
        translation_(Eigen::Matrix<Scalar, N, 1>::Zero()) {}

  template <typename LinearArg, typename TranslationArg>
  SubGLn_rx_Rn(LinearArg&& linear, TranslationArg&& translation)
      : linear_(std::forward<LinearArg>(linear)),
        translation_(std::forward<TranslationArg>(translation)) {}

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  SubGLn_rx_Rn(const MatrixXpr& m) {  // NOLINT
    static_assert(
        (MatrixXpr::ColsAtCompileTime == N + 1 ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == N ||
             MatrixXpr::RowsAtCompileTime == N + 1 ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == N + 1);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == N || m.rows() == N + 1);
    }
    linear_ = m.template block<N, N>(0, 0);
    translation_ = m.template block<N, 1>(0, N);
  }

  SubGLn_rx_Rn(const SubGLn_rx_Rn& other)
      : linear_(other.linear_), translation_(other.translation_) {}

  // The semi direct product of SubGLn and Rn is still a matrix group (a Lie
  // subgroup of GL(N+1))
  Eigen::Matrix<Scalar, N + 1, N + 1> matrix() const {
    Eigen::Matrix<Scalar, N + 1, N + 1> result =
        Eigen::Matrix<Scalar, N + 1, N + 1>::Zero();
    result.template block<N, N>(0, 0) = this->linear_.matrix();
    result.template block<N, 1>(0, N) = this->translation_.vector();
    result(N, N) = liegroup::Constants<Scalar>::kNum_1;
    return result;
  }

  const SubGLn& linear() const {
    return linear_;
  }

  SubGLn& linear() {
    return linear_;
  }

  const typename Rn::Ambient& translation() const {
    return translation_.vector();
  }

  typename Rn::Ambient& translation() {
    return translation_.vector();
  }

  template <int _part_idx>
  auto& part() {
    if constexpr (_part_idx == 0) {
      return linear_;
    } else {
      return translation_;
    }
  }

  template <int _part_idx>
  const auto& part() const {
    if constexpr (_part_idx == 0) {
      return linear_;
    } else {
      return translation_;
    }
  }

 public:
  /// @brief  Override LeftPerturbationTemplate
  template <typename LieGroup>
  using LeftPerturbationTemplate =
      affine_group_internal::LiePerturbation<LieGroup, true>;

  /// @brief  Override RightPerturbationTemplate
  template <typename LieGroup>
  using RightPerturbationTemplate =
      affine_group_internal::LiePerturbation<LieGroup, false>;

  /// @brief  Define AffinePerturbationTemplate
  template <
      typename LieGroup, typename SubGLnPerturbation, typename RnPerturbation>
  using AffinePerturbationTemplate = affine_group_internal::AffinePerturbation<
      LieGroup, SubGLnPerturbation, RnPerturbation>;

  /// @brief  Define AffineLeftPerturbationTemplate
  template <typename LieGroup>
  using AffineLeftPerturbationTemplate = AffinePerturbationTemplate<
      LieGroup, typename SubGLn::LeftPerturbation,
      typename Rn::LeftPerturbation>;

  /// @brief  Define AffineRightPerturbationTemplate
  template <typename LieGroup>
  using AffineRightPerturbationTemplate = AffinePerturbationTemplate<
      LieGroup, typename SubGLn::RightPerturbation,
      typename Rn::RightPerturbation>;

#define DEFINE_AFFINE_PERTURBATIONS(LieGroup)                                  \
  template <typename SubGLnPerturbation, typename RnPerturbation>              \
  using AffinePerturbation = affine_group_internal::AffinePerturbation<        \
      LieGroup, SubGLnPerturbation, RnPerturbation>;                           \
  using AffineLeftPerturbation = AffineLeftPerturbationTemplate<LieGroup>;     \
  using AffineRightPerturbation = AffineRightPerturbationTemplate<LieGroup>;   \
  template <typename SubSpaceType>                                             \
  using SeparateSubLeftPerturbation = liegroup_internal::SubSpacePerturbation< \
      AffineLeftPerturbation, SubSpaceType>;                                   \
  template <typename SubSpaceType>                                             \
  using SeparateSubRightPerturbation =                                         \
      liegroup_internal::SubSpacePerturbation<                                 \
          AffineRightPerturbation, SubSpaceType>;

#define DEFINE_AFFINE_OPTIMIZABLES(LieGroup)                           \
  template <typename SubGLnOptimizable, typename RnOptimizable>        \
  using SeparateOptimizable = OptimizableManifold<                     \
      LieGroup, AffinePerturbation<SubGLnOptimizable, RnOptimizable>>; \
  using SeparateLeftOptimizable =                                      \
      OptimizableManifold<LieGroup, AffineLeftPerturbation>;           \
  using SeparateRightOptimizable =                                     \
      OptimizableManifold<LieGroup, AffineRightPerturbation>;          \
  template <typename SubSpaceType, bool _share_perturbation = true>    \
  using SeparateSubLeftOptimizable = OptimizableManifold<              \
      LieGroup, SeparateSubLeftPerturbation<SubSpaceType>,             \
      _share_perturbation>;                                            \
  template <typename SubSpaceType, bool _share_perturbation = true>    \
  using SeparateSubRightOptimizable = OptimizableManifold<             \
      LieGroup, SeparateSubRightPerturbation<SubSpaceType>,            \
      _share_perturbation>;

  DEFINE_LIE_PERTURBATIONS(SubGLn_rx_Rn)
  DEFINE_LIE_OPTIMIZABLES(SubGLn_rx_Rn)

  DEFINE_AFFINE_PERTURBATIONS(SubGLn_rx_Rn)
  DEFINE_AFFINE_OPTIMIZABLES(SubGLn_rx_Rn)

  /// @brief  Override the Extension template to make AffinePerturbation
  ///         available for the extension types.
  template <
      template <typename Scalar, typename...> class DerivedExtensionTemplate,
      typename Scalar, typename... Rest>
  struct Extension
      : public LieGroupExtension<
            SubGLn_rx_Rn, DerivedExtensionTemplate, Scalar, Rest...> {
    using Base = LieGroupExtension<
        SubGLn_rx_Rn, DerivedExtensionTemplate, Scalar, Rest...>;
    using Base::Base;
    using DerivedExtension = DerivedExtensionTemplate<Scalar, Rest...>;

    /// @brief  Define AffinePerturbationTemplate
    template <
        typename LieGroup, typename SubGLnPerturbation, typename RnPerturbation>
    using AffinePerturbationTemplate =
        typename Base::template AffinePerturbationTemplate<
            LieGroup, SubGLnPerturbation, RnPerturbation>;

    /// @brief  Define AffineLeftPerturbationTemplate
    template <typename LieGroup>
    using AffineLeftPerturbationTemplate =
        typename Base::template AffineLeftPerturbationTemplate<LieGroup>;

    /// @brief  Define AffineRightPerturbationTemplate
    template <typename LieGroup>
    using AffineRightPerturbationTemplate =
        typename Base::template AffineRightPerturbationTemplate<LieGroup>;

    DEFINE_AFFINE_PERTURBATIONS(DerivedExtension)
    DEFINE_AFFINE_OPTIMIZABLES(DerivedExtension)
  };

 protected:
  template <typename MatrixXpr>
  Eigen::Matrix<
      Scalar, MatrixXpr::RowsAtCompileTime, MatrixXpr::ColsAtCompileTime>
  _mult_matrix_impl(const MatrixXpr& m) const {
    using ReturnMatrix = Eigen::Matrix<
        Scalar, MatrixXpr::RowsAtCompileTime, MatrixXpr::ColsAtCompileTime>;
    if constexpr (MatrixXpr::RowsAtCompileTime == N) {
      if constexpr (MatrixXpr::ColsAtCompileTime == 1) {
        // LOGA("_mult_matrix_impl <N, 1> A");
        return ReturnMatrix(linear_.matrix() * m + translation_.vector());
      } else {
        Eigen::Matrix<Scalar, N, MatrixXpr::ColsAtCompileTime> ts;
        if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
          ts.resize(N, m.cols());
        }
        // LOGA("_mult_matrix_impl <N, %d> B", m.cols());

        // set ts' every column to translation_
        const auto& t = translation_.vector();
        for (int i = 0; i < ts.cols(); ++i) {
          ts.col(i) = t;
        }
        return ReturnMatrix(linear_.matrix() * m + ts);
      }
    } else if constexpr (MatrixXpr::RowsAtCompileTime == N + 1) {
      // LOGA("_mult_matrix_impl <N+1, %d> C", m.cols());
      // LOGA("m:\n%s", toStr(m).c_str());
      // LOGA("matrix():\n%s", toStr(matrix()).c_str());
      // LOGA("matrix() * m:\n%s", toStr(matrix() * m).c_str());
      return ReturnMatrix(matrix() * m);
    } else {
      using ReturnMatrix =
          Eigen::Matrix<Scalar, Eigen::Dynamic, MatrixXpr::ColsAtCompileTime>;
      if (m.rows() == N) {
        if constexpr (MatrixXpr::ColsAtCompileTime == 1) {
          // LOGA("_mult_matrix_impl <N, 1> D");
          return ReturnMatrix(linear_.matrix() * m + translation_.vector());
        } else {
          Eigen::Matrix<Scalar, N, MatrixXpr::ColsAtCompileTime> ts;
          if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
            ts.resize(N, m.cols());
          }
          // LOGA("_mult_matrix_impl <N, %d> E", m.cols());

          // set ts' every column to translation_
          const auto& t = translation_.vector();
          for (int i = 0; i < ts.cols(); ++i) {
            ts.col(i) = t;
          }
          return ReturnMatrix(linear_.matrix() * m + ts);
        }
      } else {
        ASSERT(m.rows() == N + 1);
        // LOGA("_mult_matrix_impl <N+1, %d> F", m.cols());
        return ReturnMatrix(matrix() * m);
      }
    }
  }

 private:
  SubGLn linear_;
  Rn translation_;
};

/////////// Implementation /////////////

namespace affine_group_internal {

template <typename SubGLn, typename Rn, typename ApproximationOptions>
struct SubGLn_rx_Rn_Impl {
  using DefaultLieGroup = SubGLn_rx_Rn<SubGLn, Rn>;
  using LieAlgebraEndomorphism =
      typename DefaultLieGroup::LieAlgebraEndomorphism;
  using LieAlgebra = typename DefaultLieGroup::LieAlgebra;
  using Scalar = typename DefaultLieGroup::Scalar;
  static constexpr int N = DefaultLieGroup::N;
  static constexpr int kDim = DefaultLieGroup::kDim;

  template <typename LieGroup>
  static LieAlgebraEndomorphism Ad(const LieGroup& g) {
    static const Eigen::Matrix<Scalar, SubGLn::kDim, N> Zero_corner =
        Eigen::Matrix<Scalar, SubGLn::kDim, N>::Zero();
    const Eigen::Matrix<Scalar, N, N>& M = g.linear().matrix();
    const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
    Eigen::Matrix<Scalar, N, SubGLn::kDim> tensor_dot_t;

    typename SubGLn::LieAlgebraEndomorphism Ad_M = SubGLn::Ad(g.linear());
    const Eigen::Matrix<Scalar, SubGLn::kDim, SubGLn::kDim>& Ad_M_mat = Ad_M;
    for (size_t i = 0; i < SubGLn::kDim; i++) {
      Eigen::Matrix<Scalar, N, N> hat_Ad_M_i = SubGLn::hat(-Ad_M_mat.col(i));
      tensor_dot_t.col(i) = hat_Ad_M_i * t;
    }

    LieAlgebraEndomorphism Adj;

    // clang-format off
    Adj <<       Ad_M_mat,    Zero_corner,
             tensor_dot_t,    M;
    // clang-format on
    return Adj;
  }

  // Ad(g, X) = Ad(g) * X
  template <typename LieGroup>
  static LieAlgebra Ad(const LieGroup& g, const LieAlgebra& X) {
    const Eigen::Matrix<Scalar, N, N>& M = g.linear().matrix();
    const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
    // typename SubGLn::LieAlgebraEndomorphism Ad_M = SubGLn::Ad(g.linear());

    typename SubGLn::LieAlgebra ret_head =
        SubGLn::Ad(g.linear(), X.template head<SubGLn::kDim>());
    Eigen::Matrix<Scalar, N, N> hat_ret_head = SubGLn::hat(ret_head);
    LieAlgebra ret;
    ret.template head<SubGLn::kDim>() = ret_head;
    ret.template tail<N>() = hat_ret_head * (-t) + M * X.template tail<N>();
    return ret;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    static const Eigen::Matrix<Scalar, SubGLn::kDim, N> Zero_corner =
        Eigen::Matrix<Scalar, SubGLn::kDim, N>::Zero();
    typename SubGLn::LieAlgebra xi = X.template head<SubGLn::kDim>();
    Eigen::Matrix<Scalar, N, 1> eta = X.template tail<N>();
    Eigen::Matrix<Scalar, N, SubGLn::kDim> tensor_dot_eta =
        -SubGLn::JmultVector(eta);

    LieAlgebraEndomorphism adj;

    Eigen::Matrix<Scalar, SubGLn::kDim, SubGLn::kDim> ad_xi = SubGLn::ad(xi);
    Eigen::Matrix<Scalar, N, N> hat_xi = SubGLn::hat(xi);

    // clang-format off
    adj <<          ad_xi, Zero_corner,
           tensor_dot_eta, hat_xi;
    // clang-format on
    return adj;
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    LieAlgebra X;
    X.template head<SubGLn::kDim>() = SubGLn::bracket(
        X1.template head<SubGLn::kDim>(), X2.template head<SubGLn::kDim>());

    const Eigen::Matrix<Scalar, N, N>& hat_xi_1 =
        SubGLn::hat(X1.template head<SubGLn::kDim>());
    const Eigen::Matrix<Scalar, N, N>& hat_xi_2 =
        SubGLn::hat(X2.template head<SubGLn::kDim>());

    X.template tail<N>() =
        hat_xi_1 * X2.template tail<N>() - hat_xi_2 * X1.template tail<N>();
    return X;
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    typename SubGLn::LieAlgebra xi = X.template head<SubGLn::kDim>();
    Eigen::Matrix<Scalar, N, 1> eta = X.template tail<N>();

    Eigen::Matrix<Scalar, SubGLn::kDim, SubGLn::kDim> Jl_xi = SubGLn::Jl(xi);

    Eigen::Matrix<Scalar, N, SubGLn::kDim> Q = calcQ(xi, eta);
    static const Eigen::Matrix<Scalar, SubGLn::kDim, N> Zero_corner =
        Eigen::Matrix<Scalar, SubGLn::kDim, N>::Zero();
    Eigen::Matrix<Scalar, N, N> V_xi = calcV(
        xi, ApproximationOptions::kMaxOrderForJacobian,
        ApproximationOptions::kEpsForJacobian);

    LieAlgebraEndomorphism J;
    // clang-format off
    J <<  Jl_xi,  Zero_corner,
              Q,  V_xi;
    // clang-format on
    return J;
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return Jl(-X);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    typename SubGLn::LieAlgebra xi = X.template head<SubGLn::kDim>();
    Eigen::Matrix<Scalar, N, 1> eta = X.template tail<N>();

    Eigen::Matrix<Scalar, SubGLn::kDim, SubGLn::kDim> invJl_xi =
        SubGLn::invJl(xi);

    Eigen::Matrix<Scalar, N, SubGLn::kDim> Q = calcQ(xi, eta);
    static const Eigen::Matrix<Scalar, SubGLn::kDim, N> Zero_corner =
        Eigen::Matrix<Scalar, SubGLn::kDim, N>::Zero();

    Eigen::Matrix<Scalar, N, N> invV_xi = calcVinv(
        xi, nullptr, ApproximationOptions::kMaxOrderForJacobian,
        ApproximationOptions::kEpsForJacobian);

    LieAlgebraEndomorphism invJ;
    // clang-format off
    invJ <<                 invJl_xi,  Zero_corner,
            - invV_xi * Q * invJl_xi,  invV_xi;
    // clang-format on
    return invJ;
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return invJl(-X);
  }

  static Eigen::Matrix<Scalar, N, N> calcV(
      const typename SubGLn::LieAlgebra& xi) {
    return calcV(
        xi, ApproximationOptions::kMaxOrderForExp,
        ApproximationOptions::kEpsForExp);
  }

  static Eigen::Matrix<Scalar, N, N> calcVinv(
      const typename SubGLn::LieAlgebra& xi, const SubGLn* linear) {
    return calcVinv(
        xi, linear, ApproximationOptions::kMaxOrderForLog,
        ApproximationOptions::kEpsForLog);
  }

 protected:
  static Eigen::Matrix<Scalar, N, SubGLn::kDim> calcQ(
      const typename SubGLn::LieAlgebra& xi,
      const Eigen::Matrix<Scalar, N, 1>& eta) {
    return calcQ(
        xi, eta, ApproximationOptions::kMaxOrderForJacobian,
        ApproximationOptions::kEpsForJacobian);
  }

  static Eigen::Matrix<Scalar, N, N> calcV(
      const typename SubGLn::LieAlgebra& xi, const int max_order,
      const Scalar eps) {
    const Eigen::Matrix<Scalar, N, N>& hat_xi = SubGLn::hat(xi);
    return expm1OverXOnAlgebra(hat_xi, max_order, eps);
  }

  static Eigen::Matrix<Scalar, N, N> calcVinv(
      const typename SubGLn::LieAlgebra& xi, const SubGLn* linear,
      const int max_order, const Scalar eps) {
    return calcV(xi, max_order, eps).inverse();
  }

  static Eigen::Matrix<Scalar, N, SubGLn::kDim> calcQ(
      const typename SubGLn::LieAlgebra& xi,
      const Eigen::Matrix<Scalar, N, 1>& eta, const int max_order,
      const Scalar eps) {
    const Eigen::Matrix<Scalar, N, N>& hat_xi = SubGLn::hat(xi);
    const Eigen::Matrix<Scalar, SubGLn::kDim, SubGLn::kDim>& ad_xi =
        SubGLn::ad(xi);
    const Eigen::Matrix<Scalar, N, SubGLn::kDim>& H1 =
        -SubGLn::JmultVector(eta);
    Eigen::Matrix<Scalar, N, SubGLn::kDim> Hk = H1;
    Eigen::Matrix<Scalar, N, SubGLn::kDim> H1_adxi_pow_km1 = H1;
    Scalar denominator_k =
        liegroup::Constants<Scalar>::kNum_2;  // denominator_k = 1/(k+1)!
    Eigen::Matrix<Scalar, N, SubGLn::kDim> Q = Hk / denominator_k;
    int max_iter;
    if (max_order < 0) {
      max_iter = std::numeric_limits<int>::max();
    } else {
      max_iter = max_order + 1;
    }

    int k;
    for (k = 2; k < max_iter; k++) {
      const auto& H_km1 = Hk;
      H1_adxi_pow_km1 *= ad_xi;
      Hk = hat_xi * H_km1 + H1_adxi_pow_km1;
      denominator_k *= Scalar(k + 1);
      auto delta = Hk / denominator_k;
      Q += delta;
      // if (max_order < 0) {
      Scalar max_abs = delta.array().abs().maxCoeff();
      if (max_abs < eps) {
        break;
      }
      // }
    }
    if (k == max_iter) {
      LOGA("SubGLn_rx_Rn::calcQ max iter reached");
      k -= 1;
    }
    LOGA("SubGLn_rx_Rn::calcQ order: %d", k);
    return Q;
  }

  static const Eigen::Matrix<Scalar, N, N>& matrixGeneratorOfSubGLn(int i) {
    static std::vector<Eigen::Matrix<Scalar, N, N>> SubGLn_generators =
        computeMatrixGeneratorsForSubGLn();
    return SubGLn_generators[i];
  }

 private:
  static std::vector<Eigen::Matrix<Scalar, N, N>>
  computeMatrixGeneratorsForSubGLn() {
    std::vector<Eigen::Matrix<Scalar, N, N>> SubGLn_generators(SubGLn::kDim);
    for (int i = 0; i < SubGLn::kDim; ++i) {
      SubGLn_generators[i] = SubGLn::generator(i);
    }
    return SubGLn_generators;
  }
};

template <typename SubGLn_rx_Rn, bool _left_perturbation>
class LiePerturbation
    : public liegroup_internal::Perturbation<SubGLn_rx_Rn, _left_perturbation>::
          template Extension<
              LiePerturbation<SubGLn_rx_Rn, _left_perturbation>> {
  using _Base = typename liegroup_internal::
      Perturbation<SubGLn_rx_Rn, _left_perturbation>::template Extension<
          LiePerturbation<SubGLn_rx_Rn, _left_perturbation>>;
  using RetractionInterface::ExtendTransformJacobianTypes;
  using ThisPerturbation = LiePerturbation;
  using AffineLeftPerturbation = typename SubGLn_rx_Rn::AffineLeftPerturbation;
  using AffineRightPerturbation =
      typename SubGLn_rx_Rn::AffineRightPerturbation;
  using Impl = typename SubGLn_rx_Rn::Impl;
  using SubGLn = RawType<decltype(std::declval<SubGLn_rx_Rn>().linear())>;
  static const auto* getLinearPerturbation() {
    if constexpr (_left_perturbation) {
      return RetractionInterface::defaultInstance<SubGLn::LeftPerturbation>();
    } else {
      return RetractionInterface::defaultInstance<SubGLn::RightPerturbation>();
    }
  }

 public:
  using _Base::_Base;
  using _Base::kDof;
  using _Base::transformJacobian;
  using typename _Base::Scalar;
  using TransformJacobianTypes = ExtendTransformJacobianTypes<
      _Base, AffineLeftPerturbation, AffineRightPerturbation>;
  static constexpr int N = SubGLn_rx_Rn::N;
  static constexpr int kLinearDim = SubGLn_rx_Rn::kDim - N;

  /// @brief  Convert a perturbation from this type to another.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be one of
  ///           @ref LeftPerturbation, @ref RightPerturbation,
  ///           @ref AffineLeftPerturbation, or
  ///           @ref AffineRightPerturbation.
  /// @tparam _Tangent
  ///           Any type convertible to a tangent (Lie algebra) vector.
  /// @param delta
  ///           the perturbation vector to convert.
  /// @return  the converted perturbation vector.
  template <typename ToPerturbation, typename _Tangent>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> convertPerturbation(
      const _Tangent& delta, const SubGLn_rx_Rn& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    if constexpr (std::is_same_v<ThisPerturbation, ToPerturbation>) {
      return delta;
    } else if constexpr (std::is_same_v<  // NOLINT
                             ToPerturbation, AffineRightPerturbation>) {
      if constexpr (_left_perturbation) {
        // left perturbation -> separate right perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            delta.template head<kLinearDim>();
        Eigen::Matrix<Scalar, N, 1> eta = delta.template tail<N>();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        result << SubGLn::Ad(g.linear().inverse(), xi),
            SubGLn::Exp(xi) * t + Impl::calcV(xi) * eta - t;
        return result;
      } else {
        // right perturbation -> separate right perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            delta.template head<kLinearDim>();
        Eigen::Matrix<Scalar, N, 1> eta = delta.template tail<N>();
        result << xi, g.linear() * (Impl::calcV(xi) * eta);
        return result;
      }
    } else if constexpr (std::is_same_v<  // NOLINT
                             ToPerturbation, AffineLeftPerturbation>) {
      if constexpr (_left_perturbation) {
        // left perturbation -> separate left perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            delta.template head<kLinearDim>();
        Eigen::Matrix<Scalar, N, 1> eta = delta.template tail<N>();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        result << xi, SubGLn::Exp(xi) * t + Impl::calcV(xi) * eta - t;
        return result;
      } else {
        // right perturbation -> separate left perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            delta.template head<kLinearDim>();
        Eigen::Matrix<Scalar, N, 1> eta = delta.template tail<N>();
        result << SubGLn::Ad(g.linear(), xi),
            g.linear() * (Impl::calcV(xi) * eta);
        return result;
      }
    } else {
      // left perturbation <-> right perturbation
      return _Base::template convertPerturbation<ToPerturbation>(
          delta, g, to_perturbation);
    }
  }

  /// @brief  Convert a perturbation from this type to another.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be one of
  ///           @ref LeftPerturbation, @ref RightPerturbation,
  ///           @ref AffineLeftPerturbation, or
  ///           @ref AffineRightPerturbation.
  /// @return  A transformation matrix that can convert INFINITESIMAL
  ///          perturbations from the source perturbation type to
  ///          the target perturbation type.
  /// @note
  ///         For conversion between LeftPerturbation and RightPerturbation,
  ///         or that between AffineLeftPerturbation and
  ///         AffineRightPerturbation, the resulting matrix can even be
  ///         applied to non-infinitesimal perturbations.
  template <typename ToPerturbation>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> convertPerturbation(
      const SubGLn_rx_Rn& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    using LieAlgebraEndomorphism =
        typename SubGLn_rx_Rn::LieAlgebraEndomorphism;

    if constexpr (std::is_same_v<ThisPerturbation, ToPerturbation>) {
      return LieAlgebraEndomorphism::Identity();
    } else if constexpr (std::is_same_v<  // NOLINT
                             ToPerturbation, AffineRightPerturbation>) {
      if constexpr (_left_perturbation) {
        // left perturbation -> separate right perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Identity();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        SubGLn linear_inv = g.linear().inverse();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinearInv =
            SubGLn::Ad(linear_inv);
        result.template block<kLinearDim, kLinearDim>(0, 0) = AdLinearInv;
        result.template block<N, kLinearDim>(kLinearDim, 0) =
            SubGLn::JmultVector(t);
        return result;
      } else {
        // right perturbation -> separate right perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Identity();
        result.template block<N, N>(kLinearDim, kLinearDim) =
            g.linear().matrix();
        return result;
      }
    } else if constexpr (std::is_same_v<  // NOLINT
                             ToPerturbation, AffineLeftPerturbation>) {
      if constexpr (_left_perturbation) {
        // left perturbation -> separate left perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Identity();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        result.template block<N, kLinearDim>(kLinearDim, 0) =
            SubGLn::JmultVector(t);
        return result;
      } else {
        // right perturbation -> separate left perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Zero();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinear =
            SubGLn::Ad(g.linear());
        result.template block<kLinearDim, kLinearDim>(0, 0) = AdLinear;
        result.template block<N, N>(kLinearDim, kLinearDim) =
            g.linear().matrix();
        return result;
      }
    } else {
      // left perturbation <-> right perturbation
      return _Base::template convertPerturbation<ToPerturbation>(
          g, to_perturbation);
    }
  }

  template <
      typename SrcPerturbation, typename JacobianMatrixWrtSrcPerturbation,
      typename JacobianMatrixWrtThisPerturbation>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcPerturbation& jacobian_under_src_perturbation,
      const SubGLn_rx_Rn& g,
      JacobianMatrixWrtThisPerturbation* jacobian_under_this_perturbation,
      const SrcPerturbation& src_perturbation = SrcPerturbation()) const {
    if constexpr (std::is_same_v<ThisPerturbation, SrcPerturbation>) {
      if constexpr (std::is_same_v<
                        JacobianMatrixWrtSrcPerturbation,
                        JacobianMatrixWrtThisPerturbation>) {
        if (jacobian_under_this_perturbation ==
            &jacobian_under_src_perturbation) {
          return;
        }
      }
      *jacobian_under_this_perturbation = jacobian_under_src_perturbation;
      return;
    } else if constexpr (std::is_same_v<  // NOLINT
                             SrcPerturbation, AffineRightPerturbation>) {
      if constexpr (_left_perturbation) {
        // left perturbation -> separate right perturbation
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        SubGLn linear_inv = g.linear().inverse();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinearInv =
            SubGLn::Ad(linear_inv);
        Eigen::Matrix<Scalar, SubGLn_rx_Rn::kDim, kLinearDim> tmp;
        tmp << AdLinearInv, SubGLn::JmultVector(t);
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation * tmp;
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>();
        return;
      } else {
        // right perturbation -> separate right perturbation
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation.template leftCols<kLinearDim>();
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>() *
            g.linear().matrix();
        return;
      }
    } else if constexpr (std::is_same_v<  // NOLINT
                             SrcPerturbation, AffineLeftPerturbation>) {
      if constexpr (_left_perturbation) {
        // left perturbation -> separate left perturbation
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        Eigen::Matrix<Scalar, N, kLinearDim> tmp;
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation.template leftCols<kLinearDim>() +
            jacobian_under_src_perturbation.template rightCols<N>() *
                SubGLn::JmultVector(t);
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>();
        return;
      } else {
        // right perturbation -> separate left perturbation
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinear =
            SubGLn::Ad(g.linear());
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation.template leftCols<kLinearDim>() *
            AdLinear;
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>() *
            g.linear().matrix();
        return;
      }
    } else {
      _Base::transformJacobianImpl(
          jacobian_under_src_perturbation, g, jacobian_under_this_perturbation,
          src_perturbation);
      return;
    }
  }

  template <
      typename _LieGroup, typename VectorXpr, typename ResultVector,
      typename JacobianWrtG =
          Eigen::Matrix<typename _LieGroup::Scalar, Eigen::Dynamic, kDof>>
  void TransformHomoVector(
      const _LieGroup& g, const VectorXpr& v, ResultVector* result,
      JacobianWrtG* jacobian_wrt_g = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    static_assert(_LieGroup::kDim == SubGLn::kDim + N);
    using Scalar = typename _LieGroup::Scalar;
    // Alloc memory for jacobian_wrt_g if needed
    if constexpr (
        JacobianWrtG::RowsAtCompileTime == Eigen::Dynamic ||
        JacobianWrtG::ColsAtCompileTime == Eigen::Dynamic) {
      if (jacobian_wrt_g &&
          (jacobian_wrt_g->rows() == 0 || jacobian_wrt_g->cols() == 0)) {
        jacobian_wrt_g->resize(N, kDof);
      }
    }

    if constexpr (_left_perturbation) {
      Vector<N, Scalar> transformed_v = g * v;
      if (result) {
        *result = transformed_v;
      }
      if (jacobian_wrt_g) {
        jacobian_wrt_g->template leftCols<SubGLn::kDim>() =
            SubGLn::JmultVector(transformed_v);
        jacobian_wrt_g->template rightCols<N>() =
            Eigen::Matrix<Scalar, N, N>::Identity();
      }
    } else {
      if (result) {
        *result = g * v;
      }
      if (jacobian_wrt_g) {
        const auto& gm = g.linear().matrix();
        jacobian_wrt_g->template leftCols<SubGLn::kDim>() =
            gm * SubGLn::JmultVector(v);
        jacobian_wrt_g->template rightCols<N>() = gm;
      }
    }
  }
};

template <
    typename SubGLn_rx_Rn, typename SubGLnPerturbation, typename RnPerturbation>
class AffinePerturbation
    : public liegroup_internal::ProductPerturbationBase<
          AffinePerturbation<SubGLn_rx_Rn, SubGLnPerturbation, RnPerturbation>,
          SubGLn_rx_Rn, SubGLnPerturbation, RnPerturbation> {
  using _Base = typename liegroup_internal::ProductPerturbationBase<
      AffinePerturbation<SubGLn_rx_Rn, SubGLnPerturbation, RnPerturbation>,
      SubGLn_rx_Rn, SubGLnPerturbation, RnPerturbation>;

  using RetractionInterface::DeclareTransformJacobianTypes;

  using ThisPerturbation = AffinePerturbation;
  using LeftPerturbation = typename SubGLn_rx_Rn::LeftPerturbation;
  using RightPerturbation = typename SubGLn_rx_Rn::RightPerturbation;
  using AffineLeftPerturbation = typename SubGLn_rx_Rn::AffineLeftPerturbation;
  using AffineRightPerturbation =
      typename SubGLn_rx_Rn::AffineRightPerturbation;
  using Impl = typename SubGLn_rx_Rn::Impl;
  using SubGLn = RawType<decltype(std::declval<SubGLn_rx_Rn>().linear())>;

  static constexpr bool kIsAffineLeftPerturbation =
      std::is_base_of_v<ThisPerturbation, AffineLeftPerturbation>;
  static constexpr bool kIsAffineRightPerturbation =
      std::is_base_of_v<ThisPerturbation, AffineRightPerturbation>;

  static inline AffineLeftPerturbation sep_left_perturb =
      AffineLeftPerturbation();
  static inline AffineRightPerturbation sep_right_perturb =
      AffineRightPerturbation();

 public:
  using _Base::_Base;
  using _Base::kDof;
  using _Base::transformJacobian;
  using typename _Base::Scalar;
  using TransformJacobianTypes =
      DeclareTransformJacobianTypes<LeftPerturbation, RightPerturbation>;
  static constexpr int N = SubGLn_rx_Rn::N;
  static constexpr int kLinearDim = SubGLn_rx_Rn::kDim - N;

  /// @brief  Convert a perturbation from this type to another.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be one of
  ///           @ref LeftPerturbation, @ref RightPerturbation,
  ///           @ref AffineLeftPerturbation, or
  ///           @ref AffineRightPerturbation.
  /// @tparam _Tangent
  ///           Any type convertible to a tangent (Lie algebra) vector.
  /// @param delta
  ///           the perturbation vector to convert.
  /// @return  the converted perturbation vector.
  template <typename ToPerturbation, typename _Tangent>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> convertPerturbation(
      const _Tangent& delta, const SubGLn_rx_Rn& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    if constexpr (std::is_base_of_v<ThisPerturbation, ToPerturbation>) {
      return delta;
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             ToPerturbation, RightPerturbation>) {
      if constexpr (kIsAffineLeftPerturbation) {
        // separate left perturbation -> right perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            SubGLn::Ad(g.linear().inverse(), delta.template head<kLinearDim>());
        Eigen::Matrix<Scalar, N, 1> delta_t = delta.template tail<N>();
        result << xi,
            Impl::calcVinv(xi, nullptr) * (g.linear().inverse() * delta_t);
        return result;
      } else if constexpr (kIsAffineRightPerturbation) {
        // separate right perturbation -> right perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            delta.template head<kLinearDim>();
        Eigen::Matrix<Scalar, N, 1> delta_t = delta.template tail<N>();
        result << xi,
            Impl::calcVinv(xi, nullptr) * (g.linear().inverse() * delta_t);
        return result;
      } else {
        // Convert to separate right perturbation first, then to the
        // target right perturbation
        return sep_right_perturb.template convertPerturbation<ToPerturbation>(
            this->template convertPerturbation<AffineRightPerturbation>(
                delta, g),
            g);
      }
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             ToPerturbation, LeftPerturbation>) {
      if constexpr (kIsAffineLeftPerturbation) {
        // separate left perturbation -> left perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            delta.template head<kLinearDim>();
        Eigen::Matrix<Scalar, N, 1> delta_t = delta.template tail<N>();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        result << xi,
            Impl::calcVinv(xi, nullptr) * (delta_t + t - SubGLn::Exp(xi) * t);
        return result;
      } else if constexpr (kIsAffineRightPerturbation) {
        // separate right perturbation -> left perturbation
        Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
        Eigen::Matrix<Scalar, kLinearDim, 1> xi =
            SubGLn::Ad(g.linear(), delta.template head<kLinearDim>());
        Eigen::Matrix<Scalar, N, 1> delta_t = delta.template tail<N>();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        result << xi,
            Impl::calcVinv(xi, nullptr) * (delta_t + t - SubGLn::Exp(xi) * t);
        return result;
      } else {
        // Convert to separate left perturbation first, then to the
        // target left perturbation
        return sep_left_perturb.template convertPerturbation<ToPerturbation>(
            this->template convertPerturbation<AffineLeftPerturbation>(
                delta, g),
            g);
      }
    } else {
      // seprate left perturbation <-> seprate right perturbation
      return _Base::template convertPerturbation<ToPerturbation>(delta, g);
    }
  }

  /// @brief  Convert a perturbation from this type to another.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be one of
  ///           @ref LeftPerturbation, @ref RightPerturbation,
  ///           @ref AffineLeftPerturbation, or
  ///           @ref AffineRightPerturbation.
  /// @return  A transformation matrix that can convert INFINITESIMAL
  ///          perturbations from the source perturbation type to
  ///          the target perturbation type.
  /// @note
  ///         For conversion between LeftPerturbation and RightPerturbation,
  ///         or that between AffineLeftPerturbation and
  ///         AffineRightPerturbation, the resulting matrix can even be
  ///         applied to non-infinitesimal perturbations.
  template <typename ToPerturbation>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> convertPerturbation(
      const SubGLn_rx_Rn& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    using LieAlgebraEndomorphism =
        typename SubGLn_rx_Rn::LieAlgebraEndomorphism;
    if constexpr (std::is_base_of_v<ThisPerturbation, ToPerturbation>) {
      return LieAlgebraEndomorphism::Identity();
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             ToPerturbation, RightPerturbation>) {
      if constexpr (kIsAffineLeftPerturbation) {
        // separate left perturbation -> right perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Zero();
        SubGLn linear_inv = g.linear().inverse();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinearInv =
            SubGLn::Ad(linear_inv);
        result.template block<kLinearDim, kLinearDim>(0, 0) = AdLinearInv;
        result.template block<N, N>(kLinearDim, kLinearDim) =
            linear_inv.matrix();
        return result;
      } else if constexpr (kIsAffineRightPerturbation) {
        // separate right perturbation -> right perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Identity();
        result.template block<N, N>(kLinearDim, kLinearDim) =
            g.linear().inverse().matrix();
        return result;
      } else {
        // Convert to separate right perturbation first, then to the
        // target right perturbation
        return sep_right_perturb.template convertPerturbation<ToPerturbation>(
                   g) *
               this->template convertPerturbation<AffineRightPerturbation>(g);
      }
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             ToPerturbation, LeftPerturbation>) {
      if constexpr (kIsAffineLeftPerturbation) {
        // separate left perturbation -> left perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Identity();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        result.template block<N, kLinearDim>(kLinearDim, 0) =
            SubGLn::JmultVector(-t);
        return result;
      } else if constexpr (kIsAffineRightPerturbation) {
        // separate right perturbation -> left perturbation
        LieAlgebraEndomorphism result = LieAlgebraEndomorphism::Identity();
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinear =
            SubGLn::Ad(g.linear());
        result.template block<kLinearDim, kLinearDim>(0, 0) = AdLinear;
        result.template block<N, kLinearDim>(kLinearDim, 0) =
            SubGLn::JmultVector(-t) * AdLinear;
        return result;
      } else {
        // Convert to separate left perturbation first, then to the
        // target left perturbation
        return sep_left_perturb.template convertPerturbation<ToPerturbation>(
                   g) *
               this->template convertPerturbation<AffineLeftPerturbation>(g);
      }
    } else {
      return _Base::template convertPerturbation<ToPerturbation>(g);
    }
  }

  template <
      typename SrcPerturbation, typename JacobianMatrixWrtSrcPerturbation,
      typename JacobianMatrixWrtThisPerturbation>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcPerturbation& jacobian_under_src_perturbation,
      const SubGLn_rx_Rn& g,
      JacobianMatrixWrtThisPerturbation* jacobian_under_this_perturbation,
      const SrcPerturbation& src_perturbation = SrcPerturbation()) const {
    if constexpr (std::is_base_of_v<ThisPerturbation, SrcPerturbation>) {
      if constexpr (std::is_base_of_v<
                        JacobianMatrixWrtSrcPerturbation,
                        JacobianMatrixWrtThisPerturbation>) {
        if (jacobian_under_this_perturbation ==
            &jacobian_under_src_perturbation) {
          return;
        }
      }
      *jacobian_under_this_perturbation = jacobian_under_src_perturbation;
      return;
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             SrcPerturbation, RightPerturbation>) {
      if constexpr (kIsAffineLeftPerturbation) {
        // separate left perturbation -> right perturbation
        SubGLn linear_inv = g.linear().inverse();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinearInv =
            SubGLn::Ad(linear_inv);
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation.template leftCols<kLinearDim>() *
            AdLinearInv;
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>() *
            linear_inv.matrix();
        return;
      } else if constexpr (kIsAffineRightPerturbation) {
        // separate right perturbation -> right perturbation
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation.template leftCols<kLinearDim>();
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>() *
            g.linear().inverse().matrix();
        return;
      } else {
        // Convert to separate right perturbation first, then to the
        // target right perturbation
        Eigen::Matrix<
            Scalar, JacobianMatrixWrtSrcPerturbation::RowsAtCompileTime,
            SubGLn_rx_Rn::kDim>
            jacobian_under_sep_right_perturbation;
        if constexpr (
            JacobianMatrixWrtSrcPerturbation::RowsAtCompileTime ==
            Eigen::Dynamic) {
          jacobian_under_sep_right_perturbation.resize(
              jacobian_under_src_perturbation.rows(), SubGLn_rx_Rn::kDim);
        }
        sep_right_perturb.transformJacobianImpl(
            jacobian_under_src_perturbation, g,
            &jacobian_under_sep_right_perturbation, src_perturbation);
        transformJacobianImpl(
            jacobian_under_sep_right_perturbation, g,
            jacobian_under_this_perturbation, sep_right_perturb);
        return;
      }
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             SrcPerturbation, LeftPerturbation>) {
      if constexpr (kIsAffineLeftPerturbation) {
        // separate left perturbation -> left perturbation
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation.template leftCols<kLinearDim>() +
            jacobian_under_src_perturbation.template rightCols<N>() *
                SubGLn::JmultVector(-t);
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>();
        return;
      } else if constexpr (kIsAffineRightPerturbation) {
        // separate right perturbation -> left perturbation
        const Eigen::Matrix<Scalar, N, 1>& t = g.translation();
        Eigen::Matrix<Scalar, kLinearDim, kLinearDim> AdLinear =
            SubGLn::Ad(g.linear());
        Eigen::Matrix<Scalar, SubGLn_rx_Rn::kDim, kLinearDim> tmp;
        tmp << AdLinear, SubGLn::JmultVector(-t) * AdLinear;
        jacobian_under_this_perturbation->template leftCols<kLinearDim>() =
            jacobian_under_src_perturbation * tmp;
        jacobian_under_this_perturbation->template rightCols<N>() =
            jacobian_under_src_perturbation.template rightCols<N>();
        return;
      } else {
        // Convert to separate left perturbation first, then to the
        // target left perturbation
        Eigen::Matrix<
            Scalar, JacobianMatrixWrtSrcPerturbation::RowsAtCompileTime,
            SubGLn_rx_Rn::kDim>
            jacobian_under_sep_left_perturbation;
        if constexpr (
            JacobianMatrixWrtSrcPerturbation::RowsAtCompileTime ==
            Eigen::Dynamic) {
          jacobian_under_sep_left_perturbation.resize(
              jacobian_under_src_perturbation.rows(), SubGLn_rx_Rn::kDim);
        }
        sep_left_perturb.transformJacobianImpl(
            jacobian_under_src_perturbation, g,
            &jacobian_under_sep_left_perturbation, src_perturbation);
        transformJacobianImpl(
            jacobian_under_sep_left_perturbation, g,
            jacobian_under_this_perturbation, sep_left_perturb);
        return;
      }
    } else {
      // Handle the Jacobian transformation by individual parts
      _Base::transformJacobianAgainstOtherProduct(
          g, src_perturbation, jacobian_under_src_perturbation,
          jacobian_under_this_perturbation);
      return;
    }
  }

 public:  // Calculating Jacobians for LieGroup operations.
          // First we need to define some macros to reduce code duplication
#define AffinePerturbationPrepareForBinaryOp                              \
  if (!jacobian_wrt_gl && !jacobian_wrt_gr && !result) {                  \
    return;                                                               \
  }                                                                       \
  /* Alloc memory for jacobian_wrt_gl if needed */                        \
  if constexpr (                                                          \
      JacobianWrtGl::RowsAtCompileTime == Eigen::Dynamic ||               \
      JacobianWrtGl::ColsAtCompileTime == Eigen::Dynamic) {               \
    if (jacobian_wrt_gl &&                                                \
        (jacobian_wrt_gl->rows() == 0 || jacobian_wrt_gl->cols() == 0)) { \
      jacobian_wrt_gl->resize(kDof, kDof);                                \
    }                                                                     \
  } else {                                                                \
    static_assert(JacobianWrtGl::RowsAtCompileTime == kDof);              \
    static_assert(JacobianWrtGl::ColsAtCompileTime == kDof);              \
  }                                                                       \
  /* Alloc memory for jacobian_wrt_gr if needed */                        \
  if constexpr (                                                          \
      JacobianWrtGr::RowsAtCompileTime == Eigen::Dynamic ||               \
      JacobianWrtGr::ColsAtCompileTime == Eigen::Dynamic) {               \
    if (jacobian_wrt_gr &&                                                \
        (jacobian_wrt_gr->rows() == 0 || jacobian_wrt_gr->cols() == 0)) { \
      jacobian_wrt_gr->resize(kDof, kDof);                                \
    }                                                                     \
  } else {                                                                \
    static_assert(JacobianWrtGr::RowsAtCompileTime == kDof);              \
    static_assert(JacobianWrtGr::ColsAtCompileTime == kDof);              \
  }

#define AffinePerturbationHandleLinearPartForBinaryOp(Op)                      \
  /* For the linear part, let the linear_perturbation handle the operation*/   \
  {                                                                            \
    const auto& linear_perturbation = this->template part<0>();                \
    using LinearPartJacobian =                                                 \
        Eigen::Matrix<typename _LieGroup::Scalar, SubGLn::kDim, SubGLn::kDim>; \
    LinearPartJacobian J_gl, J_gr;                                             \
    LinearPartJacobian* j_gl = jacobian_wrt_gl ? &J_gl : nullptr;              \
    LinearPartJacobian* j_gr = jacobian_wrt_gr ? &J_gr : nullptr;              \
    linear_perturbation.Op(                                                    \
        gl.linear(), gr.linear(), result ? &result->linear() : nullptr, j_gl,  \
        j_gr);                                                                 \
    if (jacobian_wrt_gl) {                                                     \
      jacobian_wrt_gl->template topLeftCorner<SubGLn::kDim, SubGLn::kDim>() =  \
          J_gl;                                                                \
      jacobian_wrt_gl->template block<SubGLn::kDim, N>(0, SubGLn::kDim)        \
          .setZero();                                                          \
    }                                                                          \
    if (jacobian_wrt_gr) {                                                     \
      jacobian_wrt_gr->template topLeftCorner<SubGLn::kDim, SubGLn::kDim>() =  \
          J_gr;                                                                \
      jacobian_wrt_gr->template block<SubGLn::kDim, N>(0, SubGLn::kDim)        \
          .setZero();                                                          \
    }                                                                          \
  }

#define AffinePerturbationPrepareForUnaryOp                             \
  if (!jacobian_wrt_g && !result) {                                     \
    return;                                                             \
  }                                                                     \
  /* Alloc memory for jacobian_wrt_g if needed */                       \
  if constexpr (                                                        \
      JacobianWrtG::RowsAtCompileTime == Eigen::Dynamic ||              \
      JacobianWrtG::ColsAtCompileTime == Eigen::Dynamic) {              \
    if (jacobian_wrt_g &&                                               \
        (jacobian_wrt_g->rows() == 0 || jacobian_wrt_g->cols() == 0)) { \
      jacobian_wrt_g->resize(kDof, kDof);                               \
    }                                                                   \
  } else {                                                              \
    static_assert(JacobianWrtG::RowsAtCompileTime == kDof);             \
    static_assert(JacobianWrtG::ColsAtCompileTime == kDof);             \
  }

#define AffinePerturbationHandleLinearPartForUnaryOp(Op)                       \
  /* For the linear part, let the linear_perturbation handle the operation*/   \
  {                                                                            \
    const auto& linear_perturbation = this->template part<0>();                \
    using LinearPartJacobian =                                                 \
        Eigen::Matrix<typename _LieGroup::Scalar, SubGLn::kDim, SubGLn::kDim>; \
    LinearPartJacobian J_g;                                                    \
    LinearPartJacobian* j_g = jacobian_wrt_g ? &J_g : nullptr;                 \
    linear_perturbation.Op(                                                    \
        g.linear(), result ? &result->linear() : nullptr, j_g);                \
    if (jacobian_wrt_g) {                                                      \
      jacobian_wrt_g->template topLeftCorner<SubGLn::kDim, SubGLn::kDim>() =   \
          J_g;                                                                 \
      jacobian_wrt_g->template block<SubGLn::kDim, N>(0, SubGLn::kDim)         \
          .setZero();                                                          \
    }                                                                          \
  }

  template <
      typename _LieGroup,
      typename JacobianWrtGl =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>,
      typename JacobianWrtGr =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>>
  void Multiply(
      const _LieGroup& gl, const _LieGroup& gr, _LieGroup* result,
      JacobianWrtGl* jacobian_wrt_gl = nullptr,
      JacobianWrtGr* jacobian_wrt_gr = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;
    AffinePerturbationPrepareForBinaryOp;
    AffinePerturbationHandleLinearPartForBinaryOp(Multiply);

    // Compute the Jacobian of translation part
    if (jacobian_wrt_gl) {
      jacobian_wrt_gl->template bottomRightCorner<N, N>().setIdentity();
      if constexpr (kIsAffineLeftPerturbation) {
        jacobian_wrt_gl->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            SubGLn::JmultVector(gl.linear() * gr.translation());
      } else {
        static_assert(
            kIsAffineRightPerturbation,
            "AffinePerturbation::Multiply() is only defined for "
            "AffineLeftPerturbation or AffineRightPerturbation");
        jacobian_wrt_gl->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            gl.linear().matrix() * SubGLn::JmultVector(gr.translation());
      }
    }
    if (jacobian_wrt_gr) {
      jacobian_wrt_gr->template bottomRightCorner<N, N>() =
          gl.linear().matrix();
      jacobian_wrt_gr->template block<N, SubGLn::kDim>(SubGLn::kDim, 0)
          .setZero();
    }

    if (result) {
      result->translation() = gl.translation() + gl.linear() * gr.translation();
    }
  }

  template <
      typename _LieGroup,
      typename JacobianWrtG =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>,
      typename JacobianWrtG2 =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>>
  void Inverse(
      const _LieGroup& g, _LieGroup* result,
      JacobianWrtG* jacobian_wrt_g = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;

    AffinePerturbationPrepareForUnaryOp;
    AffinePerturbationHandleLinearPartForUnaryOp(Inverse);

    auto g_linear_inv = g.linear().inverse().matrix();
    if (jacobian_wrt_g) {
      jacobian_wrt_g->template bottomRightCorner<N, N>() = -g_linear_inv;
      if constexpr (kIsAffineLeftPerturbation) {
        jacobian_wrt_g->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            g_linear_inv * SubGLn::JmultVector(g.translation());
      } else {
        static_assert(
            kIsAffineRightPerturbation,
            "AffinePerturbation::Inverse() is only defined for "
            "AffineLeftPerturbation or AffineRightPerturbation");
        jacobian_wrt_g->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            SubGLn::JmultVector(g_linear_inv * g.translation());
      }
    }

    if (result) {
      result->translation() = g_linear_inv * (-g.translation());
    }
  }

  template <
      typename _LieGroup,
      typename JacobianWrtGl =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>,
      typename JacobianWrtGr =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>>
  void RightDelta(
      const _LieGroup& gl, const _LieGroup& gr, _LieGroup* result,
      JacobianWrtGl* jacobian_wrt_gl = nullptr,
      JacobianWrtGr* jacobian_wrt_gr = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;
    AffinePerturbationPrepareForBinaryOp;
    AffinePerturbationHandleLinearPartForBinaryOp(RightDelta);

    // Compute the Jacobian of translation part
    auto gl_linear_inv = gl.linear().inverse().matrix();
    if (jacobian_wrt_gl) {
      jacobian_wrt_gl->template bottomRightCorner<N, N>() = -gl_linear_inv;
      if constexpr (kIsAffineLeftPerturbation) {
        jacobian_wrt_gl->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            gl_linear_inv *
            SubGLn::JmultVector(gl.translation() - gr.translation());
      } else {
        static_assert(
            kIsAffineRightPerturbation,
            "AffinePerturbation::RightDelta() is only defined for "
            "AffineLeftPerturbation or AffineRightPerturbation");
        jacobian_wrt_gl->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            SubGLn::JmultVector(
                gl_linear_inv * (gl.translation() - gr.translation()));
      }
    }
    if (jacobian_wrt_gr) {
      jacobian_wrt_gr->template bottomRightCorner<N, N>() = gl_linear_inv;
      jacobian_wrt_gr->template block<N, SubGLn::kDim>(SubGLn::kDim, 0)
          .setZero();
    }

    if (result) {
      result->translation() =
          gl_linear_inv * (gr.translation() - gl.translation());
    }
  }

  template <
      typename _LieGroup,
      typename JacobianWrtGl =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>,
      typename JacobianWrtGr =
          Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>>
  void LeftDelta(
      const _LieGroup& gl, const _LieGroup& gr, _LieGroup* result,
      JacobianWrtGl* jacobian_wrt_gl = nullptr,
      JacobianWrtGr* jacobian_wrt_gr = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;
    AffinePerturbationPrepareForBinaryOp;
    AffinePerturbationHandleLinearPartForBinaryOp(LeftDelta);

    // Compute the Jacobian of translation part
    auto r_l_linear_inv = (gr.linear() * gl.linear().inverse()).matrix();
    if constexpr (kIsAffineLeftPerturbation) {
      if (jacobian_wrt_gl) {
        jacobian_wrt_gl->template bottomRightCorner<N, N>() = -r_l_linear_inv;
        jacobian_wrt_gl->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            r_l_linear_inv * SubGLn::JmultVector(gl.translation());
      }
      if (jacobian_wrt_gr) {
        jacobian_wrt_gr->template bottomRightCorner<N, N>().setIdentity();
        jacobian_wrt_gr->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
            SubGLn::JmultVector(r_l_linear_inv * (-gl.translation()));
      }
    } else {
      static_assert(
          kIsAffineRightPerturbation,
          "AffinePerturbation::LeftDelta() is only defined for "
          "AffineLeftPerturbation or AffineRightPerturbation");
      if (jacobian_wrt_gl || jacobian_wrt_gr) {
        auto gl_linear_inv = gl.linear().inverse();
        Vector<N, typename _LieGroup::Scalar> tmp =
            gl_linear_inv * gl.translation();
        Eigen::Matrix<typename _LieGroup::Scalar, N, SubGLn::kDim> J =
            gr.linear() * SubGLn::JmultVector(tmp);
        if (jacobian_wrt_gl) {
          jacobian_wrt_gl->template bottomRightCorner<N, N>() = -r_l_linear_inv;
          jacobian_wrt_gl->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) = J;
        }
        if (jacobian_wrt_gr) {
          jacobian_wrt_gr->template bottomRightCorner<N, N>().setIdentity();
          jacobian_wrt_gr->template block<N, SubGLn::kDim>(SubGLn::kDim, 0) =
              -J;
        }
      }
    }

    if (result) {
      result->translation() =
          gr.translation() - r_l_linear_inv * gl.translation();
    }
  }

  template <
      typename _LieGroup, typename VectorXpr, typename ResultVector,
      typename JacobianWrtG =
          Eigen::Matrix<typename _LieGroup::Scalar, Eigen::Dynamic, kDof>>
  void TransformVector(
      const _LieGroup& g, const VectorXpr& v, ResultVector* result,
      JacobianWrtG* jacobian_wrt_g = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    static_assert(_LieGroup::kDim == SubGLn::kDim + N);
    static_assert(
        kIsAffineLeftPerturbation || kIsAffineRightPerturbation,
        "AffinePerturbation::RightDelta() is only defined for "
        "AffineLeftPerturbation or AffineRightPerturbation");
    using Scalar = typename _LieGroup::Scalar;
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    using std::abs;
    // Alloc memory for jacobian_wrt_g if needed
    if constexpr (
        JacobianWrtG::RowsAtCompileTime == Eigen::Dynamic ||
        JacobianWrtG::ColsAtCompileTime == Eigen::Dynamic) {
      if (jacobian_wrt_g &&
          (jacobian_wrt_g->rows() == 0 || jacobian_wrt_g->cols() == 0)) {
        jacobian_wrt_g->resize(N + 1, kDof);
      }
    } else {
      static_assert(JacobianWrtG::RowsAtCompileTime == N + 1);
      static_assert(JacobianWrtG::ColsAtCompileTime == kDof);
    }

    Vector<N, Scalar> transformed_homo_v;
    Vector<N, Scalar>* p_transformed_homo_v = nullptr;
    if (result) {
      result->template tail<1>() = v.template tail<1>();
      p_transformed_homo_v = &transformed_homo_v;
    }
    if (jacobian_wrt_g) {
      jacobian_wrt_g->template bottomRows<1>().setZero();
    }

    Scalar scale = v.template tail<1>().value();
    if (abs(scale) < kEps) {
      // scale = 0. We only need to transform the homogeneous part.
      const auto& linear_perturbation = this->template part<0>();
      if (jacobian_wrt_g) {
        jacobian_wrt_g->template rightCols<N>().setZero();
        auto jacobian_wrt_g_top_left =
            jacobian_wrt_g->template topLeftCorner<N, SubGLn::kDim>();
        linear_perturbation.TransformVector(
            g.linear(), v.template head<N>(), p_transformed_homo_v,
            &jacobian_wrt_g_top_left);
      } else {
        linear_perturbation.TransformVector(
            g.linear(), v.template head<N>(), p_transformed_homo_v);
      }
      if (result) {
        result->template head<N>() = transformed_homo_v;
      }
    } else {
      if (jacobian_wrt_g) {
        auto jacobian_wrt_g_top_N_rows = jacobian_wrt_g->template topRows<N>();
        TransformHomoVector(
            g, v.template head<N>() / scale, p_transformed_homo_v,
            &jacobian_wrt_g_top_N_rows);
      } else {
        TransformHomoVector(
            g, v.template head<N>() / scale, p_transformed_homo_v);
      }
      if (result) {
        result->template head<N>() = transformed_homo_v * scale;
      }
      if (jacobian_wrt_g) {
        jacobian_wrt_g->template topRows<N>() *= scale;
      }
    }
  }

  template <
      typename _LieGroup, typename VectorXpr, typename ResultVector,
      typename JacobianWrtG =
          Eigen::Matrix<typename _LieGroup::Scalar, Eigen::Dynamic, kDof>>
  void TransformHomoVector(
      const _LieGroup& g, const VectorXpr& v, ResultVector* result,
      JacobianWrtG* jacobian_wrt_g = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    static_assert(_LieGroup::kDim == SubGLn::kDim + N);
    static_assert(
        kIsAffineLeftPerturbation || kIsAffineRightPerturbation,
        "AffinePerturbation::RightDelta() is only defined for "
        "AffineLeftPerturbation or AffineRightPerturbation");
    using Scalar = typename _LieGroup::Scalar;
    // Alloc memory for jacobian_wrt_g if needed
    if constexpr (
        JacobianWrtG::RowsAtCompileTime == Eigen::Dynamic ||
        JacobianWrtG::ColsAtCompileTime == Eigen::Dynamic) {
      if (jacobian_wrt_g &&
          (jacobian_wrt_g->rows() == 0 || jacobian_wrt_g->cols() == 0)) {
        jacobian_wrt_g->resize(N, kDof);
      }
    } else {
      static_assert(JacobianWrtG::RowsAtCompileTime == N);
      static_assert(JacobianWrtG::ColsAtCompileTime == kDof);
    }

    if constexpr (kIsAffineLeftPerturbation) {
      Vector<N, Scalar> linear_transformed_v = g.linear() * v;
      if (result) {
        *result = linear_transformed_v + g.translation();
      }
      if (jacobian_wrt_g) {
        jacobian_wrt_g->template leftCols<SubGLn::kDim>() =
            SubGLn::JmultVector(linear_transformed_v);
        jacobian_wrt_g->template rightCols<N>() =
            Eigen::Matrix<Scalar, N, N>::Identity();
      }
    } else {
      if (result) {
        *result = g * v;
      }
      if (jacobian_wrt_g) {
        // Exactly the same as the standard RightPerturbation case.
        const auto& gm = g.linear().matrix();
        jacobian_wrt_g->template leftCols<SubGLn::kDim>() =
            gm * SubGLn::JmultVector(v);
        jacobian_wrt_g->template rightCols<N>() =
            Eigen::Matrix<Scalar, N, N>::Identity();
      }
    }
  }
};

}  // namespace affine_group_internal

}  // namespace sk4slam
