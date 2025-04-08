#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/matrix_group_helper.h"
#include "sk4slam_math/optimizable_manifold.h"

namespace sk4slam {

template <typename DerivedLieGroup>
struct LieGroupBase;

template <
    typename BaseGroup,
    template <typename Scalar, typename...> class DerivedExtensionTemplate,
    typename Scalar, typename... Rest>
class LieGroupExtension;

DEFINE_HAS_MEMBER_VARIABLE(kIsLieGroupExtension);
template <typename LieGroup>
inline constexpr bool IsLieGroupExtension =
    HasMemberVariable_kIsLieGroupExtension<LieGroup>&& std::is_same_v<
        TypeOfMemberVariable_kIsLieGroupExtension<LieGroup>, std::true_type>;

template <typename DerivedLieGroup>
inline constexpr bool IsLieGroup =
    std::is_base_of_v<LieGroupBase<DerivedLieGroup>, DerivedLieGroup> ||
    IsLieGroupExtension<DerivedLieGroup>;

namespace liegroup_internal {

/// @brief Perturbation for Lie Groups
/// @details
/// Perturbation is a template class for perturbations in Lie groups.
/// It inherits from the Retraction class (see @ref RetractionInterface), which
/// defines a local map from the tangent bundle to the manifold. The
/// Perturbation class extends this by adding operations commonly used in Lie
/// groups and calculating their Jacobians.
///
/// The template parameter @c _left_perturbation determines whether the
/// perturbation is left or right:
///
/// - When @c _left_perturbation is true, the class implements left
/// perturbation, applying small changes to the left side of group elements:
///
///         g_perturbed = Exp(delta) * g,
///
///     where `Exp(delta)` is the exponential map of the perturbation vector
///     `delta` in the Lie algebra.
///
/// - When @c _left_perturbation is false, the class implements right
/// perturbation, applying small changes to the right side:
///
///         g_perturbed = g * Exp(delta)
///
template <typename LieGroup, bool _left_perturbation>
struct Perturbation;

template <typename Derived, typename BasePerturbation, typename SubSpaceType>
struct SubSpacePerturbationBase;

template <typename BasePerturbation, typename SubSpaceType>
struct SubSpacePerturbation;

template <
    typename Derived, typename ProductLieGroup, typename... PerturbationParts>
struct ProductPerturbationBase;

template <typename ProductLieGroup, typename... PerturbationParts>
struct ProductPerturbation;

inline constexpr int Dynamic = Eigen::Dynamic;

}  // namespace liegroup_internal

template <typename DerivedLieGroup>
class LieGroupBase {
  // See the trivial Lie Group `R^+` (class Rp) defined in "Rp.h"
  // to see what properties and interfaces a LieGroup (template) should support.

 public:
  /// @brief Runtime dimension of the Lie group. For fixed-sized Lie groups,
  /// this is the same with the static dimension `kDim` and the default
  /// implementation of this method returns `kDim`. For dynamic-sized Lie
  /// groups, this method must be overridden.
  int dim() const {
    if constexpr (std::is_same_v<
                      decltype(&DerivedLieGroup::dim),
                      decltype(&LieGroupBase::dim)>) {
      static_assert(
          DerivedLieGroup::kDim != liegroup_internal::Dynamic,
          "Liegroup with dynamic dimension must override the dim() method!");
      return DerivedLieGroup::kDim;
    } else {
      // Shouldn't be reached if DerivedLieGroup::dim() is overridden.
      // return derived()->dim();
      throw std::runtime_error(
          "Do NOT use LieGroupBase::dim() if the derived LieGroup overrides "
          "dim()!");
    }
  }

  /// @brief  Runtime ambient dimension of the Lie group. For fixed-sized Lie
  /// groups, this is the same with the static dimension `kAmbientDim` and the
  /// default implementation of this method returns `kAmbientDim`. For
  /// dynamic-sized Lie groups, this method must be overridden.
  int ambientDim() const {
    if constexpr (std::is_same_v<
                      decltype(&DerivedLieGroup::ambientDim),
                      decltype(&LieGroupBase::ambientDim)>) {
      static_assert(
          DerivedLieGroup::kAmbientDim != liegroup_internal::Dynamic,
          "Liegroup with dynamic ambient dimension must override the "
          "ambientDim() method!");
      return DerivedLieGroup::kAmbientDim;
    } else {
      // Shouldn't be reached if DerivedLieGroup::ambientDim() is overridden.
      // return derived()->ambientDim();
      throw std::runtime_error(
          "Do NOT use LieGroupBase::ambientDim() if the derived LieGroup "
          "overrides ambientDim()!");
    }
  }

  /// @brief  This template is used to define Lie group extensions. A Lie group
  /// extension is a new class representing the same Lie group as the
  /// original one, but with additional methods.
  ///
  /// @note Note that the extension class shouldn't contain any data members so
  /// that `sizeof(Extended LieGroup) == sizeof(Original LieGroup)` is always
  /// guaranteed.
  template <
      template <typename Scalar, typename...> class DerivedExtensionTemplate,
      typename Scalar, typename... Rest>
  using Extension = LieGroupExtension<
      DerivedLieGroup, DerivedExtensionTemplate, Scalar, Rest...>;

  /// @brief  If the current Lie group is an extension of another Lie group,
  /// the ExtensionBase should be set to the base Lie group. Otherwise, it
  /// should be set to `void`.
  using ExtensionBase = void;

  /// @name Perturbations
  /// @{

  template <typename LieGroup>
  using LeftPerturbationTemplate =
      liegroup_internal::Perturbation<LieGroup, true>;

  template <typename LieGroup>
  using RightPerturbationTemplate =
      liegroup_internal::Perturbation<LieGroup, false>;

#define DEFINE_LIE_PERTURBATIONS(LieGroup)                                     \
  using LeftPerturbation = LeftPerturbationTemplate<LieGroup>;                 \
  using RightPerturbation = RightPerturbationTemplate<LieGroup>;               \
  template <typename SubSpaceType>                                             \
  using SubLeftPerturbation =                                                  \
      liegroup_internal::SubSpacePerturbation<LeftPerturbation, SubSpaceType>; \
  template <typename SubSpaceType>                                             \
  using SubRightPerturbation = liegroup_internal::SubSpacePerturbation<        \
      RightPerturbation, SubSpaceType>;

  DEFINE_LIE_PERTURBATIONS(DerivedLieGroup);

  /// @}
  /// @name Optimizables
  /// @{

#define DEFINE_LIE_OPTIMIZABLES(LieGroup)                                    \
  using LeftOptimizable = OptimizableManifold<LieGroup, LeftPerturbation>;   \
  using RightOptimizable = OptimizableManifold<LieGroup, RightPerturbation>; \
  template <typename SubSpaceType, bool _share_perturbation = true>          \
  using SubLeftOptimizable = OptimizableManifold<                            \
      LieGroup, SubLeftPerturbation<SubSpaceType>, _share_perturbation>;     \
  template <typename SubSpaceType, bool _share_perturbation = true>          \
  using SubRightOptimizable = OptimizableManifold<                           \
      LieGroup, SubRightPerturbation<SubSpaceType>, _share_perturbation>;

  DEFINE_LIE_OPTIMIZABLES(DerivedLieGroup);

  using XOptimizable = XOptimizableManifold<DerivedLieGroup>;

  /// @}

  /// @brief  Convert a perturbation from one  type to another.
  /// @tparam FromPerturbation
  ///           the source perturbation type. Can be
  ///           @ref LeftPerturbation or @ref RightPerturbation or
  ///           other non-standard perturbation types that have provided
  ///           the required convertPerturbation() method.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be either
  ///           @ref LeftPerturbation or @ref RightPerturbation.
  /// @tparam _Tangent
  ///           Any type convertible to a tangent (Lie algebra) vector.
  /// @tparam _LieGroup
  ///           The Lie group type. Must the derived Lie group type.
  /// @param delta
  ///           the perturbation vector to convert.
  /// @return  the converted perturbation vector.
  template <
      typename FromPerturbation, typename ToPerturbation, typename _Tangent,
      typename _LieGroup = DerivedLieGroup>
  Eigen::Matrix<typename _LieGroup::Scalar, ToPerturbation::kDof, 1>
  convertPerturbation(
      const _Tangent& delta,
      const FromPerturbation& from_perturbation = FromPerturbation(),
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    return from_perturbation.template convertPerturbation<ToPerturbation>(
        delta, static_cast<const _LieGroup&>(*derived()), to_perturbation);
  }

  /// @brief  Convert a perturbation from one  type to another.
  /// @tparam FromPerturbation
  ///           the source perturbation type. Can be
  ///           @ref LeftPerturbation or @ref RightPerturbation or
  ///           other non-standard perturbation types that have provided
  ///           the required convertPerturbation() method.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be either
  ///           @ref LeftPerturbation or @ref RightPerturbation.
  /// @tparam _LieGroup
  ///           The Lie group type. Must the derived Lie group type.
  /// @return  A transformation matrix that can convert INFINITESIMAL
  ///          perturbations from the source perturbation type to
  ///          the target perturbation type.
  /// @note
  ///         For conversion between the standard LeftPerturbation and
  ///         RightPerturbation, the resulting matrix can even be applied to
  ///         non-infinitesimal perturbations.
  template <
      typename FromPerturbation, typename ToPerturbation,
      typename _LieGroup = DerivedLieGroup>
  Eigen::Matrix<
      typename _LieGroup::Scalar, ToPerturbation::kDof, FromPerturbation::kDof>
  convertPerturbation(
      const FromPerturbation& from_perturbation = FromPerturbation(),
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    return from_perturbation.template convertPerturbation<ToPerturbation>(
        static_cast<const _LieGroup&>(*derived()), to_perturbation);
  }

  template <
      typename DstPerturbation, typename SrcPerturbation,
      typename JacobianMatrixWrtSrcPerturbation,
      typename _LieGroup = DerivedLieGroup>
  auto transformJacobian(
      const JacobianMatrixWrtSrcPerturbation& jacobian_under_src_perturbation,
      const DstPerturbation& dst_perturbation = DstPerturbation(),
      const SrcPerturbation& src_perturbation = SrcPerturbation()) const {
    return dst_perturbation.transformJacobian(
        jacobian_under_src_perturbation,
        static_cast<const _LieGroup&>(*derived()), src_perturbation);
  }

 private:
  template <typename _Derived = DerivedLieGroup>
  _Derived* derived() {
    return static_cast<_Derived*>(this);
  }

  template <typename _Derived = DerivedLieGroup>
  const _Derived* derived() const {
    return static_cast<const _Derived*>(this);
  }
};

/// @brief A helper class to extend a Lie group with additional
///        functionality.
template <
    typename BaseGroup,
    template <typename Scalar, typename...> class DerivedExtensionTemplate,
    typename _Scalar, typename... Rest>
class LieGroupExtension : public BaseGroup {
 public:
  /// @brief  If the current Lie group is an extension of another Lie group,
  /// the ExtensionBase should be set to the base Lie group. Otherwise, it
  /// should be set to `void`.
  using ExtensionBase = BaseGroup;

  static inline std::true_type kIsLieGroupExtension = std::true_type();
  using BaseGroup::BaseGroup;  // inherit constructors
  using DerivedExtension = DerivedExtensionTemplate<_Scalar, Rest...>;
  using Scalar = _Scalar;
  static_assert(std::is_same_v<typename BaseGroup::Scalar, Scalar>);
  using BaseGroup::operator*;
  using LieAlgebra = typename BaseGroup::LieAlgebra;
  using LieAlgebraEndomorphism = typename BaseGroup::LieAlgebraEndomorphism;

  template <typename _ScalarType>
  DerivedExtensionTemplate<_ScalarType, Rest...> cast() const {
    DerivedExtensionTemplate<_ScalarType, Rest...> derived;
    BaseGroup& base = derived;
    base = BaseGroup::template cast<_ScalarType>();
    return derived;
  }

  static DerivedExtension Identity() {
    DerivedExtension derived;
    BaseGroup& base = derived;
    base = BaseGroup::Identity();
    return derived;
  }

  DerivedExtension operator*(const DerivedExtension& other) const {
    DerivedExtension derived;
    BaseGroup& base = derived;
    base = BaseGroup::operator*(static_cast<const BaseGroup&>(other));
    return derived;
  }

  DerivedExtension inverse() const {
    DerivedExtension derived;
    BaseGroup& base = derived;
    base = BaseGroup::inverse();
    return derived;
  }

  bool isApprox(
      const DerivedExtension& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return BaseGroup::isApprox(static_cast<const BaseGroup&>(other), eps);
  }

  static DerivedExtension Exp(const LieAlgebra& X) {
    DerivedExtension derived;
    BaseGroup& base = derived;
    base = BaseGroup::Exp(X);
    return derived;
  }

  static LieAlgebra Log(const DerivedExtension& g) {
    return BaseGroup::Log(static_cast<const BaseGroup&>(g));
  }

  static LieAlgebraEndomorphism Ad(const DerivedExtension& g) {
    return BaseGroup::Ad(static_cast<const BaseGroup&>(g));
  }

  // Ad(g, X) = Ad(g) * X
  static LieAlgebra Ad(const DerivedExtension& g, const LieAlgebra& X) {
    return BaseGroup::Ad(static_cast<const BaseGroup&>(g), X);
  }

  /// @name Perturbations
  /// @{

  template <typename LieGroup>
  using LeftPerturbationTemplate =
      typename BaseGroup::template LeftPerturbationTemplate<LieGroup>;

  template <typename LieGroup>
  using RightPerturbationTemplate =
      typename BaseGroup::template RightPerturbationTemplate<LieGroup>;

  DEFINE_LIE_PERTURBATIONS(DerivedExtension);

  /// @}
  /// @name Optimizables
  /// @{

  DEFINE_LIE_OPTIMIZABLES(DerivedExtension);

  using XOptimizable = XOptimizableManifold<DerivedExtension>;

  /// @}

  template <
      typename FromPerturbation, typename ToPerturbation, typename _Tangent,
      typename _LieGroup = DerivedExtension>
  Eigen::Matrix<typename _LieGroup::Scalar, ToPerturbation::kDof, 1>
  convertPerturbation(
      const _Tangent& delta,
      const FromPerturbation& from_perturbation = FromPerturbation(),
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    return BaseGroup::template convertPerturbation<
        FromPerturbation, ToPerturbation, _Tangent, _LieGroup>(
        delta, from_perturbation, to_perturbation);
  }

  template <
      typename FromPerturbation, typename ToPerturbation,
      typename _LieGroup = DerivedExtension>
  Eigen::Matrix<
      typename _LieGroup::Scalar, ToPerturbation::kDof, FromPerturbation::kDof>
  convertPerturbation(
      const FromPerturbation& from_perturbation = FromPerturbation(),
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    return BaseGroup::template convertPerturbation<
        FromPerturbation, ToPerturbation, _LieGroup>(
        from_perturbation, to_perturbation);
  }

  template <
      typename DstPerturbation, typename SrcPerturbation,
      typename JacobianMatrixWrtSrcPerturbation,
      typename _LieGroup = DerivedExtension>
  auto transformJacobian(
      const JacobianMatrixWrtSrcPerturbation& jacobian_under_src_perturbation,
      const DstPerturbation& dst_perturbation = DstPerturbation(),
      const SrcPerturbation& src_perturbation = SrcPerturbation()) const {
    return BaseGroup::template transformJacobian<
        DstPerturbation, SrcPerturbation, JacobianMatrixWrtSrcPerturbation,
        _LieGroup>(
        jacobian_under_src_perturbation, dst_perturbation, src_perturbation);
  }
};

////////// internal implementation //////////

namespace liegroup_internal {
template <typename __LieGroup, bool _left_perturbation>
class Perturbation
    : public RetractionBase<
          Perturbation<__LieGroup, _left_perturbation>, __LieGroup> {
  using _RetractionInterface =
      RetractionBase<Perturbation<__LieGroup, _left_perturbation>, __LieGroup>;

 public:
  using _RetractionInterface::kDof;
  using _RetractionInterface::section;
  using _RetractionInterface::transformJacobian;
  using LieGroup = __LieGroup;
  using Scalar = typename LieGroup::Scalar;

  template <typename _Tangent, typename _LieGroup>
  _LieGroup operator()(const _LieGroup& g, const _Tangent& local) const {
    if constexpr (_left_perturbation) {
      return _LieGroup::Exp(local) * g;
    } else {
      return g * _LieGroup::Exp(local);
    }
  }

  template <
      typename _Tangent, typename _LieGroup, typename JacobianWrtG,
      typename JacobianWrtL>
  _LieGroup operator()(
      const _LieGroup& g, const _Tangent& local, JacobianWrtG* jacobian_wrt_g,
      JacobianWrtL* jacobian_wrt_l) const {
    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;
    if constexpr (_left_perturbation) {
      _LieGroup exp_local = _LieGroup::Exp(local);
      _LieGroup ret = exp_local * g;
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = EndomophismMatrix(_LieGroup::Ad(exp_local));
      }
      if (jacobian_wrt_l) {
        *jacobian_wrt_l = EndomophismMatrix(_LieGroup::Jl(local));
      }
      return ret;
    } else {
      _LieGroup exp_local = _LieGroup::Exp(local);
      _LieGroup ret = g * exp_local;
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = EndomophismMatrix(_LieGroup::Ad(exp_local.inverse()));
      }
      if (jacobian_wrt_l) {
        *jacobian_wrt_l = EndomophismMatrix(_LieGroup::Jr(local));
      }
      return ret;
    }
  }

  template <typename _Tangent, typename _LieGroup>
  void sectionImpl(
      const _LieGroup& g, const _LieGroup& g2, _Tangent* delta) const {
    if constexpr (_left_perturbation) {
      *delta = _LieGroup::Log(g2 * g.inverse());
    } else {
      *delta = _LieGroup::Log(g.inverse() * g2);
    }
  }

  template <
      typename _Tangent, typename _LieGroup, typename JacobianWrtG,
      typename JacobianWrtG2>
  void sectionImpl(
      const _LieGroup& g, const _LieGroup& g2, _Tangent* delta,
      JacobianWrtG* jacobian_wrt_g, JacobianWrtG2* jacobian_wrt_g2) const {
    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;
    if constexpr (_left_perturbation) {
      *delta = _LieGroup::Log(g2 * g.inverse());
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = EndomophismMatrix(-_LieGroup::invJr(*delta));
      }
      if (jacobian_wrt_g2) {
        *jacobian_wrt_g2 = EndomophismMatrix(_LieGroup::invJl(*delta));
      }
    } else {
      *delta = _LieGroup::Log(g.inverse() * g2);
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = EndomophismMatrix(-_LieGroup::invJl(*delta));
      }
      if (jacobian_wrt_g2) {
        *jacobian_wrt_g2 = EndomophismMatrix(_LieGroup::invJr(*delta));
      }
    }
  }

  /// @brief  Convert a perturbation from this type to another.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be either
  ///           @ref LeftPerturbation or @ref RightPerturbation.
  /// @tparam _Tangent
  ///           Any type convertible to a tangent (Lie algebra) vector.
  /// @param delta
  ///           the perturbation vector to convert.
  /// @return  the converted perturbation vector.
  template <typename ToPerturbation, typename _Tangent>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> convertPerturbation(
      const _Tangent& delta, const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    using ThisPerturbation = Perturbation;
    using LeftPerturbation = typename LieGroup::LeftPerturbation;
    using RightPerturbation = typename LieGroup::RightPerturbation;
    static_assert(
        std::is_base_of_v<ToPerturbation, LeftPerturbation> ||
            std::is_base_of_v<ToPerturbation, RightPerturbation>,
        "ToPerturbation must be either LeftPerturbation or "
        "RightPerturbation");
    if constexpr (std::is_same_v<ThisPerturbation, ToPerturbation>) {
      return delta;
    } else if constexpr (std::is_base_of_v<ToPerturbation, RightPerturbation>) {
      // left perturbation -> right perturbation
      return LieGroup::Ad(g.inverse(), delta);
    } else {
      static_assert(std::is_base_of_v<ToPerturbation, LeftPerturbation>);
      // right perturbation -> left perturbation
      return LieGroup::Ad(g, delta);
    }
  }

  /// @brief  Convert a perturbation from this type to another.
  /// @tparam ToPerturbation
  ///           the target perturbation type. Must be either
  ///           @ref LeftPerturbation or @ref RightPerturbation.
  /// @return  A transformation matrix that can convert INFINITESIMAL
  ///          perturbations from the source perturbation type to
  ///          the target perturbation type.
  /// @note
  ///         For conversion between the standard LeftPerturbation and
  ///         RightPerturbation, the resulting matrix can even be applied to
  ///         non-infinitesimal perturbations.
  template <typename ToPerturbation>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> convertPerturbation(
      const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    using ThisPerturbation = Perturbation;
    using LeftPerturbation = typename LieGroup::LeftPerturbation;
    using RightPerturbation = typename LieGroup::RightPerturbation;
    using LieAlgebraEndomorphism = typename LieGroup::LieAlgebraEndomorphism;
    static_assert(
        std::is_base_of_v<ToPerturbation, LeftPerturbation> ||
            std::is_base_of_v<ToPerturbation, RightPerturbation>,
        "ToPerturbation must be either LeftPerturbation or "
        "RightPerturbation");
    if constexpr (std::is_same_v<ThisPerturbation, ToPerturbation>) {
      return LieAlgebraEndomorphism::Identity();
    } else if constexpr (std::is_base_of_v<ToPerturbation, RightPerturbation>) {
      // left perturbation -> right perturbation
      return LieGroup::Ad(g.inverse());
    } else {
      static_assert(std::is_base_of_v<ToPerturbation, LeftPerturbation>);
      // right perturbation -> left perturbation
      return LieGroup::Ad(g);
    }
  }

  using RetractionInterface::DeclareTransformJacobianTypes;
  using TransformJacobianTypes = DeclareTransformJacobianTypes<
      Perturbation<LieGroup, !_left_perturbation>>;

  template <
      typename SrcPerturbation, typename JacobianMatrixWrtSrcPerturbation,
      typename JacobianMatrixWrtThisPerturbation>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcPerturbation& jacobian_under_src_perturbation,
      const LieGroup& g,
      JacobianMatrixWrtThisPerturbation* jacobian_under_this_perturbation,
      const SrcPerturbation& src_perturbation = SrcPerturbation()) const {
    using ThisPerturbation = Perturbation;
    using LeftPerturbation = typename LieGroup::LeftPerturbation;
    using RightPerturbation = typename LieGroup::RightPerturbation;
    static_assert(
        std::is_base_of_v<SrcPerturbation, LeftPerturbation> ||
            std::is_base_of_v<SrcPerturbation, RightPerturbation>,
        "SrcPerturbation must be either LeftPerturbation or "
        "RightPerturbation");
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
    } else if constexpr (std::is_base_of_v<  // NOLINT
                             SrcPerturbation, RightPerturbation>) {
      // left perturbation -> right perturbation
      *jacobian_under_this_perturbation =
          jacobian_under_src_perturbation * LieGroup::Ad(g.inverse());
      return;
    } else {
      static_assert(std::is_base_of_v<SrcPerturbation, LeftPerturbation>);
      // right perturbation -> left perturbation
      *jacobian_under_this_perturbation =
          jacobian_under_src_perturbation * LieGroup::Ad(g);
      return;
    }
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
    if constexpr (_left_perturbation) {
      if (jacobian_wrt_gl) {
        *jacobian_wrt_gl = EndomophismMatrix::Identity();
      }
      if (jacobian_wrt_gr) {
        *jacobian_wrt_gr = EndomophismMatrix(_LieGroup::Ad(gl));
      }
    } else {
      if (jacobian_wrt_gl) {
        *jacobian_wrt_gl = EndomophismMatrix(_LieGroup::Ad(gr.inverse()));
      }
      if (jacobian_wrt_gr) {
        *jacobian_wrt_gr = EndomophismMatrix::Identity();
      }
    }

    if (result) {
      *result = gl * gr;
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
    if constexpr (_left_perturbation) {
      _LieGroup g_inv = g.inverse();
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = EndomophismMatrix(-_LieGroup::Ad(g_inv));
      }
      if (result) {
        *result = g_inv;
      }
    } else {
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = EndomophismMatrix(-_LieGroup::Ad(g));
      }
      if (result) {
        *result = g.inverse();
      }
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
    if constexpr (_left_perturbation) {
      _LieGroup gl_inv = gl.inverse();
      if (result) {
        *result = gl_inv * gr;
      }
      if (jacobian_wrt_gl) {
        *jacobian_wrt_gl = EndomophismMatrix(-_LieGroup::Ad(gl_inv));
      }
      if (jacobian_wrt_gr) {
        *jacobian_wrt_gr = EndomophismMatrix(_LieGroup::Ad(gl_inv));
      }
    } else {
      _LieGroup right_delta;
      if (result || jacobian_wrt_gl) {
        right_delta = gl.inverse() * gr;
      }
      if (result) {
        *result = right_delta;
      }
      if (jacobian_wrt_gl) {
        *jacobian_wrt_gl =
            EndomophismMatrix(-_LieGroup::Ad(right_delta.inverse()));
      }
      if (jacobian_wrt_gr) {
        *jacobian_wrt_gr = EndomophismMatrix::Identity();
      }
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
    if constexpr (_left_perturbation) {
      _LieGroup left_delta;
      if (result || jacobian_wrt_gl) {
        left_delta = gr * gl.inverse();
      }
      if (result) {
        *result = left_delta;
      }
      if (jacobian_wrt_gl) {
        *jacobian_wrt_gl = EndomophismMatrix(-_LieGroup::Ad(left_delta));
      }
      if (jacobian_wrt_gr) {
        *jacobian_wrt_gr = EndomophismMatrix::Identity();
      }
    } else {
      if (result) {
        *result = gr * gl.inverse();
      }
      if (jacobian_wrt_gl) {
        *jacobian_wrt_gl = EndomophismMatrix(-_LieGroup::Ad(gl));
      }
      if (jacobian_wrt_gr) {
        *jacobian_wrt_gr = EndomophismMatrix(_LieGroup::Ad(gl));
      }
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
    static_assert(
        MatrixGroupHelper<_LieGroup>::kIsMatrixGroup,
        "Non-matrix group can not be multiplied with a vector!");

    using Endomophism = typename _LieGroup::LieAlgebraEndomorphism;
    using EndomophismMatrix = Eigen::Matrix<
        typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>;
    if constexpr (_left_perturbation) {
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = g.matrix() * _LieGroup::JmultVector(v) *
                          EndomophismMatrix(_LieGroup::Ad(g.inverse()));
      }
    } else {
      if (jacobian_wrt_g) {
        *jacobian_wrt_g = g.matrix() * _LieGroup::JmultVector(v);
      }
    }

    if (result) {
      *result = g * v;
    }
  }
};

template <typename Derived, typename BasePerturbation, typename SubSpaceType>
class SubSpacePerturbationBase
    : public SubSpaceRetractionBase<Derived, BasePerturbation, SubSpaceType> {
  using _SubSpaceRetraction =
      SubSpaceRetractionBase<Derived, BasePerturbation, SubSpaceType>;

 public:
  using _SubSpaceRetraction::_SubSpaceRetraction;
  using _SubSpaceRetraction::kDof;
  using _SubSpaceRetraction::section;
  using LieGroup = typename _SubSpaceRetraction::Manifold;
  using Scalar = typename LieGroup::Scalar;

  void setBasePerturbation(const BasePerturbation& perturbation) {
    _SubSpaceRetraction::setBaseRetraction(perturbation);
  }

  template <typename ToPerturbation, typename _Tangent>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> convertPerturbation(
      const _Tangent& tangent, const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    Eigen::Matrix<Scalar, BasePerturbation::kDof, 1> base_tangent =
        _SubSpaceRetraction::subspace_map_(tangent);
    if constexpr (std::is_same_v<ToPerturbation, BasePerturbation>) {
      if (to_perturbation == _SubSpaceRetraction::base_retraction_) {
        return base_tangent;
      }
    }
    return _SubSpaceRetraction::base_retraction_
        .template convertPerturbation<ToPerturbation>(
            base_tangent, g, to_perturbation);
    // return g.template convertPerturbation<BasePerturbation,
    // ToPerturbation>(
    //     base_tangent);
  }

  template <typename ToPerturbation>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> convertPerturbation(
      const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    if constexpr (std::is_same_v<ToPerturbation, BasePerturbation>) {
      if (to_perturbation == _SubSpaceRetraction::base_retraction_) {
        return _SubSpaceRetraction::subspace_map_
            .template getTransformMatrix<Scalar>();
      }
    }

    return _SubSpaceRetraction::subspace_map_.transformJacobianWrtBase(
        _SubSpaceRetraction::base_retraction_
            .template convertPerturbation<ToPerturbation>(g, to_perturbation));
    // return _SubSpaceRetraction::subspace_map_.transformJacobianWrtBase(
    //     g.template convertPerturbation<BasePerturbation,
    //     ToPerturbation>());
  }
};

template <typename BasePerturbation, typename SubSpaceType>
class SubSpacePerturbation
    : public SubSpacePerturbationBase<
          SubSpacePerturbation<BasePerturbation, SubSpaceType>,
          BasePerturbation, SubSpaceType> {
  using _Base = SubSpacePerturbationBase<
      SubSpacePerturbation<BasePerturbation, SubSpaceType>, BasePerturbation,
      SubSpaceType>;

 public:
  using _Base::_Base;
};

template <
    typename Derived, typename ProductLieGroup, typename... PerturbationParts>
class ProductPerturbationBase
    : public ProductRetractionBase<
          Derived, ProductLieGroup, PerturbationParts...> {
  using _ProductRetraction =
      ProductRetractionBase<Derived, ProductLieGroup, PerturbationParts...>;

 public:
  using _ProductRetraction::_ProductRetraction;
  using _ProductRetraction::kDof;
  using _ProductRetraction::section;
  using LieGroup = ProductLieGroup;
  using Scalar = typename LieGroup::Scalar;

  template <typename ToPerturbation, typename _Tangent>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> convertPerturbation(
      const _Tangent& tangent, const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    return convertPerturbation<ToPerturbation>(
        tangent, g, to_perturbation,
        std::index_sequence_for<PerturbationParts...>{});
  }

  template <typename ToPerturbation>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> convertPerturbation(
      const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    return convertPerturbation<ToPerturbation>(
        g, to_perturbation, std::index_sequence_for<PerturbationParts...>{});
  }

  template <
      typename _LieGroup,
      typename JacobianWrtGl = Eigen::Matrix<
          typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>,
      typename JacobianWrtGr = Eigen::Matrix<
          typename _LieGroup::Scalar, _LieGroup::kDim, _LieGroup::kDim>>
  void Multiply(
      const _LieGroup& gl, const _LieGroup& gr, _LieGroup* result,
      JacobianWrtGl* jacobian_wrt_gl = nullptr,
      JacobianWrtGr* jacobian_wrt_gr = nullptr) const {
    Multiply(
        gl, gr, result, jacobian_wrt_gl, jacobian_wrt_gr,
        std::index_sequence_for<PerturbationParts...>{});
  }

  template <
      typename _LieGroup, typename JacobianWrtG = Eigen::Matrix<
                              typename _LieGroup::Scalar, kDof, kDof>>
  void Inverse(
      const _LieGroup& g, _LieGroup* result,
      JacobianWrtG* jacobian_wrt_g = nullptr) const {
    Inverse(
        g, result, jacobian_wrt_g,
        std::index_sequence_for<PerturbationParts...>{});
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
    RightDelta(
        gl, gr, result, jacobian_wrt_gl, jacobian_wrt_gr,
        std::index_sequence_for<PerturbationParts...>{});
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
    LeftDelta(
        gl, gr, result, jacobian_wrt_gl, jacobian_wrt_gr,
        std::index_sequence_for<PerturbationParts...>{});
  }

  template <
      typename _LieGroup, typename VectorXpr, typename ResultVector,
      typename JacobianWrtG =
          Eigen::Matrix<typename _LieGroup::Scalar, Eigen::Dynamic, kDof>>
  void TransformVector(
      const _LieGroup& g, const VectorXpr& v, ResultVector* result,
      JacobianWrtG* jacobian_wrt_g = nullptr) const {
    static_assert(kDof == _LieGroup::kDim);
    static_assert(
        MatrixGroupHelper<_LieGroup>::kIsMatrixGroup,
        "Non-matrix group can not be multiplied with a vector!");
    TransformVector(
        g, v, result, jacobian_wrt_g,
        std::index_sequence_for<PerturbationParts...>{});
  }

 protected:
  template <typename ToPerturbation, int _part_idx>
  using PerturbationPart = RawType<
      decltype(std::declval<ToPerturbation>().template part<_part_idx>())>;

  template <typename Vector, typename FirstElement, typename... RestElements>
  static void assignVectorParts(
      Vector& vec, const FirstElement& first, const RestElements&... rest) {
    Eigen::CommaInitializer<Vector> comma_initializer = (vec << first);
    (comma_initializer, ..., rest);
  }

  template <typename Matrix, typename FirstElement, typename... RestElements>
  static void assignMatrixDiagBlocks(
      Matrix& mat, int start_row, int start_col, const FirstElement& first,
      const RestElements&... rest) {
    mat.block(start_row, start_col, first.rows(), first.cols()) = first;
    if constexpr (sizeof...(rest) > 0) {
      assignMatrixDiagBlocks(
          mat, start_row + first.rows(), start_col + first.cols(), rest...);
    }
  }

  template <typename ToPerturbation, typename _Tangent, std::size_t... Is>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> convertPerturbation(
      const _Tangent& t, const LieGroup& g,
      const ToPerturbation& to_perturbation, std::index_sequence<Is...>) const {
    using PartitionedTangent =
        typename _ProductRetraction::template PartitionedTangentT<Scalar>;
    const Eigen::Matrix<Scalar, kDof, 1>& vt = t;
    const PartitionedTangent& pt = static_cast<const PartitionedTangent&>(vt);

    Eigen::Matrix<Scalar, ToPerturbation::kDof, 1> result;
    assignVectorParts(
        result,
        (g.template part<Is>()
             .template convertPerturbation<
                 PerturbationParts, PerturbationPart<ToPerturbation, Is>>(
                 pt.template part<Is>()))...);
    return result;
  }

  template <typename ToPerturbation, std::size_t... Is>
  Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> convertPerturbation(
      const LieGroup& g, const ToPerturbation& to_perturbation,
      std::index_sequence<Is...>) const {
    Eigen::Matrix<Scalar, ToPerturbation::kDof, kDof> result;
    result.setZero();
    assignMatrixDiagBlocks(
        result, 0, 0,
        (g.template part<Is>()
             .template convertPerturbation<
                 PerturbationParts,
                 PerturbationPart<ToPerturbation, Is>>())...);
    return result;
  }

#define ProductPerturbationUnaryOpImpl(UnaryOp)                             \
  template <std::size_t I, typename _LieGroup, typename JacobianWrtG>       \
  void UnaryOp##Part(                                                       \
      const _LieGroup& g, _LieGroup* result, JacobianWrtG* jacobian_wrt_g)  \
      const {                                                               \
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;   \
    const std::vector<int>& dimensions = getDimensions();                   \
    const std::vector<int>& offsets = getDimensionOffsets();                \
    Matrix J_part_g;                                                        \
    Matrix* j_part_g = nullptr;                                             \
    if (jacobian_wrt_g) {                                                   \
      j_part_g = &J_part_g;                                                 \
      j_part_g->resize(dimensions[I], dimensions[I]);                       \
    }                                                                       \
    this->template part<I>().UnaryOp(                                       \
        g.template part<I>(), &result->template part<I>(), j_part_g);       \
    if (jacobian_wrt_g) {                                                   \
      jacobian_wrt_g->block(                                                \
          offsets[I], offsets[I], dimensions[I], dimensions[I]) = J_part_g; \
    }                                                                       \
  }                                                                         \
  template <std::size_t... Is, typename _LieGroup, typename JacobianWrtG>   \
  void UnaryOp(                                                             \
      const _LieGroup& g, _LieGroup* result, JacobianWrtG* jacobian_wrt_g,  \
      std::index_sequence<Is...>) const {                                   \
    using JacobianMatrix =                                                  \
        Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>;              \
    JacobianMatrix J_g;                                                     \
    JacobianMatrix* j_g = nullptr;                                          \
    if (jacobian_wrt_g) {                                                   \
      j_g = &J_g;                                                           \
      j_g->setZero();                                                       \
    }                                                                       \
    (UnaryOp##Part<Is>(g, result, j_g), ...);                               \
    if (jacobian_wrt_g) {                                                   \
      *jacobian_wrt_g = J_g;                                                \
    }                                                                       \
  }

#define ProductPerturbationBinaryOpImpl(BinaryOp)                             \
  template <                                                                  \
      std::size_t I, typename _LieGroup, typename JacobianWrtGl,              \
      typename JacobianWrtGr>                                                 \
  void BinaryOp##Part(                                                        \
      const _LieGroup& gl, const _LieGroup& gr, _LieGroup* result,            \
      JacobianWrtGl* jacobian_wrt_gl, JacobianWrtGr* jacobian_wrt_gr) const { \
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;     \
    const std::vector<int>& dimensions = getDimensions();                     \
    const std::vector<int>& offsets = getDimensionOffsets();                  \
    Matrix J_part_gl, J_part_gr;                                              \
    Matrix *j_part_gl = nullptr, *j_part_gr = nullptr;                        \
    if (jacobian_wrt_gl) {                                                    \
      j_part_gl = &J_part_gl;                                                 \
      j_part_gl->resize(dimensions[I], dimensions[I]);                        \
    }                                                                         \
    if (jacobian_wrt_gr) {                                                    \
      j_part_gr = &J_part_gr;                                                 \
      j_part_gr->resize(dimensions[I], dimensions[I]);                        \
    }                                                                         \
    this->template part<I>().BinaryOp(                                        \
        gl.template part<I>(), gr.template part<I>(),                         \
        &result->template part<I>(), j_part_gl, j_part_gr);                   \
    if (jacobian_wrt_gl) {                                                    \
      jacobian_wrt_gl->block(                                                 \
          offsets[I], offsets[I], dimensions[I], dimensions[I]) = J_part_gl;  \
    }                                                                         \
    if (jacobian_wrt_gr) {                                                    \
      jacobian_wrt_gr->block(                                                 \
          offsets[I], offsets[I], dimensions[I], dimensions[I]) = J_part_gr;  \
    }                                                                         \
  }                                                                           \
  template <                                                                  \
      std::size_t... Is, typename _LieGroup, typename JacobianWrtGl,          \
      typename JacobianWrtGr>                                                 \
  void BinaryOp(                                                              \
      const _LieGroup& gl, const _LieGroup& gr, _LieGroup* result,            \
      JacobianWrtGl* jacobian_wrt_gl, JacobianWrtGr* jacobian_wrt_gr,         \
      std::index_sequence<Is...>) const {                                     \
    using JacobianMatrix =                                                    \
        Eigen::Matrix<typename _LieGroup::Scalar, kDof, kDof>;                \
    JacobianMatrix J_gl, J_gr;                                                \
    JacobianMatrix *j_gl = nullptr, *j_gr = nullptr;                          \
    if (jacobian_wrt_gl) {                                                    \
      j_gl = &J_gl;                                                           \
      j_gl->setZero();                                                        \
    }                                                                         \
    if (jacobian_wrt_gr) {                                                    \
      j_gr = &J_gr;                                                           \
      j_gr->setZero();                                                        \
    }                                                                         \
    (BinaryOp##Part<Is>(gl, gr, result, j_gl, j_gr), ...);                    \
    if (jacobian_wrt_gl) {                                                    \
      *jacobian_wrt_gl = J_gl;                                                \
    }                                                                         \
    if (jacobian_wrt_gr) {                                                    \
      *jacobian_wrt_gr = J_gr;                                                \
    }                                                                         \
  }

  ProductPerturbationUnaryOpImpl(Inverse);

  ProductPerturbationBinaryOpImpl(Multiply);

  ProductPerturbationBinaryOpImpl(RightDelta);

  ProductPerturbationBinaryOpImpl(LeftDelta);

  template <
      std::size_t I, typename _LieGroup, typename VectorXpr,
      typename ResultVector, typename JacobianWrtG>
  void TransformVectorPart(
      int& v_part_offset, int v_part_size, const _LieGroup& g,
      const VectorXpr& v, ResultVector* result,
      JacobianWrtG* jacobian_wrt_g) const {
    static_assert(
        MatrixGroupHelper<decltype(g.template part<I>())>::kIsMatrixGroup,
        "Non-matrix group can not be multiplied with a vector!");
    static_assert(MatrixGroupHelper<decltype(g.template part<I>())>::N > 0);

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const std::vector<int>& col_dimensions = getDimensions();
    const std::vector<int>& col_offsets = getDimensionOffsets();

    Matrix J_part_g;
    Matrix* j_part_g = nullptr;
    if (jacobian_wrt_g) {
      J_part_g.resize(v_part_size, col_dimensions[I]);
      j_part_g = &J_part_g;
    }
    Vector<Eigen::Dynamic, Scalar> result_part;
    Vector<Eigen::Dynamic, Scalar>* p_result_part = nullptr;
    if (result) {
      result_part.resize(v_part_size);
      p_result_part = &result_part;
    }

    this->template part<I>().TransformVector(
        g.template part<I>(), v.segment(v_part_offset, v_part_size),
        p_result_part, j_part_g);
    if (result) {
      result->segment(v_part_offset, v_part_size) = result_part;
    }
    if (jacobian_wrt_g) {
      jacobian_wrt_g->block(
          v_part_offset, col_offsets[I], v_part_size, col_dimensions[I]) =
          J_part_g;
    }
    LOGA(
        "TransformVectorPart: v_part_offset = %d, v_part_size = %d",
        v_part_offset, v_part_size);
    v_part_offset += v_part_size;
  }

  template <
      std::size_t... Is, typename _LieGroup, typename VectorXpr,
      typename ResultVector, typename JacobianWrtG>
  void TransformVector(
      const _LieGroup& g, const VectorXpr& v, ResultVector* result,
      JacobianWrtG* jacobian_wrt_g, std::index_sequence<Is...>) const {
    static_assert(kDof == _LieGroup::kDim);
    static_assert(
        MatrixGroupHelper<_LieGroup>::kIsMatrixGroup,
        "Non-matrix group can not be multiplied with a vector!");
    static constexpr int N = MatrixGroupHelper<_LieGroup>::N;
    using JacobianMatrix = Eigen::Matrix<typename _LieGroup::Scalar, N, kDof>;
    using VectorN = Vector<N, typename _LieGroup::Scalar>;
    JacobianMatrix J_g;
    JacobianMatrix* j_g = nullptr;
    if (jacobian_wrt_g) {
      j_g = &J_g;
      j_g->setZero();
    }
    VectorN result_v;
    VectorN* p_result_v = nullptr;
    if (result) {
      p_result_v = &result_v;
    }
    int v_part_offset = 0;
    (TransformVectorPart<Is>(
         v_part_offset, MatrixGroupHelper<decltype(g.template part<Is>())>::N,
         g, v, p_result_v, j_g),
     ...);
    LOGA("TransformVector: final v_part_offset = %d, N = %d", v_part_offset, N);
    ASSERT(v_part_offset == N);
    if (result) {
      *result = result_v;
    }
    if (jacobian_wrt_g) {
      *jacobian_wrt_g = J_g;
    }
  }

  static const std::vector<int>& getDimensions() {
    static const std::vector<int> dimensions = {PerturbationParts::kDof...};
    return dimensions;
  }

  static const std::vector<int>& getDimensionOffsets() {
    static const std::vector<int> offsets = generateDimensionOffsets();
    return offsets;
  }

  static std::vector<int> generateDimensionOffsets() {
    std::vector<int> offsets;
    const std::vector<int>& dimensions = getDimensions();
    offsets.push_back(0);
    for (int i = 0; i < getDimensions().size() - 1; i++) {
      offsets.push_back(offsets.back() + dimensions[i]);
    }
    ASSERT(offsets.size() == dimensions.size());
    ASSERT(offsets.back() + dimensions.back() == kDof);
    return offsets;
  }
};

template <typename ProductLieGroup, typename... PerturbationParts>
class ProductPerturbation
    : public ProductPerturbationBase<
          ProductPerturbation<ProductLieGroup, PerturbationParts...>,
          ProductLieGroup, PerturbationParts...> {
  using _Base = ProductPerturbationBase<
      ProductPerturbation<ProductLieGroup, PerturbationParts...>,
      ProductLieGroup, PerturbationParts...>;

 public:
  using _Base::_Base;
};

template <typename Perturbation>
inline constexpr bool IsProductPerturbation =
    Perturbation::kIsProduct;  // `kIsProduct` is defined in the base class
                               // `ProductRetractionBase`.
}  // namespace liegroup_internal

}  // namespace sk4slam
