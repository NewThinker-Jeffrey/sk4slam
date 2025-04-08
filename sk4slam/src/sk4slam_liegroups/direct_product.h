#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/liegroup_data_iterator.h"
#include "sk4slam_math/vector_space.h"

namespace sk4slam {

template <
    template <typename...> class DerivedProductTemplate,
    template <typename...> class AmbientProductTemplate, typename... LieGroups>
class ProductLieGroupBase;

// A simple version of ProductLieGroupBase that uses the default
// implementations for Ambient direct products.
template <
    template <typename...> class DerivedProductTemplate, typename... LieGroups>
using ProductLieGroupSimpleBase = ProductLieGroupBase<
    DerivedProductTemplate,
    ProductVectorSpace,  // for Ambient
    LieGroups...>;

// The default implementation of ProductLieGroup.
template <typename... LieGroups>
class ProductLieGroup;

////////// Direct Product of LieGroups //////////

template <
    template <typename...> class DerivedProductTemplate,
    template <typename...> class AmbientProductTemplate, typename... LieGroups>
class ProductLieGroupBase
    : public LieGroupBase<DerivedProductTemplate<LieGroups...>> {
 public:
  static constexpr int kParts = sizeof...(LieGroups);
  static constexpr int kPartDims[] = {LieGroups::kDim...};
  static constexpr int kPartAmbientDims[] = {LieGroups::kAmbientDim...};
  static constexpr int kDim = (0 + ... + LieGroups::kDim);
  static constexpr int kAmbientDim = (0 + ... + LieGroups::kAmbientDim);

 protected:
  ///////// Template metafunctions  //////////

  template <typename Fisrt, typename... Rest>
  struct _CheckScalarTypes {
    using Scalar = typename Fisrt::Scalar;
    static constexpr bool value =
        (std::is_same_v<Scalar, typename Rest::Scalar> && ...);
  };

  static_assert(
      _CheckScalarTypes<LieGroups...>::value,
      "All LieGroups must have the same scalar type");

  template <
      int _idx, int _dim_offset, int _ambient_dim_offset,
      typename... _LieGroups>
  struct _LieGroupPartWrapper;

  template <
      int _idx, int _dim_offset, int _ambient_dim_offset, typename LieGroup0,
      typename... RestLieGroups>
  struct _LieGroupPartWrapper<
      _idx, _dim_offset, _ambient_dim_offset, LieGroup0, RestLieGroups...> {
    using LieGroupPart = typename _LieGroupPartWrapper<
        _idx - 1, _dim_offset + LieGroup0::kDim,
        _ambient_dim_offset + LieGroup0::kAmbientDim,
        RestLieGroups...>::LieGroupPart;
    using LieGroup = typename LieGroupPart::LieGroup;
    using Scalar = typename LieGroup::Scalar;
  };

  template <
      int _dim_offset, int _ambient_dim_offset, typename LieGroup0,
      typename... RestLieGroups>
  struct _LieGroupPartWrapper<
      0, _dim_offset, _ambient_dim_offset, LieGroup0, RestLieGroups...> {
    struct LieGroupPart {
      using LieGroup = LieGroup0;
      static constexpr int kDimOffset = _dim_offset;
      static constexpr int kAmbientDimOffset = _ambient_dim_offset;
      static constexpr int kDim = LieGroup0::kDim;
      static constexpr int kAmbientDim = LieGroup0::kAmbientDim;
    };
  };

  template <int _idx>
  using _LieGroupPart =
      typename _LieGroupPartWrapper<_idx, 0, 0, LieGroups...>::LieGroupPart;

  using Derived = DerivedProductTemplate<LieGroups...>;

  template <typename _ScalarType>
  using _CastTemplate = DerivedProductTemplate<
      decltype(std::declval<LieGroups>().template cast<_ScalarType>())...>;

 public:
  using Scalar = typename _CheckScalarTypes<LieGroups...>::Scalar;

  using DataIterator = LieGroupDataIterator<Scalar>;

  using ConstDataIterator = LieGroupDataIterator<const Scalar>;

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using PartitionedLieAlgebra = PartitionedVector<Scalar, LieGroups::kDim...>;

  static_assert(
      std::is_same_v<Scalar, typename LieAlgebra::Scalar>,
      "Scalar type of LieGroup and LieAlgebra must be the same");

  using Ambient = AmbientProductTemplate<typename LieGroups::Ambient...>;

  using LieAlgebraEndomorphism =
      ProductLinearEndomorphism<typename LieGroups::LieAlgebraEndomorphism...>;

  static_assert(
      std::is_same_v<Scalar, typename Ambient::Scalar>,
      "Scalar type of LieGroup and Ambient must be the same");
  static_assert(
      std::is_same_v<Scalar, typename LieAlgebraEndomorphism::Scalar>,
      "Scalar type of LieGroup and LieAlgebraEndomorphism must be the same");

  ProductLieGroupBase() {}

  template <typename... Args>
  explicit ProductLieGroupBase(Args&&... parts)
      : parts_(std::forward<Args>(parts)...) {}

  // Specialization for copy construction
  ProductLieGroupBase(const Derived& other)  // NOLINT
      : parts_(other.parts_) {}

  template <std::size_t _part_idx>
  auto& part() {
    return std::get<_part_idx>(parts_);
  }

  template <std::size_t _part_idx>
  const auto& part() const {
    return std::get<_part_idx>(parts_);
  }

  DataIterator data() {
    return _data_impl(std::index_sequence_for<LieGroups...>{});
  }

  ConstDataIterator data() const {
    return _data_impl(std::index_sequence_for<LieGroups...>{});
  }

  template <typename _ScalarType>
  _CastTemplate<_ScalarType> cast() const {
    return this->template _cast_impl<_CastTemplate<_ScalarType>>(
        std::index_sequence_for<LieGroups...>{});
  }

 public:
  //// Lie Group operations ////

  template <typename _Derived = Derived>
  static _Derived Identity() {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _Derived(LieGroups::Identity()...);
  }

  template <typename _Derived = Derived>
  _Derived operator*(const _Derived& other) const {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _multiply<_Derived>(other, std::index_sequence_for<LieGroups...>{});
  }

  template <typename _Derived = Derived>
  _Derived inverse() const {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _inverse_impl<_Derived>(std::index_sequence_for<LieGroups...>{});
  }

  template <typename _Derived = Derived>
  bool isApprox(
      const _Derived& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _isApprox_impl<_Derived>(
        other, eps, std::index_sequence_for<LieGroups...>{});
  }

  template <typename _Derived = Derived>
  static _Derived Exp(const LieAlgebra& X) {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _exp_impl<_Derived>(X, std::index_sequence_for<LieGroups...>{});
  }

  template <typename _Derived = Derived>
  static LieAlgebra Log(const _Derived& g) {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _log_impl<_Derived>(g, std::index_sequence_for<LieGroups...>{});
  }

  template <typename _Derived = Derived>
  static LieAlgebraEndomorphism Ad(const _Derived& g) {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _Ad_impl<_Derived>(g, std::index_sequence_for<LieGroups...>{});
  }

  // Ad(g, X) = Ad(g) * X
  template <typename _Derived = Derived>
  static LieAlgebra Ad(const _Derived& g, const LieAlgebra& X) {
    static_assert(
        std::is_base_of<ProductLieGroupBase, _Derived>::value,
        "Invalid derived type");
    return _Ad_impl<_Derived>(g, X, std::index_sequence_for<LieGroups...>{});
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& X) {
    return _ad_impl(X, std::index_sequence_for<LieGroups...>{});
  }

  // bracket(X1, X2) = [X1, X2] = ad(X1) * X2
  static LieAlgebra bracket(const LieAlgebra& X1, const LieAlgebra& X2) {
    return _bracket_impl(X1, X2, std::index_sequence_for<LieGroups...>{});
  }

  static LieAlgebraEndomorphism Jl(const LieAlgebra& X) {
    return _Jl_impl(X, std::index_sequence_for<LieGroups...>{});
  }

  static LieAlgebraEndomorphism Jr(const LieAlgebra& X) {
    return _Jr_impl(X, std::index_sequence_for<LieGroups...>{});
  }

  static LieAlgebraEndomorphism invJl(const LieAlgebra& X) {
    return _invJl_impl(X, std::index_sequence_for<LieGroups...>{});
  }

  static LieAlgebraEndomorphism invJr(const LieAlgebra& X) {
    return _invJr_impl(X, std::index_sequence_for<LieGroups...>{});
  }

  static Ambient hat(const LieAlgebra& X) {
    return _hat_impl(X, std::index_sequence_for<LieGroups...>{});
  }

  static LieAlgebra vee(const Ambient& X_hat) {
    return _vee_impl(X_hat, std::index_sequence_for<LieGroups...>{});
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    LieAlgebra w = LieAlgebra::Zero();
    w(i) = liegroup::Constants<Scalar>::kNum_1;
    return hat(w);
  }

 public:
  /// @name Perturbations
  /// @{

  /// For a direct product of Lie groups, the Left/Right perturbation is
  /// the direct product of the Left/Right perturbations of the individual
  /// Lie groups. We can access the individual perturbations by calling
  /// LeftPerturbation::part<0>(), LeftPerturbation::get<1>(), etc.
  template <typename LieGroup>
  using LeftPerturbationTemplate = liegroup_internal::ProductPerturbation<
      LieGroup, typename LieGroups::LeftPerturbation...>;

  template <typename LieGroup>
  using RightPerturbationTemplate = liegroup_internal::ProductPerturbation<
      LieGroup, typename LieGroups::RightPerturbation...>;

  using LeftPerturbation = LeftPerturbationTemplate<Derived>;

  using RightPerturbation = RightPerturbationTemplate<Derived>;

  /// Note SubLeftPerturbation / SubRightPerturbation are not a direct
  /// product of the individual Lie groups' SubLeftPerturbation /
  /// SubRightPerturbation. We can't access the individual perturbations
  /// by calling part<>().
  template <typename SubSpaceType>
  using SubLeftPerturbation =
      liegroup_internal::SubSpacePerturbation<LeftPerturbation, SubSpaceType>;

  template <typename SubSpaceType>
  using SubRightPerturbation =
      liegroup_internal::SubSpacePerturbation<RightPerturbation, SubSpaceType>;

  /// @}
  /// @name Optimizables
  /// @{

  using LeftOptimizable = OptimizableManifold<Derived, LeftPerturbation>;

  using RightOptimizable = OptimizableManifold<Derived, RightPerturbation>;

  template <typename SubSpaceType, bool _share_perturbation = true>
  using SubLeftOptimizable = OptimizableManifold<
      Derived, SubLeftPerturbation<SubSpaceType>, _share_perturbation>;

  template <typename SubSpaceType, bool _share_perturbation = true>
  using SubRightOptimizable = OptimizableManifold<
      Derived, SubRightPerturbation<SubSpaceType>, _share_perturbation>;

  /// @}

 protected:
  // Helper for multiplication
  template <typename _Derived, std::size_t... Is>
  _Derived _multiply(const _Derived& other, std::index_sequence<Is...>) const {
    return _Derived((std::get<Is>(parts_) * std::get<Is>(other.parts_))...);
  }

  // Helper for inverse
  template <typename _Derived, std::size_t... Is>
  _Derived _inverse_impl(std::index_sequence<Is...>) const {
    return _Derived((std::get<Is>(parts_).inverse())...);
  }

  // Helper for isApprox
  template <typename _Derived, std::size_t... Is>
  bool _isApprox_impl(
      const _Derived& other, const Scalar& eps,
      std::index_sequence<Is...>) const {
    return (
        ... && std::get<Is>(parts_).isApprox(std::get<Is>(other.parts_), eps));
  }

  // Helper for data
  template <std::size_t... Is>
  DataIterator _data_impl(std::index_sequence<Is...>) {
    return DataIterator::ConcatenateDataForLieGroups(std::get<Is>(parts_)...);
  }

  // Helper for data
  template <std::size_t... Is>
  ConstDataIterator _data_impl(std::index_sequence<Is...>) const {
    return ConstDataIterator::ConcatenateDataForLieGroups(
        std::get<Is>(parts_)...);
  }

  template <typename _RetType, std::size_t... Is>
  _RetType _cast_impl(std::index_sequence<Is...>) const {
    return _RetType(
        (std::get<Is>(parts_).template cast<typename _RetType::Scalar>())...);
  }

  // Helper for exp
  template <typename _Derived, std::size_t... Is>
  static _Derived _exp_impl(const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return _Derived((LieGroups::Exp(partitioned_X.template part<Is>()))...);
  }

  // Helper for log
  template <typename _Derived, std::size_t... Is>
  static LieAlgebra _log_impl(const _Derived& g, std::index_sequence<Is...>) {
    return PartitionedLieAlgebra((LieGroups::Log(std::get<Is>(g.parts_)))...);
  }

  // Helper for Ad
  template <typename _Derived, std::size_t... Is>
  static LieAlgebraEndomorphism _Ad_impl(
      const _Derived& g, std::index_sequence<Is...>) {
    // return LieAlgebraEndomorphism((LieGroups.Ad(std::get<Is>(g.parts_)))...);
    return LieAlgebraEndomorphism(LieGroups::Ad(std::get<Is>(g.parts_))...);
  }

  template <typename _Derived, std::size_t... Is>
  static LieAlgebra _Ad_impl(
      const _Derived& g, const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return PartitionedLieAlgebra(LieGroups::Ad(
        std::get<Is>(g.parts_), partitioned_X.template part<Is>())...);
  }

  // Helper for ad
  template <std::size_t... Is>
  static LieAlgebraEndomorphism _ad_impl(
      const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return LieAlgebraEndomorphism(
        LieGroups::ad(partitioned_X.template part<Is>())...);
  }

  // Helper for Jl
  template <std::size_t... Is>
  static LieAlgebraEndomorphism _Jl_impl(
      const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return LieAlgebraEndomorphism(
        LieGroups::Jl(partitioned_X.template part<Is>())...);
  }

  // Helper for Jr
  template <std::size_t... Is>
  static LieAlgebraEndomorphism _Jr_impl(
      const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return LieAlgebraEndomorphism(
        LieGroups::Jr(partitioned_X.template part<Is>())...);
  }

  // Helper for invJl
  template <std::size_t... Is>
  static LieAlgebraEndomorphism _invJl_impl(
      const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return LieAlgebraEndomorphism(
        LieGroups::invJl(partitioned_X.template part<Is>())...);
  }

  // Helper for invJr
  template <std::size_t... Is>
  static LieAlgebraEndomorphism _invJr_impl(
      const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return LieAlgebraEndomorphism(
        LieGroups::invJr(partitioned_X.template part<Is>())...);
  }

  // Helper for bracket
  template <std::size_t... Is>
  static LieAlgebra _bracket_impl(
      const LieAlgebra& X1, const LieAlgebra& X2, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X1 =
        static_cast<const PartitionedLieAlgebra&>(X1);
    const PartitionedLieAlgebra& partitioned_X2 =
        static_cast<const PartitionedLieAlgebra&>(X2);
    return PartitionedLieAlgebra((LieGroups::bracket(
        partitioned_X1.template part<Is>(),
        partitioned_X2.template part<Is>()))...);
  }

  template <std::size_t... Is>
  static Ambient _hat_impl(const LieAlgebra& X, std::index_sequence<Is...>) {
    const PartitionedLieAlgebra& partitioned_X =
        static_cast<const PartitionedLieAlgebra&>(X);
    return Ambient((LieGroups::hat(partitioned_X.template part<Is>()))...);
  }

  template <std::size_t... Is>
  static LieAlgebra _vee_impl(
      const Ambient& X_hat, std::index_sequence<Is...>) {
    return PartitionedLieAlgebra(
        (LieGroups::vee(X_hat.template part<Is>()))...);
  }

 protected:
  std::tuple<LieGroups...> parts_;
};

// The default implementation of ProductLieGroup.
template <typename... LieGroups>
class ProductLieGroup
    : public ProductLieGroupSimpleBase<ProductLieGroup, LieGroups...> {
  using _ProductLieGroupBase =
      ProductLieGroupSimpleBase<ProductLieGroup, LieGroups...>;

 public:
  // Import the constructors from the base class
  using _ProductLieGroupBase::_ProductLieGroupBase;
};

}  // namespace sk4slam
