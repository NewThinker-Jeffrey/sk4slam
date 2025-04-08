#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

template <int _dim = Eigen::Dynamic, typename _Scalar = double>
using Vector = Eigen::Matrix<_Scalar, _dim, 1>;

using Vector1d = Vector<1, double>;
using Vector2d = Vector<2, double>;
using Vector3d = Vector<3, double>;
using Vector4d = Vector<4, double>;
using VectorXd = Vector<Eigen::Dynamic, double>;

template <typename>
inline constexpr bool IsVector = false;

template <int _dim, typename _Scalar>
inline constexpr bool IsVector<Vector<_dim, _Scalar>> = true;

template <typename ScalarType, int... _part_dims>
class PartitionedVector;

template <template <typename...> class DerivedBase, typename... VectorSpaces>
class ProductVectorSpaceBase;

template <typename... VectorSpaces>
class ProductVectorSpace;

template <template <typename...> class DerivedBase, typename... Algebras>
class ProductAlgebraBase;

template <typename... Algebras>
class ProductAlgebra;

template <template <typename...> class DerivedBase, typename... Endomorphisms>
class ProductLinearEndomorphismBase;

template <typename... Endomorphisms>
class ProductLinearEndomorphism;

////////// PartitionedVector //////////

template <typename ScalarType, int... _part_dims>
class PartitionedVector
    : public Eigen::Matrix<ScalarType, (0 + ... + _part_dims), 1> {
  ///////// Template metafunctions  //////////
  template <int _idx, int _dim_offset, int... _tmp_part_dims>
  struct _PartWrapper;

  template <int _idx, int _dim_offset, int _part0_dim, int... _rest_part_dims>
  struct _PartWrapper<_idx, _dim_offset, _part0_dim, _rest_part_dims...> {
    using Part = typename _PartWrapper<
        _idx - 1, _dim_offset + _part0_dim, _rest_part_dims...>::Part;
  };

  template <int _dim_offset, int _part0_dim, int... _rest_part_dims>
  struct _PartWrapper<0, _dim_offset, _part0_dim, _rest_part_dims...> {
    struct Part {
      static constexpr int kDimOffset = _dim_offset;
      static constexpr int kDim = _part0_dim;
    };
  };

  template <int _idx>
  using _Part = typename _PartWrapper<_idx, 0, _part_dims...>::Part;

  using EigenBase = Eigen::Matrix<ScalarType, (0 + ... + _part_dims), 1>;

 public:
  template <typename _ScalarType>
  PartitionedVector<_ScalarType, _part_dims...> cast() const {
    return PartitionedVector<_ScalarType, _part_dims...>(
        this->EigenBase::template cast<_ScalarType>());
  }

  PartitionedVector() : EigenBase() {}

  // clang-format off
  // template <typename... Arg>
  // PartitionedVector(Arg&&... args)  // NOLINT
  //     : EigenBase(std::forward<Arg>(args)...) {}
  // clang-format on

  template <typename FirstSegment, typename... RestSegments>
  PartitionedVector(
      const FirstSegment& first, const RestSegments&... rest) {  // NOLINT
    // ((*this) << first) returns an unfinished Eigen::CommaInitializer.
    Eigen::CommaInitializer<EigenBase> comma_initializer = ((*this) << first);
    // Eigen::CommaInitializer<EigenBase> comma_initializer(*this, first);

    (comma_initializer, ..., rest);
  }

  // Partial specialization of the above variadic constructor template for a
  // single argument (which is necessarily the copy constructor).
  template <typename Other>
  explicit PartitionedVector(const Other& other) : EigenBase(other) {}

  // Some helper interfaces for LieAlgebra direct product.
  template <size_t _part_idx>
  auto part() {
    return this->template segment<_Part<_part_idx>::kDim>(
        _Part<_part_idx>::kDimOffset);
  }

  template <size_t _part_idx>
  const auto part() const {
    return this->template segment<_Part<_part_idx>::kDim>(
        _Part<_part_idx>::kDimOffset);
  }
};

////////// Direct Product of VectorSpaces //////////

template <template <typename...> class DerivedBase, typename... VectorSpaces>
class ProductVectorSpaceBase {
 protected:
  ///////// Template metafunctions  //////////
  template <typename Fisrt, typename... Rest>
  struct _CheckScalarTypes {
    using Scalar = typename Fisrt::Scalar;
    static constexpr bool value =
        (std::is_same_v<Scalar, typename Rest::Scalar> && ...);
  };

  static_assert(
      _CheckScalarTypes<VectorSpaces...>::value,
      "All VectorSpaces must have the same scalar type");

  using DefaultDerived = DerivedBase<VectorSpaces...>;

  template <typename _TgtScalar, typename _PartType>
  static decltype(auto) _cast_part_impl(const _PartType& part) {
    using _SrcScalar = typename _PartType::Scalar;
    if constexpr (std::is_same<_SrcScalar, _TgtScalar>::value) {
      return static_cast<const _PartType&>(part);
    } else {
      using _DirectCastType = decltype(part.template cast<_TgtScalar>());
      using _CastBackType =
          decltype(std::declval<_DirectCastType>().template cast<_SrcScalar>());
      if constexpr (std::is_same<_PartType, _CastBackType>::value) {
        return part.template cast<_TgtScalar>();
      } else {
        // _PartType is native Eigen::Matrix
        static constexpr int kRows = _PartType::RowsAtCompileTime;
        static constexpr int kCols = _PartType::ColsAtCompileTime;
        static constexpr auto kOptions = _PartType::Options;
        static_assert(
            kRows > 0 && kCols > 0,
            "Matrix must have positive (fixed) dimensions");
        using _SrcMatrixType =
            Eigen::Matrix<_SrcScalar, kRows, kCols, kOptions>;
        using _TgtMatrixType =
            Eigen::Matrix<_TgtScalar, kRows, kCols, kOptions>;
        static_assert(
            std::is_same<_SrcMatrixType, _PartType>::value,
            "part should be of type Eigen::Matrix");
        return _TgtMatrixType(part.template cast<_TgtScalar>());
      }
    }
  }

  template <typename _TgtScalar, typename _PartType>
  using _PartCastTemplate = std::remove_cv_t<std::remove_reference_t<
      decltype(_cast_part_impl<_TgtScalar>(std::declval<_PartType>()))>>;

  template <typename _ScalarType>
  using _CastTemplate =
      DerivedBase<_PartCastTemplate<_ScalarType, VectorSpaces>...>;

 public:
  using Scalar = typename _CheckScalarTypes<VectorSpaces...>::Scalar;

  ProductVectorSpaceBase() {}

  // template <typename... _VectorSpaces>
  // explicit ProductVectorSpaceBase(_VectorSpaces&&... parts)
  //     : parts_(std::forward<_VectorSpaces>(parts)...) {}

  template <typename... _VectorSpaces>
  explicit ProductVectorSpaceBase(const _VectorSpaces&... parts)
      : parts_((parts)...) {}

  // Partial specialization of the above variadic constructor template for a
  // single argument (which is necessarily the copy constructor).
  template <typename Other>
  explicit ProductVectorSpaceBase(const Other& other) : parts_(other.parts_) {}

  template <std::size_t _part_idx>
  auto& part() {
    return std::get<_part_idx>(parts_);
  }

  template <std::size_t _part_idx>
  const auto& part() const {
    return std::get<_part_idx>(parts_);
  }

  template <typename _ScalarType>
  _CastTemplate<_ScalarType> cast() const {
    return this->template _cast_impl<_CastTemplate<_ScalarType>>(
        std::index_sequence_for<VectorSpaces...>{});
  }

 public:
  //// Vector space operations /////

  template <typename _Derived = DefaultDerived>
  static _Derived Zero() {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _Derived(VectorSpaces::Zero()...);
  }

  template <typename _Derived = DefaultDerived>
  bool isZero(const Scalar& eps = Eigen::NumTraits<Scalar>::epsilon()) const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _isZero_impl<_Derived>(
        eps, std::index_sequence_for<VectorSpaces...>{});
  }

  template <typename _Derived = DefaultDerived>
  bool isApprox(
      const _Derived& other,
      const Scalar& eps = Eigen::NumTraits<Scalar>::epsilon()) const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _isApprox_impl<_Derived>(
        other, eps, std::index_sequence_for<VectorSpaces...>{});
  }

  template <typename _Derived = DefaultDerived>
  _Derived operator+(const _Derived& other) const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _plus_impl<_Derived>(
        other, std::index_sequence_for<VectorSpaces...>{});
  }

  template <typename _Derived = DefaultDerived>
  _Derived operator-(const _Derived& other) const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _minus_impl<_Derived>(
        other, std::index_sequence_for<VectorSpaces...>{});
  }

  template <typename _Derived = DefaultDerived>
  _Derived operator*(const Scalar& scalar) const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _multiply_scalar_impl<_Derived>(
        scalar, std::index_sequence_for<VectorSpaces...>{});
  }

  template <typename _Derived = DefaultDerived>
  _Derived operator/(const Scalar& scalar) const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    const _Derived& this_cast = *static_cast<const _Derived*>(this);
    return this_cast * (1.0 / scalar);
  }

  template <typename _Derived = DefaultDerived>
  _Derived operator-() const {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    return _negate_impl<_Derived>(std::index_sequence_for<VectorSpaces...>{});
  }

  template <typename _Derived = DefaultDerived>
  _Derived& operator+=(const _Derived& other) {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    _Derived& this_cast = *static_cast<_Derived*>(this);
    this_cast = this_cast + other;
    return this_cast;
  }

  template <typename _Derived = DefaultDerived>
  _Derived& operator-=(const _Derived& other) {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    _Derived& this_cast = *static_cast<_Derived*>(this);
    this_cast = this_cast - other;
    return this_cast;
  }

  template <typename _Derived = DefaultDerived>
  _Derived& operator*=(const Scalar& scalar) {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    _Derived& this_cast = *static_cast<_Derived*>(this);
    this_cast = this_cast * scalar;
    return this_cast;
  }

  template <typename _Derived = DefaultDerived>
  _Derived& operator/=(const Scalar& scalar) {
    static_assert(
        std::is_base_of<ProductVectorSpaceBase, _Derived>::value,
        "Invalid derived type");
    _Derived& this_cast = *static_cast<_Derived*>(this);
    this_cast = this_cast / scalar;
    return this_cast;
  }

 protected:
  template <typename _Derived, std::size_t... Is>
  _Derived _plus_impl(const _Derived& other, std::index_sequence<Is...>) const {
    return _Derived(part<Is>() + other.template part<Is>()...);
  }

  template <typename _Derived, std::size_t... Is>
  _Derived _minus_impl(
      const _Derived& other, std::index_sequence<Is...>) const {
    return _Derived(part<Is>() - other.template part<Is>()...);
  }

  template <typename _Derived, std::size_t... Is>
  _Derived _multiply_scalar_impl(
      const Scalar& scalar, std::index_sequence<Is...>) const {
    return _Derived(part<Is>() * scalar...);
  }

  template <typename _Derived, std::size_t... Is>
  _Derived _negate_impl(std::index_sequence<Is...>) const {
    return _Derived(-part<Is>()...);
  }

  template <typename _Derived, std::size_t... Is>
  bool _isApprox_impl(
      const _Derived& other, const Scalar& eps,
      std::index_sequence<Is...>) const {
    return (
        ... && std::get<Is>(parts_).isApprox(std::get<Is>(other.parts_), eps));
  }

  template <typename _Derived, std::size_t... Is>
  bool _isZero_impl(const Scalar& eps, std::index_sequence<Is...>) const {
    return (... && std::get<Is>(parts_).isZero(eps));
  }

  template <typename _RetType, std::size_t... Is>
  decltype(auto) _cast_impl(std::index_sequence<Is...>) const {
    if constexpr (std::is_same_v<typename _RetType::Scalar, Scalar>) {
      return static_cast<const _RetType&>(*this);
    } else {
      return _RetType((
          _cast_part_impl<typename _RetType::Scalar>(std::get<Is>(parts_)))...);
    }
  }

 protected:
  std::tuple<VectorSpaces...> parts_;
};

template <typename... VectorSpaces>
class ProductVectorSpace
    : public ProductVectorSpaceBase<ProductVectorSpace, VectorSpaces...> {
  using _ProductVectorSpaceBase =
      ProductVectorSpaceBase<ProductVectorSpace, VectorSpaces...>;

 public:
  // Import the constructors from the base class
  using _ProductVectorSpaceBase::_ProductVectorSpaceBase;
};

////////// Direct Product of Algebras //////////
template <template <typename...> class DerivedBase, typename... Algebras>
class ProductAlgebraBase
    : public ProductVectorSpaceBase<DerivedBase, Algebras...> {
 protected:
  ///////// Template metafunctions  //////////
  using DefaultDerived = DerivedBase<Algebras...>;

  using _ProductVectorSpaceBase =
      ProductVectorSpaceBase<DerivedBase, Algebras...>;

  static_assert(
      std::is_same_v<
          DefaultDerived, typename _ProductVectorSpaceBase::DefaultDerived>,
      "the DefaultDerived types do not match between ProductAlgebraBase "
      "and ProductVectorSpaceBase!");

 public:
  // Import the constructors from the base class
  using _ProductVectorSpaceBase::_ProductVectorSpaceBase;

 public:
  //// extra operations for an Algebra  ////

  template <typename _Derived = DefaultDerived>
  static _Derived Identity() {
    static_assert(
        std::is_base_of<ProductAlgebraBase, _Derived>::value,
        "Invalid derived type");
    return _Derived(Algebras::Identity()...);
  }
  template <
      typename _Derived = DefaultDerived,
      ENABLE_IF((std::is_base_of<ProductAlgebraBase, _Derived>::value))>
  _Derived operator*(const _Derived& other) const {
    static_assert(
        std::is_base_of<ProductAlgebraBase, _Derived>::value,
        "Invalid derived type");
    return _compose_impl<_Derived>(
        other, std::index_sequence_for<Algebras...>{});
  }
  template <
      typename _Derived = DefaultDerived,
      ENABLE_IF((std::is_base_of<ProductAlgebraBase, _Derived>::value))>
  _Derived& operator*=(const _Derived& other) {
    static_assert(
        std::is_base_of<ProductAlgebraBase, _Derived>::value,
        "Invalid derived type");
    _Derived& this_cast = *static_cast<_Derived*>(this);
    this_cast = this_cast * other;
    return this_cast;
  }

  // Since we overload the operator*() and operator*=() as above, which covers
  // the operator*() and operator*=() in ProductVectorSpaceBase,
  // we need to explicitly re-declare them here.
  using _ProductVectorSpaceBase::operator*;
  using _ProductVectorSpaceBase::operator*=;

 protected:
  template <typename _Derived, std::size_t... Is>
  _Derived _compose_impl(
      const _Derived& other, std::index_sequence<Is...>) const {
    const _Derived& this_cast = *static_cast<const _Derived*>(this);
    return _Derived(
        this_cast.template part<Is>() * other.template part<Is>()...);
  }
};

template <typename... Algebras>
class ProductAlgebra : public ProductAlgebraBase<ProductAlgebra, Algebras...> {
  using _ProductAlgebraBase = ProductAlgebraBase<ProductAlgebra, Algebras...>;

 public:
  // Import the constructors from the base class
  using _ProductAlgebraBase::_ProductAlgebraBase;
};

////////// Direct Product of Linear Endomorphisms //////////

template <template <typename...> class DerivedBase, typename... Endomorphisms>
class ProductLinearEndomorphismBase
    : public ProductAlgebraBase<DerivedBase, Endomorphisms...> {
 protected:
  ///////// Template metafunctions  //////////
  using DefaultDerived = DerivedBase<Endomorphisms...>;

  using _ProductAlgebraBase = ProductAlgebraBase<DerivedBase, Endomorphisms...>;

  static_assert(
      std::is_same_v<
          DefaultDerived, typename _ProductAlgebraBase::DefaultDerived>,
      "the DefaultDerived types do not match between "
      "ProductLinearEndomorphismBase and ProductAlgebraBase!");

 public:
  // Import the constructors from the base class
  using _ProductAlgebraBase::_ProductAlgebraBase;

  // To mimic Eigen::Matrix
  static constexpr int RowsAtCompileTime =
      (0 + ... + Endomorphisms::RowsAtCompileTime);

  using Scalar = typename _ProductAlgebraBase::Scalar;

  Eigen::Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> matrix() const {
    return _to_matrix_impl(std::index_sequence_for<Endomorphisms...>{});
  }

  // static cast to Eigen::Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>
  operator Eigen::Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>() const {
    return _to_matrix_impl(std::index_sequence_for<Endomorphisms...>{});
  }

 public:
  //// extra operations for a linear Endomorphism  ////

  // The action on the product vector space:
  // Map a vector in the ProductSpace to another
  template <typename Other>
  decltype(auto) operator*(const Other& X) const {
    using Scalar = typename _ProductAlgebraBase::Scalar;
    // Since we're overloading the operator*(), which will cover
    // the operator*() in ProductAlgebraBase, we need to
    // check and redirect the call if X is a scalar or another
    // element of the algebra.
    if constexpr (std::is_same_v<Other, Scalar>) {
      // Scalar multiplication
      return this->_ProductAlgebraBase::operator*(X);
    } else if constexpr (  // NOLINT
        std::is_base_of<_ProductAlgebraBase, Other>::value) {
      // Algebra multiplication
      return this->_ProductAlgebraBase::operator*(X);
    } else {
      using _ProductSpace = Other;
      // Linear action on ProductSpace
      return _action_impl(X, std::index_sequence_for<Endomorphisms...>{});
    }
  }

  // A inverse() function is required but not every endomorphism
  // has an inverse. The behavior of inverse() is undefined
  // when it is called on an endomorphism that is not invertible.
  template <typename _Derived = DefaultDerived>
  _Derived inverse() const {
    static_assert(
        std::is_base_of<ProductLinearEndomorphismBase, _Derived>::value,
        "Invalid derived type");
    return _inverse_impl<_Derived>(std::index_sequence_for<Endomorphisms...>{});
  }

 protected:
  template <typename _ProductSpace, std::size_t... Is>
  decltype(auto) _action_impl(
      const _ProductSpace& X, std::index_sequence<Is...>) const {
    using _Scalar = typename _ProductAlgebraBase::Scalar;
    using _PartitionedVector =
        PartitionedVector<_Scalar, Endomorphisms::RowsAtCompileTime...>;
    if constexpr (std::is_base_of<_PartitionedVector, _ProductSpace>::value) {
      return _ProductSpace(
          this->template part<Is>() * X.template part<Is>()...);
    } else {
      const _PartitionedVector& partioned_X =
          static_cast<const _PartitionedVector&>(X);
      return _PartitionedVector(
          this->template part<Is>() * partioned_X.template part<Is>()...);
    }
  }

  template <typename _Derived, std::size_t... Is>
  _Derived _inverse_impl(std::index_sequence<Is...>) const {
    return _Derived((this->template part<Is>().inverse())...);
  }

  // Implementation of static cast to Eigen::Matrix

  template <int _idx, int _row_offset, int... _tmp_part_rows>
  struct _PartRowsWrapper;

  template <int _idx, int _row_offset, int _part0_rows, int... _rest_part_rows>
  struct _PartRowsWrapper<_idx, _row_offset, _part0_rows, _rest_part_rows...> {
    using Part = typename _PartRowsWrapper<
        _idx - 1, _row_offset + _part0_rows, _rest_part_rows...>::Part;
  };

  template <int _row_offset, int _part0_rows, int... _rest_part_rows>
  struct _PartRowsWrapper<0, _row_offset, _part0_rows, _rest_part_rows...> {
    struct Part {
      static constexpr size_t kRowOffset = _row_offset;
      static constexpr size_t kRows = _part0_rows;
    };
  };

  template <int _idx>
  using _PartRows = typename _PartRowsWrapper<
      _idx, 0, Endomorphisms::RowsAtCompileTime...>::Part;

  template <std::size_t... Is>
  Eigen::Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> _to_matrix_impl(
      std::index_sequence<Is...>) const {
    Eigen::Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> matrix;
    matrix.setZero();

    ((matrix.template block<_PartRows<Is>::kRows, _PartRows<Is>::kRows>(
          _PartRows<Is>::kRowOffset, _PartRows<Is>::kRowOffset) =
          static_cast<Eigen::Matrix<
              Scalar, _PartRows<Is>::kRows, _PartRows<Is>::kRows>>(
              this->template part<Is>())),
     ...);
    return matrix;
  }
};

template <typename... Endomorphisms>
class ProductLinearEndomorphism
    : public ProductLinearEndomorphismBase<
          ProductLinearEndomorphism, Endomorphisms...> {
  using _ProductLinearEndomorphismBase = ProductLinearEndomorphismBase<
      ProductLinearEndomorphism, Endomorphisms...>;

 public:
  // Import the constructors from the base class
  using _ProductLinearEndomorphismBase::_ProductLinearEndomorphismBase;
};

}  // namespace sk4slam
