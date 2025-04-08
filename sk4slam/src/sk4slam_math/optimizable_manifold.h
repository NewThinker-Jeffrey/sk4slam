#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/likely.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_math/vector_space.h"

namespace sk4slam {

template <typename Manifold>
class manifold_traits {
 public:
  using Scalar = typename Manifold::Scalar;
  static constexpr int kDim = Manifold::kDim;
  static constexpr int kAmbientDim = Manifold::kAmbientDim;

  /// @brief This template is used to cast a manifold to a new scalar type.
  ///        The Manifold must have a member function template
  ///        cast<NewScalar>() that returns a new manifold of the same
  ///        kind but with the new scalar type.
  template <typename NewScalar>
  using Cast = decltype(std::declval<Manifold>().template cast<NewScalar>());

  static bool isApprox(
      const Manifold& lhs, const Manifold& rhs,
      const Scalar& eps = Eigen::NumTraits<Scalar>::epsilon()) {
    return lhs.isApprox(rhs, eps);
  }

  /// Runtime dimension of the manifold.
  static int dim(const Manifold& manifold) {
    return manifold.dim();
  }

  /// Runtime ambient dimension of the manifold.
  static int ambientDim(const Manifold& manifold) {
    return manifold.ambientDim();
  }
};

/// @brief Specialization for vector space.
template <int _dim, typename _Scalar>
class manifold_traits<Vector<_dim, _Scalar>> {
  static inline _Scalar kNum_1 = _Scalar(1.);
  using Vector_ = Vector<_dim, _Scalar>;

 public:
  using Scalar = _Scalar;
  static constexpr int kDim = _dim;
  static constexpr int kAmbientDim = _dim;

  /// Note Eigen::Matrix::cast() doesn't return a new Eigen::Matrix object, but
  /// a Eigen::CwiseUnaryOp object. So we need to cast it to the right type.
  template <typename NewScalar>
  using Cast = Vector<kDim, NewScalar>;

  // The isApprox() function in Eigen typically evaluates the relative error
  // between two matrices. This can be misleading when both matrices have
  // norms close to zero, as the relative error might appear excessively large
  // or even infinite, despite the matrices being similar. To address this, we
  // adaptively choose between using absolute and relative error to assess
  // the closeness of the matrices.
  static bool isApprox(
      const Vector_& lhs, const Vector_& rhs,
      const Scalar& eps = Eigen::NumTraits<Scalar>::epsilon()) {
    // if (other.array().abs().maxCoeff() > kNum_1) {
    if (rhs.squaredNorm() > kNum_1) {
      return lhs.isApprox(rhs, eps);
    } else {
      return (lhs - rhs).isZero(eps);
    }
  }

  static int dim(const Vector_& vec) {
    return vec.size();
  }

  static int ambientDim(const Vector_& vec) {
    return vec.size();
  }
};

/// @brief  The CRTP base template for direct product of manifolds.
/// @tparam DerivedProductTemplate  The derived product template
/// @tparam ManifoldParts  The manifold parts in the product
///
/// @note Note the current implementation doesn't support
///       product manifold with dynamic dimension, so @c ManifoldParts
///       shouldn't contain a part with dynamic dimension!
template <
    template <typename...> class DerivedProductTemplate,
    typename... ManifoldParts>
class ProductManifoldBase {
  template <typename Fisrt, typename... Rest>
  struct _CheckScalarTypes {
    using Scalar = typename manifold_traits<Fisrt>::Scalar;
    static constexpr bool value =
        (std::is_same_v<Scalar, typename manifold_traits<Rest>::Scalar> && ...);
  };

  static_assert(
      _CheckScalarTypes<ManifoldParts...>::value,
      "All ManifoldParts must have the same scalar type");

  template <typename _ScalarType>
  using _CastTemplate = DerivedProductTemplate<
      typename manifold_traits<ManifoldParts>::template Cast<_ScalarType>...>;

  using Derived = DerivedProductTemplate<ManifoldParts...>;

 public:
  static constexpr int kParts = sizeof...(ManifoldParts);
  static constexpr int kPartDims[] = {manifold_traits<ManifoldParts>::kDim...};
  static constexpr int kPartAmbientDims[] = {
      manifold_traits<ManifoldParts>::kAmbientDim...};
  static constexpr int kDim = (0 + ... + manifold_traits<ManifoldParts>::kDim);
  static constexpr int kAmbientDim =
      (0 + ... + manifold_traits<ManifoldParts>::kAmbientDim);

  using Scalar = typename _CheckScalarTypes<ManifoldParts...>::Scalar;

  ProductManifoldBase() {}

  template <typename... _ManifoldParts>
  explicit ProductManifoldBase(const _ManifoldParts&... parts)
      : parts_((parts)...) {}

  // Partial specialization of the above variadic constructor template for a
  // single argument (which is necessarily the copy constructor).
  template <typename Other>
  explicit ProductManifoldBase(const Other& other) : parts_(other.parts_) {}

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
        std::index_sequence_for<ManifoldParts...>{});
  }

  int dim() const {
    return kDim;
  }

  int ambientDim() const {
    return kAmbientDim;
  }

  bool isApprox(
      const Derived& other,
      const Scalar& eps = Eigen::NumTraits<Scalar>::epsilon()) const {
    return this->template _isApprox_impl(
        other, eps, std::index_sequence_for<ManifoldParts...>{});
  }

 protected:
  template <typename _RetType, std::size_t... Is>
  _RetType _cast_impl(std::index_sequence<Is...>) const {
    return _RetType(
        (std::get<Is>(parts_).template cast<typename _RetType::Scalar>())...);
  }

  template <std::size_t... Is>
  bool _isApprox_impl(
      const Derived& other, const Scalar& eps,
      std::index_sequence<Is...>) const {
    return (
        ... && std::get<Is>(parts_).isApprox(std::get<Is>(other.parts_), eps));
  }

 protected:
  std::tuple<ManifoldParts...> parts_;
};

/// @brief  The default implementation of the direct product manifold.
/// @tparam ...ManifoldParts
template <typename... ManifoldParts>
class ProductManifold
    : public ProductManifoldBase<ProductManifold, ManifoldParts...> {
  using Base = ProductManifoldBase<ProductManifold, ManifoldParts...>;

 public:
  using Base::Base;
};

/// @brief This is the interface class for Retractions. All retraction
///        classes must inherit from this class to allow different
///        retractions to be stored in the same container.
///
/// Roughly, a Retraction on a Manifold @f$ M @f$ is a smooth mapping @f$ R @f$
/// from the tangent bundle @f$ TM @f$ onto @f$ M @f$, satisfying:
///
/// (i)  @f$ R_x(0_x) = x @f$
///      where @f$ 0_x @f$ denotes the origin of @f$ T_x M @f$
///      (the tangent space at @f$ x \in M @f$), and @f$ R_x @f$ denotes
///      the restriction of @f$ R @f$ to @f$ T_x M @f$.
///
/// (ii) @f$ T_0 R_x = Id @f$
///      meaning the differential of @f$ R_x @f$ at 0 is the identity.
///
/// For a more detailed reference, see Definition 4.1.1 in [Optimization
/// Algorithms on Matrix Manifolds]
/// (http://ndl.ethernet.edu.et/bitstream/123456789/24450/1/P.%20A.%20Absil.pdf).
///
/// Note that for manifolds with "nice" properties (e.g., Lie groups), the
/// domain of the retraction can be the entire tangent bundle. However, for more
/// general manifolds, retractions are typically defined only locally. This is
/// why the definition above is referred to as "roughly" correct.
///
/// We also define a "section" at each point @f$ x \in M @f$ as the inverse
/// mapping of @f$ R_x @f$. Locally, a Retraction @f$ R @f$ can be viewed as a
/// [fiber
/// bundle](https://en.wikipedia.org/wiki/Fiber_bundle#Formal_definition):
///
/// In some small neighborhood @f$ U \subset M @f$, the preimage of @f$ U @f$
/// under @f$ R @f$, denoted by @f$ E := R^{-1}[U] @f$, can be viewed as the
/// total space and projected to the base space @f$ U @f$ via @f$ {R|_E} @f$
/// (the restriction of @f$ R @f$ to @f$ E @f$). The fiber @f$ F_x @f$ at each
/// point @f$ x \in U @f$ consists of all tangent vectors @f$ y \in E @f$ such
/// that @f$ R(y) = x @f$, i.e., @f$ F_x := {R|_E}^{-1}[x] @f$.
///
/// In differential geometry, a [section]
/// (https://en.wikipedia.org/wiki/Section_(fiber_bundle))
/// @f$ \sigma @f$ of a fiber bundle (@f$ E @f$ in our case) is a mapping
/// from the base space (@f$ U \subset M @f$ in our case) to the total space
/// such that @f$ \sigma(x) \in F_x @f$, i.e., @f$ R(\sigma(x)) = x @f$.
/// It's evident that @f$ R_x^{-1} @f$ is a section.
///
struct RetractionInterface {
  static constexpr int kDof = Eigen::Dynamic;
  static constexpr int kAmbientDim = Eigen::Dynamic;
  virtual ~RetractionInterface() = default;
  virtual int dimToDof(int manifold_dim) const = 0;
  virtual bool isFixed() const {
    return false;
  }
  virtual bool operator==(const RetractionInterface& other) const = 0;
  virtual bool operator!=(const RetractionInterface& other) const {
    return !(*this == other);
  }
  virtual const char* name() const = 0;

  /// @brief  The retraction at point @f$ x \in M @f$ is the mapping
  ///         @f$ R_x @f$. This operator computes @f$ R_x(t) @f$.
  ///         i.e. for a Retraction object `r`,  `r(x, t)` returns
  ///         @f$ R_x(t) @f$.
  template <typename Tangent, typename Manifold>
  Manifold operator()(const Manifold& x, const Tangent& t) const {
    static_assert(
        std::is_same_v<typename manifold_traits<Manifold>::Scalar, double>,
        "RetractionInterface: The scalar type of the manifold must be double!");
    // return x + t;
    ASSERT(checkManifoldType<Manifold>());
    ASSERT(t.rows() == dof(x));
    Manifold x_plus_t;
    (*this)(&x, t.data(), &x_plus_t);
    return x_plus_t;
  }

  /// @brief  The section at point @f$ x \in M @f$ is the inverse mapping
  ///         of @f$ R_x @f$.
  ///
  ///         For a Retraction object `r`, if `x2 = r(x, t)`, then
  ///         `r.section(x, x2) == t`.
  ///
  /// @tparam Manifold  The underlying manifold.
  /// @tparam Tangent   The tangent vector type.
  /// @param  x        The base point.
  /// @param  x2       The point after retraction.
  template <typename Tangent = VectorXd, typename Manifold = VectorXd>
  Tangent section(const Manifold& x, const Manifold& x2) const {
    static_assert(
        std::is_same_v<typename manifold_traits<Manifold>::Scalar, double>,
        "RetractionInterface: The scalar type of the manifold must be double!");
    // return x2 - x;
    ASSERT(checkManifoldType<Manifold>());
    Tangent x2_minus_x;
    if constexpr (Tangent::RowsAtCompileTime == Eigen::Dynamic) {
      x2_minus_x.resize(dof(x), 1);
    } else {
      ASSERT(Tangent::RowsAtCompileTime == dof(x));
    }
    sectionInterface(&x, &x2, x2_minus_x.data());
    return x2_minus_x;
  }

  /// @brief Runtime dof of the retraction at `x`.
  template <typename Manifold>
  int dof(const Manifold& x) const {
    // static_assert(
    //     std::is_same_v<typename manifold_traits<Manifold>::Scalar, double>,
    //     "RetractionInterface: The scalar type of the manifold must be
    //     double!");
    // ASSERT(checkManifoldType<Manifold>());
    return dimToDof(manifold_traits<Manifold>::dim(x));
  }

  /// @brief The Jacobian matrix type.
  /// @note The Jacobians are all stored in row-major order, to be compatible
  /// with
  ///       ceres-solver.
  template <
      typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
  using JacobianMatrix = std::conditional_t<
      _cols == 1,
      Eigen::Matrix<Scalar, _rows, 1>,  // Use Eigen::RowMajor when cols == 1
                                        // will cause a compile error.
      Eigen::Matrix<Scalar, _rows, _cols, Eigen::RowMajor>>;

  using JacobianMatrixXd =
      JacobianMatrix<double, Eigen::Dynamic, Eigen::Dynamic>;

  template <
      typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
  using JacobianMap = Eigen::Map<JacobianMatrix<Scalar, _rows, _cols>>;

  template <
      typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
  using ConstJacobianMap =
      Eigen::Map<const JacobianMatrix<Scalar, _rows, _cols>>;

  template <
      int _output_cols = Eigen::Dynamic, typename Manifold = VectorXd,
      typename JacobianMatrixWrtSrcRetraction = JacobianMatrixXd,
      typename SrcRetraction = RetractionInterface,
      ENABLE_IF((std::is_same_v<
                 SrcRetraction,
                 RetractionInterface>))  // To avoid ambiguity
                                         // with the
                                         // RetractionBase::transformJacobian()
                                         // defined below
      >
  auto transformJacobian(
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      const Manifold& x, const RetractionInterface& src_retraction) const {
    static_assert(
        std::is_same_v<typename manifold_traits<Manifold>::Scalar, double>,
        "RetractionInterface: The scalar type of the manifold must be double!");
    ASSERT(checkManifoldType<Manifold>());
    using JacobianMatrixWrtThisRetraction = JacobianMatrix<
        typename JacobianMatrixWrtSrcRetraction::Scalar,
        JacobianMatrixWrtSrcRetraction::RowsAtCompileTime, _output_cols>;
    JacobianMatrixWrtThisRetraction jacobian_under_this_retraction;
    if constexpr (
        JacobianMatrixWrtSrcRetraction::RowsAtCompileTime == Eigen::Dynamic ||
        _output_cols == Eigen::Dynamic) {
      jacobian_under_this_retraction.resize(
          jacobian_under_src_retraction.rows(), dof(x));
    } else if constexpr (_output_cols != Eigen::Dynamic) {
      ASSERT(_output_cols == dof(x));
    }

    // TODO(jeffrey): Maybe we can do finer checks on
    // JacobianMatrixWrtSrcRetraction to reduce unnecessary data copying.
    using ReorderedJacobianMatrixWrtSrcRetraction = JacobianMatrix<
        typename JacobianMatrixWrtSrcRetraction::Scalar,
        JacobianMatrixWrtSrcRetraction::RowsAtCompileTime,
        JacobianMatrixWrtSrcRetraction::ColsAtCompileTime>;
    if constexpr (std::is_same_v<
                      ReorderedJacobianMatrixWrtSrcRetraction,
                      JacobianMatrixWrtSrcRetraction>) {
      transformJacobianInterface(
          &x, &src_retraction, jacobian_under_src_retraction.data(),
          jacobian_under_src_retraction.rows(),
          jacobian_under_src_retraction.cols(),
          jacobian_under_this_retraction.data());
    } else {
      // Copy the data to a matrix with correct data layout
      ReorderedJacobianMatrixWrtSrcRetraction
          reordered_jacobian_under_src_retraction =
              jacobian_under_src_retraction;
      transformJacobianInterface(
          &x, &src_retraction, reordered_jacobian_under_src_retraction.data(),
          jacobian_under_src_retraction.rows(),
          jacobian_under_src_retraction.cols(),
          jacobian_under_this_retraction.data());
    }

    return jacobian_under_this_retraction;
  }

  template <typename Retraction>
  static const Retraction* defaultInstance() {
    static_assert(std::is_base_of_v<RetractionInterface, Retraction>);
    if constexpr (std::is_same_v<RetractionInterface, Retraction>) {
      return nullptr;
    } else {
      static const Retraction instance;
      return &instance;
    }
  }

 protected:
  using AnyManifold = void;
  virtual void operator()(
      const AnyManifold* x, const double* t, AnyManifold* x_plus_t) const = 0;

  virtual void sectionInterface(
      const AnyManifold* x, const AnyManifold* x2,
      double* x2_minus_x) const = 0;

  virtual void transformJacobianInterface(
      const AnyManifold* x, const RetractionInterface* src_retraction,
      const double* jacobian_under_src_retraction, int jacobian_under_src_rows,
      int jacobian_under_src_cols,
      double* jacobian_under_this_retraction) const = 0;

  virtual bool checkManifoldType(const std::type_info& manifold_type) const = 0;

  template <typename Manifold>
  bool checkManifoldType() const {
    if (UNLIKELY(!checkManifoldType(typeid(Manifold)))) {
      LOGE(
          "Retraction checkManifoldType(): manifold type missmatch!! Input "
          "Manifold = %s, retraction name = %s",
          classname<Manifold>(), name());
      return false;
    }
    return true;
  }

 protected:  // helper functions for derived classes
  template <typename... SrcRetractions>
  struct TransformJacobianImpl;

  template <typename... SrcRetractions>
  using DeclareTransformJacobianTypes =
      TransformJacobianImpl<SrcRetractions...>;

  template <typename, typename... NewRetractions>
  struct _InsertTransformJacobianTypes {
    using type = TransformJacobianImpl<NewRetractions...>;
  };

  template <typename... OldSrcRetractions, typename... NewSrcRetractions>
  struct _InsertTransformJacobianTypes<
      TransformJacobianImpl<OldSrcRetractions...>, NewSrcRetractions...> {
    // Put the new retractions at the front
    using type =
        TransformJacobianImpl<NewSrcRetractions..., OldSrcRetractions...>;
  };

  /// The final order of the retractions checked in TransformJacobianImpl() is:
  ///    - ThisRetraction (always first),
  ///    - NewRetractions...,
  ///    - FatherRetraction,
  ///    - FatherRetraction::TransformJacobianTypes...
  /// @warning It seems the compiler sometimes falls in a dead loop if we put
  ///          FatherRetraction at the end ?

  // clang-format off
  template <typename FatherRetraction, typename... NewRetractions>
  using ExtendTransformJacobianTypes = typename
      _InsertTransformJacobianTypes<
          typename FatherRetraction::TransformJacobianTypes,
          NewRetractions...                                                            // NOLINT
          , FatherRetraction  // <-- Remove this if the comipler falls in a dead loop  // NOLINT
      >::type;
  // clang-format on
};

template <typename... SrcRetractions>
struct RetractionInterface::TransformJacobianImpl {
  template <
      typename DstRetraction, typename Manifold,
      typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtDstRetraction>
  bool operator()(
      const Manifold& x, const DstRetraction& dst_retraction,
      const RetractionInterface& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtDstRetraction* jacobian_under_dst_retraction) const {
    return transformJacobianAgainstTypes<
        DstRetraction, Manifold, JacobianMatrixWrtSrcRetraction,
        JacobianMatrixWrtDstRetraction,
        DstRetraction,     // try transformJacobianImpl() against DstRetraction
                           // first,
        SrcRetractions...  // then SrcRetractions
        >(
        x, dst_retraction, src_retraction, jacobian_under_src_retraction,
        jacobian_under_dst_retraction);
  }

 protected:
  template <
      typename DstRetraction, typename Manifold,
      typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtDstRetraction, typename... Retractions>
  bool transformJacobianAgainstTypes(
      const Manifold& x, const DstRetraction& dst_retraction,
      const RetractionInterface& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtDstRetraction* jacobian_under_dst_retraction) const {
    static_assert(
        std::is_same_v<typename manifold_traits<Manifold>::Scalar, double>,
        "RetractionInterface: The scalar type of the manifold must be "
        "double!");
    bool handled =
        (transformJacobianAgainstOneType<Retractions>(
             x, dst_retraction, src_retraction, jacobian_under_src_retraction,
             jacobian_under_dst_retraction) ||
         ...);
    return handled;
  }

  template <
      typename SrcRetraction, typename DstRetraction, typename Manifold,
      typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtDstRetraction>
  bool transformJacobianAgainstOneType(
      const Manifold& x, const DstRetraction& dst_retraction,
      const RetractionInterface& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtDstRetraction* jacobian_under_dst_retraction) const {
    // clang-format off
    // // This requires EXACT type matching.
    // if (typeid(src_retraction) != typeid(SrcRetraction)) {
    //   return false;
    // }
    // const SrcRetraction& src_retraction_cast =
    //     dynamic_cast<const SrcRetraction&>(src_retraction);
    // clang-format on

    // This only requires dynamic_cast to succeed, i.e. `src_retraction`
    // can either be an instance of `SrcRetraction` or a subclass of
    // `SrcRetraction`.
    const SrcRetraction* src_retraction_cast_ptr =
        dynamic_cast<const SrcRetraction*>(&src_retraction);
    if (!src_retraction_cast_ptr) {
      return false;
    }
    const SrcRetraction& src_retraction_cast = *src_retraction_cast_ptr;

    if constexpr (std::is_same_v<SrcRetraction, DstRetraction>) {
      if (&dst_retraction == &src_retraction_cast ||
          dst_retraction == src_retraction_cast) {
        if constexpr (std::is_same_v<
                          JacobianMatrixWrtSrcRetraction,
                          JacobianMatrixWrtDstRetraction>) {
          if (&jacobian_under_src_retraction == jacobian_under_dst_retraction) {
            return true;
          }
        }
        *jacobian_under_dst_retraction = jacobian_under_src_retraction;
        return true;
      }
    }

    dst_retraction.transformJacobianImpl(
        jacobian_under_src_retraction, x, jacobian_under_dst_retraction,
        src_retraction_cast);
    return true;
  }
};

/// @brief  This is the interface class for all retractions on a specific
///         manifold. All retractions on this manifold should inherit
///         from this class.
///
/// @tparam _Manifold  The underlying manifold type.
template <typename _Manifold>
struct RetractionInterface_ : public RetractionInterface {
  static constexpr int kAmbientDim = manifold_traits<_Manifold>::kAmbientDim;

  virtual bool isProduct() const {
    return false;
  }

  virtual std::vector<const RetractionInterface*> getParts() const {
    return {this};
  }

 protected:
  bool checkManifoldType(const std::type_info& manifold_type) const override {
    // return manifold_type == typeid(_Manifold);
    if (UNLIKELY(manifold_type != typeid(_Manifold))) {
      LOGE(
          "Retraction checkManifoldType(): manifold type missmatch!! Expected "
          "Manifold = %s",
          classname<_Manifold>());
      return false;
    }
    return true;
  }
};

/// @brief  This is the CRTP bass class for all retractions on a specific
///         manifold. It facilitates the implementation of virtual interfaces
///         for derived classes, assuming they have provided some essential
///         template functions and type declarations.
///
/// @tparam _Manifold  The underlying manifold type.
template <typename Derived, typename _Manifold>
struct RetractionBase;

template <typename BaseRetraction, typename DerivedExtension>
struct RetractionExtension;

template <typename Derived, typename _Manifold>
class RetractionBase : public RetractionInterface_<_Manifold> {
  using RetractionInterface::DeclareTransformJacobianTypes;

 public:
  using Manifold = _Manifold;
  using RetractionInterface::dof;
  using RetractionInterface::transformJacobian;

  template <typename DerivedExtension>
  using Extension = RetractionExtension<Derived, DerivedExtension>;

  static constexpr int kDof =
      manifold_traits<Manifold>::kDim;  // The derived class may override this.

  static constexpr bool kIsProduct = false;

  const char* name() const override {
    return classname<Derived>();
  }

  static const Derived* defaultInstance() {
    return RetractionInterface::defaultInstance<Derived>();
  }

  bool operator==(const RetractionInterface& other) const override {
    auto other_cast = dynamic_cast<const Derived*>(&other);
    if (other_cast == nullptr) {
      return false;
    }
    return *derived() == *other_cast;
  }

  static_assert(
      std::is_same_v<typename manifold_traits<Manifold>::Scalar, double>,
      "Retraction: The scalar type of the manifold must be double.");

  template <typename Tangent, typename Manifold>
  Tangent section(const Manifold& x, const Manifold& x2) const {
    Tangent x2_minus_x;
    if constexpr (Tangent::RowsAtCompileTime == Eigen::Dynamic) {
      x2_minus_x.resize(dof(x), 1);
    } else {
      ASSERT(Tangent::RowsAtCompileTime == dof(x));
    }
    derived()->template sectionImpl<Tangent>(x, x2, &x2_minus_x);
    return x2_minus_x;
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2>
  Tangent section(
      const Manifold& x, const Manifold& x2, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtX2* jacobian_wrt_x2) const {
    Tangent x2_minus_x;
    if constexpr (Tangent::RowsAtCompileTime == Eigen::Dynamic) {
      x2_minus_x.resize(dof(x), 1);
    } else {
      ASSERT(Tangent::RowsAtCompileTime == dof(x));
    }
    derived()->template sectionImpl<Tangent>(
        x, x2, &x2_minus_x, jacobian_wrt_x, jacobian_wrt_x2);
    return x2_minus_x;
  }

  template <typename Manifold>
  auto section(const Manifold& x, const Manifold& x2) const {
    using DefaultTagent = Vector<Derived::kDof, typename Manifold::Scalar>;
    return section<DefaultTagent, Manifold>(x, x2);
  }

  template <typename Manifold, typename JacobianWrtX, typename JacobianWrtX2>
  auto section(
      const Manifold& x, const Manifold& x2, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtX2* jacobian_wrt_x2) const {
    using DefaultTagent = Vector<Derived::kDof, typename Manifold::Scalar>;
    return section<DefaultTagent, Manifold>(
        x, x2, jacobian_wrt_x, jacobian_wrt_x2);
  }

  /// @brief Runtime dof of the retraction. For fixed-sized manifolds,
  /// this is the same with the static dimension `kDof` and the default
  /// implementation of this method returns `kDof`. For dynamic-sized
  /// manifolds, this method must be overridden.
  int dimToDof(int manifold_dim) const override {
    if constexpr (manifold_traits<Manifold>::kDim != Eigen::Dynamic) {
      ASSERT(manifold_dim == manifold_traits<Manifold>::kDim);
    }
    if constexpr (Derived::kDof != Eigen::Dynamic) {
      ASSERT(Derived::kDof <= manifold_dim);
      return Derived::kDof;
    } else {
      throw std::runtime_error(
          "Derived retraction should override dimToDof() if it has dynamic "
          "dof!");
    }
  }

  template <
      typename SrcRetraction, typename JacobianMatrixWrtSrcRetraction,
      ENABLE_IF((!std::is_same_v<SrcRetraction, RetractionInterface>))>
  auto transformJacobian(
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      const Manifold& x,
      const SrcRetraction& src_retraction = SrcRetraction()) const {
    return this->template transformJacobian<Derived::kDof>(
        jacobian_under_src_retraction, x, src_retraction);
  }

  // If the derived class is instance dependent, it should override this.
  virtual bool operator==(const Derived& other) const {
    return true;
  }

  // The derived class should override this by declaring the retraction types
  // that it can handle, for example:
  //    using TransformJacobianTypes =
  //        DeclareTransformJacobianTypes<Retraction1, Retraction2>;
  using TransformJacobianTypes = DeclareTransformJacobianTypes<>;

  // Implemente transformJacobian() for each type in TransformJacobianTypes.
  template <
      typename SrcRetraction, typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      const Manifold& x,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      const SrcRetraction& src_retraction = SrcRetraction()) const {
    throw std::runtime_error("Not implemented");
  }

 protected:
  using RetractionInterface::AnyManifold;

  template <int RowsAtCompileTime, typename Scalar = double>
  Vector<RowsAtCompileTime, Scalar> createZeroTangent(const Manifold& x) const {
    Vector<RowsAtCompileTime, Scalar> tangent;
    if constexpr (RowsAtCompileTime == Eigen::Dynamic) {
      tangent.resize(dof(x), 1);
    } else {
      ASSERT(dof(x) == RowsAtCompileTime);
    }
    tangent.setZero();
  }

  template <int RowsAtCompileTime = Eigen::Dynamic, typename Scalar = double>
  Eigen::Map<Vector<RowsAtCompileTime, Scalar>> mutableTangentFromData(
      Scalar* data, const Manifold& x) const {
    if constexpr (RowsAtCompileTime == Eigen::Dynamic) {
      return Eigen::Map<Vector<RowsAtCompileTime, Scalar>>(data, dof(x), 1);
    } else {
      ASSERT(dof(x) == RowsAtCompileTime);
      return Eigen::Map<Vector<RowsAtCompileTime, Scalar>>(data);
    }
  }

  template <int RowsAtCompileTime = Eigen::Dynamic, typename Scalar = double>
  Eigen::Map<const Vector<RowsAtCompileTime, Scalar>> tangentFromData(
      const Scalar* data, const Manifold& x) const {
    if constexpr (RowsAtCompileTime == Eigen::Dynamic) {
      return Eigen::Map<const Vector<RowsAtCompileTime, Scalar>>(
          data, dof(x), 1);
    } else {
      ASSERT(dof(x) == RowsAtCompileTime);
      return Eigen::Map<const Vector<RowsAtCompileTime, Scalar>>(data);
    }
  }

  void operator()(const AnyManifold* x, const double* t, AnyManifold* x_plus_t)
      const override {
    const Manifold* x_cast = static_cast<const Manifold*>(x);
    Manifold* x_plus_t_cast = static_cast<Manifold*>(x_plus_t);
    *x_plus_t_cast =
        (*derived())(*x_cast, tangentFromData<Derived::kDof>(t, *x_cast));
  }

  void sectionInterface(
      const AnyManifold* x, const AnyManifold* x2,
      double* x2_minus_x) const override {
    const Manifold* x_cast = static_cast<const Manifold*>(x);
    const Manifold* x2_cast = static_cast<const Manifold*>(x2);
    mutableTangentFromData<Derived::kDof>(x2_minus_x, *x_cast) =
        section(*x_cast, *x2_cast);
  }

  void transformJacobianInterface(
      const AnyManifold* x, const RetractionInterface* src_retraction,
      const double* jacobian_under_src_retraction, int jacobian_under_src_rows,
      int jacobian_under_src_cols,
      double* jacobian_under_this_retraction) const override {
    derived()
        ->template transformJacobianInterfaceImpl<
            typename Derived::TransformJacobianTypes>(
            x, derived(), src_retraction, jacobian_under_src_retraction,
            jacobian_under_src_rows, jacobian_under_src_cols,
            jacobian_under_this_retraction);
  }

 public:
  template <typename TransformJacobianTypes, typename _DerivedRetraction>
  void transformJacobianInterfaceImpl(
      const AnyManifold* x, const _DerivedRetraction* this_retraction,
      const RetractionInterface* src_retraction,
      const double* jacobian_under_src_retraction, int jacobian_under_src_rows,
      int jacobian_under_src_cols,
      double* jacobian_under_this_retraction) const {
    const Manifold* x_cast = static_cast<const Manifold*>(x);
    RetractionInterface::ConstJacobianMap<double> J_src(
        jacobian_under_src_retraction, jacobian_under_src_rows,
        jacobian_under_src_cols);
    RetractionInterface::JacobianMap<double> J_this(
        jacobian_under_this_retraction, jacobian_under_src_rows, dof(*x_cast));
    // using TransformJacobianImpl = typename Derived::TransformJacobianTypes;
    using TransformJacobianImpl = TransformJacobianTypes;
    static const TransformJacobianImpl impl;
    bool handled =
        impl(*x_cast, *this_retraction, *src_retraction, J_src, &J_this);
    if (!handled) {
      throw std::runtime_error(
          std::string("transformJacobian() for Retraction '") +
          classname<Derived>() + "' against '" + classname(*src_retraction) +
          "' has not been implemented!");
    }
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

template <typename BaseRetraction, typename Derived>
struct RetractionExtension : public BaseRetraction {
  using BaseRetraction::BaseRetraction;

  static const Derived* defaultInstance() {
    return RetractionInterface::defaultInstance<Derived>();
  }

 protected:
  using RetractionInterface::AnyManifold;
  void transformJacobianInterface(
      const AnyManifold* x, const RetractionInterface* src_retraction,
      const double* jacobian_under_src_retraction, int jacobian_under_src_rows,
      int jacobian_under_src_cols,
      double* jacobian_under_this_retraction) const override {
    derived()
        ->template transformJacobianInterfaceImpl<
            typename Derived::TransformJacobianTypes>(
            x, derived(), src_retraction, jacobian_under_src_retraction,
            jacobian_under_src_rows, jacobian_under_src_cols,
            jacobian_under_this_retraction);
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

/// @brief  This retraction is used to fix a variable.
template <typename _Manifold>
struct FixedRetraction
    : public RetractionBase<FixedRetraction<_Manifold>, _Manifold> {
  using Manifold = _Manifold;
  static constexpr int kDof = 0;
  FixedRetraction() {}
  bool isFixed() const override {
    return true;
  }

  template <typename Tangent, typename Manifold>
  Manifold operator()(const Manifold& x, const Tangent& t) const {
    throw std::runtime_error("FixedRetraction::operator()() can't be called");
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtT>
  Manifold operator()(
      const Manifold& x, const Tangent& t, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtT* jacobian_wrt_t) const {
    throw std::runtime_error("FixedRetraction::operator()() can't be called");
  }

  template <typename Tangent, typename Manifold>
  void sectionImpl(const Manifold& x, const Manifold& x2, Tangent* t) const {
    throw std::runtime_error("FixedRetraction::sectionImpl() can't be called");
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2>
  void sectionImpl(
      const Manifold& x, const Manifold& x2, Tangent* t,
      JacobianWrtX* jacobian_wrt_x, JacobianWrtX2* jacobian_wrt_x2) const {
    throw std::runtime_error("FixedRetraction::sectionImpl() can't be called");
  }

  /// Declaring retraction types that can be used in transformJacobian().
  using RetractionInterface::DeclareTransformJacobianTypes;
  using TransformJacobianTypes = DeclareTransformJacobianTypes<>;

  /// Implemente transformJacobian() for each type in TransformJacobianTypes.
  template <
      typename SrcRetraction, typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      const Manifold& x,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      const SrcRetraction& src_retraction = SrcRetraction()) const {
    throw std::runtime_error(
        "FixedRetraction::transformJacobianImpl() can't be called");
  }
};

/// @brief  This is the default Retraction for an n dimensional vector space
///         and it serves as an example for how to implement a custom
///         retraction.
template <int n = Eigen::Dynamic, typename Scalar = double>
class VectorSpaceRetraction
    : public RetractionBase<
          VectorSpaceRetraction<n, Scalar>, Vector<n, Scalar>> {
  using _RetractionInterface =
      RetractionBase<VectorSpaceRetraction<n, Scalar>, Vector<n, Scalar>>;
  using RetractionInterface::DeclareTransformJacobianTypes;

 public:
  using _RetractionInterface::kDof;
  using _RetractionInterface::section;
  using _RetractionInterface::transformJacobian;
  using typename _RetractionInterface::Manifold;
  static_assert(kDof == manifold_traits<Manifold>::kDim);

  template <typename Tangent, typename Manifold>
  Manifold operator()(const Manifold& x, const Tangent& t) const {
    return x + t;
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtT>
  Manifold operator()(
      const Manifold& x, const Tangent& t, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtT* jacobian_wrt_t) const {
    if (jacobian_wrt_x) {
      jacobian_wrt_x->setIdentity();
    }
    if (jacobian_wrt_t) {
      jacobian_wrt_t->setIdentity();
    }
    return x + t;
  }

  template <typename Tangent, typename Manifold>
  void sectionImpl(const Manifold& x, const Manifold& x2, Tangent* t) const {
    *t = x2 - x;
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2>
  void sectionImpl(
      const Manifold& x, const Manifold& x2, Tangent* t,
      JacobianWrtX* jacobian_wrt_x, JacobianWrtX2* jacobian_wrt_x2) const {
    if (jacobian_wrt_x) {
      jacobian_wrt_x->setIdentity();
      *jacobian_wrt_x = -(*jacobian_wrt_x);
    }
    if (jacobian_wrt_x2) {
      jacobian_wrt_x2->setIdentity();
    }
    *t = x2 - x;
  }

  int dimToDof(int manifold_dim) const override {
    if constexpr (kDof != Eigen::Dynamic) {
      ASSERT(manifold_dim == kDof);
      return kDof;
    } else {
      return manifold_dim;
    }
  }

  /// Declaring retraction types that can be used in transformJacobian().
  ///
  /// Here we leave the types empty, which means we can't transform
  /// Jacobians under other retractions to that of this retraction.
  ///
  /// If your custom retraction supports transformJacobian() for other
  /// retractions, you should declare the suppported retraction types
  /// like:
  /// @code
  /// using TransformJacobianTypes = DeclareTransformJacobianTypes<
  ///                                    RetractionType1,
  ///                                    RetractionType2>;
  /// @endcode
  using TransformJacobianTypes = DeclareTransformJacobianTypes<>;

  /// Implemente transformJacobian() for each type in TransformJacobianTypes.
  template <
      typename SrcRetraction, typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      const Manifold& x,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      const SrcRetraction& src_retraction = SrcRetraction()) const {
    throw std::runtime_error("Not implemented");
  }
};

/// @brief  The direct product of multiple retractions.
/// @tparam ...RetractionParts
///
/// @note Note @c RetractionParts shouldn't contain a part
///       with dynamic dimension!
template <
    typename Derived, typename ProductManifold, typename... RetractionParts>
struct ProductRetractionBase;

template <typename ProductManifold, typename... RetractionParts>
class ProductRetraction
    : public ProductRetractionBase<
          ProductRetraction<ProductManifold, RetractionParts...>,
          ProductManifold, RetractionParts...> {
  using _Base = ProductRetractionBase<
      ProductRetraction<ProductManifold, RetractionParts...>, ProductManifold,
      RetractionParts...>;

 public:
  using _Base::_Base;
};

/// @brief  A SubSpaceRetraction is the resriction of the @c BaseRetraction
///         to a subspace of the tangent space.
///         The @c SubSpaceType can be either a specialization of
///         SubSpaceByMatrix or SubSpaceByAxes (or other user defined
///         types that has implemented the necessary interfaces).
template <typename Derived, typename BaseRetraction, typename SubSpaceType>
struct SubSpaceRetractionBase;

template <typename BaseRetraction, typename SubSpaceType>
class SubSpaceRetraction : public SubSpaceRetractionBase<
                               SubSpaceRetraction<BaseRetraction, SubSpaceType>,
                               BaseRetraction, SubSpaceType> {
  using _Base = SubSpaceRetractionBase<
      SubSpaceRetraction<BaseRetraction, SubSpaceType>, BaseRetraction,
      SubSpaceType>;

 public:
  using _Base::_Base;
};

template <typename SubSpaceType, int _base_dim>
using VectorSpaceSubRetraction =
    SubSpaceRetraction<VectorSpaceRetraction<_base_dim>, SubSpaceType>;

/// @brief  A SubSpaceByMatrix is a subspace of the tangent space
///         defined by a matrix.
template <int _sub_space_dim>
struct SubSpaceByMatrix;

template <int _base_dim, int _sub_space_dim>
using SubSpaceByMatrixMap =
    typename SubSpaceByMatrix<_sub_space_dim>::template Map<_base_dim>;

/// @brief  A SubSpaceByAxes is a subspace of the tangent space
///         defined by a list of axes.
///         For example, SubSpaceByAxes<0, 1, 2> is the 3-dimensional
///         subspace spanned by the first three axes.
template <size_t... _sub_space_axes>
struct SubSpaceByAxes;

template <int _base_dim, size_t... _sub_space_axes>
using SubSpaceByAxesMap =
    typename SubSpaceByAxes<_sub_space_axes...>::template Map<_base_dim>;

/// @brief  Optimizable manifold.
template <
    typename _Manifold,
    typename _Retraction =
        VectorSpaceRetraction<manifold_traits<_Manifold>::kDim>,
    bool _share_retraction = true>
struct OptimizableManifold {
  using Scalar = typename manifold_traits<_Manifold>::Scalar;
  using Value = _Manifold;
  using Retraction = _Retraction;

  // Ensure Retraction implements RetractionInterface.
  static_assert(std::is_base_of_v<RetractionInterface, Retraction>);
  static_assert(!std::is_same_v<RetractionInterface, Retraction>);

  static constexpr bool kShareRetraction = _share_retraction;
  static constexpr int kDof = Retraction::kDof;
  static constexpr int kAmbientDim = Retraction::kAmbientDim;

  OptimizableManifold() : value_(Value()) {}
  OptimizableManifold(const Value& v) : value_(v) {}  // NOLINT

  template <
      typename __Retraction = Retraction,
      ENABLE_IF((std::is_base_of_v<Retraction, __Retraction>))>
  static OptimizableManifold Create() {
    return OptimizableManifold();
  }

  template <
      typename __Retraction = Retraction,
      ENABLE_IF((std::is_base_of_v<Retraction, __Retraction>))>
  static OptimizableManifold Create(const Value& v) {
    return OptimizableManifold(v);
  }

  Value& value() {
    return value_;
  }
  const Value& value() const {
    return value_;
  }
  void setValue(const Value& v) {
    value_ = v;
  }
  static const Retraction& retraction() {
    return *retraction_;
  }
  int dof() const {
    return retraction().dof(value_);
  }

  template <typename Tangent>
  OptimizableManifold operator+(const Tangent& t) const {
    return (*retraction_)(value_, t);
  }

  template <typename Tangent = Vector<kDof, Scalar>>
  Tangent operator-(const OptimizableManifold& x2) const {
    ASSERT(retraction_ == x2.retraction_ || *retraction_ == *x2.retraction_);
    return retraction_->template section<Tangent>(x2.value_, value_);
  }

  template <typename Tangent = Vector<kDof, Scalar>>
  Tangent operator-(const Value& x2) const {
    return retraction_->template section<Tangent>(x2, value_);
  }

  operator Value() const {
    return value_;
  }

 private:
  Value value_;

  // All instances of OptimizableManifold share the same
  // retraction object.
  static inline const Retraction* retraction_ =
      RetractionInterface::defaultInstance<Retraction>();
};

template <typename _Manifold, typename _Retraction>
struct OptimizableManifold<_Manifold, _Retraction, false> {
  using Scalar = typename manifold_traits<_Manifold>::Scalar;
  using Value = _Manifold;
  using Retraction = _Retraction;
  static constexpr bool kShareRetraction = false;
  static constexpr int kDof = Retraction::kDof;
  static constexpr int kAmbientDim = Retraction::kAmbientDim;

  /// Note that the member retraction_ is just a reference to the
  /// @p extern_retraction passed to the constructor. The extern
  /// retraction object should have a lifetime longer than the
  /// OptimizableManifold object.
  explicit OptimizableManifold(const Retraction* extern_retraction = nullptr)
      : value_(),
        retraction_(
            extern_retraction
                ? extern_retraction
                : RetractionInterface::defaultInstance<Retraction>()) {
    if (!retraction_) {
      LOGW(
          "OptimizableManifold Warning: retraction is not initialized!! "
          "_Manifold = '%s', _Retraction = '%s'",
          classname<_Manifold>(), classname<_Retraction>());
    }
  }

  OptimizableManifold(
      const Value& v,
      const Retraction* extern_retraction = nullptr)  // NOLINT
      : value_(v),
        retraction_(
            extern_retraction
                ? extern_retraction
                : RetractionInterface::defaultInstance<Retraction>()) {
    if (!retraction_) {
      LOGW(
          "OptimizableManifold Warning: retraction is not initialized!! "
          "_Manifold = '%s', _Retraction = '%s'",
          classname<_Manifold>(), classname<_Retraction>());
    }
  }

  template <
      typename __Retraction = Retraction,
      ENABLE_IF((std::is_base_of_v<Retraction, __Retraction>))>
  static OptimizableManifold Create(
      const __Retraction* extern_retraction = nullptr) {
    return OptimizableManifold(
        extern_retraction
            ? extern_retraction
            : RetractionInterface::defaultInstance<__Retraction>());
  }

  template <
      typename __Retraction = Retraction,
      ENABLE_IF((std::is_base_of_v<Retraction, __Retraction>))>
  static OptimizableManifold Create(
      const Value& v, const __Retraction* extern_retraction = nullptr) {
    return OptimizableManifold(
        v, extern_retraction
               ? extern_retraction
               : RetractionInterface::defaultInstance<__Retraction>());
  }

  Value& value() {
    return value_;
  }
  const Value& value() const {
    return value_;
  }
  void setValue(const Value& v) {
    value_ = v;
  }
  const Retraction& retraction() const {
    return *retraction_;
  }
  template <
      typename __Retraction = Retraction,
      ENABLE_IF((std::is_base_of_v<Retraction, __Retraction>))>
  void setRetraction(const __Retraction* extern_retraction = nullptr) {
    retraction_ = extern_retraction
                      ? extern_retraction
                      : RetractionInterface::defaultInstance<__Retraction>();
  }
  int dof() const {
    return retraction().dof(value_);
  }

  template <typename Tangent>
  OptimizableManifold operator+(const Tangent& t) const {
    return OptimizableManifold((*retraction_)(value_, t), retraction_);
  }

  template <typename Tangent = Vector<kDof, Scalar>>
  Tangent operator-(const OptimizableManifold& x2) const {
    ASSERT(retraction_ == x2.retraction_ || *retraction_ == *x2.retraction_);
    return retraction_->template section<Tangent>(x2.value_, value_);
  }

  template <typename Tangent = Vector<kDof, Scalar>>
  Tangent operator-(const Value& x2) const {
    return retraction_->template section<Tangent>(x2, value_);
  }

  operator Value() const {
    return value_;
  }

 private:
  Value value_;
  const Retraction* retraction_;
};

/// The flexible version of OptimizableManifold, whose retraction type can
/// be changed at runtime.
template <typename Manifold>
using XOptimizableManifold =
    OptimizableManifold<Manifold, RetractionInterface, false>;

/// Optimizable vector
template <
    int n = Eigen::Dynamic, typename Retraction = VectorSpaceRetraction<n>>
using OptimizableVector = OptimizableManifold<Vector<n>, Retraction>;

template <int n = Eigen::Dynamic>
using XOptimizableVector = OptimizableVector<n, RetractionInterface>;

////// Implmentation of SubSpaceRetraction //////

template <typename Derived, typename BaseRetraction, typename SubSpaceType>
class SubSpaceRetractionBase
    : public RetractionBase<Derived, typename BaseRetraction::Manifold> {
  using _RetractionInterface =
      RetractionBase<Derived, typename BaseRetraction::Manifold>;
  using typename _RetractionInterface::AnyManifold;

 public:
  using _RetractionInterface::dof;
  using _RetractionInterface::section;
  using typename _RetractionInterface::Manifold;
  using SubSpaceMap = typename SubSpaceType::template Map<BaseRetraction::kDof>;

  static constexpr int kDof = SubSpaceMap::kSubSpaceDim;
  static constexpr int kAmbientDim = BaseRetraction::kAmbientDim;
  static_assert(
      BaseRetraction::kDof == Eigen::Dynamic || kDof <= BaseRetraction::kDof,
      "Subspace dimension must be lessthan or equal to the base retraction "
      "dimension");

  SubSpaceRetractionBase(
      const BaseRetraction& base_retraction = BaseRetraction(),
      const SubSpaceMap& subspace_map = SubSpaceMap())
      : base_retraction_(base_retraction), subspace_map_(subspace_map) {}

  void setBaseRetraction(const BaseRetraction& retraction) {
    base_retraction_ = retraction;
  }
  void setSubSpaceMap(const SubSpaceMap& map) {
    subspace_map_ = map;
  }
  int dimToDof(int manifold_dim) const override {
    if constexpr (manifold_traits<Manifold>::kDim != Eigen::Dynamic) {
      ASSERT(manifold_dim == manifold_traits<Manifold>::kDim);
    }
    if constexpr (kDof != Eigen::Dynamic) {
      ASSERT(kDof <= manifold_dim);
      return kDof;
    } else {
      int dof = subspace_map_.subSpaceDim();
      ASSERT(dof <= manifold_dim);
      return dof;
    }
  }

  bool operator==(const Derived& other) const override {
    if (derived() == &other) {
      return true;
    } else {
      return base_retraction_ == other.base_retraction_ &&
             subspace_map_ == other.subspace_map_;
    }
  }

  template <typename Tangent, typename Manifold>
  Manifold operator()(const Manifold& x, const Tangent& t) const {
    return base_retraction_(x, subspace_map_(t));
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtT>
  Manifold operator()(
      const Manifold& x, const Tangent& t, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtT* jacobian_wrt_t) const {
    throw std::runtime_error(
        "SubSpaceRetractionBase::operator(): Subspace retraction does not "
        "support Jacobian computation");
  }

  template <typename Tangent, typename Manifold>
  void sectionImpl(const Manifold& x, const Manifold& x2, Tangent* t) const {
    using BaseTangent = Vector<BaseRetraction::kDof, typename Tangent::Scalar>;
    BaseTangent base_tangent =
        base_retraction_.template section<BaseTangent, Manifold>(x, x2);
    *t = subspace_map_.toSub(base_tangent);
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2>
  void sectionImpl(
      const Manifold& x, const Manifold& x2, Tangent* t,
      JacobianWrtX* jacobian_wrt_x, JacobianWrtX2* jacobian_wrt_x2) const {
    throw std::runtime_error(
        "SubSpaceRetractionBase::sectionImpl(): Subspace retraction does not "
        "support Jacobian computation");
  }

  template <typename TransformJacobianTypes, typename _DerivedRetraction>
  void transformJacobianInterfaceImpl(
      const AnyManifold* x, const _DerivedRetraction* this_retraction,
      const RetractionInterface* src_retraction,
      const double* jacobian_under_src_retraction, int jacobian_under_src_rows,
      int jacobian_under_src_cols,
      double* jacobian_under_this_retraction) const {
    const Manifold* x_cast = static_cast<const Manifold*>(x);
    RetractionInterface::ConstJacobianMap<double> J_src(
        jacobian_under_src_retraction, jacobian_under_src_rows,
        jacobian_under_src_cols);
    RetractionInterface::JacobianMap<double> J_this(
        jacobian_under_this_retraction, jacobian_under_src_rows, dof(*x_cast));

    using TransformJacobianImpl = TransformJacobianTypes;
    static const TransformJacobianImpl impl;
    bool handled = impl(*x_cast, *derived(), *src_retraction, J_src, &J_this);
    if (!handled) {
      // Let the base retraction handle the Jacobian transformation.
      auto jacobian_wrt_base =
          base_retraction_.transformJacobian(J_src, *x_cast, *src_retraction);
      J_this = subspace_map_.transformJacobianWrtBase(jacobian_wrt_base);
    }
  }

 protected:
  BaseRetraction base_retraction_;
  SubSpaceMap subspace_map_;

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

template <int _sub_space_dim>
struct SubSpaceByMatrix {
  template <int _base_space_dim = Eigen::Dynamic>
  struct Map {
    static constexpr int kSubSpaceDim = _sub_space_dim;
    static constexpr int kBaseSpaceDim = _base_space_dim;
    static_assert(
        kBaseSpaceDim == Eigen::Dynamic || kSubSpaceDim <= kBaseSpaceDim,
        "Subspace dimension must be less than or equal to the base space");

    template <typename Scalar>
    using TransformMatrix = RetractionInterface::JacobianMatrix<
        Scalar, kBaseSpaceDim, kSubSpaceDim>;
    using TransformMatrixd = TransformMatrix<double>;

    Map(const TransformMatrixd& transform_matrix =  // NOLINT
        TransformMatrixd())
        : transform_matrix_(transform_matrix) {
      if (transform_matrix_.rows() > 0) {
        qr_ = transform_matrix_.householderQr();
      }
    }

    void setTransformMatrix(const TransformMatrixd& transform_matrix) {
      transform_matrix_ = transform_matrix;
      if (transform_matrix_.rows() > 0) {
        qr_ = transform_matrix_.householderQr();
      }
    }

    int subSpaceDim() const {
      return transform_matrix_.cols();
    }

    int baseSpaceDim() const {
      return transform_matrix_.rows();
    }

    bool operator==(const Map& other) const {
      if (this == &other) {
        return true;
      } else {
        return (transform_matrix_ - other.transform_matrix_).isZero();
      }
    }

    template <typename SubSpaceVector>
    auto operator()(const SubSpaceVector& x) const {
      if constexpr (std::is_same_v<typename SubSpaceVector::Scalar, double>) {
        return transform_matrix_ * x;
      } else {
        return transform_matrix_
                   .template cast<typename SubSpaceVector::Scalar>() *
               x;
      }
    }

    template <typename BaseSpaceVector>
    auto toSub(const BaseSpaceVector& x_base) const {
      if constexpr (std::is_same_v<typename BaseSpaceVector::Scalar, double>) {
        return qr_.solve(x_base);
      } else {
        return qr_.template cast<typename BaseSpaceVector::Scalar>().solve(
            x_base);
      }
    }

    template <typename Scalar>
    TransformMatrix<Scalar> getTransformMatrix() const {
      return transform_matrix_.template cast<Scalar>();
    }

    template <typename JacobianMatrixWrtBase>
    RetractionInterface::JacobianMatrix<
        typename JacobianMatrixWrtBase::Scalar,
        JacobianMatrixWrtBase::RowsAtCompileTime, kSubSpaceDim>
    transformJacobianWrtBase(
        const JacobianMatrixWrtBase& jacobian_wrt_base) const {
      using Scalar = typename JacobianMatrixWrtBase::Scalar;
      using JacobianMatrixWrtSubspace = RetractionInterface::JacobianMatrix<
          Scalar, JacobianMatrixWrtBase::RowsAtCompileTime, kSubSpaceDim>;
      JacobianMatrixWrtSubspace jacobian_wrt_subspace;
      ASSERT(jacobian_wrt_base.cols() == transform_matrix_.rows());
      if constexpr (std::is_same<Scalar, double>::value) {
        jacobian_wrt_subspace = jacobian_wrt_base * transform_matrix_;
      } else {
        jacobian_wrt_subspace =
            jacobian_wrt_base * transform_matrix_.template cast<Scalar>();
      }
      return jacobian_wrt_subspace;
    }

   private:
    TransformMatrixd transform_matrix_;
    Eigen::HouseholderQR<TransformMatrixd> qr_;
  };
};

template <size_t... _sub_space_axes>
struct SubSpaceByAxes {
  template <int _base_space_dim = Eigen::Dynamic>
  struct Map {
    static constexpr int kSubSpaceDim = sizeof...(_sub_space_axes);
    static constexpr int kBaseSpaceDim = _base_space_dim;
    static_assert(
        kBaseSpaceDim == Eigen::Dynamic || kSubSpaceDim <= kBaseSpaceDim,
        "Subspace dimension must be less than or equal to the base space");
    template <typename Scalar>
    using TransformMatrix = RetractionInterface::JacobianMatrix<
        Scalar, kBaseSpaceDim, kSubSpaceDim>;
    using TransformMatrixd = TransformMatrix<double>;

    Map(int base_space_dim = kBaseSpaceDim)  // NOLINT
        : base_space_dim_(base_space_dim) {
      // Base space dimension must be equal to the specified dimension
      ASSERT(
          kBaseSpaceDim == Eigen::Dynamic || base_space_dim_ == kBaseSpaceDim);
    }

    void setBaseSpaceDim(int base_space_dim) {
      if constexpr (kBaseSpaceDim == Eigen::Dynamic) {
        base_space_dim_ = base_space_dim;
      } else {
        // Base space dimension must be equal to the specified dimension
        ASSERT(base_space_dim == kBaseSpaceDim);
      }
    }

    int subSpaceDim() const {
      return kSubSpaceDim;
    }

    int baseSpaceDim() const {
      return base_space_dim_;
    }

    bool operator==(const Map& other) const {
      return base_space_dim_ == other.base_space_dim_;
    }

    template <typename SubSpaceVector>
    auto operator()(const SubSpaceVector& x) const {
      return operator()(x, std::make_index_sequence<kSubSpaceDim>());
    }

    template <typename BaseSpaceVector, std::size_t... Is>
    auto toSub(const BaseSpaceVector& x_base) const {
      return toSub(x_base, std::make_index_sequence<kSubSpaceDim>());
    }

    template <typename Scalar>
    TransformMatrix<Scalar> getTransformMatrix() const {
      return getTransformMatrix<Scalar>(
          std::make_index_sequence<kSubSpaceDim>());
    }

    template <typename JacobianMatrixWrtBase>
    RetractionInterface::JacobianMatrix<
        typename JacobianMatrixWrtBase::Scalar,
        JacobianMatrixWrtBase::RowsAtCompileTime, kSubSpaceDim>
    transformJacobianWrtBase(
        const JacobianMatrixWrtBase& jacobian_wrt_base) const {
      return transformJacobianWrtBase(
          jacobian_wrt_base, std::make_index_sequence<kSubSpaceDim>());
    }

   private:
    static constexpr size_t kAxes[] = {_sub_space_axes...};

    template <typename SubSpaceVector, std::size_t... Is>
    auto operator()(const SubSpaceVector& x, std::index_sequence<Is...>) const {
      Vector<kBaseSpaceDim, typename SubSpaceVector::Scalar> result;
      if constexpr (kBaseSpaceDim == Eigen::Dynamic) {
        result.resize(base_space_dim_);
      }
      result.setZero();
      ((result(_sub_space_axes) = x(Is)), ...);
      return result;
    }

    template <typename BaseSpaceVector, std::size_t... Is>
    auto toSub(
        const BaseSpaceVector& x_base, std::index_sequence<Is...>) const {
      Vector<kSubSpaceDim, typename BaseSpaceVector::Scalar> result;
      ((result(Is) = x_base(_sub_space_axes)), ...);
      // for (size_t i = 0; i < kSubSpaceDim; ++i) {
      //   result(i) = x_base(kAxes[i]);
      // }
      return result;
    }

    template <typename Scalar, std::size_t... Is>
    TransformMatrix<Scalar> getTransformMatrix(
        std::index_sequence<Is...>) const {
      TransformMatrix<Scalar> result;
      if constexpr (kBaseSpaceDim == Eigen::Dynamic) {
        result.resize(base_space_dim_, kSubSpaceDim);
      }
      result.setZero();
      ((result(_sub_space_axes, Is) = Scalar(1.)), ...);
      return result;
    }

    template <typename JacobianMatrixWrtBase, std::size_t... Is>
    RetractionInterface::JacobianMatrix<
        typename JacobianMatrixWrtBase::Scalar,
        JacobianMatrixWrtBase::RowsAtCompileTime, kSubSpaceDim>
    transformJacobianWrtBase(
        const JacobianMatrixWrtBase& jacobian_wrt_base,
        std::index_sequence<Is...>) const {
      using Scalar = typename JacobianMatrixWrtBase::Scalar;
      using JacobianMatrixWrtSubspace = RetractionInterface::JacobianMatrix<
          Scalar, JacobianMatrixWrtBase::RowsAtCompileTime, kSubSpaceDim>;
      ASSERT(jacobian_wrt_base.cols() == base_space_dim_);
      JacobianMatrixWrtSubspace jacobian_wrt_subspace;
      if constexpr (
          JacobianMatrixWrtBase::RowsAtCompileTime == Eigen::Dynamic) {
        jacobian_wrt_subspace.resize(jacobian_wrt_base.rows(), kSubSpaceDim);
      }
      // jacobian_wrt_subspace.setZero();
      ((jacobian_wrt_subspace.col(Is) = jacobian_wrt_base.col(_sub_space_axes)),
       ...);
      return jacobian_wrt_subspace;
    }

    int base_space_dim_ = kBaseSpaceDim;
  };
};

////// Implmentation of ProductRetraction //////

template <
    typename Derived, typename ProductManifold, typename... RetractionParts>
class ProductRetractionBase : public RetractionBase<Derived, ProductManifold> {
  static_assert(
      ((RetractionParts::kDof != Eigen::Dynamic) && ...),
      "ProductRetraction doesn't support dynamic dimension so far! "
      "RetractionParts shouldn't contain a part with dynamic dimension!");
  using _RetractionInterface = RetractionBase<Derived, ProductManifold>;
  using typename _RetractionInterface::AnyManifold;

 public:
  using _RetractionInterface::dof;
  using _RetractionInterface::section;
  using typename _RetractionInterface::Manifold;
  static constexpr bool kIsProduct = true;

  static constexpr int kDof = (RetractionParts::kDof + ...);
  static constexpr int kAmbientDim = (RetractionParts::kAmbientDim + ...);
  // static constexpr int kNumRetractionParts = sizeof...(RetractionParts);

  ProductRetractionBase() = default;

  template <typename _First, typename _Second, typename... _Rest>
  ProductRetractionBase(_First&& first, _Second&& second, _Rest&&... rest)
      : parts_(
            std::forward<_First>(first), std::forward<_Second>(second),
            (std::forward<_Rest>(rest))...) {}

  bool isProduct() const override {
    return true;
  }

  std::vector<const RetractionInterface*> getParts() const override {
    return getParts(std::index_sequence_for<RetractionParts...>{});
  }

  template <std::size_t _part_idx>
  auto& part() {
    return std::get<_part_idx>(parts_);
  }

  template <std::size_t _part_idx>
  const auto& part() const {
    return std::get<_part_idx>(parts_);
  }

  int dimToDof(int manifold_dim) const override {
    // Currently we don't support dynamic dimension for
    // ProductRetractionBase, so we just use the
    // default implementation.
    return _RetractionInterface::dimToDof(manifold_dim);
  }

  bool operator==(const Derived& other) const override {
    if (derived() == &other) {
      return true;
    } else {
      return _equals_impl(other, std::index_sequence_for<RetractionParts...>{});
    }
  }

  template <typename Tangent, typename Manifold>
  Manifold operator()(const Manifold& x, const Tangent& t) const {
    // return _retraction_impl(x, t,
    // std::make_index_sequence<kNumRetractionParts>());
    return _retraction_impl(
        x, t, std::index_sequence_for<RetractionParts...>{});
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtT>
  Manifold operator()(
      const Manifold& x, const Tangent& t, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtT* jacobian_wrt_t) const {
    // return _retraction_impl(x, t,
    // std::make_index_sequence<kNumRetractionParts>());
    return _retraction_impl(
        x, t, jacobian_wrt_x, jacobian_wrt_t,
        std::index_sequence_for<RetractionParts...>{});
  }

  template <typename Tangent, typename Manifold>
  void sectionImpl(const Manifold& x, const Manifold& x2, Tangent* t) const {
    _section_impl<Tangent>(
        x, x2, t, std::index_sequence_for<RetractionParts...>{});
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2>
  void sectionImpl(
      const Manifold& x, const Manifold& x2, Tangent* t,
      JacobianWrtX* jacobian_wrt_x, JacobianWrtX2* jacobian_wrt_x2) const {
    _section_impl<Tangent>(
        x, x2, t, jacobian_wrt_x, jacobian_wrt_x2,
        std::index_sequence_for<RetractionParts...>{});
  }

  template <typename TransformJacobianTypes, typename _DerivedRetraction>
  void transformJacobianInterfaceImpl(
      const AnyManifold* x, const _DerivedRetraction* this_retraction,
      const RetractionInterface* src_retraction,
      const double* jacobian_under_src_retraction, int jacobian_under_src_rows,
      int jacobian_under_src_cols,
      double* jacobian_under_this_retraction) const {
    const Manifold* x_cast = static_cast<const Manifold*>(x);
    RetractionInterface::ConstJacobianMap<double> J_src(
        jacobian_under_src_retraction, jacobian_under_src_rows,
        jacobian_under_src_cols);
    RetractionInterface::JacobianMap<double> J_this(
        jacobian_under_this_retraction, jacobian_under_src_rows, dof(*x_cast));

    using TransformJacobianImpl = TransformJacobianTypes;
    static const TransformJacobianImpl impl;
    bool handled = impl(*x_cast, *derived(), *src_retraction, J_src, &J_this);
    if (!handled) {
      // Handle the Jacobian transformation by individual parts
      transformJacobianAgainstOtherProduct(
          *x_cast, *src_retraction, J_src, &J_this);
    }
  }

 protected:
  template <typename Product, int _part_idx>
  using Part =
      RawType<decltype(std::declval<Product>().template part<_part_idx>())>;

  template <typename Scalar>
  using PartitionedTangentT =
      PartitionedVector<Scalar, RetractionParts::kDof...>;

  template <std::size_t... Is>
  std::vector<const RetractionInterface*> getParts(
      std::index_sequence<Is...>) const {
    return {(&derived()->template part<Is>())...};
  }

  template <std::size_t... Is>
  bool _equals_impl(const Derived& other, std::index_sequence<Is...>) const {
    return (
        (other.template part<Is>() == derived()->template part<Is>()) && ...);
  }

  template <typename Tangent, typename Manifold, std::size_t... Is>
  Manifold _retraction_impl(
      const Manifold& x, const Tangent& t, std::index_sequence<Is...>) const {
    using PartitionedTangent = PartitionedTangentT<typename Manifold::Scalar>;
    const Vector<kDof, typename Manifold::Scalar>& vt = t;
    const PartitionedTangent& pt = static_cast<const PartitionedTangent&>(vt);
    return Manifold(
        (part<Is>()(x.template part<Is>(), pt.template part<Is>()))...);
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtT, std::size_t... Is>
  Manifold _retraction_impl(
      const Manifold& x, const Tangent& t, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtT* jacobian_wrt_t, std::index_sequence<Is...>) const {
    if (jacobian_wrt_x) {
      jacobian_wrt_x->setZero();
    }
    if (jacobian_wrt_t) {
      jacobian_wrt_t->setZero();
    }
    int start_row_idx = 0;
    Manifold ret;
    (_part_retraction_impl<Is>(
         &start_row_idx, x, t, &ret, jacobian_wrt_x, jacobian_wrt_t),
     ...);
    return ret;
  }

  template <
      size_t I, typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtT>
  void _part_retraction_impl(
      int* start_row_idx, const Manifold& x, const Tangent& t,
      Manifold* retracted_x, JacobianWrtX* jacobian_wrt_x,
      JacobianWrtT* jacobian_wrt_t) const {
    int part_dof = this->template part<I>().dof(x.template part<I>());
    using JacobianBlockTypeWrtX = decltype(jacobian_wrt_x->block(
        *start_row_idx, *start_row_idx, part_dof, part_dof));
    using JacobianBlockTypeWrtT = decltype(jacobian_wrt_t->block(
        *start_row_idx, *start_row_idx, part_dof, part_dof));

    JacobianBlockTypeWrtX* jacobian_wrt_x_part =
        jacobian_wrt_x
            ? new JacobianBlockTypeWrtX(jacobian_wrt_x->block(
                  *start_row_idx, *start_row_idx, part_dof, part_dof))
            : nullptr;
    JacobianBlockTypeWrtT* jacobian_wrt_t_part =
        jacobian_wrt_t
            ? new JacobianBlockTypeWrtT(jacobian_wrt_t->block(
                  *start_row_idx, *start_row_idx, part_dof, part_dof))
            : nullptr;
    auto t_part = t.block(*start_row_idx, 0, part_dof, 1);
    retracted_x->template part<I>() = this->template part<I>()(
        x.template part<I>(), t_part, jacobian_wrt_x_part, jacobian_wrt_t_part);
    if (jacobian_wrt_x_part) {
      delete jacobian_wrt_x_part;
    }
    if (jacobian_wrt_t_part) {
      delete jacobian_wrt_t_part;
    }
    *start_row_idx += part_dof;
  }

  template <typename Tangent, typename Manifold, std::size_t... Is>
  void _section_impl(
      const Manifold& x, const Manifold& x2, Tangent* t,
      std::index_sequence<Is...>) const {
    using PartitionedTangent = PartitionedTangentT<typename Manifold::Scalar>;
    PartitionedTangent& pt = static_cast<PartitionedTangent&>(*t);
    ((pt.template part<Is>() =
          part<Is>()
              .template section<
                  Vector<RetractionParts::kDof, typename Manifold::Scalar>,
                  decltype(x.template part<Is>())>(
                  x.template part<Is>(), x2.template part<Is>())),
     ...);
  }

  template <
      typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2, std::size_t... Is>
  void _section_impl(
      const Manifold& x, const Manifold& x2, Tangent* t,
      JacobianWrtX* jacobian_wrt_x, JacobianWrtX2* jacobian_wrt_x2,
      std::index_sequence<Is...>) const {
    int start_row_idx = 0;
    if (jacobian_wrt_x) {
      jacobian_wrt_x->setZero();
    }
    if (jacobian_wrt_x2) {
      jacobian_wrt_x2->setZero();
    }
    (_part_section_impl<Is>(
         &start_row_idx, x, x2, t, jacobian_wrt_x, jacobian_wrt_x2,
         std::index_sequence<Is...>()),
     ...);
  }

  template <
      size_t I, typename Tangent, typename Manifold, typename JacobianWrtX,
      typename JacobianWrtX2, std::size_t... Is>
  void _part_section_impl(
      int* start_row_idx, const Manifold& x, const Manifold& x2, Tangent* t,
      JacobianWrtX* jacobian_wrt_x, JacobianWrtX2* jacobian_wrt_x2,
      std::index_sequence<Is...>) const {
    int part_dof = this->template part<I>().dof(x.template part<I>());
    using JacobianBlockTypeWrtX = decltype(jacobian_wrt_x->block(
        *start_row_idx, *start_row_idx, part_dof, part_dof));
    using JacobianBlockTypeWrtX2 = decltype(jacobian_wrt_x2->block(
        *start_row_idx, *start_row_idx, part_dof, part_dof));

    JacobianBlockTypeWrtX* jacobian_wrt_x_part =
        jacobian_wrt_x
            ? new JacobianBlockTypeWrtX(jacobian_wrt_x->block(
                  *start_row_idx, *start_row_idx, part_dof, part_dof))
            : nullptr;
    JacobianBlockTypeWrtX2* jacobian_wrt_x2_part =
        jacobian_wrt_x2
            ? new JacobianBlockTypeWrtX2(jacobian_wrt_x2->block(
                  *start_row_idx, *start_row_idx, part_dof, part_dof))
            : nullptr;

    auto t_part = t->block(*start_row_idx, 0, part_dof, 1);
    t_part = this->template part<I>().section(
        x.template part<I>(), x2.template part<I>(), jacobian_wrt_x_part,
        jacobian_wrt_x2_part);
    if (jacobian_wrt_x_part) {
      delete jacobian_wrt_x_part;
    }
    if (jacobian_wrt_x2_part) {
      delete jacobian_wrt_x2_part;
    }
    *start_row_idx += part_dof;
  }

  template <
      typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void transformJacobianAgainstOtherProduct(
      const Manifold& x, const RetractionInterface& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction) const {
    const RetractionInterface_<Manifold>& src_retraction_cast =
        static_cast<const RetractionInterface_<Manifold>&>(src_retraction);
    ASSERT(src_retraction_cast.isProduct());
    std::vector<const RetractionInterface*> this_parts = getParts();
    std::vector<const RetractionInterface*> src_parts =
        src_retraction_cast.getParts();
    ASSERT(this_parts.size() == src_parts.size());
    _transform_jacobian_impl(
        x, this_parts, src_parts, jacobian_under_src_retraction,
        jacobian_under_this_retraction,
        std::index_sequence_for<RetractionParts...>{});
  }

  template <
      typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction, std::size_t... Is>
  void _transform_jacobian_impl(
      const Manifold& x,
      const std::vector<const RetractionInterface*>& this_parts,
      const std::vector<const RetractionInterface*>& src_parts,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      std::index_sequence<Is...>) const {
    int src_start_col = 0;
    int this_start_col = 0;
    (_transform_jacobian_one_part<Is>(
         x, this_parts, src_parts, jacobian_under_src_retraction,
         jacobian_under_this_retraction, &src_start_col, &this_start_col),
     ...);
  }

  template <
      std::size_t _part_idx, typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void _transform_jacobian_one_part(
      const Manifold& x,
      const std::vector<const RetractionInterface*>& this_parts,
      const std::vector<const RetractionInterface*>& src_parts,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      int* src_start_col, int* this_start_col) const {
    int src_cols = src_parts[_part_idx]->dof(x.template part<_part_idx>());
    int this_cols = this_parts[_part_idx]->dof(x.template part<_part_idx>());
    auto src_block = jacobian_under_src_retraction.block(
        0, *src_start_col, jacobian_under_src_retraction.rows(), src_cols);
    auto this_block = jacobian_under_this_retraction->block(
        0, *this_start_col, jacobian_under_this_retraction->rows(), this_cols);
    this_block = this_parts[_part_idx]->transformJacobian(
        src_block, x.template part<_part_idx>(), *src_parts[_part_idx]);
    *src_start_col += src_cols;
    *this_start_col += this_cols;
  }

 protected:  // experimental
  template <
      typename OtherProductRetraction, typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void transformJacobianAgainstOtherProduct_2(
      const Manifold& x, const OtherProductRetraction& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction) const {
    if constexpr (OtherProductRetraction::kIsProduct) {
      _transform_jacobian_impl_2(
          x, src_retraction, jacobian_under_src_retraction,
          jacobian_under_this_retraction,
          std::index_sequence_for<RetractionParts...>{});
    } else {
      throw std::runtime_error("Not implemented");
    }
  }

  template <
      typename OtherProductRetraction, typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction, std::size_t... Is>
  void _transform_jacobian_impl_2(
      const Manifold& x, const OtherProductRetraction& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      std::index_sequence<Is...>) const {
    int src_start_col = 0;
    int this_start_col = 0;
    (_transform_jacobian_one_part_2<Is>(
         x, src_retraction, jacobian_under_src_retraction,
         jacobian_under_this_retraction, &src_start_col, &this_start_col),
     ...);
  }

  template <
      std::size_t _part_idx, typename OtherProductRetraction,
      typename JacobianMatrixWrtSrcRetraction,
      typename JacobianMatrixWrtThisRetraction>
  void _transform_jacobian_one_part_2(
      const Manifold& x, const OtherProductRetraction& src_retraction,
      const JacobianMatrixWrtSrcRetraction& jacobian_under_src_retraction,
      JacobianMatrixWrtThisRetraction* jacobian_under_this_retraction,
      int* src_start_col, int* this_start_col) const {
    int src_cols = src_retraction.template part<_part_idx>().dof(
        x.template part<_part_idx>());
    int this_cols =
        derived()->template part<_part_idx>().dof(x.template part<_part_idx>());
    auto src_block = jacobian_under_src_retraction.block(
        0, *src_start_col, jacobian_under_src_retraction.rows(), src_cols);
    auto this_block = jacobian_under_this_retraction->block(
        0, *this_start_col, jacobian_under_this_retraction->rows(), this_cols);
    this_block = derived()->template part<_part_idx>().transformJacobian(
        src_block, x.template part<_part_idx>(),
        src_retraction.template part<_part_idx>());
    *src_start_col += src_cols;
    *this_start_col += this_cols;
  }

 protected:
  std::tuple<RetractionParts...> parts_;

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

}  // namespace sk4slam
