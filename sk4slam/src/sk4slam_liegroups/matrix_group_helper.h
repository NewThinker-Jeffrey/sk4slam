#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_math/vector_space.h"

namespace sk4slam {

////////// Helper for Matrix LieGroups //////////

namespace matrix_group_internal {
template <typename T>
struct IsMatrixXprT;

template <typename MatrixXpr>
using MultReturnMatrix = Eigen::Matrix<
    typename MatrixXpr::Scalar, MatrixXpr::RowsAtCompileTime,
    MatrixXpr::ColsAtCompileTime>;

template <typename MatrixXpr>
struct RightMultHelper;
struct RightMultImpl;

DEFINE_HAS_MEMBER_VARIABLE(kSupportMatrixGroupCommonOps)
}  // namespace matrix_group_internal

// Whether type T is a matrix expression.
template <typename T>
inline constexpr bool IsMatrixXpr =
    matrix_group_internal::IsMatrixXprT<RawType<T>>::value;

// For a matrix group, if you need its elements can be multiplied or assigned
// as if they were matrices, just inherit this class.
// The DerivedLieGroup should be a matrix Lie Group and can be constructed
// from a matrix.
template <typename DerivedLieGroup>
class MatrixGroupCommonOps;

template <typename DerivedLieGroup>
inline constexpr bool SupportMatrixGroupCommonOps =
    matrix_group_internal::HasMemberVariable_kSupportMatrixGroupCommonOps<
        DerivedLieGroup>;

// MatrixGroupHelper assists in verifying whether a Lie group is a matrix
// LieGroup (a Lie subgroup of GL(n) for some n), and offers auxiliary
// properties and methods for such groups.
//
// It is assumed that each matrix LieGroup possesses a member function
// `matrix()` that yields a square matrix representation of the group
// element.
// Furthermore, if the inner type 'Ambient' of the LieGroup differs from
// the matrix type, it should be able to be static casted to a square
// matrix representation matching the dimensions of the matrix() function
// of the LieGroup.
// Similarly, if the inner type 'LieAlgebraEndomorphism' of the LieGroup
// is not matrix type, it should be able to be static casted to a square
// matrix representation matching the dimension of the LieGroup (::kDim).
template <typename LieGroup>
class MatrixGroupHelper {
  DEFINE_HAS_MEMBER_FUNCTION(matrix)
  template <bool _has_matrix, typename _LieGroup>
  struct _MatrixInfo {
    using Matrix = void;
    static constexpr int kRows = 0;
    static constexpr int kCols = 0;
  };
  template <typename _LieGroup>
  struct _MatrixInfo<true, _LieGroup> {
    using Matrix =
        std::remove_reference_t<decltype(std::declval<_LieGroup>().matrix())>;
    static constexpr int kRows = Matrix::RowsAtCompileTime;
    static constexpr int kCols = Matrix::ColsAtCompileTime;
  };
  static constexpr bool kHasMatrix = HasMemberFunction_matrix<LieGroup>;
  static constexpr int kMatrixRows = _MatrixInfo<kHasMatrix, LieGroup>::kRows;
  static constexpr int kMatrixCols = _MatrixInfo<kHasMatrix, LieGroup>::kCols;

 public:
  static constexpr bool kIsMatrixGroup =
      kHasMatrix && kMatrixRows > 0 && kMatrixRows == kMatrixCols;
  static constexpr int N = (kIsMatrixGroup ? kMatrixRows : 0);
  using Matrix = typename _MatrixInfo<kHasMatrix, LieGroup>::Matrix;

  // ambientToMatrix(ambient) -> Matrix
  template <
      typename _AmbientOrItsRef, typename _Matrix = Matrix,
      ENABLE_IF(!(std::is_same_v<_Matrix, void>))>
  static decltype(auto) ambientToMatrix(_AmbientOrItsRef&& ambient) {
    using _Ambient =
        std::remove_cv_t<std::remove_reference_t<_AmbientOrItsRef>>;
    static_assert(
        std::is_same_v<_Ambient, typename LieGroup::Ambient>,
        "Ambient type mismatch!");
    static_assert(std::is_same_v<_Matrix, Matrix>, "Matrix type mismatch!");
    if constexpr (std::is_base_of<_Matrix, typename LieGroup::Ambient>::value) {
      LOGA("ambientToMatrix: DO NOT need cast");

      return std::forward<_AmbientOrItsRef>(ambient);
    } else {
      LOGA("ambientToMatrix: need cast");
      // return std::forward<_AmbientOrItsRef>(ambient).matrix();

      // if the inner type 'Ambient' of the LieGroup differs from
      // the matrix type, it should be able to be static casted to
      // a square matrix representation matching the dimensions
      // of the matrix() function of the LieGroup.
      return static_cast<Matrix>(std::forward<_AmbientOrItsRef>(ambient));
    }
  }
};

}  // namespace sk4slam

template <
    typename MatrixXpr, typename MatrixGroup,
    ENABLE_IF_N(sk4slam::SupportMatrixGroupCommonOps<MatrixGroup>&&
                    sk4slam::IsMatrixXpr<MatrixXpr>)>
decltype(auto) operator*(const MatrixXpr& m, const MatrixGroup& g);

template <
    typename Scalar, typename MatrixGroup,
    ENABLE_IF_N(
        sk4slam::SupportMatrixGroupCommonOps<MatrixGroup> &&
        (std::is_convertible_v<Scalar, typename MatrixGroup::Scalar>))>
decltype(auto) operator*(const Scalar& s, const MatrixGroup& g);

////////// template implementations //////////

namespace sk4slam {

template <typename DerivedLieGroup>
class MatrixGroupCommonOps {
 public:
  static constexpr bool kSupportMatrixGroupCommonOps = true;

  const MatrixGroupCommonOps& matrixCommonOps() const {
    return *this;
  }

  // static cast to matrix
  // With this operator you can assign a group element to a matrix, e.g.
  //    Eigen::MatrixXd m = group_element;
  template <
      typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
      int _MaxCols>
  operator Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>()
      const {
    LOGA("operator Matrix() called.");
    return _derived()->matrix();
  }

  // clang-format off
  // TODO(jeffrey):
  //    Assigning a group element to a block of a matrix is not well supported
  //    yet! i.e. the following code may not work for now:
  //
  //        Eigen::MatrixXd m(3,3)
  //        m.block<3,3>(0,0) = group_element;
  //
  //    Consider: how to implement it?
  //    Note that the following implementation is potentially dangerous! See
  //    the comment inside the function.
  //
  // template <
  //     typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
  //     typename _Derived = DerivedLieGroup,
  //     ENABLE_IF(
  //         (MatrixGroupHelper<_Derived>::kIsMatrixGroup) &&
  //         (BlockRows == MatrixGroupHelper<_Derived>::N ||
  //          BlockRows == Eigen::Dynamic) &&
  //         (BlockCols == MatrixGroupHelper<_Derived>::N ||
  //          BlockCols == Eigen::Dynamic))>
  // operator Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>() const {
  //   LOGA("operator Block() called.");
  //   // This implementation is potentially dangerous! because m is
  //   // a temporary object, so the lifetime of the data referenced by
  //   // the returned m.block() is limited to the scope of the function call.
  //   // So the following code may not work.
  //   XprType m = _derived<_Derived>()->matrix();
  //   if constexpr (BlockRows == Eigen::Dynamic &&
  //                 BlockCols == Eigen::Dynamic) {
  //     return m.template block<BlockRows, BlockCols>(
  //         0, 0,
  //         MatrixGroupHelper<_Derived>::N, MatrixGroupHelper<_Derived>::N);
  //   } else if constexpr (BlockRows == Eigen::Dynamic) {
  //     return m.template block<BlockRows, BlockCols>(
  //         0, 0, MatrixGroupHelper<_Derived>::N, BlockCols);
  //   } else if constexpr (BlockCols == Eigen::Dynamic) {
  //     return m.template block<BlockRows, BlockCols>(
  //         0, 0, BlockRows, MatrixGroupHelper<_Derived>::N);
  //   } else {
  //     return m.template block<BlockRows, BlockCols>(0, 0);
  //   }
  // }
  // clang-format on

  // Assign from matrix
  template <
      typename MatrixXpr, typename _Derived = DerivedLieGroup,
      ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  _Derived& operator=(const MatrixXpr& m) {
    LOGA(
        "operator=(const MatrixXpr&) called. (MatrixXpr = %s)",
        classname<MatrixXpr>());
    *_derived<_Derived>() = _Derived(m);
    return *_derived<_Derived>();
  }

  // When multiplying with a matrix, the group element is just treated as a
  // matrix.
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

  template <
      typename _Scalar, typename _Derived = DerivedLieGroup,
      ENABLE_IF((std::is_convertible_v<_Scalar, typename _Derived::Scalar>))>
  decltype(auto) operator*(const _Scalar& s) const {
    LOGA("operator*(const Scalar&) called.");
    return static_cast<RawType<decltype(_derived()->matrix())>>(
        _derived()->matrix() * s);
  }

  // Return
  //    J = d [Exp(x) * v] / d (x)      (Evaluated at x=0, x ∈ LieAlg, v ∈ R^N)
  template <typename _Derived = DerivedLieGroup>
  static Eigen::Matrix<
      typename _Derived::Scalar, MatrixGroupHelper<_Derived>::N, _Derived::kDim>
  JmultVector(
      const Vector<MatrixGroupHelper<_Derived>::N, typename _Derived::Scalar>&
          v) {
    Eigen::Matrix<
        typename _Derived::Scalar, MatrixGroupHelper<_Derived>::N,
        _Derived::kDim>
        J;
    for (size_t i = 0; i < _Derived::kDim; i++) {
      J.col(i) = matrixGenerator(i) * v;
    }
    return J;
  }

  // Retrun type:
  //     const Eigen::Matrix<Scalar, N, N>& .
  template <typename _Derived = DerivedLieGroup>
  static const auto& matrixGenerator(int i) {
    static const auto matrix_generators = computeMatrixGenerators();
    return matrix_generators[i];
  }

 protected:
  template <typename MatrixXpr>
  using MultReturnMatrix = matrix_group_internal::MultReturnMatrix<MatrixXpr>;

  template <typename _Derived = DerivedLieGroup>
  static auto computeMatrixGenerators() {
    static constexpr int N = MatrixGroupHelper<_Derived>::N;
    static constexpr int kDim = _Derived::kDim;
    using Scalar = typename _Derived::Scalar;
    std::vector<Eigen::Matrix<Scalar, N, N>> matrix_generators(kDim);
    for (int i = 0; i < kDim; ++i) {
      matrix_generators[i] = _Derived::generator(i);
    }
    return matrix_generators;
  }

  // NOTE:
  // The return type should be an evaluated matrix here since
  // _derived()->matrix() may return a temporary matrix:
  //     If the return type is decltype(auto), then the return type will be
  //     deduced as an Eigen::Product object which depends on the data of
  //     _derived()->matrix(). But the data might be released if the returned
  //     matrix is a temporary object, which may cause unexpected behavior.
  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  // decltype(auto)
  MultReturnMatrix<MatrixXpr> _mult_matrix_impl(const MatrixXpr& m) const {
    static_assert(
        std::is_same_v<
            typename MatrixXpr::Scalar, typename DerivedLieGroup::Scalar>);
    return _derived()->matrix() * m;
  }

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  // decltype(auto)
  MultReturnMatrix<MatrixXpr> _rmult_matrix_impl(const MatrixXpr& m) const {
    static_assert(
        std::is_same_v<
            typename MatrixXpr::Scalar, typename DerivedLieGroup::Scalar>);
    return m * _derived()->matrix();
  }

  template <
      typename MatrixXpr, typename MatrixGroup,
      ENABLE_IF_N(sk4slam::SupportMatrixGroupCommonOps<MatrixGroup>&&
                      sk4slam::IsMatrixXpr<MatrixXpr>)>
  friend decltype(auto)(::operator*)(const MatrixXpr& m, const MatrixGroup& g);

  friend class matrix_group_internal::RightMultImpl;

 private:
  template <typename _Derived = DerivedLieGroup>
  const _Derived* _derived() const {
    return static_cast<const _Derived*>(this);
  }
  template <typename _Derived = DerivedLieGroup>
  _Derived* _derived() {
    return static_cast<_Derived*>(this);
  }
};

namespace matrix_group_internal {

template <typename T>
struct IsMatrixXprT {
  static constexpr bool value = false;
};

template <
    typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
    int _MaxCols>
struct IsMatrixXprT<
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  static constexpr bool value = true;
};

template <typename UnaryOp, typename ValueType>
struct IsMatrixXprT<Eigen::CwiseUnaryOp<UnaryOp, ValueType>> {
  static constexpr bool value = true;
};

template <typename BinaryOp, typename ValueType1, typename ValueType2>
struct IsMatrixXprT<Eigen::CwiseBinaryOp<BinaryOp, ValueType1, ValueType2>> {
  static constexpr bool value = true;
};

template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
struct IsMatrixXprT<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> {
  static constexpr bool value = true;
};

template <typename Lhs, typename Rhs, int Option>
struct IsMatrixXprT<Eigen::Product<Lhs, Rhs, Option>> {
  static constexpr bool value = true;
};

struct RightMultImpl {
  template <typename MatrixXpr, typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    return g.matrixCommonOps()._rmult_matrix_impl(m);
  }
};

template <typename MatrixXpr>
struct RightMultHelper {
  static_assert(
      IsMatrixXpr<MatrixXpr>, "MatrixXpr must be a matrix expression.");
  static constexpr bool kSpecialized = false;
  template <typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    static_assert(
        !kSpecialized, "The primary template should never be instantiated.");
    return MultReturnMatrix<MatrixXpr>();
  }
};

template <
    typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
    int _MaxCols>
struct RightMultHelper<
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  using MatrixXpr =
      Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
  static constexpr bool kSpecialized = true;
  template <typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    LOGA("operator*(Matrix, const MatrixGroupCommonOps&) called.");
    return RightMultImpl::rmult(m, g);
  }
};

template <typename UnaryOp, typename ValueType>
struct RightMultHelper<Eigen::CwiseUnaryOp<UnaryOp, ValueType>> {
  using MatrixXpr = Eigen::CwiseUnaryOp<UnaryOp, ValueType>;
  static constexpr bool kSpecialized = true;
  template <typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    LOGA("operator*(CwiseUnaryOp, const MatrixGroupCommonOps&) called.");
    return RightMultImpl::rmult(m, g);
  }
};

template <typename BinaryOp, typename ValueType1, typename ValueType2>
struct RightMultHelper<Eigen::CwiseBinaryOp<BinaryOp, ValueType1, ValueType2>> {
  using MatrixXpr = Eigen::CwiseBinaryOp<BinaryOp, ValueType1, ValueType2>;
  static constexpr bool kSpecialized = true;
  template <typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    LOGA("operator*(CwiseBinaryOp, const MatrixGroupCommonOps&) called.");
    return RightMultImpl::rmult(m, g);
  }
};

template <typename Lhs, typename Rhs, int Option>
struct RightMultHelper<Eigen::Product<Lhs, Rhs, Option>> {
  using MatrixXpr = Eigen::Product<Lhs, Rhs, Option>;
  static constexpr bool kSpecialized = true;
  template <typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    LOGA("operator*(Product, const MatrixGroupCommonOps&) called.");
    return RightMultImpl::rmult(m, g);
  }
};

template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
struct RightMultHelper<
    Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> {
  using MatrixXpr = Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>;
  static constexpr bool kSpecialized = true;
  template <typename DerivedLieGroup>
  static MultReturnMatrix<MatrixXpr> rmult(
      const MatrixXpr& m, const DerivedLieGroup& g) {
    LOGA("operator*(Block, const MatrixGroupCommonOps&) called.");
    return RightMultImpl::rmult(m, g);
  }
};

}  // namespace matrix_group_internal

}  // namespace sk4slam

template <
    typename MatrixXpr, typename MatrixGroup,
    ENABLE_IF(sk4slam::SupportMatrixGroupCommonOps<MatrixGroup>&&
                  sk4slam::IsMatrixXpr<MatrixXpr>)>
decltype(auto) operator*(const MatrixXpr& m, const MatrixGroup& g) {
  using RightMultHelper =
      sk4slam::matrix_group_internal::RightMultHelper<MatrixXpr>;
  return RightMultHelper::rmult(m, g);
}

template <
    typename Scalar, typename MatrixGroup,
    ENABLE_IF(
        sk4slam::SupportMatrixGroupCommonOps<MatrixGroup> &&
        (std::is_convertible_v<Scalar, typename MatrixGroup::Scalar>))>
decltype(auto) operator*(const Scalar& s, const MatrixGroup& g) {
  LOGA("operator*(const Scalar&, const MatrixGroupCommonOps&) called.");
  return static_cast<RawType<decltype(g.matrix())>>(s * g.matrix());
}
