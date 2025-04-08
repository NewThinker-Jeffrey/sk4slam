#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_math/third_party/colmap/math/matrix.h"

namespace sk4slam {

// Perform RQ decomposition on matrix. The RQ decomposition transforms a matrix
// A into the product of an upper triangular matrix R (also known as
// right-triangular) and an orthogonal matrix Q.
template <typename MatrixType>
inline void decomposeMatrixRQ(
    const MatrixType& A, MatrixType* R, MatrixType* Q) {
  return sk4slam_colmap::DecomposeMatrixRQ(A, R, Q);
}

template <typename MatrixType, typename MatrixType2 = MatrixType>
bool safeInverseMatrix(const MatrixType& matrix, MatrixType2* inversed) {
  ASSERT(matrix.rows() == matrix.cols());
  using Scalar = typename MatrixType::Scalar;
  static const Scalar eps = Eigen::NumTraits<Scalar>::epsilon();
  Eigen::JacobiSVD<MatrixType> svd(
      matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  if (singular_values(matrix.rows() - 1) < eps) {
    return false;
  }
  Eigen::Vector3d inv_singular_values = singular_values.array().inverse();
  *inversed = svd.matrixV() * inv_singular_values.asDiagonal() *
              svd.matrixU().transpose();
  return true;
}

template <typename Vector3Like>
Eigen::Matrix<typename Vector3Like::Scalar, 3, 3> skew3(
    const Vector3Like& vec) {
  using Scalar = typename Vector3Like::Scalar;
  static const Scalar kZero = Scalar(0);
  Eigen::Matrix<Scalar, 3, 3> skew_matrix;
  // // clang-format off
  // skew_matrix <<   kZero,  -vec(2),   vec(1),
  //                 vec(2),    kZero,  -vec(0),
  //                -vec(1),   vec(0),   kZero;
  // // clang-format on
  Scalar* m = skew_matrix.data();
  const Scalar& x = vec[0];
  const Scalar& y = vec[1];
  const Scalar& z = vec[2];
  m[0] = kZero;
  m[1] = z;
  m[2] = -y;
  m[3] = -z;
  m[4] = kZero;
  m[5] = x;
  m[6] = y;
  m[7] = -x;
  m[8] = kZero;
  return skew_matrix;
}

template <typename MatrixType>
MatrixType expMat(
    const MatrixType& matrix, int max_order = -1, double eps = 1e-8) {
  ASSERT(matrix.rows() == matrix.cols());
  using Scalar = typename MatrixType::Scalar;
  MatrixType item =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
          matrix.rows(), matrix.cols());
  MatrixType sum = item;
  int max_iter;

  if (max_order < 0) {
    max_iter = std::numeric_limits<int>::max();
  } else {
    max_iter = max_order + 1;
  }

  int i;
  for (i = 1; i < max_iter; i++) {
    item *= (matrix / Scalar(i));
    sum += item;
    // if (max_order < 0) {
    Scalar max_abs = item.array().abs().maxCoeff();
    if (max_abs < eps) {
      break;
    }
    // }
  }
  if (i == max_iter) {
    LOGA("expMat max iter reached");
    i -= 1;
  }
  LOGA("expMat order: %d", i);
  return sum;
}

// Givens rotation to zero out subdiagonal elements in a row.
// This function applies a sequence of Givens rotations to eliminate subdiagonal
// elements in the specified row (`row`) of the input matrix `m`. The affected
// columns start from `start_col` and span `n_cols_to_eliminate` columns.
// If `n_cols_to_eliminate == -1`, all columns from `start_col` to the end of
// the row are processed.
template <typename MatrixType>
void GivensRotation(
    MatrixType& m, int row, int start_col = 0, int n_cols_to_eliminate = -1) {
  using Scalar = typename MatrixType::Scalar;

  // Constants for numerical stability and readability
  static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
  static const Scalar kOne = Scalar(1.0);
  static const Scalar k0p5 = Scalar(0.5);
  static const Scalar kZero = Scalar(0.0);
  using std::abs;
  using std::sqrt;

  // Ensure the target element (row, start_col) is below the diagonal.
  // Skip if `row` is not below `start_col` or if indices are out of bounds.
  if (row <= start_col || row >= m.rows() || start_col >= m.cols()) {
    return;  // No operation needed
  }

  // Determine the actual number of columns to process.
  int total_cols = m.cols();
  if (n_cols_to_eliminate == -1) {
    // Process all columns from `start_col` to the end of the row.
    n_cols_to_eliminate = std::min(total_cols - start_col, row - start_col);
  } else {
    // Limit processing to the valid range of columns and rows.
    n_cols_to_eliminate = std::min(
        n_cols_to_eliminate, std::min(total_cols - start_col, row - start_col));
  }

  // Iterate over the columns to apply Givens rotations and eliminate elements.
  for (int col = start_col; col < start_col + n_cols_to_eliminate; ++col) {
    Scalar x = m(col, col);  // Diagonal element
    Scalar y = m(row, col);  // Subdiagonal element to eliminate

    // Skip rotation if the subdiagonal element is already close to zero.
    if (abs(y) < kEps) {
      continue;
    }

    // Compute the Givens rotation coefficients.
    Scalar r2 = x * x + y * y;  // Squared magnitude
    Scalar r;
    if (abs(r2) < kEps) {
      // Use a numerically stable approximation for very small values.
      Scalar x_over_y = x / y;
      r = y * (kOne + k0p5 * x_over_y * x_over_y);
    } else {
      r = sqrt(r2);  // Compute the magnitude
    }
    Scalar c = x / r;   // cos(θ)
    Scalar s = -y / r;  // sin(θ)

    // Explicitly set the eliminated element to zero.
    m(row, col) = kZero;

    // Set the diagonal element to `r`.
    // For column `col`, we have:
    //     m(col, col) = c * x - s * y
    //                 = (x^2 / r) + (y^2 / r)
    //                 = r,
    // which is guaranteed to be non-negative. Thus, if QR decomposition
    // is performed using this method, the diagonal elements of R
    // will always be non-negative.
    m(col, col) = r;

    // Apply Givens rotation to the remaining columns.
    for (int j = col + 1; j < m.cols(); ++j) {
      Scalar temp = c * m(col, j) - s * m(row, j);
      m(row, j) = s * m(col, j) + c * m(row, j);
      m(col, j) = temp;
    }
  }
}

// QR decomposition of a matrix using Givens rotations.
// This function performs a QR decomposition of the input matrix `R` by applying
// a sequence of Givens rotations to zero out subdiagonal elements in `R`.
// The result is an upper triangular matrix `R`.
template <typename MatrixType>
void GivensQRDecomposition(MatrixType& R) {
  for (int i = 1; i < R.rows(); i++) {
    GivensRotation(
        R, i, 0, -1);  // Eliminate all subdiagonal elements in row `i`.
  }
}

// Perform QR decomposition on an augmented matrix [A|b] using Givens rotations.
// The augmented matrix consists of a matrix `A` (left side) and a column vector
// `b` (right side). The function applies Givens rotations to eliminate
// subdiagonal elements in the left part of the augmented matrix while updating
// `b` accordingly.
template <typename MatrixType>
void GivensQRForAugmentedMatrix(MatrixType& augmented) {
  int rows = augmented.rows();
  int cols = augmented.cols() - 1;  // Exclude the last column (b)

  // Apply Givens rotations to the augmented matrix.
  for (int i = 1; i < augmented.rows(); i++) {
    // Eliminate subdiagonal elements in row `i` up to column `cols`.
    GivensRotation(augmented, i, 0, std::min(cols, i));
  }
}

}  // namespace sk4slam
