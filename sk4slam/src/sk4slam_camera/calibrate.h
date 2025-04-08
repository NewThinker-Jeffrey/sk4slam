#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

/// @brief  This function computes the pinhole camera intrinsics from a set of
///         homography matrices.
/// @param homography_matrices  The set of homography matrices.
/// @return  The pinhole camera intrinsics.
///
/// This function implements the initial estimation of the pinhole camera
/// intrinsics in the paper "A flexible new technique for camera calibration"
/// by Z. Zhang. (https://ieeexplore.ieee.org/document/888718).
/// But note that we've made some simplifications in the implementation (we
/// assume the cross factor @f$ \gamma @f$ is zero).
inline std::unique_ptr<Eigen::Matrix<double, 4, 1>>
computePinholeIntrinsicsFromHomographyMatrices(
    const std::vector<Eigen::Matrix3d>& homography_matrices) {
  Eigen::Matrix<double, 4, 1> intrinsics;
  size_t n = homography_matrices.size();
  if (n < 2) {
    return nullptr;
  }

  Eigen::MatrixXd V(2 * n, 5);

  auto compute_v = [](const Eigen::Matrix3d& H, int i, int j) {
    Eigen::Matrix<double, 1, 5> v;
    v << H(0, i) * H(0, j),
        //  H(0, i) * H(1, j) + H(1, i) * H(0, j),  // this is omitted since we
        //  assume gamma = 0
        H(1, i) * H(1, j), H(2, i) * H(0, j) + H(0, i) * H(2, j),
        H(2, i) * H(1, j) + H(1, i) * H(2, j), H(2, i) * H(2, j);
    return v;
  };

  for (size_t i = 0; i < n; ++i) {
    V.row(2 * i) = compute_v(homography_matrices[i], 0, 1);
    V.row(2 * i + 1) = compute_v(homography_matrices[i], 0, 0) -
                       compute_v(homography_matrices[i], 1, 1);
  }

  // Solve Vb = 0 by SVD decomposition
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
  Eigen::Matrix<double, 5, 1> b = svd.matrixV().col(4);
  double B11 = b(0);
  // double B12 = 0;  // B12 is assumed to be zero
  double B22 = b(1);
  double B13 = b(2);
  double B23 = b(3);
  double B33 = b(4);

  double u0 = -B13 / B11;
  double v0 = -B23 / B22;
  double lambda = B33 - u0 * u0 * B11 - v0 * v0 * B22;
  double alpha = std::sqrt(lambda / B11);
  double beta = std::sqrt(lambda / B22);

  intrinsics << alpha, beta, u0, v0;
  return std::make_unique<Eigen::Matrix<double, 4, 1>>(intrinsics);
}

}  // namespace sk4slam
