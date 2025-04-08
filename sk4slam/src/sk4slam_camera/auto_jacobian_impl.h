#pragma once

#include <Eigen/Core>

#include "ceres/jet.h"
#include "sk4slam_basic/logging.h"

namespace sk4slam {

/// @brief    A helper function for project3AndAutoJacobians()
template <typename CameraModel, int N1, int N2, int N3>
bool project3AndAutoJacobiansHelper(
    const double* point, const double* proj_params, const double* dist_params,
    double* pixel, double* jacobians) {
  static constexpr int N = N1 + N2 + N3;
  using Jet = ceres::Jet<double, N>;

  // Initialize the jets.
  Eigen::Matrix<Jet, 3, 1> point_jet;
  Eigen::Matrix<Jet, CameraModel::kNumProjectionParams, 1> proj_params_jet;
  Eigen::Matrix<Jet, CameraModel::kNumDistortionParams, 1> dist_params_jet;
  int k = 0;
  if constexpr (N1 > 0) {
    for (int i = 0; i < N1; ++i) {
      point_jet[i] = Jet(point[i], k++);
    }
  } else {
    for (int i = 0; i < N1; ++i) {
      point_jet[i] = Jet(point[i]);
    }
  }

  if constexpr (N2 > 0) {
    for (int i = 0; i < N2; ++i) {
      proj_params_jet[i] = Jet(proj_params[i], k++);
    }
  } else {
    for (int i = 0; i < N2; ++i) {
      proj_params_jet[i] = Jet(proj_params[i]);
    }
  }

  if constexpr (N3 > 0) {
    for (int i = 0; i < N3; ++i) {
      dist_params_jet[i] = Jet(dist_params[i], k++);
    }
  } else {
    for (int i = 0; i < N3; ++i) {
      dist_params_jet[i] = Jet(dist_params[i]);
    }
  }
  ASSERT(k == N);

  // Compute the projection.
  Eigen::Matrix<Jet, 2, 1> pixel_jet;
  bool success = CameraModel::project3(
      point_jet, proj_params_jet, dist_params_jet, pixel_jet.data());
  Eigen::Map<Eigen::Matrix<double, N, 1>> jac0(jacobians);
  Eigen::Map<Eigen::Matrix<double, N, 1>> jac1(jacobians + N);
  pixel[0] = pixel_jet[0].a;
  pixel[1] = pixel_jet[1].a;
  jac0 = pixel_jet[0].v;
  jac1 = pixel_jet[1].v;
  return success;
}

/// @brief   Implementation of the auto jacobian function for a camera model.
///          See @ref CameraModel::project3AndComputeJacobians().
template <typename CameraModel>
bool project3AndAutoJacobians(
    const double* point, const double* proj_params, const double* dist_params,
    double* pixel, double* jacobian_wrt_point, double* jacobian_wrt_proj_params,
    double* jacobian_wrt_dist_params) {
  // Allocate the jacobian matrix.
  int n = 0;
  int n1 = 0, n2 = 0, n3 = 0;
  if (jacobian_wrt_point) {
    n1 = 3;
    n += n1;
  }
  if (jacobian_wrt_proj_params) {
    n2 = CameraModel::kNumProjectionParams;
    n += n2;
  }
  if (jacobian_wrt_dist_params) {
    n3 = CameraModel::kNumDistortionParams;
    n += n3;
  }
  if (n == 0) {
    return CameraModel::project3(point, proj_params, dist_params, pixel);
  }

  using JacobianMatrix =
      Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>;
  using JacobianMap = Eigen::Map<JacobianMatrix>;

  // // resize the jacobians to n columns
  // JacobianMatrix jacobians_mat;
  // jacobians_mat.resize(2, n);
  // double* jacobians = jacobians_mat.data();

  double jacobians
      [2 * (CameraModel::kNumProjectionParams +
            CameraModel::kNumDistortionParams + 3)];
  JacobianMap jacobians_mat(jacobians, 2, n);
  bool success = false;

  // Compute the pixel and jacobians.
  if (n1 && n2 && n3) {
    success = project3AndAutoJacobiansHelper<
        CameraModel, 3, CameraModel::kNumProjectionParams,
        CameraModel::kNumDistortionParams>(
        point, proj_params, dist_params, pixel, jacobians);
  } else if (n1 && n2) {
    success = project3AndAutoJacobiansHelper<
        CameraModel, 3, CameraModel::kNumProjectionParams, 0>(
        point, proj_params, dist_params, pixel, jacobians);
  } else if (n1 && n3) {
    success = project3AndAutoJacobiansHelper<
        CameraModel, 3, 0, CameraModel::kNumDistortionParams>(
        point, proj_params, dist_params, pixel, jacobians);
  } else if (n2 && n3) {
    success = project3AndAutoJacobiansHelper<
        CameraModel, 0, CameraModel::kNumProjectionParams,
        CameraModel::kNumDistortionParams>(
        point, proj_params, dist_params, pixel, jacobians);
  } else if (n1) {
    success = project3AndAutoJacobiansHelper<CameraModel, 3, 0, 0>(
        point, proj_params, dist_params, pixel, jacobians);
  } else if (n2) {
    success = project3AndAutoJacobiansHelper<
        CameraModel, 0, CameraModel::kNumProjectionParams, 0>(
        point, proj_params, dist_params, pixel, jacobians);
  } else {
    ASSERT(n3);
    success = project3AndAutoJacobiansHelper<
        CameraModel, 0, 0, CameraModel::kNumDistortionParams>(
        point, proj_params, dist_params, pixel, jacobians);
  }

  // Copy the jacobian matrix to the output.
  int col_idx = 0;
  if (jacobian_wrt_point) {
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
        jacobian_wrt_point_mat(jacobian_wrt_point);
    jacobian_wrt_point_mat = jacobians_mat.block<2, 3>(0, col_idx);
    col_idx += 3;
  }
  if (jacobian_wrt_proj_params) {
    typename CameraModel::template JacobianMap<
        double, 2, CameraModel::kNumProjectionParams>
        jacobian_wrt_proj_params_mat(jacobian_wrt_proj_params);
    jacobian_wrt_proj_params_mat =
        jacobians_mat.block<2, CameraModel::kNumProjectionParams>(0, col_idx);
    col_idx += CameraModel::kNumProjectionParams;
  }
  if (jacobian_wrt_dist_params) {
    typename CameraModel::template JacobianMap<
        double, 2, CameraModel::kNumDistortionParams>
        jacobian_wrt_dist_params_mat(jacobian_wrt_dist_params);
    jacobian_wrt_dist_params_mat =
        jacobians_mat.block<2, CameraModel::kNumDistortionParams>(0, col_idx);
    col_idx += CameraModel::kNumDistortionParams;
  }
  ASSERT(col_idx == n);
  return success;
}

template <typename LUT>
bool LutProject3AndAutoJacobian(
    const LUT* lut, const double* point, double* pixel,
    double* jacobian_wrt_point) {
  if (!jacobian_wrt_point) {
    return lut->project3(point, pixel);
  }

  using JacobianMatrix = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
  using JacobianMap = Eigen::Map<JacobianMatrix>;

  using Jet = ceres::Jet<double, 3>;

  Eigen::Matrix<Jet, 3, 1> point_jet;

  point_jet[0] = Jet(point[0], 0);
  point_jet[1] = Jet(point[1], 1);
  point_jet[2] = Jet(point[2], 2);

  Eigen::Matrix<Jet, 2, 1> pixel_jet;
  bool success = lut->project3(point_jet, pixel_jet.data());

  Eigen::Map<Eigen::Matrix<double, 3, 1>> jac0(jacobian_wrt_point);
  Eigen::Map<Eigen::Matrix<double, 3, 1>> jac1(jacobian_wrt_point + 3);
  pixel[0] = pixel_jet[0].a;
  pixel[1] = pixel_jet[1].a;
  jac0 = pixel_jet[0].v;
  jac1 = pixel_jet[1].v;
  return success;
}

}  // namespace sk4slam
