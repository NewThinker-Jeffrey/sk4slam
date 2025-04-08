
#include "sk4slam_geometry/essential_matrix.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_geometry/fundamental_matrix.h"
#include "sk4slam_geometry/homography_matrix.h"
#include "sk4slam_geometry/third_party/colmap/estimators/essential_matrix.h"
#include "sk4slam_geometry/third_party/colmap/geometry/essential_matrix.h"
#include "sk4slam_geometry/third_party/colmap/geometry/homography_matrix.h"
#include "sk4slam_geometry/utils.h"
#include "sk4slam_math/matrix.h"

namespace sk4slam {

std::vector<Eigen::Matrix3d> EssentialMatrix::solveWith5Points(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  std::vector<Eigen::Matrix3d> Es =
      sk4slam_colmap::EssentialMatrixFivePointEstimator::Estimate(
          Xs_2d, Xprimes_2d);

  // The solutions of 5-point (essential-matrix) algorithm should be exact.
  if (selected_indices.size() == 5) {
    for (const Eigen::Matrix3d& E : Es) {
      auto errs = FundamentalMatrix::computeSquaredAlgebraicErrors(
          E, selected_indices, point_pairs);
      // LOGA("Check the exactness of 5-point algorithm: errors: %s",
      //      toStr(errs, sqrt, Precision(6)).c_str());
      for (const auto& err : errs) {
        ASSERT(std::abs(err) < 1e-10);
      }
    }
  }
  return Es;
}

std::vector<Eigen::Matrix3d> EssentialMatrix::solveWith8Points(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  return sk4slam_colmap::EssentialMatrixEightPointEstimator::Estimate(
      Xs_2d, Xprimes_2d);
}

std::vector<Eigen::Matrix3d> EssentialMatrix::solveWithKnownRotation(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs, const Eigen::Matrix3d& R) {
  // TODO(jeffrey): implement this.

  if (selected_indices.size() < 2) {
    return std::vector<Eigen::Matrix3d>();
  }

  Eigen::MatrixXd coeffs_mat;
  coeffs_mat.resize(selected_indices.size(), 3);
  for (size_t i = 0; i < selected_indices.size(); i++) {
    const auto& X = point_pairs[selected_indices[i]].first;
    const auto& Xprime = point_pairs[selected_indices[i]].second;
    Eigen::Vector3d rotated_X3d = R * X.homogeneous();
    Eigen::Vector3d Xprime3d = Xprime.homogeneous();
    rotated_X3d.normalize();
    Xprime3d.normalize();
    coeffs_mat.row(i) = rotated_X3d.cross(Xprime3d).transpose();
  }
  // Solve the homogeneous equation Ax = 0 using SVD. The nullspace of A is
  // the solution to the homogeneous equation. But note we should first check
  // whether the nullspace of A is degenerate, i.e., the second smallest
  // singular value should be larger than a threshold.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(coeffs_mat, Eigen::ComputeFullV);
  // Check whether the nullspace of A is degenerate.
  if (svd.singularValues()(0) < 1e-6 ||
      svd.singularValues()(1) / svd.singularValues()(0) < 1e-6 ||
      svd.singularValues()(2) / svd.singularValues()(1) > 1e-1) {
    return std::vector<Eigen::Matrix3d>();
  }

  Eigen::Vector3d t = svd.matrixV().col(2);
  Eigen::Matrix3d E = skew3(t) * R;
  return std::vector<Eigen::Matrix3d>{E};
}

void EssentialMatrix::computePose(
    const Eigen::Matrix3d& E, const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs, Eigen::Matrix3d* R,
    Eigen::Vector3d* t, std::vector<Eigen::Vector3d>* points3D) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  sk4slam_colmap::PoseFromEssentialMatrix(E, Xs_2d, Xprimes_2d, R, t, points3D);
}

void EssentialMatrix::findOptimalImageObservations(
    const Eigen::Matrix3d& E, const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2, Eigen::Vector2d* optimal_point1,
    Eigen::Vector2d* optimal_point2) {
  sk4slam_colmap::FindOptimalImageObservations(
      E, point1, point2, optimal_point1, optimal_point2);
}

bool EssentialMatrix::check_H_Degenerate(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs,
    const Eigen::Matrix<double, 3, 3>& E, const double err_thr,
    Eigen::Matrix<double, 3, 3>* H, std::vector<size_t>* h_sample_indices,
    size_t points_used_for_initial_h) {
  ASSERT(selected_indices.size() == 5 || selected_indices.size() == 8);
  if (selected_indices.size() == 8) {
    return FundamentalMatrix::check_H_Degenerate(
        selected_indices, point_pairs, E, err_thr, H, h_sample_indices,
        points_used_for_initial_h);
  }

  ASSERT(H != nullptr);
  ASSERT(h_sample_indices != nullptr);
  ASSERT(selected_indices.size() == 5);

  static const std::vector<std::vector<size_t>> triplets = {{0, 1, 2}};

  static const std::vector<std::vector<size_t>> quadruplets = {{0, 1, 2, 3}};

  static const std::vector<std::vector<size_t>> quintuplets = {{0, 1, 2, 3, 4}};

  // For 5-point algorithm, force `points_used_for_initial_h` to be 5 to
  // ensure the initial H is reliable.
  points_used_for_initial_h = 5;

  const std::vector<std::vector<size_t>>* p_initial_subsets = nullptr;

  if (points_used_for_initial_h == 3) {
    p_initial_subsets = &triplets;
  } else if (points_used_for_initial_h == 4) {
    p_initial_subsets = &quadruplets;
  } else if (points_used_for_initial_h == 5) {
    p_initial_subsets = &quintuplets;
  }
  ASSERT(p_initial_subsets != nullptr);

  auto solve_Hs = [&](const std::vector<size_t>& subset) {
    LOGA(
        BLUE
        "EssentialMatrix::check_H_Degenerate(): Estimate "
        "the initial H with the subset %s" RESET,
        toStr(subset).c_str());

    if (subset.size() == 3) {
      return HomographyMatrix::solveWithKnownF(
          {selected_indices[subset[0]], selected_indices[subset[1]],
           selected_indices[subset[2]]},
          point_pairs, E);
    } else if (subset.size() == 4) {
      return HomographyMatrix::solveDLT(
          {selected_indices[subset[0]], selected_indices[subset[1]],
           selected_indices[subset[2]], selected_indices[subset[3]]},
          point_pairs);
    } else {
      ASSERT(subset.size() == 5);
      return HomographyMatrix::solveDLT(
          {selected_indices[subset[0]], selected_indices[subset[1]],
           selected_indices[subset[2]], selected_indices[subset[3]],
           selected_indices[subset[4]]},
          point_pairs);
    }
  };

  const std::vector<std::vector<size_t>>& initial_subsets = *p_initial_subsets;

  std::vector<Eigen::Matrix3d> Hs = solve_Hs(initial_subsets[0]);

  if (Hs.size() == 0) {
    LOGA("EssentialMatrix::check_H_Degenerate(): Hs.size() == 0!");
    return false;
  }
  ASSERT(Hs.size() == 1);
  *H = Hs[0];
  h_sample_indices->reserve(selected_indices.size());

  std::vector<double> errs(selected_indices.size());
  std::vector<int> is;
  is.reserve(selected_indices.size());

  // Check if all the 5 points are consistent with the homography.
  for (size_t i = 0; i < selected_indices.size(); i++) {
    const auto& pair = point_pairs[selected_indices[i]];
    const Eigen::Vector3d Xi = pair.first.homogeneous();
    const Eigen::Vector3d Xprimei = pair.second.homogeneous();
    Eigen::Vector3d xi = (*H) * Xi;
    xi /= xi.z();
    double err = (xi - Xprimei).squaredNorm();
    errs[i] = err;
    if (err < err_thr) {
      is.push_back(i);
      h_sample_indices->push_back(selected_indices[i]);
    } else {
      // h_sample_indices->clear();
      // return false;
    }
  }

  LOGA(
      BLUE "EssentialMatrix::check_H_Degenerate(): errs = %s" RESET,
      toStr(errs, sqrt).c_str());
  LOGA(
      BLUE "EssentialMatrix::check_H_Degenerate(): inliers ids = %s" RESET,
      toStr(is).c_str());

  return h_sample_indices->size() >= 5;
}

bool EssentialMatrix::CeresOptimizer::optimizeWithCeres(
    Eigen::Matrix3d* E, int max_iter) const {
  CeresProblem problem;
  CeresParamBlock param_blk(&problem, toManifold(*E));
  ceres::CostFunction* cost_func = createCostFunction(&param_blk);
  problem.AddResidualBlock(cost_func, nullptr, param_blk.LocalData());

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = max_iter;
  ceres::Solver::Summary summary;
  // static constexpr bool print_iterations = true;
  static constexpr bool print_iterations = false;
  ceres::TerminationType termination_type =
      CeresSolve(solver_options, &problem, &summary, print_iterations);
  // if (termination_type != ceres::CONVERGENCE) {
  //   return false;
  // }
  *E = fromManifold(param_blk.GetGlobal());
  return true;
}

}  // namespace sk4slam
