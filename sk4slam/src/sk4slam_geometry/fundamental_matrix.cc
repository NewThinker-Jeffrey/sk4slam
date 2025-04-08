
#include "sk4slam_geometry/fundamental_matrix.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_geometry/homography_matrix.h"
#include "sk4slam_geometry/third_party/colmap/estimators/fundamental_matrix.h"
#include "sk4slam_geometry/utils.h"
#include "sk4slam_math/matrix.h"
// #include "sk4slam_math/sac.h"  // CombinationSampler

namespace sk4slam {

std::vector<Eigen::Matrix3d> FundamentalMatrix::solveWith7Points(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  std::vector<Eigen::Matrix3d> Fs =
      sk4slam_colmap::FundamentalMatrixSevenPointEstimator::Estimate(
          Xs_2d, Xprimes_2d);

  // The solutions of 7-point (essential-matrix) algorithm should be exact.
  if (selected_indices.size() == 7) {
    for (const Eigen::Matrix3d& F : Fs) {
      auto errs = FundamentalMatrix::computeSquaredAlgebraicErrors(
          F, selected_indices, point_pairs);
      // LOGA("Check the exactness of 7-point algorithm: errors: %s",
      //      toStr(errs, sqrt, Precision(6)).c_str());
      for (const auto& err : errs) {
        ASSERT(std::abs(err) < 1e-10);
      }
    }
  }
  return Fs;
}

std::vector<Eigen::Matrix3d> FundamentalMatrix::solveWith8Points(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  return sk4slam_colmap::FundamentalMatrixEightPointEstimator::Estimate(
      Xs_2d, Xprimes_2d);
}

std::vector<Eigen::Matrix3d> FundamentalMatrix::solveWithKnownH(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs, const Eigen::Matrix3d& H) {
  if (selected_indices.size() < 2) {
    return std::vector<Eigen::Matrix3d>();
  }

  std::vector<Eigen::Vector3d> Xs, Xprimes;

  convertSelectedPointPairsToHomogeneousXsAndXprimes(
      selected_indices, point_pairs, &Xs, &Xprimes);
  ASSERT(Xs.size() == Xprimes.size());
  ASSERT(Xs.size() == selected_indices.size());

  std::vector<Eigen::Matrix3d> Fs;
  Eigen::MatrixXd transposed_coffes(3, Xs.size());
  for (size_t i = 0; i < Xs.size(); ++i) {
    transposed_coffes.col(i) = (H * Xs[i]).cross(Xprimes[i]).normalized();
  }

  Eigen::Vector3d e_prime;
  if (Xs.size() == 2) {
    Eigen::Vector3d l0 = transposed_coffes.col(0);
    Eigen::Vector3d l1 = transposed_coffes.col(1);
    e_prime = l0.cross(l1);

    if (abs(e_prime.z()) < 1e-10) {
      // Degenerate case.
      return std::vector<Eigen::Matrix3d>();
    }

    // clang-format off

    // // For debugging.
    // e_prime /= e_prime.z();
    // Eigen::MatrixXd coffes = transposed_coffes.tranpose();
    // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(coffes.leftCols<2>());
    // Eigen::Vector3d e_prime_qr << qr.solve(-coffes.col(2)), 1.0;
    // LOGI("e_prime_qr: %s, e_prime: %s, diff: %s",
    //      toStr(e_prime_qr.transpose()).c_str(),
    //      toStr(e_prime.transpose()).c_str(),
    //      toStr((e_prime_qr - e_prime).transpose()).c_str());

    // clang-format on
  } else {
    Eigen::MatrixXd coffes = transposed_coffes.transpose();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(coffes.leftCols<2>());
    if (qr.rank() < 2) {
      // Degenerate case.
      return std::vector<Eigen::Matrix3d>();
    }
    e_prime << qr.solve(-coffes.col(2)), 1.0;
  }
  Eigen::Matrix3d F = skew3(e_prime) * H;
  Fs.push_back(F);

  return Fs;
}

Eigen::Vector3d FundamentalMatrix::getEpipole(
    const Eigen::Matrix3d& F, const bool left_image = true) {
  // left_image=true: return e that `F * e = 0`
  // left_image=false: return e_prime that `e_prime^T * F = 0`

  if (left_image) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV);
    Eigen::Vector3d e = svd.matrixV().block<3, 1>(0, 2);
    return e;
  } else {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F.transpose(), Eigen::ComputeFullV);
    Eigen::Vector3d e_prime = svd.matrixV().block<3, 1>(0, 2);
    return e_prime;
  }
}

bool FundamentalMatrix::check_H_Degenerate(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs,
    const Eigen::Matrix<double, 3, 3>& F, const double err_thr,
    Eigen::Matrix<double, 3, 3>* H, std::vector<size_t>* h_sample_indices,
    size_t points_used_for_initial_h) {
  ASSERT(selected_indices.size() == 7 || selected_indices.size() == 8);
  ASSERT(H != nullptr);
  ASSERT(h_sample_indices != nullptr);
  ASSERT(
      points_used_for_initial_h == 3 || points_used_for_initial_h == 4 ||
      points_used_for_initial_h == 5);

  std::vector<Eigen::Vector3d> Xs, Xprimes;

  convertSelectedPointPairsToHomogeneousXsAndXprimes(
      selected_indices, point_pairs, &Xs, &Xprimes);
  ASSERT(Xs.size() == Xprimes.size());
  ASSERT(Xs.size() == selected_indices.size());

  h_sample_indices->reserve(Xs.size());

  static const std::vector<std::vector<size_t>> triplets = {
      {0, 1, 2}, {3, 4, 5}, {0, 1, 6}, {3, 4, 6}, {2, 5, 6}};

  static const std::vector<std::vector<size_t>> quadruplets = {
      {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4},
      {1, 2, 3, 4}, {0, 1, 4, 5}, {0, 2, 4, 5}, {0, 3, 4, 5},
      {1, 2, 4, 5}, {1, 3, 4, 5}, {2, 3, 4, 5}};

  // static std::vector<std::vector<size_t>> quintuplets;
  // if (quintuplets.empty()) {
  //   quintuplets.reserve(21);
  //   CombinationSampler sampler;
  //   for (size_t i=0; i<21; i++) {
  //     quintuplets.emplace_back(sampler.sample(5, 7));
  //   }
  //   // Ensure we have no more combinations.
  //   ASSERT(sampler.sample(5, 7).empty());
  // }
  // ASSERT(quintuplets.size() == 21);

  static const std::vector<std::vector<size_t>> quintuplets = {
      {0, 1, 2, 3, 4}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 6}, {0, 1, 2, 4, 5},
      {0, 1, 2, 4, 6}, {0, 1, 2, 5, 6}, {0, 1, 3, 4, 5}, {0, 1, 3, 4, 6},
      {0, 1, 3, 5, 6}, {0, 1, 4, 5, 6}, {0, 2, 3, 4, 5}, {0, 2, 3, 4, 6},
      {0, 2, 3, 5, 6}, {0, 2, 4, 5, 6}, {0, 3, 4, 5, 6}, {1, 2, 3, 4, 5},
      {1, 2, 3, 4, 6}, {1, 2, 3, 5, 6}, {1, 2, 4, 5, 6}, {1, 3, 4, 5, 6},
      {2, 3, 4, 5, 6}};

  const std::vector<std::vector<size_t>>* p_initial_subsets = nullptr;

  if (points_used_for_initial_h == 3) {
    p_initial_subsets = &triplets;
  } else if (points_used_for_initial_h == 4) {
    p_initial_subsets = &quadruplets;
  } else if (points_used_for_initial_h == 5) {
    p_initial_subsets = &quintuplets;
  }
  ASSERT(p_initial_subsets != nullptr);

  const size_t min_degenerate_samples = Xs.size() - 2;

  auto solve_Hs = [&](const std::vector<size_t>& subset) {
    LOGA(
        BLUE
        "FundamentalMatrix::check_H_Degenerate(): Estimate "
        "the initial H with the subset %s" RESET,
        toStr(subset).c_str());

    if (subset.size() == 3) {
      return HomographyMatrix::solveWithKnownF(
          {selected_indices[subset[0]], selected_indices[subset[1]],
           selected_indices[subset[2]]},
          point_pairs, F);
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

  auto evaluate_H = [&]() {
    std::vector<double> errs(selected_indices.size());
    std::vector<int> is;
    h_sample_indices->clear();
    h_sample_indices->reserve(Xs.size());
    is.reserve(Xs.size());
    errs.resize(Xs.size());
    for (size_t i = 0; i < Xs.size(); i++) {
      Eigen::Vector3d xi = (*H) * Xs[i];
      xi /= xi.z();
      double err = (xi - Xprimes[i]).squaredNorm();
      errs[i] = err;
      // LOGD("err: %f, err_thr: %f", err, err_thr);
      if (err < err_thr) {
        h_sample_indices->push_back(selected_indices[i]);
        is.push_back(i);
      }
    }
    LOGA(
        BLUE "FundamentalMatrix::check_H_Degenerate(): errs = %s" RESET,
        toStr(errs, sqrt).c_str());
    LOGA(
        BLUE
        "FundamentalMatrix::check_H_Degenerate(): inliers "
        "ids = %s" RESET,
        toStr(is).c_str());
    return h_sample_indices->size() >= min_degenerate_samples;
  };

  const std::vector<std::vector<size_t>>& initial_subsets = *p_initial_subsets;

  for (const auto& subset : initial_subsets) {
    std::vector<Eigen::Matrix3d> Hs = solve_Hs(subset);
    if (Hs.size() == 0) {
      // LOGA("FundamentalMatrix::check_H_Degenerate(): Hs.size() == 0!");
      continue;
    }
    ASSERT(Hs.size() == 1);
    *H = Hs[0];

    if (evaluate_H()) {
      return true;
    }
  }

  return false;
}

std::vector<double> FundamentalMatrix::computeSquaredAlgebraicErrors(
    const Eigen::Matrix<double, 3, 3>& F,
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<double> errs;
  errs.reserve(selected_indices.size());
  for (size_t i : selected_indices) {
    const auto& pair = point_pairs[i];
    double err =
        pair.second.homogeneous().transpose() * F * pair.first.homogeneous();
    errs.push_back(err * err);
  }
  return errs;
}

std::vector<double> FundamentalMatrix::computeSquaredSampsonErrors(
    const Eigen::Matrix<double, 3, 3>& F,
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<double> errs;
  errs.reserve(selected_indices.size());
  for (size_t i : selected_indices) {
    const auto& pair = point_pairs[i];
    errs.push_back(computeSquaredSampsonError(F, pair.first, pair.second));
  }
  return errs;
}

bool FundamentalMatrix::CeresOptimizer::optimizeWithCeres(
    Eigen::Matrix3d* F, int max_iter) const {
  CeresProblem problem;
  CeresParamBlock param_blk(&problem, toManifold(*F));
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
  *F = fromManifold(param_blk.GetGlobal());
  return true;
}

}  // namespace sk4slam
