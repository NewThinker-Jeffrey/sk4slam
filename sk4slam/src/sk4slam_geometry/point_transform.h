#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_math/sac.h"

namespace sk4slam {

template <int kDim, bool kEstimateScale = true>
class SimilarityTransformEstimator {
  // sR * X + t = Xprime
  //
  // pair.first = X
  // pair.second = Xprime

 public:
  static RansacOptions defaultRansacOptions() {
    return RansacOptions(
        (0.1 * 0.1),  // error_thr
        0.999,        // confidence
        1000,         // max_iter
        0.0,          // initial_inlier_ratio_thr
        0,            // local_opt_max_iter
                      // ^ None zero local_opt_max_iter may
                      // ^ significantly slow down the computation
        1             // final_opt_max_iter
    );                // NOLINT
  }

  using Point = Eigen::Matrix<double, kDim, 1>;

  using DataPoint = std::pair<Point, Point>;

  using Parameter = Eigen::Matrix<double, kDim, kDim + 1>;

  static constexpr int kMinimalSampleSize = kDim;

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const;

  std::vector<double> errors(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points, const Parameter& model) const;

  static typename Ransac<SimilarityTransformEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<SimilarityTransformEstimator>(options).solve(point_pairs);
  }

 public:  // specialized properties.
  // Do LO with all the inliers (needed by LO-RANSAC framework).
  // `param` shuold already hold a reasonable initial guess for the model
  // parameters.
  bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;
};

template <int kDim>
using EuclideanTransformEstimator = SimilarityTransformEstimator<kDim, false>;

template <int kDim>
class TranslationTransformEstimator {
 public:
  // X + t = Xprime
  //
  // pair.first = X
  // pair.second = Xprime

  static RansacOptions defaultRansacOptions() {
    return RansacOptions(
        (0.1 * 0.1),  // error_thr
        0.999,        // confidence
        1000,         // max_iter
        0.0,          // initial_inlier_ratio_thr
        0,            // local_opt_max_iter
                      // ^ None zero local_opt_max_iter may
                      // ^ significantly slow down the computation
        1             // final_opt_max_iter
    );                // NOLINT
  }

  using Point = Eigen::Matrix<double, kDim, 1>;

  using DataPoint = std::pair<Point, Point>;

  using Parameter = Eigen::Matrix<double, kDim, 1>;

  static constexpr int kMinimalSampleSize = 1;

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const;

  std::vector<double> errors(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points, const Parameter& model) const;

  static typename Ransac<TranslationTransformEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<TranslationTransformEstimator>(options).solve(point_pairs);
  }

 public:  // specialized properties.
  // Do LO with all the inliers (needed by LO-RANSAC framework).
  // `param` shuold already hold a reasonable initial guess for the model
  // parameters.
  bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;
};

}  // namespace sk4slam

#include "sk4slam_geometry/point_transform_impl.h"
