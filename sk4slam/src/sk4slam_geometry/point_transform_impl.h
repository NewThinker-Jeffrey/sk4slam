#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_geometry/third_party/colmap/estimators/similarity_transform.h"
#include "sk4slam_geometry/third_party/colmap/estimators/translation_transform.h"
#include "sk4slam_geometry/utils.h"

namespace sk4slam {

template <int kDim, bool kEstimateScale>
std::vector<
    typename SimilarityTransformEstimator<kDim, kEstimateScale>::Parameter>
SimilarityTransformEstimator<kDim, kEstimateScale>::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  if (selected_indices.size() < kMinimalSampleSize) {
    return std::vector<Parameter>();
  }

  std::vector<Point> Xs, Xprimes;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, all_points, &Xs, &Xprimes);

  return sk4slam_colmap::SimilarityTransformEstimator<
      kDim, kEstimateScale>::Estimate(Xs, Xprimes);
}

template <int kDim, bool kEstimateScale>
std::vector<double> SimilarityTransformEstimator<kDim, kEstimateScale>::errors(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points, const Parameter& model) const {
  std::vector<Point> Xs, Xprimes;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, all_points, &Xs, &Xprimes);

  std::vector<double> errs;
  sk4slam_colmap::SimilarityTransformEstimator<kDim, kEstimateScale>::Residuals(
      Xs, Xprimes, model, &errs);
  return errs;
}

template <int kDim, bool kEstimateScale>
bool SimilarityTransformEstimator<kDim, kEstimateScale>::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
  auto res = compute(inlier_indices, all_points);
  if (res.empty()) {
    return false;
  } else {
    *param = res[0];
    return true;
  }
}

template <int kDim>
std::vector<typename TranslationTransformEstimator<kDim>::Parameter>
TranslationTransformEstimator<kDim>::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  if (selected_indices.size() < kMinimalSampleSize) {
    return std::vector<Parameter>();
  }

  std::vector<Point> Xs, Xprimes;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, all_points, &Xs, &Xprimes);

  return sk4slam_colmap::TranslationTransformEstimator<kDim>::Estimate(
      Xs, Xprimes);
}

template <int kDim>
std::vector<double> TranslationTransformEstimator<kDim>::errors(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points, const Parameter& model) const {
  std::vector<Point> Xs, Xprimes;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, all_points, &Xs, &Xprimes);

  std::vector<double> errs;
  sk4slam_colmap::TranslationTransformEstimator<kDim>::Residuals(
      Xs, Xprimes, model, &errs);
  return errs;
}

template <int kDim>
bool TranslationTransformEstimator<kDim>::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
  auto res = compute(inlier_indices, all_points);
  if (res.empty()) {
    return false;
  } else {
    *param = res[0];
    return true;
  }
}

}  // namespace sk4slam
