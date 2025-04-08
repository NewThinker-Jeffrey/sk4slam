
#include "sk4slam_geometry/two_view_geometry.h"

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

bool TwoViewGeometry::computePose(
    const std::vector<PointPair>& point_pairs, Eigen::Matrix3d* R,
    Eigen::Vector3d* t, const std::vector<size_t>& selected_indices_in,
    const double H_pure_rotation_thr) const {
  std::vector<size_t> tmp_selected_indices;
  const std::vector<size_t>* p_selected_indices = &selected_indices_in;
  if (p_selected_indices->empty()) {
    tmp_selected_indices.resize(point_pairs.size());
    std::iota(tmp_selected_indices.begin(), tmp_selected_indices.end(), 0);
    p_selected_indices = &tmp_selected_indices;
  }
  const std::vector<size_t>& selected_indices = *p_selected_indices;

  if (type == TwoViewGeometryType::ROTATION) {
    *R = matrix;
    *t = Eigen::Vector3d::Zero();
    return true;
  } else if (type == TwoViewGeometryType::HOMOGRAPHY) {
    Eigen::Matrix3d HTH = matrix.transpose() * matrix;
    double scale2 = HTH.trace() / 3.0;
    Eigen::Matrix3d normalized_HTH = HTH / scale2;
    Eigen::Matrix3d S = normalized_HTH - Eigen::Matrix3d::Identity();
    double rot_err = S.lpNorm<Eigen::Infinity>();
    LOGA(
        "TwoViewGeometry::computePose(): normalized_HTH:\n%s",
        toStr(normalized_HTH).c_str());
    LOGA(
        "TwoViewGeometry::computePose(): rot_err %f "
        "(pure_rotation_thr %f):\n",
        rot_err, H_pure_rotation_thr);
    if (rot_err < H_pure_rotation_thr) {
      *R = matrix / sqrt(scale2);
      if (R->determinant() < 0) {
        *R = -(*R);
      }
      // make sure R is a perfect rotation matrix.
      // *R = SO3d::expM(SO3d::logM(*R));
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(
          *R, Eigen::ComputeFullU | Eigen::ComputeFullV);
      *R = svd.matrixU() * svd.matrixV().transpose();

      *t = Eigen::Vector3d::Zero();
      return true;
    }

    Eigen::Vector3d n;
    std::vector<Eigen::Vector3d> points3D;
    HomographyMatrix::computePose(
        matrix, selected_indices, point_pairs, R, t, &n, &points3D);
    return true;
  } else if (type == TwoViewGeometryType::ESSENTIAL) {
    std::vector<Eigen::Vector3d> points3D;
    EssentialMatrix::computePose(
        matrix, selected_indices, point_pairs, R, t, &points3D);
    return true;
  } else {  // (type == TwoViewGeometryType::FUNDAMENTAL)
    return false;
  }
}

std::vector<double> TwoViewGeometry::computeSquaredAlgebraicErrors(
    const std::vector<PointPair>& point_pairs,
    const std::vector<size_t>& selected_indices_in) const {
  std::vector<size_t> tmp_selected_indices;
  const std::vector<size_t>* p_selected_indices = &selected_indices_in;
  if (p_selected_indices->empty()) {
    tmp_selected_indices.resize(point_pairs.size());
    std::iota(tmp_selected_indices.begin(), tmp_selected_indices.end(), 0);
    p_selected_indices = &tmp_selected_indices;
  }
  const std::vector<size_t>& selected_indices = *p_selected_indices;

  if (type == TwoViewGeometryType::ROTATION ||
      type == TwoViewGeometryType::HOMOGRAPHY) {
    return HomographyMatrix::computeSquaredAlgebraicErrors(
        matrix, selected_indices, point_pairs);
  } else {
    return FundamentalMatrix::computeSquaredAlgebraicErrors(
        matrix, selected_indices, point_pairs);
  }
}

std::vector<double> TwoViewGeometry::computeSquaredSampsonErrors(
    const std::vector<PointPair>& point_pairs,
    const std::vector<size_t>& selected_indices_in) const {
  std::vector<size_t> tmp_selected_indices;
  const std::vector<size_t>* p_selected_indices = &selected_indices_in;
  if (p_selected_indices->empty()) {
    tmp_selected_indices.resize(point_pairs.size());
    std::iota(tmp_selected_indices.begin(), tmp_selected_indices.end(), 0);
    p_selected_indices = &tmp_selected_indices;
  }
  const std::vector<size_t>& selected_indices = *p_selected_indices;

  if (type == TwoViewGeometryType::ROTATION ||
      type == TwoViewGeometryType::HOMOGRAPHY) {
    return HomographyMatrix::computeSquaredSampsonErrors(
        matrix, selected_indices, point_pairs);
  } else {
    return FundamentalMatrix::computeSquaredSampsonErrors(
        matrix, selected_indices, point_pairs);
  }
}

std::vector<double> TwoViewGeometryEstimator::errors(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points, const Parameter& model) const {
  if (model.type == TwoViewGeometryType::FUNDAMENTAL ||
      model.type == TwoViewGeometryType::ESSENTIAL) {
    return FundamentalMatrix::computeSquaredSampsonErrors(
        model.matrix, selected_indices, all_points);
  } else {  // model.type == TwoViewGeometryType::HOMOGRAPHY ||
            // model.type == TwoViewGeometryType::ROTATION

    // return HomographyMatrix::computeSquaredAlgebraicErrors(
    //     model.matrix, selected_indices, all_points);

    return HomographyMatrix::computeSquaredSampsonErrors(
        model.matrix, selected_indices, all_points);
  }
}

bool TwoViewGeometryEstimator::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
  Parameter& model = *param;
  if (model.type == TwoViewGeometryType::FUNDAMENTAL) {
    using Optimizer = FundamentalMatrix::SampsonErrorOptimizer;
    // using Optimizer = FundamentalMatrix::SquaredSampsonErrorOptimizer;
    return Optimizer(&inlier_indices, &all_points)
        .optimizeWithCeres(&model.matrix);
  } else if (model.type == TwoViewGeometryType::ESSENTIAL) {
    using Optimizer = EssentialMatrix::SampsonErrorOptimizer;
    // using Optimizer = EssentialMatrix::SquaredSampsonErrorOptimizer;
    return Optimizer(&inlier_indices, &all_points)
        .optimizeWithCeres(&model.matrix);
  } else {  // model.type == TwoViewGeometryType::HOMOGRAPHY ||
            // model.type == TwoViewGeometryType::ROTATION
    std::vector<Eigen::Matrix3d> Hs =
        HomographyMatrix::solveDLT(inlier_indices, all_points);
    if (Hs.size() == 0) {
      return false;
    } else {
      ASSERT(Hs.size() == 1);
      model.matrix = Hs[0];
      // TODO(jeffrey): We need a better way to handle the rotation case.
      if (model.type == TwoViewGeometryType::ROTATION) {
        if (model.matrix.determinant() < 0) {
          model.matrix *= -1;
        }
        double scale =
            sqrt((model.matrix * model.matrix.transpose()).trace() / 3.0);
        // model.matrix = SO3d::expM(SO3d::logM(model.matrix / scale));
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            model.matrix / scale, Eigen::ComputeFullU | Eigen::ComputeFullV);
        model.matrix = svd.matrixU() * svd.matrixV().transpose();
      }
      return true;
    }
  }
}

std::vector<TwoViewGeometryEstimator::Parameter>
TwoViewGeometryEstimator::getParametersFromMatrices(
    TwoViewGeometryType type, const std::vector<Eigen::Matrix3d>& matrices) {
  std::vector<Parameter> params;
  params.reserve(matrices.size());
  for (const auto& matrix : matrices) {
    params.emplace_back(type, matrix);
  }
  return params;
}

EpipolarDegensacScoring::Score EpipolarDegensacScoring::evaluate(
    const TwoViewGeometryEstimator& model, const TwoViewGeometry& param,
    const std::vector<TwoViewGeometry::PointPair>& all_points,
    const RansacOptions& sac_options, std::vector<size_t>* inliers) const {
  Score score;
  score.num_inliers = 0;
  score.err_sum = 0;
  if (param.type == TwoViewGeometryType::ROTATION ||
      param.type == TwoViewGeometryType::HOMOGRAPHY) {
    score.degenerated = true;
  } else {
    score.degenerated = false;
  }

  inliers->clear();
  inliers->reserve(all_points.size());

  std::vector<size_t> all_indices;
  all_indices.resize(all_points.size());
  std::iota(all_indices.begin(), all_indices.end(), 0);
  auto errs = model.errors(all_indices, all_points, param);
  ASSERT(errs.size() == all_points.size());

  for (size_t i = 0; i < all_points.size(); i++) {
    double err = errs[i];
    if (err > sac_options.error_thr) {
      continue;
    }
    score.err_sum += err;
    ++score.num_inliers;
    inliers->push_back(i);
  }
  ASSERT(score.num_inliers == inliers->size());
  return score;
}

// Compare the two scores (return true if score2 is strictly better).
bool EpipolarDegensacScoring::compare(
    const Score& score1, const Score& score2) const {
  double s1, s2;
  if (score1.degenerated) {
    s1 = degenerate_score_multiplier * score1.num_inliers + 1;
  } else {
    s1 = score1.num_inliers;
  }

  if (score2.degenerated) {
    s2 = degenerate_score_multiplier * score2.num_inliers + 1;
  } else {
    s2 = score2.num_inliers;
  }
  return s1 < s2;
}

bool PlaneAndParallaxEstimator::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
  ASSERT(param->type == TwoViewGeometryType::FUNDAMENTAL);
  std::vector<Eigen::Matrix3d> Fs =
      FundamentalMatrix::solveWithKnownH(inlier_indices, all_points, H_);
  if (Fs.empty()) {
    return false;
  }
  ASSERT(Fs.size() == 1);
  param->matrix = Fs[0];
  return true;
}

bool DegeneracyAwareFundamentalEstimator::handleDegeneracy(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points,
    const std::vector<size_t>& initial_inliers, Parameter* model) {
  if (model->type == TwoViewGeometryType::HOMOGRAPHY) {
    // This could only happen at the end of the RANSAC when the
    // best model is already a homography.
    return false;
  }

  ASSERT(model->type == TwoViewGeometryType::FUNDAMENTAL);

  if (!degen_params_.enable_degeneracy_check) {
    return false;
  }

  bool is_final = false;

  // 1. Check H-degenerate sample:
  TwoViewGeometry H(TwoViewGeometryType::HOMOGRAPHY);
  std::vector<size_t> h_sample_indices;
  if (selected_indices.empty()) {
    is_final = true;
    // This only happens at the end of the RANSAC. What we need to do is to
    // estimate the homography from the initial_inliers.
    std::vector<DataPoint> initial_inlier_point_pairs;
    for (size_t i : initial_inliers) {
      initial_inlier_point_pairs.push_back(all_points[i]);
    }
    RansacOptions tmp_estimator_options = degen_params_.ransac_options;
    tmp_estimator_options.local_opt_max_iter = 0;
    tmp_estimator_options.final_opt_max_iter = 0;
    tmp_estimator_options.check_step_degeneracy = false;
    tmp_estimator_options.check_final_degeneracy = false;
    // tmp_estimator_options.min_inlier_ratio =
    // degen_params_.dominant_h_percent;
    auto h_report = HomographyEstimator::ransac(
        initial_inlier_point_pairs, tmp_estimator_options);
    H = h_report.param;
    ASSERT(H.type == TwoViewGeometryType::HOMOGRAPHY);
  } else if (!FundamentalMatrix::check_H_Degenerate(
                 selected_indices, all_points, model->matrix,
                 degen_params_.h_degen_err_thr, &(H.matrix), &h_sample_indices,
                 degen_params_.points_used_for_initial_h)) {
    LOGA(
        BLUE
        "DegeneracyAwareFundamentalEstimator::handleDegeneracy(): NOT "
        "degenerate! (F.inliers = %d)" RESET,
        initial_inliers.size());
    return false;
  }

  // 2. Check dominant H:

  // locally opt H
  RansacOptions h_estimator_options;
  std::vector<size_t> h_inliers;

  // TODO(jeffrey):
  //     Consider and test: Do we really need a two-round LO ?
  if (!is_final && degen_params_.h_degen_err_thr > 0.0) {
    // first round: use the looser err threshold
    h_estimator_options.error_thr = degen_params_.h_degen_err_thr;
    Ransac<HomographyEstimator> h_estimator(h_estimator_options);
    h_estimator.evaluateAndLocalOptimize(
        all_points, h_sample_indices, &H, &h_inliers,
        nullptr,  // score
        degen_params_.h_max_opt_iter);
  }

  {
    // second round: use the same err threshold as the ransac
    h_estimator_options.error_thr = degen_params_.ransac_options.error_thr;
    Ransac<HomographyEstimator> h_estimator(h_estimator_options);
    h_estimator.evaluateAndLocalOptimize(
        all_points, h_sample_indices, &H, &h_inliers,
        nullptr,  // score
        degen_params_.h_max_opt_iter);
  }

  // NOTE: `initial_inliers` and `h_inliers` are both sorted array.
  std::vector<size_t> common_inliers;
  std::set_intersection(
      initial_inliers.begin(), initial_inliers.end(), h_inliers.begin(),
      h_inliers.end(), std::back_inserter(common_inliers));
  if (common_inliers.size() <
      degen_params_.dominant_h_percent * initial_inliers.size()) {
    LOGA(
        BLUE
        "DegeneracyAwareFundamentalEstimator::handleDegeneracy(): NOT a "
        "dominant H! (H.%d %d/%d)" RESET,
        h_inliers.size(), common_inliers.size(), initial_inliers.size());
    return false;  // Not a dominant H.
  }

  // 3. Re-estimate F: (with off-H points)
  std::vector<DataPoint> off_h_points;
  off_h_points.reserve(all_points.size() - h_inliers.size());

  // NOTE: `h_inliers` is a sorted array.
  size_t j = 0;
  for (const size_t i : h_inliers) {
    while (j < i) {
      off_h_points.push_back(all_points[j]);
      ++j;
    }
    ASSERT(i == j);
    ++j;
  }
  while (j < all_points.size()) {
    off_h_points.push_back(all_points[j]);
    ++j;
  }

  if (off_h_points.size() < degen_params_.min_off_h_points) {
    // `all_points` is a degenerate sample, output the degenerated model.
    *model = H;
    LOGA(
        BLUE
        "DegeneracyAwareFundamentalEstimator::handleDegeneracy(): Output H "
        "(degenerate)! (H.%d vs F.%d)" RESET,
        h_inliers.size(), initial_inliers.size());
    return true;
  }

  RansacOptions f_estimator_options = degen_params_.ransac_options;

  // int min_inliers_off_h =
  //     all_points.size() *
  //       degen_params_.ransac_options.initial_min_inlier_ratio -
  //     h_inliers.size();
  int min_inliers_off_h =
      0.9 * (initial_inliers.size() - common_inliers.size());

  min_inliers_off_h =
      std::max(min_inliers_off_h, degen_params_.min_off_h_points);
  f_estimator_options.initial_min_inlier_ratio =
      static_cast<double>(min_inliers_off_h) / off_h_points.size();
  Ransac<PlaneAndParallaxEstimator> f_estimator(
      f_estimator_options, PlaneAndParallaxEstimator(H.matrix));

  auto report = f_estimator.solve(off_h_points);

  // clang-format off
  if (
      report.inliers.size() < min_inliers_off_h
      // report.inliers.size() < degen_params_.min_off_h_points
      //   ||
      // report.inliers.size() <
      //     // off-H inliers of the new F are less than that of the initial F.
      //    initial_inliers.size() - common_inliers.size()
      //   ||
      // (report.inliers.size() + h_inliers.size()) *
      //         degen_params_.dominant_h_percent <=
      //     h_inliers.size()
      ) {
    // clang-format on

    // H is still dominamt.
    *model = H;
    LOGA(
        BLUE
        "DegeneracyAwareFundamentalEstimator::handleDegeneracy(): Output H "
        "(degenerate 2)! min_inliers_off_h=%d "
        "(newOffH.%d H.%d (common.%d) vs oldF.%d)" RESET,
        min_inliers_off_h, report.inliers.size(), h_inliers.size(),
        common_inliers.size(), initial_inliers.size());
    return true;
  }

  // 4. Continue the normal RANSAC process:
  if (report.inliers.size() + h_inliers.size() <= initial_inliers.size()) {
    // The re-estimated F is not better.
    LOGA(
        BLUE
        "DegeneracyAwareFundamentalEstimator::handleDegeneracy(): Keep the "
        "original F! min_inliers_off_h=%d "
        "(newOffH.%d H.%d (common.%d) vs oldF.%d)" RESET,
        min_inliers_off_h, report.inliers.size(), h_inliers.size(),
        common_inliers.size(), initial_inliers.size());
    return false;
  }

  *model = report.param;  // update the model.
  LOGA(
      BLUE
      "DegeneracyAwareFundamentalEstimator::handleDegeneracy(): Restimate F! "
      "min_inliers_off_h=%d "
      "(newOffH.%d H.%d (common.%d) vs oldF.%d)" RESET,
      min_inliers_off_h, report.inliers.size(), h_inliers.size(),
      common_inliers.size(), initial_inliers.size());
  return true;
}

bool DegeneracyAwareEssentialEstimator::handleDegeneracy(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points,
    const std::vector<size_t>& initial_inliers, Parameter* model) {
  if (  // model->type == TwoViewGeometryType::HOMOGRAPHY ||
      model->type == TwoViewGeometryType::ROTATION) {
    // This could only happen at the end of the RANSAC when the
    // best model is already a homography.
    return false;
  }

  ASSERT(model->type == TwoViewGeometryType::ESSENTIAL);

  if (!degen_params_.enable_degeneracy_check) {
    return false;
  }

  bool is_final = false;

  // 1. Check H-degenerate sample:
  TwoViewGeometry H(TwoViewGeometryType::HOMOGRAPHY);
  std::vector<size_t> h_sample_indices;
  if (selected_indices.empty()) {
    is_final = true;
    // This only happens at the end of the RANSAC. What we need to do is to
    // estimate the homography from the initial_inliers.
    std::vector<DataPoint> initial_inlier_point_pairs;
    for (size_t i : initial_inliers) {
      initial_inlier_point_pairs.push_back(all_points[i]);
    }
    RansacOptions tmp_estimator_options = degen_params_.ransac_options;
    tmp_estimator_options.local_opt_max_iter = 0;
    tmp_estimator_options.final_opt_max_iter = 0;
    tmp_estimator_options.check_step_degeneracy = false;
    tmp_estimator_options.check_final_degeneracy = false;

    auto h_report = HomographyEstimator::ransac(
        initial_inlier_point_pairs, tmp_estimator_options);
    H = h_report.param;
    ASSERT(H.type == TwoViewGeometryType::HOMOGRAPHY);
  } else if (!EssentialMatrix::check_H_Degenerate(
                 selected_indices, all_points, model->matrix,
                 degen_params_.h_degen_err_thr, &(H.matrix), &h_sample_indices,
                 degen_params_.points_used_for_initial_h)) {
    LOGA(
        BLUE
        "DegeneracyAwareEssentialEstimator::handleDegeneracy(): NOT "
        "degenerate! (E.inliers = %d)" RESET,
        initial_inliers.size());
    return false;
  }

  // 2. Check dominant H:

  // locally opt H
  RansacOptions h_estimator_options;
  std::vector<size_t> h_inliers;

  // TODO(jeffrey):
  //     Consider and test: Do we really need a two-round LO ?
  if (!is_final && degen_params_.h_degen_err_thr > 0.0) {
    // first round: use the looser err threshold
    h_estimator_options.error_thr = degen_params_.h_degen_err_thr;
    Ransac<HomographyEstimator> h_estimator(h_estimator_options);
    h_estimator.evaluateAndLocalOptimize(
        all_points, h_sample_indices, &H, &h_inliers,
        nullptr,  // score
        degen_params_.h_max_opt_iter);
  }

  {
    // second round: use the same err threshold as the ransac
    h_estimator_options.error_thr = degen_params_.ransac_options.error_thr;
    Ransac<HomographyEstimator> h_estimator(h_estimator_options);
    h_estimator.evaluateAndLocalOptimize(
        all_points, h_sample_indices, &H, &h_inliers,
        nullptr,  // score
        degen_params_.h_max_opt_iter);
  }

  // NOTE: `initial_inliers` and `h_inliers` are both sorted array.
  std::vector<size_t> common_inliers;
  std::set_intersection(
      initial_inliers.begin(), initial_inliers.end(), h_inliers.begin(),
      h_inliers.end(), std::back_inserter(common_inliers));
  if (common_inliers.size() <
      degen_params_.dominant_h_percent * initial_inliers.size()) {
    LOGA(
        BLUE
        "DegeneracyAwareEssentialEstimator::handleDegeneracy(): NOT a "
        "dominant H! (H.%d %d/%d)" RESET,
        h_inliers.size(), common_inliers.size(), initial_inliers.size());
    return false;  // Not a dominant H.
  }

  // 3. Check pure rotation:

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  LOGA("estimated_H:\n%s", toStr(H.matrix / H.matrix(2, 2)).c_str());
  H.computePose(all_points, &R, &t, h_inliers, degen_params_.pure_rotation_thr);
  bool is_pure_rotation = (t.norm() < 1e-6);

  // 4. Continue the normal RANSAC process:
  if (is_pure_rotation) {
    model->type = TwoViewGeometryType::ROTATION;
    model->matrix = R;
    LOGA(
        BLUE
        "DegeneracyAwareEssentialEstimator::handleDegeneracy(): Output R "
        "(degenerate)! (H.%d (common.%d) vs E.%d)" RESET,
        h_inliers.size(), common_inliers.size(), initial_inliers.size());
    return true;
  } else {
    const bool always_update_E_if_H_is_dominant = true;
    // const bool always_update_E_if_H_is_dominant = false;
    if (always_update_E_if_H_is_dominant ||
        h_inliers.size() > initial_inliers.size()) {
      // clang-format off

      // model->type = TwoViewGeometryType::HOMOGRAPHY;
      // model->matrix = H.matrix;
      // LOGA(
      //     BLUE
      //     "DegeneracyAwareEssentialEstimator::handleDegeneracy(): Output H!"
      //     " (degenerate)! (H.%d (common.%d) vs E.%d)" RESET,
      //     h_inliers.size(), common_inliers.size(), initial_inliers.size());
      // return true;

      model->matrix = skew3(t) * R;  // we find a better model from H.
      LOGA(
          BLUE
          "DegeneracyAwareEssentialEstimator::handleDegeneracy(): "
          "Restimate E! "
          "(H.%d (common.%d) vs E.%d)" RESET,
          h_inliers.size(), common_inliers.size(), initial_inliers.size());
      return true;

      // clang-format on
    } else {
      LOGA(
          BLUE
          "DegeneracyAwareEssentialEstimator::handleDegeneracy(): Keep the "
          "original E! (H.%d (common.%d) vs E.%d)" RESET,
          h_inliers.size(), common_inliers.size(), initial_inliers.size());
      return false;
    }
  }
}

std::vector<TwoViewGeometryEstimator::Parameter>
Essential2PointEstimator::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  std::vector<Eigen::Matrix3d> Es = EssentialMatrix::solveWithKnownRotation(
      selected_indices, all_points, known_rotation_);
  if (!Es.empty()) {
    // If we have a solution, return it.
    ASSERT(Es.size() == 1);
    return getParametersFromMatrices(TwoViewGeometryType::ESSENTIAL, Es);
  } else {
    // Otherwise, the essential matrix is degenerate and motion is pure
    // rotation.
    if (selected_indices.size() >= 2) {
      return {TwoViewGeometry(TwoViewGeometryType::ROTATION, known_rotation_)};
    } else {
      return std::vector<Parameter>();
    }
  }
}

bool Essential2PointEstimator::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
#if 0
  if (param->type == TwoViewGeometryType::ESSENTIAL) {
    std::vector<Eigen::Matrix3d> Es = EssentialMatrix::solveWithKnownRotation(
        inlier_indices, all_points, known_rotation_);
    if (!Es.empty()) {
      // If we have a solution, return it.
      ASSERT(Es.size() == 1);
      *param = TwoViewGeometry(TwoViewGeometryType::ESSENTIAL, Es[0]);
      return true;
    } else {
      return false;  // If the essential matrix is degenerate, we cannot
                     // optimize it.
    }
  } else {
    ASSERT(param->type == TwoViewGeometryType::ROTATION);
    return false;  // If the geometry is pure rotation, there is nothing to
                   // optimize.
  }
#else
  return TwoViewGeometryEstimator::localOptimize(
      inlier_indices, all_points,
      param);  // Ignore the rotation constraint when optimizing.
#endif
}

}  // namespace sk4slam
