#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/time.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_geometry/third_party/colmap/geometry/homography_matrix.h"  // HomographyMatrixFromPose()
#include "sk4slam_geometry/two_view_geometry.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"
#include "sk4slam_math/sac.h"

namespace sk4slam {

static constexpr int kTestFinalOptMaxIter = 1;
// static constexpr int kTestFinalOptMaxIter = 0;

// The tests become less stable as the noise level increases.
// (Increasing the values of `min_off_h_points` and `pure_rotation_thr`
//  might be necessary under high noise levels)
//
// clang-format off
static const double noise_stddev =
    // 0.25 / TwoViewGeometryEstimator::kDefatultFocal;
    0.33 / TwoViewGeometryEstimator::kDefatultFocal;
    // 0.50 / TwoViewGeometryEstimator::kDefatultFocal;
    // 1.00 / TwoViewGeometryEstimator::kDefatultFocal;
// clang-format on

inline RansacOptions defaultRansacOptions() {
  RansacOptions ransac_options =
      TwoViewGeometryEstimator::defaultRansacOptions();
  ransac_options.error_thr = noise_stddev * noise_stddev * 3.0 * 3.0;
  ransac_options.final_opt_max_iter = kTestFinalOptMaxIter;
  return ransac_options;
}

std::string formatTwoViewGeometryErrors(
    const std::vector<double>& squared_errs) {
  return toStr(squared_errs, sqrt);
  // Oss oss;
  // for (size_t i = 0; i < squared_errs.size(); ++i) {
  //   oss << sqrt(squared_errs[i]) << " ";
  // }
  // return oss.str();
}

template <typename RansacReport>
void printTwoViewGeometryRansacReport(
    const RansacReport& report,
    const std::vector<TwoViewGeometryEstimator::DataPoint>& point_pairs) {
  LOGI(
      "RANSAC output model: %s\n%s", toStr(report.param.type).c_str(),
      toStr(
          report.param.matrix /
          (report.param.type == TwoViewGeometryType::HOMOGRAPHY
               ? report.param.matrix(2, 2)
               : 1.0))
          .c_str());
  LOGI(
      "RANSAC inlier_ratio: %f  (%d out of %d)", report.inlier_ratio,
      report.inliers.size(), report.n_total);
  LOGI("RANSAC iter: %d", report.iter);

  auto report_outliers = report.getOutliers();
  LOGI(
      "RANSAC outliers [%d]: %s", report_outliers.size(),
      toStr(report_outliers).c_str());
  LOGI(
      "RANSAC inliers [%d]: %s", report.inliers.size(),
      toStr(report.inliers).c_str());

  ASSERT_FALSE(report.inliers.empty());
  std::string inlier_algebraic_errs_str = formatTwoViewGeometryErrors(
      report.param.computeSquaredAlgebraicErrors(point_pairs, report.inliers));
  std::string inlier_sampson_errs_str = formatTwoViewGeometryErrors(
      report.param.computeSquaredSampsonErrors(point_pairs, report.inliers));
  std::string outlier_sampson_errs_str;
  if (!report_outliers.empty()) {
    outlier_sampson_errs_str = formatTwoViewGeometryErrors(
        report.param.computeSquaredSampsonErrors(point_pairs, report_outliers));
  }
  double inlier_ave_err = sqrt(
      report.param.computeSquaredSampsonErrorsAve(point_pairs, report.inliers));
  LOGI("Averaged Inlier Sampson Error: %f", inlier_ave_err);
  LOGI("Inliers Algebra Errors: %s", inlier_algebraic_errs_str.c_str());
  LOGI("Inliers Sampson Errors: %s", inlier_sampson_errs_str.c_str());
  LOGI("Outliers Sampson Errors: %s", outlier_sampson_errs_str.c_str());
}

template <typename RansacReport>
void checkPose(
    const RansacReport& report,
    const std::vector<TwoViewGeometryEstimator::DataPoint>& point_pairs,
    const Eigen::Isometry3d& real_T_C2_C1) {
  Eigen::Matrix3d R_C2_C1;
  Eigen::Vector3d t_C2_C1;
  bool rotation_only = (real_T_C2_C1.translation().squaredNorm() < 1e-6);
  ASSERT_FALSE(report.inliers.empty());
  ASSERT_TRUE(report.param.computePose(
      point_pairs, &R_C2_C1, &t_C2_C1, report.inliers));

  LOGI("Real R_C2_C1:\n%s", toStr(real_T_C2_C1.rotation()).c_str());
  LOGI("Recovered R_C2_C1:\n%s", toStr(R_C2_C1).c_str());

  if (!rotation_only) {
    LOGI(
        "Real t_C2_C1: %s",
        toStr(real_T_C2_C1.translation().transpose().normalized()).c_str());
    LOGI(
        "Recovered t_C2_C1: %s",
        toStr(t_C2_C1.transpose().normalized()).c_str());
  } else {
    LOGI(
        "Real t_C2_C1: %s",
        toStr(real_T_C2_C1.translation().transpose()).c_str());
    LOGI("Recovered t_C2_C1: %s", toStr(t_C2_C1.transpose()).c_str());
  }

  Eigen::Matrix3d rotation_err = real_T_C2_C1.rotation().transpose() * R_C2_C1 -
                                 Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_err;
  if (!rotation_only) {
    translation_err =
        real_T_C2_C1.translation().normalized() - t_C2_C1.normalized();
  } else {
    translation_err = real_T_C2_C1.translation() - t_C2_C1;
  }

  LOGI(
      "Recovered rotation_err(%f):\n%s", rotation_err.squaredNorm(),
      toStr(rotation_err).c_str());
  LOGI(
      "Recovered translation_err(%f): %s", translation_err.squaredNorm(),
      toStr(translation_err.transpose()).c_str());

  ASSERT_NEAR(rotation_err.squaredNorm(), 0, 1e-2);
  ASSERT_NEAR(translation_err.squaredNorm(), 0, 1e-2);
}

template <typename RansacReport>
void checkRansac(
    const RansacReport& report,
    const std::vector<TwoViewGeometryEstimator::DataPoint>& point_pairs,
    double inlier_ratio_lower_bound, double inlier_ratio_upper_bound = 1.0001) {
  ASSERT_FALSE(report.inliers.empty());

  double ransac_err_thr = sqrt(report.ransac_options.error_thr);
  // double ave_err_thr = 3.0 * sqrt(report.ransac_options.error_thr /
  // report.inliers.size());
  double ave_err_thr = ransac_err_thr;

  double inlier_ave_err = sqrt(
      report.param.computeSquaredSampsonErrorsAve(point_pairs, report.inliers));
  LOGI(
      "inlier_ave_err = %f, ave_err_thr = %f (ransac_err_thr=%f)",
      inlier_ave_err, ave_err_thr, ransac_err_thr);
  ASSERT_LE(inlier_ave_err, ave_err_thr);

  ASSERT_LE(report.inlier_ratio, inlier_ratio_upper_bound);
  ASSERT_GE(report.inlier_ratio, inlier_ratio_lower_bound);
}

inline const std::vector<TwoViewGeometryEstimator::DataPoint>&
getSimpleNonCoplanarPointPairs() {
  // The data was borrowed from the COLMAP test code.
  // 12 point-pairs in total, with the last two points being outliers.
  static const double points1_raw[] = {
      0.4964, 1.0577, 0.3650,  -0.0919, -0.5412, 0.0159, -0.5239, 0.9467,
      0.3467, 0.5301, 0.2797,  0.0012,  -0.1986, 0.0460, -0.1622, 0.5347,
      0.0796, 0.2379, -0.3946, 0.7969,  0.2,     0.7,    0.6,     0.3};

  static const double points2_raw[] = {
      0.7570, 2.7340, 0.3961,  0.6981, -0.6014, 0.7110, -0.7385, 2.2712,
      0.4177, 1.2132, 0.3052,  0.4835, -0.2171, 0.5057, -0.2059, 1.1583,
      0.0946, 0.7013, -0.6236, 3.0253, 0.5,     0.9,    0.9,     0.2};

  static const size_t num_points = 12;
  static std::vector<TwoViewGeometryEstimator::DataPoint> point_pairs;
  if (point_pairs.empty()) {
    point_pairs.resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      point_pairs[i].first =
          Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
      point_pairs[i].second =
          Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
    }
  }
  return point_pairs;
}

inline const std::vector<TwoViewGeometryEstimator::DataPoint>
getCoplanarPointPairs(
    Eigen::Isometry3d* output_T_C2_C1, Eigen::Matrix3d* output_H,
    int n_coplanar = 40, int n_non_coplanar = 0, int n_outliers = 0) {
  Eigen::Isometry3d T_G_C2 = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_G_C1;
  T_G_C1.linear() = SO3d::expM(Eigen::Vector3d(0.2, -0.1, 0.1));
  T_G_C1.translation() = Eigen::Vector3d(0.3, 0.2, 0.4);
  Eigen::Isometry3d T_C1_G = T_G_C1.inverse();
  Eigen::Isometry3d T_C2_G = T_G_C2.inverse();
  Eigen::Matrix3d R_G_Plane = SO3d::expM(Eigen::Vector3d(0.1, 0.05, 0.05));

  auto dist2d = MultivariateUniformDistribution<double>::standard(2);
  double plane_d = 2.0;
  std::vector<Eigen::Vector3d> points_3d;
  // Add coplanar points.
  for (size_t i = 0; i < n_coplanar; ++i) {
    points_3d.push_back(R_G_Plane * (plane_d * dist2d().homogeneous()));
  }
  // LOGI("T_C1_C2:\n%s", toStr((T_C1_G * T_G_C2).matrix()).c_str());
  // LOGI("T_C2_C1:\n%s", toStr((T_C2_G * T_G_C1).matrix()).c_str());
  *output_T_C2_C1 = T_C2_G * T_G_C1;
  *output_H = sk4slam_colmap::HomographyMatrixFromPose(
      Eigen::Matrix3d::Identity(),  // K1
      Eigen::Matrix3d::Identity(),  // K2
      T_G_C1.linear(), T_G_C1.translation(), -R_G_Plane.col(2), plane_d);

  // Add non-coplanar points.
  for (size_t i = 0; i < n_non_coplanar; ++i) {
    double r = UniformRealDistribution<double>(0.5, 1.5)();
    points_3d.push_back(
        R_G_Plane *
        (plane_d * dist2d().homogeneous() + Eigen::Vector3d(0, 0, r)));
  }

  std::vector<TwoViewGeometryEstimator::DataPoint> point_pairs;
  auto dist_noise = MultivariateNormalDistribution<double>::standard(2);
  for (size_t i = 0; i < points_3d.size(); ++i) {
    point_pairs.emplace_back(
        (T_C1_G * points_3d[i]).hnormalized() + noise_stddev * dist_noise(),
        (T_C2_G * points_3d[i]).hnormalized() + noise_stddev * dist_noise());
  }

  if (n_outliers > 0) {
    std::vector<size_t> outliers =
        RandomSampler().sample(n_outliers, point_pairs.size());
    for (size_t i : outliers) {
      double r = UniformRealDistribution<double>(25.0, 50.0)() /
                 TwoViewGeometryEstimator::kDefatultFocal;
      // point_pairs[i].first += dist_noise().normalized() * r;
      point_pairs[i].second += dist_noise().normalized() * r;
    }
  }

  return point_pairs;
}

inline const std::vector<TwoViewGeometryEstimator::DataPoint>
getPureRotationPointPairs(
    Eigen::Isometry3d* output_T_C2_C1, int n_total = 40, int n_outliers = 0) {
  Eigen::Matrix3d R_G_C1 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_G_C2;
  R_G_C2 = SO3d::expM(Eigen::Vector3d(0.2, -0.1, 0.1));
  Eigen::Matrix3d R_C1_G = R_G_C1.transpose();
  Eigen::Matrix3d R_C2_G = R_G_C2.transpose();

  auto dist2d = MultivariateUniformDistribution<double>::standard(2);
  double center_d = 2.0;
  std::vector<Eigen::Vector3d> points_3d;
  // Add n_total points.
  for (size_t i = 0; i < n_total; ++i) {
    Eigen::Vector3d p = center_d * dist2d().homogeneous();
    p.z() += UniformRealDistribution<double>(-0.5, 0.5)();
    points_3d.push_back(p);
  }
  *output_T_C2_C1 = Eigen::Isometry3d::Identity();
  output_T_C2_C1->linear() = R_C2_G * R_G_C1;

  std::vector<TwoViewGeometryEstimator::DataPoint> point_pairs;
  auto dist_noise = MultivariateNormalDistribution<double>::standard(2);
  for (size_t i = 0; i < points_3d.size(); ++i) {
    point_pairs.emplace_back(
        (R_C1_G * points_3d[i]).hnormalized() + noise_stddev * dist_noise(),
        (R_C2_G * points_3d[i]).hnormalized() + noise_stddev * dist_noise());
  }

  if (n_outliers > 0) {
    std::vector<size_t> outliers =
        RandomSampler().sample(n_outliers, point_pairs.size());
    for (size_t i : outliers) {
      double r = UniformRealDistribution<double>(25.0, 50.0)() /
                 TwoViewGeometryEstimator::kDefatultFocal;
      // point_pairs[i].first += dist_noise().normalized() * r;
      point_pairs[i].second += dist_noise().normalized() * r;
    }
  }

  return point_pairs;
}

}  // namespace sk4slam
