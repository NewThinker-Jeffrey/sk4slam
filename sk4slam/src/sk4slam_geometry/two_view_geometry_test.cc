#include "sk4slam_geometry/two_view_geometry_test_helper.h"

using namespace sk4slam;  // NOLINT

static const double errThrForSimpleNonCoplanarPointPairs = 0.02 * 0.02;
// static const double errThrForSimpleNonCoplanarPointPairs = 0.05 * 0.05;

TEST(EpipolarBasicTest, _8PointFundamental) {
  // Logging::setVerbose("DEBUG");
  const auto& point_pairs = getSimpleNonCoplanarPointPairs();
  TimeCounter tc;
  auto report = Fundamental8PointEstimator::ransac(
      point_pairs, RansacOptions(
                       errThrForSimpleNonCoplanarPointPairs,
                       // error_thr (for normalized images).
                       0.999,  // confidence
                       5000,   // max_iter

                       // 10 / 12.0,  // initial_min_inlier_ratio.
                       0.00,  // initial_min_inlier_ratio.
                              // Even though the actual inlier ratio is 10/12,
                              // setting `initial_min_inlier_ratio` to 10/12
                              // will lead to frequent failures of the RANSAC
                              // algorithm.

                       0,   // local_opt_max_iter
                       1    // final_opt_max_iter
                       ));  // NOLINT
  tc.tag("ransac_over");
  tc.report("EpipolarBasicTest._8PointFundamental Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::FUNDAMENTAL);
  auto report_outliers = report.getOutliers();
  ASSERT_EQ(report.inliers.size(), 10);
  ASSERT_EQ(report_outliers.size(), 2);
  ASSERT_EQ(report_outliers[0], 10);
  ASSERT_EQ(report_outliers[1], 11);
}

TEST(EpipolarBasicTest, _7PointFundamental) {
  const auto& point_pairs = getSimpleNonCoplanarPointPairs();
  TimeCounter tc;
  auto report = Fundamental7PointEstimator::ransac(
      point_pairs, RansacOptions(
                       errThrForSimpleNonCoplanarPointPairs,
                       // error_thr (for normalized images).
                       0.999,  // confidence
                       5000,   // max_iter

                       // 10 / 12.0,  // initial_min_inlier_ratio.
                       0.00,  // initial_min_inlier_ratio.
                              // Even though the actual inlier ratio is 10/12,
                              // setting `initial_min_inlier_ratio` to 10/12
                              // will lead to frequent failures of the RANSAC
                              // algorithm.

                       0,   // local_opt_max_iter
                       1    // final_opt_max_iter
                       ));  // NOLINT
  tc.tag("ransac_over");
  tc.report("EpipolarBasicTest._7PointFundamental Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::FUNDAMENTAL);
  auto report_outliers = report.getOutliers();
  ASSERT_EQ(report.inliers.size(), 10);
  ASSERT_EQ(report_outliers.size(), 2);
  ASSERT_EQ(report_outliers[0], 10);
  ASSERT_EQ(report_outliers[1], 11);
}

TEST(EpipolarBasicTest, _8PointEssential) {
  const auto& point_pairs = getSimpleNonCoplanarPointPairs();
  TimeCounter tc;
  auto report = Essential8PointEstimator::ransac(
      point_pairs, RansacOptions(
                       errThrForSimpleNonCoplanarPointPairs,
                       // error_thr (for normalized images).
                       0.999,  // confidence
                       5000,   // max_iter

                       // 10 / 12.0,  // initial_min_inlier_ratio.
                       0.00,  // initial_min_inlier_ratio.
                              // Even though the actual inlier ratio is 10/12,
                              // setting `initial_min_inlier_ratio` to 10/12
                              // will lead to frequent failures of the RANSAC
                              // algorithm.

                       0,   // local_opt_max_iter
                       1    // final_opt_max_iter
                       ));  // NOLINT
  tc.tag("ransac_over");
  tc.report("EpipolarBasicTest._8PointEssential Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  auto report_outliers = report.getOutliers();
  ASSERT_EQ(report.inliers.size(), 10);
  ASSERT_EQ(report_outliers.size(), 2);
  ASSERT_EQ(report_outliers[0], 10);
  ASSERT_EQ(report_outliers[1], 11);
}

TEST(EpipolarBasicTest, _5PointEssential) {
  const auto& point_pairs = getSimpleNonCoplanarPointPairs();
  TimeCounter tc;
  auto report = Essential5PointEstimator::ransac(
      point_pairs, RansacOptions(
                       errThrForSimpleNonCoplanarPointPairs,
                       // error_thr (for normalized images).
                       0.999,  // confidence
                       5000,   // max_iter

                       // 10 / 12.0,  // initial_min_inlier_ratio.
                       0.00,  // initial_min_inlier_ratio.
                              // Even though the actual inlier ratio is 10/12,
                              // setting `initial_min_inlier_ratio` to 10/12
                              // will lead to frequent failures of the RANSAC
                              // algorithm.

                       0,   // local_opt_max_iter
                       1    // final_opt_max_iter
                       ));  // NOLINT
  tc.tag("ransac_over");
  tc.report("EpipolarBasicTest._5PointEssential Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  auto report_outliers = report.getOutliers();
  ASSERT_EQ(report.inliers.size(), 10);
  ASSERT_EQ(report_outliers.size(), 2);
  ASSERT_EQ(report_outliers[0], 10);
  ASSERT_EQ(report_outliers[1], 11);
}

TEST(HomographyBasicTest, HomographyDLT) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 4;
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto ransac_options = defaultRansacOptions();
  // ransac_options.initial_min_inlier_ratio =
  //     // 0.70 * static_cast<double>(n_coplanar) / (n_coplanar +
  //     n_non_coplanar); 0.80 * static_cast<double>(n_coplanar) / (n_coplanar +
  //     n_non_coplanar);

  // Homography is not stable without local optimization.
  if (ransac_options.final_opt_max_iter == 0) {
    ransac_options.final_opt_max_iter = 1;
  }

  auto report = HomographyEstimator::ransac(point_pairs, ransac_options);
  tc.tag("ransac_over");
  tc.report("HomographyBasicTest.HomographyDLT Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.8);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(HomographyPureRotationTest, HomographyDLT) {
  Eigen::Isometry3d real_T_C2_C1;
  const auto point_pairs = getPureRotationPointPairs(&real_T_C2_C1);
  TimeCounter tc;

  auto ransac_options = defaultRansacOptions();

  // Homography is not stable without local optimization.
  if (ransac_options.final_opt_max_iter == 0) {
    ransac_options.final_opt_max_iter = 1;
  }

  auto report = HomographyEstimator::ransac(point_pairs, ransac_options);
  tc.tag("ransac_over");
  tc.report("HomographyPureRotationTest.HomographyDLT Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialPureRotationTest, _5Point) {
  Eigen::Isometry3d real_T_C2_C1;
  const auto point_pairs = getPureRotationPointPairs(&real_T_C2_C1);
  TimeCounter tc;
  auto report =
      Essential5PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialPureRotationTest._5Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ROTATION);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialPureRotationTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  const auto point_pairs = getPureRotationPointPairs(&real_T_C2_C1);
  TimeCounter tc;
  auto report =
      Essential8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialPureRotationTest._8Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ROTATION);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalPureRotationTest, _7Point) {
  Eigen::Isometry3d real_T_C2_C1;
  const auto point_pairs = getPureRotationPointPairs(&real_T_C2_C1);

  TimeCounter tc;
  auto report =
      Fundamental7PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalPureRotationTest._7Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalPureRotationTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  const auto point_pairs = getPureRotationPointPairs(&real_T_C2_C1);

  TimeCounter tc;
  auto report =
      Fundamental8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalPureRotationTest._8Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialCoplanarTest, _5Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 0;  // So all data points will be coplanar.
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Essential5PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialCoplanarTest._5Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialCoplanarTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 0;  // So all data points will be coplanar.
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Essential8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialCoplanarTest._8Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalCoplanarTest, _7Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 0;  // So all data points will be coplanar.
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Fundamental7PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalCoplanarTest._7Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalCoplanarTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 0;  // So all data points will be coplanar.
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Fundamental8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalCoplanarTest._8Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

////

TEST(EssentialMixedCoplanarTest, _5Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 15;
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Essential5PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialMixedCoplanarTest._5Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialMixedCoplanarTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 15;
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Essential8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialMixedCoplanarTest._8Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalMixedCoplanarTest, _7Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 15;
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Fundamental7PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalMixedCoplanarTest._7Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::FUNDAMENTAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  // checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalMixedCoplanarTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 40;
  int n_non_coplanar = 15;
  const auto point_pairs =
      getCoplanarPointPairs(&real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar);
  TimeCounter tc;
  auto report =
      Fundamental8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalMixedCoplanarTest._8Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::FUNDAMENTAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(report, point_pairs, 0.85);
  // checkPose(report, point_pairs, real_T_C2_C1);
}

////////////////////////////////////////////////

// Tests with outliers

// static constexpr int kOutliers = 40;
static constexpr int kOutliers = 25;
// static constexpr int kOutliers = 15;
// static constexpr int kOutliers = 10;

////

TEST(EssentialMixedCoplanarAndOutlierTest, _5Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 75;
  int n_non_coplanar = 25;
  int n_total = n_coplanar + n_non_coplanar;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs = getCoplanarPointPairs(
      &real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar, n_outlier);
  TimeCounter tc;
  auto report =
      Essential5PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialMixedCoplanarAndOutlierTest._5Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialMixedCoplanarAndOutlierTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 75;
  int n_non_coplanar = 25;
  int n_total = n_coplanar + n_non_coplanar;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs = getCoplanarPointPairs(
      &real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar, n_outlier);
  TimeCounter tc;
  auto report =
      Essential8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialMixedCoplanarAndOutlierTest._8Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ESSENTIAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalMixedCoplanarAndOutlierTest, _7Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 75;
  int n_non_coplanar = 25;
  int n_total = n_coplanar + n_non_coplanar;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs = getCoplanarPointPairs(
      &real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar, n_outlier);
  TimeCounter tc;
  auto report =
      Fundamental7PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalMixedCoplanarAndOutlierTest._7Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::FUNDAMENTAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  // checkPose(report, point_pairs, real_T_C2_C1);
}

///

TEST(EssentialPureRotationAndOutlierTest, _5Point) {
  Eigen::Isometry3d real_T_C2_C1;
  int n_total = 100;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs =
      getPureRotationPointPairs(&real_T_C2_C1, n_total, n_outlier);
  TimeCounter tc;
  auto report =
      Essential5PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialPureRotationAndOutlierTest._5Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ROTATION);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(EssentialPureRotationAndOutlierTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  int n_total = 100;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs =
      getPureRotationPointPairs(&real_T_C2_C1, n_total, n_outlier);
  TimeCounter tc;
  auto report =
      Essential8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("EssentialPureRotationAndOutlierTest._8Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::ROTATION);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalMixedCoplanarAndOutlierTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  Eigen::Matrix3d real_H;
  int n_coplanar = 75;
  int n_non_coplanar = 25;
  int n_total = n_coplanar + n_non_coplanar;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs = getCoplanarPointPairs(
      &real_T_C2_C1, &real_H, n_coplanar, n_non_coplanar, n_outlier);
  TimeCounter tc;
  auto report =
      Fundamental8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalMixedCoplanarAndOutlierTest._8Point Timing: ", true);
  LOGI("real_H:\n%s", toStr(real_H / real_H(2, 2)).c_str());
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::FUNDAMENTAL);
  // ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  // checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalPureRotationAndOutlierTest, _7Point) {
  Eigen::Isometry3d real_T_C2_C1;
  int n_total = 100;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs =
      getPureRotationPointPairs(&real_T_C2_C1, n_total, n_outlier);

  TimeCounter tc;
  auto report =
      Fundamental7PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalPureRotationAndOutlierTest._7Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  checkPose(report, point_pairs, real_T_C2_C1);
}

TEST(FundamentalPureRotationAndOutlierTest, _8Point) {
  Eigen::Isometry3d real_T_C2_C1;
  int n_total = 100;
  int n_outlier = kOutliers;
  double expected_inlier_ratio =
      (n_total - n_outlier) / static_cast<double>(n_total);
  const auto point_pairs =
      getPureRotationPointPairs(&real_T_C2_C1, n_total, n_outlier);

  TimeCounter tc;
  auto report =
      Fundamental8PointEstimator::degensac(point_pairs, defaultRansacOptions());
  tc.tag("degensac_over");
  tc.report("FundamentalPureRotationAndOutlierTest._8Point Timing: ", true);
  printTwoViewGeometryRansacReport(report, point_pairs);
  ASSERT_EQ(report.param.type, TwoViewGeometryType::HOMOGRAPHY);
  checkRansac(
      report, point_pairs, expected_inlier_ratio - 0.15,
      expected_inlier_ratio + 0.15);
  checkPose(report, point_pairs, real_T_C2_C1);
}

SK4SLAM_UNITTEST_ENTRYPOINT
