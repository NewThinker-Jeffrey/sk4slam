#include "sk4slam_liegroups/mean.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SE3.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"
#include "sk4slam_math/sac.h"

using namespace sk4slam;  // NOLINT

TEST(TestLieGroupMean, NoWeights) {
  using LieGroup = SE3d;
  auto dist = MultivariateNormalDistribution<double>::standard(LieGroup::kDim);
  LieGroup::LieAlgebra X;
  // X = dist();
  X << 0.8561, 2.3318, 1.9139, 1.2086, 0.3122, 0.5135;
  LOGI("X: %s\n", toStr(X.transpose()).c_str());
  LieGroup real_mean = LieGroup::Exp(X);
  LOGI(
      "real_mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(real_mean.linear().matrix()).c_str(),
      toStr(real_mean.translation().transpose()).c_str());

  std::vector<LieGroup> samples;
  for (int i = 0; i < 1000; i++) {
    samples.push_back(real_mean * LieGroup::Exp(0.1 * dist()));
  }

  LieGroup mean;
  bool success;

  LOGI("Solve with RightPerturbation:");
  success = computeLieGroupMean<false>(samples, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));

  LOGI("Solve with LeftPerturbation:");
  success = computeLieGroupMean<true>(samples, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));
}

TEST(TestLieGroupMean, ScalarWeights) {
  using LieGroup = SE3d;
  auto dist = MultivariateNormalDistribution<double>::standard(LieGroup::kDim);
  LieGroup::LieAlgebra X;
  // X = dist();
  X << 0.8561, 2.3318, 1.9139, 1.2086, 0.3122, 0.5135;
  LOGI("X: %s\n", toStr(X.transpose()).c_str());
  LieGroup real_mean = LieGroup::Exp(X);
  LOGI(
      "real_mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(real_mean.linear().matrix()).c_str(),
      toStr(real_mean.translation().transpose()).c_str());

  std::vector<LieGroup> samples;
  for (int i = 0; i < 1000; i++) {
    samples.push_back(real_mean * LieGroup::Exp(0.1 * dist()));
  }

  std::vector<double> scalar_weights(samples.size(), 1.0);

  LieGroup mean;
  bool success;

  LOGI("Solve with RightPerturbation:");
  success = computeLieGroupMean<false>(samples, scalar_weights, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));

  LOGI("Solve with LeftPerturbation:");
  success = computeLieGroupMean<true>(samples, scalar_weights, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));
}

TEST(TestLieGroupMean, DiagWeights) {
  using LieGroup = SE3d;
  auto dist = MultivariateNormalDistribution<double>::standard(LieGroup::kDim);
  LieGroup::LieAlgebra X;
  // X = dist();
  X << 0.8561, 2.3318, 1.9139, 1.2086, 0.3122, 0.5135;
  LOGI("X: %s\n", toStr(X.transpose()).c_str());
  LieGroup real_mean = LieGroup::Exp(X);
  LOGI(
      "real_mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(real_mean.linear().matrix()).c_str(),
      toStr(real_mean.translation().transpose()).c_str());

  std::vector<LieGroup> samples;
  for (int i = 0; i < 1000; i++) {
    samples.push_back(real_mean * LieGroup::Exp(0.1 * dist()));
  }

  Eigen::VectorXd diag_weight(LieGroup::kDim);
  diag_weight.setConstant(1.0);
  std::vector<Eigen::VectorXd> diag_weights(samples.size(), diag_weight);

  LieGroup mean;
  bool success;

  LOGI("Solve with RightPerturbation:");
  success = computeLieGroupMean<false>(samples, diag_weights, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));

  LOGI("Solve with LeftPerturbation:");
  success = computeLieGroupMean<true>(samples, diag_weights, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));
}

TEST(TestLieGroupMean, MatrixWeights) {
  using LieGroup = SE3d;
  auto dist = MultivariateNormalDistribution<double>::standard(LieGroup::kDim);
  LieGroup::LieAlgebra X;
  // X = dist();
  X << 0.8561, 2.3318, 1.9139, 1.2086, 0.3122, 0.5135;
  LOGI("X: %s\n", toStr(X.transpose()).c_str());
  LieGroup real_mean = LieGroup::Exp(X);
  LOGI(
      "real_mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(real_mean.linear().matrix()).c_str(),
      toStr(real_mean.translation().transpose()).c_str());

  std::vector<LieGroup> samples;
  for (int i = 0; i < 1000; i++) {
    samples.push_back(real_mean * LieGroup::Exp(0.1 * dist()));
  }

  Eigen::MatrixXd matrix_weight(LieGroup::kDim, LieGroup::kDim);
  matrix_weight.setIdentity();
  std::vector<Eigen::MatrixXd> matrix_weights(samples.size(), matrix_weight);

  LieGroup mean;
  bool success;

  LOGI("Solve with RightPerturbation:");
  success = computeLieGroupMean<false>(samples, matrix_weights, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));

  LOGI("Solve with LeftPerturbation:");
  success = computeLieGroupMean<true>(samples, matrix_weights, &mean);
  LOGI(
      "mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  ASSERT_TRUE(success);
  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));
}

TEST(TestLieGroupMean, Ransac) {
  using LieGroup = SE3d;
  auto dist = MultivariateNormalDistribution<double>::standard(LieGroup::kDim);
  LieGroup::LieAlgebra X;
  // X = dist();
  X << 0.8561, 2.3318, 1.9139, 1.2086, 0.3122, 0.5135;
  LOGI("X: %s\n", toStr(X.transpose()).c_str());
  LieGroup real_mean = LieGroup::Exp(X);
  LOGI(
      "real_mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(real_mean.linear().matrix()).c_str(),
      toStr(real_mean.translation().transpose()).c_str());

  const double sigma = 0.1;
  std::vector<LieGroup> samples;
  for (int i = 0; i < 1000; i++) {
    samples.push_back(real_mean * LieGroup::Exp(sigma * dist()));
  }

  // Set RANSAC options
  RansacOptions ransac_options;
  // 12.59 is the 0.95 quantile for the chi-square distribution
  // with 6 degrees of freedom.
  const double chi_square_95_quantile = 12.59;
  ransac_options.error_thr = chi_square_95_quantile * sigma * sigma;
  // ransac_options.initial_min_inlier_ratio = 0.95 * n_inliers / n_total;
  // ransac_options.initial_min_inlier_ratio *=
  //     0.7;  // This will increase the number of iterations and the test
  //           // can be more stable.
  ransac_options.confidence = 0.99;
  // ransac_options.max_iter = 100;

  // ransac_options.local_opt_max_iter = 0;
  ransac_options.local_opt_max_iter = 5;

  // Run RANSAC
  using SacModel = LieGroupSacModel<LieGroup>;
  SacModel sac_model(10);
  Ransac<SacModel> ransac(ransac_options, sac_model);
  auto report = ransac.solve(samples);
  ASSERT_TRUE(!report.inliers.empty());
  auto& mean = report.param;
  LOGI(
      "RANSAC output mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());

  auto report_outliers = report.getOutliers();
  std::vector<uint8_t> inliers_mask;
  report.getInliersMask(&inliers_mask);
  LOGI(
      "RANSAC inlier_ratio: %f  (%d out of %d)", report.inlier_ratio,
      report.inliers.size(), report.n_total);
  LOGI("RANSAC iter: %d", report.iter);
  LOGI(
      "RANSAC inliers_mask [%d]: %s", inliers_mask.size(),
      toStr<int>(inliers_mask).c_str());
  LOGI(
      "RANSAC outliers [%d]: %s", report_outliers.size(),
      toStr(report_outliers).c_str());
  LOGI(
      "RANSAC inliers [%d]: %s", report.inliers.size(),
      toStr(report.inliers).c_str());

  ASSERT_TRUE(mean.isApprox(real_mean, 3e-2));
}

TEST(TestLieGroupMean, RealDataRansac) {
  using LieGroup = SE3d;

  Eigen::Matrix3d R0, R1, R2, R3, R4;
  Eigen::Vector3d t0, t1, t2, t3, t4;

  R0 << 0.9834, 0.1814, 0.0042, -0.1814, 0.9834, -0.0078, -0.0056, 0.0069,
      1.0000;
  t0 << -0.0067, -0.0043, 0.0121;

  R1 << 0.9838, 0.1791, 0.0039, -0.1791, 0.9838, -0.0041, -0.0046, 0.0033,
      1.0000;
  t1 << -0.0009, 0.0005, 0.0212;

  R2 << 0.9835, 0.1808, 0.0040, -0.1808, 0.9835, -0.0089, -0.0056, 0.0080,
      1.0000;
  t2 << -0.0056, -0.0019, 0.0065;

  R3 << 0.9843, 0.1766, 0.0050, -0.1764, 0.9840, -0.0260, -0.0095, 0.0247,
      0.9996;
  t3 << 0.0082, 0.0067, -0.0389;

  R4 << 0.9838, 0.1793, 0.0084, -0.1791, 0.9837, -0.0147, -0.0109, 0.0130,
      0.9999;
  t4 << 0.0050, -0.0011, -0.0011;

  std::vector<LieGroup> samples = {
      LieGroup(R0, t0), LieGroup(R1, t1), LieGroup(R2, t2), LieGroup(R3, t3),
      LieGroup(R4, t4)};

  const double r_sigma = 0.02;
  const double t_sigma = 0.1;
  const double r_sigma_inv = 1.0 / r_sigma;
  const double t_sigma_inv = 1.0 / t_sigma;
  Eigen::Matrix<double, 6, 1> diag_weights;
  diag_weights << r_sigma_inv, r_sigma_inv, r_sigma_inv, t_sigma_inv,
      t_sigma_inv, t_sigma_inv;

  // Set RANSAC options
  RansacOptions ransac_options;
  // 12.59 is the 0.95 quantile for the chi-square distribution
  // with 6 degrees of freedom.
  const double chi_square_95_quantile = 12.59;
  ransac_options.error_thr = chi_square_95_quantile;
  // ransac_options.initial_min_inlier_ratio = 0.95 * n_inliers / n_total;
  // ransac_options.initial_min_inlier_ratio *=
  //     0.7;  // This will increase the number of iterations and the test
  //           // can be more stable.
  ransac_options.confidence = 0.99;
  // ransac_options.max_iter = 100;

  // ransac_options.local_opt_max_iter = 0;
  ransac_options.local_opt_max_iter = 5;

  // Run RANSAC
  using SacModel = LieGroupSacModel<LieGroup>;
  SacModel sac_model(10, nullptr, diag_weights);
  Ransac<SacModel> ransac(ransac_options, sac_model);
  auto report = ransac.solve(samples);
  auto& mean = report.param;
  LOGI(
      "RANSAC output mean:\n\t rotation =\n%s\n\t translation =\n%s\n",
      toStr(mean.linear().matrix()).c_str(),
      toStr(mean.translation().transpose()).c_str());
  std::vector<double> errors = sac_model.errors({0, 1, 2, 3, 4}, samples, mean);
  LOGI("RANSAC errors: %s", toStr(errors, sqrt).c_str());

  auto report_outliers = report.getOutliers();
  std::vector<uint8_t> inliers_mask;
  report.getInliersMask(&inliers_mask);
  LOGI(
      "RANSAC inlier_ratio: %f  (%d out of %d)", report.inlier_ratio,
      report.inliers.size(), report.n_total);
  LOGI("RANSAC iter: %d", report.iter);
  LOGI(
      "RANSAC inliers_mask [%d]: %s", inliers_mask.size(),
      toStr<int>(inliers_mask).c_str());
  LOGI(
      "RANSAC outliers [%d]: %s", report_outliers.size(),
      toStr(report_outliers).c_str());
  LOGI(
      "RANSAC inliers [%d]: %s", report.inliers.size(),
      toStr(report.inliers).c_str());
  ASSERT_EQ(report.inliers.size(), 5);
}

SK4SLAM_UNITTEST_ENTRYPOINT
