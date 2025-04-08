#include "sk4slam_math/sac.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"

using namespace sk4slam;  // NOLINT

TEST(TestSacSampler, RandomSampler) {
  using SacModel = VectorSacModel<>;
  RandomSampler sampler;
  const size_t n_total = 500;
  std::vector<SacModel::DataPoint> all_points(n_total);
  std::vector<size_t> indices;

  auto check = [&](size_t n, const std::vector<size_t>& indices) {
    LOGI("Sampling %d out of %d: %s", n, n_total, toStr(indices).c_str());

    ASSERT_EQ(n, indices.size());
    std::set<size_t> set(indices.begin(), indices.end());
    ASSERT_EQ(n, set.size());
    ASSERT_GE(*set.begin(), 0);
    ASSERT_GE(*set.rbegin(), *set.begin());
    ASSERT_GT(n_total, *set.rbegin());
  };

  size_t test_n_list[] = {1, 2, 3, 4, 13, 20, 53, 98, 240, 500};
  for (size_t n : test_n_list) {
    indices = sampler.sample(n, all_points);
    check(n, indices);
  }
}

TEST(TestSacSampler, CombinationSampler) {
  using SacModel = VectorSacModel<>;
  CombinationSampler sampler;
  const size_t n_total = 7;
  std::vector<SacModel::DataPoint> all_points(n_total);
  std::vector<size_t> indices;

  auto check = [&](size_t n, const std::vector<size_t>& indices) {
    LOGI("Sampling %d out of %d: %s", n, n_total, toStr(indices).c_str());

    ASSERT_EQ(n, indices.size());
    std::set<size_t> set(indices.begin(), indices.end());
    ASSERT_EQ(n, set.size());
    ASSERT_GE(*set.begin(), 0);
    ASSERT_GE(*set.rbegin(), *set.begin());
    ASSERT_GT(n_total, *set.rbegin());
  };

  size_t K = 5;
  size_t max_samples = Cnk(n_total, K);
  for (size_t i = 0; i < max_samples; i++) {
    indices = sampler.sample(5, all_points);
    check(K, indices);
  }

  // After output all combinations, the sampler
  // should return empty indices
  indices = sampler.sample(5, all_points);
  ASSERT_EQ(0, indices.size());
}

TEST(TestRansac, Simple) {
  std::random_device rd;
  std::mt19937 gen(rd());
  double sigma = 2.0;

  Eigen::Vector3d real_model(10.0, 20.0, 30.0);
  std::normal_distribution<> dis_x(
      real_model.x(), sigma);  // mean = 10, sigma = 2
  std::normal_distribution<> dis_y(
      real_model.y(), sigma);  // mean = 10, sigma = 2
  std::normal_distribution<> dis_z(
      real_model.z(), sigma);  // mean = 10, sigma = 2

  // Generate test data
  size_t n_total = 100;
  size_t n_inliers = 60;
  size_t n_outliers = n_total - n_inliers;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> data(n_total);

  // Add points
  for (int i = 0; i < n_total; ++i) {
    data[i] = Eigen::Vector3d(dis_x(gen), dis_y(gen), dis_z(gen));
  }

  // Add clustered outliers
  Eigen::Vector3d outlier_offset(1.0, 1.0, 1.0);
  // outlier_offset *= 3.0 * sigma;
  outlier_offset *= 6.0 * sigma;
  for (int i = n_inliers; i < n_inliers + n_outliers / 2; ++i) {
    data[i] += outlier_offset;
  }

  // Add random outliers
  std::normal_distribution<> dis_o(0.0, 6.0 * sigma);
  for (int i = n_inliers + n_outliers / 2; i < n_total; ++i) {
    data[i] += Eigen::Vector3d(dis_o(gen), dis_o(gen), dis_o(gen));
  }

  // print data
  LOGI("Data:");
  for (size_t i = 0; i < data.size(); i++) {
    LOGI("[%d]  %s", i, toStr(data[i].transpose()).c_str());
  }

  // Set RANSAC options
  RansacOptions ransac_options;
  // 7.814727903251179 is the 0.95 quantile for the chi-square distribution
  // with 3 degrees of freedom.
  const double chi_square_95_quantile = 7.814727903251179;
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
  Ransac<VectorSacModel<>> ransac(ransac_options);
  auto report = ransac.solve(data);

  // // Run LMedS
  // LMedS<VectorSacModel<>> lmeds(ransac_options);
  // auto report = lmeds.solve(data);

  auto report_outliers = report.getOutliers();
  std::vector<uint8_t> inliers_mask;
  report.getInliersMask(&inliers_mask);

  std::set<size_t> report_outliers_set(
      report_outliers.begin(), report_outliers.end());
  std::set<size_t> report_inliers_set(
      report.inliers.begin(), report.inliers.end());
  ASSERT_EQ(report_outliers_set.size(), report_outliers.size());
  ASSERT_EQ(report_inliers_set.size(), report.inliers.size());
  ASSERT_EQ(report_outliers_set.size() + report_inliers_set.size(), n_total);
  ASSERT_EQ(inliers_mask.size(), n_total);
  ASSERT_EQ(report.n_total, n_total);
  for (const size_t idx : report_outliers_set) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, n_total);
    ASSERT_EQ(report_inliers_set.count(idx), 0);
    ASSERT_EQ(inliers_mask[idx], 0);
  }
  for (const size_t idx : report_inliers_set) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, n_total);
    ASSERT_EQ(report_outliers_set.count(idx), 0);
    ASSERT_NE(inliers_mask[idx], 0);
  }

  Eigen::Vector3d estimate_error = report.param - real_model;

  LOGI("Real model: %s", toStr(real_model.transpose()).c_str());
  LOGI("RANSAC output model: %s", toStr(report.param.transpose()).c_str());
  LOGI("RANSAC estimate error: %f", estimate_error.norm());
  LOGI(
      "RANSAC inlier_ratio: %f  (%d out of %d)", report.inlier_ratio,
      report.inliers.size(), n_total);
  LOGI("RANSAC iter: %d", report.iter);

  std::vector<size_t> false_inliers;
  for (size_t i = 0; i < report.inliers.size(); ++i) {
    auto idx = report.inliers[i];
    if (idx >= n_inliers) {
      false_inliers.push_back(idx);
    }
  }
  LOGI(
      "RANSAC inliers_mask [%d]: %s", inliers_mask.size(),
      toStr<int>(inliers_mask).c_str());
  LOGI(
      "RANSAC outliers [%d]: %s", report_outliers.size(),
      toStr(report_outliers).c_str());
  LOGI(
      "RANSAC inliers [%d]: %s", report.inliers.size(),
      toStr(report.inliers).c_str());
  LOGI(
      "False inliers [%d]: %s", false_inliers.size(),
      toStr(false_inliers).c_str());
  int TP = report.inliers.size() - false_inliers.size();
  int FP = false_inliers.size();
  int FN = n_inliers - TP;
  double precision = static_cast<double>(TP) / (TP + FP);
  double recall = static_cast<double>(TP) / (TP + FN);
  LOGI("TP: %d, FP: %d, FN: %d", TP, FP, FN);
  LOGI("Precision(TP / (TP + FP)): %f", precision);
  LOGI("Recall(TP / (TP + FN)): %f", recall);

  double real_inlier_ratio = static_cast<double>(n_inliers) / n_total;

  // double multiplier = 4.0;
  // double error_threshold =
  //     multiplier * chi_square_95_quantile * sigma * sigma / n_inliers;
  double error_threshold =
      3.0 * sigma * sigma;  // Use larger thr to make the test stable

  LOGI(
      "Error = %f,   Threshold = %f", estimate_error.squaredNorm(),
      error_threshold);

  ASSERT_GE(precision, 0.9);
  ASSERT_GE(recall, 0.9);
  ASSERT_LE(report.inlier_ratio, real_inlier_ratio * 1.1);
  ASSERT_GE(report.inlier_ratio, real_inlier_ratio * 0.9);
  ASSERT_LE(estimate_error.squaredNorm(), error_threshold);
}

SK4SLAM_UNITTEST_ENTRYPOINT
