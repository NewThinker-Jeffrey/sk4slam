#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/polynomial.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

TEST(TestMath, Basic) {
  // TODO(jeffrey)
  LOGI("Run testing for TestMath.Basic");
}

TEST(TestMath, Matrix) {
  // TODO(jeffrey)
  LOGI("Run testing for TestMath.Matrix");
}

TEST(TestMath, RandomUniformDistribution) {
  Eigen::Vector2d lower_bounds(-1, -1);
  Eigen::Vector2d upper_bounds(1, 1);

  auto dist =
      MultivariateUniformDistribution<double>(lower_bounds, upper_bounds);

  // Generate 40000 samples
  constexpr int kSampleSize = 40000;
  Eigen::MatrixXd samples(2, kSampleSize);
  size_t n_fall_in_unit_circle = 0;
  for (int i = 0; i < kSampleSize; ++i) {
    auto tmp = dist();
    ASSERT_TRUE((tmp.array() >= lower_bounds.array()).all());
    ASSERT_TRUE((tmp.array() < upper_bounds.array()).all());
    if (tmp.norm() <= 1.0) {
      n_fall_in_unit_circle++;
    }
    samples.col(i) = tmp;
  }

  Eigen::Matrix<double, 2, 1> mean = samples.rowwise().mean();
  LOGI("Mean: %s", toStr(mean.transpose()).c_str());
  ASSERT_LE(mean.norm(), 1e-1);

  LOGI(
      "Number of points fall in unit circle: %d (expected 31415)",
      n_fall_in_unit_circle);
  ASSERT_NEAR(n_fall_in_unit_circle, 31415, 2000);
}

TEST(TestMath, RandomNormalDistribution) {
  auto dist = MultivariateNormalDistribution<double>::random(3);

  // Generate 10000 samples
  constexpr int kSampleSize = 10000;
  Eigen::MatrixXd samples(3, kSampleSize);
  for (int i = 0; i < kSampleSize; ++i) {
    samples.col(i) = dist();
  }

  Eigen::Matrix<double, 3, 1> mean = samples.rowwise().mean();
  Eigen::Matrix<double, 3, 3> cov =
      (samples * samples.transpose() - kSampleSize * mean * mean.transpose()) /
      (kSampleSize - 1);

  Eigen::Matrix<double, 3, 1> delta_mean = mean - dist.mean();
  Eigen::Matrix<double, 3, 3> delta_cov = cov - dist.cov();

  double delta_mean_squared_sum = delta_mean.squaredNorm();
  double delta_cov_squared_sum = delta_cov.squaredNorm();

  double cov_error_ratio = delta_cov_squared_sum / dist.cov().squaredNorm();
  double mahal_error =
      delta_mean.transpose() * dist.cov().inverse() * delta_mean;

  LOGI("Real Covariance:\n %s", toStr(cov).c_str());
  LOGI("Sample Covariance:\n %s", toStr(dist.cov()).c_str());
  LOGI("Delta Covariance:\n %s", toStr(delta_cov).c_str());
  LOGI("Delta Covariance squared sum: %f", delta_cov_squared_sum);
  LOGI("cov_error_ratio: %f", cov_error_ratio);

  LOGI("Real Mean: %s", toStr(mean.transpose()).c_str());
  LOGI("Sample Mean: %s", toStr(dist.mean().transpose()).c_str());
  LOGI("Delta Mean: %s", toStr(delta_mean.transpose()).c_str());
  LOGI("Delta Mean squared sum: %f", delta_mean_squared_sum);
  LOGI("mahal_error: %f", mahal_error);

  ASSERT_LE(cov_error_ratio, 1e-1);
  ASSERT_LE(mahal_error, 1e-1);
}

TEST(TestMath, Polynomial) {
  // TODO(jeffrey)
  NormalDistribution<double> gen;
  Eigen::Matrix<double, 5, 1> tmp;
  Eigen::MatrixXd coeffs = tmp.unaryExpr([&](auto x) { return gen(); });

  auto check_roots = [&](const Eigen::VectorXd& real,
                         const Eigen::VectorXd& imag) {
    ASSERT_EQ(real.size(), imag.size());
    ASSERT_EQ(real.size(), coeffs.size() - 1);
    for (int i = 0; i < real.size(); ++i) {
      auto v =
          evaluatePolynomial(coeffs, std::complex<double>(real(i), imag(i)));
      LOGI(
          "Root[%d]: %f + %fi, eval: %f + %fi", i, real(i), imag(i), v.real(),
          v.imag());
      ASSERT_NEAR(v.real(), 0, 1e-6);
      ASSERT_NEAR(v.imag(), 0, 1e-6);
    }
  };
  LOGI(
      "Polynomial coeffs[%d]: %s", coeffs.size(),
      toStr(coeffs.transpose()).c_str());

  {
    Eigen::VectorXd real, imag;
    bool use_durand_kerner = true;
    findRootsForPolynomial(coeffs, &real, &imag, use_durand_kerner);
    LOGI("Check roots for durand_kerner method:");
    check_roots(real, imag);
    Eigen::VectorXd real_roots =
        findRealRootsForPolynomial(coeffs, 1e-8, false, use_durand_kerner);
    LOGI("Real roots: %s", toStr(real_roots.transpose()).c_str());
  }

  {
    Eigen::VectorXd real, imag;
    bool use_durand_kerner = false;
    findRootsForPolynomial(coeffs, &real, &imag, use_durand_kerner);
    LOGI("Check roots for companion-matrix method:");
    check_roots(real, imag);
    Eigen::VectorXd real_roots =
        findRealRootsForPolynomial(coeffs, 1e-8, false, use_durand_kerner);
    LOGI("Real roots: %s", toStr(real_roots.transpose()).c_str());
  }

  {
    coeffs = Eigen::Matrix<double, 4, 1>();
    coeffs << 1, -1, -1, 1;
    LOGI(
        "Polynomial coeffs[%d]: %s", coeffs.size(),
        toStr(coeffs.transpose()).c_str());
    Eigen::VectorXd real_roots;
    bool merge_repeated_roots = false;
    real_roots = findRealRootsForPolynomial(coeffs, 1e-8, merge_repeated_roots);
    LOGI(
        "Real roots (merge_repeated_roots=false): %s",
        toStr(real_roots.transpose()).c_str());
    ASSERT_EQ(real_roots.size(), 3);

    merge_repeated_roots = true;
    real_roots = findRealRootsForPolynomial(coeffs, 1e-8, merge_repeated_roots);
    LOGI(
        "Real roots (merge_repeated_roots=true): %s",
        toStr(real_roots.transpose()).c_str());
    ASSERT_EQ(real_roots.size(), 2);
  }
}

SK4SLAM_UNITTEST_ENTRYPOINT
