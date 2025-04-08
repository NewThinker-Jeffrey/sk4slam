#include "sk4slam_liegroups/S3.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure S3 can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestS3d, Exp) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  for (auto& w : ws) {
    S3d g = S3d::Exp(w);
    S3d::Ambient exp_w = expOnAlgebra(S3d::hat(w));
    Eigen::Vector3d log_g = S3d::Log(g);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI("S3d g: %s", toStr(g.q().coeffs().transpose()).c_str());
    LOGI("exp_w : %s", toStr(exp_w.coeffs().transpose()).c_str());
    ASSERT_NEAR((g.q() - exp_w).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - w).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestS3d, Ad_ad) {
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  using LieAlgebraEndomorphism = S3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(S3d::ad(w));
    LieAlgebraEndomorphism Ad_Exp = S3d::Ad(S3d::Exp(w));
    LOGI(
        "exp_ad    : %s, %s, %s", toStr(exp_ad.row(0)).c_str(),
        toStr(exp_ad.row(1)).c_str(), toStr(exp_ad.row(2)).c_str());
    LOGI(
        "Ad_Exp   : %s, %s, %s", toStr(Ad_Exp.row(0)).c_str(),
        toStr(Ad_Exp.row(1)).c_str(), toStr(Ad_Exp.row(2)).c_str());
    ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

    auto any_Y = MultivariateUniformDistribution<double>::standard(S3d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - S3d::Ad(S3d::Exp(w), any_Y)).squaredNorm(), 0,
        1e-6);
    ASSERT_NEAR(
        ((S3d::ad(w) * any_Y) - S3d::bracket(w, any_Y)).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestS3d, LieJacobian) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}};

  using LieAlgebraEndomorphism = S3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LieAlgebraEndomorphism Jl = S3d::Jl(w);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<S3d>(w);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI(
        "Jl    : %s, %s, %s", toStr(Jl.row(0)).c_str(),
        toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str());
    LOGI(
        "Jl2   : %s, %s, %s", toStr(Jl2.row(0)).c_str(),
        toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str());
    ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

    ASSERT_TRUE((S3d::Jl(w) * S3d::invJl(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((S3d::Jr(w) * S3d::invJr(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(S3d::Jr(w).isApprox(S3d::Jl(-w), 1e-6));
    ASSERT_TRUE(S3d::invJr(w).isApprox(S3d::invJl(-w), 1e-6));
  }
}

TEST(TestS3d, HatVee) {
  S3d::LieAlgebra w(0.1, 0.2, 0.3);
  ASSERT_TRUE(w.isApprox(S3d::vee(S3d::hat(w)), 1e-6));
}

#ifdef TEST_CERES_JET
TEST(Test_S3_, Jet) {
  S3d::LieAlgebra X, any_Y;
  X << 0.1, 0.2, 0.3;
  any_Y = MultivariateUniformDistribution<double>::standard(S3d::kDim)();

  std::vector<S3d::LieAlgebra> Xs = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}};

  using Jet = ceres::Jet<double, 7>;
  using S3Jet = S3<Jet>;
  using LieAlgebraEndomorphism = S3Jet::LieAlgebraEndomorphism;
  const Jet Jet_eps(1e-6);
  for (const auto& X : Xs) {
    S3Jet::LieAlgebra jet_X = X.cast<Jet>();
    S3Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
    S3Jet jet_g = S3Jet::Exp(jet_X);
    S3Jet::LieAlgebra jet_log_g = S3Jet::Log(jet_g);
    LOGI(
        "jet_g : %s",
        toStr(jet_g.q().coeffs().transpose(), Precision(11)).c_str());
    LOGI("jet_X     : %s", toStr(jet_X.transpose(), Precision(20)).c_str());
    LOGI("jet_log_g : %s", toStr(jet_log_g.transpose(), Precision(20)).c_str());

    auto diff = jet_log_g - jet_X;
    auto diff_norm = diff.norm();
    LOGI("diff     : %s", toStr(diff.transpose(), Precision(20)).c_str());
    LOGI("diff_norm : %s", toStr(diff_norm, Precision(20)).c_str());
    LOGI("Jet_eps : %s", toStr(Jet_eps, Precision(11)).c_str());

    ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
    ASSERT_TRUE((S3Jet::Jl(jet_X) * S3Jet::invJl(jet_X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(S3Jet::ad(jet_X));
    LieAlgebraEndomorphism Ad_Exp = S3Jet::Ad(S3Jet::Exp(jet_X));
    ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
    ASSERT_TRUE(
        (Ad_Exp * jet_any_Y)
            .isApprox(S3Jet::Ad(S3Jet::Exp(jet_X), jet_any_Y), Jet_eps));
    ASSERT_TRUE((S3Jet::ad(jet_X) * jet_any_Y)
                    .isApprox(S3Jet::bracket(jet_X, jet_any_Y), Jet_eps));
  }
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
