#include "sk4slam_liegroups/SO2.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure SO2 can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestSO2d, Exp) {
  std::vector<SO2d::LieAlgebra> ws = {
      SO2d::LieAlgebra(0.2), SO2d::LieAlgebra(0.5), SO2d::LieAlgebra(2e-10),
      SO2d::LieAlgebra(M_PI)};

  for (auto& w : ws) {
    SO2d g = SO2d::Exp(w);
    Eigen::Matrix2d exp_w = expOnAlgebra(SO2d::hat(w));
    SO2d::LieAlgebra log_g = SO2d::Log(g);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI(
        "SO2d g: %s, %s", toStr(g.matrix().row(0)).c_str(),
        toStr(g.matrix().row(1)).c_str());
    LOGI(
        "exp_w : %s, %s", toStr(exp_w.row(0)).c_str(),
        toStr(exp_w.row(1)).c_str());

    ASSERT_NEAR((g.matrix() - exp_w).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - w).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestSO2d, Ad_ad) {
  std::vector<SO2d::LieAlgebra> ws = {
      SO2d::LieAlgebra(0.2), SO2d::LieAlgebra(0.5), SO2d::LieAlgebra(2e-10),
      SO2d::LieAlgebra(M_PI)};

  using LieAlgebraEndomorphism = SO2d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(SO2d::ad(w));
    LieAlgebraEndomorphism Ad_Exp = SO2d::Ad(SO2d::Exp(w));
    LOGI("exp_ad    : %s", toStr(exp_ad).c_str());
    LOGI("Ad_Exp   : %s", toStr(Ad_Exp).c_str());
    ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

    auto any_Y =
        MultivariateUniformDistribution<double>::standard(SO2d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - SO2d::Ad(SO2d::Exp(w), any_Y)).squaredNorm(), 0,
        1e-6);
    ASSERT_NEAR(
        ((SO2d::ad(w) * any_Y) - SO2d::bracket(w, any_Y)).squaredNorm(), 0,
        1e-6);
  }
}

TEST(TestSO2d, LieJacobian) {
  std::vector<SO2d::LieAlgebra> ws = {
      SO2d::LieAlgebra(0.2), SO2d::LieAlgebra(0.5), SO2d::LieAlgebra(2e-10),
      SO2d::LieAlgebra(M_PI)};

  using LieAlgebraEndomorphism = SO2d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LieAlgebraEndomorphism Jl = SO2d::Jl(w);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<SO2d>(w);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("Jl    : %s", toStr(Jl).c_str());
    LOGI("Jl2   : %s", toStr(Jl2).c_str());
    ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

    ASSERT_TRUE((SO2d::Jl(w) * SO2d::invJl(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((SO2d::Jr(w) * SO2d::invJr(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(SO2d::Jr(w).isApprox(SO2d::Jl(-w), 1e-6));
    ASSERT_TRUE(SO2d::invJr(w).isApprox(SO2d::invJl(-w), 1e-6));
  }
}

TEST(TestSO2d, HatVee) {
  SO2d::LieAlgebra w(0.2);
  ASSERT_TRUE(w.isApprox(SO2d::vee(SO2d::hat(w)), 1e-6));
}

#ifdef TEST_CERES_JET
TEST(Test_SO2_, Jet) {
  SO2d::LieAlgebra X, any_Y;
  X << 0.2;
  any_Y = MultivariateUniformDistribution<double>::standard(SO2d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using SO2Jet = SO2<Jet>;
  using LieAlgebraEndomorphism = SO2Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  SO2Jet::LieAlgebra jet_X = X.cast<Jet>();
  SO2Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  SO2Jet jet_g = SO2Jet::Exp(jet_X);
  SO2Jet::LieAlgebra jet_log_g = SO2Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((SO2Jet::Jl(jet_X) * SO2Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(SO2Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = SO2Jet::Ad(SO2Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(SO2Jet::Ad(SO2Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((SO2Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(SO2Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
