#include "sk4slam_liegroups/S1.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure S1 can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestS1d, Exp) {
  std::vector<S1d::LieAlgebra> ws = {
      S1d::LieAlgebra(0.2), S1d::LieAlgebra(0.5), S1d::LieAlgebra(2e-10),
      S1d::LieAlgebra(M_PI)};

  for (auto& w : ws) {
    S1d g = S1d::Exp(w);
    S1d::Complex exp_w = expOnAlgebra(S1d::hat(w));
    S1d::LieAlgebra log_g = S1d::Log(g);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI("S1d g: %s", toStr(g.c().transpose()).c_str());
    LOGI("exp_w : %s", toStr(exp_w.transpose()).c_str());
    ASSERT_NEAR((g.c() - exp_w).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - w).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestS1d, Ad_ad) {
  std::vector<S1d::LieAlgebra> ws = {
      S1d::LieAlgebra(0.2), S1d::LieAlgebra(0.5), S1d::LieAlgebra(2e-10),
      S1d::LieAlgebra(M_PI)};

  using LieAlgebraEndomorphism = S1d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(S1d::ad(w));
    LieAlgebraEndomorphism Ad_Exp = S1d::Ad(S1d::Exp(w));
    LOGI("exp_ad    : %s", toStr(exp_ad).c_str());
    LOGI("Ad_Exp   : %s", toStr(Ad_Exp).c_str());
    ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

    auto any_Y = MultivariateUniformDistribution<double>::standard(S1d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - S1d::Ad(S1d::Exp(w), any_Y)).squaredNorm(), 0,
        1e-6);
    ASSERT_NEAR(
        ((S1d::ad(w) * any_Y) - S1d::bracket(w, any_Y)).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestS1d, LieJacobian) {
  std::vector<S1d::LieAlgebra> ws = {
      S1d::LieAlgebra(0.2), S1d::LieAlgebra(0.5), S1d::LieAlgebra(2e-10),
      S1d::LieAlgebra(M_PI)};

  using LieAlgebraEndomorphism = S1d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LieAlgebraEndomorphism Jl = S1d::Jl(w);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<S1d>(w);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("Jl    : %s", toStr(Jl).c_str());
    LOGI("Jl2   : %s", toStr(Jl2).c_str());
    ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

    ASSERT_TRUE((S1d::Jl(w) * S1d::invJl(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((S1d::Jr(w) * S1d::invJr(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(S1d::Jr(w).isApprox(S1d::Jl(-w), 1e-6));
    ASSERT_TRUE(S1d::invJr(w).isApprox(S1d::invJl(-w), 1e-6));
  }
}

TEST(TestS1d, HatVee) {
  S1d::LieAlgebra w(0.2);
  ASSERT_TRUE(w.isApprox(S1d::vee(S1d::hat(w)), 1e-6));
}

#ifdef TEST_CERES_JET
TEST(Test_S1_, Jet) {
  S1d::LieAlgebra X, any_Y;
  X << 0.2;
  any_Y = MultivariateUniformDistribution<double>::standard(S1d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using S1Jet = S1<Jet>;
  using LieAlgebraEndomorphism = S1Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  S1Jet::LieAlgebra jet_X = X.cast<Jet>();
  S1Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  S1Jet jet_g = S1Jet::Exp(jet_X);
  S1Jet::LieAlgebra jet_log_g = S1Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((S1Jet::Jl(jet_X) * S1Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(S1Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = S1Jet::Ad(S1Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE((Ad_Exp * jet_any_Y)
                  .isApprox(S1Jet::Ad(S1Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((S1Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(S1Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
