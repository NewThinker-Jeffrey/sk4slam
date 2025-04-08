#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure Rp_x_SO3 can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestRp_x_SO2d, Exp) {
  std::vector<SO2d::LieAlgebra> ws = {
      SO2d::LieAlgebra(0.2), SO2d::LieAlgebra(0.5), SO2d::LieAlgebra(2e-10),
      SO2d::LieAlgebra(M_PI)};

  for (auto& w : ws) {
    double sigma = UniformRealDistribution(-0.5, 0.5)();
    Eigen::Matrix<double, 2, 1> X;
    X << sigma, w;

    Rp_x_SO2d g = Rp_x_SO2d::Exp(X);
    Eigen::Matrix<double, 2, 1> log_g = Rp_x_SO2d::Log(g);
    Eigen::Matrix2d r1 = g.matrix();
    Eigen::Matrix2d r2 = exp(sigma) * SO2d::Exp(w).matrix();
    Eigen::Matrix2d r3 = expOnAlgebra(Rp_x_SO2d::hat(X).matrix());
    LOGI("sigma : %f", sigma);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI("r1: %s, %s", toStr(r1.row(0)).c_str(), toStr(r1.row(1)).c_str());
    LOGI("r2 : %s, %s, %s", toStr(r2.row(0)).c_str(), toStr(r2.row(1)).c_str());
    LOGI("r3 : %s, %s", toStr(r3.row(0)).c_str(), toStr(r3.row(1)).c_str());

    ASSERT_NEAR((r1 - r2).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((r1 - r3).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - X).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestRp_x_SO2d, Ad_ad) {
  std::vector<SO2d::LieAlgebra> ws = {
      SO2d::LieAlgebra(0.2), SO2d::LieAlgebra(0.5), SO2d::LieAlgebra(2e-10),
      SO2d::LieAlgebra(M_PI)};

  using LieAlgebraEndomorphism = Rp_x_SO2d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    double sigma = UniformRealDistribution(-0.5, 0.5)();
    Eigen::Matrix<double, 2, 1> X;
    X << sigma, w;
    LOGI("sigma : %f", sigma);
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(Rp_x_SO2d::ad(X));
    LieAlgebraEndomorphism Ad_Exp = Rp_x_SO2d::Ad(Rp_x_SO2d::Exp(X));
    LOGI(
        "exp_ad    : %s, %s", toStr(exp_ad.part<1>().row(0)).c_str(),
        toStr(exp_ad.part<0>().row(0)).c_str());
    LOGI(
        "Ad_Exp   : %s, %s", toStr(Ad_Exp.part<1>().row(0)).c_str(),
        toStr(Ad_Exp.part<0>().row(0)).c_str());

    ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, 1e-6));

    Eigen::Matrix<double, Rp_x_SO2d::kDim, 1> any_Y =
        MultivariateUniformDistribution<double>::standard(Rp_x_SO2d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - Rp_x_SO2d::Ad(Rp_x_SO2d::Exp(X), any_Y))
            .squaredNorm(),
        0, 1e-6);
    ASSERT_NEAR(
        ((Rp_x_SO2d::ad(X) * any_Y) - Rp_x_SO2d::bracket(X, any_Y))
            .squaredNorm(),
        0, 1e-6);
  }
}

TEST(TestRp_x_SO2d, LieJacobian) {
  std::vector<SO2d::LieAlgebra> ws = {
      SO2d::LieAlgebra(0.2), SO2d::LieAlgebra(0.5), SO2d::LieAlgebra(2e-10),
      SO2d::LieAlgebra(M_PI)};

  using LieAlgebraEndomorphism = Rp_x_SO2d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    double sigma = UniformRealDistribution(-0.5, 0.5)();
    Eigen::Matrix<double, 2, 1> X;
    X << sigma, w;
    LOGI("sigma : %f", sigma);
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism Jl = Rp_x_SO2d::Jl(X);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<Rp_x_SO2d>(X);
    LOGI(
        "Jl    : %s, %s", toStr(Jl.part<1>().row(0)).c_str(),
        toStr(Jl.part<0>().row(0)).c_str());
    LOGI(
        "Jl2   : %s, %s", toStr(Jl2.part<1>().row(0)).c_str(),
        toStr(Jl2.part<0>().row(0)).c_str());
    ASSERT_TRUE(Jl.isApprox(Jl2, 1e-6));
    // std::cout << Rp_x_SO2d::invJl(w) << std::endl;  // should report error at
    // compile time
    ASSERT_TRUE((Rp_x_SO2d::Jl(X) * Rp_x_SO2d::invJl(X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((Rp_x_SO2d::Jr(X) * Rp_x_SO2d::invJr(X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(Rp_x_SO2d::Jr(X).isApprox(Rp_x_SO2d::Jl(-X), 1e-6));
    ASSERT_TRUE(Rp_x_SO2d::invJr(X).isApprox(Rp_x_SO2d::invJl(-X), 1e-6));
  }
}

TEST(TestRp_x_SO2d, HatVee) {
  SO2d::LieAlgebra w(0.3);
  double sigma = UniformRealDistribution(-0.5, 0.5)();
  Eigen::Matrix<double, 2, 1> X;
  X << sigma, w;
  ASSERT_TRUE(X.isApprox(Rp_x_SO2d::vee(Rp_x_SO2d::hat(X)), 1e-6));
}

#ifdef TEST_CERES_JET
TEST(Test_Rp_x_SO2, Jet) {
  Rp_x_SO2d::LieAlgebra X, any_Y;
  X << 0.15, 0.3;
  any_Y = MultivariateUniformDistribution<double>::standard(Rp_x_SO2d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using Rp_x_SO2_Jet = Rp_x_SO2<Jet>;
  using LieAlgebraEndomorphism = Rp_x_SO2_Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  Rp_x_SO2_Jet::LieAlgebra jet_X = X.cast<Jet>();
  Rp_x_SO2_Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  Rp_x_SO2_Jet jet_g = Rp_x_SO2_Jet::Exp(jet_X);
  Rp_x_SO2_Jet::LieAlgebra jet_log_g = Rp_x_SO2_Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((Rp_x_SO2_Jet::Jl(jet_X) * Rp_x_SO2_Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(Rp_x_SO2_Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = Rp_x_SO2_Jet::Ad(Rp_x_SO2_Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(
              Rp_x_SO2_Jet::Ad(Rp_x_SO2_Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((Rp_x_SO2_Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(Rp_x_SO2_Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
