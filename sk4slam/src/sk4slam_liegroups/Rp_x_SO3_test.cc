#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure Rp_x_SO3 operations can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestRp_x_SO3d, Exp) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  for (auto& w : ws) {
    double sigma = UniformRealDistribution(-0.5, 0.5)();
    Eigen::Matrix<double, 4, 1> X;
    X << sigma, w;

    Rp_x_SO3d g = Rp_x_SO3d::Exp(X);
    Eigen::Matrix<double, 4, 1> log_g = Rp_x_SO3d::Log(g);
    Eigen::Matrix3d r1 = g.matrix();
    Eigen::Matrix3d r2 = exp(sigma) * SO3d::Exp(w).matrix();
    Eigen::Matrix3d r3 = expOnAlgebra(Rp_x_SO3d::hat(X).matrix());
    LOGI("sigma : %f", sigma);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI(
        "r1: %s, %s, %s", toStr(r1.row(0)).c_str(), toStr(r1.row(1)).c_str(),
        toStr(r1.row(2)).c_str());
    LOGI(
        "r2 : %s, %s, %s", toStr(r2.row(0)).c_str(), toStr(r2.row(1)).c_str(),
        toStr(r2.row(2)).c_str());
    LOGI(
        "r3 : %s, %s, %s", toStr(r3.row(0)).c_str(), toStr(r3.row(1)).c_str(),
        toStr(r3.row(2)).c_str());

    ASSERT_NEAR((r1 - r2).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((r1 - r3).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - X).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestRp_x_SO3d, Ad_ad) {
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  using LieAlgebraEndomorphism = Rp_x_SO3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    double sigma = UniformRealDistribution(-0.5, 0.5)();
    Eigen::Matrix<double, 4, 1> X;
    X << sigma, w;
    LOGI("sigma : %f", sigma);
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(Rp_x_SO3d::ad(X));
    LieAlgebraEndomorphism Ad_Exp = Rp_x_SO3d::Ad(Rp_x_SO3d::Exp(X));
    LOGI(
        "exp_ad    : %s, %s, %s, %s", toStr(exp_ad.part<1>().row(0)).c_str(),
        toStr(exp_ad.part<1>().row(1)).c_str(),
        toStr(exp_ad.part<1>().row(2)).c_str(),
        toStr(exp_ad.part<0>()).c_str());
    LOGI(
        "Ad_Exp   : %s, %s, %s, %s", toStr(Ad_Exp.part<1>().row(0)).c_str(),
        toStr(Ad_Exp.part<1>().row(1)).c_str(),
        toStr(Ad_Exp.part<1>().row(2)).c_str(),
        toStr(Ad_Exp.part<0>()).c_str());

    ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, 1e-6));

    Eigen::Matrix<double, 4, 1> any_Y =
        MultivariateUniformDistribution<double>::standard(Rp_x_SO3d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - Rp_x_SO3d::Ad(Rp_x_SO3d::Exp(X), any_Y))
            .squaredNorm(),
        0, 1e-6);
    ASSERT_NEAR(
        ((Rp_x_SO3d::ad(X) * any_Y) - Rp_x_SO3d::bracket(X, any_Y))
            .squaredNorm(),
        0, 1e-6);
  }
}

TEST(TestRp_x_SO3d, LieJacobian) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  using LieAlgebraEndomorphism = Rp_x_SO3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    double sigma = UniformRealDistribution(-0.5, 0.5)();
    Eigen::Matrix<double, 4, 1> X;
    X << sigma, w;
    LOGI("sigma : %f", sigma);
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism Jl = Rp_x_SO3d::Jl(X);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<Rp_x_SO3d>(X);
    LOGI(
        "Jl    : %s, %s, %s, %s", toStr(Jl.part<1>().row(0)).c_str(),
        toStr(Jl.part<1>().row(1)).c_str(), toStr(Jl.part<1>().row(2)).c_str(),
        toStr(Jl.part<0>()).c_str());
    LOGI(
        "Jl2   : %s, %s, %s, %s", toStr(Jl2.part<1>().row(0)).c_str(),
        toStr(Jl2.part<1>().row(1)).c_str(),
        toStr(Jl2.part<1>().row(2)).c_str(), toStr(Jl2.part<0>()).c_str());
    ASSERT_TRUE(Jl.isApprox(Jl2, 1e-6));
    // std::cout << Rp_x_SO3d::invJl(w) << std::endl;  // should report error at
    // compile time
    ASSERT_TRUE((Rp_x_SO3d::Jl(X) * Rp_x_SO3d::invJl(X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((Rp_x_SO3d::Jr(X) * Rp_x_SO3d::invJr(X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(Rp_x_SO3d::Jr(X).isApprox(Rp_x_SO3d::Jl(-X), 1e-6));
    ASSERT_TRUE(Rp_x_SO3d::invJr(X).isApprox(Rp_x_SO3d::invJl(-X), 1e-6));
  }
}

TEST(TestRp_x_SO3d, HatVee) {
  SO3d::LieAlgebra w(0.1, 0.2, 0.3);
  double sigma = UniformRealDistribution(-0.5, 0.5)();
  Eigen::Matrix<double, 4, 1> X;
  X << sigma, w;
  ASSERT_TRUE(X.isApprox(Rp_x_SO3d::vee(Rp_x_SO3d::hat(X)), 1e-6));
}

TEST(TestRp_x_SO3d, ConvertPerturbation) {
  using LieAlgebraEndomorphism = Rp_x_SO3d::LieAlgebraEndomorphism;
  using LeftPerturbation = Rp_x_SO3d::LeftPerturbation;
  using RightPerturbation = Rp_x_SO3d::RightPerturbation;

  using ScaleYawOnlyPerturbation = liegroup_internal::ProductPerturbation<
      Rp_x_SO3d, Rpd::LeftPerturbation, SO3d::YawOnlyPerturbation>;
  using YawOnlyPerturbation = liegroup_internal::SubSpacePerturbation<
      ScaleYawOnlyPerturbation, SubSpaceByAxes<1>>;
  using YawFixedPerturbation = liegroup_internal::ProductPerturbation<
      Rp_x_SO3d, Rpd::LeftPerturbation, SO3d::YawFixedPerturbation<0>>;
  using ScaleYawFixedPerturbation = liegroup_internal::SubSpacePerturbation<
      YawFixedPerturbation, SubSpaceByAxes<1, 2>>;

  using LeftOptimizable = Rp_x_SO3d::LeftOptimizable;
  using RightOptimizable = Rp_x_SO3d::RightOptimizable;
  using YawOnlyOptimizable =
      OptimizableManifold<Rp_x_SO3d, YawOnlyPerturbation>;
  using YawFixedOptimizable =
      OptimizableManifold<Rp_x_SO3d, YawFixedPerturbation>;
  using ScaleYawFixedOptimizable =
      OptimizableManifold<Rp_x_SO3d, ScaleYawFixedPerturbation>;

  SO3d rot = SO3d::Exp(Eigen::Vector3d(0, 0.1, 0)) *
             SO3d::Exp(Eigen::Vector3d(0.2, 0, 0));
  Rp_x_SO3d g(2.0, rot);
  LOGI("g:\n%s", toStr(g.matrix()).c_str());

  Rp_x_SO3d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(Rp_x_SO3d::kDim)();
  LOGI("any_Y:\n%s", toStr(any_Y.transpose()).c_str());

  // left perturbation <-> right perturbation
  {
    LOGI("******** left perturbation <-> right perturbation ******** \n");
    const double approx_radius =
        1;  // conversion between left and right perturbations is exact, we
            // don't need approximation for this case
    const double approx_precision = 1e-6;  // use a high precision for this case
    auto delta_l = any_Y * approx_radius;
    auto delta_r =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(delta_l);
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    auto g_r = (RightOptimizable(g) + delta_r).value();
    LOGI("g_l:\n%s", toStr(g_l.matrix()).c_str());
    LOGI("g_r:\n%s", toStr(g_r.matrix()).c_str());
    ASSERT_TRUE(g_l.isApprox(g_r, 1e-6));

    auto delta_l2 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(delta_r);
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, 1e-6));

    Rp_x_SO3d::LieAlgebra delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>() * delta_l;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    Rp_x_SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>() * delta_r;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, approx_precision));
  }

  // yaw-fixed perturbation -> left perturbation
  {
    LOGI(
        "******** yaw-fixed perturbation -> left perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector3d delta_yf = any_Y.head<3>() * approx_radius;
    auto delta_l =
        g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>(delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_yf = (YawFixedOptimizable(g) + delta_yf).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_yf:\n%s", toStr(g_yf.matrix()).c_str());
    LOGI("g_l:\n%s", toStr(g_l.matrix()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_l, 1e-6));

    Rp_x_SO3d::LieAlgebra delta_l2 =
        g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>() *
        delta_yf;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    Rp_x_SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(
            g.convertPerturbation<YawFixedPerturbation, RightPerturbation>(
                delta_yf));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));
  }

  // yaw-fixed perturbation -> right perturbation
  {
    LOGI(
        "******** yaw-fixed perturbation -> right perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector3d delta_yf = any_Y.head<3>() * approx_radius;
    auto delta_r =
        g.convertPerturbation<YawFixedPerturbation, RightPerturbation>(
            delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    auto g_yf = (YawFixedOptimizable(g) + delta_yf).value();
    auto g_r = (RightOptimizable(g) + delta_r).value();
    LOGI("g_yf:\n%s", toStr(g_yf.matrix()).c_str());
    LOGI("g_r:\n%s", toStr(g_r.matrix()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_r, 1e-6));

    Rp_x_SO3d::LieAlgebra delta_r2 =
        g.convertPerturbation<YawFixedPerturbation, RightPerturbation>() *
        delta_yf;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    Rp_x_SO3d::LieAlgebra delta_r3 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(
            g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>(
                delta_yf));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r3:\n%s", toStr(delta_r3.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r3, 1e-6));
  }

  // scale-yaw-fixed perturbation -> left perturbation
  {
    LOGI(
        "******** scale-yaw-fixed perturbation -> left perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector2d delta_yf = any_Y.head<2>() * approx_radius;
    auto delta_l =
        g.convertPerturbation<ScaleYawFixedPerturbation, LeftPerturbation>(
            delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_yf = (ScaleYawFixedOptimizable(g) + delta_yf).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_yf:\n%s", toStr(g_yf.matrix()).c_str());
    LOGI("g_l:\n%s", toStr(g_l.matrix()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_l, 1e-6));

    Rp_x_SO3d::LieAlgebra delta_l2 =
        g.convertPerturbation<ScaleYawFixedPerturbation, LeftPerturbation>() *
        delta_yf;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    Rp_x_SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(
            g.convertPerturbation<ScaleYawFixedPerturbation, RightPerturbation>(
                delta_yf));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));
  }

  // yaw-only perturbation -> left perturbation
  {
    LOGI(
        "******** yaw-only perturbation -> left perturbation ******** "
        "\n");
    const double approx_radius =
        1;  // conversion between left and right perturbations is exact, we
            // don't need approximation for this case
    const double approx_precision = 1e-6;  // use a high precision for this case
    Eigen::Matrix<double, 1, 1> delta_y = any_Y.head<1>() * approx_radius;
    auto delta_l =
        g.convertPerturbation<YawOnlyPerturbation, LeftPerturbation>(delta_y);
    LOGI("delta_y:\n%s", toStr(delta_y.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_y = (YawOnlyOptimizable(g) + delta_y).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_y:\n%s", toStr(g_y.matrix()).c_str());
    LOGI("g_l:\n%s", toStr(g_l.matrix()).c_str());
    ASSERT_TRUE(g_y.isApprox(g_l, 1e-6));

    Rp_x_SO3d::LieAlgebra delta_l2 =
        g.convertPerturbation<YawOnlyPerturbation, LeftPerturbation>() *
        delta_y;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    Rp_x_SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(
            g.convertPerturbation<YawOnlyPerturbation, RightPerturbation>(
                delta_y));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));
  }

  // yaw-only perturbation -> right perturbation
  {
    LOGI(
        "******** yaw-only perturbation -> right perturbation ******** "
        "\n");
    const double approx_radius =
        1;  // conversion between left and right perturbations is exact, we
            // don't need approximation for this case
    const double approx_precision = 1e-6;  // use a high precision for this case
    Eigen::Matrix<double, 1, 1> delta_y = any_Y.head<1>() * approx_radius;
    auto delta_r =
        g.convertPerturbation<YawOnlyPerturbation, RightPerturbation>(delta_y);
    LOGI("delta_y:\n%s", toStr(delta_y.transpose()).c_str());
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    auto g_y = (YawOnlyOptimizable(g) + delta_y).value();
    auto g_r = (RightOptimizable(g) + delta_r).value();
    LOGI("g_y:\n%s", toStr(g_y.matrix()).c_str());
    LOGI("g_r:\n%s", toStr(g_r.matrix()).c_str());
    ASSERT_TRUE(g_y.isApprox(g_r, 1e-6));

    Rp_x_SO3d::LieAlgebra delta_r2 =
        g.convertPerturbation<YawOnlyPerturbation, RightPerturbation>() *
        delta_y;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));
    Rp_x_SO3d::LieAlgebra delta_r3 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(
            g.convertPerturbation<YawOnlyPerturbation, LeftPerturbation>(
                delta_y));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r3:\n%s", toStr(delta_r3.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r3, 1e-6));
  }
}

#ifdef TEST_CERES_JET
TEST(Test_Rp_x_SO3, Jet) {
  Rp_x_SO3d::LieAlgebra X, any_Y;
  X << 0.15, 0.1, 0.2, 0.3;
  any_Y = MultivariateUniformDistribution<double>::standard(Rp_x_SO3d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using Rp_x_SO3_Jet = Rp_x_SO3<Jet>;
  using LieAlgebraEndomorphism = Rp_x_SO3_Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  Rp_x_SO3_Jet::LieAlgebra jet_X = X.cast<Jet>();
  Rp_x_SO3_Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  Rp_x_SO3_Jet jet_g = Rp_x_SO3_Jet::Exp(jet_X);
  Rp_x_SO3_Jet::LieAlgebra jet_log_g = Rp_x_SO3_Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((Rp_x_SO3_Jet::Jl(jet_X) * Rp_x_SO3_Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(Rp_x_SO3_Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = Rp_x_SO3_Jet::Ad(Rp_x_SO3_Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(
              Rp_x_SO3_Jet::Ad(Rp_x_SO3_Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((Rp_x_SO3_Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(Rp_x_SO3_Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
