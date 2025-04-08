#include "sk4slam_liegroups/SO3.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure the SO3 can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestSO3d, Exp) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  for (auto& w : ws) {
    SO3d g = SO3d::Exp(w);
    Eigen::Matrix3d exp_w = expOnAlgebra(SO3d::hat(w));
    Eigen::Vector3d log_g = SO3d::Log(g);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI(
        "SO3d g: %s, %s, %s", toStr(g.matrix().row(0)).c_str(),
        toStr(g.matrix().row(1)).c_str(), toStr(g.matrix().row(2)).c_str());
    LOGI(
        "exp_w : %s, %s, %s", toStr(exp_w.row(0)).c_str(),
        toStr(exp_w.row(1)).c_str(), toStr(exp_w.row(2)).c_str());

    ASSERT_NEAR((g.matrix() - exp_w).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - w).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestSO3d, Ad_ad) {
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  using LieAlgebraEndomorphism = SO3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(SO3d::ad(w));
    LieAlgebraEndomorphism Ad_Exp = SO3d::Ad(SO3d::Exp(w));
    LOGI(
        "exp_ad    : %s, %s, %s", toStr(exp_ad.row(0)).c_str(),
        toStr(exp_ad.row(1)).c_str(), toStr(exp_ad.row(2)).c_str());
    LOGI(
        "Ad_Exp   : %s, %s, %s", toStr(Ad_Exp.row(0)).c_str(),
        toStr(Ad_Exp.row(1)).c_str(), toStr(Ad_Exp.row(2)).c_str());
    ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

    auto any_Y =
        MultivariateUniformDistribution<double>::standard(SO3d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - SO3d::Ad(SO3d::Exp(w), any_Y)).squaredNorm(), 0,
        1e-6);
    ASSERT_NEAR(
        ((SO3d::ad(w) * any_Y) - SO3d::bracket(w, any_Y)).squaredNorm(), 0,
        1e-6);
  }
}

TEST(TestSO3d, LieJacobian) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  using LieAlgebraEndomorphism = SO3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LieAlgebraEndomorphism Jl = SO3d::Jl(w);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<SO3d>(w);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI(
        "Jl    : %s, %s, %s", toStr(Jl.row(0)).c_str(),
        toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str());
    LOGI(
        "Jl2   : %s, %s, %s", toStr(Jl2.row(0)).c_str(),
        toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str());
    ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

    ASSERT_TRUE((SO3d::Jl(w) * SO3d::invJl(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((SO3d::Jr(w) * SO3d::invJr(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(SO3d::Jr(w).isApprox(SO3d::Jl(-w), 1e-6));
    ASSERT_TRUE(SO3d::invJr(w).isApprox(SO3d::invJl(-w), 1e-6));
  }
}

TEST(TestSO3d, HatVee) {
  SO3d::LieAlgebra w(0.1, 0.2, 0.3);
  ASSERT_TRUE(w.isApprox(SO3d::vee(SO3d::hat(w)), 1e-6));
}

TEST(TestSO3d, YawFixed) {
  using YawFixedPerturbation = SO3d::YawFixedPerturbation<0>;
  using YawFixedOptimizable = SO3d::YawFixed<0>;
  SO3d g = SO3d::Exp(Eigen::Vector3d(0, 0.1, 0)) *
           SO3d::Exp(Eigen::Vector3d(0.2, 0, 0));
  LOGI("g:\n%s", toStr(g.matrix()).c_str());
  Eigen::Vector2d delta =
      MultivariateUniformDistribution<double>::standard(2)();
  LOGI("delta:\n%s", toStr(delta.transpose()).c_str());
  SO3d g2 = (YawFixedOptimizable(g) + delta).value();
  LOGI("g2:\n%s", toStr(g2.matrix()).c_str());
  Eigen::Vector2d delta2 = YawFixedOptimizable(g2) - YawFixedOptimizable(g);
  LOGI("delta:\n%s", toStr(delta.transpose()).c_str());
  LOGI("delta2:\n%s", toStr(delta2.transpose()).c_str());
  ASSERT_NEAR((delta - delta2).squaredNorm(), 0, 1e-6);
}

TEST(TestSO3d, ConvertPerturbation) {
  using LieAlgebraEndomorphism = SO3d::LieAlgebraEndomorphism;
  using LeftPerturbation = SO3d::LeftPerturbation;
  using RightPerturbation = SO3d::RightPerturbation;
  using YawOnlyPerturbation = SO3d::YawOnlyPerturbation;
  using YawFixedPerturbation = SO3d::YawFixedPerturbation<0>;
  using LeftOptimizable = SO3d::LeftOptimizable;
  using RightOptimizable = SO3d::RightOptimizable;
  using YawOnlyOptimizable = SO3d::YawOnly;
  using YawFixedOptimizable = SO3d::YawFixed<0>;
  SO3d g = SO3d::Exp(Eigen::Vector3d(0, 0.1, 0)) *
           SO3d::Exp(Eigen::Vector3d(0.2, 0, 0));
  LOGI("g:\n%s", toStr(g.matrix()).c_str());

  SO3d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(SO3d::kDim)();
  LOGI("any_Y:\n%s", toStr(any_Y.transpose()).c_str());

  Eigen::Matrix<double, 4, LeftPerturbation::kDof> any_J =
      Eigen::Matrix<double, 4, LeftPerturbation::kDof>::Random();

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

    SO3d::LieAlgebra delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>() * delta_l;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>() * delta_r;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, approx_precision));

    ASSERT_TRUE(
        (LeftOptimizable(g_l) - LeftOptimizable(g)).isApprox(delta_l, 1e-6));
    ASSERT_TRUE(
        (RightOptimizable(g_r) - RightOptimizable(g)).isApprox(delta_r, 1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<RightPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<RightPerturbation, LeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<LeftPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<LeftPerturbation, RightPerturbation>(any_J),
                1e-6));
  }

  // yaw-fixed perturbation -> left perturbation
  {
    LOGI(
        "******** yaw-fixed perturbation -> left perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector2d delta_yf = any_Y.head<2>() * approx_radius;
    auto delta_l =
        g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>(delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_yf = (YawFixedOptimizable(g) + delta_yf).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_yf:\n%s", toStr(g_yf.matrix()).c_str());
    LOGI("g_l:\n%s", toStr(g_l.matrix()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_l, 1e-6));

    SO3d::LieAlgebra delta_l2 =
        g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>() *
        delta_yf;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(
            g.convertPerturbation<YawFixedPerturbation, RightPerturbation>(
                delta_yf));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));

    ASSERT_TRUE(
        (LeftOptimizable(g_l) - LeftOptimizable(g)).isApprox(delta_l, 1e-6));
    ASSERT_TRUE((YawFixedOptimizable(g_yf) - YawFixedOptimizable(g))
                    .isApprox(delta_yf, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<YawFixedPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
  }

  // yaw-fixed perturbation -> right perturbation
  {
    LOGI(
        "******** yaw-fixed perturbation -> right perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector2d delta_yf = any_Y.head<2>() * approx_radius;
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

    SO3d::LieAlgebra delta_r2 =
        g.convertPerturbation<YawFixedPerturbation, RightPerturbation>() *
        delta_yf;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    SO3d::LieAlgebra delta_r3 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(
            g.convertPerturbation<YawFixedPerturbation, LeftPerturbation>(
                delta_yf));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r3:\n%s", toStr(delta_r3.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r3, 1e-6));

    ASSERT_TRUE(
        (RightOptimizable(g_r) - RightOptimizable(g)).isApprox(delta_r, 1e-6));
    ASSERT_TRUE((YawFixedOptimizable(g_yf) - YawFixedOptimizable(g))
                    .isApprox(delta_yf, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawFixedPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<YawFixedPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
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

    SO3d::LieAlgebra delta_l2 =
        g.convertPerturbation<YawOnlyPerturbation, LeftPerturbation>() *
        delta_y;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    SO3d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(
            g.convertPerturbation<YawOnlyPerturbation, RightPerturbation>(
                delta_y));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));

    ASSERT_TRUE(
        (LeftOptimizable(g_l) - LeftOptimizable(g)).isApprox(delta_l, 1e-6));
    ASSERT_TRUE((YawOnlyOptimizable(g_y) - YawOnlyOptimizable(g))
                    .isApprox(delta_y, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawOnlyPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<YawOnlyPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
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

    SO3d::LieAlgebra delta_r2 =
        g.convertPerturbation<YawOnlyPerturbation, RightPerturbation>() *
        delta_y;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));
    SO3d::LieAlgebra delta_r3 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(
            g.convertPerturbation<YawOnlyPerturbation, LeftPerturbation>(
                delta_y));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r3:\n%s", toStr(delta_r3.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r3, 1e-6));

    ASSERT_TRUE(
        (RightOptimizable(g_r) - RightOptimizable(g)).isApprox(delta_r, 1e-6));
    ASSERT_TRUE((YawOnlyOptimizable(g_y) - YawOnlyOptimizable(g))
                    .isApprox(delta_y, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawOnlyPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<YawOnlyPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
  }
}

#ifdef TEST_CERES_JET
TEST(Test_SO3_, Jet) {
  SO3d::LieAlgebra X, any_Y;
  X << 0.1, 0.2, 0.3;
  any_Y = MultivariateUniformDistribution<double>::standard(SO3d::kDim)();

  std::vector<SO3d::LieAlgebra> Xs = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}, {M_PI, 0, 0}};

  using Jet = ceres::Jet<double, 7>;
  using SO3Jet = SO3<Jet>;
  using LieAlgebraEndomorphism = SO3Jet::LieAlgebraEndomorphism;
  const Jet Jet_eps(1e-6);
  for (const auto& X : Xs) {
    SO3Jet::LieAlgebra jet_X = X.cast<Jet>();
    SO3Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
    SO3Jet jet_g = SO3Jet::Exp(jet_X);
    SO3Jet::LieAlgebra jet_log_g = SO3Jet::Log(jet_g);

    LOGI("jet_X     : %s", toStr(jet_X.transpose(), Precision(20)).c_str());
    LOGI("jet_log_g : %s", toStr(jet_log_g.transpose(), Precision(20)).c_str());

    auto diff = jet_log_g - jet_X;
    auto diff_norm = diff.norm();
    LOGI("diff     : %s", toStr(diff.transpose(), Precision(20)).c_str());
    LOGI("diff_norm : %s", toStr(diff_norm, Precision(20)).c_str());
    LOGI("Jet_eps : %s", toStr(Jet_eps, Precision(11)).c_str());

    ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
    ASSERT_TRUE((SO3Jet::Jl(jet_X) * SO3Jet::invJl(jet_X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(SO3Jet::ad(jet_X));
    LieAlgebraEndomorphism Ad_Exp = SO3Jet::Ad(SO3Jet::Exp(jet_X));
    ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
    ASSERT_TRUE(
        (Ad_Exp * jet_any_Y)
            .isApprox(SO3Jet::Ad(SO3Jet::Exp(jet_X), jet_any_Y), Jet_eps));
    ASSERT_TRUE((SO3Jet::ad(jet_X) * jet_any_Y)
                    .isApprox(SO3Jet::bracket(jet_X, jet_any_Y), Jet_eps));
  }
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
