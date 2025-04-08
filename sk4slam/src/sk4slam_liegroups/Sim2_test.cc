#include "sk4slam_liegroups/Sim2.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SO2.h"
#include "sk4slam_liegroups/SubGLn_rx_Rn.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure the Sim2 operations can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(Test_Sim2, Exp) {
  Sim2d::LieAlgebra X;
  X << 0.15, 0.3, 0.5, 0.6;
  Eigen::Matrix<double, 3, 3> hatX = Eigen::Matrix<double, 3, 3>::Zero();
  hatX.block<2, 2>(0, 0) = SO2d::hat(X.segment<1>(1));
  hatX(0, 0) = X(0);
  hatX(1, 1) = X(0);
  hatX.block<2, 1>(0, 2) = X.tail<2>();
  Eigen::Matrix<double, 3, 3> exp_hatX = expOnAlgebra(hatX);

  Sim2d g = Sim2d::Exp(X);
  Sim2d::LieAlgebra log_g = Sim2d::Log(g);
  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
  double scale = sqrt(g.linear().matrix().squaredNorm() / 2.0);
  LOGI("Sim2d g.scale   : %f", scale);
  LOGI(
      "Sim2d g.rotation   : %s, %s",
      toStr(g.linear().matrix().row(0) / scale).c_str(),
      toStr(g.linear().matrix().row(1) / scale).c_str());
  LOGI(
      "Sim2d g.scaled_rotation   : %s, %s",
      toStr(g.linear().matrix().row(0)).c_str(),
      toStr(g.linear().matrix().row(1)).c_str());
  LOGI("Sim2d g.translation  : %s", toStr(g.translation().transpose()).c_str());
  LOGI(
      "exp_hatX : %s, %s, %s", toStr(exp_hatX.row(0)).c_str(),
      toStr(exp_hatX.row(1)).c_str(), toStr(exp_hatX.row(2)).c_str());

  ASSERT_NEAR(
      (g.linear().matrix() - exp_hatX.block<2, 2>(0, 0)).squaredNorm(), 0,
      1e-6);
  ASSERT_NEAR(
      (g.translation() - exp_hatX.col(2).head<2>()).squaredNorm(), 0, 1e-6);
  ASSERT_NEAR((log_g - X).squaredNorm(), 0, 1e-6);
}

TEST(Test_Sim2, Ad_ad) {
  Sim2d::LieAlgebra X;
  X << 0.15, 0.3, 0.5, 0.6;
  using LieAlgebraEndomorphism = Sim2d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(Sim2d::ad(X));
  LieAlgebraEndomorphism Ad_Exp = Sim2d::Ad(Sim2d::Exp(X));
  LOGI(
      "exp_ad    : %s, %s, %s, %s", toStr(exp_ad.row(0)).c_str(),
      toStr(exp_ad.row(1)).c_str(), toStr(exp_ad.row(2)).c_str(),
      toStr(exp_ad.row(3)).c_str());
  LOGI(
      "Ad_Exp   : %s, %s, %s, %s", toStr(Ad_Exp.row(0)).c_str(),
      toStr(Ad_Exp.row(1)).c_str(), toStr(Ad_Exp.row(2)).c_str(),
      toStr(Ad_Exp.row(3)).c_str());
  ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

  Sim2d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(Sim2d::kDim)();
  ASSERT_NEAR(
      ((Ad_Exp * any_Y) - Sim2d::Ad(Sim2d::Exp(X), any_Y)).squaredNorm(), 0,
      1e-6);
  ASSERT_NEAR(
      ((Sim2d::ad(X) * any_Y) - Sim2d::bracket(X, any_Y)).squaredNorm(), 0,
      1e-6);
}

TEST(Test_Sim2, LieJacobian) {
  Sim2d::LieAlgebra X;
  X << 0.15, 0.3, 0.5, 0.6;
  using LieAlgebraEndomorphism = Sim2d::LieAlgebraEndomorphism;

  std::vector<Sim2d::LieAlgebra> Xs;
  Xs.push_back(X);
  Xs.push_back(-X);
  X[0] = 0.0;  // sigma = 0
  Xs.push_back(X);
  Xs.push_back(-X);
  for (const auto& X : Xs) {
    LOGI("X     : %s", toStr(X.transpose()).c_str());
    LieAlgebraEndomorphism Jl = Sim2d::Jl(X);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<Sim2d>(X);
    LOGI(
        "Jl    : %s, %s, %s, %s", toStr(Jl.row(0)).c_str(),
        toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str(),
        toStr(Jl.row(3)).c_str());
    LOGI(
        "Jl2   : %s, %s, %s, %s", toStr(Jl2.row(0)).c_str(),
        toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str(),
        toStr(Jl2.row(3)).c_str());
    ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

    LieAlgebraEndomorphism invJl = Sim2d::invJl(X);
    LieAlgebraEndomorphism invJl2 = Jl2.inverse();
    LOGI(
        "invJl    : %s, %s, %s, %s", toStr(invJl.row(0)).c_str(),
        toStr(invJl.row(1)).c_str(), toStr(invJl.row(2)).c_str(),
        toStr(invJl.row(3)).c_str());
    LOGI(
        "invJl2   : %s, %s, %s, %s", toStr(invJl2.row(0)).c_str(),
        toStr(invJl2.row(1)).c_str(), toStr(invJl2.row(2)).c_str(),
        toStr(invJl2.row(3)).c_str());

    LieAlgebraEndomorphism tmp = Sim2d::Jl(X) * Sim2d::invJl(X);
    LOGI(
        "Jl * invJl: %s, %s, %s, %s", toStr(tmp.row(0)).c_str(),
        toStr(tmp.row(1)).c_str(), toStr(tmp.row(2)).c_str(),
        toStr(tmp.row(3)).c_str());

    ASSERT_TRUE((Sim2d::Jl(X) * Sim2d::invJl(X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((Sim2d::Jr(X) * Sim2d::invJr(X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(Sim2d::Jr(X).isApprox(Sim2d::Jl(-X), 1e-6));
    ASSERT_TRUE(Sim2d::invJr(X).isApprox(Sim2d::invJl(-X), 1e-6));
  }
}

TEST(Test_Sim2, HatVee) {
  Sim2d::LieAlgebra X;
  X << 0.15, 0.3, 0.5, 0.6;
  ASSERT_TRUE(X.isApprox(Sim2d::vee(Sim2d::hat(X)), 1e-6));
}

TEST(Test_Sim2, ConvertPerturbation) {
  using LieAlgebraEndomorphism = Sim2d::LieAlgebraEndomorphism;
  using LeftPerturbation = Sim2d::LeftPerturbation;
  using RightPerturbation = Sim2d::RightPerturbation;
  using AffineLeftPerturbation = Sim2d::AffineLeftPerturbation;
  using AffineRightPerturbation = Sim2d::AffineRightPerturbation;
  using LeftOptimizable = Sim2d::LeftOptimizable;
  using RightOptimizable = Sim2d::RightOptimizable;
  using SeparateLeftOptimizable = Sim2d::SeparateLeftOptimizable;
  using SeparateRightOptimizable = Sim2d::SeparateRightOptimizable;
  Sim2d::LieAlgebra X;
  X << 0.15, 0.3, 0.5, 0.6;
  Sim2d g = Sim2d::Exp(X);
  LOGI("X:\n%s", toStr(X.transpose()).c_str());
  LOGI("g.linear():\n%s", toStr(g.linear().matrix()).c_str());
  LOGI("g.translation():\n%s", toStr(g.translation().transpose()).c_str());

  Sim2d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(Sim2d::kDim)();
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
    LOGI("g_l.linear():\n%s", toStr(g_l.linear().matrix()).c_str());
    LOGI(
        "g_l.translation():\n%s", toStr(g_l.translation().transpose()).c_str());
    LOGI("g_r.linear():\n%s", toStr(g_r.linear().matrix()).c_str());
    LOGI(
        "g_r.translation():\n%s", toStr(g_r.translation().transpose()).c_str());
    ASSERT_TRUE(g_l.isApprox(g_r, 1e-6));

    auto delta_l2 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(delta_r);
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, 1e-6));

    Sim2d::LieAlgebra delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>() * delta_l;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    Sim2d::LieAlgebra delta_l3 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>() * delta_r;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, approx_precision));
  }

  // separate left perturbation <-> separate right perturbation
  {
    LOGI(
        "******** separate left perturbation <-> separate right perturbation "
        "******** \n");
    const double approx_radius =
        1;  // conversion between separate-left and separate-right
            // perturbations is exact, we don't need approximation
            // for this case
    const double approx_precision = 1e-6;  // use a high precision for this case
    Sim2d::LieAlgebra delta_sl = any_Y * approx_radius;
    auto delta_sr =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            delta_sl);
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    auto g_sl = (SeparateLeftOptimizable(g) + delta_sl).value();
    auto g_sr = (SeparateRightOptimizable(g) + delta_sr).value();
    LOGI("g_sl.linear():\n%s", toStr(g_sl.linear().matrix()).c_str());
    LOGI(
        "g_sl.translation():\n%s",
        toStr(g_sl.translation().transpose()).c_str());
    LOGI("g_sr.linear():\n%s", toStr(g_sr.linear().matrix()).c_str());
    LOGI(
        "g_sr.translation():\n%s",
        toStr(g_sr.translation().transpose()).c_str());
    ASSERT_TRUE(g_sl.isApprox(g_sr, 1e-6));

    auto delta_sl2 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            delta_sr);
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, 1e-6));

    Sim2d::LieAlgebra delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            delta_sl);
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    Sim2d::LieAlgebra delta_sl3 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            delta_sr);
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl3:\n%s", toStr(delta_sl3.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl3, 1e-6));
  }

  // left perturbation <-> separate left perturbation
  {
    LOGI(
        "******** left perturbation <-> separate left perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    auto delta_l = any_Y * approx_radius;
    auto delta_sl =
        g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>(
            delta_l);
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    auto g_sl = (SeparateLeftOptimizable(g) + delta_sl).value();
    LOGI("g_l.linear():\n%s", toStr(g_l.linear().matrix()).c_str());
    LOGI(
        "g_l.translation():\n%s", toStr(g_l.translation().transpose()).c_str());
    LOGI("g_sl.linear():\n%s", toStr(g_sl.linear().matrix()).c_str());
    LOGI(
        "g_sl.translation():\n%s",
        toStr(g_sl.translation().transpose()).c_str());
    ASSERT_TRUE(g_l.isApprox(g_sl, 1e-6));

    auto delta_l2 =
        g.convertPerturbation<AffineLeftPerturbation, LeftPerturbation>(
            delta_sl);
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, 1e-6));

    LOGI(
        "tmp:\n%s",
        toStr(g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>())
            .c_str());
    Sim2d::LieAlgebra delta_sl2 =
        g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>() *
        delta_l;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    Sim2d::LieAlgebra delta_l3 =
        g.convertPerturbation<AffineLeftPerturbation, LeftPerturbation>() *
        delta_sl;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, approx_precision));
  }

  // right perturbation -> separate left perturbation
  {
    LOGI(
        "******** right perturbation <-> separate left perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    auto delta_r = any_Y * approx_radius;
    auto delta_sl =
        g.convertPerturbation<RightPerturbation, AffineLeftPerturbation>(
            delta_r);
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    auto g_r = (RightOptimizable(g) + delta_r).value();
    auto g_sl = (SeparateLeftOptimizable(g) + delta_sl).value();
    LOGI("g_r.linear():\n%s", toStr(g_r.linear().matrix()).c_str());
    LOGI(
        "g_r.translation():\n%s", toStr(g_r.translation().transpose()).c_str());
    LOGI("g_sl.linear():\n%s", toStr(g_sl.linear().matrix()).c_str());
    LOGI(
        "g_sl.translation():\n%s",
        toStr(g_sl.translation().transpose()).c_str());
    ASSERT_TRUE(g_r.isApprox(g_sl, 1e-6));

    auto delta_r2 =
        g.convertPerturbation<AffineLeftPerturbation, RightPerturbation>(
            delta_sl);
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, 1e-6));

    Sim2d::LieAlgebra delta_sl2 =
        g.convertPerturbation<RightPerturbation, AffineLeftPerturbation>() *
        delta_r;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    Sim2d::LieAlgebra delta_r3 =
        g.convertPerturbation<AffineLeftPerturbation, RightPerturbation>() *
        delta_sl;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r3:\n%s", toStr(delta_r3.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r3, approx_precision));

    // test the conversion loop:
    //
    // right perturbation --> left perturbation --> separate left perturbation
    //        ^                                          |
    //        |-------------------------------------------
    //
    auto delta_r4 =
        g.convertPerturbation<AffineLeftPerturbation, RightPerturbation>(
            g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>(
                g.convertPerturbation<RightPerturbation, LeftPerturbation>(
                    delta_r)));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r4:\n%s", toStr(delta_r4.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r4, 1e-6));
  }

  // right perturbation <-> separate right perturbation
  {
    LOGI(
        "******** right perturbation <-> separate right perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    auto delta_r = any_Y * approx_radius;
    auto delta_sr =
        g.convertPerturbation<RightPerturbation, AffineRightPerturbation>(
            delta_r);
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    auto g_r = (RightOptimizable(g) + delta_r).value();
    auto g_sr = (SeparateRightOptimizable(g) + delta_sr).value();
    LOGI("g_r.linear():\n%s", toStr(g_r.linear().matrix()).c_str());
    LOGI(
        "g_r.translation():\n%s", toStr(g_r.translation().transpose()).c_str());
    LOGI("g_sr.linear():\n%s", toStr(g_sr.linear().matrix()).c_str());
    LOGI(
        "g_sr.translation():\n%s",
        toStr(g_sr.translation().transpose()).c_str());
    ASSERT_TRUE(g_r.isApprox(g_sr, 1e-6));

    auto delta_r2 =
        g.convertPerturbation<AffineRightPerturbation, RightPerturbation>(
            delta_sr);
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, 1e-6));

    Sim2d::LieAlgebra delta_sr2 =
        g.convertPerturbation<RightPerturbation, AffineRightPerturbation>() *
        delta_r;
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, approx_precision));

    Sim2d::LieAlgebra delta_r3 =
        g.convertPerturbation<AffineRightPerturbation, RightPerturbation>() *
        delta_sr;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r3:\n%s", toStr(delta_r3.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r3, approx_precision));

    // test the conversion loop:
    //
    // right perturbation --> separate left  --> separate right
    //        ^                                          |
    //        |-------------------------------------------
    //
    auto delta_r4 = g.convertPerturbation<
        AffineRightPerturbation, RightPerturbation>(
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            g.convertPerturbation<RightPerturbation, AffineLeftPerturbation>(
                delta_r)));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r4:\n%s", toStr(delta_r4.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r4, 1e-6));
  }

  // left perturbation <-> separate right perturbation
  {
    LOGI(
        "******** left perturbation <-> separate right perturbation ******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    auto delta_l = any_Y * approx_radius;
    auto delta_sr =
        g.convertPerturbation<LeftPerturbation, AffineRightPerturbation>(
            delta_l);
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    auto g_sr = (SeparateRightOptimizable(g) + delta_sr).value();
    LOGI("g_l.linear():\n%s", toStr(g_l.linear().matrix()).c_str());
    LOGI(
        "g_l.translation():\n%s", toStr(g_l.translation().transpose()).c_str());
    LOGI("g_sr.linear():\n%s", toStr(g_sr.linear().matrix()).c_str());
    LOGI(
        "g_sr.translation():\n%s",
        toStr(g_sr.translation().transpose()).c_str());
    ASSERT_TRUE(g_l.isApprox(g_sr, 1e-6));

    auto delta_l2 =
        g.convertPerturbation<AffineRightPerturbation, LeftPerturbation>(
            delta_sr);
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, 1e-6));

    Sim2d::LieAlgebra delta_sr2 =
        g.convertPerturbation<LeftPerturbation, AffineRightPerturbation>() *
        delta_l;
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, approx_precision));

    Sim2d::LieAlgebra delta_l3 =
        g.convertPerturbation<AffineRightPerturbation, LeftPerturbation>() *
        delta_sr;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, approx_precision));

    // test the conversion loop:
    //
    // left perturbation --> separate left  --> separate right
    //        ^                                          |
    //        |-------------------------------------------
    //
    auto delta_l4 = g.convertPerturbation<
        AffineRightPerturbation, LeftPerturbation>(
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>(
                delta_l)));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l4:\n%s", toStr(delta_l4.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l4, 1e-6));
  }
}

#ifdef TEST_CERES_JET
TEST(Test_Sim2, Jet) {
  Sim2d::LieAlgebra X, any_Y;
  X << 0.15, 0.3, 0.5, 0.6;
  any_Y = MultivariateUniformDistribution<double>::standard(Sim2d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using Sim2Jet = Sim2<Jet>;
  using LieAlgebraEndomorphism = Sim2Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  Sim2Jet::LieAlgebra jet_X = X.cast<Jet>();
  Sim2Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  Sim2Jet jet_g = Sim2Jet::Exp(jet_X);
  Sim2Jet::LieAlgebra jet_log_g = Sim2Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((Sim2Jet::Jl(jet_X) * Sim2Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(Sim2Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = Sim2Jet::Ad(Sim2Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(Sim2Jet::Ad(Sim2Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((Sim2Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(Sim2Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
