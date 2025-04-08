#include "sk4slam_liegroups/SE3.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure the SE3 operations can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(TestSE3d, Exp) {
  SE3d::LieAlgebra X;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  Eigen::Matrix<double, 4, 4> hatX = Eigen::Matrix<double, 4, 4>::Zero();
  hatX.block<3, 3>(0, 0) = SO3d::hat(Eigen::Vector3d(X.head<3>()));
  hatX.block<3, 1>(0, 3) = X.tail<3>();
  Eigen::Matrix<double, 4, 4> exp_hatX = expOnAlgebra(hatX);

  SE3d g = SE3d::Exp(X);
  SE3d::LieAlgebra log_g = SE3d::Log(g);
  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
  LOGI(
      "SE3d g.rotation   : %s, %s, %s",
      toStr(g.linear().matrix().row(0)).c_str(),
      toStr(g.linear().matrix().row(1)).c_str(),
      toStr(g.linear().matrix().row(2)).c_str());
  LOGI("SE3d g.translation  : %s", toStr(g.translation().transpose()).c_str());
  // LOGI(
  //     "exp_hatX.rotation : %s, %s, %s",
  //     toStr(exp_hatX.block<3,3>(0,0).row(0)).c_str(),
  //     toStr(exp_hatX.block<3,3>(0,0).row(1)).c_str(),
  //     toStr(exp_hatX.block<3,3>(0,0).row(2)).c_str());
  // LOGI(
  //     "exp_hatX.translation: %s",
  //       toStr(exp_hatX.block<3,1>(0,3).transpose()).c_str());
  LOGI(
      "exp_hatX : %s, %s, %s, %s", toStr(exp_hatX.row(0)).c_str(),
      toStr(exp_hatX.row(1)).c_str(), toStr(exp_hatX.row(2)).c_str(),
      toStr(exp_hatX.row(3)).c_str());

  ASSERT_NEAR(
      (g.linear().matrix() - exp_hatX.block<3, 3>(0, 0)).squaredNorm(), 0,
      1e-6);
  ASSERT_NEAR(
      (g.translation() - exp_hatX.col(3).head<3>()).squaredNorm(), 0, 1e-6);
  ASSERT_NEAR((log_g - X).squaredNorm(), 0, 1e-6);
}

TEST(TestSE3d, Ad_ad) {
  SE3d::LieAlgebra X;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  using LieAlgebraEndomorphism = SE3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(SE3d::ad(X));
  LieAlgebraEndomorphism Ad_Exp = SE3d::Ad(SE3d::Exp(X));
  LOGI(
      "exp_ad    : %s, %s, %s, %s, %s, %s", toStr(exp_ad.row(0)).c_str(),
      toStr(exp_ad.row(1)).c_str(), toStr(exp_ad.row(2)).c_str(),
      toStr(exp_ad.row(3)).c_str(), toStr(exp_ad.row(4)).c_str(),
      toStr(exp_ad.row(5)).c_str());
  LOGI(
      "Ad_Exp   : %s, %s, %s, %s, %s, %s", toStr(Ad_Exp.row(0)).c_str(),
      toStr(Ad_Exp.row(1)).c_str(), toStr(Ad_Exp.row(2)).c_str(),
      toStr(Ad_Exp.row(3)).c_str(), toStr(Ad_Exp.row(4)).c_str(),
      toStr(Ad_Exp.row(5)).c_str());
  ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

  auto any_Y = MultivariateUniformDistribution<double>::standard(SE3d::kDim)();
  ASSERT_NEAR(
      ((Ad_Exp * any_Y) - SE3d::Ad(SE3d::Exp(X), any_Y)).squaredNorm(), 0,
      1e-6);
  ASSERT_NEAR(
      ((SE3d::ad(X) * any_Y) - SE3d::bracket(X, any_Y)).squaredNorm(), 0, 1e-6);
}

TEST(TestSE3d, LieJacobian) {
  SE3d::LieAlgebra X;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  using LieAlgebraEndomorphism = SE3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LieAlgebraEndomorphism Jl = SE3d::Jl(X);
  LieAlgebraEndomorphism Jl2 = leftLieJacobian<SE3d>(X);
  LOGI(
      "Jl    : %s, %s, %s, %s, %s, %s", toStr(Jl.row(0)).c_str(),
      toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str(),
      toStr(Jl.row(3)).c_str(), toStr(Jl.row(4)).c_str(),
      toStr(Jl.row(5)).c_str());
  LOGI(
      "Jl2   : %s, %s, %s, %s, %s, %s", toStr(Jl2.row(0)).c_str(),
      toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str(),
      toStr(Jl2.row(3)).c_str(), toStr(Jl2.row(4)).c_str(),
      toStr(Jl2.row(5)).c_str());
  ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

  ASSERT_TRUE((SE3d::Jl(X) * SE3d::invJl(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE((SE3d::Jr(X) * SE3d::invJr(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE(SE3d::Jr(X).isApprox(SE3d::Jl(-X), 1e-6));
  ASSERT_TRUE(SE3d::invJr(X).isApprox(SE3d::invJl(-X), 1e-6));
}

TEST(TestSE3d, HatVee) {
  SE3d::LieAlgebra X;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  ASSERT_TRUE(X.isApprox(SE3d::vee(SE3d::hat(X)), 1e-6));
}

TEST(Test_SE3d, ConvertPerturbation) {
  using LieAlgebraEndomorphism = SE3d::LieAlgebraEndomorphism;

  // standard perturbations
  using LeftPerturbation = SE3d::LeftPerturbation;
  using RightPerturbation = SE3d::RightPerturbation;
  using AffineLeftPerturbation = SE3d::AffineLeftPerturbation;
  using AffineRightPerturbation = SE3d::AffineRightPerturbation;

  using LeftOptimizable = SE3d::LeftOptimizable;
  using RightOptimizable = SE3d::RightOptimizable;
  using SeparateLeftOptimizable = SE3d::SeparateLeftOptimizable;
  using SeparateRightOptimizable = SE3d::SeparateRightOptimizable;

  // non-standard perturbations
  using YawXYZFixedPerturbation = SE3d::YawXYZFixedPerturbation<0>;
  using YawXYZOnlyPerturbation = SE3d::YawXYZOnlyPerturbation;
  using YawXYOnlyPerturbation = SE3d::YawXYOnlyPerturbation;

  using YawXYZFixedOptimizable = SE3d::YawXYZFixed<0>;
  using YawXYZOnlyOptimizable = SE3d::YawXYZOnly;
  using YawXYOnlyOptimizable = SE3d::YawXYOnly;

  SO3d rot = SO3d::Exp(Eigen::Vector3d(0, 0.1, 0)) *
             SO3d::Exp(Eigen::Vector3d(0.2, 0, 0));
  SE3d g(rot, Eigen::Vector3d(0.4, 0.5, 0.6));
  LOGI("g.linear():\n%s", toStr(g.linear().matrix()).c_str());
  LOGI("g.translation():\n%s", toStr(g.translation().transpose()).c_str());

  SE3d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(SE3d::kDim)();
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

    SE3d::LieAlgebra delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>() * delta_l;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    SE3d::LieAlgebra delta_l3 =
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
    SE3d::LieAlgebra delta_sl = any_Y * approx_radius;
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

    SE3d::LieAlgebra delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            delta_sl);
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    SE3d::LieAlgebra delta_sl3 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            delta_sr);
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl3:\n%s", toStr(delta_sl3.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl3, 1e-6));

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE((SeparateRightOptimizable(g_sr) - SeparateRightOptimizable(g))
                    .isApprox(delta_sr, 1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     AffineRightPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    AffineRightPerturbation, AffineLeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     AffineLeftPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    AffineLeftPerturbation, AffineRightPerturbation>(any_J),
                1e-6));
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
    SE3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>() *
        delta_l;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    SE3d::LieAlgebra delta_l3 =
        g.convertPerturbation<AffineLeftPerturbation, LeftPerturbation>() *
        delta_sl;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, approx_precision));

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE(
        (LeftOptimizable(g_l) - LeftOptimizable(g)).isApprox(delta_l, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<LeftPerturbation, AffineLeftPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<AffineLeftPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<AffineLeftPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
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

    SE3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<RightPerturbation, AffineLeftPerturbation>() *
        delta_r;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    SE3d::LieAlgebra delta_r3 =
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

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE(
        (RightOptimizable(g_r) - RightOptimizable(g)).isApprox(delta_r, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<RightPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<RightPerturbation, AffineLeftPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<AffineLeftPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<AffineLeftPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
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

    SE3d::LieAlgebra delta_sr2 =
        g.convertPerturbation<RightPerturbation, AffineRightPerturbation>() *
        delta_r;
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, approx_precision));

    SE3d::LieAlgebra delta_r3 =
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

    ASSERT_TRUE((SeparateRightOptimizable(g_sr) - SeparateRightOptimizable(g))
                    .isApprox(delta_sr, 1e-6));
    ASSERT_TRUE(
        (RightOptimizable(g_r) - RightOptimizable(g)).isApprox(delta_r, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<RightPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<RightPerturbation, AffineRightPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<AffineRightPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<AffineRightPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
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

    SE3d::LieAlgebra delta_sr2 =
        g.convertPerturbation<LeftPerturbation, AffineRightPerturbation>() *
        delta_l;
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, approx_precision));

    SE3d::LieAlgebra delta_l3 =
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

    ASSERT_TRUE((SeparateRightOptimizable(g_sr) - SeparateRightOptimizable(g))
                    .isApprox(delta_sr, 1e-6));
    ASSERT_TRUE(
        (LeftOptimizable(g_l) - LeftOptimizable(g)).isApprox(delta_l, 1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<LeftPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<LeftPerturbation, AffineRightPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<AffineRightPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<AffineRightPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
  }

  ////// tests for non-standard perturbation types
  // yaw-xyz fixed perturbation -> separate left perturbation
  {
    LOGI(
        "******** yaw-xyz fixed perturbation <-> separate left perturbation "
        "******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector2d delta_yf = any_Y.head<2>() * approx_radius;
    auto delta_sl =
        g.convertPerturbation<YawXYZFixedPerturbation, AffineLeftPerturbation>(
            delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    auto g_yf = (YawXYZFixedOptimizable(g) + delta_yf).value();
    auto g_sl = (SeparateLeftOptimizable(g) + delta_sl).value();
    LOGI("g_yf.linear():\n%s", toStr(g_yf.linear().matrix()).c_str());
    LOGI(
        "g_yf.translation():\n%s",
        toStr(g_yf.translation().transpose()).c_str());
    LOGI("g_sl.linear():\n%s", toStr(g_sl.linear().matrix()).c_str());
    LOGI(
        "g_sl.translation():\n%s",
        toStr(g_sl.translation().transpose()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_sl, 1e-6));

    SE3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<
            YawXYZFixedPerturbation, AffineLeftPerturbation>() *
        delta_yf;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xyz fixed perturbation --> separate right perturbation --> separate left perturbation  // NOLINT
    //        ^                                                             |                     // NOLINT
    //        |--------------------------------------------------------------                     // NOLINT
    //
    // clang-format on
    auto delta_sl3 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            g.convertPerturbation<
                YawXYZFixedPerturbation, AffineRightPerturbation>(delta_yf));
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl3:\n%s", toStr(delta_sl3.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl3, 1e-6));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xyz fixed perturbation --> separate left perturbation --> separate right perturbation  // NOLINT
    //        ^                                                             |                     // NOLINT
    //        |--------------------------------------------------------------                     // NOLINT
    //
    // clang-format on
    auto delta_sr =
        g.convertPerturbation<YawXYZFixedPerturbation, AffineRightPerturbation>(
            delta_yf);
    auto delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            g.convertPerturbation<
                YawXYZFixedPerturbation, AffineLeftPerturbation>(delta_yf));
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE((YawXYZFixedOptimizable(g_yf) - YawXYZFixedOptimizable(g))
                    .isApprox(delta_yf, 1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     YawXYZFixedPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    YawXYZFixedPerturbation, AffineLeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     YawXYZFixedPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    YawXYZFixedPerturbation, AffineRightPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawXYZFixedPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<YawXYZFixedPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawXYZFixedPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<YawXYZFixedPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
  }

  // yaw-xyz only perturbation -> separate left perturbation
  {
    LOGI(
        "******** yaw-xyz only perturbation <-> separate left perturbation "
        "******** "
        "\n");
    const double approx_radius = 1;  // conversion between yaw-xyz only and
                                     // separate-left perturbations is exact, we
                                     // don't need approximation for this case
    const double approx_precision = 1e-6;  // use a high precision for this case
    Eigen::Vector4d delta_yo = any_Y.head<4>() * approx_radius;
    auto delta_sl =
        g.convertPerturbation<YawXYZOnlyPerturbation, AffineLeftPerturbation>(
            delta_yo);
    LOGI("delta_yo:\n%s", toStr(delta_yo.transpose()).c_str());
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    auto g_yo = (YawXYZOnlyOptimizable(g) + delta_yo).value();
    auto g_sl = (SeparateLeftOptimizable(g) + delta_sl).value();
    LOGI("g_yo.linear():\n%s", toStr(g_yo.linear().matrix()).c_str());
    LOGI(
        "g_yo.translation():\n%s",
        toStr(g_yo.translation().transpose()).c_str());
    LOGI("g_sl.linear():\n%s", toStr(g_sl.linear().matrix()).c_str());
    LOGI(
        "g_sl.translation():\n%s",
        toStr(g_sl.translation().transpose()).c_str());
    ASSERT_TRUE(g_yo.isApprox(g_sl, 1e-6));

    SE3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<
            YawXYZOnlyPerturbation, AffineLeftPerturbation>() *
        delta_yo;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xy only perturbation --> separate right perturbation --> separate left perturbation    // NOLINT
    //        ^                                                             |                     // NOLINT
    //        |--------------------------------------------------------------                     // NOLINT
    //
    // clang-format on
    auto delta_sl3 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            g.convertPerturbation<
                YawXYZOnlyPerturbation, AffineRightPerturbation>(delta_yo));
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl3:\n%s", toStr(delta_sl3.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl3, 1e-6));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xy only perturbation --> separate left perturbation --> separate right perturbation    // NOLINT
    //        ^                                                             |                     // NOLINT
    //        |--------------------------------------------------------------                     // NOLINT
    //
    // clang-format on
    auto delta_sr =
        g.convertPerturbation<YawXYZOnlyPerturbation, AffineRightPerturbation>(
            delta_yo);
    auto delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            g.convertPerturbation<
                YawXYZOnlyPerturbation, AffineLeftPerturbation>(delta_yo));
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE((YawXYZOnlyOptimizable(g_yo) - YawXYZOnlyOptimizable(g))
                    .isApprox(delta_yo, 1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     YawXYZOnlyPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    YawXYZOnlyPerturbation, AffineLeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     YawXYZOnlyPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    YawXYZOnlyPerturbation, AffineRightPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawXYZOnlyPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<YawXYZOnlyPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawXYZOnlyPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<YawXYZOnlyPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
  }

  // yaw-xy only perturbation -> separate left perturbation
  {
    LOGI(
        "******** yaw-xy only perturbation <-> separate left perturbation "
        "******** "
        "\n");
    const double approx_radius = 1;  // conversion between yaw-xy only and
                                     // separate-left perturbations is exact, we
                                     // don't need approximation for this case
    const double approx_precision = 1e-6;  // use a high precision for this case
    Eigen::Vector3d delta_yo = any_Y.head<3>() * approx_radius;
    auto delta_sl =
        g.convertPerturbation<YawXYOnlyPerturbation, AffineLeftPerturbation>(
            delta_yo);
    LOGI("delta_yo:\n%s", toStr(delta_yo.transpose()).c_str());
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    auto g_yo = (YawXYOnlyOptimizable(g) + delta_yo).value();
    auto g_sl = (SeparateLeftOptimizable(g) + delta_sl).value();
    LOGI("g_yo.linear():\n%s", toStr(g_yo.linear().matrix()).c_str());
    LOGI(
        "g_yo.translation():\n%s",
        toStr(g_yo.translation().transpose()).c_str());
    LOGI("g_sl.linear():\n%s", toStr(g_sl.linear().matrix()).c_str());
    LOGI(
        "g_sl.translation():\n%s",
        toStr(g_sl.translation().transpose()).c_str());
    ASSERT_TRUE(g_yo.isApprox(g_sl, 1e-6));

    SE3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<YawXYOnlyPerturbation, AffineLeftPerturbation>() *
        delta_yo;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xy only perturbation --> separate right perturbation --> separate left perturbation    // NOLINT
    //        ^                                                             |                     // NOLINT
    //        |--------------------------------------------------------------                     // NOLINT
    //
    // clang-format on
    auto delta_sl3 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            g.convertPerturbation<
                YawXYOnlyPerturbation, AffineRightPerturbation>(delta_yo));
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl3:\n%s", toStr(delta_sl3.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl3, 1e-6));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xy only perturbation --> separate left perturbation --> separate right perturbation    // NOLINT
    //        ^                                                             |                     // NOLINT
    //        |--------------------------------------------------------------                     // NOLINT
    //
    // clang-format on
    auto delta_sr =
        g.convertPerturbation<YawXYOnlyPerturbation, AffineRightPerturbation>(
            delta_yo);
    auto delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            g.convertPerturbation<
                YawXYOnlyPerturbation, AffineLeftPerturbation>(delta_yo));
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE((YawXYOnlyOptimizable(g_yo) - YawXYOnlyOptimizable(g))
                    .isApprox(delta_yo, 1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     YawXYOnlyPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    YawXYOnlyPerturbation, AffineLeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     YawXYOnlyPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    YawXYOnlyPerturbation, AffineRightPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawXYOnlyPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<YawXYOnlyPerturbation, LeftPerturbation>(
                    any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J *
         (g.convertPerturbation<YawXYOnlyPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<YawXYOnlyPerturbation, RightPerturbation>(
                    any_J),
                1e-6));
  }

  // yaw-xyz fixed perturbation -> left perturbation
  {
    LOGI(
        "******** yaw-xyz fixed perturbation <-> left perturbation "
        "******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector2d delta_yf = any_Y.head<2>() * approx_radius;
    auto delta_l =
        g.convertPerturbation<YawXYZFixedPerturbation, LeftPerturbation>(
            delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_yf = (YawXYZFixedOptimizable(g) + delta_yf).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_yf.linear():\n%s", toStr(g_yf.linear().matrix()).c_str());
    LOGI(
        "g_yf.translation():\n%s",
        toStr(g_yf.translation().transpose()).c_str());
    LOGI("g_l.linear():\n%s", toStr(g_l.linear().matrix()).c_str());
    LOGI(
        "g_l.translation():\n%s", toStr(g_l.translation().transpose()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_l, 1e-6));

    SE3d::LieAlgebra delta_l2 =
        g.convertPerturbation<YawXYZFixedPerturbation, LeftPerturbation>() *
        delta_yf;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xyz fixed perturbation --> right perturbation --> left perturbation  // NOLINT
    //        ^                                                      |          // NOLINT
    //        |-------------------------------------------------------          // NOLINT
    //
    // clang-format on
    auto delta_l3 = g.convertPerturbation<RightPerturbation, LeftPerturbation>(
        g.convertPerturbation<YawXYZFixedPerturbation, RightPerturbation>(
            delta_yf));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));

    // clang-format off
    // test the conversion loop:
    //
    // yaw-xyz fixed perturbation --> left perturbation --> right perturbation  // NOLINT
    //        ^                                                      |          // NOLINT
    //        |-------------------------------------------------------          // NOLINT
    //
    // clang-format on
    auto delta_r =
        g.convertPerturbation<YawXYZFixedPerturbation, RightPerturbation>(
            delta_yf);
    auto delta_r2 = g.convertPerturbation<LeftPerturbation, RightPerturbation>(
        g.convertPerturbation<YawXYZFixedPerturbation, LeftPerturbation>(
            delta_yf));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, 1e-6));
  }
}

#ifdef TEST_CERES_JET
TEST(Test_SE3, Jet) {
  SE3d::LieAlgebra X, any_Y;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  any_Y = MultivariateUniformDistribution<double>::standard(SE3d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using SE3Jet = SE3<Jet>;
  using LieAlgebraEndomorphism = SE3Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  SE3Jet::LieAlgebra jet_X = X.cast<Jet>();
  SE3Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  SE3Jet jet_g = SE3Jet::Exp(jet_X);
  SE3Jet::LieAlgebra jet_log_g = SE3Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((SE3Jet::Jl(jet_X) * SE3Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(SE3Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = SE3Jet::Ad(SE3Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(SE3Jet::Ad(SE3Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((SE3Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(SE3Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
