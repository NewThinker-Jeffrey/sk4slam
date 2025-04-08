#include "sk4slam_liegroups/Sim3.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/SubGLn_rx_Rn.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

#define TEST_CERES_JET
#ifdef TEST_CERES_JET
#include "ceres/jet.h"  // Ensure Sim3 can work with Jet type
#endif

using namespace sk4slam;  // NOLINT

TEST(Test_Sim3, Exp) {
  Sim3d::LieAlgebra X;
  X << 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  Eigen::Matrix<double, 4, 4> hatX = Eigen::Matrix<double, 4, 4>::Zero();
  hatX.block<3, 3>(0, 0) = SO3d::hat(Eigen::Vector3d(X.segment<3>(1)));
  hatX(0, 0) = X(0);
  hatX(1, 1) = X(0);
  hatX(2, 2) = X(0);
  hatX.block<3, 1>(0, 3) = X.tail<3>();
  Eigen::Matrix<double, 4, 4> exp_hatX = expOnAlgebra(hatX);

  Sim3d g = Sim3d::Exp(X);
  Sim3d::LieAlgebra log_g = Sim3d::Log(g);
  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
  double scale = sqrt(g.linear().matrix().squaredNorm() / 3.0);
  LOGI("Sim3d g.scale   : %f", scale);
  LOGI(
      "Sim3d g.rotation   : %s, %s, %s",
      toStr(g.linear().matrix().row(0) / scale).c_str(),
      toStr(g.linear().matrix().row(1) / scale).c_str(),
      toStr(g.linear().matrix().row(2) / scale).c_str());
  LOGI(
      "Sim3d g.scaled_rotation   : %s, %s, %s",
      toStr(g.linear().matrix().row(0)).c_str(),
      toStr(g.linear().matrix().row(1)).c_str(),
      toStr(g.linear().matrix().row(2)).c_str());
  LOGI("Sim3d g.translation  : %s", toStr(g.translation().transpose()).c_str());
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

TEST(Test_Sim3, Ad_ad) {
  Sim3d::LieAlgebra X;
  X << 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  using LieAlgebraEndomorphism = Sim3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(Sim3d::ad(X));
  LieAlgebraEndomorphism Ad_Exp = Sim3d::Ad(Sim3d::Exp(X));
  LOGI(
      "exp_ad    : %s, %s, %s, %s, %s, %s, %s", toStr(exp_ad.row(0)).c_str(),
      toStr(exp_ad.row(1)).c_str(), toStr(exp_ad.row(2)).c_str(),
      toStr(exp_ad.row(3)).c_str(), toStr(exp_ad.row(4)).c_str(),
      toStr(exp_ad.row(5)).c_str(), toStr(exp_ad.row(6)).c_str());
  LOGI(
      "Ad_Exp   : %s, %s, %s, %s, %s, %s, %s", toStr(Ad_Exp.row(0)).c_str(),
      toStr(Ad_Exp.row(1)).c_str(), toStr(Ad_Exp.row(2)).c_str(),
      toStr(Ad_Exp.row(3)).c_str(), toStr(Ad_Exp.row(4)).c_str(),
      toStr(Ad_Exp.row(5)).c_str(), toStr(Ad_Exp.row(6)).c_str());
  ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

  Sim3d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(Sim3d::kDim)();
  ASSERT_NEAR(
      ((Ad_Exp * any_Y) - Sim3d::Ad(Sim3d::Exp(X), any_Y)).squaredNorm(), 0,
      1e-6);
  ASSERT_NEAR(
      ((Sim3d::ad(X) * any_Y) - Sim3d::bracket(X, any_Y)).squaredNorm(), 0,
      1e-6);
}

TEST(Test_Sim3, LieJacobian) {
  Sim3d::LieAlgebra X;
  X << 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  using LieAlgebraEndomorphism = Sim3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LieAlgebraEndomorphism Jl = Sim3d::Jl(X);
  LieAlgebraEndomorphism Jl2 = leftLieJacobian<Sim3d>(X);
  LOGI(
      "Jl    : %s, %s, %s, %s, %s, %s, %s", toStr(Jl.row(0)).c_str(),
      toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str(),
      toStr(Jl.row(3)).c_str(), toStr(Jl.row(4)).c_str(),
      toStr(Jl.row(5)).c_str(), toStr(Jl.row(6)).c_str());
  LOGI(
      "Jl2   : %s, %s, %s, %s, %s, %s, %s", toStr(Jl2.row(0)).c_str(),
      toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str(),
      toStr(Jl2.row(3)).c_str(), toStr(Jl2.row(4)).c_str(),
      toStr(Jl2.row(5)).c_str(), toStr(Jl2.row(6)).c_str());
  ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

  ASSERT_TRUE((Sim3d::Jl(X) * Sim3d::invJl(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE((Sim3d::Jr(X) * Sim3d::invJr(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE(Sim3d::Jr(X).isApprox(Sim3d::Jl(-X), 1e-6));
  ASSERT_TRUE(Sim3d::invJr(X).isApprox(Sim3d::invJl(-X), 1e-6));
}

TEST(Test_Sim3, HatVee) {
  Sim3d::LieAlgebra X;
  X << 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  ASSERT_TRUE(X.isApprox(Sim3d::vee(Sim3d::hat(X)), 1e-6));
}

TEST(Test_Sim3, UpToOrder) {
  using ApproxSim3d = Sim3UpToOrder<double, 10>;
  ApproxSim3d::LieAlgebra X;
  X << 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  using LieAlgebraEndomorphism = ApproxSim3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LieAlgebraEndomorphism Jl = ApproxSim3d::Jl(X);
  LieAlgebraEndomorphism Jl2 = leftLieJacobian<ApproxSim3d>(X);
  LOGI(
      "Jl    : %s, %s, %s, %s, %s, %s, %s", toStr(Jl.row(0)).c_str(),
      toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str(),
      toStr(Jl.row(3)).c_str(), toStr(Jl.row(4)).c_str(),
      toStr(Jl.row(5)).c_str(), toStr(Jl.row(6)).c_str());
  LOGI(
      "Jl2   : %s, %s, %s, %s, %s, %s, %s", toStr(Jl2.row(0)).c_str(),
      toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str(),
      toStr(Jl2.row(3)).c_str(), toStr(Jl2.row(4)).c_str(),
      toStr(Jl2.row(5)).c_str(), toStr(Jl2.row(6)).c_str());
  ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

  ASSERT_TRUE((ApproxSim3d::Jl(X) * ApproxSim3d::invJl(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE((ApproxSim3d::Jr(X) * ApproxSim3d::invJr(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE(ApproxSim3d::Jr(X).isApprox(ApproxSim3d::Jl(-X), 1e-6));
  ASSERT_TRUE(ApproxSim3d::invJr(X).isApprox(ApproxSim3d::invJl(-X), 1e-6));
}

TEST(Test_Sim3, ConvertPerturbation) {
  using LieAlgebraEndomorphism = Sim3d::LieAlgebraEndomorphism;
  using LeftPerturbation = Sim3d::LeftPerturbation;
  using RightPerturbation = Sim3d::RightPerturbation;
  using AffineLeftPerturbation = Sim3d::AffineLeftPerturbation;
  using AffineRightPerturbation = Sim3d::AffineRightPerturbation;
  using LeftOptimizable = Sim3d::LeftOptimizable;
  using RightOptimizable = Sim3d::RightOptimizable;
  using SeparateLeftOptimizable = Sim3d::SeparateLeftOptimizable;
  using SeparateRightOptimizable = Sim3d::SeparateRightOptimizable;

  // non-standard perturbations
  using __YawFixedPerturbation = liegroup_internal::ProductPerturbation<
      Rp_x_SO3d, Rpd::LeftPerturbation, SO3d::YawFixedPerturbation<0>>;
  using __ScaleYawFixedPerturbation = liegroup_internal::SubSpacePerturbation<
      __YawFixedPerturbation, SubSpaceByAxes<1, 2>>;
  using __ZFixedPerturbation = liegroup_internal::SubSpacePerturbation<
      R3d::LeftPerturbation, SubSpaceByAxes<0, 1>>;
  using ScaleYawZFixedPerturbation = Sim3d::AffinePerturbation<
      __ScaleYawFixedPerturbation, __ZFixedPerturbation>;
  using ScaleYawZFixedOptimizable =
      OptimizableManifold<Sim3d, ScaleYawZFixedPerturbation>;

  SO3d rot = SO3d::Exp(Eigen::Vector3d(0, 0.1, 0)) *
             SO3d::Exp(Eigen::Vector3d(0.2, 0, 0));
  Rp_x_SO3d(2.0, rot);

  Sim3d g(Rp_x_SO3d(2.0, rot), Eigen::Vector3d(0.4, 0.5, 0.6));
  LOGI("g.linear():\n%s", toStr(g.linear().matrix()).c_str());
  LOGI("g.translation():\n%s", toStr(g.translation().transpose()).c_str());

  Sim3d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(Sim3d::kDim)();
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

    Sim3d::LieAlgebra delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>() * delta_l;
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, approx_precision));

    Sim3d::LieAlgebra delta_l3 =
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
    Sim3d::LieAlgebra delta_sl = any_Y * approx_radius;
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

    Sim3d::LieAlgebra delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            delta_sl);
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    Sim3d::LieAlgebra delta_sl3 =
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
    Sim3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<LeftPerturbation, AffineLeftPerturbation>() *
        delta_l;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    Sim3d::LieAlgebra delta_l3 =
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

    Sim3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<RightPerturbation, AffineLeftPerturbation>() *
        delta_r;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    Sim3d::LieAlgebra delta_r3 =
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

    Sim3d::LieAlgebra delta_sr2 =
        g.convertPerturbation<RightPerturbation, AffineRightPerturbation>() *
        delta_r;
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, approx_precision));

    Sim3d::LieAlgebra delta_r3 =
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

    Sim3d::LieAlgebra delta_sr2 =
        g.convertPerturbation<LeftPerturbation, AffineRightPerturbation>() *
        delta_l;
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, approx_precision));

    Sim3d::LieAlgebra delta_l3 =
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
  // scale-yaw-z fixed perturbation -> separate left perturbation
  {
    LOGI(
        "******** scale-yaw-z fixed perturbation <-> separate left "
        "perturbation "
        "******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector4d delta_yf = any_Y.head<4>() * approx_radius;
    auto delta_sl = g.convertPerturbation<
        ScaleYawZFixedPerturbation, AffineLeftPerturbation>(delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    auto g_yf = (ScaleYawZFixedOptimizable(g) + delta_yf).value();
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

    Sim3d::LieAlgebra delta_sl2 =
        g.convertPerturbation<
            ScaleYawZFixedPerturbation, AffineLeftPerturbation>() *
        delta_yf;
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl2:\n%s", toStr(delta_sl2.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl2, approx_precision));

    // clang-format off
    // test the conversion loop:
    //
    // scale-yaw-z fixed perturbation --> separate right perturbation --> separate left perturbation  // NOLINT
    //        ^                                                             |                         // NOLINT
    //        |--------------------------------------------------------------                         // NOLINT
    //
    // clang-format on
    auto delta_sl3 =
        g.convertPerturbation<AffineRightPerturbation, AffineLeftPerturbation>(
            g.convertPerturbation<
                ScaleYawZFixedPerturbation, AffineRightPerturbation>(delta_yf));
    LOGI("delta_sl:\n%s", toStr(delta_sl.transpose()).c_str());
    LOGI("delta_sl3:\n%s", toStr(delta_sl3.transpose()).c_str());
    ASSERT_TRUE(delta_sl.isApprox(delta_sl3, 1e-6));

    // clang-format off
    // test the conversion loop:
    //
    // scale-yaw-z fixed perturbation --> separate left perturbation --> separate right perturbation  // NOLINT
    //        ^                                                             |                         // NOLINT
    //        |--------------------------------------------------------------                         // NOLINT
    //
    // clang-format on
    auto delta_sr = g.convertPerturbation<
        ScaleYawZFixedPerturbation, AffineRightPerturbation>(delta_yf);
    auto delta_sr2 =
        g.convertPerturbation<AffineLeftPerturbation, AffineRightPerturbation>(
            g.convertPerturbation<
                ScaleYawZFixedPerturbation, AffineLeftPerturbation>(delta_yf));
    LOGI("delta_sr:\n%s", toStr(delta_sr.transpose()).c_str());
    LOGI("delta_sr2:\n%s", toStr(delta_sr2.transpose()).c_str());
    ASSERT_TRUE(delta_sr.isApprox(delta_sr2, 1e-6));

    ASSERT_TRUE((SeparateLeftOptimizable(g_sl) - SeparateLeftOptimizable(g))
                    .isApprox(delta_sl, 1e-6));
    ASSERT_TRUE((ScaleYawZFixedOptimizable(g_yf) - ScaleYawZFixedOptimizable(g))
                    .isApprox(delta_yf, 1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     ScaleYawZFixedPerturbation, AffineLeftPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    ScaleYawZFixedPerturbation, AffineLeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     ScaleYawZFixedPerturbation, AffineRightPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    ScaleYawZFixedPerturbation, AffineRightPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     ScaleYawZFixedPerturbation, LeftPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    ScaleYawZFixedPerturbation, LeftPerturbation>(any_J),
                1e-6));
    ASSERT_TRUE(
        (any_J * (g.convertPerturbation<
                     ScaleYawZFixedPerturbation, RightPerturbation>()))
            .isApprox(
                g.transformJacobian<
                    ScaleYawZFixedPerturbation, RightPerturbation>(any_J),
                1e-6));
  }

  // scale-yaw-z fixed perturbation -> left perturbation
  {
    LOGI(
        "******** scale-yaw-z fixed perturbation <-> left "
        "perturbation "
        "******** "
        "\n");
    const double approx_radius =
        0.01;  // make it small to ensure the approximation accurate enough
    const double approx_precision = 1e-2;  // precision of the approximation
    Eigen::Vector4d delta_yf = any_Y.head<4>() * approx_radius;
    auto delta_l =
        g.convertPerturbation<ScaleYawZFixedPerturbation, LeftPerturbation>(
            delta_yf);
    LOGI("delta_yf:\n%s", toStr(delta_yf.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_yf = (ScaleYawZFixedOptimizable(g) + delta_yf).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_yf.linear():\n%s", toStr(g_yf.linear().matrix()).c_str());
    LOGI(
        "g_yf.translation():\n%s",
        toStr(g_yf.translation().transpose()).c_str());
    LOGI("g_l.linear():\n%s", toStr(g_l.linear().matrix()).c_str());
    LOGI(
        "g_l.translation():\n%s", toStr(g_l.translation().transpose()).c_str());
    ASSERT_TRUE(g_yf.isApprox(g_l, 1e-6));

    Sim3d::LieAlgebra delta_l2 =
        g.convertPerturbation<ScaleYawZFixedPerturbation, LeftPerturbation>() *
        delta_yf;
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, approx_precision));

    // clang-format off
    // test the conversion loop:
    //
    // scale-yaw-z fixed perturbation --> right perturbation --> left perturbation  // NOLINT
    //        ^                                                          |          // NOLINT
    //        |-----------------------------------------------------------          // NOLINT
    //
    // clang-format on
    auto delta_l3 = g.convertPerturbation<RightPerturbation, LeftPerturbation>(
        g.convertPerturbation<ScaleYawZFixedPerturbation, RightPerturbation>(
            delta_yf));
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_l3:\n%s", toStr(delta_l3.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l3, 1e-6));

    // clang-format off
    // test the conversion loop:
    //
    // scale-yaw-z fixed perturbation --> left perturbation --> right perturbation  // NOLINT
    //        ^                                                          |          // NOLINT
    //        |-----------------------------------------------------------          // NOLINT
    //
    // clang-format on
    auto delta_r =
        g.convertPerturbation<ScaleYawZFixedPerturbation, RightPerturbation>(
            delta_yf);
    auto delta_r2 = g.convertPerturbation<LeftPerturbation, RightPerturbation>(
        g.convertPerturbation<ScaleYawZFixedPerturbation, LeftPerturbation>(
            delta_yf));
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, 1e-6));
  }
}

#ifdef TEST_CERES_JET
TEST(Test_Sim3, Jet) {
  Sim3d::LieAlgebra X, any_Y;
  X << 0.15, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  any_Y = MultivariateUniformDistribution<double>::standard(Sim3d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using Sim3Jet = Sim3<Jet>;
  using LieAlgebraEndomorphism = Sim3Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  Sim3Jet::LieAlgebra jet_X = X.cast<Jet>();
  Sim3Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  Sim3Jet jet_g = Sim3Jet::Exp(jet_X);
  Sim3Jet::LieAlgebra jet_log_g = Sim3Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((Sim3Jet::Jl(jet_X) * Sim3Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(Sim3Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = Sim3Jet::Ad(Sim3Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(Sim3Jet::Ad(Sim3Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((Sim3Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(Sim3Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
