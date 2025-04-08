#include "sk4slam_liegroups/SubGLn.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

template <typename ScalarType>
struct SO3SubStructure {
  using Scalar = ScalarType;
  static constexpr int N = 3;
  static constexpr int kDim = 3;

  static Eigen::Matrix<Scalar, N, N> hat(
      const Eigen::Matrix<Scalar, kDim, 1>& X) {
    return SO3<Scalar>::hat(X);
  }

  static Eigen::Matrix<Scalar, kDim, 1> vee(
      const Eigen::Matrix<Scalar, N, N>& hat_X) {
    return SO3<Scalar>::vee(hat_X);
  }

  static Eigen::Matrix<Scalar, N, N> generator(int i) {
    Eigen::Matrix<Scalar, kDim, 1> X = Eigen::Matrix<Scalar, kDim, 1>::Zero();
    X(i) = 1;
    return hat(X);
  }
};

template <typename Scalar>
using SlowSO3 = SubGLnUpToOrder<Scalar, SO3SubStructure, -1, -1, -1>;

using SlowSO3d = SlowSO3<double>;

template <typename Scalar>
using ApproxGL3 = SubGLnUpToOrder<Scalar, GLUnits<3>::Structure, 10, 15, 10>;

using ApproxGL3d = ApproxGL3<double>;

TEST(TestSlowSO3d, Exp) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}};

  for (auto& w : ws) {
    SlowSO3d g = SlowSO3d::Exp(w);
    Eigen::Matrix3d exp_w = SO3d::expM(w);
    Eigen::Vector3d log_g = SlowSO3d::Log(g);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI("log_g : %s", toStr(log_g.transpose()).c_str());
    LOGI(
        "SlowSO3d g: %s, %s, %s", toStr(g.matrix().row(0)).c_str(),
        toStr(g.matrix().row(1)).c_str(), toStr(g.matrix().row(2)).c_str());
    LOGI(
        "SO3exp_w : %s, %s, %s", toStr(exp_w.row(0)).c_str(),
        toStr(exp_w.row(1)).c_str(), toStr(exp_w.row(2)).c_str());

    ASSERT_NEAR((g.matrix() - exp_w).squaredNorm(), 0, 1e-6);
    ASSERT_NEAR((log_g - w).squaredNorm(), 0, 1e-6);
  }
}

TEST(TestSlowSO3d, Ad_ad) {
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}};

  using LieAlgebraEndomorphism = SlowSO3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LOGI("w     : %s", toStr(w.transpose()).c_str());

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(SlowSO3d::ad(w));
    LieAlgebraEndomorphism Ad_Exp = SlowSO3d::Ad(SlowSO3d::Exp(w));
    LOGI(
        "exp_ad    : %s, %s, %s", toStr(exp_ad.row(0)).c_str(),
        toStr(exp_ad.row(1)).c_str(), toStr(exp_ad.row(2)).c_str());
    LOGI(
        "Ad_Exp   : %s, %s, %s", toStr(Ad_Exp.row(0)).c_str(),
        toStr(Ad_Exp.row(1)).c_str(), toStr(Ad_Exp.row(2)).c_str());
    ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

    auto any_Y =
        MultivariateUniformDistribution<double>::standard(SlowSO3d::kDim)();
    ASSERT_NEAR(
        ((Ad_Exp * any_Y) - SlowSO3d::Ad(SlowSO3d::Exp(w), any_Y))
            .squaredNorm(),
        0, 1e-6);
    ASSERT_NEAR(
        ((SlowSO3d::ad(w) * any_Y) - SlowSO3d::bracket(w, any_Y)).squaredNorm(),
        0, 1e-6);
  }
}

TEST(TestSlowSO3d, LieJacobian) {
  Eigen::Vector3d w(0.1, 0.2, 0.3);
  std::vector<Eigen::Vector3d> ws = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}};

  using LieAlgebraEndomorphism = SlowSO3d::LieAlgebraEndomorphism;
  for (auto& w : ws) {
    LieAlgebraEndomorphism Jl = SlowSO3d::Jl(w);
    LieAlgebraEndomorphism Jl2 = leftLieJacobian<SlowSO3d>(w);
    LOGI("w     : %s", toStr(w.transpose()).c_str());
    LOGI(
        "Jl    : %s, %s, %s", toStr(Jl.row(0)).c_str(),
        toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str());
    LOGI(
        "Jl2   : %s, %s, %s", toStr(Jl2.row(0)).c_str(),
        toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str());
    ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

    ASSERT_TRUE((SlowSO3d::Jl(w) * SlowSO3d::invJl(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE((SlowSO3d::Jr(w) * SlowSO3d::invJr(w))
                    .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
    ASSERT_TRUE(SlowSO3d::Jr(w).isApprox(SlowSO3d::Jl(-w), 1e-6));
    ASSERT_TRUE(SlowSO3d::invJr(w).isApprox(SlowSO3d::invJl(-w), 1e-6));
  }
}

TEST(TestSlowSO3d, HatVee) {
  SlowSO3d::LieAlgebra w(0.1, 0.2, 0.3);
  ASSERT_TRUE(w.isApprox(SlowSO3d::vee(SlowSO3d::hat(w)), 1e-6));
}

#ifdef TEST_CERES_JET
TEST(Test_SlowSO3_, Jet) {
  SlowSO3d::LieAlgebra X, any_Y;
  X << 0.1, 0.2, 0.3;
  any_Y = MultivariateUniformDistribution<double>::standard(SlowSO3d::kDim)();

  std::vector<SlowSO3d::LieAlgebra> Xs = {
      {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {1e-10, 2e-10, 3e-10}};

  using Jet = ceres::Jet<double, 7>;
  using SlowSO3Jet = SlowSO3<Jet>;
  using LieAlgebraEndomorphism = SlowSO3Jet::LieAlgebraEndomorphism;
  const Jet Jet_eps(1e-6);
  for (const auto& X : Xs) {
    SlowSO3Jet::LieAlgebra jet_X = X.cast<Jet>();
    SlowSO3Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
    SlowSO3Jet jet_g = SlowSO3Jet::Exp(jet_X);
    SlowSO3Jet::LieAlgebra jet_log_g = SlowSO3Jet::Log(jet_g);
    ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
    ASSERT_TRUE((SlowSO3Jet::Jl(jet_X) * SlowSO3Jet::invJl(jet_X))
                    .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

    LieAlgebraEndomorphism exp_ad = expOnAlgebra(SlowSO3Jet::ad(jet_X));
    LieAlgebraEndomorphism Ad_Exp = SlowSO3Jet::Ad(SlowSO3Jet::Exp(jet_X));
    ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
    ASSERT_TRUE(
        (Ad_Exp * jet_any_Y)
            .isApprox(
                SlowSO3Jet::Ad(SlowSO3Jet::Exp(jet_X), jet_any_Y), Jet_eps));
    ASSERT_TRUE((SlowSO3Jet::ad(jet_X) * jet_any_Y)
                    .isApprox(SlowSO3Jet::bracket(jet_X, jet_any_Y), Jet_eps));
  }
}
#endif

TEST(Test_ApproxGL3, Exp) {
  ApproxGL3d::LieAlgebra X;
  X << 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23;
  ApproxGL3d g = ApproxGL3d::Exp(X);
  ApproxGL3d::LieAlgebra log_g = ApproxGL3d::Log(g);
  LOGI("hat X     :\n%s", toStr(ApproxGL3d::hat(X)).c_str());
  LOGI("hat log_g :\n%s", toStr(ApproxGL3d::hat(log_g)).c_str());
  ASSERT_NEAR((log_g - X).squaredNorm(), 0, 1e-6);
}

TEST(Test_ApproxGL3, Ad_ad) {
  ApproxGL3d::LieAlgebra X;
  X << 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23;
  using LieAlgebraEndomorphism = ApproxGL3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(ApproxGL3d::ad(X));
  LieAlgebraEndomorphism Ad_Exp = ApproxGL3d::Ad(ApproxGL3d::Exp(X));
  LOGI(
      "exp_ad    : %s, %s, %s, %s, %s, %s, %s, %s, %s",
      toStr(exp_ad.row(0)).c_str(), toStr(exp_ad.row(1)).c_str(),
      toStr(exp_ad.row(2)).c_str(), toStr(exp_ad.row(3)).c_str(),
      toStr(exp_ad.row(4)).c_str(), toStr(exp_ad.row(5)).c_str(),
      toStr(exp_ad.row(6)).c_str(), toStr(exp_ad.row(7)).c_str(),
      toStr(exp_ad.row(8)).c_str());
  LOGI(
      "Ad_Exp   : %s, %s, %s, %s, %s, %s, %s, %s, %s",
      toStr(Ad_Exp.row(0)).c_str(), toStr(Ad_Exp.row(1)).c_str(),
      toStr(Ad_Exp.row(2)).c_str(), toStr(Ad_Exp.row(3)).c_str(),
      toStr(Ad_Exp.row(4)).c_str(), toStr(Ad_Exp.row(5)).c_str(),
      toStr(Ad_Exp.row(6)).c_str(), toStr(Ad_Exp.row(7)).c_str(),
      toStr(Ad_Exp.row(8)).c_str());
  ASSERT_NEAR((exp_ad - Ad_Exp).squaredNorm(), 0, 1e-6);

  ApproxGL3d::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(ApproxGL3d::kDim)();
  ASSERT_NEAR(
      ((Ad_Exp * any_Y) - ApproxGL3d::Ad(ApproxGL3d::Exp(X), any_Y))
          .squaredNorm(),
      0, 1e-6);
  ASSERT_NEAR(
      ((ApproxGL3d::ad(X) * any_Y) - ApproxGL3d::bracket(X, any_Y))
          .squaredNorm(),
      0, 1e-6);
}

TEST(Test_ApproxGL3, LieJacobian) {
  ApproxGL3d::LieAlgebra X;
  X << 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23;
  using LieAlgebraEndomorphism = ApproxGL3d::LieAlgebraEndomorphism;

  LOGI("X     : %s", toStr(X.transpose()).c_str());
  LieAlgebraEndomorphism Jl = ApproxGL3d::Jl(X);
  LieAlgebraEndomorphism Jl2 = leftLieJacobian<ApproxGL3d>(X);
  LOGI(
      "Jl    : %s, %s, %s, %s, %s, %s, %s, %s, %s", toStr(Jl.row(0)).c_str(),
      toStr(Jl.row(1)).c_str(), toStr(Jl.row(2)).c_str(),
      toStr(Jl.row(3)).c_str(), toStr(Jl.row(4)).c_str(),
      toStr(Jl.row(5)).c_str(), toStr(Jl.row(6)).c_str(),
      toStr(Jl.row(7)).c_str(), toStr(Jl.row(8)).c_str());
  LOGI(
      "Jl2   : %s, %s, %s, %s, %s, %s, %s, %s, %s", toStr(Jl2.row(0)).c_str(),
      toStr(Jl2.row(1)).c_str(), toStr(Jl2.row(2)).c_str(),
      toStr(Jl2.row(3)).c_str(), toStr(Jl2.row(4)).c_str(),
      toStr(Jl2.row(5)).c_str(), toStr(Jl2.row(6)).c_str(),
      toStr(Jl2.row(7)).c_str(), toStr(Jl2.row(8)).c_str());
  ASSERT_NEAR((Jl - Jl2).squaredNorm(), 0, 1e-6);

  ASSERT_TRUE((ApproxGL3d::Jl(X) * ApproxGL3d::invJl(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE((ApproxGL3d::Jr(X) * ApproxGL3d::invJr(X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE(ApproxGL3d::Jr(X).isApprox(ApproxGL3d::Jl(-X), 1e-6));
  ASSERT_TRUE(ApproxGL3d::invJr(X).isApprox(ApproxGL3d::invJl(-X), 1e-6));
}

TEST(Test_ApproxGL3, HatVee) {
  ApproxGL3d::LieAlgebra X;
  X << 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23;
  ASSERT_TRUE(X.isApprox(ApproxGL3d::vee(ApproxGL3d::hat(X)), 1e-6));
}

#ifdef TEST_CERES_JET
TEST(Test_ApproxGL3, Jet) {
  ApproxGL3d::LieAlgebra X, any_Y;
  X << 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23;
  any_Y = MultivariateUniformDistribution<double>::standard(ApproxGL3d::kDim)();

  using Jet = ceres::Jet<double, 7>;
  using ApproxGL3Jet = ApproxGL3<Jet>;
  using LieAlgebraEndomorphism = ApproxGL3Jet::LieAlgebraEndomorphism;
  Jet Jet_eps(1e-6);
  ApproxGL3Jet::LieAlgebra jet_X = X.cast<Jet>();
  ApproxGL3Jet::LieAlgebra jet_any_Y = any_Y.cast<Jet>();
  ApproxGL3Jet jet_g = ApproxGL3Jet::Exp(jet_X);
  ApproxGL3Jet::LieAlgebra jet_log_g = ApproxGL3Jet::Log(jet_g);
  ASSERT_TRUE(jet_log_g.isApprox(jet_X, Jet_eps));
  ASSERT_TRUE((ApproxGL3Jet::Jl(jet_X) * ApproxGL3Jet::invJl(jet_X))
                  .isApprox(LieAlgebraEndomorphism::Identity(), Jet_eps));

  LieAlgebraEndomorphism exp_ad = expOnAlgebra(ApproxGL3Jet::ad(jet_X));
  LieAlgebraEndomorphism Ad_Exp = ApproxGL3Jet::Ad(ApproxGL3Jet::Exp(jet_X));
  ASSERT_TRUE(exp_ad.isApprox(Ad_Exp, Jet_eps));
  ASSERT_TRUE(
      (Ad_Exp * jet_any_Y)
          .isApprox(
              ApproxGL3Jet::Ad(ApproxGL3Jet::Exp(jet_X), jet_any_Y), Jet_eps));
  ASSERT_TRUE((ApproxGL3Jet::ad(jet_X) * jet_any_Y)
                  .isApprox(ApproxGL3Jet::bracket(jet_X, jet_any_Y), Jet_eps));
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
