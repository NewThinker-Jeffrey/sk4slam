#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SE3.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/Sim3.h"
#include "sk4slam_liegroups/liegroup_test_helper.h"

using namespace sk4slam;  // NOLINT

TEST(TestIsLieGroup, Main) {
  ASSERT_TRUE((std::is_base_of_v<LieGroupBase<SO3d>, SO3d>));
  ASSERT_FALSE(IsLieGroupExtension<SO3d>);

  ASSERT_FALSE((std::is_base_of_v<LieGroupBase<SE3d>, SE3d>));
  ASSERT_TRUE(IsLieGroupExtension<SE3d>);

  ASSERT_TRUE(IsLieGroup<SO3d>);
  ASSERT_TRUE(IsLieGroup<SE3d>);
}

TEST(TestLieGroup, Cast) {
  {
    SO3d gd;
    auto gf = gd.cast<float>();
    LOGI("SO3f: %s", toOneLineStr(gf.matrix()).c_str());
  }

  {
    ProductLieGroup<SO3d, SO3d, SO3d> dddd;
    auto ffff = dddd.cast<float>();
    LOGI(
        "ProductLieGroup<float>::part<0>: %s",
        toOneLineStr(ffff.part<0>().matrix()).c_str());
  }

  {
    SE3d gd;
    auto gf = gd.cast<float>();
    LOGI("SE3f: %s", toOneLineStr(gf.matrix()).c_str());
  }
}

TEST(TestProductLieGroup, Main) {
  ProductLieGroup<SO3d, SO3d, SO3d> dddd;
  ASSERT_EQ(dddd.kDim, 9);
  ASSERT_EQ(dddd.kAmbientDim, 27);
  ASSERT_EQ(dddd.kParts, 3);
  for (size_t i = 0; i < dddd.kParts; i++) {
    ASSERT_EQ(dddd.kPartDims[i], 3);
    ASSERT_EQ(dddd.kPartAmbientDims[i], 9);
  }
  ASSERT_EQ(dddd.part<0>().kDim, 3);

  decltype(dddd.cast<float>()) ffff;
  ASSERT_EQ(ffff.kDim, 9);
  ASSERT_EQ(ffff.kAmbientDim, 27);
  ASSERT_EQ(ffff.kParts, 3);
  for (size_t i = 0; i < ffff.kParts; i++) {
    ASSERT_EQ(ffff.kPartDims[i], 3);
    ASSERT_EQ(ffff.kPartAmbientDims[i], 9);
  }
  ASSERT_EQ(ffff.part<0>().kDim, 3);

  // cast from double to float and vice versa
  ASSERT_TRUE(ffff.cast<double>().cast<float>().isApprox(ffff, 1e-10));
  ASSERT_TRUE(dddd.cast<float>().cast<double>().isApprox(dddd, 1e-10));

  Eigen::Vector3d w1(0.1, 0.2, 0.3);
  Eigen::Vector3d w2(0.4, 0.5, 0.6);
  Eigen::Matrix<double, 6, 1> w;
  w << w1, w2;

  SO3d r1 = SO3d::Exp(w1);
  SO3d r2 = SO3d::Exp(w2);

  // using DirectProduct = ProductLieGroup2<SO3d, SO3d>;
  using DirectProduct = ProductLieGroup<SO3d, SO3d>;
  DirectProduct dp1(r1, r2);
  DirectProduct dp2 = DirectProduct::Exp(w);

  ASSERT_EQ(DirectProduct::kDim, 6);
  ASSERT_EQ(DirectProduct::kAmbientDim, 18);

  Oss oss_r1, oss_r2;
  auto r1_iter = r1.data();
  for (size_t i = 0; i < r1.kAmbientDim; i += 3) {
    ASSERT_EQ(&(r1.data()[i]), &(*r1_iter));
    oss_r1 << *(r1_iter++) << " " << *(r1_iter++) << " " << *(r1_iter++)
           << "\n";
  }

  auto r2_iter = r2.data();
  for (size_t i = 0; i < r2.kAmbientDim; i += 3) {
    ASSERT_EQ(&(r2.data()[i]), &(*r2_iter));
    oss_r2 << *(r2_iter++) << " " << *(r2_iter++) << " " << *(r2_iter++)
           << "\n";
  }

  Oss oss_dp1, oss_dp2;
  auto dp1_iter = dp1.data();
  for (size_t i = 0; i < dp1.kAmbientDim; i += 3) {
    ASSERT_EQ(&(dp1.data()[i]), &(*dp1_iter));
    oss_dp1 << *(dp1_iter++) << " " << *(dp1_iter++) << " " << *(dp1_iter++)
            << "\n";
  }

  auto dp2_iter = dp2.data();
  for (size_t i = 0; i < dp2.kAmbientDim; i += 3) {
    ASSERT_EQ(&(dp2.data()[i]), &(*dp2_iter));
    oss_dp2 << *(dp2_iter++) << " " << *(dp2_iter++) << " " << *(dp2_iter++)
            << "\n";
  }

  LOGI("r1:\n%s", oss_r1.str().c_str());
  LOGI("r2:\n%s", oss_r2.str().c_str());
  LOGI("dp1:\n%s", oss_dp1.str().c_str());
  LOGI("dp2:\n%s", oss_dp2.str().c_str());

  ASSERT_TRUE(dp1.isApprox(dp2));

  ASSERT_TRUE(DirectProduct::Log(dp2).isApprox(w));

  ASSERT_TRUE(DirectProduct::Jl(w).part<0>().isApprox(SO3d::Jl(w1)));
  ASSERT_TRUE(DirectProduct::Jl(w).part<1>().isApprox(SO3d::Jl(w2)));

  // test conversion from endomorphism to matrix
  // auto Jl_mat = static_cast<Eigen::Matrix<double, DirectProduct::kDim,
  // DirectProduct::kDim>>(DirectProduct::Jl(w));
  Eigen::Matrix<double, DirectProduct::kDim, DirectProduct::kDim> Jl_mat =
      DirectProduct::Jl(w);
  LOGI("Jl part 0:\n%s", toStr(DirectProduct::Jl(w).part<0>()).c_str());
  LOGI("Jl part 1:\n%s", toStr(DirectProduct::Jl(w).part<1>()).c_str());
  LOGI("Jl_mat:\n%s", toStr(Jl_mat).c_str());
  ASSERT_TRUE(
      (Jl_mat.block<3, 3>(0, 0)).isApprox(DirectProduct::Jl(w).part<0>()));
  ASSERT_TRUE(
      (Jl_mat.block<3, 3>(3, 3)).isApprox(DirectProduct::Jl(w).part<1>()));
  ASSERT_TRUE((Jl_mat.block<3, 3>(0, 3)).isZero());
  ASSERT_TRUE((Jl_mat.block<3, 3>(3, 0)).isZero());

  DirectProduct::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(DirectProduct::kDim)();
  ASSERT_NEAR(
      ((DirectProduct::Ad(dp2) * any_Y) - DirectProduct::Ad(dp2, any_Y))
          .squaredNorm(),
      0, 1e-6);
  ASSERT_NEAR(
      ((DirectProduct::ad(w) * any_Y) - DirectProduct::bracket(w, any_Y))
          .squaredNorm(),
      0, 1e-6);

  ASSERT_TRUE(DirectProduct::Ad(dp2).part<0>().isApprox(SO3d::Ad(r1)));
  ASSERT_TRUE(DirectProduct::Ad(dp2).part<1>().isApprox(SO3d::Ad(r2)));
  auto Ad_dp2 = DirectProduct::Ad(dp2);
  auto Ad_dp2f = Ad_dp2.cast<float>();
  auto Ad_dp2fd = Ad_dp2f.cast<double>();
  LOGI(
      "Ad_dp2:\n%s\n%s", toStr(Ad_dp2.part<0>()).c_str(),
      toStr(Ad_dp2.part<1>()).c_str());
  LOGI(
      "Ad_dp2f:\n%s\n%s", toStr(Ad_dp2f.part<0>()).c_str(),
      toStr(Ad_dp2f.part<1>()).c_str());
  LOGI(
      "Ad_dp2fd:\n%s\n%s", toStr(Ad_dp2fd.part<0>()).c_str(),
      toStr(Ad_dp2fd.part<1>()).c_str());

  ASSERT_TRUE((std::is_same_v<decltype(Ad_dp2), decltype(Ad_dp2fd)>));
  ASSERT_TRUE(Ad_dp2.cast<float>().cast<double>().isApprox(Ad_dp2, 1e-6));
  ASSERT_TRUE(Ad_dp2f.cast<double>().cast<float>().isApprox(Ad_dp2f, 1e-6));

  ASSERT_TRUE(DirectProduct::ad(w).part<0>().isApprox(SO3d::ad(w1)));
  ASSERT_TRUE(DirectProduct::ad(w).part<1>().isApprox(SO3d::ad(w2)));

  ASSERT_TRUE(leftLieJacobian<DirectProduct>(w).part<0>().isApprox(
      leftLieJacobian<SO3d>(w1)));
  ASSERT_TRUE(leftLieJacobian<DirectProduct>(w).part<1>().isApprox(
      leftLieJacobian<SO3d>(w2)));

  // test leftLieJacobianOfXOnY()
  ASSERT_TRUE(leftLieJacobianOfXOnY<DirectProduct>(w, any_Y).isApprox(
      leftLieJacobian<DirectProduct>(w) * any_Y, 1e-6));

  using LieAlgebraEndomorphism = DirectProduct::LieAlgebraEndomorphism;
  ASSERT_TRUE((DirectProduct::Jl(w) * DirectProduct::invJl(w))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE((DirectProduct::Jr(w) * DirectProduct::invJr(w))
                  .isApprox(LieAlgebraEndomorphism::Identity(), 1e-6));
  ASSERT_TRUE(DirectProduct::Jr(w).isApprox(DirectProduct::Jl(-w), 1e-6));
  ASSERT_TRUE(DirectProduct::invJr(w).isApprox(DirectProduct::invJl(-w), 1e-6));

  // test invLeftLieJacobian()
  ASSERT_TRUE(
      (leftLieJacobian<DirectProduct>(w) * invLeftLieJacobian<DirectProduct>(w))
          .isApprox(DirectProduct::LieAlgebraEndomorphism::Identity(), 1e-6));

  // Test BCH approximation of BCHInvLeftLieJacobianOfXOnY().
  // We need w to be small.
  w /= (w.norm() / 0.9);  // make sure w1 is small (norm=0.9)
  auto res_with_BCH = BCHInvLeftLieJacobianOfXOnY<DirectProduct>(w, any_Y);
  auto res_with_BCH2 = BCHInvLeftLieJacobian<DirectProduct>(w) * any_Y;
  auto res_without_BCH = invLeftLieJacobian<DirectProduct>(w) * any_Y;
  LOGI("Y               : %s", toStr(any_Y.transpose()).c_str());
  LOGI("res_without_BCH : %s", toStr(res_without_BCH.transpose()).c_str());
  LOGI("res_with_BCH2   : %s", toStr(res_with_BCH2.transpose()).c_str());
  LOGI("res_with_BCH    : %s", toStr(res_with_BCH.transpose()).c_str());
  // ASSERT_TRUE(res_with_BCH.isApprox(res_with_BCH2, 1e-6));
  auto err_BCH = res_with_BCH - res_without_BCH;
  LOGI("err_BCH         : %s", toStr(err_BCH.transpose()).c_str());
  LOGI("err_BCH_norm    : %f", err_BCH.norm());
  auto err_BCH2 = res_with_BCH2 - res_without_BCH;
  LOGI("err_BCH2        : %s", toStr(err_BCH2.transpose()).c_str());
  LOGI("err_BCH2_norm   : %f", err_BCH2.norm());
  ASSERT_NEAR(err_BCH.norm(), 0, 1e-6);
  ASSERT_NEAR(err_BCH2.norm(), 0, 1e-6);

  // Test BCH approximation of invLeftLieJacobian().
  ASSERT_TRUE(invLeftLieJacobian<DirectProduct>(w).isApprox(
      BCHInvLeftLieJacobian<DirectProduct>(w), 1e-6));
}

TEST(TestProductLieGroup, HatVee) {
  // using DirectProduct = ProductLieGroup<SO3d, Rpd>;
  using DirectProduct = ProductLieGroup<SO3d, SO3d>;
  DirectProduct::LieAlgebra X;
  // X << 0.1, 0.2, 0.3, 0.4;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  auto X_hat = DirectProduct::hat(X);
  LOGI("X:\n%s", toStr(X.transpose()).c_str());
  LOGI(
      "hat(X):\n%s\n\n%s", toStr(X_hat.part<0>()).c_str(),
      toStr(X_hat.part<1>()).c_str());
  LOGI(
      "vee(hat(X)):\n%s", toStr(DirectProduct::vee(X_hat).transpose()).c_str());
  ASSERT_TRUE(X.isApprox(DirectProduct::vee(X_hat), 1e-6));
}

TEST(TestLieGroup, GroupOpJacobian) {
  LIE_GROUP_TEST_GroupOpJacobian(SO3d, LeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(SO3d, RightPerturbation);

  using SO3_x_SO3 = ProductLieGroup<SO3d, SO3d>;
  LIE_GROUP_TEST_GroupOpJacobian(SO3_x_SO3, LeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(SO3_x_SO3, RightPerturbation);

  LIE_GROUP_TEST_GroupOpJacobian(Rp_x_SO3d, LeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(Rp_x_SO3d, RightPerturbation);

  LIE_GROUP_TEST_GroupOpJacobian(SE3d, LeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(SE3d, RightPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(SE3d, AffineLeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(SE3d, AffineRightPerturbation);

  LIE_GROUP_TEST_GroupOpJacobian(Sim3d, LeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(Sim3d, RightPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(Sim3d, AffineLeftPerturbation);
  LIE_GROUP_TEST_GroupOpJacobian(Sim3d, AffineRightPerturbation);
}

TEST(TestLieGroup, TransformVectorJacobian) {
  LIE_GROUP_TEST_TransformVectorJacobian(Rp_x_SO3d, LeftPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(Rp_x_SO3d, RightPerturbation);

  LIE_GROUP_TEST_TransformVectorJacobian(SE3d, LeftPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(SE3d, RightPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(SE3d, AffineLeftPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(SE3d, AffineRightPerturbation);

  LIE_GROUP_TEST_TransformVectorJacobian(Sim3d, LeftPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(Sim3d, RightPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(Sim3d, AffineLeftPerturbation);
  LIE_GROUP_TEST_TransformVectorJacobian(Sim3d, AffineRightPerturbation);
}

TEST(TestLieGroup, TransformHomoVectorJacobian) {
  LIE_GROUP_TEST_TransformHomoVectorJacobian(SE3d, LeftPerturbation);
  LIE_GROUP_TEST_TransformHomoVectorJacobian(SE3d, RightPerturbation);
  LIE_GROUP_TEST_TransformHomoVectorJacobian(SE3d, AffineLeftPerturbation);
  LIE_GROUP_TEST_TransformHomoVectorJacobian(SE3d, AffineRightPerturbation);

  LIE_GROUP_TEST_TransformHomoVectorJacobian(Sim3d, LeftPerturbation);
  LIE_GROUP_TEST_TransformHomoVectorJacobian(Sim3d, RightPerturbation);
  LIE_GROUP_TEST_TransformHomoVectorJacobian(Sim3d, AffineLeftPerturbation);
  LIE_GROUP_TEST_TransformHomoVectorJacobian(Sim3d, AffineRightPerturbation);
}

TEST(TestProductLieGroup, ConvertPerturbation) {
  // using DirectProduct = ProductLieGroup<SO3d, Rpd>;
  using DirectProduct = ProductLieGroup<SO3d, SO3d>;
  using LeftPerturbation = DirectProduct::LeftPerturbation;
  using RightPerturbation = DirectProduct::RightPerturbation;
  using LeftOptimizable = DirectProduct::LeftOptimizable;
  using RightOptimizable = DirectProduct::RightOptimizable;
  DirectProduct::LieAlgebra X;
  // X << 0.1, 0.2, 0.3, 0.4;
  X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  DirectProduct g = DirectProduct::Exp(X);
  LOGI("X:\n%s", toStr(X.transpose()).c_str());
  LOGI("g.part<0>():\n%s", toStr(g.part<0>().matrix()).c_str());
  LOGI("g.part<1>():\n%s", toStr(g.part<1>().matrix()).c_str());

  DirectProduct::LieAlgebra any_Y =
      MultivariateUniformDistribution<double>::standard(DirectProduct::kDim)();
  LOGI("any_Y:\n%s", toStr(any_Y.transpose()).c_str());

  // left perturbation -> right perturbation
  {
    auto delta_l = any_Y;
    auto delta_r =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(delta_l);
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    auto g_r = (RightOptimizable(g) + delta_r).value();
    LOGI("g_l.part<0>():\n%s", toStr(g_l.part<0>().matrix()).c_str());
    LOGI("g_l.part<1>():\n%s", toStr(g_l.part<1>().matrix()).c_str());
    LOGI("g_r.part<0>():\n%s", toStr(g_r.part<0>().matrix()).c_str());
    LOGI("g_r.part<1>():\n%s", toStr(g_r.part<1>().matrix()).c_str());
    ASSERT_TRUE(g_l.isApprox(g_r, 1e-6));

    auto delta_l2 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(delta_r);
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, 1e-6));

    DirectProduct::LieAlgebra delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>() * delta_l;
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, 1e-6));
  }

  // right perturbation -> left perturbation
  {
    auto delta_r = any_Y;
    auto delta_l =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>(delta_r);
    LOGI("delta_r:\n%s", toStr(delta_r.transpose()).c_str());
    LOGI("delta_l:\n%s", toStr(delta_l.transpose()).c_str());
    auto g_r = (RightOptimizable(g) + delta_r).value();
    auto g_l = (LeftOptimizable(g) + delta_l).value();
    LOGI("g_l.part<0>():\n%s", toStr(g_l.part<0>().matrix()).c_str());
    LOGI("g_l.part<1>():\n%s", toStr(g_l.part<1>().matrix()).c_str());
    LOGI("g_r.part<0>():\n%s", toStr(g_r.part<0>().matrix()).c_str());
    LOGI("g_r.part<1>():\n%s", toStr(g_r.part<1>().matrix()).c_str());
    ASSERT_TRUE(g_l.isApprox(g_r, 1e-6));

    auto delta_r2 =
        g.convertPerturbation<LeftPerturbation, RightPerturbation>(delta_l);
    LOGI("delta_r2:\n%s", toStr(delta_r2.transpose()).c_str());
    ASSERT_TRUE(delta_r.isApprox(delta_r2, 1e-6));

    DirectProduct::LieAlgebra delta_l2 =
        g.convertPerturbation<RightPerturbation, LeftPerturbation>() * delta_r;
    LOGI("delta_l2:\n%s", toStr(delta_l2.transpose()).c_str());
    ASSERT_TRUE(delta_l.isApprox(delta_l2, 1e-6));
  }
}

#define TEST_BCH
#ifdef TEST_BCH

TEST(TestLieGroup, BCH) {
  SO3d::LieAlgebra X(0.42, 0.12, 0.34);
  SO3d::LieAlgebra Y(0.12, 0.44, 0.29);
  // X *= 2;
  // Y *= 2;
  LOGI("X = %s", toStr(X.transpose(), Precision(12)).c_str());
  LOGI("Y = %s", toStr(Y.transpose(), Precision(12)).c_str());

  SO3d::LieAlgebra Ztruth = SO3d::Log(SO3d::Exp(X) * SO3d::Exp(Y));
  for (int xy_order = 2; xy_order < 11 + 1; xy_order++) {
    SO3d::LieAlgebra Z = BCH<SO3d>(X, Y, xy_order);
    SO3d::LieAlgebra Zwiki = BCHwiki<SO3d>(X, Y, xy_order);
    LOGI("approx order %d:", xy_order);

    LOGI(
        "                 Zwiki = %s",
        toStr(Zwiki.transpose(), Precision(12)).c_str());
    LOGI(
        "                   Z   = %s",
        toStr(Z.transpose(), Precision(12)).c_str());
    LOGI(
        "[ ground truth ]  log  = %s",
        toStr(Ztruth.transpose(), Precision(12)).c_str());
    LOGI("    Z error = %.12f", (Z - Ztruth).norm());
    LOGI("Zwiki error = %.12f", (Zwiki - Ztruth).norm());
  }
}
#endif  // TEST_BCH

SK4SLAM_UNITTEST_ENTRYPOINT
