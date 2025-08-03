#include "sk4slam_geometry/pnp_test_helper.h"

using namespace sk4slam;  // NOLINT

TEST(PnPTest, P3P) {
  Eigen::Isometry3d real_T_C_Obj;
  int n = 4;
  // const auto point_pairs = getGeneral3D2DPointPairs(&real_T_C_Obj, n);
  const auto point_pairs = getCoplanar3D2DPointPairs(&real_T_C_Obj, n);

  P3PEstimator estimator;

  // std::vector<size_t> selected = {0, 1, 2};
  // auto poses = estimator.compute(selected, point_pairs);
  // int best_i = 0;
  // if (poses.size() > 1) {
  //   double best_err =
  //       computeSquaredErrorsSum(&estimator, point_pairs, poses[best_i]);
  //   for (int i = 1; i < poses.size(); ++i) {
  //     double err = computeSquaredErrorsSum(&estimator, point_pairs,
  //     poses[i]); if (err < best_err) {
  //       best_err = err;
  //       best_i = i;
  //     }
  //   }
  // }

  auto poses = estimator.compute2(point_pairs);
  ASSERT_EQ(poses.size(), 1);
  int best_i = 0;

  checkReprojErr(&estimator, point_pairs, poses[best_i]);
  checkPose(poses[best_i], real_T_C_Obj);
}

// TODO(jeffrey): @2025.08.01  This test gets unstable when we
// run it on ubuntu 22.04 (built with clang 14 + ros2-humble colcon) for unknown
// reason.  However it works fine on ubuntu 20.04 (built with clang 10 +
// ros1-noetic catkin).
TEST(PnPTest, EPNP_4Points) {
  Eigen::Isometry3d real_T_C_Obj;
  int n = 4;

  // EPnP will fail if the points are coplanar
  // const auto point_pairs = getCoplanar3D2DPointPairs(&real_T_C_Obj, n);
  const auto point_pairs = getGeneral3D2DPointPairs(&real_T_C_Obj, n);

  int best_i = 0;

  EPNPEstimator estimator;
  auto poses = estimator.compute2(point_pairs);
  ASSERT_EQ(poses.size(), 1);
  checkReprojErr(&estimator, point_pairs, poses[best_i]);
  checkPose(poses[best_i], real_T_C_Obj);
}

TEST(PnPTest, EPNP_ManyPoints) {
  Eigen::Isometry3d real_T_C_Obj;
  int n = 8;

  // EPnP will fail if the points are coplanar
  // const auto point_pairs = getCoplanar3D2DPointPairs(&real_T_C_Obj, n);
  const auto point_pairs = getGeneral3D2DPointPairs(&real_T_C_Obj, n);

  int best_i = 0;

  EPNPEstimator estimator;
  auto poses = estimator.compute2(point_pairs);
  ASSERT_EQ(poses.size(), 1);
  checkReprojErr(&estimator, point_pairs, poses[best_i]);
  checkPose(poses[best_i], real_T_C_Obj);
}

TEST(PnPTest, CoplanarP4P) {
  Eigen::Isometry3d real_T_C_Obj;
  int n = 4;
  const auto point_pairs = getCoplanar3D2DPointPairs(&real_T_C_Obj, n);
  int best_i = 0;

  CoplanarP4PEstimator estimator;
  auto poses = estimator.compute2(point_pairs);
  ASSERT_EQ(poses.size(), 1);
  checkReprojErr(&estimator, point_pairs, poses[best_i]);
  checkPose(poses[best_i], real_T_C_Obj);
}

TEST(PnPTest, CoplanarP4P_Refine) {
  Eigen::Isometry3d real_T_C_Obj;
  int n = 4;
  const double noise_sigma = 0.001;
  const auto point_pairs =
      getCoplanar3D2DPointPairs(&real_T_C_Obj, n, noise_sigma);
  int best_i = 0;

  CoplanarP4PEstimator estimator;
  auto poses = estimator.compute2(point_pairs);
  ASSERT_EQ(poses.size(), 1);
  double err_before_opt =
      computeSquaredErrorsSum(&estimator, point_pairs, poses[best_i]);
  bool print_iterations = true;
  Eigen::Matrix<double, 6, 6> cov;
  {
    std::vector<Eigen::Matrix2d> observation_cov(
        point_pairs.size(),
        noise_sigma * noise_sigma * Eigen::Matrix2d::Identity());
    ASSERT_TRUE(estimator.refinePnP2(
        point_pairs, &poses[best_i], false, 5, &cov, &observation_cov, nullptr,
        nullptr, print_iterations));
  }
  {
    ASSERT_TRUE(estimator.refinePnP2(
        point_pairs, &poses[best_i], false, 5, &cov, nullptr, nullptr, nullptr,
        print_iterations));
  }
  double err_after_opt =
      computeSquaredErrorsSum(&estimator, point_pairs, poses[best_i]);
  LOGI(
      "err_before_opt: %f, err_after_opt: %f", sqrt(err_before_opt),
      sqrt(err_after_opt));
  ASSERT_LT(sqrt(err_after_opt), sqrt(err_before_opt));

  checkReprojErr(&estimator, point_pairs, poses[best_i], noise_sigma);
  checkPose(poses[best_i], real_T_C_Obj);
}

SK4SLAM_UNITTEST_ENTRYPOINT
