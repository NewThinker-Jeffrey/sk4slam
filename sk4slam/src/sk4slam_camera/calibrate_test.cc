#include "sk4slam_camera/calibrate.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_camera/camera_model_factory.h"
#include "sk4slam_camera/radtan.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

TEST(TestCalibrate, computePinholeIntrinsicsFromHomographyMatrices) {
  static const double fx = 425.780;
  static const double fy = 324.799;
  static const double cx = 436.380;
  static const double cy = 239.206;
  Eigen::Matrix3d K;
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

  Eigen::Matrix3d R1, R2, R3;
  Eigen::Vector3d t1, t2, t3;
  R1 = SO3d::expM(Eigen::Vector3d(0.1, 0.2, 0.1)).matrix();
  R2 = SO3d::expM(Eigen::Vector3d(0.2, 0.1, 0.2)).matrix();
  R3 = SO3d::expM(Eigen::Vector3d(0.1, 0.1, 0.3)).matrix();
  t1 << 0.1, 0.2, 1.3;
  t2 << 0.2, 0.3, 1.4;
  t3 << 0.4, 0.5, 1.5;

  double scale1 = 1.0;
  double scale2 = 1.0;
  double scale3 = 1.0;

  Eigen::Matrix3d Rt1, Rt2, Rt3;
  Rt1 << R1.col(0), R1.col(1), t1;
  Rt2 << R2.col(0), R2.col(1), t2;
  Rt3 << R3.col(0), R3.col(1), t3;

  Eigen::Matrix3d H1, H2, H3;
  H1 = K * scale1 * Rt1;
  H2 = K * scale2 * Rt2;
  H3 = K * scale3 * Rt3;

  std::vector<Eigen::Matrix3d> Hs;
  Hs.push_back(H1);
  Hs.push_back(H2);

  // solve with 2 homography matrices
  auto est_intrinsics2 = computePinholeIntrinsicsFromHomographyMatrices(Hs);
  ASSERT_TRUE(est_intrinsics2);
  auto& intrinsics2 = *est_intrinsics2;
  LOGI("Real intrinsics: fx=%f, fy=%f, cx=%f, cy=%f", fx, fy, cx, cy);
  LOGI(
      "Estimated intrinsics(2): fx=%f, fy=%f, cx=%f, cy=%f", intrinsics2[0],
      intrinsics2[1], intrinsics2[2], intrinsics2[3]);
  ASSERT_NEAR(intrinsics2[0], fx, 1e-3);
  ASSERT_NEAR(intrinsics2[1], fy, 1e-3);
  ASSERT_NEAR(intrinsics2[2], cx, 1e-3);
  ASSERT_NEAR(intrinsics2[3], cy, 1e-3);

  Hs.push_back(H3);
  // solve with 3 homography matrices
  auto est_intrinsics3 = computePinholeIntrinsicsFromHomographyMatrices(Hs);
  ASSERT_TRUE(est_intrinsics3);
  auto& intrinsics3 = *est_intrinsics3;
  LOGI(
      "Estimated intrinsics(3): fx=%f, fy=%f, cx=%f, cy=%f", intrinsics3[0],
      intrinsics3[1], intrinsics3[2], intrinsics3[3]);
  ASSERT_NEAR(intrinsics3[0], fx, 1e-3);
  ASSERT_NEAR(intrinsics3[1], fy, 1e-3);
  ASSERT_NEAR(intrinsics3[2], cx, 1e-3);
  ASSERT_NEAR(intrinsics3[3], cy, 1e-3);
}

SK4SLAM_UNITTEST_ENTRYPOINT
