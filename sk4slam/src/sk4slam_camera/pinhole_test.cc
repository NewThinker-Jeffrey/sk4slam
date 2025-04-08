#include "sk4slam_camera/pinhole.h"

#include <opencv2/calib3d.hpp>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

static const double fx = 425.780;
static const double fy = 424.799;
static const double cx = 436.380;
static const double cy = 239.206;

static const int width = 848;
static const int height = 480;

TEST(TestPinhole, Basic) {
  Pinhole::Intrinsicsd intrinsics;
  intrinsics << fx, fy, cx, cy;
  Pinhole cam_model(intrinsics);

  Eigen::Vector2d random_pixel(201.32, 234.23);
  LOGI("random pixel: %s", toStr(random_pixel.transpose()).c_str());

  auto bproj3_res = cam_model.backProject3(random_pixel);
  LOGI(
      "backprojected3 point: %s", toStr(bproj3_res.second.transpose()).c_str());
  LOGI(
      "backprojected3 point(hnormalized): %s",
      toStr(bproj3_res.second.hnormalized().transpose()).c_str());
  ASSERT_TRUE(bproj3_res.first);
  ASSERT_NEAR(bproj3_res.second.norm(), 1.0, 1e-6);
  auto reproj3_res = cam_model.project3(bproj3_res.second);
  LOGI("reprojected3 point: %s", toStr(reproj3_res.second.transpose()).c_str());
  ASSERT_TRUE(reproj3_res.first);
  ASSERT_NEAR(reproj3_res.second.x(), random_pixel.x(), 1e-6);
  ASSERT_NEAR(reproj3_res.second.y(), random_pixel.y(), 1e-6);

  auto bproj2_res = cam_model.backProject2(random_pixel);
  LOGI(
      "backprojected2 point: %s", toStr(bproj2_res.second.transpose()).c_str());
  ASSERT_TRUE(bproj2_res.first);
  ASSERT_NEAR(
      (bproj3_res.second.hnormalized() - bproj2_res.second).norm(), 0.0, 1e-6);
  auto reproj2_res = cam_model.project2(bproj2_res.second);
  LOGI("reprojected2 point: %s", toStr(reproj2_res.second.transpose()).c_str());
  ASSERT_TRUE(reproj2_res.first);
  ASSERT_NEAR(reproj2_res.second.x(), random_pixel.x(), 1e-6);
  ASSERT_NEAR(reproj2_res.second.y(), random_pixel.y(), 1e-6);

  cv::Point2d random_pixel_cv(random_pixel.x(), random_pixel.y());
  auto bproj3_res_cv = cam_model.backProject3(random_pixel_cv);
  LOGI(
      "backprojected3 point: %f %f %f", bproj3_res_cv.second.x,
      bproj3_res_cv.second.y, bproj3_res_cv.second.z);
  ASSERT_TRUE(bproj3_res_cv.first);
  ASSERT_NEAR(bproj3_res_cv.second.x, bproj3_res.second.x(), 1e-6);
  ASSERT_NEAR(bproj3_res_cv.second.y, bproj3_res.second.y(), 1e-6);
  ASSERT_NEAR(bproj3_res_cv.second.z, bproj3_res.second.z(), 1e-6);
  auto reproj3_res_cv = cam_model.project3(bproj3_res_cv.second);
  LOGI(
      "reprojected3 point: %f %f", reproj3_res_cv.second.x,
      reproj3_res_cv.second.y);
  ASSERT_TRUE(reproj3_res_cv.first);
  ASSERT_NEAR(reproj3_res_cv.second.x, reproj3_res.second.x(), 1e-6);
  ASSERT_NEAR(reproj3_res_cv.second.y, reproj3_res.second.y(), 1e-6);

  auto bproj2_res_cv = cam_model.backProject2(random_pixel_cv);
  LOGI(
      "backprojected2 point: %f %f", bproj2_res_cv.second.x,
      bproj2_res_cv.second.y);
  ASSERT_TRUE(bproj2_res_cv.first);
  ASSERT_NEAR(bproj2_res_cv.second.x, bproj2_res.second.x(), 1e-6);
  ASSERT_NEAR(bproj2_res_cv.second.y, bproj2_res.second.y(), 1e-6);
  auto reproj2_res_cv = cam_model.project2(bproj2_res_cv.second);
  LOGI(
      "reprojected2 point: %f %f", reproj2_res_cv.second.x,
      reproj2_res_cv.second.y);
  ASSERT_TRUE(reproj2_res_cv.first);
  ASSERT_NEAR(reproj2_res_cv.second.x, random_pixel_cv.x, 1e-6);
  ASSERT_NEAR(reproj2_res_cv.second.y, random_pixel_cv.y, 1e-6);

  // Our implementation of backProject2() should be equivalent to OpenCV's
  // undistortPoints()
  std::vector<cv::Point2d> random_pixels_cv = {random_pixel_cv};
  std::vector<cv::Point2d> undistorted_points_cv;
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
  cv::undistortPoints(
      random_pixels_cv, undistorted_points_cv, camera_matrix,
      distortion_coefficients);
  cv::Point2d undistorted_point_cv = undistorted_points_cv[0];
  LOGI(
      "undistorted point cv: %f %f", undistorted_point_cv.x,
      undistorted_point_cv.y);
  ASSERT_NEAR(undistorted_point_cv.x, bproj2_res.second.x(), 1e-6);
  ASSERT_NEAR(undistorted_point_cv.y, bproj2_res.second.y(), 1e-6);
}

TEST(TestPinhole, Jacobian) {
  using CamModel = Pinhole;
  CamModel::Intrinsicsd intrinsics;
  intrinsics << fx, fy, cx, cy;
  CamModel cam_model(intrinsics);

  Eigen::Vector3d random_point(1.0, 1.2, 2.0);
  LOGI("random point: %s", toStr(random_point.transpose()).c_str());
  using JacobianWrtPoint = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
  using JacobianWrtProjectionParams =
      Eigen::Matrix<double, 2, CamModel::kNumProjectionParams, Eigen::RowMajor>;
  using JacobianWrtDistortionParams =
      Eigen::Matrix<double, 2, CamModel::kNumDistortionParams, Eigen::RowMajor>;

  Eigen::Vector2d pixel;
  JacobianWrtPoint jacobian_wrt_point;
  JacobianWrtProjectionParams jacobian_wrt_projection_params;
  JacobianWrtDistortionParams jacobian_wrt_distortion_params;
  CamModel::project3AndComputeJacobians(
      random_point.data(), cam_model.projectionParams(),
      cam_model.distortionParams(), pixel.data(), jacobian_wrt_point.data(),
      jacobian_wrt_projection_params.data(),
      jacobian_wrt_distortion_params.data());

  LOGI("pixel: %s", toStr(pixel.transpose()).c_str());
  LOGI("jacobian wrt point: \n%s", toStr(jacobian_wrt_point).c_str());
  LOGI(
      "jacobian wrt projection params: \n%s",
      toStr(jacobian_wrt_projection_params).c_str());
  LOGI(
      "jacobian wrt distortion params: \n%s",
      toStr(jacobian_wrt_distortion_params).c_str());

  Eigen::Vector2d pixel_auto;
  JacobianWrtPoint jacobian_wrt_point_auto;
  JacobianWrtProjectionParams jacobian_wrt_projection_params_auto;
  JacobianWrtDistortionParams jacobian_wrt_distortion_params_auto;
  project3AndAutoJacobians<CamModel>(
      random_point.data(), cam_model.projectionParams(),
      cam_model.distortionParams(), pixel_auto.data(),
      jacobian_wrt_point_auto.data(),
      jacobian_wrt_projection_params_auto.data(),
      jacobian_wrt_distortion_params_auto.data());

  LOGI("pixel_auto: %s", toStr(pixel_auto.transpose()).c_str());
  LOGI("auto jacobian wrt point: \n%s", toStr(jacobian_wrt_point_auto).c_str());
  LOGI(
      "auto jacobian wrt projection params_auto: \n%s",
      toStr(jacobian_wrt_projection_params_auto).c_str());
  LOGI(
      "auto jacobian wrt distortion params_auto: \n%s",
      toStr(jacobian_wrt_distortion_params_auto).c_str());

  ASSERT_NEAR((pixel_auto - pixel).norm(), 0, 1e-6);
  ASSERT_NEAR((jacobian_wrt_point_auto - jacobian_wrt_point).norm(), 0, 1e-6);
  ASSERT_NEAR(
      (jacobian_wrt_projection_params_auto - jacobian_wrt_projection_params)
          .norm(),
      0, 1e-6);
  ASSERT_NEAR(
      (jacobian_wrt_distortion_params_auto - jacobian_wrt_distortion_params)
          .norm(),
      0, 1e-6);
}

SK4SLAM_UNITTEST_ENTRYPOINT
