#include "sk4slam_camera/equidistant.h"

#include <opencv2/calib3d.hpp>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

static const double fx = 527.9990706330082;
static const double fy = 527.963495807245;
static const double cx = 399.18451401412665;
static const double cy = 172.8193108347693;
static const double k1 = -0.03559759964255725;
static const double k2 = -0.005093721310999416;
static const double k3 = 0.019716282737702494;
static const double k4 = -0.01583280039499382;

static const int width = 800;
static const int height = 400;

TEST(TestEquidistant, Basic) {
  Equidistant::Intrinsicsd intrinsics;
  intrinsics << fx, fy, cx, cy, k1, k2, k3, k4;
  Equidistant cam_model(intrinsics);

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
  // fisheye::undistortPoints()
  std::vector<cv::Point2d> random_pixels_cv = {random_pixel_cv};
  std::vector<cv::Point2d> undistorted_points_cv;
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 4) << k1, k2, k3, k4);
  cv::fisheye::undistortPoints(
      random_pixels_cv, undistorted_points_cv, camera_matrix,
      distortion_coefficients);
  cv::Point2d undistorted_point_cv = undistorted_points_cv[0];
  LOGI(
      "undistorted point cv: %f %f", undistorted_point_cv.x,
      undistorted_point_cv.y);
  ASSERT_NEAR(undistorted_point_cv.x, bproj2_res.second.x(), 1e-6);
  ASSERT_NEAR(undistorted_point_cv.y, bproj2_res.second.y(), 1e-6);
}

TEST(TestEquidistant, Jacobian) {
  using CamModel = Equidistant;
  CamModel::Intrinsicsd intrinsics;
  intrinsics << fx, fy, cx, cy, k1, k2, k3, k4;
  CamModel cam_model(intrinsics);

  // Eigen::Vector3d random_point(1.0, 1.2, 2.0);
  // Eigen::Vector3d random_point(1e-6, -1e-6, 1.0);
  Eigen::Vector3d random_point(1e-8, -1e-8, 1.0);
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

TEST(TestEquidistant, LUT) {
  // return;
  using LUT = CameraModelInterface::LUT;
  using CamModel = Equidistant;
  CamModel::Intrinsicsd intrinsics;
  intrinsics << fx, fy, cx, cy, k1, k2, k3, k4;
  CamModel cam_model(intrinsics);

  std::unique_ptr<LUT> lut = cam_model.makeLUT(width, height, true, true);

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

  auto lut_bproj3_res = lut->backProject3(random_pixel);
  LOGI(
      "lut backprojected3 point: %s",
      toStr(lut_bproj3_res.second.transpose()).c_str());
  ASSERT_TRUE(lut_bproj3_res.first);
  ASSERT_NEAR((bproj3_res.second - lut_bproj3_res.second).norm(), 0.0, 1e-6);
  auto lut_reproj3_res = lut->project3(lut_bproj3_res.second);
  LOGI(
      "lut reprojected3 point: %s",
      toStr(lut_reproj3_res.second.transpose()).c_str());
  ASSERT_TRUE(lut_reproj3_res.first);
  ASSERT_NEAR(lut_reproj3_res.second.x(), random_pixel.x(), 1e-4);
  ASSERT_NEAR(lut_reproj3_res.second.y(), random_pixel.y(), 1e-4);

  auto lut_bproj2_res = lut->backProject2(random_pixel);
  LOGI(
      "lut backprojected2 point: %s",
      toStr(lut_bproj2_res.second.transpose()).c_str());
  ASSERT_TRUE(lut_bproj2_res.first);
  ASSERT_NEAR((bproj2_res.second - lut_bproj2_res.second).norm(), 0.0, 1e-6);
  auto lut_reproj2_res = lut->project2(lut_bproj2_res.second);
  LOGI(
      "lut reprojected2 point: %s",
      toStr(lut_reproj2_res.second.transpose()).c_str());
  ASSERT_TRUE(lut_reproj2_res.first);
  ASSERT_NEAR(lut_reproj2_res.second.x(), random_pixel.x(), 1e-3);
  ASSERT_NEAR(lut_reproj2_res.second.y(), random_pixel.y(), 1e-3);
}

SK4SLAM_UNITTEST_ENTRYPOINT
