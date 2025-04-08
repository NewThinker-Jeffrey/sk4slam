#include "sk4slam_camera/camera_model.h"

#include <opencv2/calib3d.hpp>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_camera/camera_model_factory.h"
#include "sk4slam_camera/radtan.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

TEST(TestCameraModel, Factory) {
  {
    CameraModelType camera_model_type = CameraModelType::RADTAN5;
    Eigen::VectorXd intrinsics(9);

    static const double fx = 425.780;
    static const double fy = 424.799;
    static const double cx = 436.380;
    static const double cy = 239.206;
    static const double k1 = -0.056;
    static const double k2 = 0.068;
    static const double p1 = 0.005;
    static const double p2 = 0.005;
    static const double k3 = -0.022;

    static const int width = 848;
    static const int height = 480;

    intrinsics << fx, fy, cx, cy, k1, k2, p1, p2, k3;
    auto cam_model = createCameraModel(camera_model_type, intrinsics);

    ASSERT_NE(cam_model, nullptr);
  }
  {
    CameraModelType camera_model_type = CameraModelType::EQUIDISTANT;
    Eigen::VectorXd intrinsics(8);

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

    intrinsics << fx, fy, cx, cy, k1, k2, k3, k4;
    auto cam_model = createCameraModel(camera_model_type, intrinsics);

    Eigen::Vector3d random_point(1.0, 1.2, 2.0);
    Eigen::Vector2d pixel;
    ASSERT_TRUE(project3WithCameraModel(
        CameraModelType::EQUIDISTANT, random_point,
        cam_model->projectionParams(), cam_model->distortionParams(),
        pixel.data()));
    ASSERT_TRUE(
        project3WithCameraModel(cam_model.get(), random_point, pixel.data()));
    ASSERT_NE(cam_model, nullptr);
  }
}

TEST(TestCameraModel, LUT) {
  static const double fx = 425.780;
  static const double fy = 424.799;
  static const double cx = 436.380;
  static const double cy = 239.206;
  static const double k1 = -0.056;
  static const double k2 = 0.068;
  static const double p1 = 0.005;
  static const double p2 = 0.005;
  static const double k3 = -0.022;

  static const int width = 848;
  static const int height = 480;

  using LUT = CameraModelInterface::LUT;
  using CamModel = RadTan<5>;
  CamModel::Intrinsicsd intrinsics;
  intrinsics << fx, fy, cx, cy, k1, k2, p1, p2, k3;
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
  ASSERT_NEAR(lut_reproj2_res.second.x(), random_pixel.x(), 1e-4);
  ASSERT_NEAR(lut_reproj2_res.second.y(), random_pixel.y(), 1e-4);

  using JacobianMatrix2x3 = CameraModelInterface::JacobianMatrix<double, 2, 3>;
  using JacobianMatrix2x2 = CameraModelInterface::JacobianMatrix<double, 2, 2>;

  {
    Eigen::Vector2d pixel(random_pixel.x(), random_pixel.y());
    Eigen::Vector2d point2_1, point2_2;
    JacobianMatrix2x2 jacobian1, jacobian2;
    ASSERT_TRUE(cam_model.backProject2AndComputeJacobian(
        pixel.data(), point2_1.data(), jacobian1.data()));
    ASSERT_TRUE(lut->backProject2AndComputeJacobian(
        pixel.data(), point2_2.data(), jacobian2.data()));
    ASSERT_NEAR(point2_1.x(), point2_2.x(), 1e-4);
    ASSERT_NEAR(point2_1.y(), point2_2.y(), 1e-4);
    LOGI("jacobian1: \n%s", toStr(jacobian1).c_str());
    LOGI("jacobian2: \n%s", toStr(jacobian2).c_str());
    ASSERT_TRUE(jacobian1.isApprox(jacobian2, 1e-4));
  }

  {
    Eigen::Vector3d point3 = bproj3_res.second;
    Eigen::Vector2d pixel1, pixel2;
    JacobianMatrix2x3 jacobian1, jacobian2;
    ASSERT_TRUE(cam_model.project3AndComputeJacobian(
        point3.data(), pixel1.data(), jacobian1.data()));
    ASSERT_TRUE(lut->project3AndComputeJacobian(
        point3.data(), pixel2.data(), jacobian2.data()));
    ASSERT_NEAR(pixel1.x(), pixel2.x(), 1e-3);
    ASSERT_NEAR(pixel1.y(), pixel2.y(), 1e-3);
    LOGI("jacobian1: \n%s", toStr(jacobian1).c_str());
    LOGI("jacobian2: \n%s", toStr(jacobian2).c_str());
    ASSERT_TRUE(jacobian1.isApprox(jacobian2, 1e-4));
  }

  {
    Eigen::Vector3d point3 = bproj3_res.second;
    Eigen::Vector2d pixel1, pixel2;
    JacobianMatrix2x3 jacobian1, jacobian2;
    ASSERT_TRUE(cam_model.project3AndComputeJacobian(
        point3.data(), pixel1.data(), jacobian1.data()));
    ASSERT_TRUE(LutProject3AndAutoJacobian(
        lut.get(), point3.data(), pixel2.data(), jacobian2.data()));
    ASSERT_NEAR(pixel1.x(), pixel2.x(), 1e-3);
    ASSERT_NEAR(pixel1.y(), pixel2.y(), 1e-3);
    LOGI("jacobian1: \n%s", toStr(jacobian1).c_str());
    LOGI("jacobian2: \n%s", toStr(jacobian2).c_str());
    ASSERT_TRUE(jacobian1.isApprox(jacobian2, 1e-3));
  }
}

SK4SLAM_UNITTEST_ENTRYPOINT
