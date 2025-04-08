#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/time.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_geometry/pnp.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"
#include "sk4slam_math/sac.h"

namespace sk4slam {

inline const std::vector<PNPEstimator::DataPoint> getGeneral3D2DPointPairs(
    Eigen::Isometry3d* output_T, int n, const double noise_sigma = 0.0) {
  Eigen::Matrix3d R_C_Obj = SO3d::expM(Eigen::Vector3d(0.1, 0.05, 0.05));
  Eigen::Vector3d t_C_Obj(0.2, 0.1, 2.0);
  Eigen::Isometry3d T_C_Obj(R_C_Obj);
  T_C_Obj.translation() = t_C_Obj;
  *output_T = T_C_Obj;

  auto dist2d = MultivariateUniformDistribution<double>::standard(2);
  auto dist3d = MultivariateUniformDistribution<double>::standard(3);
  std::vector<PNPEstimator::DataPoint> point_pairs;

  std::vector<Eigen::Vector3d> points_3d = {
      {-1.0, -1.0, 0.0},
      {-1.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},
      {1.0, -1.0, 0.5}  // ensure the first 4 points are not coplanar
  };
  while (points_3d.size() < n) {
    points_3d.emplace_back(dist3d());
  }

  std::vector<Eigen::Vector2d> points_2d;
  for (size_t i = 0; i < n; ++i) {
    // LOGI("random_3: %f, %f, %f", random_3[0], random_3[1], random_3[2]);
    Eigen::Vector3d& obj_p3 = points_3d[i];
    obj_p3 *= t_C_Obj.norm() * 0.1;

    Eigen::Vector3d p_in_C = T_C_Obj * obj_p3;
    Eigen::Vector2d pixel = p_in_C.hnormalized() + dist2d() * noise_sigma;
    point_pairs.emplace_back(pixel, obj_p3);
    points_2d.emplace_back(pixel);
  }
  LOGI("General 3d points:\n%s", toStr(points_3d, [](const Eigen::Vector3d& p) {
                                   return p.transpose();
                                 }).c_str());
  LOGI("pixels: \n%s", toStr(points_2d, [](const Eigen::Vector2d& p) {
                         return p.transpose();
                       }).c_str());
  return point_pairs;
}

inline const std::vector<PNPEstimator::DataPoint> getCoplanar3D2DPointPairs(
    Eigen::Isometry3d* output_T, int n, const double noise_sigma = 0.0) {
  Eigen::Matrix3d R_C_Plane = SO3d::expM(Eigen::Vector3d(0.1, 0.05, 0.05));
  Eigen::Vector3d t_C_Plane(0.2, 0.1, 2.0);
  Eigen::Isometry3d T_C_Plane(R_C_Plane);
  T_C_Plane.translation() = t_C_Plane;
  *output_T = T_C_Plane;

  auto dist2d = MultivariateUniformDistribution<double>::standard(2);
  std::vector<PNPEstimator::DataPoint> point_pairs;

  // Add coplanar points.
  std::vector<Eigen::Vector3d> points_3d = {
      {-1.0, -1.0, 0.0},
      {-1.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},  // ensure the first 3 points are not on a line
      {1.0, -1.0, 0.0}};
  while (points_3d.size() < n) {
    Eigen::Vector2d obj_p2 = dist2d();
    points_3d.emplace_back(obj_p2.x(), obj_p2.y(), 0.0);
  }
  std::vector<Eigen::Vector2d> points_2d;
  for (size_t i = 0; i < n; ++i) {
    Eigen::Vector3d& obj_p3 = points_3d[i];
    obj_p3 *= t_C_Plane.norm() * 0.1;
    Eigen::Vector3d p_in_C = T_C_Plane * obj_p3;
    Eigen::Vector2d pixel = p_in_C.hnormalized() + dist2d() * noise_sigma;
    point_pairs.emplace_back(pixel, obj_p3);
    points_2d.emplace_back(pixel);
  }
  LOGI(
      "Coplanar 3d points:\n%s", toStr(points_3d, [](const Eigen::Vector3d& p) {
                                   return p.transpose();
                                 }).c_str());
  LOGI("pixels: \n%s", toStr(points_2d, [](const Eigen::Vector2d& p) {
                         return p.transpose();
                       }).c_str());
  return point_pairs;
}

double computeSquaredErrorsSum(
    PNPEstimator* estimator,
    const std::vector<PNPEstimator::DataPoint>& point_pairs,
    const PNPEstimator::Parameter& model) {
  std::vector<double> errs = estimator->errors2(point_pairs, model);
  return std::accumulate(errs.begin(), errs.end(), 0.0);
}

void checkReprojErr(
    PNPEstimator* estimator,
    const std::vector<PNPEstimator::DataPoint>& point_pairs,
    const PNPEstimator::Parameter& model, const double noise_sigma = 0.0) {
  std::vector<double> errs = estimator->errors2(point_pairs, model);
  LOGI("Reproj errors:\n%s", toStr(errs, sqrt).c_str());
  double err_ave = std::accumulate(errs.begin(), errs.end(), 0.0) / errs.size();
  LOGI("Reproj error average: %f", sqrt(err_ave));
  ASSERT_NEAR(sqrt(err_ave), 0, 3.0 * noise_sigma + 1e-3);
}

void checkPose(
    const PNPEstimator::Parameter& pose, const Eigen::Isometry3d& ref_pose) {
  Eigen::Matrix3d est_R = pose.block(0, 0, 3, 3);
  Eigen::Vector3d est_t = pose.block(0, 3, 3, 1);
  LOGI("Real R_C_Obj:\n%s", toStr(ref_pose.rotation()).c_str());
  LOGI("Recovered R_C_Obj:\n%s", toStr(est_R).c_str());
  LOGI("Real t_C_Obj: %s", toStr(ref_pose.translation().transpose()).c_str());
  LOGI("Recovered t_C_Obj: %s", toStr(est_t.transpose()).c_str());

  Eigen::Matrix3d rotation_err =
      ref_pose.rotation().transpose() * est_R - Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_err = ref_pose.translation() - est_t;

  LOGI(
      "Recovered rotation_err(%f):\n%s", rotation_err.squaredNorm(),
      toStr(rotation_err).c_str());
  LOGI(
      "Recovered translation_err(%f): %s", translation_err.squaredNorm(),
      toStr(translation_err.transpose()).c_str());
  ASSERT_NEAR(rotation_err.squaredNorm(), 0, 1e-2);
  ASSERT_NEAR(translation_err.squaredNorm(), 0, 1e-2);
}

}  // namespace sk4slam
