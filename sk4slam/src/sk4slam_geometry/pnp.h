#pragma once

#include <Eigen/Core>
#include <numeric>

#include "sk4slam_math/sac.h"

namespace sk4slam {

class PNPEstimator {
 public:
  using DataPoint = std::pair<Eigen::Vector2d, Eigen::Vector3d>;

  // using Parameter = Eigen::Isometry3d;
  using Parameter = Eigen::Matrix<double, 3, 4>;

  // Constants used to compute the default RANSAC options
  static constexpr int kDefatultFocal = 400;
  static constexpr int kDefatultFocalSquare = kDefatultFocal * kDefatultFocal;
  static RansacOptions defaultRansacOptions() {
    return RansacOptions(
        (3.0 * 3.0) /
            kDefatultFocalSquare,  // error_thr (for normalized images)
        0.999,                     // confidence
        1000,                      // max_iter
        0.0,                       // initial_inlier_ratio_thr
        0,                         // local_opt_max_iter
                                   // ^ None zero local_opt_max_iter may
                                   // ^ significantly slow down the computation
        1                          // final_opt_max_iter
    );                             // NOLINT
  }

  PNPEstimator(
      const Eigen::Vector3d* lo_cam_position_prior = nullptr,
      const Eigen::Matrix3d* lo_cam_position_prior_cov = nullptr)
      : lo_cam_position_prior_(lo_cam_position_prior),
        lo_cam_position_prior_cov_(lo_cam_position_prior_cov) {}

  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const = 0;

  std::vector<double> errors(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points, const Parameter& model) const;

  // Do LO with all the inliers (needed by LO-RANSAC framework).
  // `param` shuold already hold a reasonable initial guess for the model
  // parameters.
  bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;

 public:
  /// @brief  Refine the PnP solution using the provided inliers, and compute
  /// the covariance of the refined solution optionally.
  ///
  /// Note the covariance is with respect to the AffineRightPerturbation (See
  /// @ref Pose3WithCovariance for details).
  static bool refinePnP(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param,
      bool fix_rotation = false, const int max_iterations = 5,
      Eigen::Matrix<double, 6, 6>* cov = nullptr,
      const std::vector<Eigen::Matrix2d>* observation_cov = nullptr,
      const Eigen::Vector3d* cam_position_prior = nullptr,
      const Eigen::Matrix3d* cam_position_prior_cov = nullptr,
      bool print_iterations = false);

  static bool refinePnP2(
      const std::vector<DataPoint>& all_points, Parameter* param,
      bool fix_rotation = false, const int max_iterations = 5,
      Eigen::Matrix<double, 6, 6>* cov = nullptr,
      const std::vector<Eigen::Matrix2d>* observation_cov = nullptr,
      const Eigen::Vector3d* cam_position_prior = nullptr,
      const Eigen::Matrix3d* cam_position_prior_cov = nullptr,
      bool print_iterations = false) {
    std::vector<size_t> selected_indices;
    selected_indices.resize(all_points.size());
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    return refinePnP(
        selected_indices, all_points, param, fix_rotation, max_iterations, cov,
        observation_cov, cam_position_prior, cam_position_prior_cov,
        print_iterations);
  }

 public:
  std::vector<Parameter> compute2(
      const std::vector<DataPoint>& all_points) const {
    std::vector<size_t> selected_indices;
    selected_indices.resize(all_points.size());
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    return compute(selected_indices, all_points);
  }

  std::vector<double> errors2(
      const std::vector<DataPoint>& all_points, const Parameter& model) const {
    std::vector<size_t> selected_indices;
    selected_indices.resize(all_points.size());
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    return errors(selected_indices, all_points, model);
  }

  bool localOptimize2(
      const std::vector<DataPoint>& all_points, Parameter* param) const {
    std::vector<size_t> inlier_indices;
    inlier_indices.resize(all_points.size());
    std::iota(inlier_indices.begin(), inlier_indices.end(), 0);
    return localOptimize(inlier_indices, all_points, param);
  }

 protected:
  const Eigen::Vector3d* lo_cam_position_prior_;
  const Eigen::Matrix3d* lo_cam_position_prior_cov_;
};

class EPNPEstimator : public PNPEstimator {
 public:
  static constexpr int kMinimalSampleSize = 4;

  using PNPEstimator::PNPEstimator;  // inherit constructor

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const override;

  static Ransac<EPNPEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions(),
      const Eigen::Vector3d* lo_cam_position_prior = nullptr,
      const Eigen::Matrix3d* lo_cam_position_prior_cov = nullptr) {
    return Ransac<EPNPEstimator>(
               options,
               EPNPEstimator(lo_cam_position_prior, lo_cam_position_prior_cov))
        .solve(point_pairs);
  }
};

class P3PEstimator : public PNPEstimator {
 public:  // basic properties.
  static constexpr int kMinimalSampleSize = 3;

  using PNPEstimator::PNPEstimator;  // inherit constructor

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const override;

  static Ransac<P3PEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions(),
      const Eigen::Vector3d* lo_cam_position_prior = nullptr,
      const Eigen::Matrix3d* lo_cam_position_prior_cov = nullptr) {
    return Ransac<P3PEstimator>(
               options,
               P3PEstimator(lo_cam_position_prior, lo_cam_position_prior_cov))
        .solve(point_pairs);
  }
};

class CoplanarP4PEstimator : public PNPEstimator {
 public:
  static constexpr int kMinimalSampleSize = 4;

  using PNPEstimator::PNPEstimator;  // inherit constructor

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const override;

  static Ransac<CoplanarP4PEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions(),
      const Eigen::Vector3d* lo_cam_position_prior = nullptr,
      const Eigen::Matrix3d* lo_cam_position_prior_cov = nullptr) {
    return Ransac<CoplanarP4PEstimator>(
               options, CoplanarP4PEstimator(
                            lo_cam_position_prior, lo_cam_position_prior_cov))
        .solve(point_pairs);
  }
};

class KnownRotationP2PEstimator : public PNPEstimator {
  Eigen::Matrix3d known_rotation_;

 public:
  static constexpr int kMinimalSampleSize = 2;

  KnownRotationP2PEstimator(
      const Eigen::Matrix3d& known_rotation,
      const Eigen::Vector3d* lo_cam_position_prior = nullptr,
      const Eigen::Matrix3d* lo_cam_position_prior_cov = nullptr)
      : PNPEstimator(lo_cam_position_prior, lo_cam_position_prior_cov),
        known_rotation_(known_rotation) {}

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const override;

  bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;

  static Ransac<KnownRotationP2PEstimator>::Report ransac(
      const Eigen::Matrix3d& known_rotation,
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions(),
      const Eigen::Vector3d* lo_cam_position_prior = nullptr,
      const Eigen::Matrix3d* lo_cam_position_prior_cov = nullptr) {
    return Ransac<KnownRotationP2PEstimator>(
               options, KnownRotationP2PEstimator(
                            known_rotation, lo_cam_position_prior,
                            lo_cam_position_prior_cov))
        .solve(point_pairs);
  }
};

}  // namespace sk4slam
