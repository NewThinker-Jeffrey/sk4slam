#include "sk4slam_geometry/pnp.h"

#include <Eigen/Dense>

#include "sk4slam_backends/ceres/ceres_helper.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_geometry/third_party/colmap/estimators/absolute_pose.h"
#include "sk4slam_geometry/third_party/colmap/estimators/homography_matrix.h"
#include "sk4slam_geometry/utils.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/matrix.h"

namespace sk4slam {

namespace {
using SO3ParamBlock = CeresManifoldBlock<SO3d::RightOptimizable>;

class PnPCostFunction : public ceres::SizedCostFunction<ceres::DYNAMIC, 3, 3> {
 public:
  using PointPair = PNPEstimator::DataPoint;
  PnPCostFunction(
      const std::vector<size_t>* inlier_indices,
      const std::vector<PointPair>* point_pairs, SO3ParamBlock* so3_block,
      const std::vector<Eigen::Matrix2d>* observation_cov = nullptr)
      : inlier_indices_(inlier_indices),
        point_pairs_(point_pairs),
        so3_block_(so3_block),
        observation_cov_(observation_cov) {
    set_num_residuals(inlier_indices->size() * 2);
  }

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const {
    Eigen::Map<const Eigen::Vector3d> t(parameters[0]);
    SO3d rot = so3_block_->MapToGlobal(parameters[1]);
    const Eigen::Matrix3d& R = rot.matrix();

    Eigen::Matrix2d tmp_sqrt_info;
    Eigen::Matrix2d* sqrt_info = nullptr;
    if (observation_cov_) {
      sqrt_info = &tmp_sqrt_info;
    }

    for (size_t i = 0; i < inlier_indices_->size(); ++i) {
      const size_t idx = (*inlier_indices_)[i];
      const PointPair& point_pair = (*point_pairs_)[idx];
      const Eigen::Vector2d& p2 = point_pair.first;
      const Eigen::Vector3d& p3 = point_pair.second;
      Eigen::Vector3d pinC = R * p3 + t;
      Eigen::Vector2d p2n(pinC[0] / pinC[2], pinC[1] / pinC[2]);
      Eigen::Map<Eigen::Vector2d> residual(residuals + 2 * i);
      residual = p2n - p2;

      if (sqrt_info) {
        *sqrt_info = (*observation_cov_)[idx].llt().matrixL().solve(
            Eigen::Matrix2d::Identity());
        residual = (*sqrt_info) * residual;
      }

      if (jacobians) {
        const double& x = pinC[0];
        const double& y = pinC[1];
        const double& z = pinC[2];
        const double z2 = z * z;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> pp2n_ppinC;
        pp2n_ppinC << 1.0 / z, 0.0, -x / z2, 0.0, 1.0 / z, -y / z2;
        if (jacobians[0]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jt(
              jacobians[0] + 6 * i);
          Jt = pp2n_ppinC;
          if (sqrt_info) {
            Jt = (*sqrt_info) * Jt;
          }
        }
        if (jacobians[1]) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jr(
              jacobians[1] + 6 * i);
          Jr = pp2n_ppinC * R * skew3(-p3);
          if (sqrt_info) {
            Jr = (*sqrt_info) * Jr;
          }
        }
      }
    }
    return true;
  }

 private:
  const std::vector<size_t>* inlier_indices_;
  const std::vector<PointPair>* point_pairs_;
  SO3ParamBlock* so3_block_;
  const std::vector<Eigen::Matrix2d>* observation_cov_;
};

class CameraPositionPrior : public ceres::SizedCostFunction<3, 3, 3> {
 public:
  using PointPair = PNPEstimator::DataPoint;
  CameraPositionPrior(
      SO3ParamBlock* so3_block, const Eigen::Vector3d& cam_position_prior,
      const Eigen::Matrix3d& cam_position_cov)
      : so3_block_(so3_block),
        sqrt_info_(cam_position_cov.llt().matrixL().solve(
            Eigen::Matrix3d::Identity())) {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const {
    Eigen::Map<const Eigen::Vector3d> t(parameters[0]);
    SO3d rot = so3_block_->MapToGlobal(parameters[1]);
    const Eigen::Matrix3d& R = rot.matrix();

    Eigen::Vector3d invR_t = R.transpose() * t;
    Eigen::Vector3d error = cam_position_ + invR_t;
    if (jacobians) {
      if (jacobians[0]) {
        auto Jt = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            jacobians[0]);
        Jt = sqrt_info_ * R.transpose();
      }
      if (jacobians[1]) {
        auto Jr = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            jacobians[1]);
        Jr = sqrt_info_ * SO3d::hat(invR_t);
      }
    }

    if (residuals) {
      Eigen::Map<Eigen::Vector3d> residuals_vec(residuals);
      residuals_vec = sqrt_info_ * error;
    }
    return true;
  }

 private:
  SO3ParamBlock* so3_block_;
  Eigen::Matrix3d sqrt_info_;
  Eigen::Vector3d cam_position_;
};

}  // namespace

bool PNPEstimator::refinePnP(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param,
    bool fix_rotation, const int max_iterations,
    Eigen::Matrix<double, 6, 6>* cov,
    const std::vector<Eigen::Matrix2d>* observation_cov,
    const Eigen::Vector3d* cam_position_prior,
    const Eigen::Matrix3d* cam_position_prior_cov, bool print_iterations) {
  CeresProblem problem;
  SO3ParamBlock so3_block(&problem, SO3d(param->block<3, 3>(0, 0)));
  Eigen::Vector3d translation = param->block<3, 1>(0, 3);
  problem.AddResidualBlock(
      new PnPCostFunction(
          &inlier_indices, &all_points, &so3_block, observation_cov),
      nullptr, translation.data(), so3_block.LocalData());
  if (fix_rotation) {
    problem.SetParameterBlockConstant(so3_block.LocalData());
  }

  if (cam_position_prior) {
    ASSERT(cam_position_prior_cov);
    problem.AddResidualBlock(
        new CameraPositionPrior(
            &so3_block, *cam_position_prior, *cam_position_prior_cov),
        nullptr, translation.data(), so3_block.LocalData());
  }

  bool ret = true;

  if (max_iterations != 0) {
    ceres::Solver::Options options;
    options.max_num_iterations = max_iterations;
    ceres::Solver::Summary summary;
    CeresSolve(options, &problem, &summary, print_iterations);
    // if (summary.termination_type == ceres::CONVERGENCE) {
    //   ret = false;
    // }
    param->block<3, 3>(0, 0) = so3_block.GetGlobal().matrix();
    param->block<3, 1>(0, 3) = translation;
  }

  if (cov) {
    using ParameterBlocks = std::vector<const double*>;
    ceres::Covariance::Options cov_options;
    ceres::Covariance covariance(cov_options);
    ParameterBlocks parameter_blocks = {
        so3_block.LocalData(), translation.data()};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      LOGE("covariance computing error !!!");
      ret = false;
    } else {
      // The covariance returned from ceres is in row-major order, while by
      // default Eigen::Matrix stores data in column-major order. However, we
      // don't have to converse the storage order since the covariance matrix
      // itself is a symmetric matrix.
      //
      // http://ceres-solver.org/nnls_covariance.html#_CPPv4NK5ceres18GetCovarianceBlockEPKdPKdPd
      //
      covariance.GetCovarianceMatrixInTangentSpace(
          parameter_blocks, cov->data());
      LOGA("cov = \n%s", toStr(*cov, Precision(8)).c_str());
    }
  }

  return ret;
}

std::vector<double> PNPEstimator::errors(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points, const Parameter& model) const {
  std::vector<Eigen::Vector2d> Xs;
  std::vector<Eigen::Vector3d> Xprimes;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, all_points, &Xs, &Xprimes);

  std::vector<double> errs;
  // sk4slam_colmap::EPNPEstimator::Residuals(Xs, Xprimes,
  // convertIsometry3ToMatrix3x4(model), &errs);

  sk4slam_colmap::EPNPEstimator::Residuals(Xs, Xprimes, model, &errs);
  return errs;
}

bool PNPEstimator::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
  // Eigen::Matrix<double, 6, 6> cov;
  // {
  //   std::vector<Eigen::Matrix2d> observation_cov(all_points.size(), 0.01 *
  //   Eigen::Matrix2d::Identity()); refinePnP(inlier_indices, all_points,
  //   param, false, 1, &cov, &observation_cov);
  // }
  // {
  //   refinePnP(inlier_indices, all_points, param, false, 1, &cov);
  // }

  return refinePnP(
      inlier_indices, all_points, param, false, 1, nullptr, nullptr,
      lo_cam_position_prior_, lo_cam_position_prior_cov_);
}

std::vector<PNPEstimator::Parameter> EPNPEstimator::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  if (selected_indices.size() < kMinimalSampleSize) {
    return std::vector<Parameter>();
  }

  std::vector<Eigen::Vector2d> Xs;
  std::vector<Eigen::Vector3d> Xprimes;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, all_points, &Xs, &Xprimes);

  // return
  // convertMatrix3x4ToIsometry3(sk4slam_colmap::EPNPEstimator::Estimate(Xs,
  // Xprimes));
  return sk4slam_colmap::EPNPEstimator::Estimate(Xs, Xprimes);
}

std::vector<PNPEstimator::Parameter> P3PEstimator::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  if (selected_indices.size() < kMinimalSampleSize) {
    return std::vector<Parameter>();
  }

  std::vector<Eigen::Vector2d> Xs(3);
  std::vector<Eigen::Vector3d> Xprimes(3);

  // compute all the posible solutions
  for (size_t i = 0; i < 3; ++i) {
    size_t idx = selected_indices[i];
    Xs[i] = all_points[idx].first;
    Xprimes[i] = all_points[idx].second;
  }
  auto poses = sk4slam_colmap::P3PEstimator::Estimate(Xs, Xprimes);
  if (poses.empty()) {
    return std::vector<Parameter>();
  }
  if (selected_indices.size() == 3) {
    return poses;
  }

  // find the best pose if we have more than 3 points
  LOGA(
      "P3P: We get %d solutions with the first 3 points, but we only return "
      "the "
      "best one that minimizes the reprojection errors of all the points (%d "
      "in total)",
      poses.size(), selected_indices.size());
  auto errors_sum = [&](const Parameter& pose) {
    std::vector<double> errors =
        this->errors(selected_indices, all_points, pose);
    return std::accumulate(errors.begin(), errors.end(), 0.0);
  };
  int best_i = 0;
  double best_err = errors_sum(poses[best_i]);
  for (int i = 1; i < poses.size(); ++i) {
    double err = errors_sum(poses[i]);
    if (err < best_err) {
      best_err = err;
      best_i = i;
    }
  }
  return {poses[best_i]};
}

std::vector<PNPEstimator::Parameter> CoplanarP4PEstimator::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  if (selected_indices.size() < kMinimalSampleSize) {
    return std::vector<Parameter>();
  }

  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  Xs_2d.resize(selected_indices.size());
  Xprimes_2d.resize(selected_indices.size());
  for (size_t i = 0; i < selected_indices.size(); ++i) {
    size_t idx = selected_indices[i];
    Xs_2d[i] = all_points[idx].first;
    const Eigen::Vector3d& Xprime_3d = all_points[idx].second;
    Xprimes_2d[i] << Xprime_3d.x(), Xprime_3d.y();
  }

  // We need to compute the homography matrix H that:
  //    H * Xprime = X
  std::vector<Eigen::Matrix3d> Hs =
      sk4slam_colmap::HomographyMatrixEstimator::Estimate(
          // Xs_2d, Xprimes_2d);
          Xprimes_2d,
          Xs_2d);  // We need to compute the homography matrix H that: H
                   // * Xprime = X
  if (Hs.empty()) {
    return std::vector<Parameter>();
  }

  // Compare the two equations:
  //   [R1  R2  t ] * [x'  y'  1]^T = alpha * [x  y  1]^T
  //   [H1  H2  H3] * [x'  y'  1]^T =  beta * [x  y  1]^T
  // We get:
  //    R1 = H1 / s
  //    R2 = H2 / s
  //    t  = H3 / s
  //   |s| = sqrt(|H1| * |H2|)
  //    The sign of s shuold make t.z positive.
  const Eigen::Matrix3d& H = Hs[0];
  Eigen::Vector3d H1 = H.col(0);
  Eigen::Vector3d H2 = H.col(1);
  Eigen::Vector3d H3 = H.col(2);
  double s = std::sqrt(H1.norm() * H2.norm());
  if (H3.z() < 0) {
    s = -s;
  }

  Eigen::Vector3d R1 = H1 / s;
  Eigen::Vector3d R2 = H2 / s;
  Eigen::Vector3d R3 = R1.cross(R2);
  Eigen::Vector3d t = H3 / s;

  Eigen::Matrix3d R;
  R << R1, R2, R3;

  // Make sure R is a perfect rotation matrix
  // R = SO3d::expM(SO3d::logM(R));
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  R = svd.matrixU() * svd.matrixV().transpose();

  LOGA("H: \n%s", toStr(H).c_str());
  LOGA("s: %f", s);
  LOGA("R1: %s", toStr(R1.transpose()).c_str());
  LOGA("R2: %s", toStr(R2.transpose()).c_str());
  LOGA("R3: %s", toStr(R3.transpose()).c_str());
  LOGA("t: %s", toStr(t.transpose()).c_str());
  LOGA("R: \n%s", toStr(R).c_str());

  std::vector<Parameter> parameters(1);
  parameters[0] << R, t;
  return parameters;
}

std::vector<PNPEstimator::Parameter> KnownRotationP2PEstimator::compute(
    const std::vector<size_t>& selected_indices,
    const std::vector<DataPoint>& all_points) const {
  if (selected_indices.size() < kMinimalSampleSize) {
    return std::vector<Parameter>();
  }
  Eigen::MatrixXd A(selected_indices.size() * 2, 3);
  Eigen::VectorXd b(selected_indices.size() * 2);

  for (size_t i = 0; i < selected_indices.size(); ++i) {
    size_t idx = selected_indices[i];
    const Eigen::Vector2d& X = all_points[idx].first;
    const Eigen::Vector3d& Xprime = all_points[idx].second;
    const double& u = X.x();
    const double& v = X.y();
    Eigen::Vector3d Rp = known_rotation_ * Xprime;

    A.row(i * 2) << 1, 0, -u;
    b(i * 2) = u * Rp.z() - Rp.x();

    A.row(i * 2 + 1) << 0, 1, -v;
    b(i * 2 + 1) = v * Rp.z() - Rp.y();
  }

  // Solve by QR decomposition
  auto qr = A.colPivHouseholderQr();
  if (qr.rank() < 3) {
    return {};
  }

  Eigen::Vector3d qr_solution = qr.solve(b);
  Eigen::Vector3d t = qr_solution;
  PNPEstimator::Parameter param;
  param << known_rotation_, t;

  return {param};
}

bool KnownRotationP2PEstimator::localOptimize(
    const std::vector<size_t>& inlier_indices,
    const std::vector<DataPoint>& all_points, Parameter* param) const {
  return refinePnP(
      inlier_indices, all_points, param, true, 1, nullptr, nullptr,
      lo_cam_position_prior_, lo_cam_position_prior_cov_);
}

}  // namespace sk4slam
