#include "sk4slam_geometry/homography_matrix.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_geometry/fundamental_matrix.h"
#include "sk4slam_geometry/third_party/colmap/estimators/homography_matrix.h"
#include "sk4slam_geometry/third_party/colmap/geometry/homography_matrix.h"
#include "sk4slam_geometry/utils.h"
#include "sk4slam_math/matrix.h"

namespace sk4slam {

std::vector<Eigen::Matrix3d> HomographyMatrix::solveDLT(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  return sk4slam_colmap::HomographyMatrixEstimator::Estimate(Xs_2d, Xprimes_2d);
}

// At least 3 points are needed to compute the homography if the fundamental
// matrix is known.
std::vector<Eigen::Matrix3d> HomographyMatrix::solveWithKnownF(
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs, const Eigen::Matrix3d& F) {
  ASSERT(selected_indices.size() >= 3);

  std::vector<Eigen::Vector3d> Xs, Xprimes;

  convertSelectedPointPairsToHomogeneousXsAndXprimes(
      selected_indices, point_pairs, &Xs, &Xprimes);
  ASSERT(Xs.size() == Xprimes.size());
  ASSERT(Xs.size() == selected_indices.size());

  // See "Multiple View Geometry in Computer Vision - Second Edition",
  // Result 13.6

  Eigen::Vector3d e_prime = FundamentalMatrix::getEpipole(F, false);
  Eigen::Matrix3d e_prime_skew = skew3(e_prime);
  Eigen::Matrix3d A = e_prime_skew * F;

  // TODO(jeffrey): use more points if possible? (now we only use 3 points)
  std::vector<Eigen::Vector3d> xs = {Xs[0], Xs[1], Xs[2]};
  std::vector<Eigen::Vector3d> xprimes = {Xprimes[0], Xprimes[1], Xprimes[2]};

  Eigen::Matrix3d M;
  M << xs[0].transpose(), xs[1].transpose(), xs[2].transpose();
  Eigen::Matrix3d M_inv;
  if (!safeInverseMatrix(M, &M_inv)) {
    return std::vector<Eigen::Matrix3d>();
  }

  Eigen::Vector3d b;
  for (size_t i = 0; i < 3; i++) {
    auto tmp1 = xprimes[i].cross(A * xs[i]);  // xi' X (A*xi)
    auto tmp2 = xprimes[i].cross(e_prime);    // xi' X e'
    b[i] = tmp1.dot(tmp2) / tmp2.squaredNorm();
  }

  Eigen::Matrix3d H = A - e_prime * (M_inv * b).transpose();

  std::vector<Eigen::Matrix3d> Hs;
  Hs.push_back(H);
  return Hs;
}

void HomographyMatrix::computePose(
    const Eigen::Matrix3d& H, const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs, Eigen::Matrix3d* R,
    Eigen::Vector3d* t, Eigen::Vector3d* n,
    std::vector<Eigen::Vector3d>* points3D) {
  std::vector<Eigen::Vector2d> Xs_2d, Xprimes_2d;
  convertSelectedPointPairsToXsAndXprimes(
      selected_indices, point_pairs, &Xs_2d, &Xprimes_2d);

  static const Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  sk4slam_colmap::PoseFromHomographyMatrix(
      H, K, K, Xs_2d, Xprimes_2d, R, t, n, points3D);
}

std::vector<double> HomographyMatrix::computeSquaredSampsonErrors(
    const Eigen::Matrix<double, 3, 3>& H,
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<double> errs;
  errs.reserve(selected_indices.size());

  for (size_t i : selected_indices) {
    const auto& pair = point_pairs[i];
    // We multiply by 0.5 (i.e. /2) since the Sampson error for homography has
    // 2 degrees of freedom.
    errs.push_back(
        0.5 * computeSquaredSampsonError(H, pair.first, pair.second));
  }

  return errs;
}

std::vector<double> HomographyMatrix::computeSquaredAlgebraicErrors(
    const Eigen::Matrix<double, 3, 3>& H,
    const std::vector<size_t>& selected_indices,
    const std::vector<PointPair>& point_pairs) {
  std::vector<double> errs;
  errs.reserve(selected_indices.size());

  for (size_t i : selected_indices) {
    const auto& pair = point_pairs[i];
    Eigen::Vector2d algebra_err;

    Eigen::Vector3d Hx = H * pair.first.homogeneous();
    // algebra_err << - Hx[1] + pair.second[1] * Hx[2],
    //                  Hx[0] - pair.second[0] * Hx[2];
    algebra_err << -Hx[1] / Hx[2] + pair.second[1],
        Hx[0] / Hx[2] - pair.second[0];
    errs.push_back(0.5 * algebra_err.squaredNorm());
  }

  return errs;
}

}  // namespace sk4slam
