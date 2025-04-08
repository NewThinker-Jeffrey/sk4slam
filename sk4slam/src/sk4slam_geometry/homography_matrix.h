#pragma once

#include <Eigen/Core>
#include <numeric>

namespace sk4slam {

class HomographyMatrix {
 public:
  // H * X = Xprime
  //
  // pair.first = X
  // pair.second = Xprime

  using PointPair = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

  // At least 4 points are needed to compute the homography.
  // (1 solution at most)
  static std::vector<Eigen::Matrix3d> solveDLT(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  // At least 3 points are needed to compute the homography if the fundamental
  // matrix is known. (1 solution at most)
  static std::vector<Eigen::Matrix3d> solveWithKnownF(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs, const Eigen::Matrix3d& F);

  // This function assumes the camera intrinsics 'K = I' and all the image
  // points are on the normalized imaging plane. We actually borrowed the code
  // from COLMAP.
  //
  // Recover the most probable pose from the given homography matrix.
  //
  // The pose of the first image is assumed to be P = [I | 0].
  //
  // @param H            3x3 homography matrix.
  // @param selected_indices The indices of the point pairs to use.
  // @param point_pairs      All point pairs
  // @param R            Most probable 3x3 rotation matrix.
  // @param t            Most probable 3x1 translation vector.
  // @param n            Most probable 3x1 normal vector.
  // @param points3D     Triangulated 3D points infront of camera
  //                     (only if homography is not pure-rotational).
  static void computePose(
      const Eigen::Matrix3d& H, const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs, Eigen::Matrix3d* R,
      Eigen::Vector3d* t, Eigen::Vector3d* n,
      std::vector<Eigen::Vector3d>* points3D);

  static std::vector<double> computeSquaredAlgebraicErrors(
      const Eigen::Matrix<double, 3, 3>& H,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  static std::vector<double> computeSquaredSampsonErrors(
      const Eigen::Matrix<double, 3, 3>& H,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  static double computeSquaredSampsonErrorsSum(
      const Eigen::Matrix<double, 3, 3>& H,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs) {
    std::vector<double> errs =
        computeSquaredSampsonErrors(H, selected_indices, point_pairs);
    return std::accumulate(errs.begin(), errs.end(), 0.0);
  }

  static double computeSquaredSampsonErrorsAve(
      const Eigen::Matrix<double, 3, 3>& H,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs) {
    return computeSquaredSampsonErrorsSum(H, selected_indices, point_pairs) /
           selected_indices.size();
  }

  template <typename Scalar>
  static Eigen::Matrix<Scalar, 4, 1> computeSampsonError(
      const Eigen::Matrix<Scalar, 3, 3>& H,
      const Eigen::Matrix<Scalar, 2, 1>& x,
      const Eigen::Matrix<Scalar, 2, 1>& xprime) {
    Eigen::Matrix<Scalar, 3, 1> Hx = H * x.homogeneous();
    Eigen::Matrix<Scalar, 2, 1> algebra_err;
    Eigen::Matrix<Scalar, 2, 2> pe_over_px, pe_over_pxprime;
    // clang-format off
    algebra_err << - Hx[1] + xprime[1] * Hx[2],
                     Hx[0] - xprime[0] * Hx[2];
    pe_over_px << -H(1,0) + xprime[1] * H(2,0),  -H(1,1) + xprime[1] * H(2,1),  // NOLINT
                   H(0,0) - xprime[0] * H(2,0),   H(0,1) - xprime[0] * H(2,1);  // NOLINT
    pe_over_pxprime << Scalar(0),  Hx[2],
                          -Hx[2],  Scalar(0);
    Eigen::Matrix<Scalar, 2, 4> J;
    J << pe_over_px, pe_over_pxprime;
    Eigen::Matrix<Scalar, 4, 2> JT = J.transpose();
    Eigen::Matrix<Scalar, 2, 2> JJT = J * JT;
    // clang-format on
    return -JT * JJT.inverse() * algebra_err;
  }

  template <typename Scalar>
  static Scalar computeSquaredSampsonError(
      const Eigen::Matrix<Scalar, 3, 3>& H,
      const Eigen::Matrix<Scalar, 2, 1>& x,
      const Eigen::Matrix<Scalar, 2, 1>& xprime) {
    Eigen::Matrix<Scalar, 3, 1> Hx = H * x.homogeneous();
    Eigen::Matrix<Scalar, 2, 1> algebra_err;
    Eigen::Matrix<Scalar, 2, 2> pe_over_px, pe_over_pxprime;
    // clang-format off
    algebra_err << - Hx[1] + xprime[1] * Hx[2],
                     Hx[0] - xprime[0] * Hx[2];
    pe_over_px << -H(1,0) + xprime[1] * H(2,0),  -H(1,1) + xprime[1] * H(2,1),  // NOLINT
                   H(0,0) - xprime[0] * H(2,0),   H(0,1) - xprime[0] * H(2,1);  // NOLINT
    pe_over_pxprime << Scalar(0),  Hx[2],
                          -Hx[2],  Scalar(0);
    Eigen::Matrix<Scalar, 2, 4> J;
    J << pe_over_px, pe_over_pxprime;
    Eigen::Matrix<Scalar, 4, 2> JT = J.transpose();
    Eigen::Matrix<Scalar, 2, 2> JJT = J * JT;
    // clang-format on
    return algebra_err.transpose() * JJT.inverse() * algebra_err;
  }
};

}  // namespace sk4slam
