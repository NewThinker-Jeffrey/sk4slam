#pragma once

#include <Eigen/Core>
#include <numeric>

#include "sk4slam_geometry/fundamental_matrix.h"

namespace sk4slam {

class EssentialMatrix {
 public:
  // Xprime^T * E * X = 0
  //
  // pair.first = X
  // pair.second = Xprime

  using PointPair = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

  // The 5-point algorithm. (Might output multiple solutions, 10 at most)
  static std::vector<Eigen::Matrix3d> solveWith5Points(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  // The 8-point algorithm. (1 solution at most)
  static std::vector<Eigen::Matrix3d> solveWith8Points(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  // The 2-point algorithm (with known rotation).
  // TODO(jeffrey): implement this.
  static std::vector<Eigen::Matrix3d> solveWithKnownRotation(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs, const Eigen::Matrix3d& R);

  // This function assumes the camera intrinsics 'K = I' and all the image
  // points are on the normalized imaging plane. We actually borrowed the code
  // from COLMAP.
  //
  // Recover the most probable pose from the given essential matrix.
  //
  // The pose of the first image is assumed to be P = [I | 0].
  //
  // @param E            3x3 essential matrix.
  // @param selected_indices The indices of the point pairs to use.
  // @param point_pairs      All point pairs
  // @param R            Most probable 3x3 rotation matrix.
  // @param t            Most probable 3x1 translation vector.
  // @param points3D     Triangulated 3D points infront of camera.
  static void computePose(
      const Eigen::Matrix3d& E, const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs, Eigen::Matrix3d* R,
      Eigen::Vector3d* t, std::vector<Eigen::Vector3d>* points3D);

  // This function is also borrowed from COLMAP.
  // Find optimal image points, such that:
  //
  //     optimal_point1^t * E * optimal_point2 = 0
  //
  // as described in:
  //
  //   Lindstrom, P., "Triangulation made easy",
  //   Computer Vision and Pattern Recognition (CVPR),
  //   2010 IEEE Conference on , vol., no., pp.1554,1561, 13-18 June 2010
  //
  // @param E                Essential or fundamental matrix.
  // @param point1           Corresponding 2D point in first image.
  // @param point2           Corresponding 2D point in second image.
  // @param optimal_point1   Estimated optimal image point in the first image.
  // @param optimal_point2   Estimated optimal image point in the second image.
  static void findOptimalImageObservations(
      const Eigen::Matrix3d& E, const Eigen::Vector2d& point1,
      const Eigen::Vector2d& point2, Eigen::Vector2d* optimal_point1,
      Eigen::Vector2d* optimal_point2);

  // `points_used_for_initial_h` can be  3,4,or 5:
  //     - 3 is much faster but less accurate (and depends on the accuracy of E)
  //     - 5 is the most accurate but slowest (does not depend on E)
  //     - 4 is a compromise (does not depend on E)
  //
  // If we're using 5-point algorithm, i.e. selected_indices.size() = 5,
  // then `points_used_for_initial_h` is forced to be 5 to ensure that
  // the initial H is reliable.
  static bool check_H_Degenerate(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs,
      const Eigen::Matrix<double, 3, 3>& E, const double err_thr,
      Eigen::Matrix<double, 3, 3>* H, std::vector<size_t>* h_sample_indices,
      size_t points_used_for_initial_h = 3);

  using Manifold = ProductLieGroup<SO3d, SO3d>;

  using Optimizable =
      Manifold::SubRightOptimizable<SubSpaceByAxes<0, 1, /*2,*/ 3, 4, 5>>;

  using CeresParamBlock = CeresManifoldBlock<Optimizable>;

  static Manifold toManifold(const Eigen::Matrix3d& E) {
    using Matrix3 = Eigen::Matrix3d;
    using Vector3 = Eigen::Vector3d;

    Eigen::JacobiSVD<Matrix3> svd(E, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Matrix3 U = svd.matrixU();
    Matrix3 V = svd.matrixV();
    Vector3 singular_values = svd.singularValues();

    // Ensure U and V are special orthogonal (det = 1)
    Vector3 VX = V.col(0);
    Vector3 VY = V.col(1);
    Vector3 VZ = V.col(2);
    if (VX.cross(VY).dot(VZ) < 0) {
      V.col(2) *= -1;
      U.col(2) *= -1;
    }

    return Manifold(SO3d(U), SO3d(V));
  }

  template <typename Manifold>
  static Eigen::Matrix<typename Manifold::Scalar, 3, 3> fromManifold(
      const Manifold& mp) {
    using Scalar = typename Manifold::Scalar;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    Matrix3 U = mp.template part<0>().matrix();
    Matrix3 V = mp.template part<1>().matrix();
    Vector3 singular_values(Scalar(1.0), Scalar(1.0), Scalar(0.0));
    return U * singular_values.asDiagonal() * V.transpose();
  }

  struct CeresOptimizer {
    const std::vector<size_t>* selected_indices;
    const std::vector<PointPair>* point_pairs;

    CeresOptimizer(
        const std::vector<size_t>* selected_indices_in,
        const std::vector<PointPair>* point_pairs_in)
        : selected_indices(selected_indices_in), point_pairs(point_pairs_in) {}

    // F should already be initialized with a good estimate.
    bool optimizeWithCeres(Eigen::Matrix3d* F, int max_iter = 1) const;

   protected:
    virtual ceres::CostFunction* createCostFunction(
        CeresParamBlock* param_blk) const = 0;
  };

  struct SampsonErrorOptimizer : public CeresOptimizer {
    SampsonErrorOptimizer(
        const std::vector<size_t>* selected_indices,
        const std::vector<PointPair>* point_pairs)
        : CeresOptimizer(selected_indices, point_pairs) {}

    int residualSize() const {
      return 4 * selected_indices->size();
    }

    struct Functor {
      Functor(
          const SampsonErrorOptimizer* optimizer, CeresParamBlock* param_blk)
          : optimizer_(optimizer), param_blk_(param_blk) {}

      template <typename T>
      bool operator()(const T* const E_local, T* residuals) const {
        auto E = fromManifold(param_blk_->MapToGlobal(E_local));
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res_eigen(
            residuals, optimizer_->residualSize());
        for (size_t i = 0; i < optimizer_->selected_indices->size(); ++i) {
          size_t idx = optimizer_->selected_indices->at(i);
          Eigen::Matrix<T, 2, 1> x =
              optimizer_->point_pairs->at(idx).first.cast<T>();
          Eigen::Matrix<T, 2, 1> xprime =
              optimizer_->point_pairs->at(idx).second.cast<T>();
          res_eigen.template segment<4>(4 * i) =
              FundamentalMatrix::computeSampsonError(E, x, xprime);
        }
        return true;
      }

     private:
      CeresParamBlock* param_blk_;
      const SampsonErrorOptimizer* optimizer_;
    };

   protected:
    virtual ceres::CostFunction* createCostFunction(
        CeresParamBlock* param_blk) const {
      auto cost_func =
          new ceres::AutoDiffCostFunction<Functor, -1, Optimizable::kDof>(
              new Functor(this, param_blk), residualSize());
      return cost_func;
    }
  };

  struct SquaredSampsonErrorOptimizer : public CeresOptimizer {
    SquaredSampsonErrorOptimizer(
        const std::vector<size_t>* selected_indices,
        const std::vector<PointPair>* point_pairs)
        : CeresOptimizer(selected_indices, point_pairs) {}

    int residualSize() const {
      return 1 * selected_indices->size();
    }

    struct Functor {
      Functor(
          const SquaredSampsonErrorOptimizer* optimizer,
          CeresParamBlock* param_blk)
          : optimizer_(optimizer), param_blk_(param_blk) {}

      template <typename T>
      bool operator()(const T* const E_local, T* residuals) const {
        auto E = fromManifold(param_blk_->MapToGlobal(E_local));
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res_eigen(
            residuals, optimizer_->residualSize());
        for (size_t i = 0; i < optimizer_->selected_indices->size(); ++i) {
          size_t idx = optimizer_->selected_indices->at(i);
          Eigen::Matrix<T, 2, 1> x =
              optimizer_->point_pairs->at(idx).first.cast<T>();
          Eigen::Matrix<T, 2, 1> xprime =
              optimizer_->point_pairs->at(idx).second.cast<T>();
          res_eigen[i] =
              FundamentalMatrix::computeSquaredSampsonError(E, x, xprime);
        }
        return true;
      }

     private:
      CeresParamBlock* param_blk_;
      const SquaredSampsonErrorOptimizer* optimizer_;
    };

   protected:
    virtual ceres::CostFunction* createCostFunction(
        CeresParamBlock* param_blk) const {
      auto cost_func =
          new ceres::AutoDiffCostFunction<Functor, -1, Optimizable::kDof>(
              new Functor(this, param_blk), residualSize());
      return cost_func;
    }
  };
};

}  // namespace sk4slam
