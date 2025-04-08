#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <numeric>

#include "sk4slam_backends/ceres/ceres_helper.h"
#include "sk4slam_liegroups/Rp.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/direct_product.h"

namespace sk4slam {

class FundamentalMatrix {
 public:
  // Xprime^T * F * X = 0
  //
  // pair.first = X
  // pair.second = Xprime

  using PointPair = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

  // The 7-point algorithm. (Might output multiple solutions, 3 at most)
  static std::vector<Eigen::Matrix3d> solveWith7Points(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  // The 8-point algorithm. (1 solution at most)
  static std::vector<Eigen::Matrix3d> solveWith8Points(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  // The plane-and-parallax algorithm. (1 solution at most)
  static std::vector<Eigen::Matrix3d> solveWithKnownH(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs, const Eigen::Matrix3d& H);

  static Eigen::Vector3d getEpipole(
      const Eigen::Matrix3d& F, const bool left_image);

  // `points_used_for_initial_h` can be  3,4,or 5:
  //     - 3 is much faster but less accurate (and depends on the accuracy of F)
  //     - 5 is the most accurate but slowest (does not depend on F)
  //     - 4 is a compromise (does not depend on F)
  static bool check_H_Degenerate(
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs,
      const Eigen::Matrix<double, 3, 3>& F, const double err_thr,
      Eigen::Matrix<double, 3, 3>* H, std::vector<size_t>* h_sample_indices,
      size_t points_used_for_initial_h = 3);

  static std::vector<double> computeSquaredAlgebraicErrors(
      const Eigen::Matrix<double, 3, 3>& F,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  static std::vector<double> computeSquaredSampsonErrors(
      const Eigen::Matrix<double, 3, 3>& F,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs);

  static double computeSquaredSampsonErrorsSum(
      const Eigen::Matrix<double, 3, 3>& F,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs) {
    std::vector<double> errs =
        computeSquaredSampsonErrors(F, selected_indices, point_pairs);
    return std::accumulate(errs.begin(), errs.end(), 0.0);
  }

  static double computeSquaredSampsonErrorsAve(
      const Eigen::Matrix<double, 3, 3>& F,
      const std::vector<size_t>& selected_indices,
      const std::vector<PointPair>& point_pairs) {
    return computeSquaredSampsonErrorsSum(F, selected_indices, point_pairs) /
           selected_indices.size();
  }

  using Manifold = ProductLieGroup<SO3d, SO3d, Rpd>;

  using Optimizable = Manifold::RightOptimizable;

  using CeresParamBlock = CeresManifoldBlock<Optimizable>;

  static Manifold toManifold(const Eigen::Matrix3d& F) {
    using Matrix3 = Eigen::Matrix3d;
    using Vector3 = Eigen::Vector3d;

    Eigen::JacobiSVD<Matrix3> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
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

    double scale = singular_values[1] / singular_values[0];
    return Manifold(SO3d(U), SO3d(V), Rpd(scale));
  }

  template <typename Manifold>
  static Eigen::Matrix<typename Manifold::Scalar, 3, 3> fromManifold(
      const Manifold& mp) {
    using Scalar = typename Manifold::Scalar;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    Matrix3 U = mp.template part<0>().matrix();
    Matrix3 V = mp.template part<1>().matrix();
    Vector3 singular_values(
        Scalar(1.0), mp.template part<2>().value(), Scalar(0.0));
    return U * singular_values.asDiagonal() * V.transpose();
  }

  template <typename Scalar>
  static Eigen::Matrix<Scalar, 4, 1> computeSampsonError(
      const Eigen::Matrix<Scalar, 3, 3>& F,
      const Eigen::Matrix<Scalar, 2, 1>& x,
      const Eigen::Matrix<Scalar, 2, 1>& xprime) {
    Eigen::Matrix<Scalar, 3, 1> Ft_xprime =
        F.transpose() * xprime.homogeneous();
    Eigen::Matrix<Scalar, 3, 1> F_x = F * x.homogeneous();
    Eigen::Matrix<Scalar, 1, 2> pe_over_px, pe_over_pxprime;
    Scalar algebra_err = xprime.homogeneous().dot(F_x);
    Scalar denominator = Ft_xprime[0] * Ft_xprime[0] +
                         Ft_xprime[1] * Ft_xprime[1] + F_x[0] * F_x[0] +
                         F_x[1] * F_x[1];

    pe_over_px << Ft_xprime[0], Ft_xprime[1];
    pe_over_pxprime << F_x[0], F_x[1];
    Eigen::Matrix<Scalar, 4, 1> JT;
    JT << pe_over_px.transpose(), pe_over_pxprime.transpose();
    return -JT * (algebra_err / denominator);
  }

  template <typename Scalar>
  static Scalar computeSquaredSampsonError(
      const Eigen::Matrix<Scalar, 3, 3>& F,
      const Eigen::Matrix<Scalar, 2, 1>& x,
      const Eigen::Matrix<Scalar, 2, 1>& xprime) {
    Eigen::Matrix<Scalar, 3, 1> Ft_xprime =
        F.transpose() * xprime.homogeneous();
    Eigen::Matrix<Scalar, 3, 1> F_x = F * x.homogeneous();
    Eigen::Matrix<Scalar, 1, 2> pe_over_px, pe_over_pxprime;
    Scalar algebra_err = xprime.homogeneous().dot(F_x);
    Scalar denominator = Ft_xprime[0] * Ft_xprime[0] +
                         Ft_xprime[1] * Ft_xprime[1] + F_x[0] * F_x[0] +
                         F_x[1] * F_x[1];
    return algebra_err * algebra_err / denominator;
  }

  struct CeresOptimizer {
    const std::vector<size_t>* selected_indices;
    const std::vector<PointPair>* point_pairs;

    CeresOptimizer(
        const std::vector<size_t>* selected_indices_in,
        const std::vector<PointPair>* point_pairs_in)
        : selected_indices(selected_indices_in), point_pairs(point_pairs_in) {}

    // F should already be initialized with a good estimate.
    // NOTE: This function is not reenterable since we need use modify the
    //       member `tmp_param_blk_`
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
      bool operator()(const T* const F_local, T* residuals) const {
        auto F = fromManifold(param_blk_->MapToGlobal(F_local));
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res_eigen(
            residuals, optimizer_->residualSize());
        for (size_t i = 0; i < optimizer_->selected_indices->size(); ++i) {
          size_t idx = optimizer_->selected_indices->at(i);
          Eigen::Matrix<T, 2, 1> x =
              optimizer_->point_pairs->at(idx).first.cast<T>();
          Eigen::Matrix<T, 2, 1> xprime =
              optimizer_->point_pairs->at(idx).second.cast<T>();
          res_eigen.template segment<4>(4 * i) =
              computeSampsonError(F, x, xprime);
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
      bool operator()(const T* const F_local, T* residuals) const {
        auto F = fromManifold(param_blk_->MapToGlobal(F_local));
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res_eigen(
            residuals, optimizer_->residualSize());
        for (size_t i = 0; i < optimizer_->selected_indices->size(); ++i) {
          size_t idx = optimizer_->selected_indices->at(i);
          Eigen::Matrix<T, 2, 1> x =
              optimizer_->point_pairs->at(idx).first.cast<T>();
          Eigen::Matrix<T, 2, 1> xprime =
              optimizer_->point_pairs->at(idx).second.cast<T>();
          res_eigen[i] = computeSquaredSampsonError(F, x, xprime);
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
