
#define SK4SLAM_TEST_CERES_CONSTANT_BLOCK
#include "sk4slam_backends/ceres/ceres_helper.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

#define TEST_MANUAL_JACOBIAN_SE3
#define TEST_AUTO_JACOBIAN_SE3
#define TEST_MANUAL_JACOBIAN_SO3
#define TEST_AUTO_JACOBIAN_SO3

static constexpr int max_iter = 50;
// static constexpr int max_iter = 10;

using SO3Block = CeresManifoldBlock<SO3d::RightOptimizable>;

class PointErrUnderSE3 : public ceres::SizedCostFunction<3, 3, 3> {
 public:
  PointErrUnderSE3(
      const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
      const std::vector<const CeresManifoldBlockInterface*>& manifold_blocks)
      : p1_(p1), p2_(p2), manifold_blocks_(manifold_blocks) {}

  bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const override {
    const SO3Block* R_blk = dynamic_cast<const SO3Block*>(manifold_blocks_[0]);

    ASSERT(R_blk);

    SO3d R = R_blk->MapToGlobal(parameters[0]);
    Eigen::Vector3d t(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d delta = R.matrix() * p1_ + t - p2_;
    Eigen::Map<Eigen::Vector3d>(residuals) << delta;
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        Eigen::Matrix<double, 3, 3> J = -R.matrix() * skew3(p1_);

        // We transpose J since Eigen stores matrix data column-major while
        // Ceres stores it row-major.
        Eigen::Map<Eigen::Matrix<double, 3, 3>>(jacobians[0]) << J.transpose();
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 3, 3>>(jacobians[1])
            << Eigen::Matrix3d::Identity();
      }
    }
    return true;
  }

 private:
  Eigen::Vector3d p1_, p2_;
  std::vector<const CeresManifoldBlockInterface*> manifold_blocks_;
};

// #define SE3_FUNCTOR_R_BEFORE_T

class PointErrFunctorUnderSE3 {
 public:
  PointErrFunctorUnderSE3(
      const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
      const std::vector<const CeresManifoldBlockInterface*>& manifold_blocks)
      : p1_(p1), p2_(p2), manifold_blocks_(manifold_blocks) {}

  template <typename T>
#ifdef SE3_FUNCTOR_R_BEFORE_T
  bool operator()(const T* const so3, const T* const t, T* residual) const {
#else
  bool operator()(const T* const t, const T* const so3, T* residual) const {
#endif
    const SO3Block* R_blk = dynamic_cast<const SO3Block*>(manifold_blocks_[0]);
    SO3<T> R = R_blk->MapToGlobal(so3);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_eigen(t);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_eigen(residual);
    Eigen::Matrix<T, 3, 1> p1 = p1_.cast<T>();
    Eigen::Matrix<T, 3, 1> p2 = p2_.cast<T>();
    residual_eigen = R.matrix() * p1 + t_eigen - p2;
    return true;
  }

 private:
  Eigen::Vector3d p1_, p2_;
  std::vector<const CeresManifoldBlockInterface*> manifold_blocks_;
};  // NOLINT

class PointErrFunctorUnderSO3 {
 public:
  PointErrFunctorUnderSO3(
      const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
      const std::vector<const CeresManifoldBlockInterface*>& manifold_blocks,
      bool count_computation = false)
      : p1_(p1),
        p2_(p2),
        manifold_blocks_(manifold_blocks),
        count_computation_(count_computation) {}

  template <typename T>
  bool operator()(const T* const so3, const T* const SO3R, T* residual) const {
    const SO3Block* R_blk = dynamic_cast<const SO3Block*>(manifold_blocks_[0]);
    if (count_computation_) {
      if constexpr (std::is_same<T, double>::value) {
        ++n_pure_cost_computation_;

#ifdef SK4SLAM_TEST_CERES_CONSTANT_BLOCK
        // The data pointer of a parameter-block that is marked Constant
        // used in the cost-function is always pointing to the user-provided
        // data (the user_state).
        ASSERT(SO3R == R_blk->GlobalData());
        LOGI("For scalar, SO3R == R_blk->GlobalData()");
#endif
      } else {
        ++n_jacobian_computation_;
#ifdef SK4SLAM_TEST_CERES_CONSTANT_BLOCK
        for (size_t i = 0; i < R_blk->GlobalSize(); i++) {
          // The data of a parameter-block that is marked Constant
          // used in the cost-function is always synced with the
          // user-provided data (the user_state).
          ASSERT(SO3R[i].a == R_blk->GlobalData()[i]);
        }
        LOGI("For Jet, SO3R[*] == R_blk->GlobalData()[*]");
#endif
        LOGI("Jacobian computation count: %d", n_jacobian_computation_);
      }
      // LOGI("Computation count: (pure_cost, jacobian) = (%d, %d), total = %d",
      //      n_pure_cost_computation_, n_jacobian_computation_,
      //      n_pure_cost_computation_ + n_jacobian_computation_);
    }
    SO3<T> R = R_blk->MapToGlobal(so3);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_eigen(residual);
    Eigen::Matrix<T, 3, 1> p1 = p1_.cast<T>();
    Eigen::Matrix<T, 3, 1> p2 = p2_.cast<T>();
    residual_eigen = R.matrix() * p1 - p2;
    return true;
  }

 private:
  Eigen::Vector3d p1_, p2_;
  std::vector<const CeresManifoldBlockInterface*> manifold_blocks_;

  const bool count_computation_;
  mutable size_t n_pure_cost_computation_ = 0;
  mutable size_t n_jacobian_computation_ = 0;
};

class PointErrUnderSO3 : public ceres::SizedCostFunction<3, 3> {
 public:
  PointErrUnderSO3(
      const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
      const std::vector<const CeresManifoldBlockInterface*>& manifold_blocks)
      : p1_(p1), p2_(p2), manifold_blocks_(manifold_blocks) {}

  bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const override {
    const SO3Block* R_blk = dynamic_cast<const SO3Block*>(manifold_blocks_[0]);

    ASSERT(R_blk);

    SO3d R = R_blk->MapToGlobal(parameters[0]);
    Eigen::Vector3d delta = R.matrix() * p1_ - p2_;
    Eigen::Map<Eigen::Vector3d>(residuals) << delta;
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        Eigen::Matrix<double, 3, 3> J = -R.matrix() * skew3(p1_);

        // We transpose J since Eigen stores matrix data column-major while
        // Ceres stores it row-major.
        Eigen::Map<Eigen::Matrix<double, 3, 3>>(jacobians[0]) << J.transpose();
      }
    }
    return true;
  }

 private:
  Eigen::Vector3d p1_, p2_;
  std::vector<const CeresManifoldBlockInterface*> manifold_blocks_;
};

struct TestContext {
  Eigen::MatrixXd samples;
  Eigen::MatrixXd rotated_samples;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  // Generate 100 samples
  static constexpr int kSampleSize = 100;

  explicit TestContext(bool translation = true) {
    auto dist = MultivariateNormalDistribution<double>::standard(3);
    samples = Eigen::MatrixXd(3, kSampleSize);
    for (int i = 0; i < kSampleSize; ++i) {
      samples.col(i) = dist();
    }

    Eigen::Vector3d rot_vec(0.1, 0.2, 0.3);
    R = SO3d::expM(rot_vec);
    if (translation) {
      t = dist();
    } else {
      t.setZero();
    }

    rotated_samples = R * samples;

    // Add translation and noise to rotated_samples
    double sigma = 0.2;
    for (int i = 0; i < kSampleSize; ++i) {
      rotated_samples.col(i) += t + sigma * dist();
    }
  }
};

TestContext context_se3;
TestContext context_so3(false);

#ifdef TEST_MANUAL_JACOBIAN_SE3
TEST(TestCeresHelper, ManualJacobianSE3) {
  Logging::setVerbose("DEBUG");
  LOGI("=================TestCeresHelper, ManualJacobianSE3===============");
  TestContext& c = context_se3;

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = max_iter;
  CeresProblem problem;

  SO3Block R_blk(&problem, SO3d::Identity());
  Eigen::Vector3d t_blk(0, 0, 0);

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    problem.AddResidualBlock(
        new PointErrUnderSE3(
            c.samples.col(i), c.rotated_samples.col(i), {&R_blk}),
        nullptr, R_blk.LocalData(), t_blk.data());
  }

  ceres::Solver::Summary summary;
  ceres::TerminationType termination_type =
      CeresSolve(solver_options, &problem, &summary, true);
  LOGI("termination_type: %d", termination_type);

  Eigen::Matrix3d diffR = R_blk.GetGlobal().matrix() - c.R;
  Eigen::Vector3d diffT = t_blk - c.t;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R_blk.GetGlobal().matrix()).c_str());
  LOGI("Real translation: %s", toStr(c.t.transpose()).c_str());
  LOGI("Estimated translation: %s", toStr(t_blk.transpose()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());
  LOGI("Translation error: %f", diffT.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
  ASSERT_LE(diffT.squaredNorm(), 1e-1);
}
#endif

#ifdef TEST_AUTO_JACOBIAN_SE3
TEST(TestCeresHelper, AutoJacobianSE3) {
  Logging::setVerbose("DEBUG");
  LOGI("=================TestCeresHelper, AutoJacobianSE3===============");
  TestContext& c = context_se3;

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = max_iter;
  CeresProblem problem;

  SO3Block R_blk(&problem, SO3d::Identity());
  Eigen::Vector3d t_blk(0, 0, 0);

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    auto functor = new PointErrFunctorUnderSE3(
        c.samples.col(i), c.rotated_samples.col(i), {&R_blk});
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            PointErrFunctorUnderSE3,
            3,  // residual size
            3, 3>(functor),
        nullptr,
#ifdef SE3_FUNCTOR_R_BEFORE_T
        R_blk.LocalData(), t_blk.data());
#else
        t_blk.data(), R_blk.LocalData());
#endif
  }

  ceres::Solver::Summary summary;
  ceres::TerminationType termination_type =
      CeresSolve(solver_options, &problem, &summary, true);
  LOGI("termination_type: %d", termination_type);

  Eigen::Matrix3d diffR = R_blk.GetGlobal().matrix() - c.R;
  Eigen::Vector3d diffT = t_blk - c.t;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R_blk.GetGlobal().matrix()).c_str());
  LOGI("Real translation: %s", toStr(c.t.transpose()).c_str());
  LOGI("Estimated translation: %s", toStr(t_blk.transpose()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());
  LOGI("Translation error: %f", diffT.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
  ASSERT_LE(diffT.squaredNorm(), 1e-1);
}
#endif

#ifdef TEST_MANUAL_JACOBIAN_SO3
TEST(TestCeresHelper, ManualJacobianSO3) {
  Logging::setVerbose("DEBUG");
  LOGI("=================TestCeresHelper, ManualJacobianSO3===============");
  TestContext& c = context_so3;

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = max_iter;
  CeresProblem problem;

  SO3Block R_blk(&problem, SO3d::Identity());
  Eigen::Vector3d t_blk(0, 0, 0);

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    problem.AddResidualBlock(
        new PointErrUnderSO3(
            c.samples.col(i), c.rotated_samples.col(i), {&R_blk}),
        nullptr, R_blk.LocalData());
  }

  ceres::Solver::Summary summary;
  ceres::TerminationType termination_type =
      CeresSolve(solver_options, &problem, &summary, true);
  LOGI("termination_type: %d", termination_type);

  Eigen::Matrix3d diffR = R_blk.GetGlobal().matrix() - c.R;
  Eigen::Vector3d diffT = t_blk - c.t;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R_blk.GetGlobal().matrix()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
}
#endif

#ifdef TEST_AUTO_JACOBIAN_SO3
TEST(TestCeresHelper, AutoJacobianSO3) {
  Logging::setVerbose("DEBUG");
  LOGI("=================TestCeresHelper, AutoJacobianSO3===============");
  TestContext& c = context_so3;

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = max_iter;
  CeresProblem problem;

  SO3Block R_blk(&problem, SO3d::Identity());

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    auto functor = new PointErrFunctorUnderSO3(
        c.samples.col(i), c.rotated_samples.col(i), {&R_blk}, i == 0);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            PointErrFunctorUnderSO3,
            3,  // residual size
#ifdef SK4SLAM_TEST_CERES_CONSTANT_BLOCK
            3, 9>(functor),
        nullptr, R_blk.LocalData(), R_blk.GlobalData());
#else
            3>(functor),  // NOLINT
        nullptr, R_blk.LocalData());
#endif
  }

  ceres::Solver::Summary summary;
  ceres::TerminationType termination_type =
      CeresSolve(solver_options, &problem, &summary, true);
  LOGI("termination_type: %d", termination_type);

  Eigen::Matrix3d diffR = R_blk.GetGlobal().matrix() - c.R;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R_blk.GetGlobal().matrix()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
}
#endif

SK4SLAM_UNITTEST_ENTRYPOINT
