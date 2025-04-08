
#include "sk4slam_backends/gtsam/gtsam_helper.h"

#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/BatchFixedLagSmoother.h"
#include "gtsam/nonlinear/ISAM2.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/nonlinear/NonlinearEquality.h"
#include "gtsam/nonlinear/NonlinearISAM.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

// build gtsam with the cmake option '-DGTSAM_USE_SYSTEM_EIGEN=ON'.

// using OptimizableSO3d = SO3d::RightOptimizable;
using OptimizableSO3d = SO3d::XOptimizable;

using PerturbationSO3d = SO3d::RightPerturbation;
// using OptimizableR3d = gtsam::Point3;
using OptimizableR3d = Eigen::Vector3d;

class PointErrUnderSE3
    : public gtsam::NoiseModelFactorN<OptimizableSO3d, OptimizableR3d> {
 protected:
  using Base = gtsam::NoiseModelFactorN<OptimizableSO3d, OptimizableR3d>;

 public:
  template <typename... Keys>
  PointErrUnderSE3(
      const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, gtsam::Key R_key,
      gtsam::Key t_key, GtsamLossFunctionPtr loss_func = nullptr)
      : Base(gtsamLossFunctionToNoiseModel(3, loss_func), R_key, t_key),
        p1_(p1),
        p2_(p2) {}

  gtsam::Vector evaluateError(
      const OptimizableSO3d& rot, const OptimizableR3d& p,
      gtsam::OptionalMatrixType J_rot,
      gtsam::OptionalMatrixType J_p) const override {
    const SO3d& R = rot.value();
    const Eigen::Vector3d& t = p;

    Eigen::Vector3d error = R.matrix() * p1_ + t - p2_;
    if (J_rot != gtsam::OptionalMatrixNone) {
      Eigen::Matrix<double, 3, 3> J = -R.matrix() * skew3(p1_);
      *J_rot = J;
    }
    if (J_p != gtsam::OptionalMatrixNone) {
      Eigen::Matrix<double, 3, 3> J = Eigen::Matrix<double, 3, 3>::Identity();
      *J_p = J;
    }

    return error;
  }

 private:
  Eigen::Vector3d p1_, p2_;
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

TEST(TestGtsamHelper, Optimize) {
  TestContext& c = context_se3;

  // Construct the factor graph
  gtsam::NonlinearFactorGraph graph;
  GtsamLossFunctionPtr loss_func =
      gtsam::noiseModel::mEstimator::Huber::Create(1.0);

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    graph.add(PointErrUnderSE3(
        c.samples.col(i), c.rotated_samples.col(i), gtsam::Symbol('R', 0),
        gtsam::Symbol('t', 0), loss_func));
  }

  // // solve with gtsam
  gtsam::Values initial_estimate;
  initial_estimate.insert(
      gtsam::Symbol('R', 0), OptimizableSO3d::Create<PerturbationSO3d>());
  initial_estimate.insert(
      gtsam::Symbol('t', 0), OptimizableR3d(Eigen::Vector3d::Zero()));

  gtsam::LevenbergMarquardtParams opt_params;
  opt_params.maxIterations = 100;
  opt_params.verbosity = gtsam::NonlinearOptimizerParams::TERMINATION;
  opt_params.orderingType = gtsam::Ordering::COLAMD;
  opt_params.linearSolverType =
      gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;
  opt_params.iterationHook = [](size_t iter, double oldError, double newError) {
    LOGI("iter: %lu, oldError: %f, newError: %f", iter, oldError, newError);
  };

  gtsam::LevenbergMarquardtOptimizer optimizer(
      graph, initial_estimate, opt_params);
  gtsam::Values result = optimizer.optimize();
  // result.print("Result:\n");

  ASSERT_TRUE(result.exists(gtsam::Symbol('R', 0)));
  ASSERT_TRUE(result.exists(gtsam::Symbol('t', 0)));
  const SO3d& R =
      result.exists<OptimizableSO3d>(gtsam::Symbol('R', 0))->value();
  const Eigen::Vector3d& t =
      *result.exists<OptimizableR3d>(gtsam::Symbol('t', 0));

  Eigen::Matrix3d diffR = R.matrix() - c.R;
  Eigen::Vector3d diffT = t - c.t;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R.matrix()).c_str());
  LOGI("Real translation: %s", toStr(c.t.transpose()).c_str());
  LOGI("Estimated translation: %s", toStr(t.transpose()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());
  LOGI("Translation error: %f", diffT.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
  ASSERT_LE(diffT.squaredNorm(), 1e-1);
}

TEST(TestGtsamHelper, OptimizeWithPrior) {
  TestContext& c = context_se3;

  // Construct the factor graph
  gtsam::NonlinearFactorGraph graph;
  GtsamLossFunctionPtr loss_func =
      gtsam::noiseModel::mEstimator::Huber::Create(1.0);

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    graph.add(PointErrUnderSE3(
        c.samples.col(i), c.rotated_samples.col(i), gtsam::Symbol('R', 0),
        gtsam::Symbol('t', 0), loss_func));
  }
  graph.add(gtsam::PriorFactor<OptimizableSO3d>(
      gtsam::Symbol('R', 0), OptimizableSO3d::Create<PerturbationSO3d>(c.R)));
  graph.add(gtsam::PriorFactor<OptimizableR3d>(
      gtsam::Symbol('t', 0), OptimizableR3d(c.t)));

  // // solve with gtsam
  gtsam::Values initial_estimate;
  initial_estimate.insert(
      gtsam::Symbol('R', 0), OptimizableSO3d::Create<PerturbationSO3d>(c.R));
  initial_estimate.insert(
      gtsam::Symbol('t', 0), OptimizableR3d(Eigen::Vector3d(c.t)));

  gtsam::LevenbergMarquardtParams opt_params;
  opt_params.maxIterations = 100;
  opt_params.verbosity = gtsam::NonlinearOptimizerParams::TERMINATION;
  opt_params.orderingType = gtsam::Ordering::COLAMD;
  opt_params.linearSolverType =
      gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;
  opt_params.iterationHook = [](size_t iter, double oldError, double newError) {
    LOGI("iter: %lu, oldError: %f, newError: %f", iter, oldError, newError);
  };

  gtsam::LevenbergMarquardtOptimizer optimizer(
      graph, initial_estimate, opt_params);
  gtsam::Values result = optimizer.optimize();
  // result.print("Result:\n");

  ASSERT_TRUE(result.exists(gtsam::Symbol('R', 0)));
  ASSERT_TRUE(result.exists(gtsam::Symbol('t', 0)));
  const SO3d& R =
      result.exists<OptimizableSO3d>(gtsam::Symbol('R', 0))->value();
  const Eigen::Vector3d& t =
      *result.exists<OptimizableR3d>(gtsam::Symbol('t', 0));

  Eigen::Matrix3d diffR = R.matrix() - c.R;
  Eigen::Vector3d diffT = t - c.t;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R.matrix()).c_str());
  LOGI("Real translation: %s", toStr(c.t.transpose()).c_str());
  LOGI("Estimated translation: %s", toStr(t.transpose()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());
  LOGI("Translation error: %f", diffT.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
  ASSERT_LE(diffT.squaredNorm(), 1e-1);
}

TEST(TestGtsamHelper, OptimizeWithEquality) {
  TestContext& c = context_se3;

  // Construct the factor graph
  gtsam::NonlinearFactorGraph graph;
  GtsamLossFunctionPtr loss_func =
      gtsam::noiseModel::mEstimator::Huber::Create(1.0);

  for (size_t i = 0; i < c.kSampleSize; ++i) {
    graph.add(PointErrUnderSE3(
        c.samples.col(i), c.rotated_samples.col(i), gtsam::Symbol('R', 0),
        gtsam::Symbol('t', 0), loss_func));
  }

  graph.add(gtsam::NonlinearEquality<OptimizableSO3d>(
      gtsam::Symbol('R', 0), OptimizableSO3d::Create<PerturbationSO3d>(c.R)));
  graph.add(gtsam::NonlinearEquality<OptimizableR3d>(
      gtsam::Symbol('t', 0), OptimizableR3d(c.t)));

  // // solve with gtsam
  gtsam::Values initial_estimate;
  initial_estimate.insert(
      gtsam::Symbol('R', 0), OptimizableSO3d::Create<PerturbationSO3d>(c.R));
  initial_estimate.insert(
      gtsam::Symbol('t', 0), OptimizableR3d(Eigen::Vector3d(c.t)));
  // initial_estimate.insert(gtsam::Symbol('R', 0),
  // OptimizableSO3d::Create<PerturbationSO3d>()); initial_estimate.insert(
  //     gtsam::Symbol('t', 0), OptimizableR3d(Eigen::Vector3d::Zero()));

  gtsam::LevenbergMarquardtParams opt_params;
  opt_params.maxIterations = 100;
  opt_params.verbosity = gtsam::NonlinearOptimizerParams::TERMINATION;
  opt_params.orderingType = gtsam::Ordering::COLAMD;
  opt_params.linearSolverType =
      gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;
  opt_params.iterationHook = [](size_t iter, double oldError, double newError) {
    LOGI("iter: %lu, oldError: %f, newError: %f", iter, oldError, newError);
  };

  gtsam::LevenbergMarquardtOptimizer optimizer(
      graph, initial_estimate, opt_params);
  gtsam::Values result = optimizer.optimize();
  // result.print("Result:\n");

  ASSERT_TRUE(result.exists(gtsam::Symbol('R', 0)));
  ASSERT_TRUE(result.exists(gtsam::Symbol('t', 0)));
  const SO3d& R =
      result.exists<OptimizableSO3d>(gtsam::Symbol('R', 0))->value();
  const Eigen::Vector3d& t =
      *result.exists<OptimizableR3d>(gtsam::Symbol('t', 0));

  Eigen::Matrix3d diffR = R.matrix() - c.R;
  Eigen::Vector3d diffT = t - c.t;
  LOGI("Real Rotation:\n %s", toStr(c.R).c_str());
  LOGI("Estimated Rotation:\n %s", toStr(R.matrix()).c_str());
  LOGI("Real translation: %s", toStr(c.t.transpose()).c_str());
  LOGI("Estimated translation: %s", toStr(t.transpose()).c_str());
  LOGI("Rotation error: %f", diffR.squaredNorm());
  LOGI("Translation error: %f", diffT.squaredNorm());

  ASSERT_LE(diffR.squaredNorm(), 1e-1);
  ASSERT_LE(diffT.squaredNorm(), 1e-1);
}

TEST(TestGtsamHelper, ConstructISAM2) {
  gtsam::ISAM2Params params;
  gtsam::ISAM2 isam2(params);
  isam2.printStats();
}

TEST(TestGtsamHelper, ConstructFixedLagSmoother) {
  gtsam::BatchFixedLagSmoother smoother;
  smoother.print();
}

SK4SLAM_UNITTEST_ENTRYPOINT
