#pragma once

#include <Eigen/Core>

#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "sk4slam_math/optimizable_manifold.h"

namespace sk4slam {

/// Define aliases for gtsam loss functions
using GtsamLossFunction = gtsam::noiseModel::mEstimator::Base;
using GtsamLossFunctionPtr = GtsamLossFunction::shared_ptr;

using GtsamFair = gtsam::noiseModel::mEstimator::Fair;
using GtsamFairPtr = GtsamFair::shared_ptr;

using GtsamHuber = gtsam::noiseModel::mEstimator::Huber;
using GtsamHuberPtr = GtsamHuber::shared_ptr;

using GtsamCauchy = gtsam::noiseModel::mEstimator::Cauchy;
using GtsamCauchyPtr = GtsamCauchy::shared_ptr;

using GtsamTukey = gtsam::noiseModel::mEstimator::Tukey;
using GtsamTukeyPtr = GtsamTukey::shared_ptr;

using GtsamWelsch = gtsam::noiseModel::mEstimator::Welsch;
using GtsamWelschPtr = GtsamWelsch::shared_ptr;

using GtsamGemanMcClure = gtsam::noiseModel::mEstimator::GemanMcClure;
using GtsamGemanMcClurePtr = GtsamGemanMcClure::shared_ptr;

using GtsamDCS = gtsam::noiseModel::mEstimator::DCS;
using GtsamDCSPtr = GtsamDCS::shared_ptr;

using GtsamL2WithDeadZone = gtsam::noiseModel::mEstimator::L2WithDeadZone;
using GtsamL2WithDeadZonePtr = GtsamL2WithDeadZone::shared_ptr;

using GtsamAsymmetricTukey = gtsam::noiseModel::mEstimator::AsymmetricTukey;
using GtsamAsymmetricTukeyPtr = GtsamAsymmetricTukey::shared_ptr;

using GtsamAsymmetricCauchy = gtsam::noiseModel::mEstimator::AsymmetricCauchy;
using GtsamAsymmetricCauchyPtr = GtsamAsymmetricCauchy::shared_ptr;

inline gtsam::SharedNoiseModel gtsamLossFunctionToNoiseModel(
    int dim, const GtsamLossFunctionPtr& loss_function) {
  if (loss_function == nullptr) {
    return nullptr;  // same with Unit ?
    // return gtsam::noiseModel::Unit::Create(dim);
  } else {
    return gtsam::noiseModel::Robust::Create(
        loss_function, gtsam::noiseModel::Unit::Create(dim));
  }
}

}  // namespace sk4slam

namespace gtsam {

template <typename Manifold, typename Retraction, bool share_retraction>
struct traits<
    sk4slam::OptimizableManifold<Manifold, Retraction, share_retraction>> {
  using OptimizableManifold =
      sk4slam::OptimizableManifold<Manifold, Retraction, share_retraction>;

  // Dimension of the manifold
  enum { dimension = OptimizableManifold::kDof };

  // Typedefs required by all manifold types.
  typedef OptimizableManifold ManifoldType;
  typedef manifold_tag structure_category;
  typedef Eigen::Matrix<double, dimension, 1> TangentVector;

  // Return run-time dimensionality
  static size_t GetDimension(const OptimizableManifold& point) {
    return point.dof();
  }

  // Local coordinates
  static TangentVector Local(
      const OptimizableManifold& origin, const OptimizableManifold& other) {
    // throw std::runtime_error(
    //     "gtsam::traits<sk4slam::OptimizableManifold>::localCoordinates() "
    //     "is not implemented!");
    return other - origin;
  }

  // Retraction back to manifold
  static OptimizableManifold Retract(
      const OptimizableManifold& origin, const TangentVector& v) {
    return origin + v;
  }

  static bool Equals(
      const OptimizableManifold& lhs, const OptimizableManifold& rhs,
      const double eps) {
    using Value = typename OptimizableManifold::Value;
    return sk4slam::manifold_traits<Value>::isApprox(
        lhs.value(), rhs.value(), eps);
  }

  static void Print(const OptimizableManifold& point, const std::string& str) {
    using Value = typename OptimizableManifold::Value;
    const double* data = reinterpret_cast<const double*>(&point.value());
    Eigen::Map<const Eigen::Matrix<double, 1, sizeof(Value) / sizeof(double)>>
        m(data);
    std::cout << str << ": " << m << std::endl;
  }
};

inline Matrix* OptionalMatrixNone = nullptr;

}  // namespace gtsam
