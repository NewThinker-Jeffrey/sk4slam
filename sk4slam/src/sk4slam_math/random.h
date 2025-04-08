#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <memory>
#include <random>

#include "sk4slam_basic/likely.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/prng.h"

namespace sk4slam {

template <typename IntType>
class UniformIntDistribution : public std::uniform_int_distribution<IntType> {
 public:
  // NOTE: the range is [lower_bound, upper_bound], i.e. upper_bound itself
  //       is also a possible value.
  UniformIntDistribution(IntType lower_bound, IntType upper_bound)
      : std::uniform_int_distribution<IntType>(lower_bound, upper_bound) {}
  using std::uniform_int_distribution<IntType>::operator();
  IntType operator()() {
    return (*this)(getPRNG());
  }
  // the min(), max() methods are derived from std::uniform_int_distribution.
};

template <typename ScalarType = double>
class UniformRealDistribution
    : public std::uniform_real_distribution<ScalarType> {
 public:
  // NOTE: the range is [lower_bound, upper_bound), i.e. upper_bound is NOT
  //       a possible value.
  UniformRealDistribution(ScalarType lower_bound, ScalarType upper_bound)
      : std::uniform_real_distribution<ScalarType>(lower_bound, upper_bound) {}
  using std::uniform_real_distribution<ScalarType>::operator();
  ScalarType operator()() {
    return (*this)(getPRNG());
  }
  // the min(), max() methods are derived from std::uniform_real_distribution.
};

template <typename ScalarType = double>
class NormalDistribution : public std::normal_distribution<ScalarType> {
 public:
  NormalDistribution(
      ScalarType mean = ScalarType(0), ScalarType sigma = ScalarType(1))
      : std::normal_distribution<ScalarType>(mean, sigma) {}
  using std::normal_distribution<ScalarType>::operator();
  ScalarType operator()() {
    return (*this)(getPRNG());
  }
  // the mean(), sigma() methods are derived from std::normal_distribution.
};

template <typename ScalarType = double>
class MultivariateUniformDistribution {
 public:
  using Vector = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

  MultivariateUniformDistribution(
      const Vector& lower_bounds, const Vector& upper_bounds)
      : lower_bounds_(lower_bounds), upper_bounds_(upper_bounds) {
    ASSERT(lower_bounds_.size() == upper_bounds_.size());
    univar_dists_.reserve(lower_bounds_.size());
    for (size_t i = 0; i < lower_bounds_.size(); ++i) {
      univar_dists_.emplace_back(lower_bounds_(i), upper_bounds_(i));
    }
  }

  template <class UniformRandomNumberGenerator>
  Vector operator()(UniformRandomNumberGenerator& urng) {
    Vector result(lower_bounds_.size());
    for (size_t i = 0; i < lower_bounds_.size(); ++i) {
      result(i) = univar_dists_[i](urng);
    }
    return result;
  }

  Vector operator()() {
    return (*this)(getPRNG());
  }

  static MultivariateUniformDistribution standard(size_t dim) {
    Vector lower_bounds = -1 * Vector::Ones(dim);
    Vector upper_bounds = Vector::Ones(dim);
    return MultivariateUniformDistribution(lower_bounds, upper_bounds);
  }

  static MultivariateUniformDistribution standard01(size_t dim) {
    Vector lower_bounds = Vector::Zero(dim);
    Vector upper_bounds = Vector::Ones(dim);
    return MultivariateUniformDistribution(lower_bounds, upper_bounds);
  }

  const Vector& min() const {
    return lower_bounds_;
  }

  const Vector& max() const {
    return upper_bounds_;
  }

 protected:
  Vector lower_bounds_;
  Vector upper_bounds_;
  std::vector<UniformRealDistribution<ScalarType>> univar_dists_;
};

template <typename ScalarType = double>
class MultivariateNormalDistribution {
 public:
  using Matrix = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

  MultivariateNormalDistribution(const Vector& mean, const Matrix& cov)
      : mean_(mean), cov_(cov) {
    // Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(cov);
    // sqrt_cov_ = eigen_solver.eigenvectors() *
    // eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();

    sqrt_cov_ = cov.llt().matrixL();
  }

  explicit MultivariateNormalDistribution(const Matrix& cov)
      : MultivariateNormalDistribution(Vector::Zero(cov.rows()), cov) {}

  static MultivariateNormalDistribution standard(size_t dim) {
    return MultivariateNormalDistribution(
        Vector::Zero(dim), Matrix::Identity(dim, dim));
  }

  static MultivariateNormalDistribution random(size_t dim) {
    Vector mean = Vector::Random(dim);
    Matrix cov = Matrix::Random(dim, dim);
    cov = cov * cov.transpose();  // ensure cov is self-adjoint.
    return MultivariateNormalDistribution(mean, cov);
  }

  template <class UniformRandomNumberGenerator>
  Vector operator()(UniformRandomNumberGenerator& urng) {
    auto& dist = univarStandardDist();
    return mean_ + sqrt_cov_ * Vector{mean_.size()}.unaryExpr(
                                   [&](auto x) { return dist(urng); });
  }

  Vector operator()() {
    auto& dist = univarStandardDist();
    auto& urng = getPRNG();
    return mean_ + sqrt_cov_ * Vector{mean_.size()}.unaryExpr(
                                   [&](auto x) { return dist(urng); });
  }

  const Vector& mean() const {
    return mean_;
  }
  const Matrix& cov() const {
    return cov_;
  }
  const Matrix& sqrtCov() const {
    return sqrt_cov_;
  }

 protected:
  static NormalDistribution<ScalarType>& univarStandardDist() {
    static NormalDistribution<ScalarType> dist;
    return dist;
  }

 protected:
  Vector mean_;
  Matrix cov_;
  Matrix sqrt_cov_;
};

// Fisher-Yates shuffling.
//
// Note that the vector may not contain more values than UINT32_MAX. This
// restriction comes from the fact that the 32-bit version of the
// Mersenne Twister PRNG is significantly faster.
//
// @param elems            Vector of elements to shuffle.
// @param num_to_shuffle   Optional parameter, specifying the number of first
//                         N elements in the vector to shuffle.
template <typename T>
inline void shuffle(uint32_t num_to_shuffle, std::vector<T>* elems) {
  ASSERT(num_to_shuffle <= elems->size());
  const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);

  for (uint32_t i = 0; i < num_to_shuffle; ++i) {
    const auto j = UniformIntDistribution<uint32_t>(i, last_idx)();
    std::swap((*elems)[i], (*elems)[j]);
  }
}

}  // namespace sk4slam
