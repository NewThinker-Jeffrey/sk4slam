#pragma once

#include <Eigen/Core>
#include <memory>
#include <numeric>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/constants.h"

namespace sk4slam {

/// @brief Compute the weighted errors of the input Lie group elements
///        with respect to the @p expected.
///        This function is intended to be used internally only.
/// @tparam _left_perturbation
///         If true, the error is defined to be a left perturbation (computed as
///         "element * @p expected .inverse()");
///         If false, the error is defined to be a right perturbation (computed
///         as " @p expected .inverse() * element").
/// @tparam WeigthType
///         @c WeigthType can be a general weight matrix of size (N, N) (N is
///         the dimension of the Lie group), a vector of size N (in which case
///         it is interpreted as a diagonal weight matrix with the diagonal
///         elements equal to the vector elements), or a scalar (in which case
///         it is interpreted as a diagonal weight matrix with all diagonal
///         elements equal to the scalar).
///         @c WeigthType can also be a pointer to a weight
///         matrix/vector/scalar.
/// @tparam LieGroup
///         The Lie group type.
/// @param selected_indices
///         The indices of the elements to compute the errors for.
/// @param elements
///         The elements to compute the errors for.
/// @param weights
///         The weights to use for computing the errors. If nullptr, uniform
///         weights are used; otherwise, the size of @p weights must match
///         the size of @p selected_indices, or be 1 (in which case the same
///         weight matrix is applied to all elements).
/// @param expected
///         The expected value.
/// @param residuals
///         The output errors. If nullptr, the residuals are not computed.
/// @param jacobians
///         The jacobians of the errors. If nullptr, the jacobians are not
///         computed.
template <bool _left_perturbation, typename WeigthType, typename LieGroup>
void _computeLieGroupErrors(
    const std::vector<size_t>& selected_indices,
    const std::vector<LieGroup>& elements,
    const std::vector<WeigthType>* weights,  // nullptr for uniform weights
    const LieGroup& expected,
    Eigen::Matrix<typename LieGroup::Scalar, Eigen::Dynamic, 1>* residuals,
    Eigen::Matrix<typename LieGroup::Scalar, Eigen::Dynamic, Eigen::Dynamic>*
        jacobians) {
  using Scalar = typename LieGroup::Scalar;
  int rows = selected_indices.size() * LieGroup::kDim;
  if (!residuals && !jacobians) {
    return;
  }
  if (residuals) {
    residuals->resize(rows);
  }
  if (jacobians) {
    jacobians->resize(rows, LieGroup::kDim);
  }
  Eigen::Matrix<Scalar, LieGroup::kDim, 1> residual;
  Eigen::Matrix<Scalar, LieGroup::kDim, LieGroup::kDim> jacobian;
  LieGroup expected_inv = expected.inverse();
  for (int i = 0; i < selected_indices.size(); ++i) {
    size_t idx = selected_indices[i];
    if constexpr (_left_perturbation) {
      // exp(residual) * center = elements[idx]
      residual = LieGroup::Log(elements[idx] * expected_inv);
      if (jacobians) {
        jacobian = -LieGroup::Jr(residual);
      }
    } else {
      // center * exp(residual) = elements[idx]
      residual = LieGroup::Log(expected_inv * elements[idx]);
      if (jacobians) {
        jacobian = -LieGroup::Jl(residual);
      }
    }

    if (weights) {
      size_t weight_idx = (weights->size() == 1 ? 0 : i);
      if constexpr (std::is_same_v<std::remove_cv_t<WeigthType>, Scalar>) {
        // Weights are scalars.
        if (residuals) {
          residual *= (*weights)[weight_idx];
        }
        if (jacobians) {
          jacobian *= (*weights)[weight_idx];
        }
      } else {
        // Weights are matrices.
        using Weight = std::remove_cv_t<
            std::remove_pointer_t<std::remove_cv_t<WeigthType>>>;
        const Weight* weight_ptr = nullptr;
        if constexpr (std::is_pointer_v<std::remove_cv_t<WeigthType>>) {
          // WeigthType is pointer to matrix.
          weight_ptr = (*weights)[weight_idx];
        } else {
          // WeigthType is matrix.
          weight_ptr = &(*weights)[weight_idx];
        }
        const auto& w_mat = *weight_ptr;

        if constexpr (
            Weight::RowsAtCompileTime == 1 && Weight::ColsAtCompileTime == 1) {
          // Weights are scalars.
          const Scalar& w = w_mat[0];
          if (residuals) {
            residual *= w;
          }
          if (jacobians) {
            jacobian *= w;
          }
        } else if constexpr (Weight::ColsAtCompileTime == 1) {
          // Weights are diagonal.
          ASSERT(w_mat.rows() == LieGroup::kDim);
          auto diag = w_mat.asDiagonal();
          if (residuals) {
            residual = diag * residual;
          }
          if (jacobians) {
            jacobian = diag * jacobian;
          }
        } else {
          if (w_mat.rows() == 1 && w_mat.cols() == 1) {
            // Weights are scalars.
            const Scalar& w = w_mat(0, 0);
            if (residuals) {
              residual *= w;
            }
            if (jacobians) {
              jacobian *= w;
            }
          } else if (w_mat.cols() == 1) {
            // Weights are diagonal.
            ASSERT(w_mat.rows() == LieGroup::kDim);
            auto diag = w_mat.asDiagonal();
            if (residuals) {
              residual = diag * residual;
            }
            if (jacobians) {
              jacobian = diag * jacobian;
            }
          } else {
            // Weights are general (matrices).
            ASSERT(
                w_mat.rows() == LieGroup::kDim &&
                w_mat.cols() == LieGroup::kDim);
            if (residuals) {
              residual = w_mat * residual;
            }
            if (jacobians) {
              jacobian = w_mat * jacobian;
            }
          }
        }
      }
    }

    if (residuals) {
      residuals->template block<LieGroup::kDim, 1>(i * LieGroup::kDim, 0) =
          residual;
    }
    if (jacobians) {
      jacobians->template block<LieGroup::kDim, LieGroup::kDim>(
          i * LieGroup::kDim, 0) = jacobian;
    }
  }
}

/// @brief Compute the mean of a set of Lie group elements.
///        This function is intended to be used internally only.
/// @tparam _left_perturbation
/// @tparam WeigthType
/// @tparam LieGroup
///         See the same template parameters in @ref _computeLieGroupErrors().
/// @param selected_indices
///         Indices of the elements to be used in the mean computation.
/// @param elements
///         The set of Lie group elements.
/// @param initial_estimate
///         Initial estimate of the mean.
/// @param output_mean
///         The computed mean.
/// @param weights
///         See the same parameter in @ref _computeLieGroupErrors().
/// @param max_iterations
///         The maximum number of iterations to perform. If negative, the
///         iteration won't stop until convergence.
/// @param tolerance
///         The tolerance for convergence. If the squared norm of the update
///         delta is smaller than this value, the iteration will stop.
template <bool _left_perturbation, typename WeigthType, typename LieGroup>
bool _computeLieGroupMean(
    const std::vector<size_t>& selected_indices,
    const std::vector<LieGroup>& elements, const LieGroup& initial_estimate,
    LieGroup* output_mean, const std::vector<WeigthType>* weights = nullptr,
    int max_iterations = -1,
    typename LieGroup::Scalar tolerance =
        liegroup::Constants<typename LieGroup::Scalar>::kEps) {
  if (selected_indices.empty()) {
    return false;
  }

  using Scalar = typename LieGroup::Scalar;
  Scalar tolerance2 = tolerance * tolerance;
  int rows = selected_indices.size() * LieGroup::kDim;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(rows);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(rows, LieGroup::kDim);

  LieGroup center = initial_estimate;
  int max_iter = max_iterations;
  if (max_iter < 0) {
    // set max_iter to the max value of int.
    max_iter = std::numeric_limits<int>::max();
  }

  int iter;
  for (iter = 0; iter < max_iter; ++iter) {
    _computeLieGroupErrors<_left_perturbation, WeigthType, LieGroup>(
        selected_indices, elements, weights, center, &residuals, &J);
    // LOGA(
    //     "iter: %d, rmse = %f", iter,
    //     sqrt(residuals.squaredNorm() / selected_indices.size()));

    // Perform one step of Gauss-Newton.
    // Solve delta from
    //    J * delta = -residuals
    // by QR decomposition.

    Eigen::Matrix<Scalar, LieGroup::kDim, 1> delta;
    // delta = J.colPivHouseholderQr().solve(-residuals);
    auto qr = J.colPivHouseholderQr();
    if (qr.rank() < LieGroup::kDim) {
      // Degenerate case.
      return false;
    }
    delta = qr.solve(-residuals);

    if (delta.squaredNorm() < tolerance2) {
      break;
    }

    if constexpr (_left_perturbation) {
      center = LieGroup::Exp(delta) * center;
    } else {
      center = center * LieGroup::Exp(delta);
    }
  }

  if (max_iter > 0) {
    LOGA(
        "computeLieGroupMean(): iter = %d (max_iter = %d)", iter,
        max_iterations);
  }

  if (output_mean) {
    *output_mean = center;
  }

  return true;
}

/// @brief  This function is a convenient overload of the above
///         @ref _computeLieGroupMean() when all the elements are selected.
template <bool _left_perturbation, typename WeigthType, typename LieGroup>
bool _computeLieGroupMean(
    const std::vector<LieGroup>& elements, const LieGroup& initial_estimate,
    LieGroup* output_mean, const std::vector<WeigthType>* weights = nullptr,
    int max_iterations = -1,
    typename LieGroup::Scalar tolerance =
        liegroup::Constants<typename LieGroup::Scalar>::kEps) {
  std::vector<size_t> selected_indices;
  selected_indices.resize(elements.size());
  std::iota(selected_indices.begin(), selected_indices.end(), 0);
  return _computeLieGroupMean<_left_perturbation>(
      selected_indices, elements, initial_estimate, output_mean, weights,
      max_iterations, tolerance);
}

/// @brief  Compute the mean of a set of Lie group elements.
/// @tparam _left_perturbation
///         If true, the error is defined to be a left perturbation (computed as
///         "element * @p expected .inverse()");
///         If false, the error is defined to be a right perturbation (computed
///         as " @p expected .inverse() * element").
/// @tparam LieGroup
///         The Lie group type.
/// @tparam WeightMatrix
///           The type of the weight matrices.
///           It can be a matrix of size (N, N), or of (N, 1), or of (1, 1)
///           (N is the dimension of the Lie group).
///           When it's of (1, 1), it is interpreted as a diagonal weight matrix
///           with the diagonal elements equal to the vector elements;
///           When it's of (1, 1), it is interpreted as a diagonal weight matrix
///           with all diagonal elements equal to the scalar.
/// @param elements
///           The elements to compute the mean of.
/// @param matrix_weights
///           the weight matrices for the elements.
///           It's size must be the same as the size of `elements`, or be 1 if
///           all the elements have the same weight matrix.
/// @param mean
///           The computed mean.
/// @param get_initial_estimate
///           A function that returns an initial estimate of the mean. If
///           `nullptr`, the first element of `elements` is used as the initial
///           estimate.
/// @param max_iterations
///           The maximum number of iterations to perform. If negative, the
///           iteration won't stop until convergence.
/// @param tolerance
///           The tolerance for convergence. If the squared norm of the update
///           delta is smaller than this value, the iteration will stop.
template <
    bool _left_perturbation, typename LieGroup, typename WeightMatrix,
    ENABLE_IF(!(std::is_same_v<
                std::remove_cv_t<WeightMatrix>, typename LieGroup::Scalar>))>
bool computeLieGroupMean(
    const std::vector<LieGroup>& elements,
    const std::vector<WeightMatrix>& matrix_weights, LieGroup* mean,
    const std::function<LieGroup(const std::vector<LieGroup>&)>&
        get_initial_estimate = nullptr,
    int max_iterations = -1,
    typename LieGroup::Scalar tolerance =
        liegroup::Constants<typename LieGroup::Scalar>::kEps) {
  LieGroup initial_estimate = elements[0];
  if (get_initial_estimate) {
    initial_estimate = get_initial_estimate(elements);
  }

  return _computeLieGroupMean<_left_perturbation>(
      elements, initial_estimate, mean, &matrix_weights, max_iterations,
      tolerance);
}

/// @brief  A convenient overload for the above @ref computeLieGroupMean()
///         when the weight matrices are given by pointers.
template <bool _left_perturbation, typename LieGroup, typename WeightMatrix>
bool computeLieGroupMean(
    const std::vector<LieGroup>& elements,
    const std::vector<WeightMatrix*>& matrix_weights, LieGroup* mean,
    const std::function<LieGroup(const std::vector<LieGroup>&)>&
        get_initial_estimate = nullptr,
    int max_iterations = -1,
    typename LieGroup::Scalar tolerance =
        liegroup::Constants<typename LieGroup::Scalar>::kEps) {
  LieGroup initial_estimate = elements[0];
  if (get_initial_estimate) {
    initial_estimate = get_initial_estimate(elements);
  }

  return _computeLieGroupMean<_left_perturbation>(
      elements, initial_estimate, mean, &matrix_weights, max_iterations,
      tolerance);
}

/// @brief  A convenient overload for the above @ref computeLieGroupMean()
///         when scalar weights are used.
template <bool _left_perturbation, typename LieGroup>
bool computeLieGroupMean(
    const std::vector<LieGroup>& elements,
    const std::vector<typename LieGroup::Scalar>& scalar_weights,
    LieGroup* mean,
    const std::function<LieGroup(const std::vector<LieGroup>&)>&
        get_initial_estimate = nullptr,
    int max_iterations = -1,
    typename LieGroup::Scalar tolerance =
        liegroup::Constants<typename LieGroup::Scalar>::kEps) {
  LieGroup initial_estimate = elements[0];
  if (get_initial_estimate) {
    initial_estimate = get_initial_estimate(elements);
  }

  return _computeLieGroupMean<_left_perturbation>(
      elements, initial_estimate, mean, &scalar_weights, max_iterations,
      tolerance);
}

/// @brief  A convenient overload for the above @ref computeLieGroupMean()
///         when uniform weights are used.
template <bool _left_perturbation, typename LieGroup>
bool computeLieGroupMean(
    const std::vector<LieGroup>& elements, LieGroup* mean,
    const std::function<LieGroup(const std::vector<LieGroup>&)>&
        get_initial_estimate = nullptr,
    int max_iterations = -1,
    typename LieGroup::Scalar tolerance =
        liegroup::Constants<typename LieGroup::Scalar>::kEps) {
  LieGroup initial_estimate = elements[0];
  if (get_initial_estimate) {
    initial_estimate = get_initial_estimate(elements);
  }

  return _computeLieGroupMean<_left_perturbation, typename LieGroup::Scalar>(
      elements, initial_estimate, mean, nullptr, max_iterations, tolerance);
}

/// @brief The RANSAC model for computing the mean of a set of Lie group
///        elements.
/// @tparam LieGroup
///           The Lie group type.
/// @tparam _left_perturbation
///           If true, the error is defined to be a left perturbation (computed
///           as "element * @p expected .inverse()"); If false, the error is
///           defined to be a right perturbation (computed as " @p expected
///           .inverse() * element").
template <typename LieGroup, bool _left_perturbation = false>
class LieGroupSacModel {
 public:
  using DataPoint = LieGroup;
  using Parameter = LieGroup;
  using Scalar = typename LieGroup::Scalar;
  using SampleWeightMarix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  static constexpr int kMinimalSampleSize = 1;

  /// @param max_iterations
  ///         The maximum number of internal iterations when
  ///         compute the mean of a set of Lie group elements,
  ///         See @ref computeLieGroupMean().
  /// @param get_initial_estimate
  ///         A function that returns an initial estimate of the mean. If
  ///         `nullptr`, the first element of `elements` is used as the initial
  ///         estimate. See @ref computeLieGroupMean().
  /// @param sample_weight_matrix
  ///         The sample weight matrix used for all the samples.
  ///         It can be of size (N,N), (N,1) or (1,1) (N is the dimension of the
  ///         Lie group). See @ref computeLieGroupMean().
  /// @param tolerance
  ///         The tolerance for convergence. See @ref computeLieGroupMean().
  explicit LieGroupSacModel(
      int max_iterations = -1,
      const std::function<
          LieGroup(const std::vector<size_t>&, const std::vector<DataPoint>&)>&
          get_initial_estimate = nullptr,
      const SampleWeightMarix& sample_weight_matrix = SampleWeightMarix(),
      Scalar tolerance = liegroup::Constants<Scalar>::kEps)
      : get_initial_estimate_(get_initial_estimate),
        max_iterations_(max_iterations),
        sample_weight_matrix_(sample_weight_matrix),
        tolerance_(tolerance) {}

  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    if (selected_indices.size() < kMinimalSampleSize) {
      return std::vector<Parameter>();
    }

    if (selected_indices.size() == 1) {
      return std::vector<Parameter>({all_points[selected_indices[0]]});
    }

    LieGroup initial_estimate = all_points[selected_indices[0]];
    if (get_initial_estimate_) {
      initial_estimate = get_initial_estimate_(selected_indices, all_points);
    }

    LieGroup mean;
    auto weights = getSampleWeights();
    bool success =
        _computeLieGroupMean<_left_perturbation, const SampleWeightMarix*>(
            selected_indices, all_points, initial_estimate, &mean,
            weights.get(), max_iterations_, tolerance_);
    if (!success) {
      LOGI("DEBUG_LIE_RANSAC: Failed to compute mean!!!");
      return std::vector<Parameter>();
    } else {
      return std::vector<Parameter>({mean});
    }
  }

  std::vector<Scalar> errors(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points, const Parameter& model) const {
    std::vector<Scalar> errs;
    if (selected_indices.empty()) {
      return errs;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals;
    auto weights = getSampleWeights();
    _computeLieGroupErrors<
        _left_perturbation, const SampleWeightMarix*, LieGroup>(
        selected_indices, all_points, weights.get(), model, &residuals,
        nullptr);
    ASSERT(residuals.rows() == selected_indices.size() * LieGroup::kDim);
    errs.reserve(selected_indices.size());
    for (size_t i = 0; i < selected_indices.size(); ++i) {
      auto residual =
          residuals.template segment<LieGroup::kDim>(i * LieGroup::kDim);
      errs.push_back(residual.squaredNorm());
    }
    return errs;
  }

  bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const {
    auto res = compute(inlier_indices, all_points);
    if (res.empty()) {
      return false;
    } else {
      *param = res[0];
      return true;
    }
  }

 private:
  std::unique_ptr<std::vector<const SampleWeightMarix*>> getSampleWeights()
      const {
    if (sample_weight_matrix_.rows()) {
      return std::make_unique<std::vector<const SampleWeightMarix*>>(
          1, &sample_weight_matrix_);
    } else {
      return nullptr;
    }
  }

 private:
  std::function<LieGroup(
      const std::vector<size_t>&, const std::vector<DataPoint>&)>
      get_initial_estimate_;
  int max_iterations_;
  Scalar tolerance_;
  SampleWeightMarix sample_weight_matrix_;
};

}  // namespace sk4slam
