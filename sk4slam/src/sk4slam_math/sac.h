#pragma once

#include <Eigen/Core>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_cpp/binary_search.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

namespace sk4slam {

//////////////////// SAC Interface examples ////////////////////

template <typename Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>>
struct VectorSacModel {
 public:  // basic properties.
  using DataPoint = Vector;
  using Parameter = Vector;
  // Note `DataPoint` and `Parameter` can also be user-defined structures.

  // static constexpr int kMinimalSampleSize = 3;
  static constexpr int kMinimalSampleSize = 1;

  // Compute the model parameters from selected points.
  // Return all the possible parameters that are consistent with the points.
  //
  // Instead of passing a vector of the selected points, we pass a vector of
  // the indices of the selected points and a vector containing all the points.
  // This is done to avoid unnecessary copying of the points.
  //
  std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    if (selected_indices.size() < kMinimalSampleSize) {
      return std::vector<Parameter>();
    }

    DataPoint sum = all_points[selected_indices[0]];
    for (size_t i = 1; i < selected_indices.size(); i++) {
      sum += all_points[selected_indices[i]];
    }

    std::vector<Parameter> params({sum / selected_indices.size()});

    // static constexpr bool kDebug = true;
    static constexpr bool kDebug = false;
    if constexpr (kDebug) {
      Oss oss_indices;
      for (size_t i : selected_indices) {
        oss_indices << i << " ";
      }
      LOGA(
          "param: %s, points: %s", toStr(params[0].transpose()).c_str(),
          oss_indices.str().c_str());
    }

    return params;
  }

  // The errors (non-negative) of points with respect to the model.
  std::vector<double> errors(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points, const Parameter& model) const {
    std::vector<double> errs;
    errs.reserve(selected_indices.size());
    DataPoint delta;
    for (const auto& idx : selected_indices) {
      const auto& point = all_points[idx];
      delta = point - model;
      errs.push_back(delta.squaredNorm());
    }
    return errs;
  }

 public:  // specialized properties.
  // The similarity of point1 and point2 with respect to the model.
  //
  // When verifying a model, votes from similar points may be considered
  // replicated votes or their importance shuold be reduced.
  double similarity(
      const DataPoint& point1, const DataPoint& point2,
      const Parameter& model) const {
    return 0;
  }

  // Do LO with all the inliers (needed by LO-RANSAC framework).
  // `param` shuold already hold a reasonable initial guess for the model
  // parameters.
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

  // - About Degeneracy Handling:
  //
  // Sometimes a model is not computable from the selected points because
  // the points are distributed on a submanifold that doesn't uniquly
  // determine the model, which yields a degenerate case. For example when our
  // model is a plane while our selected points all lay near a straight line.
  //
  // In this case, inner steps might be applied to try recovering the model
  // with extra points off the submanifold.
  //
  // However if all the points lay near the submanifold, then either a refined
  // degenerate model is adopted or the function just returns false. If you
  // need to return a degenerate model, the type `Parameter` should be capable
  // to describe it, i.e. our model is a heterougeneous model and `Parameter`
  // should include paramters for both the original model and the degenerate
  // model, as well as a flag or something to indicate which model is adopted.
  //
  // The argument `model` should already contain an initial value that was
  // computed from the selected points, and the function may update `model`
  // if the degeneracy is detected and handled.
  // The argument `initial_inliers` is the set of inliers for the initial
  // model.
  // The function returns true if the `model` is updated, false otherwise.
  bool handleDegeneracy(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points,
      const std::vector<size_t>& initial_inliers, Parameter* model) {
    return false;
  }
};

// clang-format off

// struct SacSamplerExample {
//   // Note: this function is not reenterable.
//   std::vector<size_t> sample(
//       size_t k,
//       const std::vector<SacModelType::DataPoint>&
//           all_points) const {
//     std::vector<size_t> indices;
//     indices.resize(k);
//     std::iota(indices.begin(), indices.end(), 0);
//     return indices;
//   }
// };

// struct SacScoringExample {

//   using Score = double;

//   // Compute the score.
//   Score evaluate(
//       const SacModelType& model,
//       const SacModelType::Parameter& param,
//       const std::vector<SacModelType::DataPoint>& all_points,
//       const SacOptionsType& sac_options,
//       std::vector<size_t>* inliers) const {
//     return 0;
//   }

//   // Compare the two scores (return true if score2 is strictly better).
//   bool compare(const Score& score1, const Score& score2) const {
//     return score1 < score2;
//   }
// };

// clang-format on

//////////////////// RANSAC implementation ////////////////////

class RandomSampler {
 public:
  // Note: this function is not reenterable.
  std::vector<size_t> sample(size_t k, size_t n) const {
    std::vector<size_t> indices;
    if (k > n) {
      return indices;
    }

    if (all_indices_.size() != n) {
      all_indices_.resize(n);
      std::iota(all_indices_.begin(), all_indices_.end(), 0);
      // distribution_ =
      //     std::uniform_int_distribution<size_t>(0, n - 1);
    }

    indices.reserve(k);
    shuffle(k, &all_indices_);

    for (size_t i = 0; i < k; i++) {
      indices.push_back(all_indices_[i]);
    }
    return indices;
  }

  // Note: this function is not reenterable.
  template <typename DataPoint>
  std::vector<size_t> sample(
      size_t k, const std::vector<DataPoint>& all_points) const {
    return sample(k, all_points.size());
  }

 private:
  mutable std::vector<size_t> all_indices_;
  // mutable std::uniform_int_distribution<size_t> distribution_;
};

class CombinationSampler {
 public:
  // Note: this function is not reenterable.
  std::vector<size_t> sample(size_t k, size_t n) const {
    std::vector<size_t> indices;
    if (k > n) {
      return indices;
    }

    if (all_indices_.size() != n || k_ != k) {
      all_indices_.resize(n);
      std::iota(all_indices_.begin(), all_indices_.end(), 0);
      generated_samples_ = 0;
      k_ = k;
      Cnk_ = Cnk(n, k);
    }

    if (generated_samples_ < Cnk_) {
      indices.reserve(k);
      for (size_t i = 0; i < k; i++) {
        indices.push_back(all_indices_[i]);
      }
      nextCombination(
          all_indices_.begin(), all_indices_.begin() + k, all_indices_.end());
      ++generated_samples_;
    }
    return indices;
  }

  // Note: this function is not reenterable.
  template <typename DataPoint>
  std::vector<size_t> sample(
      size_t k, const std::vector<DataPoint>& all_points) const {
    return sample(k, all_points.size());
  }

 private:
  mutable std::vector<size_t> all_indices_;
  mutable size_t generated_samples_;
  mutable size_t k_ = 0;
  mutable size_t Cnk_ = 0;
};

struct RansacOptions {
  // Error threshold for a point to be considered consistent with a model.
  double error_thr;

  // `min_inlier_ratio` and `confidence` are used to compute the maximum
  // number of iterations. The iteration will be aborted when the minimum
  // probability that we already have one sample free from outlier points
  // reaches `confidence`.
  double confidence;
  double initial_min_inlier_ratio;
  // `min_inlier_ratio` is dynamically estimated during the  RANSAC process.

  int local_opt_max_iter;
  int final_opt_max_iter;

  // Whether to deal with degenerate cases.
  bool check_step_degeneracy;
  bool check_final_degeneracy;

  // Sometimes the maximum number of iterations computed with
  // `min_inlier_ratio` and `confidence` can be extremely large but we still
  // want to bound the computation time.
  //
  // `max_iter` is used to bound the maximum number of iterations.
  int max_iter;

  int getMaxIter(
      int minimal_points, double min_inlier_ratio = 0.0, int N = -1) const {
    double prob_no_outliers;
    if (N < 0) {
      ASSERT(min_inlier_ratio > 0.0);
      prob_no_outliers = pow(min_inlier_ratio, minimal_points);
    } else {
      int min_inliers = static_cast<int>(N * min_inlier_ratio);
      min_inliers = std::max(minimal_points, min_inliers);
      if (min_inliers > N) {
        return 0;
      }
      prob_no_outliers = 1.0;
      for (int i = 0; i < minimal_points; i++) {
        prob_no_outliers *= (min_inliers - i) / static_cast<double>(N - i);
      }
    }
    double prob_at_least_one_outlier = 1 - prob_no_outliers;
    const double& P = prob_at_least_one_outlier;
    const double& conf = confidence;

    // 'P ^ K' is the propability that we sample K times and all the samples
    // contain some outlier points. We need this propability to be less than
    // (1 - conf), i.e.
    //   P^K < (1 - conf)
    //   K * log(P) < log(1 - conf)
    //   K > log(1 - conf) / log(P)    NOTE: "P < 1" so "log(P) < 0"

    // Note K can be even larger than 2^32.
    double K = log(1 - conf) / log(P);
    if (K > max_iter) {
      // LOGD(
      //     BLUE
      //     "The expected number of iterations according to confidence"
      //     " is %.0f, however the max_iter is %d." RESET,
      //     K, max_iter);
      return max_iter;
    } else {
      // return ceil(Kd);
      return static_cast<int>(K);
    }
  }

  RansacOptions(
      double error_thr_in = 0.0, double confidence_in = 0.999,
      int max_iter_in = 1000,  // std::numeric_limits<int>::max(),
      double initial_min_inlier_ratio_in = 0.0, int local_opt_max_iter_in = 0,
      int final_opt_max_iter_in = 0, bool check_step_degeneracy_in = false,
      bool check_final_degeneracy_in = false)
      : error_thr(error_thr_in),
        confidence(confidence_in),
        initial_min_inlier_ratio(initial_min_inlier_ratio_in),
        local_opt_max_iter(local_opt_max_iter_in),
        final_opt_max_iter(final_opt_max_iter_in),
        check_step_degeneracy(check_step_degeneracy_in),
        check_final_degeneracy(check_final_degeneracy_in),
        max_iter(max_iter_in) {}
};

struct RansacInlierScoring {
  struct Score {
    // The number of inliers.
    size_t num_inliers = 0;

    // TODO(jeffrey): As a secondary criterion, maybe it's better to compute
    // the median (as in LMedS), instead of the sum, of all inlier errs?
    // Reconsider it.
    //
    // The sum of all inlier errs.
    double err_sum = 0;
  };

  // Compute the score.
  template <typename SacModelType>
  Score evaluate(
      const SacModelType& model, const typename SacModelType::Parameter& param,
      const std::vector<typename SacModelType::DataPoint>& all_points,
      const RansacOptions& sac_options, std::vector<size_t>* inliers) const {
    Score score;
    score.num_inliers = 0;
    score.err_sum = 0;

    inliers->clear();
    inliers->reserve(all_points.size());

    std::vector<size_t> all_indices;
    all_indices.resize(all_points.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    auto errs = model.errors(all_indices, all_points, param);
    ASSERT(errs.size() == all_points.size());

    for (size_t i = 0; i < all_points.size(); i++) {
      double err = errs[i];
      if (err > sac_options.error_thr) {
        continue;
      }
      score.err_sum += err;
      ++score.num_inliers;
      inliers->push_back(i);
    }
    ASSERT(score.num_inliers == inliers->size());

    return score;
  }

  // Compare the two scores (return true if score2 is strictly better).
  bool compare(const Score& score1, const Score& score2) const {
    if (score1.num_inliers < score2.num_inliers) {
      return true;
    } else {
      return score1.num_inliers == score2.num_inliers &&
             score1.err_sum > score2.err_sum;
    }
  }
};

template <
    typename ModelType, typename ScoringType = RansacInlierScoring,
    typename SamplerType = RandomSampler>
class Ransac {
  DEFINE_HAS_MEMBER_FUNCTION(localOptimize)

  DEFINE_HAS_MEMBER_FUNCTION(handleDegeneracy)

  DEFINE_HAS_MEMBER_FUNCTION(similarity)

  static constexpr bool kHasLocalOptimize = HasMemberFunction_localOptimize<
      ModelType, const std::vector<size_t>&,
      const std::vector<typename ModelType::DataPoint>&,
      typename ModelType::Parameter*>;

  static constexpr bool kHasHandleDegeneracy =
      HasMemberFunction_handleDegeneracy<
          ModelType, const std::vector<size_t>&,
          const std::vector<typename ModelType::DataPoint>&,
          const std::vector<size_t>&, typename ModelType::Parameter*>;

  static constexpr bool kHasSimilarity = HasMemberFunction_similarity<
      ModelType, const typename ModelType::DataPoint&,
      const typename ModelType::DataPoint&,
      const typename ModelType::Parameter&>;

 public:
  struct Report {
    typename ModelType::Parameter param;
    std::vector<size_t> inliers;  // sorted indices
    double inlier_ratio;
    size_t n_total;

    RansacOptions ransac_options;
    typename ScoringType::Score score;
    int iter = 0;

    std::vector<size_t> getOutliers() const {
      std::vector<size_t> outliers;
      outliers.reserve(n_total - inliers.size());
      size_t j = 0;
      for (size_t i = 0; i < n_total; i++) {
        // Note inliers are sorted
        if (j < inliers.size() && i == inliers[j]) {
          ++j;
        } else {
          outliers.push_back(i);
        }
      }
      return outliers;
    }

    template <
        typename MaskValueType = uint8_t,
        MaskValueType _inlier_v = MaskValueType(1),
        MaskValueType _outlier_v = MaskValueType(0)>
    void getInliersMask(std::vector<MaskValueType>* mask) const {
      mask->reserve(n_total);
      size_t j = 0;
      for (size_t i = 0; i < n_total; i++) {
        // Note inliers are sorted
        if (j < inliers.size() && i == inliers[j]) {
          ++j;
          mask->push_back(_inlier_v);
        } else {
          mask->push_back(_outlier_v);
        }
      }
    }
  };

 public:
  Ransac(
      const RansacOptions& options = RansacOptions(),
      ModelType model = ModelType(), ScoringType scoring = ScoringType(),
      SamplerType sampler = SamplerType())
      : options_(options),
        model_(new ModelType(std::move(model))),
        scoring_(std::move(scoring)),
        sampler_(std::move(sampler)) {}

  virtual ~Ransac() {}

  virtual Report solve(
      const std::vector<typename ModelType::DataPoint>& all_points) {
    Report report;
    report.iter = 0;
    report.n_total = all_points.size();
    report.ransac_options = options_;

    double min_inlier_ratio = options_.initial_min_inlier_ratio;
    int max_iter = options_.getMaxIter(
        ModelType::kMinimalSampleSize, min_inlier_ratio, all_points.size());
    typename ScoringType::Score best_score_before_lo;

    for (int iter = 0; iter < max_iter; ++iter) {
      ++report.iter;
      std::vector<size_t> selected_indices =
          sampler_.sample(ModelType::kMinimalSampleSize, all_points);

      if (selected_indices.empty()) {
        // We have no more samples.
        break;
      }

      ASSERT(selected_indices.size() == ModelType::kMinimalSampleSize);
      // if (selected_indices.size() < ModelType::kMinimalSampleSize) {
      //   continue;
      // }

      std::vector<typename ModelType::Parameter> new_params =
          model_->compute(selected_indices, all_points);

      const bool only_keep_the_best_one_in_new_params = true;
      // const bool only_keep_the_best_one_in_new_params = false;

      if (only_keep_the_best_one_in_new_params && new_params.size() > 1) {
        // choose the best one
        bool is_first_solution = true;
        typename ScoringType::Score local_best_score;
        typename ModelType::Parameter local_best_param;
        typename std::vector<size_t> local_best_inliers;
        for (auto& new_param : new_params) {
          std::vector<size_t> inliers;
          typename ScoringType::Score score = scoring_.evaluate(
              *model_, new_param, all_points, options_, &inliers);
          if (is_first_solution || scoring_.compare(local_best_score, score)) {
            is_first_solution = false;
            std::swap(local_best_score, score);
            std::swap(local_best_param, new_param);
            std::swap(local_best_inliers, inliers);
          }
        }
        // only keep the best one in new_params
        new_params.clear();
        new_params.push_back(local_best_param);
      }

      for (auto& new_param : new_params) {
        std::vector<size_t> inliers;
        typename ScoringType::Score score = scoring_.evaluate(
            *model_, new_param, all_points, options_, &inliers);
        typename ScoringType::Score score_before_lo = score;
        size_t n_inliners_before_lo = inliers.size();

        if constexpr (kHasLocalOptimize) {
          if (options_.local_opt_max_iter > 0) {
            // TODO(jeffrey): Consider this.
            // bool need_lo = true;
            bool need_lo =
                !scoring_.compare(score_before_lo, best_score_before_lo);

            if (need_lo) {
              evaluateAndLocalOptimize(
                  all_points, selected_indices, &new_param, &inliers, &score,
                  options_.local_opt_max_iter, true);
            }
          }
        } else {
          LOGA("Ransac: No method localOptimize!! (1)");
        }

        if constexpr (kHasHandleDegeneracy) {
          // Check whether there is a degenerate case and try handle it.
          // If the degeneracy is handled, then the `param`, `inliers`
          // and `score` should be updated.
          if (options_.check_step_degeneracy) {
            const bool only_check_degeneracy_for_higher_score = true;
            if (!only_check_degeneracy_for_higher_score ||
                scoring_.compare(report.score, score)) {
              if (model_->handleDegeneracy(
                      selected_indices, all_points, inliers, &new_param)) {
                // new_param has been updated, revaluate the score and inliers.
                score = scoring_.evaluate(
                    *model_, new_param, all_points, options_, &inliers);
              }
            }
          }
        } else {
          // LOGA("Ransac: No method handleDegeneracy!! (2)");
        }

        if (scoring_.compare(report.score, score)) {
          std::swap(report.score, score);
          std::swap(report.param, new_param);
          std::swap(report.inliers, inliers);
          std::swap(best_score_before_lo, score_before_lo);

          // Update the minimum inlier ratio and max_iter
          int min_inliers =
              std::min(n_inliners_before_lo, report.inliers.size());

          // TODO(jeffrey): Make these hyperparameters configurable.
          static const double min_inliers_multiplier = 0.95;
          static const int min_inliers_reduction = 3;
          min_inliers = std::min(
              static_cast<int>(min_inliers_multiplier * min_inliers),
              min_inliers - min_inliers_reduction);

          double new_min_inlier_ratio = std::max(
              min_inlier_ratio,
              min_inliers / static_cast<double>(all_points.size()));
          int new_max_iter = options_.getMaxIter(
              ModelType::kMinimalSampleSize, new_min_inlier_ratio,
              all_points.size());
          LOGA(
              BLUE
              "UpdateMinInlierRatio: "
              "min_inliers: %d "
              "(n_inliners_before_lo=%d, report.inliers.size()=%d); "
              "old min_inlier_ratio: %f, new min_inlier_ratio: %f, "
              "current_iter: %d, old_max_iter: %d, new_max_iter: %d",
              min_inliers, n_inliners_before_lo, report.inliers.size(),
              min_inlier_ratio, new_min_inlier_ratio, report.iter, max_iter,
              new_max_iter);
          min_inlier_ratio = new_min_inlier_ratio;
          max_iter = new_max_iter;
        }
      }
    }

    if constexpr (kHasHandleDegeneracy) {
      if (options_.check_final_degeneracy && report.inliers.size() > 0) {
        if (model_->handleDegeneracy(
                {}, all_points, report.inliers, &report.param)) {
          // report.param has been updated, revaluate the score and inliers.
          report.score = scoring_.evaluate(
              *model_, report.param, all_points, options_, &report.inliers);
        }
      }
    }

    if constexpr (kHasLocalOptimize) {
      // Run the final optimization to refine the model and expand the inliers
      // if possible.
      if (options_.final_opt_max_iter > 0 && report.inliers.size() > 0) {
        evaluateAndLocalOptimize(
            all_points, {}, &report.param, &report.inliers, &report.score,
            options_.final_opt_max_iter, true);
      }
    }

    report.inlier_ratio =
        static_cast<double>(report.inliers.size()) / all_points.size();
    return report;
  }

  // NOTE:
  //     - `param` must be already initialized.
  //     - If `score` and `inliers` are both provided (non-nullptr), then
  //       whether they have been initilized should be indicated by
  //       `score_and_inliers_already_initialized`:
  //                - true if `score` and `inliers` are already initialized,
  //                - false otherwise.
  //       By default, `score` and `inliers` are not initialized.
  //
  //       Also note that, if `score` and `inliers` are provided but not
  //       initialized, they will be ensured to be initialized after the
  //       functions return even if `local_opt_max_iter=0`.
  virtual void evaluateAndLocalOptimize(
      const std::vector<typename ModelType::DataPoint>& all_points,
      std::vector<size_t> selected_indices,
      typename ModelType::Parameter* param,
      std::vector<size_t>* inliers = nullptr,
      typename ScoringType::Score* score = nullptr, int local_opt_max_iter = -1,
      bool score_and_inliers_already_initialized = false) {
    std::vector<size_t> tmp_inliers;
    typename ScoringType::Score tmp_score;
    if (!inliers) {
      inliers = &tmp_inliers;
      score_and_inliers_already_initialized = false;
    }
    if (!score) {
      score = &tmp_score;
      score_and_inliers_already_initialized = false;
    }

    if (!score_and_inliers_already_initialized) {
      typename ScoringType::Score score =
          scoring_.evaluate(*model_, *param, all_points, options_, inliers);
    }

    if constexpr (kHasLocalOptimize) {
      if (local_opt_max_iter < 0) {
        local_opt_max_iter = options_.local_opt_max_iter;
      }

      // Check whether all the selected points are inliers.
      auto check_selected_points =
          [&selected_indices](const std::vector<size_t>& lo_inliers) {
            for (size_t idx : selected_indices) {
              // lo_inliers is sorted so we can use binary search.
              if (!binarySearchAny(lo_inliers, idx)) {
                return false;
              }
            }
            return true;
          };

      for (size_t lo = 0; lo < local_opt_max_iter; lo++) {
        if (inliers->size() <= ModelType::kMinimalSampleSize) {
          // LOGA(BLUE "Ransac::localOptimize(): Number of inliers is smaller "
          //           "than the min sample size! (skip local optimization)"
          //           RESET);
          break;  // we only do local optimization if the number of inliers is
                  // larger than the minimal sample size
        }

        typename ModelType::Parameter lo_param = *param;
        std::vector<size_t> lo_inliers;
        if (model_->localOptimize(*inliers, all_points, &lo_param)) {
          typename ScoringType::Score lo_score = scoring_.evaluate(
              *model_, lo_param, all_points, options_, &lo_inliers);
          // All selected points should be inliers.
          if (!check_selected_points(lo_inliers)) {
            // clang-format off
            // LOGA(BLUE
            //      "Ransac::localOptimize(): Some selected points "
            //      "become outliers after local optimization!" RESET);
            // clang-format on
            break;
          }
          if (scoring_.compare(*score, lo_score)) {
            std::swap(*score, lo_score);
            std::swap(*param, lo_param);
            std::swap(*inliers, lo_inliers);
          } else {
            break;
          }
        } else {
          break;
        }
      }
    } else {
      LOGA("Ransac: No method localOptimize!! (2)");
      return;
    }
  }

  const ModelType& model() const {
    return *model_;
  }

  const SamplerType& sampler() const {
    return sampler_;
  }

  const ScoringType& scoring() const {
    return scoring_;
  }

  const RansacOptions& options() const {
    return options_;
  }

 protected:
  const RansacOptions options_;
  std::unique_ptr<ModelType> model_;
  SamplerType sampler_;
  ScoringType scoring_;
};

// LMedS
struct LMedSScoring {
  struct Score {
    double median_err = std::numeric_limits<double>::max();
  };

  // Compute the score.
  template <typename SacModelType>
  Score evaluate(
      const SacModelType& model, const typename SacModelType::Parameter& param,
      const std::vector<typename SacModelType::DataPoint>& all_points,
      const RansacOptions& sac_options, std::vector<size_t>* inliers) {
    Score score;
    score.median_err = std::numeric_limits<double>::max();

    std::vector<size_t> all_indices;
    all_indices.resize(all_points.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    auto errs = model.errors(all_indices, all_points, param);
    ASSERT(errs.size() == all_points.size());

    std::nth_element(errs.begin(), errs.begin() + errs.size() / 2, errs.end());
    score.median_err = errs[errs.size() / 2];

    // Find inliers
    if (inliers) {
      inliers->clear();
      inliers->reserve(all_points.size());
      for (size_t i = 0; i < all_points.size(); i++) {
        const auto& err = errs[i];
        if (err <= sac_options.error_thr) {
          inliers->push_back(i);
        }
      }
    }

    return score;
  }

  // Compare the two scores (return true if score2 is strictly better).
  bool compare(const Score& score1, const Score& score2) {
    return score2.median_err < score1.median_err;
  }
};

template <typename ModelType, typename SamplerType = RandomSampler>
using LMedS = Ransac<ModelType, LMedSScoring, SamplerType>;

}  // namespace sk4slam
