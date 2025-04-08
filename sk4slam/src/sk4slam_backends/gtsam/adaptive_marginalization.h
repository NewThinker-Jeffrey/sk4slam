#pragma once

#include <functional>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_unstable/nonlinear/BayesTreeMarginalizationHelper.h>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sk4slam {

/// @class AdaptiveMarginalization
/// @brief A class for adaptive variable marginalization in ISAM2.
///
/// @details
/// This class implements an adaptive marginalization stratagy designed to:
/// 1. **Maximize Efficiency**:
///    - Prioritizes marginalizing leaf keys with zero or minimal overhead.
///    - Escalates to more costly strategies (balanced or forceful
///    marginalization) only when necessary.
/// 2. **Ensure Accuracy**:
///    - Evaluates the linearization error introduced by marginalizing
///    variables.
///    - Ensures that only variables causing negligible linearization errors are
///    marginalized, maintaining high solution accuracy.
///
/// It supports three levels of marginalization, automatically adapting
/// to the problem's requirements:
/// - **Smart Marginalization** (default):
///   - Marginalizes leaf keys with minimal computational overhead and
///   controlled linearization error.
///   - Ideal for most scenarios, featuring both high efficiency and accuracy.
/// - **Balanced Marginalization**:
///   - Applied when Smart Marginalization cannot sufficiently reduce the factor
///   graph size.
///   - Marginalizes keys with controlled linearization error and limited
///   computational overhead.
/// - **Forceful Marginalization** (fallback):
///   - Used as a last resort when the factor graph remains too large after
///   Balanced Marginalization.
///   - Aggressively shrinks the graph by marginalizing keys with the smallest
///   linearization errors.
///
/// This strategy ensures that marginalization operations are computationally
/// efficient while minimizing their impact on solution accuracy.
/// By adaptively controlling the overhead and linearization errors introduced
/// by marginalization, it is particularly effective for handling complex factor
/// graphs in optimization problems like SLAM and large-scale non-linear least
/// squares, where both efficiency and accuracy are critical.
class AdaptiveMarginalization
    : public gtsam::BayesTreeMarginalizationHelper<gtsam::ISAM2> {
  using BayesTreeMarginalizationHelper =
      gtsam::BayesTreeMarginalizationHelper<gtsam::ISAM2>;

 public:
  using FactorIndex = gtsam::FactorIndex;
  using Key = gtsam::Key;

  struct Options {
    /// @brief Minimum number (> 0) of variables to retain in the graph.
    /// @details
    /// This parameter sets the minimum number of variables that must remain
    /// in the graph. If the number of variables falls below this threshold,
    /// marginalization will be skipped to prevent prematurely reducing the
    /// graph's complexity.
    int min_num_variables;

    /// @brief Threshold (> @ref min_num_variables) for the number of variables
    /// to trigger marginalization.
    /// @details
    /// When the number of variables exceeds this threshold, marginalization is
    /// triggered.
    /// @note Negative values represent an infinite threshold.
    int max_num_variables;

    /// @brief Threshold for the number of factors to trigger marginalization.
    /// @details
    /// If the number of factors exceeds this threshold (while the number of
    /// variables is above @ref min_num_variables), marginalization will be
    /// triggered.
    /// @note Negative values represent an infinite threshold.
    int max_num_factors;

    /// @brief Threshold for the number of variable-factor connections to
    /// trigger marginalization.
    /// @details
    /// When the number of variable-factor connections exceeds this threshold
    /// (while the number of variables is above @ref min_num_variables),
    /// marginalization will be triggered. This helps manage graph complexity
    /// when there are many variable-factor connections.
    /// @note Negative values represent an infinite threshold.
    int max_num_connections;

    /// @brief Threshold for acceptable marginalization overhead (per variable)
    /// during balanced marginalization.
    /// @details
    /// This defines the maximum allowable computational overhead (per variable)
    /// during the balanced marginalization process. If the overhead for a
    /// variable exceeds this threshold, balanced marginalization will skip the
    /// variable. The goal is to balance accuracy with computational efficiency,
    /// ensuring marginalization does not incur excessive cost.
    /// @note Negative values represent an infinite threshold.
    double marginalization_overhead_threshold;

    /// @brief Threshold for acceptable marginalization error when marginalizing
    /// a variable.
    /// @details
    /// This parameter defines the maximum allowable error for a variable during
    /// the marginalization process. If the marginalization error of a variable
    /// exceeds this threshold, the variable will not be marginalized unless the
    /// forceful marginalization process is triggered.
    ///
    /// Both **linearization error** and **marginalization error** are
    /// considered relative in nature, meaning that the errors are expressed as
    /// ratios or relative magnitudes, rather than absolute values. This allows
    /// a single threshold to be applied uniformly across all variables,
    /// regardless of their scale or dimensionality. Since the errors are
    /// relative, this threshold is typically set to 1.0, meaning that the error
    /// should not exceed 100% of the respective quantities.
    double marginalization_error_threshold;

    /// @brief  Whether to use the Aggressive Policy in Smart Marginalization.
    /// @details If enabled, the algorithm will immediately marginalize all
    /// variables that are marked marginalizable and do not exceed the error
    /// thresholds, even if the graph size is already below the specified
    /// thresholds; Otherwise (Lazy Policy), the algorithm won't attempt to
    /// marginalize any variable unless the graph size exceeds the thresholds.
    ///
    /// Aggressive Policy can be beneficial in scenarios where efficiency is
    /// more critical than accuracy.
    bool aggressive = false;

    Options(
        int min_num_variables = 10, int max_num_variables = 50,
        int max_num_factors = -1, int max_num_connections = -1,
        double marginalization_overhead_threshold = -1,
        double marginalization_error_threshold = 1.0, bool aggressive = false);
    virtual ~Options() = default;
  };

  AdaptiveMarginalization(
      std::shared_ptr<const Options> options,
      const gtsam::ISAM2* isam = nullptr, bool debug = false)
      : isam_(isam), options_(std::move(options)), debug_(debug) {}

  virtual ~AdaptiveMarginalization() = default;

  void bindISAM2(const gtsam::ISAM2* isam);

  /// @brief Determines which keys to marginalize.
  /// @details
  /// This function first applies **smart marginalization** to select variables
  /// with minimal overhead and acceptable error. If more variables need to be
  /// marginalized, **balanced marginalization** is triggered. In extreme cases,
  /// **forceful marginalization** is applied as a last resort.
  /// @param force_marginalize_keys A set of variable keys forced to
  /// marginalize.
  /// @param force_keep_keys A set of variable keys forced to keep.
  /// @param already_affected_keys  A set of keys already affected (e.g., by
  /// adding or removing factors) before the marginalization process. This is
  /// used only in balanced marginalization to avoid double-counting overhead
  /// caused by these keys, ensuring a more accurate estimate of the additional
  /// overhead incurred by marginalizing variables.
  /// @return A set of keys to marginalize.
  virtual std::unordered_set<Key> determineKeysToMarginalize(
      const std::unordered_set<Key>& force_marginalize_keys = {},
      const std::unordered_set<Key>& force_keep_keys = {},
      const std::unordered_set<Key>& already_affected_keys = {},
      const std::unordered_set<Key>& no_relinear_keys = {}) const;

  /// This function identifies keys that need to be re-eliminated before
  /// performing marginalization.
  template <typename KeysContainer>
  static std::unordered_set<Key> gatherAdditionalKeysToReEliminate(
      const gtsam::ISAM2& isam, const KeysContainer& marginalizableKeys) {
    return BayesTreeMarginalizationHelper::gatherAdditionalKeysToReEliminate(
        isam, toKeyVector(marginalizableKeys));
  }

 protected:
  /// @brief   Checks if a key is marginalizable. Subclasses must override to
  /// customize the marginalization criteria.
  /// @param key  The key to be evaluated for marginalization.
  virtual bool isMarginalizable(Key key) const = 0;

  /// @brief   Iterates over all marginalizable keys in order.
  /// Subclasses can override to improve iteration speed and customize the
  /// order.
  /// @param f  A function to process each key. Iteration stops if it returns
  /// true.
  /// @return  True if the function `f` returns true for any key.
  virtual bool iterateOverMarginalizableKeys(
      const std::function<bool(const Key&)>& f) const;

  /// @brief   Determines if further marginalization is required after a set
  /// of keys have been marginalized.
  /// @param assumed_marginalized_keys  The set of (assumed) already
  /// marginalized keys.
  /// @param assumed_marginalized_factors  The set of (assumed) already
  /// marginalized factors.
  virtual bool needsFurtherMarginalization(
      const std::unordered_set<Key>& assumed_marginalized_keys = {},
      const std::unordered_set<FactorIndex>& assumed_marginalized_factors = {})
      const;

  /// @brief   Iterates over all leaf keys that are ready for marginalization.
  /// Subclasses can override to customize iteration order.
  /// @param f  A function to process each key. Iteration stops if it returns
  /// true.
  /// @param assumed_marginalized_keys  The set of already marginalized keys.
  /// @return  True if the function `f` returns true for any key.
  virtual bool iterateOverLeafKeys(
      const std::function<bool(const Key&)>& f,
      const std::unordered_set<Key>& assumed_marginalized_keys = {}) const;

  /// @brief Computes the linearization error for a specific key.
  ///
  /// This method calculates the linearization error using a default
  /// implementation. The error is computed as the ratio of the difference
  /// between the current linearization point and the optimal estimate to the
  /// relinearization threshold associated with the given key in the underlying
  /// ISAM2 instance.
  ///
  /// Subclasses can override this method to define a custom error calculation,
  /// allowing for more sophisticated or context-specific definitions of
  /// linearization error.
  ///
  /// @param key The key for which the linearization error is computed.
  /// @return The computed linearization error as a double.
  virtual double evaluateLinearizationError(const Key& key) const;

  /// @brief Computes the marginalization error for a given key.
  ///
  /// This method provides a default implementation for calculating the
  /// marginalization error. The marginalization error is defined as the maximum
  /// linearization error among all keys connected to the input key through
  /// factors (referred to as "involved keys"). Keys with fixed linearization
  /// points are excluded from this calculation, as they do not contribute
  /// additional linearization error.
  ///
  /// Subclasses can override this method to define a custom calculation for
  /// marginalization error tailored to specific requirements or contexts.
  ///
  /// @param key The key for which the marginalization error is to be computed.
  /// @return The computed marginalization error as a double.
  virtual double evaluateMarginalizationError(const Key& key) const;

  /// @brief Estimates the additional overhead caused by marginalizing a key.
  /// @details
  /// This virtual function provides a default implementation that uses a
  /// heuristic approach to estimate the additional computational overhead
  /// due to re-elimination (in QR decomposition) when marginalizing a key.
  /// This overhead reflects the primary computational cost of the
  /// marginalization process, ensuring that it does not introduce excessive
  /// computational burden.
  ///
  /// Subclasses can override this method to provide more precise or specialized
  /// overhead estimates if needed.
  ///
  /// @param key The key for which the overhead is being estimated.
  /// @return The estimated overhead introduced by marginalizing the specified
  /// key.
  virtual double evaluateMarginalizationOverhead(const Key& key) const;

 protected:
  /// @brief   Helper function for `iterateOverLeafKeys()`, iterates over leaf
  /// keys in a subtree, considering dependencies and assumed marginalized keys.
  /// @param subtree_root  The root of the subtree to iterate over.
  /// @param f  A function to process the keys. Stops if it returns true.
  /// @param dependencies  Stores keys that are dependencies and should not
  /// be marginalized.
  /// @param assumed_marginalized_keys  The keys that are (assumed) already
  /// marginalized. Note the set may be extended by calls to @c f during the
  /// iteration.
  /// @return   True if @c f returns true, false otherwise.
  static bool iterateOverLeafKeysInSubtree(
      const gtsam::ISAM2Clique::shared_ptr& subtree_root,
      const std::function<bool(const Key&)>& f,
      std::unordered_set<Key>* dependencies,
      const std::unordered_set<Key>& assumed_marginalized_keys);

  /// @brief   Iterates over keys in a subtree.
  /// @param subtree_root  The root of the subtree to iterate over.
  /// @param f  A function to process the keys. Stops if it returns true.
  /// @return   True if @c f returns true, false otherwise.
  static bool iterateOverKeysInSubtree(
      const gtsam::ISAM2Clique::shared_ptr& subtree_root,
      const std::function<bool(const Key&)>& f);

  /// @brief Gathers all keys involved in the re-elimination process when
  /// marginalizing a key.
  /// @param keys_to_marginalize  A set of keys to marginalize.
  /// @param update_cached_already_affected_cliques  If true, updates the
  /// cached already-affected cliques for the specified keys.
  /// @return  A set of keys that require re-elimination.
  std::unordered_set<Key> gatherAdditionalKeysToReEliminateAlongPathToRoot(
      const std::unordered_set<Key>& keys_to_marginalize,
      bool update_cached_already_affected_cliques = false) const;

  /// @brief Gathers all cliques along the path from the specified cliques to
  /// the root of the ISAM2 Bayes tree.
  /// @param source_cliques  A set of cliques from which to begin tracing the
  /// path.
  /// @param extra_terminate_cliques  A set of additional cliques to be treated
  /// as terminating points (e.g., "fake" roots). But note that these
  /// cliques are not included in the returned set.
  /// @return A set of cliques along the path to the root.
  std::unordered_set<const Clique*> gatherCliquesAlongPathToRoot(
      const std::unordered_set<const Clique*>& source_cliques,
      const std::unordered_set<const Clique*>& extra_terminate_cliques = {})
      const;

 protected:
  double getCachedLinearizationError(const Key& key) const;

  void resetCachedLinearizationErrors() const;

  void testLinearizationErrors() const;  // For debugging only

  double getCachedMarginalizationError(const Key& key) const;

  void resetCachedMarginalizationErrors() const;

  void initCachedAlreadyAffectedCliques(
      const std::unordered_set<Key>& already_affected_keys) const;

  void updateCachedAlreadyAffectedCliques(
      const std::unordered_set<const Clique*>& new_affected_cliques) const;

  void warnForPoorLinearization(
      const std::unordered_set<Key>& keys_to_be_marginalized,
      const std::unordered_set<Key>& forceful_marginalization_keys) const;

  bool isLinearizationFixed(Key key) const;

 protected:  // Templated helper functions for subclasses
  /// This function identifies cliques that need to be re-eliminated before
  /// performing marginalization.
  template <typename KeysContainer>
  static std::unordered_set<const Clique*> gatherAdditionalCliquesToReEliminate(
      const gtsam::ISAM2& isam, const KeysContainer& marginalizableKeys) {
    return BayesTreeMarginalizationHelper::gatherAdditionalCliquesToReEliminate(
        isam, toKeyVector(marginalizableKeys));
  }

  /// Converts a KeysContainer to a KeyVector
  template <typename KeysContainer>
  static gtsam::KeyVector toKeyVector(const KeysContainer& keys) {
    gtsam::KeyVector key_vector(keys.begin(), keys.end());
    return key_vector;
  }

  /// Identifies the factors that are connected to the given keys
  /// in the underlying factor graph.
  template <typename KeysContainer>
  std::unordered_set<FactorIndex> getFactorsInvolved(
      const KeysContainer& keys) const {
    std::unordered_set<FactorIndex> factors_involved;
    const gtsam::VariableIndex& key_to_factor_indices =
        isam_->getVariableIndex();
    for (const Key& key : keys) {
      const auto& cur_factor_indices = key_to_factor_indices[key];
      factors_involved.insert(
          cur_factor_indices.begin(), cur_factor_indices.end());
    }
    return factors_involved;
  }

  /// Returns the total number of variable-factor connections
  /// involving the given factors.
  template <typename FactorIndexContainer>
  int countConnections(const FactorIndexContainer& factor_indices) const {
    int num_connections = 0;
    const gtsam::NonlinearFactorGraph& underlying_factor_graph =
        isam_->getFactorsUnsafe();
    for (const FactorIndex& factor_index : factor_indices) {
      auto factor = underlying_factor_graph.at(factor_index);
      num_connections += factor->keys().size();
    }
    return num_connections;
  }

 protected:
  const gtsam::ISAM2* isam_;
  std::shared_ptr<const Options> options_;
  bool debug_;

 private:
  mutable std::unordered_map<Key, double> cached_linearization_errors_;
  mutable std::unordered_map<Key, double> cached_marginaization_errors_;
  mutable std::unordered_set<const Clique*> cached_already_affected_cliques_;
  mutable const std::unordered_set<Key>* user_defined_no_relinear_keys_;
};

}  // namespace sk4slam
