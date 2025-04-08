#pragma once

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <memory>
#include <string>

#include "sk4slam_backends/gtsam/adaptive_marginalization.h"
#include "sk4slam_basic/time.h"

namespace sk4slam {

/// @brief IncrementalSmoother is a class that wraps ISAM2 and provides a
/// simplified interface for adding new factors and updating the solution.
/// It also provides a method for selecting marginalization variables and
/// creating a variable ordering that respects the marginalization constraints.
///
/// The class is refactored from the original IncrementalFixedLagSmoother class
/// in the GTSAM library.
class IncrementalSmoother {
 public:
  using Key = gtsam::Key;
  using KeyVector = gtsam::FastVector<Key>;
  using KeyList = gtsam::FastList<Key>;
  using KeySet = gtsam::KeySet;
  using FactorIndex = gtsam::FactorIndex;
  using FactorIndices = gtsam::FactorIndices;
  using Values = gtsam::Values;
  using VectorValues = gtsam::VectorValues;
  using NonlinearFactor = gtsam::NonlinearFactor;
  using NonlinearFactorPtr = std::shared_ptr<NonlinearFactor>;
  using NonlinearFactorGraph = gtsam::NonlinearFactorGraph;
  using ISAM2 = gtsam::ISAM2;
  using ISAM2Params = gtsam::ISAM2Params;
  using ISAM2Result = gtsam::ISAM2Result;
  using VariableOrderingConstraints = gtsam::FastMap<Key, int>;
  using Matrix = gtsam::Matrix;

  struct Result {
    Result() = default;
    Result(const Result&) = default;
    Result& operator=(const Result&) = default;
    Result(Result&&) = default;
    Result& operator=(Result&&) = default;
    virtual ~Result() = default;

    KeyList marginalized_keys;
    FactorIndices new_marginal_factor_indices;
    FactorIndices marginalized_factor_indices;
    FactorIndices removed_factor_indices;

    template <typename ValueType>
    const ValueType getMarginalizedValue(Key j) const {
      return marginalized_values_.template at<ValueType>(j);
    }

    template <typename FactorType>
    std::shared_ptr<FactorType> getRemovedFactor(FactorIndex i) const {
      return std::dynamic_pointer_cast<FactorType>(removed_factors_.at(i));
    }

    template <typename FactorType>
    std::shared_ptr<FactorType> getMarginalizedFactor(FactorIndex i) const {
      return std::dynamic_pointer_cast<FactorType>(marginalized_factors_.at(i));
    }

    template <typename FactorType>
    std::shared_ptr<FactorType> getNewMarginalFactor(FactorIndex i) const {
      return std::dynamic_pointer_cast<FactorType>(new_marginal_factors_.at(i));
    }

   private:
    friend class IncrementalSmoother;
    std::unordered_map<FactorIndex, NonlinearFactorPtr> removed_factors_;
    std::unordered_map<FactorIndex, NonlinearFactorPtr> marginalized_factors_;
    std::unordered_map<FactorIndex, NonlinearFactorPtr> new_marginal_factors_;
    Values marginalized_values_;
  };

  static const ISAM2Params& defaultISAM2Params();

  explicit IncrementalSmoother(
      const ISAM2Params& params = defaultISAM2Params(),
      AdaptiveMarginalization* marginalizer = nullptr, bool debug = false);

  virtual ~IncrementalSmoother() = default;

  /// Mark variables as temporary, which means they will be marginalized out
  /// during the next updates.
  void markTemporaryVariables(const std::unordered_set<Key>& keys) {
    temporary_keys_.insert(keys.begin(), keys.end());
  }

  /// @brief Updates the solution by integrating new factors, re-linearizing as
  /// necessary, and optionally removing or marginalizing variables.
  ///
  /// This function is responsible for:
  /// - Adding new factors to the factor graph.
  /// - Initializing new variables with provided values.
  /// - Managing marginalization of variables based on custom or default logic.
  ///
  /// @param new_factors        A collection of new factors to be added to the
  /// graph. These factors
  ///                           can involve both existing variables (already in
  ///                           the graph) and/or new variables.
  /// @param new_theta          Initial values for newly introduced variables.
  /// These values should
  ///                           correspond only to variables that are not yet
  ///                           part of the graph.
  /// @param factors_to_remove  (Optional) A list of indices identifying factors
  /// to be removed
  ///                           from the graph during the update. If no factors
  ///                           are to be removed, this parameter can be left
  ///                           empty.
  /// @param force_marginalize_keys
  ///                           (Optional) A set of variable keys to be
  ///                           explicitly marginalized during the update. This
  ///                           overrides the default marginalization logic
  ///                           implemented in @ref
  ///                           selectMarginalizationVariables(). Typically,
  ///                           these keys should correspond to variables that
  ///                           were previously marked as temporary using @ref
  ///                           markTemporaryVariables().
  ///
  /// @return A unique pointer to a `Result` object, which contains metadata
  /// such as marginalized variables, newly added marginal factors, and removed
  /// factors.
  virtual std::unique_ptr<const Result> update(
      const NonlinearFactorGraph& new_factors = NonlinearFactorGraph(),
      const Values& new_theta = Values(),  //
      const FactorIndices& factors_to_remove = FactorIndices(),
      const std::unordered_set<Key>& force_marginalize_keys =
          std::unordered_set<Key>(),
      const std::unordered_set<Key>& no_relinear_keys = {},
      TimeCounter* tc = nullptr);

  /// @brief Run extra isam2 updates until the solution converges.
  /// @param max_iterations The maximum number of iterations to run.
  /// @return The number of iterations run.
  virtual int runExtraUpdatesUntilConverged(int max_iterations = 10);

  /// @returns true if the solution has converged.
  virtual bool isConverged() const {
    return converged_;
  }

  /// Compute an estimate from the incomplete linear delta computed during the
  /// last update. This delta is incomplete because it was not updated below
  /// wildfire_threshold.  If only a single variable is needed, it is faster to
  /// call calculateEstimate(const KEY&).
  Values calculateEstimate() const {
    return isam_.calculateEstimate();
  }

  /// Compute an estimate for a single variable using its incomplete linear
  /// delta computed during the last update.  This is faster than calling the
  /// no-argument version of calculateEstimate, which operates on all variables.
  /// @param key
  /// @return
  template <class VALUE>
  VALUE calculateEstimate(Key key) const {
    return isam_.calculateEstimate<VALUE>(key);
  }

  /// return the current set of iSAM2 parameters
  const ISAM2Params& params() const {
    return isam_.params();
  }

  /// Access the current set of factors
  const NonlinearFactorGraph& getFactors() const {
    return isam_.getFactorsUnsafe();
  }

  /// Access the current linearization point
  const Values& getLinearizationPoint() const {
    return isam_.getLinearizationPoint();
  }

  /// Access the current set of deltas to the linearization point
  const VectorValues& getDelta() const {
    return isam_.getDelta();
  }

  /// Calculate marginal covariance on given variable
  Matrix marginalCovariance(Key key) const {
    return isam_.marginalCovariance(key);
  }

  /// Get results of isam2 in the last update() call.
  const ISAM2Result& getISAM2Result() const {
    return isam_result_;
  }

  /// Get the iSAM2 object which is used for the inference internally
  const ISAM2& getISAM2() const {
    return isam_;
  }

 public:
  /// Print the symbolic Bayes tree of the current iSAM2 object
  virtual void printSymbolicBayesTree(
      const std::string& label = "Bayes tree",
      std::ostream& os = std::cout) const {
    printSymbolicBayesTree(isam_, label, os);
  }

  static void printSymbolicBayesTree(
      const ISAM2& isam, const std::string& label = "Bayes tree",
      std::ostream& os = std::cout);

  template <class KeyContainer>
  static void printKeys(
      const KeyContainer& keys, const std::string& label,
      std::ostream& os = std::cout) {
    os << label;
    for (Key key : keys) {
      os << " " << gtsam::DefaultKeyFormatter(key);
    }
    os << std::endl;
  }

 protected:
  /// @brief Determines which keys to marginalize.
  /// @param force_marginalize_keys A set of variable keys forced to
  /// marginalize.
  /// @param force_keep_keys A set of variable keys that must be kept.
  /// @param already_affected_keys  A set of keys already impacted (e.g., due to
  /// adding or removing factors) before the marginalization process. When the
  /// marginalization variables are selected according to the additional
  /// computational overhead introduced by marginalizing them, knowning what
  /// have already been affected can help estimate the extra overhead more
  /// accurately.
  virtual std::unordered_set<Key> selectMarginalizationVariables(
      const std::unordered_set<Key>& force_marginalize_keys = {},
      const std::unordered_set<Key>& force_keep_keys = {},
      const std::unordered_set<Key>& already_affected_keys = {},
      const std::unordered_set<Key>& no_relinear_keys = {}) const;

  /// @brief  Create a variable ordering that respects the marginalization
  ///         constraints and optimizes the computational overhead.
  /// @param new_factors  The new factors to be added to the iSAM2 object.
  /// @param new_theta     The new values for the new factors.
  /// @param factors_to_remove   The indices of the factors to be removed.
  /// @param marginalizableKeys   The keys that will be marginalized.
  /// @return  The variable ordering constraints.
  virtual VariableOrderingConstraints createOrderingConstraints(
      const NonlinearFactorGraph& new_factors, const Values& new_theta,
      const FactorIndices& factors_to_remove,
      const std::unordered_set<Key>& marginalizableKeys) const;

 protected:
  /// An iSAM2 object used to perform inference.
  ISAM2 isam_;

  /// Store results of latest isam2 update
  ISAM2Result isam_result_;

  AdaptiveMarginalization* marginalizer_;

  /// Store the keys that are currently in the iSAM2 object
  std::unordered_set<Key> cached_keys_;

  /// Store the keys that are marked as temporary
  std::unordered_set<Key> temporary_keys_;

  /// Store the variable ordering used in the last update()
  VariableOrderingConstraints constrained_keys_;

  gtsam::FastList<Key> cached_no_relinear_keys_;

  bool converged_;

  bool debug_;
};

/// @brief  An incremental smoother that supports both persistent variables and
///         stamped variables. The smoother associates each stamped variable
///         with a timestamp. This class mimics the basic interface to the
///         original IncrementalFixedLagSmoother class in the GTSAM library.
class TemporalSmoother : public IncrementalSmoother {
 public:
  using Timestamp = double;
  /// Typedef for a Key-Timestamp map/database
  using KeyTimestampMap = std::unordered_map<Key, Timestamp>;
  using TimestampKeyMap = std::multimap<Timestamp, Key>;

  TemporalSmoother(
      const ISAM2Params& params = defaultISAM2Params(),
      AdaptiveMarginalization* marginalizer = nullptr, const bool debug = false)
      : IncrementalSmoother(params, marginalizer, debug) {}

  ~TemporalSmoother() override = default;

  static inline const Timestamp kPersistentVariableTime = 0.0;

  /// @brief Updates the solution by integrating new factors, re-linearizing as
  /// necessary, and optionally removing or marginalizing variables.
  ///
  /// This function is responsible for:
  /// - Adding new factors to the factor graph.
  /// - Initializing new variables with provided values.
  /// - Managing marginalization of variables based on custom or default logic.
  ///
  /// @param new_factors        A collection of new factors to be added to the
  /// graph. These factors can involve both existing variables (already in the
  /// graph) and/or new variables.
  /// @param new_theta          Initial values for newly introduced variables.
  /// These values should correspond only to variables that are not yet part of
  /// the graph.
  /// @param timestamps         (Optional) A map from keys to real time stamps.
  /// It can be used to set time stamps for new variables, or to update the
  /// time stamps of existing variables. Variables not associated with a
  /// timestamp, or with a non-positive timestamp, will be considered as
  /// persistent.
  /// @param factors_to_remove  (Optional) A list of indices identifying factors
  /// to be removed from the graph during the update. If no factors are to be
  /// removed, this parameter can be left empty.
  /// @param force_marginalize_keys (Optional) A set of variable keys to be
  /// explicitly marginalized during the update. This overrides the default
  /// marginalization logic implemented in @ref
  /// selectMarginalizationVariables(). Typically, these keys should correspond
  /// to variables that were previously marked as temporary using @ref
  /// markTemporaryVariables().
  ///
  /// @return A unique pointer to a `Result` object, which contains metadata
  /// such as marginalized variables, newly added marginal factors, and removed
  /// factors.
  virtual std::unique_ptr<const Result> update(
      const NonlinearFactorGraph& new_factors = NonlinearFactorGraph(),
      const Values& new_theta = Values(),  //
      const KeyTimestampMap& timestamps = KeyTimestampMap(),
      const FactorIndices& factors_to_remove = FactorIndices(),
      const std::unordered_set<Key>& force_marginalize_keys =
          std::unordered_set<Key>(),
      const std::unordered_set<Key>& no_relinear_keys = {},
      TimeCounter* tc = nullptr);

 protected:
  /// @brief  Create a variable ordering that respects the marginalization
  ///         constraints and optimizes the computational overhead.
  /// @param new_factors  The new factors to be added to the iSAM2 object.
  /// @param new_theta     The new values for the new factors.
  /// @param factors_to_remove   The indices of the factors to be removed.
  /// @param marginalizableKeys   The keys that will be marginalized.
  /// @return  The variable ordering constraints.
  VariableOrderingConstraints createOrderingConstraints(
      const NonlinearFactorGraph& new_factors, const Values& new_theta,
      const FactorIndices& factors_to_remove,
      const std::unordered_set<Key>& marginalizableKeys) const override;

 protected:
  /// For a TemporalSmoother, the update function inherited from
  /// IncrementalSmoother should not be used (except for the usage in
  /// derived classes when implementing the update function).
  using IncrementalSmoother::update;

  /// Update the Timestamps associated with the keys
  void updateKeyTimestampMap(const KeyTimestampMap& newTimestamps);

  /// Erase keys from the Key-Timestamps database
  void eraseKeyTimestampMap(const KeyVector& keys);

  /// Find the most recent timestamp of the system
  Timestamp getCurrentTimestamp() const;

  /// Find all of the keys associated with timestamps before the provided time
  KeyVector findKeysBefore(Timestamp timestamp) const;

  /// Find all of the keys associated with timestamps before the provided time
  KeyVector findKeysAfter(Timestamp timestamp) const;

 protected:
  /// The current timestamp associated with each tracked key. Note we only
  /// keep track of stamped variables.
  TimestampKeyMap timestamp_key_map_;
  KeyTimestampMap key_timestamp_map_;
  friend class AdaptiveLagMarginalization;

  // TODO(jeffrey): Is it necessary to keep track of persistent keys?
};

/// A fixed-lag incremental smoother that auotomatically marginalizes out
/// variables that are older than a specified lag.
/// This class mimics the basic interface to the original
/// IncrementalFixedLagSmoother class in the GTSAM library.
class FixedLagSmoother : public TemporalSmoother {
 public:
  using Lag = double;

  FixedLagSmoother(
      Lag smoother_lag = 5.0,
      const ISAM2Params& parameters = defaultISAM2Params(),
      const bool debug = false);

  ~FixedLagSmoother() override = default;

 protected:
  std::unordered_set<Key> selectMarginalizationVariables(
      const std::unordered_set<Key>& force_marginalize_keys = {},
      const std::unordered_set<Key>& force_keep_keys = {},
      const std::unordered_set<Key>& already_affected_keys = {},
      const std::unordered_set<Key>& no_relinear_keys = {}) const override;

 private:
  /// The lag of the smoother. Negative values are interpreted as infinity (i.e.
  /// never marginalize).
  Lag smoother_lag_;
};

class AdaptiveLagMarginalization : public AdaptiveMarginalization {
 public:
  using Key = TemporalSmoother::Key;
  using Timestamp = TemporalSmoother::Timestamp;
  using Lag = double;

  AdaptiveLagMarginalization(
      std::shared_ptr<const Options> options, Lag smoother_lag = 5.0,
      bool debug = false)
      : AdaptiveMarginalization(options, nullptr, debug),
        smoother_(nullptr),
        smoother_lag_(smoother_lag) {}

  ~AdaptiveLagMarginalization() override = default;

  void bindSmoother(const TemporalSmoother* temporal_smoother);

 protected:
  Timestamp marginalTime() const {
    return smoother_lag_ < 0
               ? -1.0
               : (smoother_->getCurrentTimestamp() - smoother_lag_);
  }

  bool isMarginalizable(Key key) const override {
    auto it = smoother_->key_timestamp_map_.find(key);
    if (it == smoother_->key_timestamp_map_.end()) {
      // Variables not in the key_timestamp_map_ are consisdered persistent.
      return false;
    }
    return marginalTime() > it->second;
  }

  /// @brief Overrides the default implementation of
  /// `iterateOverMarginalizableKeys`.
  /// @details Iterates over keys grouped by their associated cliques in
  /// timestamp order, stopping at the marginalization time defined by
  /// `marginalTime()`. The timestamp of each clique is determined by the
  /// earliest timestamp among its frontal variables.
  bool iterateOverMarginalizableKeys(
      const std::function<bool(const Key&)>& f) const override;

  bool iterateOverMarginalizableKeys(
      const std::function<bool(const Key&)>& f,
      std::unordered_set<Key>* visited_keys) const;

 private:
  /// The min guaranteed lag of the smoother. Negative values are interpreted as
  /// infinity (i.e. never marginalize).
  Lag smoother_lag_;

  const TemporalSmoother* smoother_;
};

class AdaptiveLagSmoother : public TemporalSmoother {
 public:
  using Lag = AdaptiveLagMarginalization::Lag;
  using MarginalizationOptions = AdaptiveLagMarginalization::Options;

  AdaptiveLagSmoother(
      const MarginalizationOptions& marg_options, Lag smoother_lag = 5.0,
      const ISAM2Params& parameters = defaultISAM2Params(),
      const bool debug = false);

  ~AdaptiveLagSmoother() override = default;

 private:
  std::unique_ptr<AdaptiveLagMarginalization> adaptive_lag_marginalizer_;
};

}  // namespace sk4slam
