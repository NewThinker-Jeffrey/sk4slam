#include "sk4slam_backends/gtsam/adaptive_marginalization.h"

#include "gtsam_unstable/nonlinear/BayesTreeMarginalizationHelper.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

AdaptiveMarginalization::Options::Options(
    int set_min_num_variables, int set_max_num_variables,
    int set_max_num_factors, int set_max_num_connections,
    double set_marginalization_overhead_threshold,
    double set_marginalization_error_threshold, bool set_aggressive)
    : min_num_variables(set_min_num_variables),
      max_num_variables(set_max_num_variables),
      max_num_factors(set_max_num_factors),
      max_num_connections(set_max_num_connections),
      marginalization_overhead_threshold(
          set_marginalization_overhead_threshold),
      marginalization_error_threshold(set_marginalization_error_threshold),
      aggressive(set_aggressive) {
  ASSERT(min_num_variables > 0);
  ASSERT(max_num_variables < 0 || max_num_variables > min_num_variables);
}

void AdaptiveMarginalization::bindISAM2(const gtsam::ISAM2* isam) {
  ASSERT(isam_ == nullptr || isam_ == isam);
  isam_ = isam;
}

bool AdaptiveMarginalization::isLinearizationFixed(Key key) const {
  ASSERT(user_defined_no_relinear_keys_);
  return isam_->getFixedVariables().count(key) ||
         user_defined_no_relinear_keys_->count(key);
}

void AdaptiveMarginalization::warnForPoorLinearization(
    const std::unordered_set<Key>& keys_to_be_marginalized,
    const std::unordered_set<Key>& forceful_marginalization_keys) const {
  for (const auto& key : keys_to_be_marginalized) {
    bool is_linearization_point_fixed = isLinearizationFixed(key);
    double error = getCachedLinearizationError(key);
    if (error > 1.0) {
      static const char* lin_fixed_msg =
          "This may indicate significant updates occurred after its "
          "linearization point was fixed.";
      static const char* lin_non_fixed_msg =
          "This may indicate forceful marginalization has been triggered, "
          "Otherwise it might be a bug in the adaptive marginalization!";
      LOGW(
          "AdaptiveMarginalization::determineKeysToMarginalize: "
          "Marginalizing poorly linearized %s key: %s with linearization "
          "error %f. %s",
          is_linearization_point_fixed ? "Fixed" : "Non-fixed",
          gtsam::DefaultKeyFormatter(key).c_str(), error,
          is_linearization_point_fixed ? lin_fixed_msg : lin_non_fixed_msg);
      ASSERT(
          is_linearization_point_fixed ||
          forceful_marginalization_keys.count(key));
    }
  }
}

std::unordered_set<gtsam::Key>
AdaptiveMarginalization::determineKeysToMarginalize(
    const std::unordered_set<Key>& force_marginalize_keys,
    const std::unordered_set<Key>& force_keep_keys,
    const std::unordered_set<Key>& already_affected_keys,
    const std::unordered_set<Key>& no_relinear_keys) const {
  user_defined_no_relinear_keys_ = &no_relinear_keys;

  std::unordered_set<Key> keys_to_be_marginalized = force_marginalize_keys;
  std::unordered_set<FactorIndex> factors_to_be_marginalized =
      getFactorsInvolved(keys_to_be_marginalized);
  const gtsam::VariableIndex& key_to_factor_indices = isam_->getVariableIndex();

  if (!options_->aggressive &&
      !needsFurtherMarginalization(
          keys_to_be_marginalized, factors_to_be_marginalized)) {
    warnForPoorLinearization(keys_to_be_marginalized, force_marginalize_keys);
    return keys_to_be_marginalized;
  }

  // Clear cached data for linearization and marginalization errors
  resetCachedLinearizationErrors();
  resetCachedMarginalizationErrors();

  auto keys_to_be_marginalized_str = [&keys_to_be_marginalized]() {
    return toStr(
        std::set<Key>(
            keys_to_be_marginalized.begin(), keys_to_be_marginalized.end()),
        [](const Key& key) { return gtsam::DefaultKeyFormatter(key); });
  };

  // Step 1: Perform smart marginalization
  auto handle_leaf_key = [this, &keys_to_be_marginalized, &force_keep_keys,
                          &factors_to_be_marginalized](const Key& key) {
    if (keys_to_be_marginalized.count(key)) {
      return false;  // Ensure no key is marginalized twice
    }
    if (force_keep_keys.count(key)) {
      return false;
    }
    if (isMarginalizable(key) &&
        getCachedMarginalizationError(key) <
            options_->marginalization_error_threshold) {
      keys_to_be_marginalized.insert(key);
      const auto& affected_factors = isam_->getVariableIndex()[key];
      factors_to_be_marginalized.insert(
          affected_factors.begin(), affected_factors.end());
      if (options_->aggressive ||
          needsFurtherMarginalization(
              keys_to_be_marginalized, factors_to_be_marginalized)) {
        return false;  // Need additional marginalization
      } else {
        return true;  // Marginalization is sufficient
      }
    } else {
      return false;  // Skip this key
    }
  };

  // Iterate over leaf keys and apply smart marginalization
  if (iterateOverLeafKeys(handle_leaf_key, keys_to_be_marginalized) ||
      !needsFurtherMarginalization(
          keys_to_be_marginalized, factors_to_be_marginalized)) {
    LOGD(
        "AdaptiveMarginalization: Smart marginalization completed. "
        "Find %d variables and %d factors to marginalize. Keys = %s",
        keys_to_be_marginalized.size(), factors_to_be_marginalized.size(),
        keys_to_be_marginalized_str().c_str());
    warnForPoorLinearization(keys_to_be_marginalized, force_marginalize_keys);
    return keys_to_be_marginalized;
  }

  // Step 2: Perform balanced marginalization if smart marginalization is
  // insufficient
  LOGW(
      "AdaptiveMarginalization: Smart marginalization insufficient. "
      "Only %d variables and %d factors found. Switching to balanced "
      "marginalization. Keys = %s",
      keys_to_be_marginalized.size(), factors_to_be_marginalized.size(),
      keys_to_be_marginalized_str().c_str());

  initCachedAlreadyAffectedCliques(already_affected_keys);
  if (!force_marginalize_keys.empty()) {
    gatherAdditionalKeysToReEliminateAlongPathToRoot(
        force_marginalize_keys,
        true);  // true for updating the cached already affected cliques
  }

  using KeyErrorPair = std::pair<Key, double>;
  std::priority_queue<
      KeyErrorPair, std::vector<KeyErrorPair>,
      std::function<bool(const KeyErrorPair&, const KeyErrorPair&)>>
      marginalization_error_min_heap(
          [](const KeyErrorPair& a, const KeyErrorPair& b) {
            return a.second > b.second;
          });

  auto handle_balanced_key = [this, &keys_to_be_marginalized, &force_keep_keys,
                              &factors_to_be_marginalized,
                              &key_to_factor_indices,
                              &marginalization_error_min_heap](const Key& key) {
    if (keys_to_be_marginalized.count(key)) {
      return false;  // Skip already marginalized keys
    }
    if (force_keep_keys.count(key)) {
      return false;
    }
    if (key_to_factor_indices.find(key) == key_to_factor_indices.end()) {
      // The key was newly added and not yet processed by ISAM2. Skip it.
      LOGW(
          "AdaptiveMarginalization: Key %s has not yet been added to ISAM2.",
          gtsam::DefaultKeyFormatter(key).c_str());
      return false;
    }

    double marg_error = getCachedMarginalizationError(key);
    if (marg_error < options_->marginalization_error_threshold) {
      bool marginalize = true;
      if (options_->marginalization_overhead_threshold > 0) {
        double marg_overhead = evaluateMarginalizationOverhead(key);
        if (marg_overhead > options_->marginalization_overhead_threshold) {
          marginalize = false;
        }
      }
      if (marginalize) {
        // Add key to marginalization set
        keys_to_be_marginalized.insert(key);
        const auto& affected_factors = isam_->getVariableIndex()[key];
        factors_to_be_marginalized.insert(
            affected_factors.begin(), affected_factors.end());

        // Check if more marginalization is needed
        if (needsFurtherMarginalization(
                keys_to_be_marginalized, factors_to_be_marginalized)) {
          // Update cached already affected cliques for future iterations
          gatherAdditionalKeysToReEliminateAlongPathToRoot(
              {key}, true);  // true for updating the cached already
                             // affected cliques
          return false;
        } else {
          return true;  // Marginalization is sufficient
        }
      }
    }
    // Add key to heap for potential forceful marginalization
    ASSERT(keys_to_be_marginalized.count(key) == 0);
    marginalization_error_min_heap.emplace(key, marg_error);
    return false;
  };

  if (iterateOverMarginalizableKeys(handle_balanced_key)) {
    LOGW(
        "AdaptiveMarginalization: Balanced marginalization completed. "
        "Find %d variables and %d factors to marginalize. Keys = %s",
        keys_to_be_marginalized.size(), factors_to_be_marginalized.size(),
        keys_to_be_marginalized_str().c_str());
    warnForPoorLinearization(keys_to_be_marginalized, force_marginalize_keys);
    return keys_to_be_marginalized;
  }

  // Step 3: Perform forceful marginalization if balanced marginalization is
  // insufficient
  LOGW(
      "AdaptiveMarginalization: Balanced marginalization insufficient. "
      "Only %d variables and %d factors found. Switching to forceful "
      "marginalization. Keys = %s",
      keys_to_be_marginalized.size(), factors_to_be_marginalized.size(),
      keys_to_be_marginalized_str().c_str());

  double last_marg_error = 0.0;
  std::unordered_set<Key> forceful_marginalization_keys =
      force_marginalize_keys;
  while (!marginalization_error_min_heap.empty()) {
    const auto& top = marginalization_error_min_heap.top();
    const Key& key = top.first;
    const double& marg_error = top.second;
    if (forceful_marginalization_keys.count(key) != 0) {
      LOGE(
          "AdaptiveMarginalization: The key %s has been found in the "
          "min-heap twice! This should not happen! It's likely that "
          "the function iterateOverMarginalizableKeys() repeatedly "
          "visited that key, which is a SOFT BUG!",
          gtsam::DefaultKeyFormatter(key).c_str());
      // ASSERT(forceful_marginalization_keys.count(key) == 0);
      marginalization_error_min_heap.pop();
      continue;
    }
    ASSERT(keys_to_be_marginalized.count(key) == 0);
    ASSERT(marg_error >= last_marg_error);
    last_marg_error = marg_error;
    forceful_marginalization_keys.insert(key);
    keys_to_be_marginalized.insert(key);
    marginalization_error_min_heap.pop();
  }
  LOGW(
      "AdaptiveMarginalization: Forceful marginalization completed. "
      "Find %d variables and %d factors to marginalize. Last marginalization "
      "error: %f. Keys = %s",
      keys_to_be_marginalized.size(), factors_to_be_marginalized.size(),
      last_marg_error, keys_to_be_marginalized_str().c_str());
  warnForPoorLinearization(
      keys_to_be_marginalized, forceful_marginalization_keys);
  return keys_to_be_marginalized;
}

bool AdaptiveMarginalization::needsFurtherMarginalization(
    const std::unordered_set<Key>& assumed_marginalized_keys,
    const std::unordered_set<FactorIndex>& assumed_marginalized_factors) const {
  const gtsam::VariableIndex& key_to_factor_indices = isam_->getVariableIndex();
  int rest_keys =
      key_to_factor_indices.size() - assumed_marginalized_keys.size();
  int rest_factors =
      key_to_factor_indices.nFactors() - assumed_marginalized_factors.size();
  int rest_connections = -1;

  if (debug_) {
    rest_connections = key_to_factor_indices.nEntries() -
                       countConnections(assumed_marginalized_factors);
  }

  bool needs_marginalization = false;
  if (rest_keys > options_->min_num_variables) {
    if (options_->max_num_variables > 0 &&
        rest_keys > options_->max_num_variables) {
      needs_marginalization = true;
    } else if (
        options_->max_num_factors > 0 &&
        rest_factors > options_->max_num_factors) {
      needs_marginalization = true;
    } else if (options_->max_num_connections > 0) {
      if (rest_connections < 0) {
        rest_connections = key_to_factor_indices.nEntries() -
                           countConnections(assumed_marginalized_factors);
      }
      if (rest_connections > options_->max_num_connections) {
        needs_marginalization = true;
      }
    }
  }

  if (debug_) {
    LOGI(
        "AdaptiveMarginalization: needsFurtherMarginalization = %d, "
        "rest_keys = %d, rest_factors = %d, rest_connections = %d",
        needs_marginalization, rest_keys, rest_factors, rest_connections);
  }
  return needs_marginalization;
}

double AdaptiveMarginalization::evaluateLinearizationError(
    const Key& key) const {
  using gtsam::Symbol;
  using gtsam::Vector;
  using ThresholdByVarType = gtsam::FastMap<char, Vector>;
  const Vector& delta = isam_->getDelta().at(key);
  bool is_linearization_point_fixed = false;
  if (isLinearizationFixed(key)) {
    is_linearization_point_fixed = true;
  }

  const auto& threshold = isam_->params().relinearizeThreshold;
  double error;
  if (std::holds_alternative<double>(threshold)) {
    double cur_threshold = std::get<double>(threshold);
    double max_delta = delta.lpNorm<Eigen::Infinity>();
    error = max_delta / cur_threshold;
    if (debug_) {
      LOGI(
          "AdaptiveMarginalization: %s linearization error of key %s: %f",
          is_linearization_point_fixed ? "Fixed" : "Non-fixed",
          gtsam::DefaultKeyFormatter(key).c_str(), error);
    }
  } else if (std::holds_alternative<ThresholdByVarType>(threshold)) {
    // Find the threshold for this variable type
    const Vector& cur_threshold =
        std::get<ThresholdByVarType>(threshold).at(Symbol(key).chr());
    // Verify the threshold vector matches the actual variable size
    if (cur_threshold.rows() != delta.rows()) {
      throw std::invalid_argument(
          "AdaptiveMarginalization failed to evaluate linearization error "
          "for the variable '" +
          std::string(1, Symbol(key).chr()) +
          "' since its dimension does not match the threshold vector passed "
          "to ISAM2Params");
    }
    error = (delta.array() / cur_threshold.array()).cwiseAbs().maxCoeff();
    if (debug_) {
      LOGI(
          "AdaptiveMarginalization: %s linearization error of key %s: %f",
          is_linearization_point_fixed ? "Fixed" : "Non-fixed",
          gtsam::DefaultKeyFormatter(key).c_str(), error);
    }
  } else {
    throw std::runtime_error("Unknown relinearization threshold type");
  }

  return error;
  // return is_linearization_point_fixed ? 0 : error;
}

double AdaptiveMarginalization::evaluateMarginalizationError(
    const Key& key) const {
  if (!isam_->params().enableRelinearization) {
    // return 0.0;  // No variable can be relinearized, so maginalizing
    //              // variables will not introduce any additional
    //              // linearization error.

    // Though we can return immediately, we still do the rest of the
    // computation for debugging purposes.
  }

  const gtsam::VariableIndex& key_to_factor_indices = isam_->getVariableIndex();
  const gtsam::NonlinearFactorGraph& underlying_factor_graph =
      isam_->getFactorsUnsafe();
  std::unordered_set<Key> involved_keys;
  involved_keys.insert(key);
  for (const FactorIndex& factor_index : key_to_factor_indices[key]) {
    auto factor = underlying_factor_graph.at(factor_index);
    const gtsam::KeyVector& keys = factor->keys();
    involved_keys.insert(keys.begin(), keys.end());
  }

  // Identify the variable with the largest linearization error among the
  // involved variables. Note: Variables whose linearization points are already
  // fixed will no longer introduce additional linearization errors. Therefore,
  // such variables are considered to have a linearization error of 0.
  double marg_error = 0.0;
  Key marg_error_key = *std::max_element(
      involved_keys.begin(), involved_keys.end(),
      [this](const Key& a, const Key& b) {
        bool is_linearization_point_fixed_a = isLinearizationFixed(a);
        bool is_linearization_point_fixed_b = isLinearizationFixed(b);
        if (is_linearization_point_fixed_a == is_linearization_point_fixed_b) {
          return getCachedLinearizationError(a) <
                 getCachedLinearizationError(b);
        } else {
          return is_linearization_point_fixed_a;
        }
      });
  if (isam_->params().enableRelinearization &&
      !isLinearizationFixed(marg_error_key)) {
    marg_error = getCachedLinearizationError(marg_error_key);
  }

  if (debug_) {
    LOGI(
        "AdaptiveMarginalization: marginalization error of key %s: %f (from "
        "%s)",
        gtsam::DefaultKeyFormatter(key).c_str(), marg_error,
        gtsam::DefaultKeyFormatter(marg_error_key).c_str());
  }
  return marg_error;
}

std::unordered_set<const AdaptiveMarginalization::Clique*>
AdaptiveMarginalization::gatherCliquesAlongPathToRoot(
    const std::unordered_set<const Clique*>& source_cliques,
    const std::unordered_set<const Clique*>& extra_terminate_cliques) const {
  // Find all cliques that are on the path from any source clique to the root of
  // the Bayes tree
  std::unordered_set<const Clique*> cliques;
  for (const Clique* clique : source_cliques) {
    if (extra_terminate_cliques.count(clique)) {
      continue;
    }
    // ASSERT(extra_terminate_cliques.count(clique) == 0);

    while (cliques.insert(clique).second && !clique->isRoot()) {
      clique = clique->parent().get();
      if (extra_terminate_cliques.count(clique)) {
        break;
      }
    }
  }
  return cliques;
}

std::unordered_set<AdaptiveMarginalization::Key>
AdaptiveMarginalization::gatherAdditionalKeysToReEliminateAlongPathToRoot(
    const std::unordered_set<Key>& keys_to_marginalize,
    bool update_cached_already_affected_cliques) const {
  // Find all cliques that are on the path from any clique containing a key in
  // the source_keys to the root of the Bayes tree
  std::unordered_set<const Clique*> source_cliques =
      gatherAdditionalCliquesToReEliminate(
          *isam_, toKeyVector(keys_to_marginalize));

  std::unordered_set<const Clique*> cliques = gatherCliquesAlongPathToRoot(
      source_cliques, cached_already_affected_cliques_);

  // Update the cached already_affected_cliques if required
  if (update_cached_already_affected_cliques) {
    updateCachedAlreadyAffectedCliques(cliques);
  }

  // Gather all keys that are in the cliques
  std::unordered_set<Key> additional_keys_to_reeliminate;
  for (const Clique* clique : cliques) {
    auto& conditional = clique->conditional();
    for (const Key& key : conditional->frontals()) {
      additional_keys_to_reeliminate.insert(key);
    }
  }
  return additional_keys_to_reeliminate;
}

double AdaptiveMarginalization::evaluateMarginalizationOverhead(
    const Key& key) const {
  std::unordered_set<Key> additional_keys_to_reeliminate =
      gatherAdditionalKeysToReEliminateAlongPathToRoot({key});

  std::unordered_set<FactorIndex> additional_factor_indices =
      getFactorsInvolved(additional_keys_to_reeliminate);

  double overhead = 0;
  const gtsam::NonlinearFactorGraph& underlying_factor_graph =
      isam_->getFactorsUnsafe();
  for (const FactorIndex& factor_index : additional_factor_indices) {
    auto factor = underlying_factor_graph.at(factor_index);
    int rows = factor->dim();
    int cols = 0;
    for (const Key& ikey : factor->keys()) {
      cols += isam_->getDelta().at(ikey).rows();
    }
    ASSERT(rows > 0);
    ASSERT(cols > 0);
    overhead += rows * cols * cols;
  }
  if (debug_) {
    LOGI(
        "AdaptiveMarginalization: marginalization overhead of key %s: %f",
        gtsam::DefaultKeyFormatter(key).c_str(), overhead);
  }
  return overhead;
}

bool AdaptiveMarginalization::iterateOverLeafKeysInSubtree(
    const gtsam::ISAM2Clique::shared_ptr& subtree_root,
    const std::function<bool(const Key&)>& f,
    std::unordered_set<Key>* dependencies,
    const std::unordered_set<Key>& assumed_marginalized_keys) {
  for (const sharedClique& child : subtree_root->children) {
    if (iterateOverLeafKeysInSubtree(
            child, f, dependencies, assumed_marginalized_keys)) {
      return true;  // if f returns true, stop iteration
    }
  }
  auto& conditional = subtree_root->conditional();
  bool all_frontals_marginalized = true;
  for (Key key : conditional->frontals()) {
    if (assumed_marginalized_keys.count(key)) {
      continue;
    }
    if (dependencies->count(key)) {
      all_frontals_marginalized = false;
      break;
    }
    if (f(key)) {
      return true;  // if f returns true, stop iteration
    }

    if (assumed_marginalized_keys.count(key)) {
      // The key was added to assumed_marginalized_keys during the call to
      // f(key).
      continue;
    } else {
      all_frontals_marginalized = false;
      break;
    }
  }

  if (!all_frontals_marginalized) {
    dependencies->insert(
        conditional->beginParents(), conditional->endParents());
  }
  return false;
}

bool AdaptiveMarginalization::iterateOverKeysInSubtree(
    const gtsam::ISAM2Clique::shared_ptr& subtree_root,
    const std::function<bool(const Key&)>& f) {
  for (const sharedClique& child : subtree_root->children) {
    if (iterateOverKeysInSubtree(child, f)) {
      return true;  // if f returns true, stop iteration
    }
  }

  auto& conditional = subtree_root->conditional();
  for (Key key : conditional->frontals()) {
    if (f(key)) {
      return true;  // if f returns true, stop iteration
    }
  }
  return false;
}

bool AdaptiveMarginalization::iterateOverMarginalizableKeys(
    const std::function<bool(const Key&)>& f) const {
  std::function<bool(const Key&)> f_wrapped = [this, &f](const Key& key) {
    return isMarginalizable(key) ? f(key) : false;
  };
  for (const sharedClique& root : isam_->roots()) {
    std::vector<sharedClique> leaf_cliques;
    if (iterateOverKeysInSubtree(root, f_wrapped)) {
      return true;  // Stop iteration if function f returns true
    }
  }
  return false;
}

bool AdaptiveMarginalization::iterateOverLeafKeys(
    const std::function<bool(const Key&)>& f,
    const std::unordered_set<Key>& assumed_marginalized_keys) const {
  std::unordered_set<Key> dependencies;
  for (const sharedClique& root : isam_->roots()) {
    std::vector<sharedClique> leaf_cliques;
    if (iterateOverLeafKeysInSubtree(
            root, f, &dependencies, assumed_marginalized_keys)) {
      return true;  // Stop iteration if function f returns true
    }
  }
  return false;
}

double AdaptiveMarginalization::getCachedLinearizationError(
    const Key& key) const {
  auto cache_it = cached_linearization_errors_.find(key);
  if (cache_it != cached_linearization_errors_.end()) {
    return cache_it->second;
  } else {
    double linearization_error = evaluateLinearizationError(key);
    cached_linearization_errors_[key] = linearization_error;
    return linearization_error;
  }
}

void AdaptiveMarginalization::resetCachedLinearizationErrors() const {
  cached_linearization_errors_.clear();
}

void AdaptiveMarginalization::testLinearizationErrors()
    const {  // For debugging only
  int n_relinearize = 0;
  std::function<bool(const Key&)> f = [this, &n_relinearize](const Key& key) {
    if (evaluateLinearizationError(key) > 1.0) {
      ++n_relinearize;
    }
    return false;
  };
  for (const sharedClique& root : isam_->roots()) {
    std::vector<sharedClique> leaf_cliques;
    iterateOverKeysInSubtree(root, f);
  }
  LOGD(
      "AdptiveMarginalization: testLinearizationErrors: %d variables need "
      "relinearization",
      n_relinearize);
}

double AdaptiveMarginalization::getCachedMarginalizationError(
    const Key& key) const {
  auto cache_it = cached_marginaization_errors_.find(key);
  if (cache_it != cached_marginaization_errors_.end()) {
    return cache_it->second;
  } else {
    double marginalization_error = evaluateMarginalizationError(key);
    cached_marginaization_errors_[key] = marginalization_error;
    return marginalization_error;
  }
}

void AdaptiveMarginalization::resetCachedMarginalizationErrors() const {
  cached_marginaization_errors_.clear();
}

void AdaptiveMarginalization::initCachedAlreadyAffectedCliques(
    const std::unordered_set<Key>& already_affected_keys) const {
  cached_already_affected_cliques_.clear();
  std::unordered_set<const Clique*> source_cliques;
  for (Key key : already_affected_keys) {
    source_cliques.insert((*isam_)[key].get());
  }
  std::unordered_set<const Clique*> new_affected_cliques =
      gatherCliquesAlongPathToRoot(
          source_cliques, cached_already_affected_cliques_);
  updateCachedAlreadyAffectedCliques(new_affected_cliques);
}

void AdaptiveMarginalization::updateCachedAlreadyAffectedCliques(
    const std::unordered_set<const Clique*>& new_affected_cliques) const {
  cached_already_affected_cliques_.insert(
      new_affected_cliques.begin(), new_affected_cliques.end());
}

}  // namespace sk4slam
