#include "sk4slam_backends/gtsam/incremental_smoother.h"

#include "gtsam_unstable/nonlinear/BayesTreeMarginalizationHelper.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

namespace {
void printSymbolicTreeHelper(
    const gtsam::ISAM2Clique::shared_ptr& clique, const std::string indent,
    std::ostream& os) {
  using gtsam::ISAM2Clique;
  using gtsam::Key;

  // Print the current clique
  os << indent << "P( ";
  for (Key key : clique->conditional()->frontals()) {
    os << gtsam::DefaultKeyFormatter(key) << " ";
  }
  if (clique->conditional()->nrParents() > 0)
    os << "| ";
  for (Key key : clique->conditional()->parents()) {
    os << gtsam::DefaultKeyFormatter(key) << " ";
  }
  os << ")" << std::endl;

  // Recursively print all of the children
  for (const ISAM2Clique::shared_ptr& child : clique->children) {
    printSymbolicTreeHelper(child, indent + " ", os);
  }
}

std::unordered_set<gtsam::Key> removeAlreadyAffectedKeys(
    const gtsam::ISAM2& isam,
    const std::unordered_set<gtsam::Key>& additional_keys,
    const std::unordered_set<gtsam::Key>& affected_keys) {
  using Key = gtsam::Key;
  using Clique = gtsam::ISAM2::Clique;
  std::unordered_set<const Clique*> already_affected_cliques;
  // Find the cliques that contain the affected keys
  std::unordered_set<const Clique*> source_cliques;
  for (Key key : affected_keys) {
    source_cliques.insert(isam[key].get());
  }
  // Gather all cliques that are on the path from any source clique to
  // the root of the Bayes tree
  for (const Clique* clique : source_cliques) {
    while (already_affected_cliques.insert(clique).second &&
           !clique->isRoot()) {
      clique = clique->parent().get();
    }
  }
  // Gather all keys in the affected cliques
  std::unordered_set<Key> keys_affected_before_marginalization;
  for (const Clique* clique : already_affected_cliques) {
    for (const Key& key : clique->conditional()->frontals()) {
      keys_affected_before_marginalization.insert(key);
    }
  }
  // Remove those already affected keys from the additional keys and print
  // the remaining
  std::unordered_set<Key> filtered_additional_keys;
  for (const Key& key : additional_keys) {
    if (!keys_affected_before_marginalization.count(key)) {
      filtered_additional_keys.insert(key);
    }
  }
  return filtered_additional_keys;
}

void debugMarginalizationReEliminationOverhead(
    const gtsam::ISAM2& isam,
    const std::unordered_set<gtsam::Key>& additional_keys,
    const std::unordered_set<gtsam::Key>& affected_keys) {
  using Key = gtsam::Key;

  // Remove those already affected keys from the additional keys and print
  // the remaining
  std::unordered_set<Key> filtered_additional_keys =
      removeAlreadyAffectedKeys(isam, additional_keys, affected_keys);
  std::set<Key> ordered_additional_keys(
      filtered_additional_keys.begin(), filtered_additional_keys.end());
  ASSERT(ordered_additional_keys.size() == filtered_additional_keys.size());
  LOGI(
      "IncrementalSmoother: Additional Keys to Re-Eliminate for "
      "Marginalization [%lu / %lu before-removing-already-affected]: %s",
      ordered_additional_keys.size(), additional_keys.size(),
      toStr(ordered_additional_keys, [](const Key& key) {
        return gtsam::DefaultKeyFormatter(key);
      }).c_str());
}
}  // namespace

const IncrementalSmoother::ISAM2Params&
IncrementalSmoother::defaultISAM2Params() {
  static auto make_default_params = []() {
    ISAM2Params params;
    params.findUnusedFactorSlots = true;
    params.relinearizeSkip = 1;
    // params.factorization = ISAM2Params::QR;
    return params;
  };
  static ISAM2Params default_params = make_default_params();
  return default_params;
}

IncrementalSmoother::IncrementalSmoother(
    const ISAM2Params& params, AdaptiveMarginalization* marginalizer,
    bool debug)
    : isam_(params),
      marginalizer_(marginalizer),
      debug_(debug),
      converged_(false) {
  if (marginalizer_) {
    marginalizer_->bindISAM2(&isam_);
  }
}

std::unordered_set<IncrementalSmoother::Key>
IncrementalSmoother::selectMarginalizationVariables(
    const std::unordered_set<Key>& force_marginalize_keys,
    const std::unordered_set<Key>& force_keep_keys,
    const std::unordered_set<Key>& already_affected_keys,
    const std::unordered_set<Key>& no_relinear_keys) const {
  // AdaptiveMarginalization::testLinearizationErrors();
  if (marginalizer_) {
    return marginalizer_->determineKeysToMarginalize(
        force_marginalize_keys, force_keep_keys, already_affected_keys,
        no_relinear_keys);
  } else {
    return {};
  }
}

void IncrementalSmoother::printSymbolicBayesTree(
    const ISAM2& isam, const std::string& label, std::ostream& os) {
  os << label << std::endl;
  if (!isam.roots().empty()) {
    for (const ISAM2::sharedClique& root : isam.roots()) {
      printSymbolicTreeHelper(root, "", os);
    }
  } else {
    os << "{Empty Tree}" << std::endl;
  }
}

IncrementalSmoother::VariableOrderingConstraints
IncrementalSmoother::createOrderingConstraints(
    const NonlinearFactorGraph& new_factors, const Values& new_theta,
    const FactorIndices& factors_to_remove,
    const std::unordered_set<Key>& marginalizable_keys) const {
  gtsam::FastMap<gtsam::Key, int> constrained_keys;
  // Generate ordering constraints so that the marginalizable variables will
  // be eliminated first. Set all variables to Group1
  for (const gtsam::Key& key : cached_keys_) {
    constrained_keys[key] = 1;
  }

  int new_factor_group = 2;
  for (const gtsam::Key& key : new_factors.keys()) {
    constrained_keys[key] = new_factor_group;
  }

  int new_theta_group = 3;
  for (const auto& item : new_theta) {
    const gtsam::Key& new_key = item.key;
    constrained_keys[new_key] = new_theta_group;
  }

  // Set marginalizable variables to Group0
  for (gtsam::Key key : marginalizable_keys) {
    constrained_keys[key] = 0;
  }
  return constrained_keys;
}

int IncrementalSmoother::runExtraUpdatesUntilConverged(int max_iterations) {
  const bool debug = debug_;

  if (converged_ || getISAM2Result().variablesRelinearized == 0) {
    // Already converged in the last update() or runExtraUpdatesUntilConverged()
    // call.
    converged_ = true;
    return 0;
  }

  // Ensure we're using consistent variable ordering constraints
  // with that in the last update() call.
  VariableOrderingConstraints constrained_keys;
  if (!constrained_keys_.empty()) {
    // Some varaibles in constrained_keys_ may have been removed (marginalized)
    // since the last update() call. We need to copy over the ones that are
    // still in the graph.
    for (const Key& key : cached_keys_) {
      constrained_keys[key] = constrained_keys_.at(key);
    }
  }

  auto log_result = [](const ISAM2Result& isam_result, int i) {
    isam_result.print(formatStr(
        "IncrementalSmoother::runExtraUpdatesUntilConverged(): isam result "
        "(iteration %d): ",
        i + 1));
  };

  if (debug) {
    log_result(getISAM2Result(), 0);
  }

  for (int i = 0; i < max_iterations; ++i) {
    // ISAM2Result isam_result = isam_.update();
    ISAM2Result isam_result = isam_.update(
        {}, {}, {}, constrained_keys, cached_no_relinear_keys_, {});
    if (debug) {
      log_result(isam_result, i);
    }

    // Check for convergence: stop if no variables were relinearized.
    if (isam_result.variablesRelinearized == 0) {
      converged_ = true;
      return i + 1;
    }
  }

  converged_ = false;
  return max_iterations;
}

std::unique_ptr<const IncrementalSmoother::Result> IncrementalSmoother::update(
    const NonlinearFactorGraph& new_factors, const Values& new_theta,
    const FactorIndices& factors_to_remove,
    const std::unordered_set<Key>& force_marginalize_keys,
    const std::unordered_set<Key>& no_relinear_keys, TimeCounter* tc) {
  const bool debug = debug_;
  auto result = std::make_unique<Result>();

  if (debug) {
    std::cout << "IncrementalSmoother::update() Start" << std::endl;
    printSymbolicBayesTree("Bayes Tree Before Update:");
    std::cout << "END" << std::endl;
  }
  // Gather affected old keys from new factors
  std::unordered_set<Key> affected_keys;
  std::unordered_set<Key> keys_in_new_factors;
  for (const auto& item : new_factors) {
    ASSERT(item);
    keys_in_new_factors.insert(item->keys().begin(), item->keys().end());
  }
  for (const auto& item : new_theta) {
    const Key& new_key = item.key;
    keys_in_new_factors.erase(
        new_key);  // ensure affected_keys does not contain new keys
  }
  affected_keys.insert(keys_in_new_factors.begin(), keys_in_new_factors.end());

  // Gather removed_factors and affected_keys from removed factors
  const NonlinearFactorGraph& underlying_factor_graph = getFactors();
  if (factors_to_remove.size() > 0) {
    for (const FactorIndex& factor_index : factors_to_remove) {
      const auto& factor = underlying_factor_graph.at(factor_index);
      affected_keys.insert(factor->keys().begin(), factor->keys().end());
      result->removed_factors_[factor_index] = factor;
    }
  }

  // Additional calls to `isam_.update()` can improve the linearization accuracy
  // of variables before selecting marginalization candidates.
  // By default, automatic extra updates are skipped, giving users full control
  // over the update process.
  //
  // For optional refinement, users can invoke `runExtraUpdatesUntilConverged()`
  // manually after each `update()` call to ensure relinearization convergence.
  //
  // runExtraUpdatesUntilConverged();

  // Select the keys to marginalize.
  const auto& force_keep_keys = keys_in_new_factors;
  std::unordered_set<Key> marginalizable_keys = selectMarginalizationVariables(
      force_marginalize_keys, force_keep_keys, affected_keys, no_relinear_keys);

  if (debug) {
    printKeys(marginalizable_keys, "Marginalizable Keys: ");
  }
  if (tc) {
    tc->tag("marginalizableKeysSelected");
  }

  // Add the new keys to cached_keys_, which are needed in the following
  // `createOrderingConstraints()` call.
  for (const auto& item : new_theta) {
    ASSERT(cached_keys_.insert(item.key).second);
  }

  // Force iSAM2 to put the marginalizable variables and temporary keys at the
  // beginning
  std::unordered_set<Key> group0_keys(
      marginalizable_keys.begin(), marginalizable_keys.end());
  group0_keys.insert(temporary_keys_.begin(), temporary_keys_.end());
  VariableOrderingConstraints constrained_keys = createOrderingConstraints(
      new_factors, new_theta, factors_to_remove, group0_keys);

  if (debug) {
    std::cout << "Constrained Keys: ";
    const auto& variableIndex = isam_.getVariableIndex();
    std::set<Key> group_zero;
    if (constrained_keys.size() > 0) {
      for (VariableOrderingConstraints::const_iterator iter =
               constrained_keys.begin();
           iter != constrained_keys.end(); ++iter) {
        if (iter->second == 0) {
          group_zero.insert(iter->first);
        }
        std::cout << gtsam::DefaultKeyFormatter(iter->first) << "("
                  << iter->second << ")  ";
        ASSERT(
            new_theta.exists(iter->first) ||
            variableIndex.find(iter->first) != variableIndex.end());
      }
    }
    std::cout << std::endl;
    printKeys(group_zero, "Group 0: ");
  }

  if (tc) {
    tc->tag("OrderConstrained");
  }

  // Mark additional keys to re-eliminate due to marginalization
  std::unordered_set<Key> additional_keys =
      AdaptiveMarginalization::gatherAdditionalKeysToReEliminate(
          isam_, marginalizable_keys);
  bool debug_marginalization_overhead = true;
  if (debug_marginalization_overhead && !marginalizable_keys.empty()) {
    // Remove those already affected keys from the additional keys and print
    // the remaining
    debugMarginalizationReEliminationOverhead(
        isam_, additional_keys, affected_keys);
  }
  KeyList additional_marked_keys(
      additional_keys.begin(), additional_keys.end());
  if (tc) {
    tc->tag("additionalKeysToReEliminateGathered");
  }

  // Gather no_relinear_keys
  cached_no_relinear_keys_.clear();
  for (const Key& key : no_relinear_keys) {
    if (cached_keys_.count(key)) {
      cached_no_relinear_keys_.push_back(key);
    }
  }
  if (debug) {
    if (!force_marginalize_keys.empty()) {
      std::set<Key> ordered_force_marg_keys(
          force_marginalize_keys.begin(), force_marginalize_keys.end());
      printKeys(ordered_force_marg_keys, "Force Marginalize Keys: ");
    }
    if (!cached_no_relinear_keys_.empty()) {
      printKeys(cached_no_relinear_keys_, "No Relinearization Keys: ");
    }
    std::cout << "Relinearizable Keys: ";
    std::set<Key> ordered_cached_keys(cached_keys_.begin(), cached_keys_.end());
    for (const Key& key : ordered_cached_keys) {
      if (!isam_.getFixedVariables().count(key) &&
          !no_relinear_keys.count(key)) {
        std::cout << gtsam::DefaultKeyFormatter(key) << "  ";
      }
    }
    std::cout << std::endl;
  }

  // Update iSAM2
  std::swap(constrained_keys_, constrained_keys);
  converged_ = false;
  try {
    isam_result_ = isam_.update(
        new_factors, new_theta, factors_to_remove, constrained_keys_,
        cached_no_relinear_keys_, additional_marked_keys);
  } catch (...) {
    const auto& variableIndex = isam_.getVariableIndex();
    if (constrained_keys_.size() > 0) {
      for (VariableOrderingConstraints::const_iterator iter =
               constrained_keys_.begin();
           iter != constrained_keys_.end(); ++iter) {
        if (variableIndex.find(iter->first) == variableIndex.end()) {
          std::cout << "Constrained key "
                    << gtsam::DefaultKeyFormatter(iter->first)
                    << " does not exist in iSAM2!" << std::endl;
        }
      }
    }
    throw;
  }

  result->removed_factor_indices = std::move(factors_to_remove);

  if (debug) {
    isam_result_.print("IncrementalSmoother::update(): isam result: ");
    printSymbolicBayesTree("Bayes Tree After Update, Before Marginalization:");
    std::cout << "END" << std::endl;
  }
  if (tc) {
    tc->tag("isam2Updated");
  }

  // Marginalize out any needed variables
  if (marginalizable_keys.size() > 0) {
    // Cache values for the variables that will be marginalized
    for (const Key& key : marginalizable_keys) {
      result->marginalized_values_.insert(key, isam_.calculateEstimate(key));
    }
    ASSERT(result->marginalized_values_.size() == marginalizable_keys.size());

    // Get the factors that will be marginalized
    const gtsam::VariableIndex& key_to_factor_indices =
        isam_.getVariableIndex();
    std::unordered_map<FactorIndex, NonlinearFactorPtr>
        predicted_marginalized_factors;
    for (const Key& key : marginalizable_keys) {
      for (const FactorIndex& factor_index : key_to_factor_indices[key]) {
        auto it = predicted_marginalized_factors.find(factor_index);
        if (it == predicted_marginalized_factors.end()) {
          auto factor = underlying_factor_graph.at(factor_index);
          ASSERT(factor != nullptr);
          predicted_marginalized_factors[factor_index] = std::move(factor);
        }
      }
    }
    if (tc) {
      tc->tag("marginalizableValuesCached");
    }

    // Marginalize
    KeyList leaf_keys(marginalizable_keys.begin(), marginalizable_keys.end());
    FactorIndices marginal_factor_indices, marginalized_factor_indices;
    isam_.marginalizeLeaves(
        leaf_keys, &marginal_factor_indices, &marginalized_factor_indices);
    if (tc) {
      tc->tag("marginalized");
    }

    ASSERT(
        predicted_marginalized_factors.size() ==
        marginalized_factor_indices.size());
    result->marginalized_factors_ = std::move(predicted_marginalized_factors);
    for (const FactorIndex& factor_index : marginal_factor_indices) {
      result->new_marginal_factors_[factor_index] =
          underlying_factor_graph.at(factor_index);
    }
    result->marginalized_keys = std::move(leaf_keys);
    ASSERT(
        result->marginalized_keys.size() ==
        result->marginalized_values_.size());
    result->new_marginal_factor_indices = std::move(marginal_factor_indices);
    result->marginalized_factor_indices =
        std::move(marginalized_factor_indices);

    for (const Key& key : result->marginalized_keys) {
      cached_keys_.erase(key);
      temporary_keys_.erase(key);
    }
    if (tc) {
      tc->tag("marginalizationRecorded");
    }
  }

  if (debug) {
    printSymbolicBayesTree("Final Bayes Tree:");
    std::cout << "END" << std::endl;
  }

  if (debug) {
    std::cout << "IncrementalSmoother::update() Finish" << std::endl;
  }

  return result;
}

void TemporalSmoother::updateKeyTimestampMap(
    const KeyTimestampMap& timestamps) {
  // Loop through each key and add/update it in the map
  for (const auto& key_timestamp : timestamps) {
    // Check to see if this key already exists in the database
    KeyTimestampMap::iterator key_iter =
        key_timestamp_map_.find(key_timestamp.first);

    // If the key already exists
    if (key_iter != key_timestamp_map_.end()) {
      // Find the entry in the Timestamp-Key database
#if 0  // Old implementation (slow, but works)
      std::pair<TimestampKeyMap::iterator, TimestampKeyMap::iterator> range =
          timestamp_key_map_.equal_range(key_iter->second);
      TimestampKeyMap::iterator time_iter = range.first;
      while (time_iter->second != key_timestamp.first) {
        ++time_iter;
      }
#else  // New implementation (not tested)
      TimestampKeyMap::iterator time_iter =
          timestamp_key_map_.lower_bound(key_iter->second);
      ASSERT(time_iter != timestamp_key_map_.end());
      ASSERT(time_iter->first == key_iter->second);
      // Traverse the equal range in the Timestamp-Key database to find the
      // entry for this key
      std::unordered_set<Key> keys_to_update_in_adavance;
      while (time_iter->second != key_timestamp.first) {
        auto tmp_it = timestamps.find(time_iter->second);
        if (tmp_it != timestamps.end() && tmp_it->second != time_iter->first) {
          // If any key that needs to update its timestamp is found in the
          // timestamp-key map, we remove it in advance to avoid repeated
          // traversal of the same equal range.
          keys_to_update_in_adavance.insert(time_iter->second);
          time_iter = timestamp_key_map_.erase(time_iter);
        } else {
          ++time_iter;
        }
      }
      ASSERT(time_iter != timestamp_key_map_.end());
      ASSERT(time_iter->first == key_iter->second);
      ASSERT(time_iter->second == key_timestamp.first);
#endif
      // remove the entry in the Timestamp-Key database
      timestamp_key_map_.erase(time_iter);

      if (key_timestamp.second > 0) {
        // insert an entry at the new time
        timestamp_key_map_.insert(TimestampKeyMap::value_type(
            key_timestamp.second, key_timestamp.first));
        // update the Key-Timestamp database
        key_iter->second = key_timestamp.second;
      } else {
        // If the inpute timestamp is non-positive, we consider the variable as
        // persistent and do not add it to the database.
        key_timestamp_map_.erase(key_iter);
      }

      for (const auto& key : keys_to_update_in_adavance) {
        auto updated_time = timestamps.at(key);
        if (updated_time > 0) {
          // update the Key-Timestamp database
          key_timestamp_map_[key] = updated_time;
          timestamp_key_map_.insert(
              TimestampKeyMap::value_type(updated_time, key));
        } else {
          // If the inpute timestamp is non-positive, we consider the variable
          // as persistent and do not add it to the database.
          key_timestamp_map_.erase(key);
        }
      }
    } else {
      // If the inpute timestamp is non-positive, we consider the variable as
      // persistent and do not add it to the database.
      if (key_timestamp.second > 0) {
        // Add the Key-Timestamp database
        key_timestamp_map_.insert(key_timestamp);
        // Add the key to the Timestamp-Key database
        timestamp_key_map_.insert(TimestampKeyMap::value_type(
            key_timestamp.second, key_timestamp.first));
      }
    }
  }
}

void TemporalSmoother::eraseKeyTimestampMap(const KeyVector& keys) {
  for (Key key : keys) {
    // Erase the key from the Timestamp->Key map
    Timestamp timestamp = key_timestamp_map_.at(key);

    TimestampKeyMap::iterator iter = timestamp_key_map_.lower_bound(timestamp);
    while (iter != timestamp_key_map_.end() && iter->first == timestamp) {
      if (iter->second == key) {
        timestamp_key_map_.erase(iter++);
      } else {
        ++iter;
      }
    }
    // Erase the key from the Key->Timestamp map
    key_timestamp_map_.erase(key);
  }
}

TemporalSmoother::Timestamp TemporalSmoother::getCurrentTimestamp() const {
  if (timestamp_key_map_.size() > 0) {
    return timestamp_key_map_.rbegin()->first;
  } else {
    return -std::numeric_limits<Timestamp>::max();
  }
}

TemporalSmoother::KeyVector TemporalSmoother::findKeysBefore(
    Timestamp timestamp) const {
  KeyVector keys;
  TimestampKeyMap::const_iterator end =
      timestamp_key_map_.lower_bound(timestamp);
  for (TimestampKeyMap::const_iterator iter = timestamp_key_map_.begin();
       iter != end; ++iter) {
    keys.push_back(iter->second);
  }
  return keys;
}

TemporalSmoother::KeyVector TemporalSmoother::findKeysAfter(
    Timestamp timestamp) const {
  KeyVector keys;
  TimestampKeyMap::const_iterator begin =
      timestamp_key_map_.upper_bound(timestamp);
  for (TimestampKeyMap::const_iterator iter = begin;
       iter != timestamp_key_map_.end(); ++iter) {
    keys.push_back(iter->second);
  }
  return keys;
}

TemporalSmoother::VariableOrderingConstraints
TemporalSmoother::createOrderingConstraints(
    const NonlinearFactorGraph& new_factors, const Values& new_theta,
    const FactorIndices& factors_to_remove,
    const std::unordered_set<Key>& marginalizable_keys) const {
  // The default implementation:
  return IncrementalSmoother::createOrderingConstraints(
      new_factors, new_theta, factors_to_remove, marginalizable_keys);

  // clang-format off
  // Experimental ordering constraints: Order the variables by their timestamp.
  //
  // This does not work for now. It seems that some bugs in GTSAM are causing
  // invalid_arguments exceptions:
  // ```
  // terminate called after throwing an instance of 'std::invalid_argument'
  //   what():  EliminationTree: given ordering contains variables that are not involved in the factor graph                 // NOLINT
  // ```
  // clang-format on
  //
  // Update: We have fixed the bug in gtsam mentioned above in the commit
  // b642906625d732ec95fcde02304c55937ebac0b6.
  // However, currently the experimental ordering constraints does not bring
  // better performance in VIO-like problems. We keep it here for future
  // refinement.

  std::map<Timestamp, int> time_to_group;
  const int start_group = 1;  // 0 is reserved for marginalizable variables
  const int max_group = cached_keys_.size();
  int curr_group = start_group;

  // Iterate over timestamp_key_map_
  Timestamp prev_time = -1;
  for (const auto& [timestamp, key] : timestamp_key_map_) {
    ASSERT(timestamp >= prev_time);
    if (timestamp > prev_time && cached_keys_.count(key) &&
        !marginalizable_keys.count(key)) {
      time_to_group[timestamp] = curr_group;
      prev_time = timestamp;
      ASSERT(curr_group < max_group);
      if (curr_group < max_group) {
        curr_group++;
      }
    }
  }

  // Set group for all keys
  gtsam::FastMap<gtsam::Key, int> constrained_keys;
  for (const Key& key : cached_keys_) {
    if (marginalizable_keys.count(key)) {
      // Marginalizable variables are always in Group0
      constrained_keys[key] = 0;
      continue;
    }

    auto it_key = key_timestamp_map_.find(key);
    if (it_key != key_timestamp_map_.end()) {
      // For stamped variables, set the group based on their timestamp
      constrained_keys[key] = time_to_group.at(it_key->second);
    } else {
      // For persistent variables, set their group to `start_group`, i.e.
      // they're in the same group as the newest variables
      constrained_keys[key] = start_group;
    }
  }
  return constrained_keys;
}

std::unique_ptr<const IncrementalSmoother::Result> TemporalSmoother::update(
    const NonlinearFactorGraph& new_factors,
    const Values& new_theta,  //
    const KeyTimestampMap& timestamps, const FactorIndices& factors_to_remove,
    const std::unordered_set<Key>& force_marginalize_keys,
    const std::unordered_set<Key>& no_relinear_keys, TimeCounter* tc) {
  const bool debug = debug_;

  // Update the Timestamps associated with the factor keys
  updateKeyTimestampMap(timestamps);
  if (tc) {
    tc->tag("keyTimeMapUpdated");
  }

  auto ret = IncrementalSmoother::update(
      new_factors, new_theta, factors_to_remove, force_marginalize_keys,
      no_relinear_keys, tc);
  if (tc) {
    tc->tag("smootherUpdated");
  }

  if (ret->marginalized_keys.size() > 0) {
    // Remove marginalized keys from the KeyTimestampMap
    KeyVector keys_to_erase(
        ret->marginalized_keys.begin(), ret->marginalized_keys.end());
    eraseKeyTimestampMap(keys_to_erase);
  }
  if (tc) {
    tc->tag("keyTimeMapErased");
  }

  return ret;
}

FixedLagSmoother::FixedLagSmoother(
    Lag smoother_lag, const ISAM2Params& parameters, const bool debug)
    : smoother_lag_(smoother_lag),
      TemporalSmoother(parameters, nullptr, debug) {}

std::unordered_set<IncrementalSmoother::Key>
FixedLagSmoother::selectMarginalizationVariables(
    const std::unordered_set<Key>& force_marginalize_keys,
    const std::unordered_set<Key>& force_keep_keys,
    const std::unordered_set<Key>& already_affected_keys,
    const std::unordered_set<Key>& no_relinear_keys) const {
  const bool debug = debug_;

  // Get current timestamp
  Timestamp current_timestamp = getCurrentTimestamp();
  if (debug) {
    std::cout << "Current Timestamp: " << current_timestamp << std::endl;
  }

  std::unordered_set<Key> keys_to_marginalize = force_marginalize_keys;
  if (smoother_lag_ < 0) {
    return keys_to_marginalize;
  }

  // Find the set of variables to be marginalized out
  KeyVector keys_vec = findKeysBefore(current_timestamp - smoother_lag_);
  for (const Key& key : keys_vec) {
    if (force_keep_keys.count(key) == 0) {
      keys_to_marginalize.insert(key);
    }
  }
  return keys_to_marginalize;
}

void AdaptiveLagMarginalization::bindSmoother(
    const TemporalSmoother* temporal_smoother) {
  ASSERT(smoother_ == nullptr || smoother_ == temporal_smoother);
  smoother_ = temporal_smoother;
  bindISAM2(&smoother_->getISAM2());
}

bool AdaptiveLagMarginalization::iterateOverMarginalizableKeys(
    const std::function<bool(const Key&)>& f) const {
  return iterateOverMarginalizableKeys(f, nullptr);
}

bool AdaptiveLagMarginalization::iterateOverMarginalizableKeys(
    const std::function<bool(const Key&)>& f,
    std::unordered_set<Key>* visited_keys_p) const {
  using TimestampKeyMap = TemporalSmoother::TimestampKeyMap;
  TimestampKeyMap::const_iterator end =
      smoother_->timestamp_key_map_.lower_bound(marginalTime());
  std::unordered_set<Key> tmp_visited_keys;
  if (visited_keys_p == nullptr) {
    visited_keys_p = &tmp_visited_keys;
  }
  std::unordered_set<Key>& visited_keys = *visited_keys_p;
  const auto& isam2 = smoother_->getISAM2();
  const gtsam::VariableIndex& key_to_factor_indices = isam_->getVariableIndex();
  for (TimestampKeyMap::const_iterator iter =
           smoother_->timestamp_key_map_.begin();
       iter != end; ++iter) {
    const Key& key = iter->second;
    if (visited_keys.count(key)) {
      continue;
    }
    {
      // Invariants check
      auto it_key_to_time = smoother_->key_timestamp_map_.find(key);
      ASSERT(it_key_to_time != smoother_->key_timestamp_map_.end());
      ASSERT(it_key_to_time->second == iter->first);
    }
    if (key_to_factor_indices.find(key) == key_to_factor_indices.end()) {
      // The key was newly added by updateKeyTimestampMap() and not yet
      // processed by ISAM2. Skip it.
      continue;
    }
    const Clique* clique = smoother_->getISAM2()[key].get();
    for (const Key& ikey : clique->conditional()->frontals()) {
      if (!isMarginalizable(ikey)) {
        continue;
      }
      visited_keys.insert(ikey);
      if (f(ikey)) {
        return true;
      }
    }
  }
  return false;
}

AdaptiveLagSmoother::AdaptiveLagSmoother(
    const MarginalizationOptions& marg_options, Lag smoother_lag,
    const ISAM2Params& parameters, const bool debug)
    : adaptive_lag_marginalizer_(std::make_unique<AdaptiveLagMarginalization>(
          std::make_shared<MarginalizationOptions>(marg_options), smoother_lag,
          debug)),
      TemporalSmoother(parameters, adaptive_lag_marginalizer_.get(), debug) {
  adaptive_lag_marginalizer_->bindSmoother(this);
}

}  // namespace sk4slam
