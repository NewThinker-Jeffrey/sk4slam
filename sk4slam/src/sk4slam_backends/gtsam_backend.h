#pragma once

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include "sk4slam_backends/factor_base.h"
#include "sk4slam_backends/generic_factors.h"
#include "sk4slam_backends/gtsam/gtsam_helper.h"
#include "sk4slam_backends/gtsam/incremental_smoother.h"

namespace sk4slam {

class GtsamBackend : public OptimizerBackendInterface {
 public:
  using Key = gtsam::Key;
  using NoiseModel = gtsam::noiseModel::Base;
  using NoiseModelPtr = NoiseModel::shared_ptr;
  using RobustKernel = gtsam::noiseModel::mEstimator::Base;
  using RobustKernelPtr = RobustKernel::shared_ptr;
  using FactorId = size_t;
  using FactorIndex = gtsam::FactorIndex;
  using BackendFactorInterface = gtsam::NoiseModelFactor;
  using BackendFactorInterfacePtr = std::shared_ptr<gtsam::NoiseModelFactor>;
  using Graph = gtsam::NonlinearFactorGraph;

  struct NoiseHelper;
  struct RobustHelper;
  bool debug_{false};

  explicit GtsamBackend(bool use_empty_slots = false)
      : use_empty_slots_(use_empty_slots) {}

  GtsamBackend(const GtsamBackend&) = delete;
  GtsamBackend& operator=(const GtsamBackend&) = delete;

  ~GtsamBackend() override = default;

  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addVariable(
      Key backend_key, const Manifold& initial_value,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    static_assert(
        std::is_same_v<typename CustomizedRetraction::Manifold, Manifold>);
    VariableKey key = toVariableKey(backend_key);
    customized_retractions_[backend_key] = customized_retraction;
    if constexpr (manifold_traits<Manifold>::kDim == Eigen::Dynamic) {
      setManifoldDimensionForDynamicVariable(
          key, manifold_traits<Manifold>::dim(initial_value));
    }

    if constexpr (std::is_same_v<
                      FixedRetraction<Manifold>, CustomizedRetraction>) {
      // Do not add fixed variables to `estimate_`.
      addFixedVariableValue<Manifold>(key, initial_value);
    } else {
      // Add ordinary variables to `estimate_`.
      estimate_.insert(
          backend_key,
          XOptimizableManifold<Manifold>(initial_value, customized_retraction));
    }
    LOGA(
        "GtsamBackend::addVariable: backend_key = %s, variable_key = %s",
        gtsam::DefaultKeyFormatter(backend_key).c_str(),
        gtsam::DefaultKeyFormatter(toBackendKey(key)).c_str());
    return key;
  }

  template <typename Manifold>
  VariableKey addVariable(Key backend_key, const Manifold& initial_value) {
    return addVariable(
        backend_key, initial_value,
        GlobalDefaultRetractionByManifold::get(&initial_value));
  }

  template <typename Manifold>
  VariableKey addFixedVariable(Key backend_key, const Manifold& initial_value) {
    return addVariable(
        backend_key, initial_value,
        RetractionInterface::defaultInstance<FixedRetraction<Manifold>>());
  }

  bool hasVariable(Key backend_key) const {
    return estimate_.exists(backend_key) ||
           hasFixedVariable(toVariableKey(backend_key));
  }

  void removeVariable(Key backend_key) {
    estimate_.erase(backend_key);
    customized_retractions_.erase(backend_key);

    auto variable_key = toVariableKey(backend_key);
    removeDynamicVariable(variable_key);
    removeFixedVariableValue(variable_key);
  }

  template <typename Manifold>
  const Manifold& getEstimate(Key backend_key) const {
    using XOptimizable = XOptimizableManifold<Manifold>;
    LOGA(
        "GtsamBackend::getEstimate: backend_key = %s",
        gtsam::DefaultKeyFormatter(backend_key).c_str());
    auto ptr = estimate_.exists<XOptimizable>(backend_key);
    if (ptr) {
      return ptr->value();
    } else {
      // Otherwise it should be a fixed variable
      VariableKey variable_key = toVariableKey(backend_key);
      ASSERT(hasFixedVariable(variable_key));
      return getFixedVariableValue<Manifold>(variable_key);
    }
  }

  template <typename FrontendFactor>
  FactorId addFactor(
      FrontendFactor&& frontend_factor,
      const NoiseModelPtr& noise_model = nullptr,
      const RobustKernelPtr& robust_kernel = nullptr) {
    return addFactor(createBackendFactor(
        std::move(frontend_factor), noise_model, robust_kernel));
  }

  void removeFactor(FactorId factor_id) {
    auto it = factor_id_to_graph_index_.find(factor_id);
    ASSERT(it != factor_id_to_graph_index_.end());
    gtsam::FactorIndex idx = it->second;
    graph_.remove(idx);
    factor_id_to_graph_index_.erase(it);
  }

  void eraseNullFactors() {
    Graph new_graph;
    for (auto& [factor_id, idx] : factor_id_to_graph_index_) {
      auto factor = std::move(graph_.at(idx));
      ASSERT(factor);
      idx = new_graph.size();
      new_graph.push_back(std::move(factor));
    }
    std::swap(graph_, new_graph);
  }

  template <typename FrontendFactor>
  static std::shared_ptr<FrontendFactor> getFrontendFactor(
      const BackendFactorInterfacePtr& backend_factor) {
    return std::dynamic_pointer_cast<BackendFactor<FrontendFactor>>(
        backend_factor);
  }

  const BackendFactorInterface* getFactor(FactorId factor_id) const {
    auto it = factor_id_to_graph_index_.find(factor_id);
    ASSERT(it != factor_id_to_graph_index_.end());
    gtsam::FactorIndex idx = it->second;
    return dynamic_cast<const BackendFactorInterface*>(graph_.at(idx).get());
  }

  template <typename FrontendFactor>
  const FrontendFactor* getFactor(FactorId factor_id) const {
    auto it = factor_id_to_graph_index_.find(factor_id);
    ASSERT(it != factor_id_to_graph_index_.end());
    gtsam::FactorIndex idx = it->second;
    return dynamic_cast<const BackendFactor<FrontendFactor>*>(
        graph_.at(idx).get());
  }

  VectorXd unwhitenedError(
      FactorId factor_id, gtsam::OptionalMatrixVecType H = nullptr) const {
    const BackendFactorInterface* factor = getFactor(factor_id);
    if (!factor) {
      return VectorXd();
    }
    return factor->unwhitenedError(estimate_, H);
  }

  template <typename SolverOptions>
  auto solve(const SolverOptions& params, bool print_iterations = false) {
    static_assert(
        std::is_same_v<SolverOptions, gtsam::GaussNewtonParams> ||
            std::is_same_v<SolverOptions, gtsam::LevenbergMarquardtParams> ||
            std::is_same_v<SolverOptions, gtsam::DoglegParams>,
        "Invalid solver options");
    using Optimizer = typename SolverOptions::OptimizerType;
    SolverOptions local_params = params;
    if (print_iterations) {
      local_params.verbosity = gtsam::NonlinearOptimizerParams::TERMINATION;
    } else {
      local_params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
    }

    auto optimizer =
        std::make_shared<Optimizer>(graph_, estimate_, local_params);
    gtsam::Values new_estimate = optimizer->optimize();
    std::swap(estimate_, new_estimate);
    return optimizer;
  }

 public:
  void useEmptySlots(bool use_empty_slots = true) {
    use_empty_slots_ = use_empty_slots;
  }

  static VariableKey toVariableKey(Key key) {
    return VariableKey(key);
  }
  static Key toBackendKey(VariableKey key) {
    return key.getImpl();
  }

 protected:
  const RetractionInterface* getCustomizedRetraction(
      VariableKey key) const override {
    auto iter = customized_retractions_.find(toBackendKey(key));
    if (iter != customized_retractions_.end()) {
      return iter->second;
    } else {
      return nullptr;
    }
  }

  template <typename FrontendFactor>
  class BackendFactor;

 protected:
  template <typename FrontendFactor>
  BackendFactorInterfacePtr createBackendFactor(
      FrontendFactor&& fronted_factor,
      const NoiseModelPtr& noise_model = nullptr,
      const RobustKernelPtr& robust_kernel = nullptr) {
    return std::make_shared<BackendFactor<FrontendFactor>>(
        std::move(fronted_factor), this, noise_model, robust_kernel);
  }
  FactorId nextFactorId() {
    return next_factor_id_++;
  }
  FactorId addFactor(BackendFactorInterfacePtr backend_factor) {
    if (use_empty_slots_) {
      std::vector<BackendFactorInterfacePtr> new_factors = {
          std::move(backend_factor)};
      gtsam::FactorIndex idx =
          graph_.add_factors(new_factors, use_empty_slots_)[0];
      FactorId factor_id = nextFactorId();
      factor_id_to_graph_index_[factor_id] = idx;
      return factor_id;
    } else {
      gtsam::FactorIndex idx = graph_.size();
      graph_.push_back(std::move(backend_factor));
      FactorId factor_id = nextFactorId();
      factor_id_to_graph_index_[factor_id] = idx;
      return factor_id;
    }
  }

 protected:
  std::unordered_map<Key, const RetractionInterface*> customized_retractions_;
  Graph graph_;
  gtsam::Values estimate_;
  FactorId next_factor_id_ = 0;
  std::unordered_map<FactorId, gtsam::FactorIndex> factor_id_to_graph_index_;

  bool use_empty_slots_;  // This can bound the memory usage of the graph with
                          // the cost of slowing down the addFactor() method.
};

class ISAM2BackendCommon : private GtsamBackend {
 public:
  using GtsamBackend::BackendFactorInterface;
  using GtsamBackend::BackendFactorInterfacePtr;
  using GtsamBackend::FactorId;
  using GtsamBackend::FactorIndex;
  using GtsamBackend::Key;
  using GtsamBackend::RobustKernel;
  using GtsamBackend::RobustKernelPtr;

  using GtsamBackend::NoiseHelper;
  using GtsamBackend::RobustHelper;

  using GtsamBackend::addFactor;
  using GtsamBackend::removeVariable;
  using GtsamBackend::toBackendKey;
  using GtsamBackend::toVariableKey;
  // using GtsamBackend::hasVariable;
  using GtsamBackend::debug_;

  ISAM2BackendCommon() : GtsamBackend(false) {}

  ~ISAM2BackendCommon() override = default;

  // Get the internal ISAM2 object
  virtual const gtsam::ISAM2& getISAM2() const = 0;

  // template <typename... Args>
  // VariableKey addVariable(Args&&... args) {
  //   VariableKey key = GtsamBackend::addVariable(std::forward<Args>(args)...);
  //   new_keys_.insert(key);
  //   return key;
  // }
  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addVariable(
      Key backend_key, const Manifold& initial_value,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    VariableKey key = GtsamBackend::addVariable(
        backend_key, initial_value, customized_retraction);
    if constexpr (std::is_same_v<
                      FixedRetraction<Manifold>, CustomizedRetraction>) {
      // Do not add fixed variables to `new_keys_`.
    } else {
      // Add ordinary variables to `new_keys_`.
      new_keys_.insert(key);
    }
    return key;
  }

  template <typename Manifold>
  VariableKey addVariable(Key backend_key, const Manifold& initial_value) {
    VariableKey key = GtsamBackend::addVariable(backend_key, initial_value);
    new_keys_.insert(key);
    return key;
  }

  // template <typename... Args>
  // VariableKey addFixedVariable(Args&&... args) {
  //   VariableKey key =
  //       GtsamBackend::addFixedVariable(std::forward<Args>(args)...);
  //   new_keys_.insert(key);
  //   return key;
  // }

  template <typename Manifold>
  VariableKey addFixedVariable(Key backend_key, const Manifold& initial_value) {
    VariableKey key =
        GtsamBackend::addFixedVariable(backend_key, initial_value);
    // new_keys_.insert(key);  // Do not add fixed variables to `new_keys_`.
    return key;
  }

  bool hasVariable(Key backend_key) const {
    bool ret = GtsamBackend::hasVariable(backend_key) ||
               getISAM2().valueExists(backend_key);

    // Invariance check
    VariableKey key = toVariableKey(backend_key);
    bool ret1 = new_keys_.count(key);
    bool ret2 = getISAM2().valueExists(backend_key);
    bool ret3 = hasFixedVariable(key);
    ASSERT(ret == (ret1 || ret2 || ret3));
    ASSERT(ret3 || ret1 == !ret2);

    return ret;
  }

  bool isVariableUpdated(Key backend_key) const {
    if (hasVariable(backend_key)) {
      bool ret = getISAM2().valueExists(backend_key);

      // Invariance check
      VariableKey key = toVariableKey(backend_key);
      ASSERT(hasFixedVariable(key) || ret == !new_keys_.count(key));

      return ret;
    } else {
      return false;
    }
  }

  /// Override the getEstimate() method to return the estimate from ISAM2.
  /// Note we're returning a copy of the estimate since ISAM2 itself doesn't
  /// provide a method to get a reference to the current estimate, see
  /// `ISAM2::calculateEstimate()`;
  template <typename Manifold>
  Manifold getEstimate(Key backend_key) const {
    using XOptimizable = XOptimizableManifold<Manifold>;
    LOGA(
        "GtsamBackend.ISAM2::getEstimate: backend_key = %s",
        gtsam::DefaultKeyFormatter(backend_key).c_str());
    // ASSERT(getISAM2().valueExists(backend_key));

    if (!getISAM2().valueExists(backend_key)) {
      // If the variable has not been updated by ISAM2 yet, return the initial
      // value instead.
      return GtsamBackend::getEstimate<Manifold>(backend_key);
    }

    return getISAM2().calculateEstimate<XOptimizable>(backend_key).value();
  }

  void removeFactor(FactorId factor_id) {
    auto it = factor_id_to_isam_index_.find(factor_id);
    if (it != factor_id_to_isam_index_.end()) {
      // The factor has already been added to ISAM2
      ASSERT(!factor_id_to_graph_index_.count(factor_id));

      gtsam::FactorIndex isam_index = it->second;
      auto it2 = isam_index_to_factor_id_.find(isam_index);
      ASSERT(it2 != isam_index_to_factor_id_.end());
      ASSERT(it2->second == factor_id);
      factor_id_to_isam_index_.erase(it);
      isam_index_to_factor_id_.erase(it2);
    } else {
      // // It's discouraged to call removeFactor() before the factor is added
      // to
      // // ISAM2 (i.e., before calling update() after addFactor()).
      // throw std::runtime_error(
      //     "ISAM2BackendCommon::removeFactor(): Factor has not been added to "
      //     "ISAM2yet!");

      // The factor has not been added to ISAM2 yet.
      GtsamBackend::removeFactor(factor_id);
    }
  }

  const BackendFactorInterface* getFactor(FactorId factor_id) const {
    auto it = factor_id_to_isam_index_.find(factor_id);
    if (it != factor_id_to_isam_index_.end()) {
      // The factor has already been added to ISAM2
      ASSERT(!factor_id_to_graph_index_.count(factor_id));

      gtsam::FactorIndex isam_index = it->second;
      auto it2 = isam_index_to_factor_id_.find(isam_index);
      ASSERT(it2 != isam_index_to_factor_id_.end());
      return dynamic_cast<const BackendFactorInterface*>(
          getISAM2().getFactorsUnsafe().at(isam_index).get());
    } else {
      // The factor has not been added to ISAM2 yet.
      return GtsamBackend::getFactor(factor_id);
    }
  }

  template <typename FrontendFactor>
  const FrontendFactor* getFactor(FactorId factor_id) const {
    auto it = factor_id_to_isam_index_.find(factor_id);
    if (it != factor_id_to_isam_index_.end()) {
      // The factor has already been added to ISAM2
      ASSERT(!factor_id_to_graph_index_.count(factor_id));

      gtsam::FactorIndex isam_index = it->second;
      auto it2 = isam_index_to_factor_id_.find(isam_index);
      ASSERT(it2 != isam_index_to_factor_id_.end());
      return dynamic_cast<const BackendFactor<FrontendFactor>*>(
          getISAM2().getFactorsUnsafe().at(isam_index).get());
    } else {
      // The factor has not been added to ISAM2 yet.
      return GtsamBackend::getFactor<FrontendFactor>(factor_id);
    }
  }

  VectorXd unwhitenedError(
      FactorId factor_id, gtsam::OptionalMatrixVecType H = nullptr) const {
    const BackendFactorInterface* factor = getFactor(factor_id);
    if (!factor) {
      return VectorXd();
    }
    // const gtsam::Values& x = getISAM2().getLinearizationPoint();
    gtsam::Values x;
    for (const Key& backend_key : factor->keys()) {
      if (!getISAM2().valueExists(backend_key)) {
        // If the variable has not been updated by ISAM2 yet, use the initial
        // value.
        x.insert(backend_key, estimate_.at(backend_key));
      } else {
        // If the variable has been updated by ISAM2, use the updated value.
        x.insert(backend_key, getISAM2().calculateEstimate(backend_key));
      }
    }
    return factor->unwhitenedError(x, H);
  }

 protected:
  gtsam::FactorIndices getFactorIndices(
      const gtsam::FastVector<FactorId>& factor_ids) const {
    gtsam::FactorIndices factor_indices;
    factor_indices.reserve(factor_ids.size());
    for (const auto& factor_id : factor_ids) {
      auto it = factor_id_to_isam_index_.find(factor_id);
      ASSERT(it != factor_id_to_isam_index_.end());
      factor_indices.push_back(it->second);
    }
    return factor_indices;
  }

  gtsam::FactorIndex getFactorIndex(FactorId factor_id) const {
    auto it = factor_id_to_isam_index_.find(factor_id);
    ASSERT(it != factor_id_to_isam_index_.end());
    return it->second;
  }

  gtsam::FastVector<FactorId> getFactorIds(
      const gtsam::FactorIndices& factor_indices) const {
    gtsam::FastVector<FactorId> factor_ids;
    factor_ids.reserve(factor_indices.size());
    for (const auto& factor_index : factor_indices) {
      auto it = isam_index_to_factor_id_.find(factor_index);
      ASSERT(it != isam_index_to_factor_id_.end());
      factor_ids.push_back(it->second);
    }
    return factor_ids;
  }

  FactorId getFactorId(gtsam::FactorIndex factor_index) const {
    auto it = isam_index_to_factor_id_.find(factor_index);
    ASSERT(it != isam_index_to_factor_id_.end());
    return it->second;
  }

  void prepareForUpdate(
      const gtsam::NonlinearFactorGraph** new_factors,
      gtsam::Values* new_theta) {
    eraseNullFactors();
    *new_factors = &graph_;
    std::unordered_set<VariableKey> keys_in_new_factors;
    for (const auto& factor : graph_) {
      ASSERT(factor);
      for (const auto& backend_key : factor->keys()) {
        keys_in_new_factors.insert(toVariableKey(backend_key));
      }
    }

    for (VariableKey key : new_keys_) {
      if (keys_in_new_factors.count(key) == 0) {
        continue;
      }
      Key backend_key = toBackendKey(key);
      ASSERT(estimate_.exists(backend_key));
      new_theta->insert(backend_key, estimate_.at(backend_key));
    }
  }

  void postUpdate(
      const gtsam::Values used_new_theta,
      const gtsam::FactorIndices& new_factor_indices,
      const gtsam::FactorIndices& remove_factor_indices =
          gtsam::FactorIndices()) {
    if (remove_factor_indices.size() > 0) {
      std::unordered_set<gtsam::FactorIndex> checked_removed_factors;
      checked_removed_factors.rehash(remove_factor_indices.size() * 2);
      for (const auto& isam_index : remove_factor_indices) {
        auto it = isam_index_to_factor_id_.find(isam_index);
        ASSERT(it != isam_index_to_factor_id_.end());
        const auto& factor_id = it->second;
        auto it2 = factor_id_to_isam_index_.find(factor_id);
        ASSERT(it2 != factor_id_to_isam_index_.end());
        ASSERT(it2->second == isam_index);

        isam_index_to_factor_id_.erase(it);
        factor_id_to_isam_index_.erase(it2);
        checked_removed_factors.insert(isam_index);

        // The factors to remove shuoldn't include any new factor;
        ASSERT(factor_id_to_graph_index_.count(factor_id) == 0);
      }
      ASSERT(checked_removed_factors.size() == remove_factor_indices.size());
      if (debug_) {
        LOGI(
            "GtsamBackend.ISAM2: Removed %d factors from isam, indices: %s",
            checked_removed_factors.size(),
            toStr(checked_removed_factors).c_str());
      }
    }

    ASSERT(factor_id_to_graph_index_.size() == new_factor_indices.size());
    if (factor_id_to_graph_index_.size() > 0) {
      std::unordered_set<gtsam::FactorIndex> checked_new_factors;
      checked_new_factors.rehash(new_factor_indices.size() * 2);
      for (const auto& item : factor_id_to_graph_index_) {
        const auto& factor_id = item.first;
        const auto& graph_index = item.second;
        auto isam_index = new_factor_indices.at(graph_index);
        checked_new_factors.insert(isam_index);

        // isam_index_to_factor_id_[isam_index] = factor_id;
        // factor_id_to_isam_index_[factor_id] = isam_index;
        auto intert_res = isam_index_to_factor_id_.insert(
            std::make_pair(isam_index, factor_id));
        ASSERT(intert_res.second);
        auto intert_res2 = factor_id_to_isam_index_.insert(
            std::make_pair(factor_id, isam_index));
        ASSERT(intert_res2.second);
      }
      ASSERT(checked_new_factors.size() == new_factor_indices.size());
      if (debug_) {
        LOGI(
            "GtsamBackend.ISAM2: Added %d new factors to isam, indices: %s",
            checked_new_factors.size(), toStr(checked_new_factors).c_str());
      }
    }

    for (auto iter = used_new_theta.begin(); iter != used_new_theta.end();
         ++iter) {
      auto key = iter->key;
      new_keys_.erase(key);
    }

    factor_id_to_graph_index_.clear();
    graph_ = Graph();
    ASSERT(graph_.size() == 0);
    // ASSERT(new_keys_.size() == 0);
  }

  void postMarginalize(
      const gtsam::FastList<Key>& marginalized_keys,
      const gtsam::FactorIndices& marginal_factor_indices,
      const gtsam::FactorIndices& marginalized_factor_indices,
      gtsam::FastVector<FactorId>* marginal_factor_ids = nullptr,
      gtsam::FastVector<FactorId>* marginalized_factor_ids = nullptr) {
    // Remove entries for the marginalized factors
    if (marginalized_factor_indices.size() > 0) {
      if (marginalized_factor_ids) {
        *marginalized_factor_ids = getFactorIds(marginalized_factor_indices);
      }

      std::unordered_set<gtsam::FactorIndex> checked_marginalized_factors;
      checked_marginalized_factors.rehash(
          marginalized_factor_indices.size() * 2);
      for (const auto& isam_index : marginalized_factor_indices) {
        auto it = isam_index_to_factor_id_.find(isam_index);

        // TODO(jeffrey): Bug: Assert failed here.
        ASSERT(it != isam_index_to_factor_id_.end());

        auto factor_id = it->second;
        auto it2 = factor_id_to_isam_index_.find(factor_id);
        ASSERT(it2 != factor_id_to_isam_index_.end());

        isam_index_to_factor_id_.erase(it);
        factor_id_to_isam_index_.erase(it2);
        checked_marginalized_factors.insert(isam_index);
      }
      ASSERT(
          checked_marginalized_factors.size() ==
          marginalized_factor_indices.size());
      if (debug_) {
        LOGI(
            "GtsamBackend.ISAM2: Marginalized %d factors from isam, indices: "
            "%s",
            checked_marginalized_factors.size(),
            toStr(checked_marginalized_factors).c_str());
      }
    }

    // Add new entries for the marginal factors
    if (marginal_factor_indices.size() > 0) {
      std::unordered_set<gtsam::FactorIndex> checked_marginal_factors;
      checked_marginal_factors.rehash(marginal_factor_indices.size() * 2);
      for (const auto& isam_index : marginal_factor_indices) {
        auto factor_id = nextFactorId();
        auto insert_res = isam_index_to_factor_id_.insert(
            std::make_pair(isam_index, factor_id));
        ASSERT(insert_res.second);
        auto insert_res2 = factor_id_to_isam_index_.insert(
            std::make_pair(factor_id, isam_index));
        ASSERT(insert_res2.second);
        checked_marginal_factors.insert(isam_index);
      }
      ASSERT(checked_marginal_factors.size() == marginal_factor_indices.size());
      if (debug_) {
        LOGI(
            "GtsamBackend.ISAM2: Added %d marginal factors to isam, indices: "
            "%s",
            checked_marginal_factors.size(),
            toStr(checked_marginal_factors).c_str());
      }
      if (marginal_factor_ids) {
        *marginal_factor_ids = getFactorIds(marginal_factor_indices);
      }
    }

    // Remove entries for the marginalized keys
    for (Key backend_key : marginalized_keys) {
      // Eusure that the backend key is removed from the isam
      ASSERT(!getISAM2().nodes().exists(backend_key));
      ASSERT(!getISAM2().valueExists(backend_key));
      GtsamBackend::removeVariable(backend_key);
    }
    if (debug_) {
      std::set<Key> ordered_keys(
          marginalized_keys.begin(), marginalized_keys.end());
      ASSERT(ordered_keys.size() == marginalized_keys.size());
      LOGI(
          "GtsamBackend.ISAM2: Marginalized %d variables from isam, keys: %s",
          marginalized_keys.size(),
          toStr(ordered_keys, [](const Key& backend_key) {
            return gtsam::DefaultKeyFormatter(backend_key);
          }).c_str());
    }
  }

 private:
  std::unordered_map<FactorId, gtsam::FactorIndex> factor_id_to_isam_index_;
  std::unordered_map<gtsam::FactorIndex, FactorId> isam_index_to_factor_id_;
  std::unordered_set<VariableKey> new_keys_;
};

class ISAM2Backend : public ISAM2BackendCommon {
 public:
  explicit ISAM2Backend(
      const gtsam::ISAM2Params& parameters = gtsam::ISAM2Params())
      : ISAM2BackendCommon(),
        isam_(std::make_unique<gtsam::ISAM2>(parameters)) {}

  explicit ISAM2Backend(std::unique_ptr<gtsam::ISAM2> isam)
      : ISAM2BackendCommon(), isam_(std::move(isam)) {}

  ~ISAM2Backend() override = default;

  // Get the internal ISAM2 object
  const gtsam::ISAM2& getISAM2() const override {
    return *isam_;
  }

  gtsam::ISAM2Result update(
      const gtsam::FastVector<FactorId>& remove_factor_ids =
          gtsam::FastVector<FactorId>(),
      const std::optional<gtsam::FastMap<Key, int>>& constrained_keys = {},
      const std::optional<gtsam::FastList<Key>>& no_relin_keys = {},
      const std::optional<gtsam::FastList<Key>>& extra_reelim_keys = {},
      bool force_relinearize = false) {
    gtsam::ISAM2UpdateParams params;
    params.constrainedKeys = constrained_keys;
    params.extraReelimKeys = extra_reelim_keys;
    params.force_relinearize = force_relinearize;
    params.noRelinKeys = no_relin_keys;
    params.removeFactorIndices = getFactorIndices(remove_factor_ids);
    return update(&params);
  }

  gtsam::ISAM2Result update(
      gtsam::ISAM2UpdateParams update_params,
      const gtsam::FastVector<FactorId>& remove_factor_ids =
          gtsam::FastVector<FactorId>()) {
    ASSERT(update_params.removeFactorIndices.empty());
    update_params.removeFactorIndices = getFactorIndices(remove_factor_ids);
    return update(&update_params);
  }

  void marginalizeLeaves(
      const gtsam::FastList<Key>& leafKeys,
      gtsam::FastVector<FactorId>* marginal_factor_ids = nullptr,
      gtsam::FastVector<FactorId>* marginalized_factor_ids = nullptr) {
    gtsam::FactorIndices marginal_factor_indices, marginalized_factor_indices;
    isam_->marginalizeLeaves(
        leafKeys, &marginal_factor_indices, &marginalized_factor_indices);
    postMarginalize(
        leafKeys, marginal_factor_indices, marginalized_factor_indices,
        marginal_factor_ids, marginalized_factor_ids);
  }

  template <class... OptArgs>
  void marginalizeLeaves(
      const gtsam::FastList<Key>& leafKeys, OptArgs&&... optArgs) {
    // dereference the optional arguments and pass
    // it to the pointer version
    marginalizeLeaves(leafKeys, (&optArgs)...);
  }

 private:
  gtsam::ISAM2Result update(const gtsam::ISAM2UpdateParams* update_params) {
    gtsam::Values new_theta;
    const gtsam::NonlinearFactorGraph* new_factors;
    prepareForUpdate(&new_factors, &new_theta);
    gtsam::ISAM2Result isam2_result =
        isam_->update(*new_factors, new_theta, *update_params);

    postUpdate(
        new_theta, isam2_result.newFactorsIndices,
        update_params->removeFactorIndices);
    return isam2_result;
  }

 private:
  std::unique_ptr<gtsam::ISAM2> isam_;
};

namespace gtsam_backend_internal {

template <typename IncrementalSmootherResult>
struct IncrementalSmootherResultWrapper {
  using FactorId = GtsamBackend::FactorId;
  using Key = GtsamBackend::Key;
  IncrementalSmootherResultWrapper(
      std::unique_ptr<const IncrementalSmootherResult> result,
      gtsam::FastVector<FactorId> removed_factor_ids,
      gtsam::FastVector<FactorId> new_marginal_factor_ids,
      gtsam::FastVector<FactorId> marginalized_factor_ids)
      : result_(std::move(result)),
        removed_factor_ids_(std::move(removed_factor_ids)),
        new_marginal_factor_ids_(std::move(new_marginal_factor_ids)),
        marginalized_factor_ids_(std::move(marginalized_factor_ids)) {
    ASSERT(
        result_->removed_factor_indices.size() == removed_factor_ids_.size());
    ASSERT(
        result_->new_marginal_factor_indices.size() ==
        new_marginal_factor_ids_.size());
    ASSERT(
        result_->marginalized_factor_indices.size() ==
        marginalized_factor_ids_.size());
    for (size_t i = 0; i < removed_factor_ids_.size(); ++i) {
      removed_factors_map_[removed_factor_ids_[i]] =
          result_->removed_factor_indices[i];
    }
    for (size_t i = 0; i < new_marginal_factor_ids_.size(); ++i) {
      new_marginal_factors_map_[new_marginal_factor_ids_[i]] =
          result_->new_marginal_factor_indices[i];
    }
    for (size_t i = 0; i < marginalized_factor_ids_.size(); ++i) {
      marginalized_factors_map_[marginalized_factor_ids_[i]] =
          result_->marginalized_factor_indices[i];
    }
  }

  const IncrementalSmootherResult& internal() const {
    return *result_;
  }

  const gtsam::FastVector<FactorId>& removedFactorIds() const {
    return removed_factor_ids_;
  }

  const gtsam::FastVector<FactorId>& newMarginalFactorIds() const {
    return new_marginal_factor_ids_;
  }

  const gtsam::FastVector<FactorId>& marginalizedFactorIds() const {
    return marginalized_factor_ids_;
  }

  const gtsam::KeyList& marginalizedKeys() const {
    return result_->marginalized_keys;
  }

  template <typename Manifold>
  const XOptimizableManifold<Manifold> getMarginalizedOptimizable(Key j) const {
    using XOptimizable = XOptimizableManifold<Manifold>;
    return result_->template getMarginalizedValue<XOptimizable>(j);
  }

  template <typename Manifold>
  const Manifold getMarginalizedValue(Key j) const {
    return getMarginalizedOptimizable<Manifold>(j).value();
  }

  template <typename FrontendFactor>
  std::shared_ptr<FrontendFactor> getRemovedFactor(FactorId factor_id) const {
    return GtsamBackend::getFrontendFactor<FrontendFactor>(
        result_
            ->template getRemovedFactor<GtsamBackend::BackendFactorInterface>(
                removed_factors_map_.at(factor_id)));
  }

  template <typename FrontendFactor>
  std::shared_ptr<FrontendFactor> getNewMarginalFactor(
      FactorId factor_id) const {
    return GtsamBackend::getFrontendFactor<FrontendFactor>(
        result_->template getNewMarginalFactor<
            GtsamBackend::BackendFactorInterface>(
            new_marginal_factors_map_.at(factor_id)));
  }

  template <typename FrontendFactor>
  std::shared_ptr<FrontendFactor> getMarginalizedFactor(
      FactorId factor_id) const {
    return GtsamBackend::getFrontendFactor<FrontendFactor>(
        result_->template getMarginalizedFactor<
            GtsamBackend::BackendFactorInterface>(
            marginalized_factors_map_.at(factor_id)));
  }

 private:
  std::unique_ptr<const IncrementalSmootherResult> result_;

  gtsam::FastVector<FactorId> removed_factor_ids_;
  gtsam::FastVector<FactorId> new_marginal_factor_ids_;
  gtsam::FastVector<FactorId> marginalized_factor_ids_;

  std::unordered_map<FactorId, gtsam::FactorIndex> removed_factors_map_;
  std::unordered_map<FactorId, gtsam::FactorIndex> marginalized_factors_map_;
  std::unordered_map<FactorId, gtsam::FactorIndex> new_marginal_factors_map_;
};

}  // namespace gtsam_backend_internal

template <typename IncrementalSmoother>
class IncrementalSmootherBackend : public ISAM2BackendCommon {
 public:
  using ResultWrapper =
      gtsam_backend_internal::IncrementalSmootherResultWrapper<
          typename IncrementalSmoother::Result>;

  template <typename... SmootherConstructArgs>
  explicit IncrementalSmootherBackend(SmootherConstructArgs&&... args)
      : ISAM2BackendCommon(),
        smoother_(std::make_unique<IncrementalSmoother>(
            std::forward<SmootherConstructArgs>(args)...)) {}

  explicit IncrementalSmootherBackend(
      std::unique_ptr<IncrementalSmoother> smoother)
      : ISAM2BackendCommon(), smoother_(std::move(smoother)) {}

  ~IncrementalSmootherBackend() override {}

  // template <typename... Args>
  // VariableKey addTemporaryVariable(Args&&... args) {
  //   VariableKey key =
  //       ISAM2BackendCommon::addVariable(std::forward<Args>(args)...);
  //   smoother_->markTemporaryVariables({toBackendKey(key)});
  //   return key;
  // }

  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addTemporaryVariable(
      gtsam::Key backend_key, const Manifold& initial_value,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    VariableKey key = ISAM2BackendCommon::addVariable(
        backend_key, initial_value, customized_retraction);
    smoother_->markTemporaryVariables({toBackendKey(key)});
    return key;
  }

  template <typename Manifold>
  VariableKey addTemporaryVariable(
      gtsam::Key backend_key, const Manifold& initial_value) {
    VariableKey key =
        ISAM2BackendCommon::addVariable(backend_key, initial_value);
    smoother_->markTemporaryVariables({toBackendKey(key)});
    return key;
  }

  // Get the internal ISAM2 object
  const gtsam::ISAM2& getISAM2() const override {
    return smoother_->getISAM2();
  }

  const IncrementalSmoother& getSmoother() const {
    return *smoother_;
  }

  const std::unordered_set<FactorId>& getMarginalFactors() const {
    return marginal_factors_;
  }

  std::unique_ptr<ResultWrapper> update(
      const gtsam::FastVector<FactorId>& factor_ids_to_remove =
          gtsam::FastVector<FactorId>(),
      const std::unordered_set<gtsam::Key>& force_marginalize_keys = {},
      const std::unordered_set<gtsam::Key>& no_relinear_keys = {},
      TimeCounter* tc = nullptr) {
    gtsam::FactorIndices factor_indices_to_remove =
        getFactorIndices(factor_ids_to_remove);
    gtsam::Values new_theta;
    const gtsam::NonlinearFactorGraph* new_factors;
    prepareForUpdate(&new_factors, &new_theta);
    if (tc) {
      tc->tag("prepared");
    }

    auto internal_smoother_result = smootherUpdate(
        *new_factors, new_theta, factor_indices_to_remove,
        force_marginalize_keys, no_relinear_keys, tc);
    if (tc) {
      tc->tag("updated");
    }

    const gtsam::ISAM2Result& isam2_result = smoother_->getISAM2Result();
    postUpdate(
        new_theta, isam2_result.newFactorsIndices, factor_indices_to_remove);
    if (tc) {
      tc->tag("postUpdateFinished");
    }

    gtsam::FastVector<FactorId> marginal_factor_ids;
    gtsam::FastVector<FactorId> marginalized_factor_ids;
    postMarginalize(
        internal_smoother_result->marginalized_keys,
        internal_smoother_result->new_marginal_factor_indices,
        internal_smoother_result->marginalized_factor_indices,
        &marginal_factor_ids, &marginalized_factor_ids);

    // Update marginal_factors_
    int n_removed_marginal = 0;
    int n_marginalized_marginal = 0;
    for (const auto& factor_id : factor_ids_to_remove) {
      n_removed_marginal += marginal_factors_.erase(factor_id);
    }
    for (const auto& factor_id : marginalized_factor_ids) {
      n_marginalized_marginal += marginal_factors_.erase(factor_id);
    }
    for (const auto& factor_id : marginal_factor_ids) {
      ASSERT(marginal_factors_.insert(factor_id).second);
    }
    if (debug_) {
      LOGI(
          "GtsamBackend.ISAM2: marginal_factors change: removed %d, "
          "marginalized %d, added %d, remaining %d",
          n_removed_marginal, n_marginalized_marginal,
          marginal_factor_ids.size(), marginal_factors_.size());
    }
    if (tc) {
      tc->tag("postMarginalizeFinished");
    }

    return std::make_unique<ResultWrapper>(
        std::move(internal_smoother_result), factor_ids_to_remove,
        std::move(marginal_factor_ids), std::move(marginalized_factor_ids));
  }

  int runExtraUpdatesUntilConverged(int max_iterations = 10) {
    return smoother_->runExtraUpdatesUntilConverged(max_iterations);
  }

  bool isConverged() const {
    return smoother_->isConverged();
  }

 protected:
  virtual std::unique_ptr<const typename IncrementalSmoother::Result>
  smootherUpdate(
      const gtsam::NonlinearFactorGraph& new_factors,
      const gtsam::Values& new_theta,  //
      const gtsam::FactorIndices& factors_to_remove,
      const std::unordered_set<gtsam::Key>& force_marginalize_keys,
      const std::unordered_set<gtsam::Key>& no_relinear_keys, TimeCounter* tc) {
    if constexpr (kHasBasicUpdate) {
      return smoother_->update(
          new_factors, new_theta, factors_to_remove, force_marginalize_keys,
          no_relinear_keys, tc);
    } else {
      throw std::runtime_error("IncrementalSmoother::update() not accecsible!");
    }
  }

 protected:
  std::unique_ptr<IncrementalSmoother> smoother_;
  std::unordered_set<FactorId>
      marginal_factors_;  /// Marginal factors that are currently in the
                          /// underlying factor graph.

 private:
  DEFINE_HAS_MEMBER_FUNCTION(update)
  static constexpr bool kHasBasicUpdate = HasMemberFunction_update<
      IncrementalSmoother, const gtsam::NonlinearFactorGraph&,
      const gtsam::Values&, const gtsam::FactorIndices&,
      const std::unordered_set<gtsam::Key>&>;
};

template <typename TemporalSmoother>
class TemporalSmootherBackend
    : public IncrementalSmootherBackend<TemporalSmoother> {
  using Base = IncrementalSmootherBackend<TemporalSmoother>;

 public:
  using Timestamp = typename TemporalSmoother::Timestamp;
  using Base::Base;
  using Base::debug_;
  ~TemporalSmootherBackend() override {}

  static inline const Timestamp kPersistentVariableTime =
      TemporalSmoother::kPersistentVariableTime;

  /// Override ISAM2BackendCommon::addVariable() to add timestamps
  /// to the new keys

  // template <typename... Args>
  // VariableKey addVariable(Timestamp time, Args&&... args) {
  //   VariableKey key = Base::addVariable(std::forward<Args>(args)...);
  //   times_for_new_keys_[Base::toBackendKey(key)] = time;
  //   return key;
  // }

  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addVariable(
      Timestamp time, gtsam::Key backend_key, const Manifold& initial_value,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    VariableKey key =
        Base::addVariable(backend_key, initial_value, customized_retraction);
    times_for_new_keys_[Base::toBackendKey(key)] = time;
    return key;
  }

  template <typename Manifold>
  VariableKey addVariable(
      Timestamp time, gtsam::Key backend_key, const Manifold& initial_value) {
    VariableKey key = Base::addVariable(backend_key, initial_value);
    times_for_new_keys_[Base::toBackendKey(key)] = time;
    return key;
  }

  // template <typename... Args>
  // VariableKey addFixedVariable(Timestamp time, Args&&... args) {
  //   VariableKey key = Base::addFixedVariable(std::forward<Args>(args)...);
  //   times_for_new_keys_[Base::toBackendKey(key)] = time;
  //   return key;
  // }

  template <typename Manifold>
  VariableKey addFixedVariable(
      Timestamp time, gtsam::Key backend_key, const Manifold& initial_value) {
    VariableKey key = Base::addFixedVariable(backend_key, initial_value);
    times_for_new_keys_[Base::toBackendKey(key)] = time;
    return key;
  }

  // template <typename... Args>
  // VariableKey addTemporaryVariable(Timestamp time, Args&&... args) {
  //   VariableKey key = Base::addVariable(std::forward<Args>(args)...);
  //   times_for_new_keys_[Base::toBackendKey(key)] = time;
  //   return key;
  // }
  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addTemporaryVariable(
      Timestamp time, gtsam::Key backend_key, const Manifold& initial_value,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    VariableKey key =
        Base::addVariable(backend_key, initial_value, customized_retraction);
    times_for_new_keys_[Base::toBackendKey(key)] = time;
    return key;
  }

  template <typename Manifold>
  VariableKey addTemporaryVariable(
      Timestamp time, gtsam::Key backend_key, const Manifold& initial_value) {
    VariableKey key = Base::addVariable(backend_key, initial_value);
    times_for_new_keys_[Base::toBackendKey(key)] = time;
    return key;
  }

  /// Override ISAM2BackendCommon::removeVariable() to remove timestamps
  /// for the keys
  void removeVariable(gtsam::Key backend_key) {
    times_for_new_keys_.erase(backend_key);
    ISAM2BackendCommon::removeVariable(backend_key);
  }

  void updateVariableTime(gtsam::Key backend_key, Timestamp time) {
    times_for_new_keys_[backend_key] = time;
  }

 protected:
  std::unique_ptr<const typename TemporalSmoother::Result> smootherUpdate(
      const gtsam::NonlinearFactorGraph& new_factors,
      const gtsam::Values& new_theta,  //
      const gtsam::FactorIndices& factors_to_remove,
      const std::unordered_set<gtsam::Key>& force_marginalize_keys,
      const std::unordered_set<gtsam::Key>& no_relinear_keys,
      TimeCounter* tc) override {
    if (debug_) {
      LOGI(
          "GtsamBackend.TemporalSmoother.update: new_theta.size() = "
          "%d, new_factors.size() = %d",
          new_theta.size(), new_factors.size());
      for (const auto& [key, time] : times_for_new_keys_) {
        LOGD(
            "GtsamBackend.TemporalSmoother.update: times key = %s, "
            "time = %f",
            gtsam::DefaultKeyFormatter(key).c_str(), time);
      }
      for (const auto& [key, value] : new_theta) {
        LOGD(
            "GtsamBackend.TemporalSmoother.update: thetas key = %s,",
            gtsam::DefaultKeyFormatter(key).c_str());
      }
    }
    auto smoother_result = Base::smoother_->update(
        new_factors, new_theta, times_for_new_keys_, factors_to_remove,
        force_marginalize_keys, no_relinear_keys, tc);
    times_for_new_keys_.clear();
    return smoother_result;
  }

 protected:
  std::unordered_map<gtsam::Key, double> times_for_new_keys_;
};

using FixedLagSmootherBackend = TemporalSmootherBackend<FixedLagSmoother>;

using AdaptiveLagSmootherBackend = TemporalSmootherBackend<AdaptiveLagSmoother>;

struct GtsamBackend::NoiseHelper {
  static gtsam::noiseModel::Gaussian::shared_ptr SqrtInfo(
      const gtsam::Matrix& R) {
    return gtsam::noiseModel::Gaussian::SqrtInformation(R);
  }

  static gtsam::noiseModel::Gaussian::shared_ptr Info(const gtsam::Matrix& M) {
    return gtsam::noiseModel::Gaussian::Information(M);
  }

  static gtsam::noiseModel::Gaussian::shared_ptr Cov(
      const gtsam::Matrix& covariance) {
    return gtsam::noiseModel::Gaussian::Covariance(covariance);
  }

  static gtsam::noiseModel::Diagonal::shared_ptr Sigmas(
      const gtsam::Vector& sigmas) {
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  }

  static gtsam::noiseModel::Diagonal::shared_ptr Variances(
      const gtsam::Vector& variances) {
    return gtsam::noiseModel::Diagonal::Variances(variances);
  }

  /// @param precisions   the diagonal of the information matrix, i.e., weights
  static gtsam::noiseModel::Diagonal::shared_ptr Precisions(
      const gtsam::Vector& precisions) {
    return gtsam::noiseModel::Diagonal::Precisions(precisions);
  }

  static gtsam::noiseModel::Isotropic::shared_ptr Sigma(
      size_t dim, double sigma) {
    return gtsam::noiseModel::Isotropic::Sigma(dim, sigma);
  }

  static gtsam::noiseModel::Isotropic::shared_ptr Variance(
      size_t dim, double variance) {
    return gtsam::noiseModel::Isotropic::Variance(dim, variance);
  }

  static gtsam::noiseModel::Isotropic::shared_ptr Precision(
      size_t dim, double precision) {
    return gtsam::noiseModel::Isotropic::Precision(dim, precision);
  }

  static gtsam::noiseModel::Unit::shared_ptr Unit(size_t dim) {
    return gtsam::noiseModel::Unit::Create(dim);
  }
};

struct GtsamBackend::RobustHelper {
  static gtsam::noiseModel::mEstimator::Fair::shared_ptr Fair(
      double c = 1.3998) {
    return gtsam::noiseModel::mEstimator::Fair::Create(c);
  }

  static gtsam::noiseModel::mEstimator::Huber::shared_ptr Huber(
      double k = 1.345) {
    return gtsam::noiseModel::mEstimator::Huber::Create(k);
  }

  static gtsam::noiseModel::mEstimator::Cauchy::shared_ptr Cauchy(
      double k = 0.1) {
    return gtsam::noiseModel::mEstimator::Cauchy::Create(k);
  }

  static gtsam::noiseModel::mEstimator::Tukey::shared_ptr Tukey(
      double k = 4.6851) {
    return gtsam::noiseModel::mEstimator::Tukey::Create(k);
  }

  static gtsam::noiseModel::mEstimator::Welsch::shared_ptr Welsch(
      double k = 2.9846) {
    return gtsam::noiseModel::mEstimator::Welsch::Create(k);
  }

  static gtsam::noiseModel::mEstimator::GemanMcClure::shared_ptr GemanMcClure(
      double k = 1.0) {
    return gtsam::noiseModel::mEstimator::GemanMcClure::Create(k);
  }

  static gtsam::noiseModel::mEstimator::DCS::shared_ptr DCS(double k = 1.0) {
    return gtsam::noiseModel::mEstimator::DCS::Create(k);
  }

  static gtsam::noiseModel::mEstimator::L2WithDeadZone::shared_ptr
  L2WithDeadZone(double k = 1.0) {
    return gtsam::noiseModel::mEstimator::L2WithDeadZone::Create(k);
  }

  static gtsam::noiseModel::mEstimator::AsymmetricTukey::shared_ptr
  AsymmetricTukey(double k = 4.6851) {
    return gtsam::noiseModel::mEstimator::AsymmetricTukey::Create(k);
  }

  static gtsam::noiseModel::mEstimator::AsymmetricCauchy::shared_ptr
  AsymmetricCauchy(double k = 0.1) {
    return gtsam::noiseModel::mEstimator::AsymmetricCauchy::Create(k);
  }

  static GtsamBackend::RobustKernelPtr Create(
      const std::string& type, double k = 0.0) {
    if (k == 0.0) {
      // Using the default parameters
      if (type.empty()) {
        return nullptr;
      } else if (type == "Fair") {
        return Fair();
      } else if (type == "Huber") {
        return Huber();
      } else if (type == "Cauchy") {
        return Cauchy();
      } else if (type == "Tukey") {
        return Tukey();
      } else if (type == "Welsch") {
        return Welsch();
      } else if (type == "GemanMcClure") {
        return GemanMcClure();
      } else if (type == "DCS") {
        return DCS();
      } else if (type == "L2WithDeadZone") {
        return L2WithDeadZone();
      } else if (type == "AsymmetricTukey") {
        return AsymmetricTukey();
      } else if (type == "AsymmetricCauchy") {
        return AsymmetricCauchy();
      } else {
        return nullptr;
      }
    } else {
      if (type.empty()) {
        return nullptr;
      } else if (type == "Fair") {
        return Fair(k);
      } else if (type == "Huber") {
        return Huber(k);
      } else if (type == "Cauchy") {
        return Cauchy(k);
      } else if (type == "Tukey") {
        return Tukey(k);
      } else if (type == "Welsch") {
        return Welsch(k);
      } else if (type == "GemanMcClure") {
        return GemanMcClure(k);
      } else if (type == "DCS") {
        return DCS(k);
      } else if (type == "L2WithDeadZone") {
        return L2WithDeadZone(k);
      } else if (type == "AsymmetricTukey") {
        return AsymmetricTukey(k);
      } else if (type == "AsymmetricCauchy") {
        return AsymmetricCauchy(k);
      } else {
        return nullptr;
      }
    }
  }
};

template <typename FrontendFactor>
// // Why FrontendFactor must be first?
// class GtsamBackend::BackendFactor : public gtsam::NoiseModelFactor,
//                                     public FrontendFactor {
class GtsamBackend::BackendFactor : public FrontendFactor,
                                    public gtsam::NoiseModelFactor {
  using BackendBase = gtsam::NoiseModelFactor;

 public:
  BackendFactor(
      FrontendFactor&& frontend_factor, GtsamBackend* backend,
      const NoiseModelPtr& noise_model = nullptr,
      const RobustKernelPtr& robust_kernel = nullptr)
      : FrontendFactor(std::move(frontend_factor)),
        BackendBase(
            robust_kernel
                ? gtsam::noiseModel::Robust::Create(
                      robust_kernel,
                      noise_model ? noise_model
                                  : NoiseHelper::Unit(
                                        frontend(backend).getResidualDim()))
                : noise_model,
            getOptimizableBackendKeys(backend)) {}

  BackendFactor(const BackendFactor&) = delete;
  BackendFactor& operator=(const BackendFactor&) = delete;

  gtsam::Vector unwhitenedError(
      const gtsam::Values& x,
      gtsam::OptionalMatrixVecType H = nullptr) const override {
    if (this->BackendBase::active(x)) {
      // TODO(jeffrey): How to reduce data copy?
      // Note `gtsam::Values::at<ValueType>(Key)` itself returns a copy,
      // not a reference. We have to copy the value again to `value_tuple`.
      typename FrontendFactor::VariableTypes value_tuple;
      getValues(x, &value_tuple);

      std::vector<const void*> variables;
      getVariablePointers(value_tuple, &variables);

      std::vector<gtsam::Matrix*> output_jacobians;
      std::vector<uint8_t> optimizable_mask = this->getOptimizableMask();
      getJacobianPointers(H, optimizable_mask, &output_jacobians, &value_tuple);
      return this->callEvaluateError(variables, output_jacobians);
    } else {
      return gtsam::Vector::Zero(this->BackendBase::dim());
    }
  }

 private:
  template <typename Manifold, std::size_t I>
  void getValue(
      const gtsam::Values& x,
      typename FrontendFactor::VariableTypes* value_tuple) const {
    VariableKey key = this->getVariableKey(I);
    Key backend_key = toBackendKey(key);
    if (x.exists(backend_key)) {
      // Note `gtsam::Values::at<ValueType>(Key)` itself returns a copy,
      // not a reference. We have to copy the value again to `value_tuple`.
      std::get<I>(*value_tuple) =
          x.at<XOptimizableManifold<Manifold>>(backend_key).value();
    }
  }

  template <typename... VariableTypes>
  void getValues(
      const gtsam::Values& x, std::tuple<VariableTypes...>* value_tuple) const {
    getValues(x, value_tuple, std::index_sequence_for<VariableTypes...>());
  }

  template <typename... VariableTypes, std::size_t... Is>
  void getValues(
      const gtsam::Values& x, std::tuple<VariableTypes...>* value_tuple,
      std::index_sequence<Is...>) const {
    ((getValue<VariableTypes, Is>(x, value_tuple)), ...);
  }

  template <typename... VariableTypes>
  void getVariablePointers(
      const std::tuple<VariableTypes...>& value_tuple,
      std::vector<const void*>* variables) const {
    getVariablePointers(
        value_tuple, variables, std::index_sequence_for<VariableTypes...>());
  }

  template <typename... VariableTypes, std::size_t... Is>
  void getVariablePointers(
      const std::tuple<VariableTypes...>& value_tuple,
      std::vector<const void*>* variables, std::index_sequence<Is...>) const {
    variables->resize(sizeof...(VariableTypes));
    ((variables->at(Is) = &std::get<Is>(value_tuple)), ...);
  }

  template <typename... VariableTypes>
  void getJacobianPointers(
      gtsam::OptionalMatrixVecType H,
      const std::vector<uint8_t>& optimizable_mask,
      std::vector<gtsam::Matrix*>* output_jacobians,
      const std::tuple<VariableTypes...>*) const {
    if (H == nullptr) {
      output_jacobians->resize(sizeof...(VariableTypes), nullptr);
    } else {
      ASSERT(this->getOptimizableVariableIndices().size() == H->size());
      getJacobianPointers(
          H, optimizable_mask, output_jacobians,
          std::index_sequence_for<VariableTypes...>());
    }
  }

  template <typename... VariableTypes, std::size_t... Is>
  void getJacobianPointers(
      gtsam::OptionalMatrixVecType H,
      const std::vector<uint8_t>& optimizable_mask,
      std::vector<gtsam::Matrix*>* output_jacobians,
      std::index_sequence<Is...>) const {
    output_jacobians->resize(sizeof...(Is));
    int j = 0;
    ((output_jacobians->at(Is) = optimizable_mask[Is] ? &(*H)[j++] : nullptr),
     ...);
    ASSERT(j == H->size());
  }

 private:
  void setBackend(OptimizerBackendInterface* backend) {
    if (this->optimizer_backend_ == nullptr) {
      this->optimizer_backend_ = backend;
    } else {
      ASSERT(this->optimizer_backend_ == backend);
    }
  }

  const FrontendFactor& frontend(OptimizerBackendInterface* backend = nullptr) {
    if (backend != nullptr) {
      setBackend(backend);
    }
    ASSERT(this->optimizer_backend_ != nullptr);
    return static_cast<const FrontendFactor&>(*this);
  }

  std::vector<Key> getOptimizableBackendKeys(
      OptimizerBackendInterface* backend = nullptr) {
    if (backend != nullptr) {
      setBackend(backend);
    }
    auto vkeys = this->getOptimizableVariableKeys();
    std::vector<Key> backend_keys;
    backend_keys.reserve(vkeys.size());
    for (const auto& vkey : vkeys) {
      backend_keys.push_back(toBackendKey(vkey));
    }
    LOGA(
        "getOptimizableBackendKeys for %s: %s", classname<FrontendFactor>(),
        toStr(backend_keys, [](Key key) {
          return gtsam::DefaultKeyFormatter(key);
        }).c_str());
    return backend_keys;
  }
};

}  // namespace sk4slam
