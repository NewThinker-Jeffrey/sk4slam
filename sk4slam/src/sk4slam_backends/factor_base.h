#pragma once

#include <Eigen/Core>
#include <unordered_map>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_math/optimizable_manifold.h"

#define DEBUG_WITH_GTSAM_BACKEND
#ifdef DEBUG_WITH_GTSAM_BACKEND
#include <gtsam/inference/Key.h>

#include "sk4slam_pose/pose.h"
#endif  // DEBUG_WITH_GTSAM_BACKEND

namespace sk4slam {
struct VariableKey;
}  // namespace sk4slam

inline std::ostream& operator<<(
    std::ostream& os, const sk4slam::VariableKey& key);

namespace sk4slam {

struct VariableKey {
  using Impl = uint64_t;
  VariableKey(uint64_t key = 0) : key_(key) {}  // NOLINT
  VariableKey(const VariableKey& other) : key_(other.key_) {}
  VariableKey(const std::nullptr_t&) : key_(0) {}  // NOLINT
  VariableKey& operator=(const VariableKey& other) {
    key_ = other.key_;
    return *this;
  }
  VariableKey& operator=(const std::nullptr_t&) {
    key_ = 0;
    return *this;
  }

  bool operator==(const VariableKey& other) const {
    return key_ == other.key_;
  }
  bool operator!=(const VariableKey& other) const {
    return key_ != other.key_;
  }
  bool operator<(const VariableKey& other) const {
    return key_ < other.key_;
  }
  bool operator<=(const VariableKey& other) const {
    return key_ <= other.key_;
  }
  bool operator>(const VariableKey& other) const {
    return key_ > other.key_;
  }
  bool operator>=(const VariableKey& other) const {
    return key_ >= other.key_;
  }
  std::size_t hash() const {
    return key_;
  }

  bool isNull() const {
    return key_ == 0;
  }
  operator bool() const {
    return key_ != 0;
  }

  const Impl& getImpl() const {
    return key_;
  }

 protected:
  Impl key_;
  friend std::ostream&(::operator<<)(std::ostream& os, const VariableKey& key);
};

static inline const VariableKey null_variable = VariableKey(0ul);

inline std::string toStr(const VariableKey& key) {
  Oss oss;
  oss << key;
  return oss.str();
}

}  // namespace sk4slam

namespace std {
template <>
struct hash<sk4slam::VariableKey> {
  size_t operator()(const sk4slam::VariableKey& key) const {
    return key.hash();
  }
};
}  // namespace std

std::ostream& operator<<(std::ostream& os, const sk4slam::VariableKey& key) {
  // LOGA("key.key_ = %lu", key.key_);
  os << key.key_;
  return os;
}

namespace sk4slam {

template <typename Derived>
class FactorBase;

class GlobalDefaultRetractionByManifold {
 public:
  /// Returns the default retraction for the given manifold.
  ///
  /// Derivied classes can override or specialize this method to
  /// provide a different default retraction for a given manifold
  /// type or to provide a default retraction for a manifold type
  /// that is not supported by the default implementation.
  /// For example, in the derived class, you can add a specialization
  /// to provide a different default retraction for a specific manifold
  /// type:
  ///
  /// ```cpp
  /// class MyDerived : public GlobalDefaultRetractionByManifold {
  ///   using Base = GlobalDefaultRetractionByManifold;
  /// public:
  ///   using Base::get;
  ///   OVERRIDE_DEFAULT_RETRACTION(MyManifold, MyRetraction)
  /// };
  /// ```
  template <typename Manifold>
  static auto get(const Manifold*) {
    static_assert(
        IsLieGroup<Manifold> || IsVector<Manifold>,
        "Only Lie groups and vectors are supported!");
    static constexpr int kDim = manifold_traits<Manifold>::kDim;
    using Scalar = typename manifold_traits<Manifold>::Scalar;
    if constexpr (IsVector<Manifold>) {
      using DefaultRetraction = VectorSpaceRetraction<kDim, Scalar>;
      return RetractionInterface::defaultInstance<DefaultRetraction>();
    } else {
      static_assert(IsLieGroup<Manifold>);
      if constexpr (Is_SubGLn_rx_Rn<Manifold>) {
        using DefaultRetraction = typename Manifold::AffineLeftPerturbation;
        // using DefaultRetraction = typename Manifold::RightPerturbation;
        return RetractionInterface::defaultInstance<DefaultRetraction>();
      } else {
        using DefaultRetraction = typename Manifold::LeftPerturbation;
        // using DefaultRetraction = typename Manifold::RightPerturbation;
        return RetractionInterface::defaultInstance<DefaultRetraction>();
      }
    }
  }

#define OVERRIDE_DEFAULT_RETRACTION(manifold_type, retraction_type) \
  static auto get(const retraction_type*) {                         \
    return RetractionInterface::defaultInstance<retraction_type>(); \
  }
};

template <typename Manifold, typename Impl = GlobalDefaultRetractionByManifold>
using GetDefaultRetractionByManifold =
    RawType<decltype(Impl::get(std::declval<const Manifold*>())), true>;

struct OptimizerBackendInterface {
  virtual ~OptimizerBackendInterface() = default;

  template <typename Derived>
  friend class FactorBase;

  template <typename Manifold>
  const Manifold& getFixedVariableValue(VariableKey key) const {
    auto it = fixed_values_.find(key);
    ASSERT(it != fixed_values_.end());
    ASSERT(it->second.type == classname<Manifold>());
    return *static_cast<const Manifold*>(it->second.data.get());
  }

  bool hasFixedVariable(VariableKey key) const {
    return fixed_values_.find(key) != fixed_values_.end();
  }

 protected:
  virtual const RetractionInterface* getCustomizedRetraction(
      VariableKey key) const = 0;

  void setManifoldDimensionForDynamicVariable(VariableKey key, int dimension) {
    dimensions_for_dynamic_variables_[key] = dimension;
  }

  int getManifoldDimensionForDynamicVariable(VariableKey key) const {
    auto it = dimensions_for_dynamic_variables_.find(key);
    ASSERT(it != dimensions_for_dynamic_variables_.end());
    return it->second;
  }

  void removeDynamicVariable(VariableKey key) {
    dimensions_for_dynamic_variables_.erase(key);
  }

  template <typename Manifold>
  void addFixedVariableValue(VariableKey key, const Manifold& manifold) {
    auto it = fixed_values_.find(key);
    if (it != fixed_values_.end()) {
      ASSERT(it->second.type == classname<Manifold>());
      it->second.data =
          std::make_shared<Manifold>(manifold);  // Update the value
    } else {
      fixed_values_[key] = FixedValue{
          std::make_shared<Manifold>(manifold), classname<Manifold>()};
    }
  }

  void removeFixedVariableValue(VariableKey key) {
    fixed_values_.erase(key);
  }

 private:
  std::unordered_map<VariableKey, int> dimensions_for_dynamic_variables_;

  struct FixedValue {
    std::shared_ptr<void> data;
    const char* type;
    template <typename Manifold>
    FixedValue Create(const Manifold& manifold) {
      return FixedValue{
          std::make_shared<Manifold>(manifold), classname<Manifold>()};
    }
  };
  std::unordered_map<VariableKey, FixedValue> fixed_values_;
};

struct FactorInterface {
  virtual ~FactorInterface() = default;
};

template <typename Derived>
class FactorBase : public FactorInterface {
 public:
  using JacobianMatrixXd = RetractionInterface::JacobianMatrixXd;

  explicit FactorBase(const std::vector<VariableKey>& variable_keys)
      : variable_keys_(variable_keys), optimizer_backend_(nullptr) {
    ASSERT(
        variable_keys_.size() ==
        std::tuple_size_v<typename Derived::VariableTypes>);
    // LOGA("FactorBase::FactorBase(): variable_keys_.size() = %d",
    // variable_keys_.size());
    setDefaultRetractions();
  }

  /// The derived class must define the following static members.
  static constexpr int kResidualDim = Eigen::Dynamic;

  /// The derived class may override this if its residual has
  /// dynamic size.
  virtual int getResidualDim() const {
    if constexpr (std::is_same_v<
                      decltype(&Derived::getResidualDim),
                      decltype(&FactorBase::getResidualDim)>) {
      static_assert(
          Derived::kResidualDim != Eigen::Dynamic,
          "Factor with dynamic residual size must override the "
          "getResidualDim() method!");
      return Derived::kResidualDim;
    } else {
      // Shouldn't be reached if Derived::getResidualDim() is overridden.
      // return derived()->getResidualDim();
      throw std::runtime_error(formatStr(
          "Do NOT use FactorBase::getResidualDim() if the derived retraction "
          "overrides getResidualDim()! Derived = %s",
          classname<Derived>()));
    }
  }

  const VariableKey& getVariableKey(int i) const {
    return variable_keys_[i];
  }

 protected:
  const OptimizerBackendInterface* optimizer_backend_;

  /// @brief Macro for derived factor classes to specify the types of their
  /// variables.
  ///
  /// This macro allows derived classes to define the types of variables that
  /// they operate on, simplifying the declaration process.
#define DECLARE_VARIABLE_TYPES(...) \
  using VariableTypes = std::tuple<__VA_ARGS__>;

  /// @brief Macro for derived factor classes to specifie the retraction types
  /// used in linearization for each variable.
  ///
  /// Derived factor classes can use this to define the retraction type for each
  /// variable individually. However, it is usually more convenient and readable
  /// to use @ref DefaultRetractionByManifold, which specifies retraction types
  /// based on variable types.
#define DECLARE_DEFAULT_RETRACTIONS(...) \
  using DefaultRetractionTypes = std::tuple<__VA_ARGS__>;

  /// @brief Provides a default mechanism to specify retraction types based on
  /// variable types.
  ///
  /// Derived factor classes can override or extend this struct to customize the
  /// retraction types used during linearization according to their specific
  /// needs. By default, it uses `GlobalDefaultRetractionByManifold`.
  using DefaultRetractionByManifold = GlobalDefaultRetractionByManifold;

  std::vector<size_t> getOptimizableVariableIndices() const {
    std::vector<size_t> indices;
    ASSERT(optimizer_backend_);
    // LOGA("variable_keys_.size() = %d", variable_keys_.size());
    for (size_t i = 0; i < variable_keys_.size(); ++i) {
      // LOGA("i = %d, variable_keys_[i] = %s", i,
      // toStr(variable_keys_[i]).c_str());
      if (variable_keys_[i]) {
        const RetractionInterface* retraction = getCustomizedRetraction(i);
        bool is_fixed = (retraction && retraction->isFixed());
        if (!is_fixed) {
          indices.push_back(i);
        }
      }
    }
    return indices;
  }

  std::vector<uint8_t> getOptimizableMask() const {
    std::vector<uint8_t> mask(variable_keys_.size(), 0);
    std::vector<size_t> indices = getOptimizableVariableIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      mask[indices[i]] = 1;
    }
    return mask;
  }

  std::vector<VariableKey> getOptimizableVariableKeys() const {
    std::vector<size_t> indices = getOptimizableVariableIndices();
    std::vector<VariableKey> keys(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      keys[i] = variable_keys_[indices[i]];
    }
    return keys;
  }

  std::vector<int> getOptimizableVariableDofs() const {
    std::vector<int> dofs = getVariableDofs();
    std::vector<size_t> indices = getOptimizableVariableIndices();
    std::vector<int> ret(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      ret[i] = dofs[indices[i]];
    }
    return ret;
  }

  template <typename OutputJacobianMatrix>
  VectorXd callEvaluateError(
      const std::vector<const void*>& variables,
      const std::vector<OutputJacobianMatrix*>& output_jacobians) const {
    static const typename Derived::VariableTypes* vptr = nullptr;
    return evaluateError(variables, output_jacobians, vptr);
  }

  template <typename OutputJacobianMatrix>
  VectorXd evaluateErrorWithOptimizablesOnly(
      const std::vector<const void*>& optimizable_variables,
      const std::vector<OutputJacobianMatrix*>& output_optimizable_jacobians)
      const {
    std::vector<const void*> variables;
    std::vector<OutputJacobianMatrix*> output_jacobians;
    variables.resize(variable_keys_.size(), nullptr);
    output_jacobians.resize(variable_keys_.size(), nullptr);
    std::vector<size_t> optimizable_variable_indices =
        getOptimizableVariableIndices();
    ASSERT(optimizable_variable_indices.size() == optimizable_variables.size());
    ASSERT(
        optimizable_variable_indices.size() ==
        output_optimizable_jacobians.size());
    for (size_t i = 0; i < optimizable_variables.size(); ++i) {
      variables[optimizable_variable_indices[i]] = optimizable_variables[i];
      output_jacobians[optimizable_variable_indices[i]] =
          output_optimizable_jacobians[i];
    }
    return evaluateError(variables, output_jacobians);
  }

  template <typename Manifold>
  int getManifoldDimension(int i) const {
    ASSERT(optimizer_backend_);
    if constexpr (manifold_traits<Manifold>::kDim != Eigen::Dynamic) {
      return manifold_traits<Manifold>::kDim;
    } else {
      if (variable_keys_[i]) {
        // return the runtime dimension of the manifold
        return optimizer_backend_->getManifoldDimensionForDynamicVariable(
            variable_keys_[i]);
      } else {
        // return the compile-time dimension (Dynamic) of the manifold
        // (whatever, it shuoldn't be used when the key is null)
        return manifold_traits<Manifold>::kDim;
      }
    }
  }

  template <typename Manifold>
  int getDefaultVariableDof(size_t i) const {
    if (!variable_keys_[i]) {
      // null variable key means the variable is not used in this factor,
      // set its dof to 0
      return 0;
    }
    return getDefaultRetraction(i)->dimToDof(getManifoldDimension<Manifold>(i));
  }

  // For debugging
  static std::string formatVariableKeys(
      const std::vector<VariableKey>& variable_keys) {
    Oss oss;
    oss << "{";
    for (int i = 0; i < variable_keys.size(); ++i) {
      oss << toStr(variable_keys[i]);
      if (i < variable_keys.size() - 1) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  std::string formatVariableKeys() const {
    return formatVariableKeys(variable_keys_);
  }

 private:
  const RetractionInterface* getDefaultRetraction(int i) const {
    return default_retractions_[i];
  }

  const RetractionInterface* getCustomizedRetraction(int i) const {
    ASSERT(optimizer_backend_);
    if (variable_keys_[i]) {
      return optimizer_backend_->getCustomizedRetraction(variable_keys_[i]);
    } else {
      return nullptr;
    }
  }

  bool isVariableFixed(int i) const {
    ASSERT(optimizer_backend_);
    bool is_fixed = false;
    if (variable_keys_[i]) {
      is_fixed = optimizer_backend_->hasFixedVariable(variable_keys_[i]);
    }

    // Invariants check
    const RetractionInterface* retraction = getCustomizedRetraction(i);
    bool has_fixed_retraction = (retraction && retraction->isFixed());
    ASSERT(has_fixed_retraction == is_fixed);

    return is_fixed;
  }

  template <typename Manifold>
  const Manifold& getFixedValue(int i) const {
    const FixedRetraction<Manifold>* fixed_retraction =
        static_cast<const FixedRetraction<Manifold>*>(
            getCustomizedRetraction(i));
    ASSERT(fixed_retraction);
    // return fixed_retraction->value();
    ASSERT(optimizer_backend_);
    return optimizer_backend_->getFixedVariableValue<Manifold>(
        variable_keys_[i]);
  }

  std::vector<int> getVariableDofs() const {
    static const typename Derived::VariableTypes* vptr = nullptr;
    return getVariableDofs(vptr);
  }

  template <typename... VariableTypes>
  std::vector<int> getVariableDofs(
      const std::tuple<VariableTypes...>* vptr) const {
    return getVariableDofs(vptr, std::index_sequence_for<VariableTypes...>());
  }

  template <typename... VariableTypes, std::size_t... Is>
  std::vector<int> getVariableDofs(
      const std::tuple<VariableTypes...>* vptr,
      std::index_sequence<Is...>) const {
    return {getVariableDof<VariableTypes, Is>()...};
  }

  template <typename Manifold, std::size_t I>
  int getVariableDof() const {
    if (variable_keys_[I]) {
      const RetractionInterface* retraction = getCustomizedRetraction(I);
      ASSERT(retraction);
      return retraction->dimToDof(getManifoldDimension<Manifold>(I));
    } else {
      // null variable key means the variable is not used in this factor,
      // set its dof to 0
      return 0;
    }
  }

  std::vector<int> getDefaultVariableDofs() const {
    static const typename Derived::VariableTypes* vptr = nullptr;
    return getDefaultVariableDofs(vptr);
  }

  template <typename... VariableTypes>
  std::vector<int> getDefaultVariableDofs(
      const std::tuple<VariableTypes...>* vptr) const {
    return getDefaultVariableDofs(
        vptr, std::index_sequence_for<VariableTypes...>());
  }

  template <typename... VariableTypes, std::size_t... Is>
  std::vector<int> getDefaultVariableDofs(
      const std::tuple<VariableTypes...>* vptr,
      std::index_sequence<Is...>) const {
    return {getDefaultVariableDof<VariableTypes>(Is)...};
  }

 private:
  template <typename OutputJacobianMatrix, typename... VariableTypes>
  VectorXd evaluateError(
      const std::vector<const void*>& variables,
      const std::vector<OutputJacobianMatrix*>& output_jacobians,
      const std::tuple<VariableTypes...>* vptr) const {
    return evaluateError(
        variables, output_jacobians, vptr,
        std::index_sequence_for<VariableTypes...>());
  }

  template <
      typename OutputJacobianMatrix, typename... VariableTypes,
      std::size_t... Is>
  VectorXd evaluateError(
      const std::vector<const void*>& variables,
      const std::vector<OutputJacobianMatrix*>& output_jacobians,
      const std::tuple<VariableTypes...>* vptr,
      std::index_sequence<Is...>) const {
    std::vector<JacobianMatrixXd> jacobians_under_default_retractions;
    std::vector<JacobianMatrixXd*> jacobian_ptrs_under_default_retractions;
    jacobians_under_default_retractions.resize(sizeof...(VariableTypes));
    jacobian_ptrs_under_default_retractions.resize(
        sizeof...(VariableTypes), nullptr);
    std::vector<int> default_variable_dofs = getDefaultVariableDofs();
    ASSERT(default_variable_dofs.size() == sizeof...(VariableTypes));
    int residual_size = derived()->getResidualDim();
    for (int i = 0; i < sizeof...(VariableTypes); ++i) {
      if (!variable_keys_[i] || isVariableFixed(i)) {
        ASSERT(output_jacobians[i] == nullptr);
      }
      if (output_jacobians[i] == nullptr) {
        continue;
      }
      // allocates memory for jacobian under default retraction
      jacobians_under_default_retractions[i].resize(
          // output_jacobians[i]->rows(), default_variable_dofs[i]);
          residual_size, default_variable_dofs[i]);
      jacobian_ptrs_under_default_retractions[i] =
          &jacobians_under_default_retractions[i];
    }

    VectorXd ret = derived()->evaluateError(
        (getValue<VariableTypes>(variables, Is))...,
        (jacobian_ptrs_under_default_retractions[Is])...);

    // Convert the jacobians if the evaluateError() function succeeded.
    if (ret.size() > 0) {
      ((convertJacobian<VariableTypes>(
           variables, jacobian_ptrs_under_default_retractions, output_jacobians,
           Is)),
       ...);
    }

    return ret;
  }

  template <typename Manifold>
  const Manifold& getValue(
      const std::vector<const void*>& variables, int i) const {
    if (!variable_keys_[i]) {
      // null variable key means the variable is not used in this factor,
      // so we can return any value
      static const Manifold dummy;
      return dummy;
    } else if (isVariableFixed(i)) {
      const Manifold& ret = getFixedValue<Manifold>(i);
      if constexpr (manifold_traits<Manifold>::kDim == Eigen::Dynamic) {
        // check dimension
        ASSERT(
            manifold_traits<Manifold>::dim(ret) ==
            getManifoldDimension<Manifold>(i));
      }
      return ret;
    } else {
      const Manifold& ret = *static_cast<const Manifold*>(variables[i]);
#ifdef DEBUG_WITH_GTSAM_BACKEND
      if constexpr (std::is_same_v<Pose3d, Manifold>) {
        LOGA(
            "FactorBase::getValue: Pose3d %s (Gtsam format key): rot = %s, pos "
            "= %s",
            gtsam::DefaultKeyFormatter(variable_keys_[i].getImpl()).c_str(),
            toOneLineStr(ret.rotation().matrix()).c_str(),
            toStr(ret.translation().transpose()).c_str());
      }
#endif  // DEBUG_WITH_GTSAM_BACKEND
      if constexpr (manifold_traits<Manifold>::kDim == Eigen::Dynamic) {
        // check dimension
        ASSERT(
            manifold_traits<Manifold>::dim(ret) ==
            getManifoldDimension<Manifold>(i));
      }
      return ret;
    }
  }

  template <
      typename Manifold, typename InputJacobianMatrix,
      typename OutputJacobianMatrix>
  void convertJacobian(
      const std::vector<const void*>& variables,
      const std::vector<InputJacobianMatrix*>& input_jacobians,
      const std::vector<OutputJacobianMatrix*>& output_jacobians, int i) const {
    if (!variable_keys_[i]) {
      // null variable key means the variable is not used in this factor,
      // so we shouldn't compute its jacobian.
      ASSERT(!input_jacobians[i]);
      ASSERT(!output_jacobians[i]);
      return;
    }
    if (input_jacobians[i] == nullptr) {
      ASSERT(!output_jacobians[i]);
      return;
    }

    // Convert the jacobian under default retraction to the jacobian
    // under the customized retraction.
    const RetractionInterface* customized_retraction =
        getCustomizedRetraction(i);
    const RetractionInterface* default_retraction = getDefaultRetraction(i);
    if (customized_retraction && customized_retraction != default_retraction &&
        *default_retraction != *customized_retraction) {
#ifdef DEBUG_WITH_GTSAM_BACKEND
      LOGA(
          "FactorBase: need transformJacobian for variable %s (Gtsam format "
          "key), Factor = %s, i = %d",
          gtsam::DefaultKeyFormatter(variable_keys_[i].getImpl()).c_str(),
          classname<Derived>(), i);
#endif  // DEBUG_WITH_GTSAM_BACKEND
      *output_jacobians[i] =
          customized_retraction
              ->transformJacobian<OutputJacobianMatrix::ColsAtCompileTime>(
                  *input_jacobians[i], getValue<Manifold>(variables, i),
                  *default_retraction);
    } else {
      // *output_jacobians[i] = *input_jacobians[i];
      *output_jacobians[i] = std::move(*input_jacobians[i]);
    }
  }

 private:
  using DefaultRetractionTypes = void;

  template <typename DefaultRetractionByManifold, typename VariableTuple>
  struct DefaultRetractionTypesForVariableTypes {
    using type = void;
  };
  template <typename DefaultRetractionByManifold, typename... VariableTypes>
  struct DefaultRetractionTypesForVariableTypes<
      DefaultRetractionByManifold, std::tuple<VariableTypes...>> {
    using type = std::tuple<GetDefaultRetractionByManifold<VariableTypes>...>;
  };

  void setDefaultRetractions() {
    using DerivedDefaultRetractionTypes =
        typename Derived::DefaultRetractionTypes;
    using DerivedDefaultRetractionByManifold =
        typename Derived::DefaultRetractionByManifold;
    if constexpr (std::is_same_v<DerivedDefaultRetractionTypes, void>) {
      using VariableTypes = typename Derived::VariableTypes;
      using DefaultRetractionTypes =
          typename DefaultRetractionTypesForVariableTypes<
              DerivedDefaultRetractionByManifold, VariableTypes>::type;
      static_assert(!std::is_same_v<DefaultRetractionTypes, void>);
      static_assert(
          std::tuple_size_v<DefaultRetractionTypes> ==
          std::tuple_size_v<typename Derived::VariableTypes>);
      static const DefaultRetractionTypes* ptr = nullptr;
      setDefaultRetractions(ptr);
    } else {
      using DefaultRetractionTypes = DerivedDefaultRetractionTypes;
      static_assert(
          std::tuple_size_v<DefaultRetractionTypes> ==
          std::tuple_size_v<typename Derived::VariableTypes>);
      static const DefaultRetractionTypes* ptr = nullptr;
      setDefaultRetractions(ptr);
    }
  }

  template <typename... DefaultRetractionTypes>
  void setDefaultRetractions(const std::tuple<DefaultRetractionTypes...>* ptr) {
    setDefaultRetractions(
        ptr, std::index_sequence_for<DefaultRetractionTypes...>{});
  }
  template <typename... DefaultRetractionTypes, std::size_t... Is>
  void setDefaultRetractions(
      const std::tuple<DefaultRetractionTypes...>* ptr,
      std::index_sequence<Is...>) {
    default_retractions_.resize(sizeof...(Is));
    ((default_retractions_[Is] =
          RetractionInterface::defaultInstance<DefaultRetractionTypes>()),
     ...);
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
  Derived* derived() {
    return static_cast<Derived*>(this);
  }

 private:
  std::vector<VariableKey> variable_keys_;
  std::vector<const RetractionInterface*>
      default_retractions_;  // TODO(jeffrey): make it static?
};

template <typename Derived, typename ParentFactor>
class SubFactorBase : public ParentFactor {
 public:
  using ParentFactor::ParentFactor;
  using typename ParentFactor::JacobianMatrixXd;

  template <typename OutputJacobianMatrix>
  VectorXd callEvaluateError(
      const std::vector<const void*>& variables,
      const std::vector<OutputJacobianMatrix*>& output_jacobians) const {
    std::vector<JacobianMatrixXd*> full_jacobian_ptrs(
        output_jacobians.size(), nullptr);
    std::vector<JacobianMatrixXd> full_jacobians(output_jacobians.size());
    int full_error_dim = ParentFactor::getResidualDim();
    int sub_error_dim = derived()->getResidualDim();

    for (size_t i = 0; i < output_jacobians.size(); ++i) {
      if (output_jacobians[i] != nullptr) {
        full_jacobian_ptrs[i] = &full_jacobians[i];
      }
    }
    auto full_error =
        ParentFactor::callEvaluateError(variables, full_jacobian_ptrs);
    ASSERT(full_error.rows() == full_error_dim);
    VectorXd sub_error;
    derived()->getSubError(full_error, &sub_error);
    ASSERT(sub_error.rows() == sub_error_dim);

    for (size_t i = 0; i < output_jacobians.size(); ++i) {
      if (output_jacobians[i] != nullptr) {
        ASSERT(full_jacobians[i].rows() == full_error_dim);
        derived()->getSubJacobian(full_jacobians[i], output_jacobians[i]);
        ASSERT(output_jacobians[i]->rows() == sub_error_dim);
        ASSERT(output_jacobians[i]->cols() == full_jacobians[i].cols());
      }
    }
    return sub_error;
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

template <typename ParentFactor, int _start_row, int _n_rows>
class SubFactorByRowBlock
    : public SubFactorBase<
          SubFactorByRowBlock<ParentFactor, _start_row, _n_rows>,
          ParentFactor> {
  using Base = SubFactorBase<SubFactorByRowBlock, ParentFactor>;

 public:
  using Base::Base;
  using typename Base::JacobianMatrixXd;

  int getResidualDim() const override {
    return _n_rows;
  }

  template <typename OutputJacobianMatrix>
  void getSubJacobian(
      const JacobianMatrixXd& full_jacobian,
      OutputJacobianMatrix* output_jacobian) const {
    *output_jacobian =
        full_jacobian.block(_start_row, 0, _n_rows, output_jacobian->cols());
  }

  void getSubError(const VectorXd& full_error, VectorXd* output_error) const {
    *output_error = full_error.segment(_start_row, _n_rows);
  }
};

template <typename ParentFactor, int _tail_rows>
using SubFactorByHeadRows = SubFactorByRowBlock<ParentFactor, 0, _tail_rows>;

template <typename ParentFactor, int _tail_rows>
class SubFactorByTailRows
    : public SubFactorBase<
          SubFactorByTailRows<ParentFactor, _tail_rows>, ParentFactor> {
  using Base = SubFactorBase<SubFactorByTailRows, ParentFactor>;

 public:
  using Base::Base;
  using typename Base::JacobianMatrixXd;

  int getResidualDim() const override {
    return _tail_rows;
  }

  template <typename OutputJacobianMatrix>
  void getSubJacobian(
      const JacobianMatrixXd& full_jacobian,
      OutputJacobianMatrix* output_jacobian) const {
    *output_jacobian = full_jacobian.bottomRows(_tail_rows);
  }

  void getSubError(const VectorXd& full_error, VectorXd* output_error) const {
    *output_error = full_error.tail(_tail_rows);
  }
};

template <typename ParentFactor, int... _row_indices>
class SubFactorByRowIndices
    : public SubFactorBase<
          SubFactorByRowIndices<ParentFactor, _row_indices...>, ParentFactor> {
  using Base = SubFactorBase<SubFactorByRowIndices, ParentFactor>;

 public:
  using Base::Base;
  using typename Base::JacobianMatrixXd;

  int getResidualDim() const override {
    return sizeof...(_row_indices);
  }

  template <typename OutputJacobianMatrix>
  void getSubJacobian(
      const JacobianMatrixXd& full_jacobian,
      OutputJacobianMatrix* output_jacobian) const {
    int j = 0;
    (getSubValue<_row_indices>(j++, full_jacobian, output_jacobian), ...);
    ASSERT(j == sizeof...(_row_indices));
  }

  void getSubError(const VectorXd& full_error, VectorXd* output_error) const {
    int j = 0;
    (getSubValue<_row_indices>(j++, full_error, output_error), ...);
    ASSERT(j == sizeof...(_row_indices));
  }

 private:
  template <int I, typename FullValue, typename SubValue>
  void getSubValue(
      int j, const FullValue& full_value, SubValue* sub_value) const {
    sub_value->row(j) = full_value.row(I);
  }
};

}  // namespace sk4slam
