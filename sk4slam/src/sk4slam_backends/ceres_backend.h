#pragma once

#include "ceres/cost_function.h"
#include "ceres/loss_function.h"
#include "sk4slam_backends/ceres/ceres_helper.h"
#include "sk4slam_backends/factor_base.h"
#include "sk4slam_backends/generic_factors.h"

namespace sk4slam {

class CeresBackend : public OptimizerBackendInterface {
 public:
  using Key = const void*;
  struct NoiseModel;
  using NoiseModelPtr = std::shared_ptr<NoiseModel>;
  using RobustKernel = ceres::LossFunction;
  // using RobustKernelPtr = RobustKernel*;
  using RobustKernelPtr = std::shared_ptr<RobustKernel>;
  using FactorId = ceres::ResidualBlockId;
  class BackendFactorInterface;
  using BackendFactorInterfacePtr = BackendFactorInterface*;
  using Graph = CeresProblem;

  struct NoiseHelper;
  struct RobustHelper;

  CeresBackend() : graph_(defaultCeresProblemOptions()) {}

  CeresBackend(const CeresBackend&) = delete;
  CeresBackend& operator=(const CeresBackend&) = delete;

  ~CeresBackend() override = default;

  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addVariable(
      Key backend_key, const Manifold& initial_value,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    static_assert(
        std::is_same_v<typename CustomizedRetraction::Manifold, Manifold>);
    using XOptimizable = XOptimizableManifold<Manifold>;
    using ManifoldBlock = CeresManifoldBlock<XOptimizable>;
    auto manifold_block = std::make_unique<ManifoldBlock>(
        &graph_, initial_value, customized_retraction);
    VariableKey key = toVariableKey(backend_key);
    manifold_blocks_[key] = std::move(manifold_block);
    if constexpr (manifold_traits<Manifold>::kDim == Eigen::Dynamic) {
      setManifoldDimensionForDynamicVariable(
          key, manifold_traits<Manifold>::dim(initial_value));
    }
    if constexpr (std::is_same_v<
                      FixedRetraction<Manifold>, CustomizedRetraction>) {
      addFixedVariableValue<Manifold>(key, initial_value);
    }
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

  /// This overload allows the external value to be updated directly
  /// during optimization, avoiding the overhead of copying the
  /// optimized value back.
  template <typename CustomizedRetraction, typename Manifold>
  VariableKey addVariable(
      Manifold* extern_value_to_optimize,
      const CustomizedRetraction* customized_retraction =
          RetractionInterface::defaultInstance<CustomizedRetraction>()) {
    static_assert(
        std::is_same_v<typename CustomizedRetraction::Manifold, Manifold>);
    using XOptimizable = XOptimizableManifold<Manifold>;
    using ManifoldBlock = CeresManifoldBlock<XOptimizable>;
    auto manifold_block = std::make_unique<ManifoldBlock>(
        &graph_, extern_value_to_optimize, customized_retraction);
    Key backend_key = extern_value_to_optimize;
    VariableKey key = toVariableKey(backend_key);
    manifold_blocks_[key] = std::move(manifold_block);
    if constexpr (manifold_traits<Manifold>::kDim == Eigen::Dynamic) {
      setManifoldDimensionForDynamicVariable(
          key, manifold_traits<Manifold>::dim(*extern_value_to_optimize));
    }
    if constexpr (std::is_same_v<
                      FixedRetraction<Manifold>, CustomizedRetraction>) {
      addFixedVariableValue<Manifold>(key, *extern_value_to_optimize);
    }
    return key;
  }

  template <typename Manifold>
  VariableKey addVariable(Manifold* extern_value_to_optimize) {
    return addVariable(
        extern_value_to_optimize,
        GlobalDefaultRetractionByManifold::get(extern_value_to_optimize));
  }

  template <typename Manifold>
  VariableKey addFixedVariable(Manifold* extern_value_to_optimize) {
    return addVariable(
        extern_value_to_optimize,
        RetractionInterface::defaultInstance<FixedRetraction<Manifold>>());
  }

  void removeVariable(Key backend_key) {
    auto variable_key = toVariableKey(backend_key);
    manifold_blocks_.erase(variable_key);
    removeDynamicVariable(variable_key);
    removeFixedVariableValue(variable_key);
  }

  template <typename Manifold>
  const Manifold& getEstimate(Key backend_key) const {
    using XOptimizable = XOptimizableManifold<Manifold>;
    using ManifoldBlock = CeresManifoldBlock<XOptimizable>;
    auto iter = manifold_blocks_.find(toVariableKey(backend_key));
    ASSERT(iter != manifold_blocks_.end());
    auto ptr = iter->second.get();
    ASSERT(ptr);
    ManifoldBlock* manifold_block = dynamic_cast<ManifoldBlock*>(ptr);
    ASSERT(manifold_block);
    return manifold_block->GetGlobal();
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
    graph_.RemoveResidualBlock(factor_id);
  }

  template <typename FrontendFactor>
  const FrontendFactor* getFactor(FactorId factor_id) {
    return dynamic_cast<const BackendFactor<FrontendFactor>*>(
        graph_.GetCostFunctionForResidualBlock(factor_id));
  }

  std::shared_ptr<ceres::Solver::Summary> solve(
      const ceres::Solver::Options& params, bool print_iterations = false) {
    // CeresManifoldBlock is incompatible with nonmonotonic_steps, and
    // incompatible with inner iterations. See comments in CeresSolve().
    ASSERT(!params.use_nonmonotonic_steps);
    ASSERT(params.minimizer_type == ceres::TRUST_REGION);
    auto summary = std::make_shared<ceres::Solver::Summary>();
    CeresSolve(params, &graph_, summary.get(), print_iterations);
    return summary;
  }

 public:
  static VariableKey toVariableKey(Key key) {
    return reinterpret_cast<uint64_t>(key);
  }

  static Key toBackendKey(VariableKey key) {
    return reinterpret_cast<Key>(key.getImpl());
  }

  class BackendFactorInterface : public ceres::CostFunction {
    friend class CeresBackend;

   protected:
    virtual RobustKernel* GetRobustKernel() const = 0;
    virtual const std::vector<double*>& GetParameterBlocks() const = 0;
  };

 protected:
  const RetractionInterface* getCustomizedRetraction(
      VariableKey key) const override {
    return getManifoldBlock(key)->Retraction();
  }

 private:
  template <typename _RobustKernelPtr = RobustKernelPtr>
  static ceres::Problem::Options defaultCeresProblemOptions() {
    ceres::Problem::Options option;
    if constexpr (std::is_same_v<
                      _RobustKernelPtr, std::shared_ptr<RobustKernel>>) {
      option.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    } else {
      static_assert(std::is_same_v<_RobustKernelPtr, RobustKernel*>);
    }
    return option;
  }

  CeresManifoldBlockInterface* getManifoldBlock(VariableKey key) const {
    return manifold_blocks_.at(key).get();
  }

  template <typename FrontendFactor>
  BackendFactorInterfacePtr createBackendFactor(
      FrontendFactor&& fronted_factor,
      const NoiseModelPtr& noise_model = nullptr,
      const RobustKernelPtr& robust_kernel = nullptr) {
    using _BackendFactor = BackendFactor<FrontendFactor>;
    return new _BackendFactor(
        std::move(fronted_factor), this, noise_model, robust_kernel);
  }

  FactorId addFactor(BackendFactorInterfacePtr backend_factor) {
    return graph_.AddResidualBlock(
        backend_factor, backend_factor->GetRobustKernel(),
        backend_factor->GetParameterBlocks());
  }

 private:
  template <typename FrontendFactor>
  class BackendFactor;

  template <typename FrontendFactor>
  friend class BackendFactor;

 private:
  std::unordered_map<VariableKey, std::unique_ptr<CeresManifoldBlockInterface>>
      manifold_blocks_;
  Graph graph_;
};

struct CeresBackend::NoiseModel {
  using CeresJacobianMapXd = CeresJacobianMap<double>;
  using CeresJacobianMatrixXd = CeresJacobianMatrix<double>;
  virtual ~NoiseModel() = default;
  virtual void whitenInPlace(VectorXd* residual) const = 0;
  virtual void whitenInPlace(CeresJacobianMapXd* jacobian) const = 0;
};

namespace ceres_noise {
template <typename Derived>
struct NoiseModelImpl : public CeresBackend::NoiseModel {
  void whitenInPlace(VectorXd* residual) const override {
    return derived()->whitenInPlace(*residual);
  }
  void whitenInPlace(CeresJacobianMapXd* jacobian) const override {
    return derived()->whitenInPlace(*jacobian);
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

class Gaussian : public NoiseModelImpl<Gaussian> {
  CeresJacobianMatrixXd sqrt_info_;
  template <typename MatrixXpr>
  explicit Gaussian(const MatrixXpr& sqrt_info) : sqrt_info_(sqrt_info) {
    ASSERT(sqrt_info_.rows() == sqrt_info_.cols());
  }

 public:
  using Ptr = Gaussian*;
  template <typename Matrix>
  void whitenInPlace(Matrix& m) const {
    m = sqrt_info_ * m;
  }
  int dim() const {
    return sqrt_info_.rows();
  }
  template <typename MatrixXpr>
  static Ptr SqrtInformation(const MatrixXpr& sqrt_info) {
    return new Gaussian(sqrt_info);
  }
  template <typename MatrixXpr>
  static Ptr Information(const MatrixXpr& info) {
    return SqrtInformation(info.llt().matrixU());
  }
  template <typename MatrixXpr>
  static Ptr Covariance(const MatrixXpr& cov) {
    return SqrtInformation(cov.llt().matrixL().solve(
        Eigen::MatrixXd::Identity(cov.rows(), cov.cols())));
  }
};

class Diagonal : public NoiseModelImpl<Diagonal> {
  VectorXd inv_sigmas_;
  template <typename VectorXpr>
  explicit Diagonal(const VectorXpr& inv_sigmas) : inv_sigmas_(inv_sigmas) {
    ASSERT(inv_sigmas_.size() > 0);
    // Every sigma must be positive
    ASSERT((inv_sigmas_.array() > 0).all());
    // Every sigma must be finite
    ASSERT((inv_sigmas_.array().isFinite()).all());
    // Every sigma must be non-NaN
    ASSERT(!(inv_sigmas_.array().isNaN()).any());
  }

 public:
  using Ptr = Diagonal*;
  template <typename Matrix>
  void whitenInPlace(Matrix& m) const {
    for (int i = 0; i < m.rows(); ++i) {
      m.row(i) *= inv_sigmas_[i];
    }
  }
  int dim() const {
    return inv_sigmas_.size();
  }
  template <typename VectorXpr>
  static Ptr Sigmas(const VectorXpr& sigmas) {
    return new Diagonal(sigmas.array().inverse());
  }
  template <typename VectorXpr>
  static Ptr Variances(const VectorXpr& variances) {
    return Sigmas(variances.cwiseSqrt());
  }
  template <typename VectorXpr>
  static Ptr Precisions(const VectorXpr& precisions) {
    return new Diagonal(precisions.cwiseSqrt());
  }
};

class Isotropic : public NoiseModelImpl<Isotropic> {
  int dim_;
  double inv_sigma_;
  Isotropic(int dim, double inv_sigma) : dim_(dim), inv_sigma_(inv_sigma) {
    ASSERT(dim_ > 0);
    ASSERT(inv_sigma_ > 0);
  }

 public:
  using Ptr = Isotropic*;
  template <typename Matrix>
  void whitenInPlace(Matrix& m) const {
    m *= inv_sigma_;
  }
  int dim() const {
    return dim_;
  }
  static Ptr Sigma(int dim, double sigma) {
    return new Isotropic(dim, 1.0 / sigma);
  }
  static Ptr Variance(int dim, double variance) {
    return Sigma(dim, sqrt(variance));
  }
  static Ptr Precision(int dim, double precision) {
    return Sigma(dim, 1.0 / precision);
  }
};

class Unit : public NoiseModelImpl<Unit> {
  int dim_;
  explicit Unit(int dim) : dim_(dim) {}

 public:
  using Ptr = Unit*;
  template <typename Matrix>
  void whitenInPlace(Matrix& m) const {}
  int dim() const {
    return dim_;
  }
  static Ptr Create(int dim) {
    return new Unit(dim);
  }
};
}  // namespace ceres_noise

struct CeresBackend::NoiseHelper {
  template <typename MatrixXpr>
  static ceres_noise::Gaussian::Ptr SqrtInfo(const MatrixXpr& R) {
    return ceres_noise::Gaussian::SqrtInformation(R);
  }

  template <typename MatrixXpr>
  static ceres_noise::Gaussian::Ptr Info(const MatrixXpr& M) {
    return ceres_noise::Gaussian::Information(M);
  }

  template <typename MatrixXpr>
  static ceres_noise::Gaussian::Ptr Cov(const MatrixXpr& covariance) {
    return ceres_noise::Gaussian::Covariance(covariance);
  }

  template <typename VectorXpr>
  static ceres_noise::Diagonal::Ptr Sigmas(const VectorXpr& sigmas) {
    return ceres_noise::Diagonal::Sigmas(sigmas);
  }

  template <typename VectorXpr>
  static ceres_noise::Diagonal::Ptr Variances(const VectorXpr& variances) {
    return ceres_noise::Diagonal::Variances(variances);
  }

  /// @param sigma   the standard deviation of the Gaussian noise model
  template <typename VectorXpr>
  static ceres_noise::Diagonal::Ptr Precisions(const VectorXpr& precisions) {
    return ceres_noise::Diagonal::Precisions(precisions);
  }

  static ceres_noise::Isotropic::Ptr Sigma(size_t dim, double sigma) {
    return ceres_noise::Isotropic::Sigma(dim, sigma);
  }

  static ceres_noise::Isotropic::Ptr Variance(size_t dim, double variance) {
    return ceres_noise::Isotropic::Variance(dim, variance);
  }

  static ceres_noise::Isotropic::Ptr Precision(size_t dim, double precision) {
    return ceres_noise::Isotropic::Precision(dim, precision);
  }

  static ceres_noise::Unit::Ptr Unit(size_t dim) {
    return ceres_noise::Unit::Create(dim);
  }
};

struct CeresBackend::RobustHelper {
  template <typename LossFunction, typename _RobustKernelPtr = RobustKernelPtr>
  using Ptr = std::conditional_t<
      std::is_same_v<_RobustKernelPtr, RobustKernel*>, LossFunction*,
      std::shared_ptr<LossFunction>>;

  static Ptr<ceres::HuberLoss> Huber(double k = 1.345) {
    return Ptr<ceres::HuberLoss>(new ceres::HuberLoss(k));
  }

  static Ptr<ceres::CauchyLoss> Cauchy(double k = 0.1) {
    return Ptr<ceres::CauchyLoss>(new ceres::CauchyLoss(k));
  }

  static Ptr<ceres::TukeyLoss> Tukey(double k = 4.6851) {
    return Ptr<ceres::TukeyLoss>(new ceres::TukeyLoss(k));
  }

  // TODO(jeffrey): set default value for k
  static Ptr<ceres::ArctanLoss> Arctan(double k) {
    return Ptr<ceres::ArctanLoss>(new ceres::ArctanLoss(k));
  }

  // TODO(jeffrey): set default value for k
  static Ptr<ceres::SoftLOneLoss> SoftLOne(double k) {
    return Ptr<ceres::SoftLOneLoss>(new ceres::SoftLOneLoss(k));
  }

  // TODO(jeffrey): set default value for a and b
  static Ptr<ceres::TolerantLoss> TolerantLoss(double a, double b) {
    return Ptr<ceres::TolerantLoss>(new ceres::TolerantLoss(a, b));
  }
};

template <typename FrontendFactor>
// class CeresBackend::BackendFactor : public FrontendFactor,
//                                     public BackendFactorInterface {
class CeresBackend::BackendFactor : public BackendFactorInterface,
                                    public FrontendFactor {
  using BackendBase = BackendFactorInterface;
  using CeresJacobianMapXd = CeresJacobianMap<double>;

 public:
  BackendFactor(
      FrontendFactor&& frontend_factor, CeresBackend* backend,
      const NoiseModelPtr& noise_model = nullptr,
      const RobustKernelPtr& robust_kernel = nullptr)
      : FrontendFactor(std::move(frontend_factor)),
        noise_model_(noise_model),
        robust_kernel_(robust_kernel),
        BackendBase() {
    setBackend(backend);
    this->BackendBase::set_num_residuals(this->getResidualDim());
    std::vector<int32_t>& parameter_block_sizes =
        *(this->BackendBase::mutable_parameter_block_sizes());
    parameter_block_sizes = this->getOptimizableVariableDofs();
    std::vector<VariableKey> keys = this->getOptimizableVariableKeys();
    ASSERT(parameter_block_sizes.size() == keys.size());
    parameter_blocks_.resize(keys.size());
    for (size_t j = 0; j < keys.size(); ++j) {
      parameter_blocks_[j] =
          this->backend()->getManifoldBlock(keys[j])->LocalData();
    }
  }

  BackendFactor(const BackendFactor&) = delete;
  BackendFactor& operator=(const BackendFactor&) = delete;

  bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians_data) const override {
    std::vector<uint8_t> optimizable_mask = this->getOptimizableMask();

    // TODO(jeffrey): How to reduce data copy?
    typename FrontendFactor::VariableTypes value_tuple;
    getValues(parameters, optimizable_mask, &value_tuple);

    std::vector<const void*> variables;
    getVariablePointers(value_tuple, &variables);

    std::vector<CeresJacobianMapXd*> output_jacobians;
    getJacobianPointers(
        jacobians_data, optimizable_mask, &output_jacobians, &value_tuple);

    VectorXd residuals_vec;
    residuals_vec = this->callEvaluateError(variables, output_jacobians);
    releaseJacobianPointers(&output_jacobians);

    if (residuals_vec.size() == 0) {
      return false;
    } else {
      Eigen::Map<VectorXd> residuals_map(residuals, this->getResidualDim());
      if (noise_model_) {
        noise_model_->whitenInPlace(&residuals_vec);
        for (CeresJacobianMapXd* jacobian : output_jacobians) {
          if (jacobian) {
            noise_model_->whitenInPlace(jacobian);
          }
        }
      }
      residuals_map = residuals_vec;
      return true;
    }
  }

 protected:
  RobustKernel* GetRobustKernel() const override {
    return getRobustKernelRawPtr(robust_kernel_);
  }
  const std::vector<double*>& GetParameterBlocks() const override {
    return parameter_blocks_;
  }

 private:
  template <typename PtrType>
  RobustKernel* getRobustKernelRawPtr(const PtrType& ptr) const {
    if constexpr (std::is_same_v<PtrType, std::shared_ptr<RobustKernel>>) {
      return ptr.get();
    } else {
      static_assert(std::is_same_v<PtrType, RobustKernel*>);
      return ptr;
    }
  }

  template <typename Manifold, std::size_t I>
  void getValue(
      double const* const* parameters,
      const std::vector<uint8_t>& optimizable_mask, int* j,
      typename FrontendFactor::VariableTypes* value_tuple) const {
    using XOptimizable = XOptimizableManifold<Manifold>;
    using ManifoldBlock = CeresManifoldBlock<XOptimizable>;
    if (!optimizable_mask[I]) {
      return;
    }

    VariableKey key = this->getVariableKey(I);
    const ManifoldBlock* manifold_block =
        dynamic_cast<const ManifoldBlock*>(backend()->getManifoldBlock(key));
    ASSERT(manifold_block);
    std::get<I>(*value_tuple) = manifold_block->MapToGlobal(parameters[(*j)++]);
  }

  template <typename... VariableTypes>
  void getValues(
      double const* const* parameters,
      const std::vector<uint8_t>& optimizable_mask,
      std::tuple<VariableTypes...>* value_tuple) const {
    getValues(
        parameters, optimizable_mask, value_tuple,
        std::index_sequence_for<VariableTypes...>());
  }

  template <typename... VariableTypes, std::size_t... Is>
  void getValues(
      double const* const* parameters,
      const std::vector<uint8_t>& optimizable_mask,
      std::tuple<VariableTypes...>* value_tuple,
      std::index_sequence<Is...>) const {
    int j = 0;
    ((getValue<VariableTypes, Is>(
         parameters, optimizable_mask, &j, value_tuple)),
     ...);
    ASSERT(j == this->getOptimizableVariableIndices().size());
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

  template <typename Manifold, std::size_t I>
  void getJacobianPointer(
      double** jacobians_data, const std::vector<uint8_t>& optimizable_mask,
      int* j, std::vector<CeresJacobianMapXd*>* output_jacobians) const {
    if (!optimizable_mask[I]) {
      output_jacobians->at(I) = nullptr;
      return;
    }

    double* jacobian_data = (optimizable_mask[I] && jacobians_data[*j])
                                ? jacobians_data[*j]
                                : nullptr;
    ++(*j);

    if (jacobian_data) {
      using XOptimizable = XOptimizableManifold<Manifold>;
      using ManifoldBlock = CeresManifoldBlock<XOptimizable>;
      VariableKey key = this->getVariableKey(I);
      const ManifoldBlock* manifold_block =
          dynamic_cast<const ManifoldBlock*>(backend()->getManifoldBlock(key));
      ASSERT(manifold_block);
      output_jacobians->at(I) = new CeresJacobianMapXd(
          jacobian_data, this->getResidualDim(), manifold_block->LocalSize());
    } else {
      output_jacobians->at(I) = nullptr;
    }
  }

  template <typename... VariableTypes>
  void getJacobianPointers(
      double** jacobians_data, const std::vector<uint8_t>& optimizable_mask,
      std::vector<CeresJacobianMapXd*>* output_jacobians,
      const std::tuple<VariableTypes...>*) const {
    if (jacobians_data == nullptr) {
      output_jacobians->resize(sizeof...(VariableTypes), nullptr);
    } else {
      getJacobianPointers<VariableTypes...>(
          jacobians_data, optimizable_mask, output_jacobians,
          std::index_sequence_for<VariableTypes...>());
    }
  }

  template <typename... VariableTypes, std::size_t... Is>
  void getJacobianPointers(
      double** jacobians_data, const std::vector<uint8_t>& optimizable_mask,
      std::vector<CeresJacobianMapXd*>* output_jacobians,
      std::index_sequence<Is...>) const {
    output_jacobians->resize(sizeof...(Is));
    int j = 0;
    ((getJacobianPointer<VariableTypes, Is>(
         jacobians_data, optimizable_mask, &j, output_jacobians)),
     ...);
    ASSERT(j == this->getOptimizableVariableIndices().size());
  }

  static void releaseJacobianPointers(
      std::vector<CeresJacobianMapXd*>* output_jacobians) {
    for (auto jacobian : *output_jacobians) {
      if (jacobian) {
        delete jacobian;
      }
    }
    output_jacobians->clear();
  }

 private:
  void setBackend(OptimizerBackendInterface* backend) {
    if (this->optimizer_backend_ == nullptr) {
      this->optimizer_backend_ = backend;
    } else {
      ASSERT(this->optimizer_backend_ == backend);
    }
  }

  const CeresBackend* backend() const {
    return static_cast<const CeresBackend*>(this->optimizer_backend_);
  }

 private:
  RobustKernelPtr robust_kernel_;
  NoiseModelPtr noise_model_;
  std::vector<double*> parameter_blocks_;
};
}  // namespace sk4slam
