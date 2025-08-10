#pragma once

#include <Eigen/Core>
#include <unordered_set>

#include "ceres/ceres.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/optimizable_manifold.h"

namespace sk4slam {

/// @brief The Jacobian matrix type.
/// @note The Jacobians are all stored in row-major order, to be compatible
/// with
///       ceres-solver.
template <
    typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
using CeresJacobianMatrix = std::conditional_t<
    _cols == 1,
    Eigen::Matrix<Scalar, _rows, 1>,  // Use Eigen::RowMajor when cols == 1
                                      // will cause a compile error.
    Eigen::Matrix<Scalar, _rows, _cols, Eigen::RowMajor>>;

template <
    typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
using CeresJacobianMap = Eigen::Map<CeresJacobianMatrix<Scalar, _rows, _cols>>;

template <
    typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
using ConstCeresJacobianMap =
    Eigen::Map<const CeresJacobianMatrix<Scalar, _rows, _cols>>;

// Here we propose a new interface style to handle over-parameterized
// parameter blocks.
//
// First, let's look at the current interface of Ceres.
// In Ceres, over-parameterized parameter blocks are managed by the
// LocalParameterization class (or the Manifold class since v2.0.0).
// The ceres framework first computes the Jacobian of the residual
// w.r.t the global (over-parameterized) parameters, and then it applies
// the local parameterization to obtain the Jacobian w.r.t. the minimal
// local parameters.
//
// However, this approach can be inefficient, especially when computing
// the Jacobian of the residual w.r.t the local parameters is much easier
// than computing it w.r.t the global parameters.
//
// For example, if the parameter block is an SO(3) rotation and the residual
// is defined to be the rotation error (an so(3) vector), then the Jacobian of
// the residual w.r.t the local parameters is just the identity matrix. But
// if we first compute the Jacobian w.r.t the global parameters (a flattened
// 3x3 matrix) and then apply the local parameterization, we have to perform
// more computations, and it becomes quite tedious and error-prone to write
// the Jacobians.
//
// To address this issue, we propose a new interface style. The idea is to
// use the local parameters directly when evaluating the residuals and
// Jacobians, and then update the global parameters and reset the values of
// the local parameters after each optimization iteration.
//
// In this file, we follow the naming convention of the ceres framework,
// i.e. upper camel case for member functions.
//
// The `CeresManifoldBlock` class (template) is a local parameterization that is
// used to update the global parameters and reset the values of the local
// parameters after each optimization iteration. The
// `CeresManifoldBlockInterface` class is just an interface that is used to
// uniformly handle all the over-parameterized parameter blocks.
//
// The `OptimizableManifold` template parameter defines the dimensions of
// the local and the global parameters, the type of the global parameter,
// and the plus operation for the local parameterization.
// Note, while the type of the local parameter is always `Eigen::VectorXd`,
// global parameter can use any type that supports the interface `.data()`
// which returns a iterator of the global parameters. The returned `iter`
// should support the operations `++iter` (move to the next element) and
// `*iter` (return a reference to the current element).
//
// The `CeresProblem` class extends the `ceres::Problem` class and adds the
// functionality to register auto-local parameter blocks, see
// `RegisterManifoldBlock()` below.
//
// The `CeresSolve` function works with the `CeresProblem` class to run
// the optimization and update the global parameters and reset the values of the
// local parameters after each optimization iteration.

//////// Interfaces ////////

class CeresProblem;

inline ceres::TerminationType CeresSolve(
    const ceres::Solver::Options& solver_options, CeresProblem* problem,
    ceres::Solver::Summary* summary = nullptr, bool print_iterations = false);

template <typename OptimizableManifold>
class CeresManifoldBlock;

//////// Implementation ////////

class CeresManifoldBlockInterface {
 public:
  virtual ~CeresManifoldBlockInterface() {}
  virtual size_t LocalSize() const = 0;
  virtual size_t GlobalSize() const = 0;
  virtual double* LocalData() = 0;
  virtual const double* LocalData() const = 0;
  virtual const RetractionInterface* Retraction() const = 0;

 protected:
  inline void BindCeresProblem(CeresProblem* problem);
  virtual void FinalUpdate() = 0;

 protected:
  // See ceres::EvaluationCallback::PrepareForEvaluation.
  bool evaluate_jacobians_ = false;
  bool new_evaluation_point_ = false;

  mutable bool scalar_already_evaluated_ = false;
  mutable bool jet_already_evaluated_ = false;
  friend class CeresProblem;

#ifdef SK4SLAM_TEST_CERES_CONSTANT_BLOCK
  // Only for testing purposes.
 public:
  virtual double* GlobalData() = 0;
  virtual const double* GlobalData() const = 0;
#endif
};

// TODO(jeffrey):
//    - rename the classes and members.
template <typename OptimizableManifold>
class CeresManifoldBlock : public CeresManifoldBlockInterface {
 public:
  static constexpr int kLocalSize = OptimizableManifold::kDof;
  static constexpr int kGlobalSize = OptimizableManifold::kAmbientDim;
  // static_assert(kLocalSize > 0, "Local size must be positive.");
  // static_assert(kGlobalSize > 0, "Global size must be positive.");
  // static_assert(
  //     kLocalSize <= kGlobalSize,
  //     "Local size can not be larger than kGlobalSize.");

  template <typename ScalarType = double>
  using GlobalParameter = typename manifold_traits<
      typename OptimizableManifold::Value>::template Cast<ScalarType>;

  using LocalVector = Eigen::Matrix<double, kLocalSize, 1>;

  enum {
    NeedsToAlign =
        ((kLocalSize != Eigen::Dynamic && (sizeof(LocalVector) % 16) == 0) ||
         (sizeof(GlobalParameter<>) % 16) == 0)
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)

 public:
  /// @param initial_global   The initial value of the global parameter.
  template <
      typename _OptimizableManifold = OptimizableManifold,
      ENABLE_IF(_OptimizableManifold::kShareRetraction)>
  CeresManifoldBlock(
      CeresProblem* problem, const GlobalParameter<>& initial_global)
      : optimizable_(initial_global),
        problem_(problem),
        global_base_jet_(nullptr) {
    global_ = &optimizable_.value();
    *global_ = initial_global;
    local_ = ZeroLocalVec();
    local_zero_ = ZeroLocalVec();

    tmp_global_cached_ = *global_;

    BindCeresProblem(problem);
  }

  /// @p extern_global is a pointer to the external global parameter that is
  /// going to be optimized. This constructor is helpful to avoid unnecessary
  /// copies of the optimized parameters.
  /// Befor calling this constructor, * @p extern_global should already contain
  /// a reasonable initial value.
  template <
      typename _OptimizableManifold = OptimizableManifold,
      ENABLE_IF(_OptimizableManifold::kShareRetraction)>
  CeresManifoldBlock(CeresProblem* problem, GlobalParameter<>* extern_global)
      : optimizable_(GlobalParameter<>()),
        problem_(problem),
        global_base_jet_(nullptr) {
    global_ = extern_global;
    local_ = ZeroLocalVec();
    local_zero_ = ZeroLocalVec();

    tmp_global_cached_ = *global_;
    BindCeresProblem(problem);
  }

  /// Note that a reference to the @p extern_retraction will be
  /// stored inernally (See @ref OptimizableManifold::OptimizableManifold() ).
  /// So the caller must ensure that the @p extern_retraction is alive
  /// for the lifetime of this object.
  template <
      typename _OptimizableManifold = OptimizableManifold,
      ENABLE_IF(!_OptimizableManifold::kShareRetraction)>
  CeresManifoldBlock(
      CeresProblem* problem, const GlobalParameter<>& initial_global,
      const typename OptimizableManifold::Retraction* extern_retraction)
      : optimizable_(initial_global, extern_retraction),
        problem_(problem),
        global_base_jet_(nullptr) {
    global_ = &optimizable_.value();
    *global_ = initial_global;
    local_ = ZeroLocalVec();
    local_zero_ = ZeroLocalVec();

    tmp_global_cached_ = *global_;
    BindCeresProblem(problem);
  }

  /// @p extern_global is a pointer to the external global parameter that is
  /// going to be optimized. This constructor is helpful to avoid unnecessary
  /// copies of the optimized parameters.
  /// Befor calling this constructor, * @p extern_global should already contain
  /// a reasonable initial value.
  ///
  /// Also note that a reference to the @p extern_retraction will be
  /// stored inernally (See @ref OptimizableManifold::OptimizableManifold() ).
  /// So the caller must ensure that the @p extern_retraction is alive
  /// for the lifetime of this object.
  template <
      typename _OptimizableManifold = OptimizableManifold,
      ENABLE_IF(!_OptimizableManifold::kShareRetraction)>
  CeresManifoldBlock(
      CeresProblem* problem, GlobalParameter<>* extern_global,
      const typename OptimizableManifold::Retraction* extern_retraction)
      : optimizable_(GlobalParameter<>(), extern_retraction),
        problem_(problem),
        global_base_jet_(nullptr) {
    global_ = extern_global;
    local_ = ZeroLocalVec();
    local_zero_ = ZeroLocalVec();

    tmp_global_cached_ = *global_;
    BindCeresProblem(problem);
  }

  CeresManifoldBlock(const CeresManifoldBlock& other) = delete;
  CeresManifoldBlock& operator=(const CeresManifoldBlock& other) = delete;

  LocalVector ZeroLocalVec() const {
    if constexpr (kLocalSize != Eigen::Dynamic) {
      return LocalVector::Zero();
    } else {
      LocalVector zero(LocalSize());
      zero.setZero();
      return zero;
    }
  }

  template <typename ScalarType>
  GlobalParameter<ScalarType> MapToGlobal(const ScalarType* local_data) const {
    if constexpr (std::is_same<ScalarType, double>::value) {
      if (!scalar_already_evaluated_) {
        Eigen::Map<const LocalVector> new_local(local_data, LocalSize());
        tmp_global_cached_ = optimizable_.retraction()(
            *global_, LocalVector(new_local - local_zero_));
        // We use "optimizable_.retraction()(a,b)" instead of the
        // more readable "optimizable_ + b" because the latter does not
        // work when we are using external global_;

        // if (evaluate_jacobians_ && new_evaluation_point_) {
        if (evaluate_jacobians_) {
          // New step is adopted, update global_;
          UpdateGlobalAndResetLocal(tmp_global_cached_, new_local);
        }
        scalar_already_evaluated_ = true;
      }

      return tmp_global_cached_;
    } else {
      return MapToGlobalJet(local_data);
    }
  }

  const GlobalParameter<>& GetGlobal() {
    return *global_;
  }

  size_t LocalSize() const override {
    // return kLocalSize;
    return optimizable_.retraction().dof(*global_);
  }

  size_t GlobalSize() const override {
    // return kGlobalSize;
    return manifold_traits<GlobalParameter<>>::ambientDim(*global_);
  }

  double* LocalData() override {
    return local_.data();
  }

  const double* LocalData() const override {
    return local_.data();
  }

  const RetractionInterface* Retraction() const override {
    return &optimizable_.retraction();
  }

  ~CeresManifoldBlock() override {
    ReleaseBaseJet();
  }

 protected:
  template <typename LocalVectorType>
  void UpdateGlobalAndResetLocal(
      const GlobalParameter<>& new_global,
      const LocalVectorType& new_local_zero) const {
    *global_ = new_global;
    local_zero_ = new_local_zero;
  }

  void FinalUpdate() override {
    if (IsLocalZero(local_.data())) {
      // LOGA("CeresManifoldBlock::FinalUpdate(): local is zero, skip update");
      return;
    }

    LocalVector delta_local = local_ - local_zero_;
    LOGA(
        "CeresManifoldBlock::FinalUpdate(): delta_local = %s",
        toStr(delta_local.transpose()).c_str());
    UpdateGlobalAndResetLocal(
        optimizable_.retraction()(*global_, delta_local), local_);
  }

 private:
  // For AutoDiff
  using BaseJet = ceres::Jet<double, kLocalSize>;

  void ReleaseBaseJet() {
    if (global_base_jet_) {
      delete global_base_jet_;
      global_base_jet_ = nullptr;
    }
  }

  template <typename ScalarType>
  bool IsLocalZero(const ScalarType* local_data) const {
    if constexpr (std::is_same<ScalarType, double>::value) {
      for (size_t i = 0; i < LocalSize(); i++) {
        // if (std::abs(local_data[i] - local_zero_[i]) > 1e-6) {
        if (local_data[i] != local_zero_[i]) {
          return false;
        }
      }
    } else {
      for (size_t i = 0; i < LocalSize(); i++) {
        // if (std::abs(local_data[i] - local_zero_[i]) > 1e-6) {
        if (local_data[i].a != local_zero_[i]) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename JetType>
  GlobalParameter<JetType> MapToGlobalJet(const JetType* local_data) const {
    ASSERT(
        kLocalSize !=
        Eigen::Dynamic);  // auto diff only supports fixed size local parameters
    ASSERT(evaluate_jacobians_);

    if (!global_base_jet_) {
      global_base_jet_ = new GlobalParameter<BaseJet>();
    }

    // scalar_already_evaluated_
    if (!jet_already_evaluated_) {
      if (!scalar_already_evaluated_) {
        double scalar_local_data[kLocalSize];
        for (size_t i = 0; i < kLocalSize; i++) {
          scalar_local_data[i] = local_data[i].a;
        }
        MapToGlobal<double>(scalar_local_data);
      }
      ASSERT(scalar_already_evaluated_);

      auto global_data_iter = global_->data();
      auto global_base_jet_data_iter = global_base_jet_->data();
      for (size_t i = 0; i < GlobalSize(); i++) {
        (*global_base_jet_data_iter).a = *global_data_iter;
        (*global_base_jet_data_iter).v.setZero();
        ++global_base_jet_data_iter;
        ++global_data_iter;
      }

      Eigen::Matrix<BaseJet, kLocalSize, 1> local_base_jet;
      for (size_t i = 0; i < kLocalSize; i++) {
        local_base_jet[i] = BaseJet(local_data[i].a - local_zero_[i], i);
        // local_base_jet[i] = BaseJet(0.0, i);
        // Local parameters are assumed to be zero.
        ASSERT(std::abs(local_base_jet[i].a) < 1e-6);
      }

      // Though we know local_base_jet = 0 (its real part), we still need to
      // calculate it to get the Jacobian (its imaginary part).
      *global_base_jet_ =
          optimizable_.retraction()(*global_base_jet_, local_base_jet);
      jet_already_evaluated_ = true;
    }

    const auto& local0v = local_data[0].v;
    size_t jet_v_dim = local0v.size();
    int dim_offset = -1;
    for (size_t i = 0; i < jet_v_dim; i++) {
      // Find the first non-zero value (and it's assumed to be 1.0).
      if (local0v[i] > 1e-6 && std::abs(1.0 - local0v[i]) < 1e-6) {
        dim_offset = i;
        break;
      }
    }
    // LOGA("CeresManifoldBlock::MapToGlobalJet(): dim_offset = %d",
    // dim_offset);
    ASSERT(dim_offset >= 0);

    GlobalParameter<JetType> global_jet;
    auto global_jet_data_iter = global_jet.data();
    auto global_base_jet_data_iter = global_base_jet_->data();
    for (size_t i = 0; i < GlobalSize(); i++) {
      (*global_jet_data_iter).a = (*global_base_jet_data_iter).a;
      (*global_jet_data_iter).v.template block<kLocalSize, 1>(dim_offset, 0) =
          (*global_base_jet_data_iter).v;
      ++global_jet_data_iter;
      ++global_base_jet_data_iter;
    }
    return global_jet;
  }

#ifdef SK4SLAM_TEST_CERES_CONSTANT_BLOCK
  // Only for unit test.
 public:
  double* GlobalData() override {
    return global_->data();
  }
  const double* GlobalData() const override {
    return global_->data();
  }
#endif

 private:
  OptimizableManifold optimizable_;
  CeresProblem* problem_;

  LocalVector local_;
  mutable LocalVector local_zero_;

  GlobalParameter<>* global_;
  mutable GlobalParameter<> tmp_global_cached_;

  // For AutoDiff
  mutable GlobalParameter<BaseJet>* global_base_jet_;
  // We don't need extra cache for the Jet version GlobalParameter since
  // when it's needed to compute the Jet version (to evaluate the jacobian)
  // the step is already adopted and no temporary cache is needed.
};

class CeresProblem : public ceres::Problem {
  using Base = ceres::Problem;

  // Will not take ownership of the CeresManifoldBlockInterface.
  void RegisterManifoldBlock(
      CeresManifoldBlockInterface* local_parameter_block) {
    if (local_parameter_blocks_.insert(local_parameter_block).second) {
      AddParameterBlock(
          local_parameter_block->LocalData(),
          local_parameter_block->LocalSize());

#ifdef SK4SLAM_TEST_CERES_CONSTANT_BLOCK
      // Only for testing purposes.
      AddParameterBlock(
          local_parameter_block->GlobalData(),
          local_parameter_block->GlobalSize());
      SetParameterBlockConstant(local_parameter_block->GlobalData());
#endif
    }
  }

  void PrepareForEvaluation(
      bool evaluate_jacobians, bool new_evaluation_point) {
    for (CeresManifoldBlockInterface* local_parameter_block :
         local_parameter_blocks_) {
      local_parameter_block->evaluate_jacobians_ = evaluate_jacobians;
      local_parameter_block->new_evaluation_point_ = new_evaluation_point;
      local_parameter_block->scalar_already_evaluated_ = false;
      local_parameter_block->jet_already_evaluated_ = false;
    }
  }

  void FinalUpdateManifoldBlocks() {
    for (CeresManifoldBlockInterface* local_parameter_block :
         local_parameter_blocks_) {
      local_parameter_block->FinalUpdate();
    }
  }

  friend class CeresManifoldBlockInterface;
  friend class ManifoldEvaluationCallback;
  friend ceres::TerminationType CeresSolve(
      const ceres::Solver::Options&, CeresProblem*, ceres::Solver::Summary*,
      bool);
  std::unordered_set<CeresManifoldBlockInterface*> local_parameter_blocks_;

 public:
  class ManifoldEvaluationCallback : public ceres::EvaluationCallback {
   public:
    virtual ~ManifoldEvaluationCallback() {}
    ManifoldEvaluationCallback(
        CeresProblem* problem, ceres::EvaluationCallback* other = nullptr)
        : problem_(problem), other_(other) {}

    // Called before Ceres requests residuals or jacobians for a given setting
    // of
    // the parameters. User parameters (the double* values provided to the cost
    // functions) are fixed until the next call to PrepareForEvaluation(). If
    // new_evaluation_point == true, then this is a new point that is different
    // from the last evaluated point. Otherwise, it is the same point that was
    // evaluated previously (either jacobian or residual) and the user can use
    // cached results from previous evaluations.
    virtual void PrepareForEvaluation(
        bool evaluate_jacobians, bool new_evaluation_point) {
      // LOGA(
      //     "EVA ManifoldEvaluationCallback: "
      //     " (evaluate_jacobians, new_evaluation_point) = (%d, %d)",
      //     evaluate_jacobians, new_evaluation_point);
      problem_->PrepareForEvaluation(evaluate_jacobians, new_evaluation_point);
      if (other_) {
        other_->PrepareForEvaluation(evaluate_jacobians, new_evaluation_point);
      }
    }

   private:
    CeresProblem* problem_;
    ceres::EvaluationCallback* other_;
  };

 public:
#if CERES_VERSION_MAJOR >= 2
  // Ceres 2.x and later: evaluation_callback is moved to Problem::options().
  // We need to override the evaluation_callback before constructing the
  // Problem.
  explicit CeresProblem(const Options& options = Options())
      : eva_callback_(this, options.evaluation_callback),
        Base(overrideEvaluationCallback(options, &eva_callback_)) {}

 private:
  ManifoldEvaluationCallback eva_callback_;
  static Options overrideEvaluationCallback(
      const Options& options, ManifoldEvaluationCallback* new_eva_cb) {
    Options new_options = options;
    new_options.evaluation_callback = new_eva_cb;
    return new_options;
  }
#else
  // Ceres 1.x: still uses Solver::Options::evaluation_callback.
  // We don't need to adapt the constructor.
  using Base::Base;
#endif
};

inline void CeresManifoldBlockInterface::BindCeresProblem(
    CeresProblem* problem) {
  problem->RegisterManifoldBlock(this);
}

class CeresIterationRecordCallback : public ceres::IterationCallback {
 public:
  CeresIterationRecordCallback(
      double initial_trust_region_radius, bool print_iterations = false)
      : iteration_(0),
        print_iterations_(print_iterations),
        initial_trust_region_radius_(initial_trust_region_radius) {}

  virtual ~CeresIterationRecordCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    iteration_summaries_.emplace_back(summary);

    if (iteration_ == 0) {
      if (print_iterations_) {
        std::cout << "iter      cost      cost_change  |gradient|   |step|    "
                     "tr_ratio "
                     " tr_radius  ls_iter  iter_time  total_time"
                  << std::endl;  // NOLINT
      }
    }

    if (!(iteration_ > 0 && summary.iteration == 0)) {
      if (print_iterations_) {
        const char* kReportRowFormat =
            "% 4d % 8e   % 3.2e   % 3.2e  % 3.2e  % 3.2e % 3.2e     % 4d   % "
            "3.2e   % 3.2e";  // NOLINT
        std::string report_row = formatStr(
            kReportRowFormat, iteration_, summary.cost, summary.cost_change,
            summary.gradient_max_norm, summary.step_norm,
            summary.relative_decrease, summary.trust_region_radius,
            summary.linear_solver_iterations, summary.iteration_time_in_seconds,
            summary.cumulative_time_in_seconds);
        std::cout << report_row << std::endl;
      }
      ++iteration_;
    }
    return ceres::SOLVER_CONTINUE;
  }

  bool print_iterations_;
  int iteration_;
  double initial_trust_region_radius_;
  std::vector<ceres::IterationSummary> iteration_summaries_;
};

inline ceres::TerminationType CeresSolve(
    const ceres::Solver::Options& solver_options, CeresProblem* problem,
    ceres::Solver::Summary* summary, bool print_iterations) {
  // CeresManifoldBlock is incompatible with nonmonotonic_steps, and
  // incompatible with inner iterations (since we use evaluation_callback).
  ASSERT(!solver_options.use_nonmonotonic_steps);
  ASSERT(solver_options.minimizer_type == ceres::TRUST_REGION);

  ceres::Solver::Summary tmp_summary;
  if (!summary) {
    summary = &tmp_summary;
  }

  ceres::Solver::Options local_options = solver_options;
  // local_options.update_state_every_iteration = true;

  // CeresIterationRecordCallback record_callback(
  //     solver_options.initial_trust_region_radius, print_iterations);
  // if (print_iterations) {
  //   local_options.callbacks.push_back(&record_callback);
  // }

  if (print_iterations) {
    local_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
  } else {
    local_options.logging_type = ceres::SILENT;
  }

#if CERES_VERSION_MAJOR >= 2
  // Ceres 2.x and later: set evaluation_callback via Problem::options()
  // Nothing to do here.
#else
  // Ceres 1.x: still uses Solver::Options::evaluation_callback
  CeresProblem::ManifoldEvaluationCallback eva_callback(
      problem, local_options.evaluation_callback);
  local_options.evaluation_callback = &eva_callback;
#endif

  ceres::Solve(local_options, problem, summary);
  problem->FinalUpdateManifoldBlocks();
  return summary->termination_type;
}

}  // namespace sk4slam
