#pragma once

#include "sk4slam_backends/factor_base.h"
#include "sk4slam_basic/reflection.h"

namespace sk4slam {

namespace generic_factors_internal {
class DefaultRetractionByManifold {
 public:
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
      using DefaultRetraction = typename Manifold::RightPerturbation;
      return RetractionInterface::defaultInstance<DefaultRetraction>();

      // if constexpr (Is_SubGLn_rx_Rn<Manifold>) {
      //   using DefaultRetraction = typename
      //   Manifold::AffineLeftPerturbation;
      //   // typename Manifold::AffineRightPerturbation;
      //   // typename Manifold::LeftPerturbation;
      //   // typename Manifold::RightPerturbation;
      //   return RetractionInterface::defaultInstance<DefaultRetraction>();
      // } else {
      //   using DefaultRetraction = typename Manifold::RightPerturbation;
      //   return RetractionInterface::defaultInstance<DefaultRetraction>();
      // }
    }
  }
};

template <typename Manifold, typename Retraction>
struct DeltaType_ {
  using type = std::conditional_t<
      IsLieGroup<Manifold>, Manifold, Vector<Retraction::kDof, double>>;
};

template <typename Manifold, typename Retraction>
class BetweenImpl;

template <typename Manifold, typename Retraction>
class InterpolationImpl;
}  // namespace generic_factors_internal

template <
    typename Manifold, bool _require_accurate_jacobian = false,
    typename _DefaultRetraction = GetDefaultRetractionByManifold<
        Manifold, generic_factors_internal::DefaultRetractionByManifold>>
class PriorFactor : public FactorBase<PriorFactor<Manifold>> {
  using Base = FactorBase<PriorFactor<Manifold>>;
  using XOptimizable = XOptimizableManifold<Manifold>;

 public:
  using DefaultRetraction = _DefaultRetraction;
  using typename Base::JacobianMatrixXd;
  DECLARE_VARIABLE_TYPES(Manifold)
  DECLARE_DEFAULT_RETRACTIONS(DefaultRetraction)

  static constexpr int kResidualDim = XOptimizable::kDof;

  int getResidualDim() const override {
    return retraction_->dof(prior_);
  }

  PriorFactor(
      const Manifold& prior_value, VariableKey key,
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>())
      : prior_(prior_value), retraction_(retraction), Base({key}) {}

  static VectorXd evaluateError(
      const Manifold& prior_value, const Manifold& x,
      JacobianMatrixXd* H = nullptr,
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>()) {
    if (!H) {
      return -retraction->section(x, prior_value);  // - (prior_value - x);
    } else {
      if constexpr (_require_accurate_jacobian) {
        JacobianMatrixXd* H_prior = nullptr;
        VectorXd error = -retraction->section(x, prior_value, H, H_prior);
        *H = -(*H);
        return error;
      } else {
        VectorXd error = -retraction->section(x, prior_value);
        ASSERT(H->rows() == H->cols());
        ASSERT(H->rows() == error.size());
        *H = JacobianMatrixXd::Identity(H->rows(), H->cols());
        return error;
      }
    }
  }

  virtual VectorXd evaluateError(
      const Manifold& x, JacobianMatrixXd* H = nullptr) const {
    return evaluateError(prior_, x, H, retraction_);
  }

 private:
  Manifold prior_;
  const DefaultRetraction* retraction_;
};

template <
    typename Manifold,
    typename _DefaultRetraction = GetDefaultRetractionByManifold<
        Manifold, generic_factors_internal::DefaultRetractionByManifold>>
class BetweenFactor : public FactorBase<BetweenFactor<Manifold>> {
  using Base = FactorBase<BetweenFactor<Manifold>>;
  using XOptimizable = XOptimizableManifold<Manifold>;

 public:
  using typename Base::JacobianMatrixXd;
  using DefaultRetraction = _DefaultRetraction;
  using DeltaType = typename generic_factors_internal::DeltaType_<
      Manifold, DefaultRetraction>::type;
  DECLARE_VARIABLE_TYPES(Manifold, Manifold)
  DECLARE_DEFAULT_RETRACTIONS(DefaultRetraction, DefaultRetraction)

  static constexpr int kResidualDim = XOptimizable::kDof;

  int getResidualDim() const override {
    if constexpr (std::is_same_v<DeltaType, Manifold>) {
      return retraction_->dof(observed_delta_);
    } else {
      return observed_delta_.rows();
    }
  }

  BetweenFactor(
      const DeltaType& delta_value, VariableKey key0, VariableKey key1,
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>())
      : observed_delta_(delta_value),
        retraction_(retraction),
        Base({key0, key1}) {}

  static VectorXd evaluateError(
      const DeltaType& observed_delta, const Manifold& x0, const Manifold& x1,
      JacobianMatrixXd* H0 = nullptr, JacobianMatrixXd* H1 = nullptr,
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>()) {
    int dof = retraction->dof(x0);
    VectorXd error(dof);
    generic_factors_internal::BetweenImpl<Manifold, DefaultRetraction>::
        evaluateError(observed_delta, x0, x1, &error, H0, H1);
    return error;
  }

  virtual VectorXd evaluateError(
      const Manifold& x0, const Manifold& x1, JacobianMatrixXd* H0 = nullptr,
      JacobianMatrixXd* H1 = nullptr) const {
    return evaluateError(observed_delta_, x0, x1, H0, H1, retraction_);
  }

 private:
  DeltaType observed_delta_;
  const DefaultRetraction* retraction_;
};

template <
    typename Manifold,
    typename _DefaultRetraction = GetDefaultRetractionByManifold<
        Manifold, generic_factors_internal::DefaultRetractionByManifold>>
class InterpolationFactor : public FactorBase<InterpolationFactor<Manifold>> {
  using Base = FactorBase<InterpolationFactor<Manifold>>;
  using XOptimizable = XOptimizableManifold<Manifold>;

 public:
  using DefaultRetraction = _DefaultRetraction;
  using typename Base::JacobianMatrixXd;
  DECLARE_VARIABLE_TYPES(Manifold, Manifold, Manifold)
  DECLARE_DEFAULT_RETRACTIONS(
      DefaultRetraction, DefaultRetraction, DefaultRetraction)

  static constexpr int kResidualDim = XOptimizable::kDof;

  int getResidualDim() const override {
    if constexpr (kResidualDim == Eigen::Dynamic) {
      return Base::template getDefaultVariableDof<Manifold>(0);
    } else {
      return kResidualDim;
    }
  }

  InterpolationFactor(
      double alpha, VariableKey key0, VariableKey key1,
      VariableKey key2,  // key2 is the interpolated variable
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>())
      : alpha_(alpha),
        observed_interpolation_(nullptr),
        Base({key0, key1, key2}),
        retraction_(retraction) {}

  InterpolationFactor(
      double alpha, const Manifold& observed_interpolation, VariableKey key0,
      VariableKey key1,
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>())
      : alpha_(alpha),
        observed_interpolation_(new Manifold(observed_interpolation)),
        Base({key0, key1, null_variable}),
        retraction_(retraction) {}

  static VectorXd evaluateError(
      double alpha, const Manifold& x0, const Manifold& x1, const Manifold& x2,
      JacobianMatrixXd* H0 = nullptr, JacobianMatrixXd* H1 = nullptr,
      JacobianMatrixXd* H2 = nullptr,
      const DefaultRetraction* retraction =
          RetractionInterface::defaultInstance<DefaultRetraction>()) {
    VectorXd error(retraction->dof(x0));
    generic_factors_internal::InterpolationImpl<Manifold, DefaultRetraction>::
        evaluateError(alpha, x0, x1, x2, &error, H0, H1, H2);
    return error;
  }

  virtual VectorXd evaluateError(
      const Manifold& x0, const Manifold& x1, const Manifold& x2,
      JacobianMatrixXd* H0 = nullptr, JacobianMatrixXd* H1 = nullptr,
      JacobianMatrixXd* H2 = nullptr) const {
    // error = x2 - ((1 - alpha) * x0 + alpha * x1)
    if (observed_interpolation_) {
      ASSERT(H2 == nullptr);
      return evaluateError(
          alpha_, x0, x1, *observed_interpolation_, H0, H1, H2, retraction_);
    } else {
      return evaluateError(alpha_, x0, x1, x2, H0, H1, H2, retraction_);
    }
  }

 private:
  double alpha_;
  const DefaultRetraction* retraction_;
  std::unique_ptr<Manifold> observed_interpolation_;
};

namespace generic_factors_internal {

template <typename Manifold, typename Retraction>
class BetweenImpl {
  static_assert(
      IsLieGroup<Manifold> || IsVector<Manifold>,
      "Only Lie groups and vectors are supported!");
  static constexpr int kDim = manifold_traits<Manifold>::kDim;
  using Scalar = typename manifold_traits<Manifold>::Scalar;
  static_assert(std::is_same_v<Scalar, double>);
  using DeltaType =
      typename generic_factors_internal::DeltaType_<Manifold, Retraction>::type;

 public:
  template <typename BetweenErrorVector, typename BetweenJacobianType>
  static void evaluateError(
      const DeltaType& observed_delta, const Manifold& x0, const Manifold& x1,
      BetweenErrorVector* error, BetweenJacobianType* H0,
      BetweenJacobianType* H1) {
    using JacobianMatrixXd = RetractionInterface::JacobianMatrixXd;
    LOGA(
        "BetweenImpl: Manifold = %s,  Retraction = %s", classname<Manifold>(),
        classname<Retraction>());
    if constexpr (IsVector<Manifold>) {
      static_assert(
          std::is_same_v<Retraction, VectorSpaceRetraction<kDim, Scalar>>);
      if (H0) {
        *H0 = JacobianMatrixXd::Identity(x0.size(), x0.size());
      }
      if (H1) {
        *H1 = -JacobianMatrixXd::Identity(x0.size(), x0.size());
      }
      *error = observed_delta - (x1 - x0);
      return;
    } else if constexpr (IsLieGroup<Manifold>) {
      using LieGroup = Manifold;
      if constexpr (std::is_same_v<
                        Retraction, typename LieGroup::LeftPerturbation>) {
        // const LieGroup predicted_delta = x1 * x0.inverse();
        const LieGroup predicted_delta_inv = x0 * x1.inverse();
        *error = LieGroup::Log(observed_delta * predicted_delta_inv);
        if (H0) {
          *H0 = LieGroup::invJl(*error) * LieGroup::Ad(observed_delta);
          // LOGI("BetweenFactor: H0 =\n%s", toStr(*H0).c_str());
        }
        if (H1) {
          *H1 = -LieGroup::invJr(*error);
          // LOGI("BetweenFactor: H1 =\n%s", toStr(*H1).c_str());
        }
      } else if constexpr (std::is_same_v<  // NOLINT
                               Retraction,
                               typename LieGroup::RightPerturbation>) {
        // const LieGroup predicted_delta = x0.inverse() * x1;
        const LieGroup predicted_delta_inv = x1.inverse() * x0;
        *error = LieGroup::Log(predicted_delta_inv * observed_delta);
        if (H0) {
          *H0 =
              LieGroup::invJr(*error) * LieGroup::Ad(observed_delta.inverse());
          // LOGI("BetweenFactor: H0 =\n%s", toStr(*H0).c_str());
        }
        if (H1) {
          *H1 = -LieGroup::invJl(*error);
          // LOGI("BetweenFactor: H1 =\n%s", toStr(*H1).c_str());
        }
      } else {
        static_assert(Is_SubGLn_rx_Rn<LieGroup>);
        static const BetweenJacobianType empty_jacobian;
        // Handle Separate Perturbations for SubGLn_rx_Rn
        using SubGLn = typename LieGroup::SubGLn;
        using Rn = typename LieGroup::Rn;
        using RetractionPart0 =
            RawType<decltype(std::declval<Retraction>().template part<0>())>;
        using RetractionPart1 =
            RawType<decltype(std::declval<Retraction>().template part<1>())>;
        auto error_part0 = error->block(0, 0, SubGLn::kDim, 1);
        auto error_part1 = error->block(SubGLn::kDim, 0, Rn::kDim, 1);
        JacobianMatrixXd H0_part0, H0_part1, H1_part0, H1_part1;
        JacobianMatrixXd* H0_part0_ptr = H0 ? &H0_part0 : nullptr;
        JacobianMatrixXd* H0_part1_ptr = H0 ? &H0_part1 : nullptr;
        JacobianMatrixXd* H1_part0_ptr = H1 ? &H1_part0 : nullptr;
        JacobianMatrixXd* H1_part1_ptr = H1 ? &H1_part1 : nullptr;
        BetweenImpl<SubGLn, RetractionPart0>::evaluateError(
            observed_delta.template part<0>(), x0.template part<0>(),
            x1.template part<0>(), &error_part0, H0_part0_ptr, H1_part0_ptr);
        BetweenImpl<Rn, RetractionPart1>::evaluateError(
            observed_delta.template part<1>(), x0.template part<1>(),
            x1.template part<1>(), &error_part1, H0_part1_ptr, H1_part1_ptr);
        if (H0) {
          *H0 = JacobianMatrixXd::Zero(error->rows(), error->rows());
          H0->block(0, 0, SubGLn::kDim, SubGLn::kDim) << H0_part0;
          H0->block(SubGLn::kDim, SubGLn::kDim, Rn::kDim, Rn::kDim) << H0_part1;
        }
        if (H1) {
          *H1 = JacobianMatrixXd::Zero(error->rows(), error->rows());
          H1->block(0, 0, SubGLn::kDim, SubGLn::kDim) << H1_part0;
          H1->block(SubGLn::kDim, SubGLn::kDim, Rn::kDim, Rn::kDim) << H1_part1;
        }
      }
      return;
    } else {
      // TODO(jeffrey): This branch is not tested.
      using Optimizable = OptimizableManifold<Manifold, Retraction>;
      if (!H0 && !H1) {
        Optimizable ox1(x1);
        *error = observed_delta - (ox1 - x0);
      } else {
        const Retraction& retraction =
            *RetractionInterface::defaultInstance<Retraction>();
        int dof = retraction.dof(x0);
        JacobianMatrixXd pw_px0, pw_px1;
        JacobianMatrixXd *pw_px0_ptr = nullptr, *pw_px1_ptr = nullptr;
        if (H0) {
          pw_px0.resize(dof, dof);
          pw_px0_ptr = &pw_px0;
        }
        if (H1) {
          pw_px1.resize(dof, dof);
          pw_px1_ptr = &pw_px1;
        }
        VectorXd w =
            retraction.section(x0, x1, pw_px0_ptr, pw_px1_ptr);  // w = ox1 - x0
        *error = observed_delta - w;
        if (H0) {
          *H0 = -pw_px0;
        }
        if (H1) {
          *H1 = -pw_px1;
        }
      }
    }
  }
};

template <typename Manifold, typename Retraction>
class InterpolationImpl {
  static_assert(
      IsLieGroup<Manifold> || IsVector<Manifold>,
      "Only Lie groups and vectors are supported!");
  static constexpr int kDim = manifold_traits<Manifold>::kDim;
  using Scalar = typename manifold_traits<Manifold>::Scalar;
  static_assert(std::is_same_v<Scalar, double>);

 public:
  template <
      typename InterpolationErrorVector, typename InterpolationJacobianType>
  static void evaluateError(
      double alpha, const Manifold& x0, const Manifold& x1, const Manifold& x2,
      InterpolationErrorVector* error, InterpolationJacobianType* H0,
      InterpolationJacobianType* H1, InterpolationJacobianType* H2) {
    using JacobianMatrixXd = RetractionInterface::JacobianMatrixXd;
    LOGA(
        "InterpolationImpl: Manifold = %s,  Retraction = %s",
        classname<Manifold>(), classname<Retraction>());
    if constexpr (IsVector<Manifold>) {
      static_assert(
          std::is_same_v<Retraction, VectorSpaceRetraction<kDim, Scalar>>);
      if (H0) {
        *H0 = (alpha - 1) * JacobianMatrixXd::Identity(x0.size(), x0.size());
      }
      if (H1) {
        *H1 = (-alpha) * JacobianMatrixXd::Identity(x0.size(), x0.size());
      }
      if (H2) {
        *H2 = JacobianMatrixXd::Identity(x0.size(), x0.size());
      }
      *error = x2 - ((1 - alpha) * x0 + alpha * x1);
    } else {
      using Optimizable = OptimizableManifold<Manifold, Retraction>;
      if (!H0 && !H1 && !H2) {
        Optimizable ox0(x0), ox1(x1), ox2(x2);
        *error = ox2 - (ox0 + alpha * (ox1 - ox0));
      } else {
        const Retraction& retraction =
            *RetractionInterface::defaultInstance<Retraction>();
        int dof = retraction.dof(x0);
        JacobianMatrixXd py_px0, py_px1;
        JacobianMatrixXd *py_px0_ptr = nullptr, *py_px1_ptr = nullptr;
        if (H0) {
          py_px0.resize(dof, dof);
          py_px0_ptr = &py_px0;
        }
        if (H1) {
          py_px1.resize(dof, dof);
          py_px1_ptr = &py_px1;
        }
        VectorXd y = retraction.section(
            x0, x1, py_px0_ptr, py_px1_ptr);  // y = ox1 - ox0
        VectorXd z = alpha * y;               // z = alpha * (ox1 - ox0)
        JacobianMatrixXd pw_px0, pw_pz;
        JacobianMatrixXd *pw_px0_ptr = nullptr, *pw_pz_ptr = nullptr;
        if (H0) {
          pw_px0.resize(dof, dof);
          pw_px0_ptr = &pw_px0;
        }
        if (H0 || H1) {
          pw_pz.resize(dof, dof);
          pw_pz_ptr = &pw_pz;
        }
        Manifold w = retraction(
            x0, z, pw_px0_ptr, pw_pz_ptr);  // w = ox0 + alpha * (ox1 - ox0)

        JacobianMatrixXd perr_pw, perr_px2;
        JacobianMatrixXd *perr_pw_ptr = nullptr, *perr_px2_ptr = nullptr;
        if (H0 || H1) {
          perr_pw.resize(dof, dof);
          perr_pw_ptr = &perr_pw;
        }
        if (H2) {
          perr_px2.resize(dof, dof);
          perr_px2_ptr = &perr_px2;
        }

        *error = retraction.section(
            w, x2, perr_pw_ptr, perr_px2_ptr);  // error = x2 - w
        if (H0) {
          *H0 = perr_pw * (pw_px0 + pw_pz * alpha * py_px0);
        }
        if (H1) {
          *H1 = perr_pw * pw_pz * alpha * py_px1;
        }
        if (H2) {
          *H2 = perr_px2;
        }
      }
      return;
    }
  }
};

}  // namespace generic_factors_internal
}  // namespace sk4slam
