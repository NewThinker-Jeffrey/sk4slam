
#include "sk4slam_backends/generic_factors.h"

#include "gtsam/inference/Symbol.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SE3.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT
using gtsam::symbol_shorthand::X;

// Define the templated test fixture.
template <class Manifold>
class TestGenericFactors : public testing::Test {
  static Manifold getDefaultValue() {
    if constexpr (manifold_traits<Manifold>::kDim == Eigen::Dynamic) {
      static_assert(std::is_same_v<Manifold, VectorXd>);
      return Vector3d::Zero();
    } else if constexpr (IsVector<Manifold>) {
      return Manifold::Zero();
    } else {
      static_assert(IsLieGroup<Manifold>);
      return Manifold::Identity();
    }
  }

  using PriorFactor_ = PriorFactor<Manifold, true>;
  using BetweenFactor_ = BetweenFactor<Manifold>;
  using InterpolationFactor_ = InterpolationFactor<Manifold>;

 public:
  void testPriorFactor() {
    using DefaultRetraction = typename PriorFactor_::DefaultRetraction;
    using JacobianMatrixXd = typename PriorFactor_::JacobianMatrixXd;
    using Optimizable = OptimizableManifold<Manifold, DefaultRetraction>;
    static const Optimizable default_p(getDefaultValue());
    static const int dof = default_p.dof();
    static auto dist = MultivariateUniformDistribution<double>::standard(dof);
    const double delta_scale = 0.001;

    auto random = default_p + dist();
    PriorFactor_ factor(random.value(), X(0));
    JacobianMatrixXd J0(dof, dof);

    auto p0 = default_p + dist();
    VectorXd error_a = factor.evaluateError(p0.value(), &J0);

    VectorXd dp0 = delta_scale * dist();
    VectorXd J0dp0 = J0 * dp0;
    LOGI("dp0  : %s", toStr(dp0.transpose(), Precision(6)).c_str());
    LOGI("J0dp0: %s", toStr(J0dp0.transpose(), Precision(6)).c_str());

    VectorXd error_b = factor.evaluateError((p0 + dp0).value());
    VectorXd derror = error_b - error_a;
    VectorXd approx_derror = J0dp0;

    LOGI("error_a: %s", toStr(error_a.transpose(), Precision(6)).c_str());
    LOGI("error_b: %s", toStr(error_b.transpose(), Precision(6)).c_str());
    LOGI("derror       : %s", toStr(derror.transpose(), Precision(6)).c_str());
    LOGI(
        "approx_derror: %s",
        toStr(approx_derror.transpose(), Precision(6)).c_str());
    double r = (approx_derror - derror).norm();
    LOGI("r: %f  (thr: %f)", r, sqrt(delta_scale) * delta_scale);
    ASSERT_LE(r, sqrt(delta_scale) * delta_scale);
  }

  void testBetweenFactor() {
    using DefaultRetraction = typename BetweenFactor_::DefaultRetraction;
    using JacobianMatrixXd = typename BetweenFactor_::JacobianMatrixXd;
    using Optimizable = OptimizableManifold<Manifold, DefaultRetraction>;
    static const Optimizable default_p(getDefaultValue());
    static const int dof = default_p.dof();
    static auto dist = MultivariateUniformDistribution<double>::standard(dof);
    const double delta_scale = 0.001;

    auto random = default_p + dist();
    BetweenFactor_ factor(random.value(), X(0), X(1));
    JacobianMatrixXd J0(dof, dof);
    JacobianMatrixXd J1(dof, dof);

    auto p0 = default_p + dist();
    auto p1 = default_p + dist();
    VectorXd error_a = factor.evaluateError(p0.value(), p1.value(), &J0, &J1);

    VectorXd dp0 = delta_scale * dist();
    VectorXd dp1 = delta_scale * dist();
    VectorXd J0dp0 = J0 * dp0;
    VectorXd J1dp1 = J1 * dp1;
    LOGI("dp0  : %s", toStr(dp0.transpose(), Precision(6)).c_str());
    LOGI("dp1  : %s", toStr(dp1.transpose(), Precision(6)).c_str());
    LOGI("J0dp0: %s", toStr(J0dp0.transpose(), Precision(6)).c_str());
    LOGI("J1dp1: %s", toStr(J1dp1.transpose(), Precision(6)).c_str());

    VectorXd error_b =
        factor.evaluateError((p0 + dp0).value(), (p1 + dp1).value());
    VectorXd derror = error_b - error_a;
    VectorXd approx_derror = J0dp0 + J1dp1;

    LOGI("error_a: %s", toStr(error_a.transpose(), Precision(6)).c_str());
    LOGI("error_b: %s", toStr(error_b.transpose(), Precision(6)).c_str());
    LOGI("derror       : %s", toStr(derror.transpose(), Precision(6)).c_str());
    LOGI(
        "approx_derror: %s",
        toStr(approx_derror.transpose(), Precision(6)).c_str());
    double r = (approx_derror - derror).norm();
    LOGI("r: %f  (thr: %f)", r, sqrt(delta_scale) * delta_scale);
    ASSERT_LE(r, sqrt(delta_scale) * delta_scale);
  }

  void testInterpolationFactor() {
    using DefaultRetraction = typename InterpolationFactor_::DefaultRetraction;
    using JacobianMatrixXd = typename InterpolationFactor_::JacobianMatrixXd;
    using Optimizable = OptimizableManifold<Manifold, DefaultRetraction>;
    static const Optimizable default_p(getDefaultValue());
    static const int dof = default_p.dof();
    static auto dist = MultivariateUniformDistribution<double>::standard(dof);
    const double delta_scale = 0.001;

    const double alpha = 0.3;
    InterpolationFactor_ factor(alpha, X(0), X(1), X(2));
    JacobianMatrixXd J0(dof, dof);
    JacobianMatrixXd J1(dof, dof);
    JacobianMatrixXd J2(dof, dof);

    auto p0 = default_p + dist();
    auto p1 = default_p + dist();
    auto p2 = default_p + dist();
    VectorXd error_a =
        factor.evaluateError(p0.value(), p1.value(), p2.value(), &J0, &J1, &J2);

    VectorXd dp0 = delta_scale * dist();
    VectorXd dp1 = delta_scale * dist();
    VectorXd dp2 = delta_scale * dist();
    VectorXd J0dp0 = J0 * dp0;
    VectorXd J1dp1 = J1 * dp1;
    VectorXd J2dp2 = J2 * dp2;
    LOGI("dp0  : %s", toStr(dp0.transpose(), Precision(6)).c_str());
    LOGI("dp1  : %s", toStr(dp1.transpose(), Precision(6)).c_str());
    LOGI("dp2  : %s", toStr(dp2.transpose(), Precision(6)).c_str());
    LOGI("J0dp0: %s", toStr(J0dp0.transpose(), Precision(6)).c_str());
    LOGI("J1dp1: %s", toStr(J1dp1.transpose(), Precision(6)).c_str());
    LOGI("J2dp2: %s", toStr(J2dp2.transpose(), Precision(6)).c_str());

    VectorXd error_b = factor.evaluateError(
        (p0 + dp0).value(), (p1 + dp1).value(), (p2 + dp2).value());
    VectorXd derror = error_b - error_a;
    VectorXd approx_derror = J0dp0 + J1dp1 + J2dp2;

    LOGI("error_a: %s", toStr(error_a.transpose(), Precision(6)).c_str());
    LOGI("error_b: %s", toStr(error_b.transpose(), Precision(6)).c_str());
    LOGI("derror       : %s", toStr(derror.transpose(), Precision(6)).c_str());
    LOGI(
        "approx_derror: %s",
        toStr(approx_derror.transpose(), Precision(6)).c_str());
    double r = (approx_derror - derror).norm();
    LOGI("r: %f  (thr: %f)", r, sqrt(delta_scale) * delta_scale);
    ASSERT_LE(r, sqrt(delta_scale) * delta_scale);
  }
};

using rxSim2d = SubGLn_rx_Rn<Rp_x_SO2d, R2d>;
using rxSubGL3d = SubGLn_rx_Rn<rxSim2d, R3d>;

// Register the types to test.
using Manifolds = ::testing::Types<Vector4d, VectorXd, SE3d, rxSubGL3d>;

// NOTE: use TYPED_TEST_SUITE instead for newer versions of gtest
TYPED_TEST_SUITE(TestGenericFactors, Manifolds);

TYPED_TEST(TestGenericFactors, testPriorFactor) {
  this->testPriorFactor();
}

TYPED_TEST(TestGenericFactors, testBetweenFactor) {
  this->testBetweenFactor();
}

TYPED_TEST(TestGenericFactors, testInterpolationFactor) {
  this->testInterpolationFactor();
}

SK4SLAM_UNITTEST_ENTRYPOINT
