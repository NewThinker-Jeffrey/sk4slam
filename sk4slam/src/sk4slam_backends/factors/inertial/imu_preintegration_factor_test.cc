
#include "sk4slam_backends/factors/inertial/imu_preintegration_factor.h"

#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/ISAM2.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h"
#include "sk4slam_backends/ceres_backend.h"
#include "sk4slam_backends/gtsam_backend.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;           // NOLINT
using gtsam::symbol_shorthand::A;  // for acc bias
using gtsam::symbol_shorthand::B;  // for T_M_I (B for body = imu)
using gtsam::symbol_shorthand::G;  // for gyro bias
using gtsam::symbol_shorthand::T;  // for imu time offset
using gtsam::symbol_shorthand::V;  // for v_M_I (V for velocity)

class TestImuPreIntegrationFactor : public testing::Test {
  using JacobianMatrixXd = ImuPreIntegrationFactor::JacobianMatrixXd;
  template <int n>
  static VectorXd randomVector() {
    static auto dist = MultivariateUniformDistribution<double>::standard(n);
    return dist();
  }
  static Pose3d randomPose() {
    return Pose3d::Exp(randomVector<6>());
  }
  /// @brief   Generate a sine wave with random phase
  static std::vector<std::pair<double, double>> sineWave(
      int N = 50, double freq = 2.0, double dt = 0.005, double amplitude = 1.0,
      double phase = randomVector<1>()[0]) {
    std::vector<std::pair<double, double>> result;
    for (int i = 0; i < N; ++i) {
      result.emplace_back(i * dt, amplitude * std::sin(freq * i * dt + phase));
    }
    return result;
  }
  struct ImuIntegrationContext {
    std::vector<double> timestamps;
    std::vector<Vector3d> accels;
    std::vector<Vector3d> gyros;
    Vector3d bias_acc;
    Vector3d bias_gyro;
    ImuIntegration::State initial_state;
    ImuIntegration::State final_state;
    ImuIntegration integration;
    std::shared_ptr<ImuIntegration> pre_integration;
  };

  static std::unique_ptr<ImuIntegrationContext> createImuIntegrationContext(
      int N = 50) {
    Vector3d random_rot_axis = randomVector<3>().normalized();
    Vector3d random_acc_axis = randomVector<3>().normalized();

    std::unique_ptr<ImuIntegrationContext> iic(new ImuIntegrationContext);
    iic->bias_acc = 0.2 * randomVector<3>();
    iic->bias_gyro = 0.02 * randomVector<3>();
    iic->initial_state.pose() = randomPose();
    iic->initial_state.v() = randomVector<3>();

    auto sine_wave1 = sineWave(N);
    auto sine_wave2 = sineWave(N);
    for (int i = 0; i < sine_wave1.size(); ++i) {
      iic->timestamps.push_back(sine_wave1[i].first);
      iic->gyros.push_back(
          sine_wave1[i].second * random_rot_axis + iic->bias_gyro);
    }

    ImuIntegration::Options rot_only_options =
        ImuIntegration::Options::StateBuffer();
    rot_only_options.rotation_only = true;
    ImuIntegration rot_only(
        rot_only_options, ImuSigmas(), iic->bias_gyro, iic->bias_acc,
        iic->initial_state);
    for (int i = 0; i < iic->timestamps.size(); ++i) {
      rot_only.update(iic->timestamps[i], iic->gyros[i]);
    }

    Vector3d G(0, 0, 9.81);
    for (int i = 0; i < iic->timestamps.size(); ++i) {
      double time = iic->timestamps[i];
      ImuIntegration::State state;
      ASSERT(rot_only.retrieveState(time, &state));
      iic->accels.push_back(
          state.R().inverse() * G + sine_wave2[i].second * random_acc_axis +
          iic->bias_acc);
    }

    iic->integration = ImuIntegration(
        ImuIntegration::Options(), ImuSigmas(), iic->bias_gyro, iic->bias_acc,
        iic->initial_state);
    for (int i = 0; i < iic->timestamps.size(); ++i) {
      iic->integration.update(
          iic->timestamps[i], iic->gyros[i], iic->accels[i]);
    }

    iic->final_state = iic->initial_state;
    for (int i = 1; i < iic->timestamps.size(); ++i) {
      double dt = iic->timestamps[i] - iic->timestamps[i - 1];
      Vector3d prev_gyro = iic->gyros[i - 1];
      Vector3d prev_acc = iic->accels[i - 1];
      Vector3d gyro = iic->gyros[i];
      Vector3d acc = iic->accels[i];
      Vector3d mean_gyro = (prev_gyro + gyro) / 2.0;
      Vector3d mean_acc = (prev_acc + acc) / 2.0;
      Vector3d unbiased_gyro = mean_gyro - iic->bias_gyro;
      Vector3d unbiased_acc = mean_acc - iic->bias_acc;
      Vector3d acc_in_ref = iic->final_state.R() * unbiased_acc;
      SO3d deltaR = SO3d::Exp(unbiased_gyro * dt);

      ImuIntegration::State new_state;
      new_state.R() = iic->final_state.R() * deltaR;
      new_state.v() = iic->final_state.v() + (acc_in_ref - G) * dt;
      new_state.p() = iic->final_state.p() + iic->final_state.v() * dt +
                      0.5 * (acc_in_ref - G) * dt * dt;
      iic->final_state = new_state;
    }
    ASSERT(iic->integration.retrieveLatestState().isApprox(
        iic->final_state, 1e-6));

    // Pre-Integration
    iic->pre_integration.reset(new ImuIntegration(
        ImuIntegration::Options::PreIntegration(), ImuSigmas(), iic->bias_gyro,
        iic->bias_acc));
    for (int i = 0; i < iic->timestamps.size(); ++i) {
      iic->pre_integration->update(
          iic->timestamps[i], iic->gyros[i], iic->accels[i]);
    }

    double DT = iic->pre_integration->timeWindow();
    ImuIntegration::State pre_integ_state =
        iic->pre_integration->retrieveLatestState(
            true, 9.81, iic->initial_state.R().inverse() * Vector3d(0, 0, 1));
    pre_integ_state.R() = iic->initial_state.R() * pre_integ_state.R();
    pre_integ_state.p() = iic->initial_state.p() + iic->initial_state.v() * DT +
                          iic->initial_state.R() * pre_integ_state.p();
    pre_integ_state.v() =
        iic->initial_state.v() + iic->initial_state.R() * pre_integ_state.v();
    ASSERT(pre_integ_state.isApprox(iic->final_state, 1e-6));

    return iic;
  }

  struct Context {
    VariableKey B0, V0, G0, A0, T0, B1, V1;
    Pose3d B0_val, B1_val;
    Vector3d V0_val, V1_val;
    Vector3d G0_val;
    Vector3d A0_val;
    Vector1d T0_val;

    JacobianMatrixXd J_B0, J_V0, J_G0, J_A0, J_T0, J_B1, J_V1;
    JacobianMatrixXd *j_B0, *j_V0, *j_G0, *j_A0, *j_T0, *j_B1, *j_V1;
    std::unique_ptr<ImuIntegrationContext> iic;
  };
  static std::unique_ptr<Context> createContext() {
    auto c = std::make_unique<Context>();
    c->iic = createImuIntegrationContext();

    c->B0 = B(0);
    c->V0 = V(0);
    c->G0 = G(0);
    c->A0 = A(0);
    c->T0 = null_variable;  // T(0);
    c->B1 = B(1);
    c->V1 = V(1);

    c->B0_val = c->iic->initial_state.pose();
    c->V0_val = c->iic->initial_state.v();
    c->G0_val = c->iic->bias_gyro;
    c->A0_val = c->iic->bias_acc;
    c->T0_val = 0.001 * randomVector<1>();
    c->B1_val = c->iic->final_state.pose();
    c->V1_val = c->iic->final_state.v();

    // Allocate jacobians
    c->j_B0 = c->j_V0 = c->j_G0 = c->j_A0 = c->j_T0 = c->j_B1 = c->j_V1 =
        nullptr;

    const int kResidualDim = 9;
    if (c->B0 != null_variable) {
      c->J_B0.resize(kResidualDim, 6);
      c->j_B0 = &c->J_B0;
    }
    if (c->V0 != null_variable) {
      c->J_V0.resize(kResidualDim, 3);
      c->j_V0 = &c->J_V0;
    }
    if (c->G0 != null_variable) {
      c->J_G0.resize(kResidualDim, 3);
      c->j_G0 = &c->J_G0;
    }
    if (c->A0 != null_variable) {
      c->J_A0.resize(kResidualDim, 3);
      c->j_A0 = &c->J_A0;
    }
    if (c->T0 != null_variable) {
      c->J_T0.resize(kResidualDim, 1);
    }
    if (c->B1 != null_variable) {
      c->J_B1.resize(kResidualDim, 6);
      c->j_B1 = &c->J_B1;
    }
    if (c->V1 != null_variable) {
      c->J_V1.resize(kResidualDim, 3);
      c->j_V1 = &c->J_V1;
    }
    return c;
  }

 public:
  void testError() {
    auto c = createContext();
    ImuPreIntegrationFactor factor(
        c->iic->pre_integration,
        {c->B0, c->V0, c->G0, c->A0,  //  c->T0,
         c->B1, c->V1},
        // Use large threshold to prevent re-integration
        1.0, 1.0);
    VectorXd error = factor.evaluateError(
        c->B0_val, c->V0_val, c->G0_val, c->A0_val,  // c->T0_val,
        c->B1_val, c->V1_val);
    LOGI("Error: %s", toStr(error.transpose()).c_str());
    ASSERT_TRUE(error.norm() < 1e-6);
  }

  void testJacobian() {
    using OptimizablePose =
        OptimizableManifold<Pose3d, Pose3d::AffineLeftPerturbation>;
    auto c = createContext();
    ImuPreIntegrationFactor factor(
        c->iic->pre_integration,
        {c->B0, c->V0, c->G0, c->A0,  //  c->T0,
         c->B1, c->V1},
        // Use large threshold to prevent re-integration
        1.0, 1.0);
    VectorXd error_a = factor.evaluateError(
        c->B0_val, c->V0_val, c->G0_val, c->A0_val,                // c->T0_val,
        c->B1_val, c->V1_val, c->j_B0, c->j_V0, c->j_G0, c->j_A0,  // c->j_T0,
        c->j_B1, c->j_V1);
    LOGI("error_a: %s", toStr(error_a.transpose()).c_str());
    ASSERT_TRUE(error_a.norm() < 1e-6);

    double delta_scale = 0.001;
    VectorXd d_B0 = delta_scale * randomVector<6>();
    VectorXd d_V0 = delta_scale * randomVector<3>();
    VectorXd d_G0 = delta_scale * randomVector<3>();
    VectorXd d_A0 = delta_scale * randomVector<3>();
    VectorXd d_T0 = delta_scale * randomVector<1>();
    VectorXd d_B1 = delta_scale * randomVector<6>();
    VectorXd d_V1 = delta_scale * randomVector<3>();

    // for debug
    {
      // d_G0.setZero();
      // d_A0.setZero();

      // d_B0.head<3>().setZero();
      // d_B1.head<3>().setZero();

      // d_B0.tail<3>().setZero();
      // d_B1.tail<3>().setZero();

      // d_V1.setZero();
      // d_V0.setZero();
    }

    if (!c->j_B0) {
      d_B0.setZero();
    }
    if (!c->j_V0) {
      d_V0.setZero();
    }
    if (!c->j_G0) {
      d_G0.setZero();
    }
    if (!c->j_A0) {
      d_A0.setZero();
    }
    if (!c->j_T0) {
      d_T0.setZero();
    }
    if (!c->j_B1) {
      d_B1.setZero();
    }
    if (!c->j_V1) {
      d_V1.setZero();
    }

    VectorXd error_b = factor.evaluateError(
        OptimizablePose(c->B0_val) + d_B0, c->V0_val + d_V0, c->G0_val + d_G0,
        c->A0_val + d_A0,  // c->T0_val + d_T0,
        OptimizablePose(c->B1_val) + d_B1, c->V1_val + d_V1);

    LOGI("d_B0: %s", toStr(d_B0.transpose()).c_str());
    LOGI("d_V0: %s", toStr(d_V0.transpose()).c_str());
    LOGI("d_G0: %s", toStr(d_G0.transpose()).c_str());
    LOGI("d_A0: %s", toStr(d_A0.transpose()).c_str());
    LOGI("d_T0: %s", toStr(d_T0.transpose()).c_str());
    LOGI("d_B1: %s", toStr(d_B1.transpose()).c_str());
    LOGI("d_V1: %s", toStr(d_V1.transpose()).c_str());

    LOGI("error_b: %s", toStr(error_b.transpose()).c_str());

    VectorXd d_error = (error_b - error_a);
    LOGI("d_error: %s", toStr(d_error.transpose()).c_str());

    const int kResidualDim = 9;
    VectorXd d_error_approx(kResidualDim);
    d_error_approx.setZero();
    if (c->j_B0) {
      d_error_approx += (*(c->j_B0)) * d_B0;
    }
    if (c->j_V0) {
      d_error_approx += (*(c->j_V0)) * d_V0;
    }
    if (c->j_G0) {
      d_error_approx += (*(c->j_G0)) * d_G0;
    }
    if (c->j_A0) {
      d_error_approx += (*(c->j_A0)) * d_A0;
    }
    if (c->j_T0) {
      d_error_approx += (*(c->j_T0)) * d_T0;
    }
    if (c->j_B1) {
      d_error_approx += (*(c->j_B1)) * d_B1;
    }
    if (c->j_V1) {
      d_error_approx += (*(c->j_V1)) * d_V1;
    }

    LOGI("d_error_approx: %s", toStr(d_error_approx.transpose()).c_str());

    double diff = (d_error - d_error_approx).norm();
    double d_error_norm = d_error.norm();
    double relative_diff = diff / d_error_norm;
    LOGI(
        "diff: %f, d_error_norm: %f, relative_diff: %f", diff, d_error_norm,
        relative_diff);
    ASSERT_LE(relative_diff, 1e-2);
  }
};

TEST_F(TestImuPreIntegrationFactor, testError) {
  testError();
}

TEST_F(TestImuPreIntegrationFactor, testJacobian) {
  testJacobian();
}

TEST(TestCreateImuPreIntegrationFactor, GtsamBackend) {
  GtsamBackend backend;

  auto B0 = backend.addVariable(B(0), Pose3d::Identity());
  auto V0 = backend.addVariable(V(0), Vector3d(0, 0, 0));
  auto G0 = backend.addVariable(G(0), Vector3d(0, 0, 0));
  auto A0 = backend.addVariable(A(0), Vector3d(0, 0, 0));
  // auto T0 = backend.addVariable(T(0), Vector1d(0));
  auto B1 = backend.addVariable(B(1), Pose3d::Identity());
  auto V1 = backend.addVariable(V(1), Vector3d(0, 0, 0));

  std::shared_ptr<ImuIntegration> imu_integration(
      new ImuIntegration(ImuIntegration::Options::PreIntegration()));
  auto factor_id = backend.addFactor(ImuPreIntegrationFactor(
      imu_integration, {B0, V0, G0, A0,  //  T0
                        B1, V1}));
  auto factor_ptr = backend.getFactor<ImuPreIntegrationFactor>(factor_id);
  ASSERT_TRUE(factor_ptr);

  ASSERT_EQ(factor_ptr->getPreIntegration(), imu_integration);

  std::cout << "TestCreateImuPreIntegrationFactor GtsamBackend" << std::endl;
}

TEST(TestCreateImuPreIntegrationFactor, CeresBackend) {
  CeresBackend backend;

  Pose3d B0_val = Pose3d::Identity();
  Vector3d V0_val = Vector3d::Zero();
  Vector3d G0_val = Vector3d::Zero();
  Vector3d A0_val = Vector3d::Zero();
  // Vector1d T0_val(0);
  Pose3d B1_val = Pose3d::Identity();
  Vector3d V1_val = Vector3d::Zero();

  auto B0 = backend.addVariable(&B0_val);
  auto V0 = backend.addVariable(&V0_val);
  auto G0 = backend.addVariable(&G0_val);
  auto A0 = backend.addVariable(&A0_val);
  // auto T0 = backend.addVariable(&T0_val);
  auto B1 = backend.addVariable(&B1_val);
  auto V1 = backend.addVariable(&V1_val);

  std::shared_ptr<ImuIntegration> imu_integration(
      new ImuIntegration(ImuIntegration::Options::PreIntegration()));
  auto factor_id = backend.addFactor(ImuPreIntegrationFactor(
      imu_integration, {B0, V0, G0, A0,  //  T0
                        B1, V1}));
  auto factor_ptr = backend.getFactor<ImuPreIntegrationFactor>(factor_id);
  ASSERT_TRUE(factor_ptr);

  ASSERT_EQ(factor_ptr->getPreIntegration(), imu_integration);

  std::cout << "TestCreateImuPreIntegrationFactor CeresBackend" << std::endl;
}

SK4SLAM_UNITTEST_ENTRYPOINT
