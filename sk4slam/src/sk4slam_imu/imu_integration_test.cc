
#include "sk4slam_imu/imu_integration.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

class TestImuIntegration : public testing::Test {
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
  struct Context {
    std::vector<double> timestamps;
    std::vector<Vector3d> accels;
    std::vector<Vector3d> gyros;
    Vector3d bias_acc;
    Vector3d bias_gyro;
    ImuIntegration::State initial_state;
    ImuIntegration::State final_state;
    ImuIntegration integration;
  };

  static std::unique_ptr<Context> createContext(int N = 50) {
    Vector3d random_rot_axis = randomVector<3>().normalized();
    Vector3d random_acc_axis = randomVector<3>().normalized();

    std::unique_ptr<Context> c(new Context);
    c->bias_acc = 0.2 * randomVector<3>();
    c->bias_gyro = 0.02 * randomVector<3>();
    c->initial_state.pose() = randomPose();
    c->initial_state.v() = randomVector<3>();

    auto sine_wave1 = sineWave(N);
    auto sine_wave2 = sineWave(N);
    for (int i = 0; i < sine_wave1.size(); ++i) {
      c->timestamps.push_back(sine_wave1[i].first);
      c->gyros.push_back(sine_wave1[i].second * random_rot_axis + c->bias_gyro);
    }

    ImuIntegration::Options rot_only_options =
        ImuIntegration::Options::StateBuffer();
    rot_only_options.rotation_only = true;
    ImuIntegration rot_only(
        rot_only_options, ImuSigmas(), c->bias_gyro, c->bias_acc,
        c->initial_state);
    for (int i = 0; i < c->timestamps.size(); ++i) {
      rot_only.update(c->timestamps[i], c->gyros[i]);
    }

    Vector3d G(0, 0, 9.81);
    for (int i = 0; i < c->timestamps.size(); ++i) {
      double time = c->timestamps[i];
      ImuIntegration::State state;
      ASSERT(rot_only.retrieveState(time, &state));
      c->accels.push_back(
          state.R().inverse() * G + sine_wave2[i].second * random_acc_axis +
          c->bias_acc);
    }

    c->integration = ImuIntegration(
        ImuIntegration::Options(), ImuSigmas(), c->bias_gyro, c->bias_acc,
        c->initial_state);
    for (int i = 0; i < c->timestamps.size(); ++i) {
      c->integration.update(c->timestamps[i], c->gyros[i], c->accels[i]);
    }

    c->final_state = c->initial_state;
    for (int i = 1; i < c->timestamps.size(); ++i) {
      double dt = c->timestamps[i] - c->timestamps[i - 1];
      Vector3d prev_gyro = c->gyros[i - 1];
      Vector3d prev_acc = c->accels[i - 1];
      Vector3d gyro = c->gyros[i];
      Vector3d acc = c->accels[i];
      Vector3d mean_gyro = (prev_gyro + gyro) / 2.0;
      Vector3d mean_acc = (prev_acc + acc) / 2.0;
      Vector3d unbiased_gyro = mean_gyro - c->bias_gyro;
      Vector3d unbiased_acc = mean_acc - c->bias_acc;
      Vector3d acc_in_ref = c->final_state.R() * unbiased_acc;
      SO3d deltaR = SO3d::Exp(unbiased_gyro * dt);

      ImuIntegration::State new_state;
      new_state.R() = c->final_state.R() * deltaR;
      new_state.v() = c->final_state.v() + (acc_in_ref - G) * dt;
      new_state.p() = c->final_state.p() + c->final_state.v() * dt +
                      0.5 * (acc_in_ref - G) * dt * dt;
      c->final_state = new_state;
    }
    return c;
  }

  static void printState(
      const ImuIntegration::State& state, const std::string& tag = "") {
    LOGI(
        "%s:\np = %s,  v = %s, R = \n%s", tag.c_str(),
        toStr(state.p().transpose()).c_str(),
        toStr(state.v().transpose()).c_str(),
        toStr(state.R().matrix()).c_str());
  }

 public:
  void test() {
    auto c = createContext();
    printState(c->initial_state, "initial state");
    printState(c->final_state, "final state");
    printState(c->integration.retrieveLatestState(), "integration final state");

    ImuIntegration::State interpolated_state;
    double interpolated_time = c->timestamps.back() - 0.001;
    ASSERT_TRUE(
        c->integration.retrieveState(interpolated_time, &interpolated_state));
    printState(interpolated_state, "interpolated final state");

    ASSERT_TRUE(
        c->integration.retrieveLatestState().isApprox(c->final_state, 1e-6));
  }

  void testJacobian(int N, double delta_scale = 1e-3) {
    auto c = createContext(N);
    printState(c->initial_state, "initial state");
    ImuIntegration::State old_integration_state =
        c->integration.retrieveLatestState();
    printState(old_integration_state, "old_integration_state");

    Eigen::MatrixXd J_state = c->integration.getLatestResult().J_state;
    Eigen::MatrixXd J_bias = c->integration.getLatestResult().J_bias;
    LOGI(
        "jacobian w.r.t. initial state =\n%s",
        toStr(J_state, Precision(8)).c_str());
    LOGI("jacobian w.r.t. bias =\n%s", toStr(J_bias, Precision(8)).c_str());

    Eigen::MatrixXd Q = c->integration.getLatestResult().process_noise_cov;
    LOGI("process noise cov =\n%s", toStr(Q, Precision(8)).c_str());

    VectorXd d_state = delta_scale * randomVector<9>();
    // d_state.setZero();
    LOGI("d_state = %s", toStr(d_state.transpose(), Precision(8)).c_str());
    VectorXd d_bg = delta_scale * randomVector<3>();
    // d_bg.setZero();
    LOGI("d_bg = %s", toStr(d_bg.transpose(), Precision(8)).c_str());
    VectorXd d_ba = delta_scale * randomVector<3>();
    // d_ba.setZero();
    LOGI("d_ba = %s", toStr(d_ba.transpose(), Precision(8)).c_str());

    c->integration.repropagate(
        c->bias_gyro + d_bg, c->bias_acc + d_ba,
        ImuIntegration::OptimizableState(c->initial_state) + d_state);
    ImuIntegration::State new_integration_state =
        c->integration.retrieveLatestState();
    printState(new_integration_state, "new_integration_state");

    VectorXd state_change =
        ImuIntegration::OptimizableState(new_integration_state) -
        old_integration_state;

    Vector<6> d_bias;
    d_bias << d_bg, d_ba;
    VectorXd approx_state_change = J_state * d_state + J_bias * d_bias;
    LOGI(
        "       state_change = %s",
        toStr(state_change.transpose(), Precision(8)).c_str());
    LOGI(
        "approx_state_change = %s",
        toStr(approx_state_change.transpose(), Precision(8)).c_str());

    double approx_error = (state_change - approx_state_change).norm();
    double relative_approx_error = approx_error / state_change.norm();
    LOGI(
        "approx_error = %.8f, state_change.norm = %.8f, relative_approx_error "
        "= %f",
        approx_error, state_change.norm(), relative_approx_error);
    ASSERT_LE(relative_approx_error, 1e-3);
  }

  void testCov() {
    Eigen::MatrixXd random_mat = Eigen::MatrixXd::Random(15, 15);
    Eigen::MatrixXd prior_cov = random_mat * random_mat.transpose();

    auto c = createContext();
    Eigen::MatrixXd integration_cov =
        c->integration.propagateCovariance(prior_cov, true);

    ImuEKFPropagation ekf_propagation;
    ekf_propagation.init(
        prior_cov, c->initial_state, c->bias_gyro, c->bias_acc);
    for (int i = 0; i < c->timestamps.size(); ++i) {
      ekf_propagation.propagate(c->timestamps[i], c->gyros[i], c->accels[i]);
      if (i % 5 == 0) {
        // Test applyEKFUpdate()
        Eigen::MatrixXd new_cov = ekf_propagation.getCov();
        ImuIntegration::State new_state = ekf_propagation.retrieveLatestState();
        Eigen::MatrixXd new_gyro_bias = ekf_propagation.getGyroBias();
        Eigen::MatrixXd new_acc_bias = ekf_propagation.getAccBias();
        LOGI("Apply EKF update at %d", i);
        ekf_propagation.applyEKFUpdate(
            new_cov, new_state, new_gyro_bias, new_acc_bias);
      }
    }
    Eigen::MatrixXd ekf_cov = ekf_propagation.getCov();

    ImuIntegration::State integration_state =
        c->integration.retrieveLatestState();
    ImuIntegration::State ekf_state = ekf_propagation.retrieveLatestState();
    printState(integration_state, "integration_state");
    printState(ekf_state, "ekf_state");
    ASSERT_TRUE(integration_state.isApprox(ekf_state, 1e-6));

    LOGI("prior_cov =\n%s", toStr(prior_cov, Precision(6)).c_str());
    LOGI("integration_cov =\n%s", toStr(integration_cov, Precision(6)).c_str());
    LOGI("ekf_cov =\n%s", toStr(ekf_cov, Precision(6)).c_str());

    double cov_diff = (integration_cov - ekf_cov).norm();
    LOGI("cov_diff = %f", cov_diff);
    ASSERT_LE(cov_diff, 1e-6);
  }
};

TEST_F(TestImuIntegration, testError) {
  test();
}

TEST_F(TestImuIntegration, testJacobian) {
  testJacobian(2, 1e-3);   // One step jacobian
  testJacobian(50, 1e-3);  // Multistep jacobian
}

TEST_F(TestImuIntegration, testCov) {
  testCov();
}

SK4SLAM_UNITTEST_ENTRYPOINT
