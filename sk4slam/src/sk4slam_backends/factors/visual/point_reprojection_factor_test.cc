
#include "sk4slam_backends/factors/visual/point_reprojection_factor.h"

#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/ISAM2.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h"
#include "sk4slam_backends/ceres_backend.h"
#include "sk4slam_backends/gtsam_backend.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_camera/camera_model_factory.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;           // NOLINT
using gtsam::symbol_shorthand::B;  // for T_M_B
using gtsam::symbol_shorthand::E;  // for T_B_C (camera extrinsics)
using gtsam::symbol_shorthand::I;  // for camera intrinsics
using gtsam::symbol_shorthand::M;  // for T_G_M
using gtsam::symbol_shorthand::P;  // for 3D points
using sk4slam::CameraModelType;

static const double fx = 425.780;
static const double fy = 424.799;
static const double cx = 436.380;
static const double cy = 239.206;
static const double k1 = -0.056;
static const double k2 = 0.068;
static const double p1 = 0.005;
static const double p2 = 0.005;
static const double k3 = -0.022;

class TestPointReprojectionFactor : public testing::Test {
  using JacobianMatrixXd = PointReprojectionFactor::JacobianMatrixXd;
  template <int n>
  static VectorXd randomVector() {
    static auto dist = MultivariateUniformDistribution<double>::standard(n);
    return dist();
  }
  static Pose3d randomPose() {
    return Pose3d::Exp(randomVector<6>());
  }
  struct Context {
    VariableKey M1, B1, E1, M2, B2, E2, P1, I1;
    Pose3d M1_val, B1_val, E1_val, M2_val, B2_val, E2_val;
    Vector3d P1_val;
    Eigen::VectorXd I1_val;
    JacobianMatrixXd J_M1, J_B1, J_E1, J_M2, J_B2, J_E2, J_P1, J_I1;
    JacobianMatrixXd *j_M1, *j_B1, *j_E1, *j_M2, *j_B2, *j_E2, *j_P1, *j_I1;

    Vector2d measurement;
    PointReprojectionFactor::PointRepresentation point_representation;
    std::unique_ptr<CameraModelInterface> cam;
  };
  static std::unique_ptr<Context> createContext(
      PointReprojectionFactor::PointRepresentation point_representation =
          PointReprojectionFactor::MAP_XYZ,
      bool common_map = true, bool common_keyframe = false,
      bool common_extrinsic = true) {
    Eigen::VectorXd I1_val(9);
    I1_val << fx, fy, cx, cy, k1, k2, p1, p2, k3;
    auto cam = createCameraModel<CameraModelType::RADTAN5>(I1_val);

    VariableKey M1(M(1)), B1(B(1)), E1(E(1));
    Pose3d M1_val, B1_val = randomPose(), E1_val = randomPose();
    VariableKey M2;
    Pose3d M2_val;

    if (common_map) {
      M2 = M1;
      M1_val = Pose3d::Identity();
      M2_val = M1_val;
    } else {
      M2 = VariableKey(M(2));
      M1_val = randomPose();
      M2_val = M1_val * Pose3d::Exp(0.1 * randomVector<6>());
    }

    VariableKey B2, E2;
    Pose3d B2_val, E2_val;
    if (common_keyframe) {
      ASSERT(common_map);  // common_keyframe implies common_map
      B2 = B1;
      B2_val = B1_val;
    } else {
      B2 = VariableKey(B(2));
      B2_val = M2_val.inverse() * (M1_val * B1_val) *
               Pose3d::Exp(0.1 * randomVector<6>());
    }

    if (common_extrinsic) {
      E2 = E1;
      E2_val = E1_val;
    } else {
      E2 = VariableKey(E(2));
      E2_val = E1_val * Pose3d::Exp(0.1 * randomVector<6>());
    }

    if (point_representation == PointReprojectionFactor::MAP_XYZ) {
      // If the point is not anchored to keyframe, then the variable key
      // for BODYr_POSE_IN_MAPr and CAMr_POSE_IN_BODYr should be null.
      B2 = null_variable;
      E2 = null_variable;
    }

    VariableKey I1(I(1)), P1(P(1));
    Vector3d p_in_ref_cam(0.5, 0.5, 5);
    LOGI("p_in_ref_cam: %s", toStr(p_in_ref_cam.transpose()).c_str());
    Eigen::Vector3d P1_val;
    if (point_representation == PointReprojectionFactor::MAP_XYZ) {
      P1_val = B2_val * (E2_val * p_in_ref_cam);
    } else if (point_representation == PointReprojectionFactor::CAMERA_xyZ) {
      const Vector3d& p = p_in_ref_cam;
      P1_val = Eigen::Vector3d(p.x() / p.z(), p.y() / p.z(), p.z());
    } else {
      ASSERT(
          point_representation ==
          PointReprojectionFactor::CAMERA_INVERSE_DEPTH);
      const Vector3d& p = p_in_ref_cam;
      P1_val = Eigen::Vector3d(p.x() / p.z(), p.y() / p.z(), 1 / p.z());
    }
    LOGI("P1_val: %s", toStr(P1_val.transpose()).c_str());

    Vector3d p_in_world = M2_val * (B2_val * (E2_val * p_in_ref_cam));
    LOGI("p_in_world: %s", toStr(p_in_world.transpose()).c_str());

    Vector3d p_in_obs_body = B1_val.inverse() * (M1_val.inverse() * p_in_world);
    Vector3d p_in_obs_cam = E1_val.inverse() * p_in_obs_body;
    LOGI("p_in_obs_body: %s", toStr(p_in_obs_body.transpose()).c_str());
    LOGI("p_in_obs_cam: %s", toStr(p_in_obs_cam.transpose()).c_str());

    Vector2d measurement = cam->project3(p_in_obs_cam).second;
    LOGI("Measurement: %s", toStr(measurement.transpose()).c_str());

    if (common_map) {
      M2_val = randomPose();  // The value should be ignored by the factor
    }
    if (common_keyframe) {
      B2_val = randomPose();  // The value should be ignored by the factor
    }
    if (common_extrinsic) {
      E2_val = randomPose();  // The value should be ignored by the factor
    }

    auto c = std::make_unique<Context>();
    c->M1 = M1;
    c->B1 = B1;
    c->E1 = E1;
    c->M2 = M2;
    c->B2 = B2;
    c->E2 = E2;
    c->P1 = P1;
    c->I1 = I1;
    c->M1_val = M1_val;
    c->B1_val = B1_val;
    c->E1_val = E1_val;
    c->M2_val = M2_val;
    c->B2_val = B2_val;
    c->E2_val = E2_val;
    c->P1_val = P1_val;
    c->I1_val = I1_val;
    c->measurement = measurement;
    c->cam = std::move(cam);
    c->point_representation = point_representation;

    c->j_M1 = c->j_B1 = c->j_E1 = c->j_M2 = c->j_B2 = c->j_E2 = c->j_P1 =
        c->j_I1 = nullptr;
    if (c->M1 != null_variable && c->M2 != null_variable && c->M2 != c->M1) {
      // If we're handling more than one maps
      c->J_M1.resize(2, 6);
      c->j_M1 = &c->J_M1;
    }
    if (c->B1 != null_variable) {
      c->J_B1.resize(2, 6);
      c->j_B1 = &c->J_B1;
    }
    if (c->E1 != null_variable) {
      c->J_E1.resize(2, 6);
      c->j_E1 = &c->J_E1;
    }
    if (c->M2 != null_variable && c->M2 != c->M1) {
      c->J_M2.resize(2, 6);
      c->j_M2 = &c->J_M2;
    }
    if (c->B2 != null_variable && c->B2 != c->B1) {
      c->J_B2.resize(2, 6);
      c->j_B2 = &c->J_B2;
    }
    if (c->E2 != null_variable && c->E2 != c->E1) {
      c->J_E2.resize(2, 6);
      c->j_E2 = &c->J_E2;
    }
    if (c->P1 != null_variable) {
      c->J_P1.resize(2, 3);
      c->j_P1 = &c->J_P1;
    }
    if (c->I1 != null_variable) {
      c->J_I1.resize(2, 1);
      c->j_I1 = &c->J_I1;
    }
    return c;
  }

 public:
  void testError(
      PointReprojectionFactor::PointRepresentation point_representation =
          PointReprojectionFactor::MAP_XYZ,
      bool common_map = true, bool common_keyframe = false,
      bool common_extrinsic = true) {
    auto c = createContext(
        point_representation, common_map, common_keyframe, common_extrinsic);
    PointReprojectionFactor factor(
        c->measurement, c->point_representation, c->cam.get(),
        {c->M1, c->B1, c->E1, c->M2, c->B2, c->E2, c->P1, c->I1});

    Vector2d error = factor.evaluateError(
        c->M1_val, c->B1_val, c->E1_val, c->M2_val, c->B2_val, c->E2_val,
        c->P1_val, c->I1_val);
    LOGI("Error: %s", toStr(error.transpose()).c_str());
    ASSERT_TRUE(error.norm() < 1e-6);
  }

  void testJacobian(
      PointReprojectionFactor::PointRepresentation point_representation =
          PointReprojectionFactor::MAP_XYZ,
      bool common_map = true, bool common_keyframe = false,
      bool common_extrinsic = true) {
    using OptimizablePose =
        OptimizableManifold<Pose3d, Pose3d::AffineLeftPerturbation>;
    auto c = createContext(
        point_representation, common_map, common_keyframe, common_extrinsic);
    PointReprojectionFactor factor(
        c->measurement, c->point_representation, c->cam.get(),
        {c->M1, c->B1, c->E1, c->M2, c->B2, c->E2, c->P1, c->I1});
    Vector2d error_a = factor.evaluateError(
        c->M1_val, c->B1_val, c->E1_val, c->M2_val, c->B2_val, c->E2_val,
        c->P1_val, c->I1_val, c->j_M1, c->j_B1, c->j_E1, c->j_M2, c->j_B2,
        c->j_E2, c->j_P1, c->j_I1);
    LOGI("error_a: %s", toStr(error_a.transpose()).c_str());
    ASSERT_TRUE(error_a.norm() < 1e-6);

    double delta_scale = 0.001;
    VectorXd d_M1 = delta_scale * randomVector<6>();
    VectorXd d_B1 = delta_scale * randomVector<6>();
    VectorXd d_E1 = delta_scale * randomVector<6>();
    VectorXd d_M2 = delta_scale * randomVector<6>();
    VectorXd d_B2 = delta_scale * randomVector<6>();
    VectorXd d_E2 = delta_scale * randomVector<6>();
    VectorXd d_P1 = delta_scale * randomVector<3>();
    VectorXd d_I1 = delta_scale * randomVector<9>();

    if (!c->j_M1) {
      d_M1.setZero();
    }
    if (!c->j_B1) {
      d_B1.setZero();
    }
    if (!c->j_E1) {
      d_E1.setZero();
    }
    if (!c->j_M2) {
      d_M2.setZero();
    }
    if (!c->j_B2) {
      d_B2.setZero();
    }
    if (!c->j_E2) {
      d_E2.setZero();
    }
    if (!c->j_P1) {
      d_P1.setZero();
    }
    if (!c->j_I1) {
      d_I1.setZero();
    }

    Vector2d error_b = factor.evaluateError(
        OptimizablePose(c->M1_val) + d_M1, OptimizablePose(c->B1_val) + d_B1,
        OptimizablePose(c->E1_val) + d_E1, OptimizablePose(c->M2_val) + d_M2,
        OptimizablePose(c->B2_val) + d_B2, OptimizablePose(c->E2_val) + d_E2,
        c->P1_val + d_P1, c->I1_val + d_I1);

    LOGI("d_M1: %s", toStr(d_M1.transpose()).c_str());
    LOGI("d_B1: %s", toStr(d_B1.transpose()).c_str());
    LOGI("d_E1: %s", toStr(d_E1.transpose()).c_str());
    LOGI("d_M2: %s", toStr(d_M2.transpose()).c_str());
    LOGI("d_B2: %s", toStr(d_B2.transpose()).c_str());
    LOGI("d_E2: %s", toStr(d_E2.transpose()).c_str());
    LOGI("d_P1: %s", toStr(d_P1.transpose()).c_str());
    LOGI("d_I1: %s", toStr(d_I1.transpose()).c_str());

    LOGI("error_b: %s", toStr(error_b.transpose()).c_str());

    Vector2d d_error = (error_b - error_a);
    LOGI("d_error: %s", toStr(d_error.transpose()).c_str());

    Vector2d d_error_approx(0, 0);
    if (c->j_M1) {
      d_error_approx += (*(c->j_M1)) * d_M1;
    }
    if (c->j_B1) {
      d_error_approx += (*(c->j_B1)) * d_B1;
    }
    if (c->j_E1) {
      d_error_approx += (*(c->j_E1)) * d_E1;
    }
    if (c->j_M2) {
      d_error_approx += (*(c->j_M2)) * d_M2;
    }
    if (c->j_B2) {
      d_error_approx += (*(c->j_B2)) * d_B2;
    }
    if (c->j_E2) {
      d_error_approx += (*(c->j_E2)) * d_E2;
    }
    if (c->j_P1) {
      d_error_approx += (*(c->j_P1)) * d_P1;
    }
    if (c->j_I1) {
      d_error_approx += (*(c->j_I1)) * d_I1;
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

TEST_F(TestPointReprojectionFactor, testError_MAP_XYZ) {
  // Common map, Different keyframe, Common extrinsic
  testError(PointReprojectionFactor::MAP_XYZ, true, false, true);

  // Common keyframe, Common extrinsic
  testError(PointReprojectionFactor::MAP_XYZ, true, true, true);

  // Different map, Different keyframe, Common extrinsic
  testError(PointReprojectionFactor::MAP_XYZ, false, false, true);

  // Different map, Different keyframe, Different extrinsic
  testError(PointReprojectionFactor::MAP_XYZ, false, false, false);
}

TEST_F(TestPointReprojectionFactor, testError_CAMERA_xyZ) {
  // Common map, Different keyframe, Common extrinsic
  testError(PointReprojectionFactor::CAMERA_xyZ, true, false, true);

  // Common keyframe, Common extrinsic
  testError(PointReprojectionFactor::CAMERA_xyZ, true, true, true);

  // Different map, Different keyframe, Common extrinsic
  testError(PointReprojectionFactor::CAMERA_xyZ, false, false, true);

  // Different map, Different keyframe, Different extrinsic
  testError(PointReprojectionFactor::CAMERA_xyZ, false, false, false);
}

TEST_F(TestPointReprojectionFactor, testError_CAMERA_INVERSE_DEPTH) {
  // Common map, Different keyframe, Common extrinsic
  testError(PointReprojectionFactor::CAMERA_INVERSE_DEPTH, true, false, true);

  // Common keyframe, Common extrinsic
  testError(PointReprojectionFactor::CAMERA_INVERSE_DEPTH, true, true, true);

  // Different map, Different keyframe, Common extrinsic
  testError(PointReprojectionFactor::CAMERA_INVERSE_DEPTH, false, false, true);

  // Different map, Different keyframe, Different extrinsic
  testError(PointReprojectionFactor::CAMERA_INVERSE_DEPTH, false, false, false);
}

TEST_F(TestPointReprojectionFactor, testJacobian_MAP_XYZ) {
  // Common map, Different keyframe, Common extrinsic
  testJacobian(PointReprojectionFactor::MAP_XYZ, true, false, true);

  // Common keyframe, Common extrinsic
  testJacobian(PointReprojectionFactor::MAP_XYZ, true, true, true);

  // Different map, Different keyframe, Common extrinsic
  testJacobian(PointReprojectionFactor::MAP_XYZ, false, false, true);

  // Different map, Different keyframe, Different extrinsic
  testJacobian(PointReprojectionFactor::MAP_XYZ, false, false, false);
}

TEST(TestCreatePointReprojectionFactor, GtsamBackend) {
  GtsamBackend backend;
  auto cam = createCameraModel<CameraModelType::PINHOLE>(Vector4d(1, 1, 0, 0));

  auto M1 = backend.addVariable(M(1), Pose3d::Identity());
  auto B1 = backend.addVariable(B(1), Pose3d::Identity());
  auto E1 = backend.addVariable(E(1), Pose3d::Identity());
  auto M2 = backend.addVariable(M(2), Pose3d::Identity());
  auto B2 = backend.addVariable(B(2), Pose3d::Identity());
  auto E2 = backend.addVariable(E(2), Pose3d::Identity());
  auto I1 = backend.addVariable(I(1), cam->intrinsicsVector());
  auto P1 = backend.addVariable(P(1), Vector3d(0, 0, 1));

  Vector2d measurement(0.01, 0.02);
  auto factor_id = backend.addFactor(PointReprojectionFactor(
      measurement, PointReprojectionFactor::CAMERA_INVERSE_DEPTH, cam.get(),
      {M1, B1, E1, M2, B2, E2, P1, I1}));
  auto factor_ptr = backend.getFactor<PointReprojectionFactor>(factor_id);
  ASSERT_TRUE(factor_ptr);
  LOGI(
      "factor_ptr->measurement() = %s",
      toStr(factor_ptr->measurement().transpose()).c_str());
  ASSERT_NEAR(factor_ptr->measurement()(0), measurement(0), 1e-6);
  ASSERT_NEAR(factor_ptr->measurement()(1), measurement(1), 1e-6);
  std::cout << "TestCreatePointReprojectionFactor GtsamBackend" << std::endl;
}

TEST(TestCreatePointReprojectionFactor, CeresBackend) {
  CeresBackend backend;
  auto cam = createCameraModel<CameraModelType::PINHOLE>(Vector4d(1, 1, 0, 0));
  Pose3d M1_value = Pose3d::Identity();
  Pose3d B1_value = Pose3d::Identity();
  Pose3d E1_value = Pose3d::Identity();
  Pose3d M2_value = Pose3d::Identity();
  Pose3d B2_value = Pose3d::Identity();
  Pose3d E2_value = Pose3d::Identity();
  Vector3d P1_value = Vector3d(0, 0, 1);
  VectorXd I1_value = cam->intrinsicsVector();

  auto M1 = backend.addVariable(&M1_value);
  auto B1 = backend.addVariable(&B1_value);
  auto E1 = backend.addVariable(&E1_value);
  auto M2 = backend.addVariable(&M2_value);
  auto B2 = backend.addVariable(&B2_value);
  auto E2 = backend.addVariable(&E2_value);
  auto P1 = backend.addVariable(&P1_value);
  auto I1 = backend.addVariable(&I1_value);

  Vector2d measurement(0.01, 0.02);
  auto factor_id = backend.addFactor(PointReprojectionFactor(
      measurement, PointReprojectionFactor::CAMERA_INVERSE_DEPTH, cam.get(),
      {M1, B1, E1, M2, B2, E2, P1, I1}));
  auto factor_ptr = backend.getFactor<PointReprojectionFactor>(factor_id);
  ASSERT_TRUE(factor_ptr);
  LOGI(
      "factor_ptr->measurement() = %s",
      toStr(factor_ptr->measurement().transpose()).c_str());
  ASSERT_NEAR(factor_ptr->measurement()(0), measurement(0), 1e-6);
  ASSERT_NEAR(factor_ptr->measurement()(1), measurement(1), 1e-6);

  std::cout << "TestCreatePointReprojectionFactor CeresBackend" << std::endl;
}

SK4SLAM_UNITTEST_ENTRYPOINT
