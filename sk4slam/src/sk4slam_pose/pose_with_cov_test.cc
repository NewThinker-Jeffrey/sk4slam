#include "sk4slam_pose/pose_with_cov.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/matrix.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

using Vector3 = Eigen::Matrix<double, 3, 1>;
using Matrix3 = Eigen::Matrix<double, 3, 3>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;

// Define the templated test fixture.
template <class _Perturbation>
class TestPoseWithCov : public testing::Test {
 public:
  using Perturbation = _Perturbation;
  using Pose3dWithCov = Pose3WithCov<Perturbation, double>;
  using Pose = typename Pose3dWithCov::Pose;

  void testMultiplyPose() {
    Pose3dWithCov pose_with_cov1 = makeRandomPoseWithCov();
    LOGI("p: %s", toStr(pose_with_cov1.translation().transpose()).c_str());
    LOGI("R: \n%s", toStr(pose_with_cov1.rotation().matrix()).c_str());
    LOGI("pose1: \n%s", toStr(pose_with_cov1.pose().matrix()).c_str());
    LOGI("pose1_cov: \n%s", toStr(pose_with_cov1.cov(), Precision(6)).c_str());

    Pose pose2 = makeRandomPose();
    LOGI("pose2: \n%s", toStr(pose2.matrix()).c_str());

    Pose3dWithCov pose12_with_cov = pose_with_cov1 * pose2;
    LOGI("pose12: \n%s", toStr(pose12_with_cov.pose().matrix()).c_str());
    LOGI(
        "pose12_cov: \n%s", toStr(pose12_with_cov.cov(), Precision(6)).c_str());
    Pose3dWithCov pose12_inv_with_cov = pose12_with_cov.inverse();
    LOGI(
        "pose12_inv: \n%s", toStr(pose12_inv_with_cov.pose().matrix()).c_str());
    LOGI(
        "pose12_inv_cov: \n%s",
        toStr(pose12_inv_with_cov.cov(), Precision(6)).c_str());

    Pose3dWithCov pose12_inv2_with_cov =
        pose2.inverse() * pose_with_cov1.inverse();
    LOGI(
        "pose12_inv2: \n%s",
        toStr(pose12_inv2_with_cov.pose().matrix()).c_str());
    LOGI(
        "pose12_inv2_cov: \n%s",
        toStr(pose12_inv2_with_cov.cov(), Precision(6)).c_str());

    EXPECT_TRUE(
        pose12_inv_with_cov.pose().isApprox(pose12_inv2_with_cov.pose(), 1e-6));
    EXPECT_TRUE(
        pose12_inv_with_cov.cov().isApprox(pose12_inv2_with_cov.cov(), 1e-6));
  }

  void testMultiplyPoseWithCov() {
    Pose3dWithCov pose_with_cov1 = makeRandomPoseWithCov();
    LOGI("p: %s", toStr(pose_with_cov1.translation().transpose()).c_str());
    LOGI("R: \n%s", toStr(pose_with_cov1.rotation().matrix()).c_str());
    LOGI("pose1: \n%s", toStr(pose_with_cov1.pose().matrix()).c_str());
    LOGI("pose1_cov: \n%s", toStr(pose_with_cov1.cov(), Precision(6)).c_str());

    Pose3dWithCov pose_with_cov2 = makeRandomPoseWithCov();
    Pose3dWithCov pose12_with_cov = pose_with_cov1 * pose_with_cov2;
    LOGI("pose12: \n%s", toStr(pose12_with_cov.pose().matrix()).c_str());
    LOGI(
        "pose12_cov: \n%s", toStr(pose12_with_cov.cov(), Precision(6)).c_str());
    Pose3dWithCov pose12_inv_with_cov = pose12_with_cov.inverse();
    LOGI(
        "pose12_inv: \n%s", toStr(pose12_inv_with_cov.pose().matrix()).c_str());
    LOGI(
        "pose12_inv_cov: \n%s",
        toStr(pose12_inv_with_cov.cov(), Precision(6)).c_str());
    Pose3dWithCov pose12_inv2_with_cov =
        pose_with_cov2.inverse() * pose_with_cov1.inverse();
    LOGI(
        "pose12_inv2: \n%s",
        toStr(pose12_inv2_with_cov.pose().matrix()).c_str());
    LOGI(
        "pose12_inv2_cov: \n%s",
        toStr(pose12_inv2_with_cov.cov(), Precision(6)).c_str());

    EXPECT_TRUE(
        pose12_inv_with_cov.pose().isApprox(pose12_inv2_with_cov.pose(), 1e-6));
    EXPECT_TRUE(
        pose12_inv_with_cov.cov().isApprox(pose12_inv2_with_cov.cov(), 1e-6));
  }

  void testPerturbationConversion() {
    Pose3dWithCov pose_with_cov1 = makeRandomPoseWithCov();
    LOGI(
        "original_cov: \n%s",
        toStr(pose_with_cov1.cov(), Precision(12)).c_str());

#undef TEST_WITH_NEW_PERTURBATION
#define TEST_WITH_NEW_PERTURBATION(NewPerturbation)                   \
  if constexpr (!std::is_same_v<NewPerturbation, Perturbation>) {     \
    using Pose3WithOtherCov = Pose3WithCov<NewPerturbation>;          \
    Pose3WithOtherCov pose_with_other_cov;                            \
    pose_with_other_cov.pose() = pose_with_cov1.pose();               \
    pose_with_other_cov.setCov<Perturbation>(pose_with_cov1.cov());   \
    LOGI("NewPerturbation: %s", #NewPerturbation);                    \
    LOGI(                                                             \
        "converted_back_cov: \n%s",                                   \
        toStr(pose_with_other_cov.cov<Perturbation>(), Precision(12)) \
            .c_str());                                                \
    ASSERT_TRUE(pose_with_other_cov.cov<Perturbation>().isApprox(     \
        pose_with_cov1.cov(), 1e-6));                                 \
  }

    TEST_WITH_NEW_PERTURBATION(Pose3d::AffineRightPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::AffineLeftPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::RightPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::LeftPerturbation)
  }

  void testRightPoseDeltaCov() {
    Pose pose0 = makeRandomPose();
    Pose pose1 = makeRandomPose();
    Matrix12 joint_cov = makeRandomCov12();
    Pose3dWithCov pose_delta_with_cov =
        Pose3dWithCov::RightPoseDelta(pose0, pose1, joint_cov);
    LOGI(
        "original_delta_cov: \n%s",
        toStr(pose_delta_with_cov.cov(), Precision(12)).c_str());

#undef TEST_WITH_NEW_PERTURBATION
#define TEST_WITH_NEW_PERTURBATION(NewPerturbation)                         \
  if constexpr (!std::is_same_v<NewPerturbation, Perturbation>) {           \
    using Pose3WithOtherCov = Pose3WithCov<NewPerturbation>;                \
    Matrix12 joint_other_cov =                                              \
        Pose3WithOtherCov::ConvertJointPoseCov<Perturbation>(               \
            pose0, pose1, joint_cov);                                       \
    Pose3WithOtherCov pose_delta_with_other_cov =                           \
        Pose3WithOtherCov::RightPoseDelta(pose0, pose1, joint_other_cov);   \
    LOGI("NewPerturbation: %s", #NewPerturbation);                          \
    LOGI(                                                                   \
        "converted_back_delta_cov: \n%s",                                   \
        toStr(pose_delta_with_other_cov.cov<Perturbation>(), Precision(12)) \
            .c_str());                                                      \
    ASSERT_TRUE(pose_delta_with_other_cov.cov<Perturbation>().isApprox(     \
        pose_delta_with_cov.cov(), 1e-6));                                  \
  }

    TEST_WITH_NEW_PERTURBATION(Pose3d::RightPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::LeftPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::AffineRightPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::AffineLeftPerturbation)
  }

  void testLeftPoseDeltaCov() {
    Pose pose0 = makeRandomPose();
    Pose pose1 = makeRandomPose();
    Matrix12 joint_cov = makeRandomCov12();
    Pose3dWithCov pose_delta_with_cov =
        Pose3dWithCov::LeftPoseDelta(pose0, pose1, joint_cov);
    LOGI(
        "original_delta_cov: \n%s",
        toStr(pose_delta_with_cov.cov(), Precision(12)).c_str());

#undef TEST_WITH_NEW_PERTURBATION
#define TEST_WITH_NEW_PERTURBATION(NewPerturbation)                         \
  if constexpr (!std::is_same_v<NewPerturbation, Perturbation>) {           \
    using Pose3WithOtherCov = Pose3WithCov<NewPerturbation>;                \
    Matrix12 joint_other_cov =                                              \
        Pose3WithOtherCov::ConvertJointPoseCov<Perturbation>(               \
            pose0, pose1, joint_cov);                                       \
    Pose3WithOtherCov pose_delta_with_other_cov =                           \
        Pose3WithOtherCov::LeftPoseDelta(pose0, pose1, joint_other_cov);    \
    LOGI("NewPerturbation: %s", #NewPerturbation);                          \
    LOGI(                                                                   \
        "converted_back_delta_cov: \n%s",                                   \
        toStr(pose_delta_with_other_cov.cov<Perturbation>(), Precision(12)) \
            .c_str());                                                      \
    ASSERT_TRUE(pose_delta_with_other_cov.cov<Perturbation>().isApprox(     \
        pose_delta_with_cov.cov(), 1e-6));                                  \
  }

    TEST_WITH_NEW_PERTURBATION(Pose3d::RightPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::LeftPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::AffineRightPerturbation)
    TEST_WITH_NEW_PERTURBATION(Pose3d::AffineLeftPerturbation)
  }

 protected:
  Pose3dWithCov makeRandomPoseWithCov() {
    return Pose3dWithCov(makeRandomPose(), makeRandomCov());
  }

  Matrix6 makeRandomCov() {
    static MultivariateUniformDistribution<double> dist6 =
        MultivariateUniformDistribution<double>::standard(6);
    Matrix6 sqrt_cov;
    for (int i = 0; i < 6; ++i) {
      sqrt_cov.col(i) = dist6();
    }
    sqrt_cov *= 0.05;
    Matrix6 cov = sqrt_cov * sqrt_cov.transpose();
    return cov;
  }

  Matrix12 makeRandomCov12() {
    static MultivariateUniformDistribution<double> dist12 =
        MultivariateUniformDistribution<double>::standard(12);
    Matrix12 sqrt_cov;
    for (int i = 0; i < 12; ++i) {
      sqrt_cov.col(i) = dist12();
    }
    sqrt_cov *= 0.05;
    Matrix12 cov = sqrt_cov * sqrt_cov.transpose();
    return cov;
  }

  Pose makeRandomPose() {
    static MultivariateUniformDistribution<double> dist3 =
        MultivariateUniformDistribution<double>::standard(3);
    return Pose(expMat(skew3(dist3())), dist3());
  }
};

// Register the types to test.
using Perturbations = ::testing::Types<
    Pose3d::AffineLeftPerturbation, Pose3d::AffineRightPerturbation,
    Pose3d::LeftPerturbation, Pose3d::RightPerturbation>;

// NOTE: use TYPED_TEST_SUITE instead for newer versions of gtest
TYPED_TEST_SUITE(TestPoseWithCov, Perturbations);

TYPED_TEST(TestPoseWithCov, MultiplyPose) {
  this->testMultiplyPose();
}

TYPED_TEST(TestPoseWithCov, MultiplyPoseWithCov) {
  this->testMultiplyPoseWithCov();
}

TYPED_TEST(TestPoseWithCov, PerturbationConversion) {
  this->testPerturbationConversion();
}

TYPED_TEST(TestPoseWithCov, RightPoseDeltaCov) {
  this->testRightPoseDeltaCov();
}

TYPED_TEST(TestPoseWithCov, LeftPoseDeltaCov) {
  this->testLeftPoseDeltaCov();
}

SK4SLAM_UNITTEST_ENTRYPOINT
