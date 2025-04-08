#include "sk4slam_math/optimizable_manifold.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

TEST(TestOptimizableManifold, Basic) {
  ASSERT_EQ((manifold_traits<ProductManifold<Vector3d, Vector2d>>::kDim), 5);

  ASSERT_TRUE(
      (std::is_same_v<
          Eigen::Vector3f, manifold_traits<Eigen::Vector3d>::Cast<float>>));
  ASSERT_TRUE(
      (std::is_same_v<
          Eigen::VectorXf, manifold_traits<Eigen::VectorXd>::Cast<float>>));
  ASSERT_EQ((manifold_traits<Eigen::Vector3d>::kDim), 3);
  ASSERT_EQ((manifold_traits<Eigen::Vector3f>::kDim), 3);
  ASSERT_EQ((manifold_traits<Eigen::VectorXd>::kDim), -1);
  ASSERT_EQ((manifold_traits<Eigen::VectorXf>::kDim), -1);

  ASSERT_EQ((OptimizableManifold<Eigen::Vector3d>::kDof), 3);
  ASSERT_EQ((OptimizableManifold<Eigen::Vector3d>::kAmbientDim), 3);

  {
    using SubRetraction = VectorSpaceSubRetraction<SubSpaceByAxes<0, 1>, 3>;
    ASSERT_EQ((OptimizableManifold<Eigen::Vector3d, SubRetraction>::kDof), 2);
    ASSERT_EQ(
        (OptimizableManifold<Eigen::Vector3d, SubRetraction>::kAmbientDim), 3);
  }

  {
    using SubRetraction = VectorSpaceSubRetraction<SubSpaceByMatrix<2>, 3>;
    ASSERT_EQ((OptimizableManifold<Eigen::Vector3d, SubRetraction>::kDof), 2);
    ASSERT_EQ(
        (OptimizableManifold<Eigen::Vector3d, SubRetraction>::kAmbientDim), 3);
  }
}

SK4SLAM_UNITTEST_ENTRYPOINT
