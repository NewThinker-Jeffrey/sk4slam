#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_geometry/essential_matrix.h"
#include "sk4slam_geometry/fundamental_matrix.h"
#include "sk4slam_geometry/homography_matrix.h"
#include "sk4slam_geometry/pnp.h"
#include "sk4slam_geometry/point_transform.h"
#include "sk4slam_geometry/utils.h"
#include "sk4slam_math/sac.h"

using namespace sk4slam;  // NOLINT

// CV to Eigen
TEST(TestCvToEigen, CvToEigen) {
  cv::Point2f cv_point2f(1.0, 2.0);
  cv::Point3f cv_point3f(1.0, 2.0, 3.0);
  using PointPair = std::pair<Eigen::Vector2d, Eigen::Vector3d>;
  std::vector<cv::Point2f> cv_Xs = {cv_point2f};
  std::vector<cv::Point3f> cv_Xprimes = {cv_point3f};
  std::vector<PointPair> point_pairs;
  convertCvPointPairsToEigen(cv_Xs, cv_Xprimes, &point_pairs);
  LOGI("point_pairs: %s", toStr(point_pairs, [](const PointPair& pair) {
                            return toStr(pair.first.transpose()) + " -> " +
                                   toStr(pair.second.transpose());
                          }).c_str());
  ASSERT_EQ(point_pairs.size(), 1);
  ASSERT_NEAR(
      (point_pairs[0].first - Eigen::Vector2d(1.0, 2.0)).norm(), 0.0, 1e-6);
  ASSERT_NEAR(
      (point_pairs[0].second - Eigen::Vector3d(1.0, 2.0, 3.0)).norm(), 0.0,
      1e-6);
}

// PnP
TEST(TestEPNPEstimator, Ransac) {
  // TODO(jeffrey)
  LOGI("Run testing for TestEPNPEstimator.Ransac");
  Ransac<EPNPEstimator> ransac;
}

TEST(TestP3PEstimator, Ransac) {
  // TODO(jeffrey)
  LOGI("Run testing for TestP3PEstimator.Ransac");
  Ransac<P3PEstimator> ransac;
}

// Transforms
TEST(TestSimilarityTransformEstimator, Ransac) {
  // TODO(jeffrey)
  LOGI("Run testing for TestSimilarityTransformEstimator.Ransac");
  Ransac<SimilarityTransformEstimator<3>> ransac;
}

TEST(TestEuclideanTransformEstimator, Ransac) {
  // TODO(jeffrey)
  LOGI("Run testing for TestEuclideanTransformEstimator.Ransac");
  Ransac<EuclideanTransformEstimator<3>> ransac;
}

TEST(TestTranslationTransformEstimator, Ransac) {
  // TODO(jeffrey)
  LOGI("Run testing for TestTranslationTransformEstimator.Ransac");
  Ransac<TranslationTransformEstimator<3>> ransac;
}

SK4SLAM_UNITTEST_ENTRYPOINT
