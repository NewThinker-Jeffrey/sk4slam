#include "sk4slam_math/matrix.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/polynomial.h"
#include "sk4slam_math/random.h"

using namespace sk4slam;  // NOLINT

TEST(TestMath, Skew3) {
  Eigen::Vector3d xi(0.1, 0.2, 0.3);
  Eigen::Vector3d eta(0.4, 0.5, 0.6);
  Eigen::Matrix3d hat_xi = skew3(xi);
  Eigen::Matrix3d hat_eta = skew3(eta);
  double theta2 = xi.squaredNorm();

  {
    // Verify the equation:
    //    hat_xi^2 = xi * xi^T - I * theta^2
    LOGI(
        "Verifying equation 1: hat_xi^2 = xi * xi^T - I * theta^2 (where theta "
        "is the norm of xi)");
    Eigen::Matrix3d hat_xi2 = hat_xi * hat_xi;
    Eigen::Matrix3d res =
        xi * xi.transpose() - theta2 * Eigen::Matrix3d::Identity();
    LOGI("hat_xi^2:\n%s", toStr(hat_xi2).c_str());
    LOGI("xi * xi^T - I * theta^2:\n%s", toStr(res).c_str());
    ASSERT_TRUE(hat_xi2.isApprox(res, 1e-6));
  }

  {
    // Verify the equation:
    //    hat_xi^2 * hat_eta * hat_xi^2 =
    // -theta^2 * (hat_eta * hat_xi^2 + hat_xi^2 * hat_eta + theta^2 * hat_eta)
    LOGI("Verifying equation 2:");
    Eigen::Matrix3d hat_xi2 = hat_xi * hat_xi;
    Eigen::Matrix3d r1 = hat_xi2 * hat_eta * hat_xi2;
    Eigen::Matrix3d r2 =
        -theta2 * (hat_eta * hat_xi2 + hat_xi2 * hat_eta + theta2 * hat_eta);
    LOGI("r1:\n%s", toStr(r1).c_str());
    LOGI("r2:\n%s", toStr(r2).c_str());
    ASSERT_TRUE(r1.isApprox(r2, 1e-6));
  }
}

// Test case: Out-of-bound start_col
TEST(GivensRotationTest, OutOfBoundStartCol) {
  Eigen::MatrixXd A(3, 3);
  Eigen::MatrixXd expected(3, 3);
  // clang-format off
  A << 4, 1, 2,
       3, 4, 5,
       2, 5, 6;

  expected = A;  // No change expected
  // clang-format on

  // Start column is out of bounds
  GivensRotation(A, 2, 3, 1);

  ASSERT_TRUE(A.isApprox(expected, 1e-6))
      << "Matrix should remain unchanged for out-of-bound start_col:\n"
      << "Expected:\n"
      << expected << "\nActual:\n"
      << A;
}

// Test case: Invalid row and column position
TEST(GivensRotationTest, InvalidPosition) {
  Eigen::MatrixXd A(3, 3);
  Eigen::MatrixXd expected(3, 3);
  // clang-format off
  A << 4, 1, 2,
       3, 4, 5,
       2, 5, 6;

  expected = A;  // No change expected
  // clang-format on

  // Row is above the diagonal
  GivensRotation(A, 0, 1, 1);

  ASSERT_TRUE(A.isApprox(expected, 1e-6))
      << "Matrix should remain unchanged for invalid position:\n"
      << "Expected:\n"
      << expected << "\nActual:\n"
      << A;
}

// Test case: Solve least squares problem with Givens QR and Eigen QR
TEST(GivensRotationTest, LeastSquaresSolutionAugmented) {
  // Generate a random overdetermined system (8 rows, 4 columns)
  int rows = 8, cols = 4;

  // Random number generator
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  // Generate random matrix A and vector b
  Eigen::MatrixXd A(rows, cols);
  Eigen::VectorXd b(rows);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      A(i, j) = dist(gen);
    }
    b(i) = dist(gen);
  }

  // Solve using Eigen's QR decomposition
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
  Eigen::VectorXd x_eigen = qr.solve(b);

  // Solve using Givens QR decomposition on augmented matrix [A|b]
  Eigen::MatrixXd augmented(rows, cols + 1);
  augmented << A, b;  // Concatenate A and b
  std::cout << "augmented:\n" << augmented << std::endl;

  GivensQRForAugmentedMatrix(augmented);

  // Extract R and transformed b
  Eigen::MatrixXd R = augmented.block<4, 4>(0, 0);
  Eigen::VectorXd b_transformed = augmented.block<4, 1>(0, 4);
  std::cout << "QR decomposed:\n" << augmented << std::endl;
  std::cout << "R:\n" << R << std::endl;
  std::cout << "b_transformed: " << b_transformed.transpose() << std::endl;

  // Solve R * x = b_transformed for x (least squares solution)
  Eigen::VectorXd x_givens =
      R.triangularView<Eigen::Upper>().solve(b_transformed);
  std::cout << "x_givens: " << x_givens.transpose() << std::endl;
  std::cout << "x_eigen : " << x_eigen.transpose() << std::endl;

  // Compare the two solutions
  ASSERT_TRUE(x_givens.isApprox(x_eigen, 1e-6))
      << "Least squares solution from Givens QR does not match Eigen QR:\n"
      << "Eigen solution:\n"
      << x_eigen << "\nGivens solution:\n"
      << x_givens;
}

SK4SLAM_UNITTEST_ENTRYPOINT
