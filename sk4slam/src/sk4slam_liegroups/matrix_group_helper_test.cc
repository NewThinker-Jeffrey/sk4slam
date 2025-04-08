#include "sk4slam_liegroups/matrix_group_helper.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/Rn.h"
#include "sk4slam_liegroups/Rp.h"
#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SE3.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/Sim3.h"

using namespace sk4slam;  // NOLINT

TEST(TestMatrixGroupHelper, MatrixGroupHelper) {
  ASSERT_TRUE(MatrixGroupHelper<SO3d>::kIsMatrixGroup);
  ASSERT_EQ(MatrixGroupHelper<SO3d>::N, 3);
  {
    using LieGroup = SO3d;
    using MatrixType = Eigen::Matrix<double, 3, 3>;
    LieGroup::LieAlgebra X(0.1, 0.2, 0.3);
    auto hat_X = LieGroup::hat(X);
    LOGI("hat_X:\n%s", toStr(hat_X).c_str());

    // For SO3, ambientToMatrix() returns a const lvalue reference to the matrix
    using RetType =
        decltype(MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType>));
    ASSERT_FALSE((std::is_same_v<RetType, const MatrixType&>));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType&&>));
    ASSERT_TRUE((std::is_same_v<RetType, MatrixType&>));

    const MatrixType& ret_matrix =
        MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X);
    LOGI("ret_matrix:\n%s", toStr(ret_matrix).c_str());
    ASSERT_EQ(&ret_matrix, &hat_X);
  }

  ASSERT_TRUE(MatrixGroupHelper<Rpd>::kIsMatrixGroup);
  ASSERT_EQ(MatrixGroupHelper<Rpd>::N, 1);
  {
    using LieGroup = Rpd;
    using MatrixType = Eigen::Matrix<double, 1, 1>;
    LieGroup::LieAlgebra X(0.123);
    auto hat_X = LieGroup::hat(X);
    LOGI("hat_X:\n%s", toStr(hat_X).c_str());

    // For R^+, ambientToMatrix() returns a const lvalue reference to the matrix
    using RetType =
        decltype(MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType>));
    ASSERT_FALSE((std::is_same_v<RetType, const MatrixType&>));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType&&>));
    ASSERT_TRUE((std::is_same_v<RetType, MatrixType&>));

    const MatrixType& ret_matrix =
        MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X);
    LOGI("ret_matrix:\n%s", toStr(ret_matrix).c_str());
    ASSERT_EQ(&ret_matrix, &hat_X);
  }

  ASSERT_TRUE(MatrixGroupHelper<SE3d>::kIsMatrixGroup);
  ASSERT_EQ(MatrixGroupHelper<SE3d>::N, 4);
  {
    using LieGroup = SE3d;
    using MatrixType = Eigen::Matrix<double, 4, 4>;
    LieGroup::LieAlgebra X;
    X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    auto hat_X = LieGroup::hat(X);
    MatrixType hat_Xmat = Eigen::Matrix<double, 4, 4>::Zero();
    hat_Xmat.block<3, 3>(0, 0) = hat_X.part<0>();
    hat_Xmat.block<3, 1>(0, 3) = hat_X.part<1>();
    LOGI("hat_X:\n%s", toStr(hat_Xmat).c_str());

    // For SE3, ambientToMatrix() returns a temporary matrix (not a reference)
    using RetType =
        decltype(MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X));
    ASSERT_TRUE((std::is_same_v<RetType, MatrixType>));
    ASSERT_FALSE((std::is_same_v<RetType, const MatrixType&>));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType&&>));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType&>));

    // Although ambientToMatrix() returns a temporary matrix, it is safe to
    // capture the return value using a const lvalue reference because C++
    // allows for lifetime extension of temporary objects when they are bound
    // to a const lvalue reference.
    const MatrixType& ret_matrix =
        MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X);
    LOGI("ret_matrix:\n%s", toStr(ret_matrix).c_str());
    ASSERT_TRUE(ret_matrix.isApprox(hat_Xmat, 1e-10));
  }

  ASSERT_TRUE(MatrixGroupHelper<Rp_x_SO3d>::kIsMatrixGroup);
  ASSERT_EQ(MatrixGroupHelper<Rp_x_SO3d>::N, 3);
  {
    using LieGroup = Rp_x_SO3d;
    using MatrixType = Eigen::Matrix<double, 3, 3>;
    LieGroup::LieAlgebra X;
    X << 0.15, 0.1, 0.2, 0.3;
    auto hat_X = LieGroup::hat(X);
    MatrixType hat_Xmat = Eigen::Matrix<double, 3, 3>::Zero();
    hat_Xmat = hat_X;
    LOGI("hat_X:\n%s", toStr(hat_Xmat).c_str());

    // For Rp_x_SO3, ambientToMatrix() returns a temporary matrix (not a
    // reference)
    using RetType =
        decltype(MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X));
    ASSERT_TRUE((std::is_same_v<RetType, MatrixType>));
    ASSERT_FALSE((std::is_same_v<RetType, const MatrixType&>));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType&&>));
    ASSERT_FALSE((std::is_same_v<RetType, MatrixType&>));

    // Although ambientToMatrix() returns a temporary matrix, it is safe to
    // capture the return value using a const lvalue reference because C++
    // allows for lifetime extension of temporary objects when they are bound
    // to a const lvalue reference.
    const MatrixType& ret_matrix =
        MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X);
    LOGI("ret_matrix:\n%s", toStr(ret_matrix).c_str());
    ASSERT_TRUE(ret_matrix.isApprox(hat_Xmat, 1e-10));
  }

  ASSERT_FALSE(MatrixGroupHelper<R2d>::kIsMatrixGroup);
  ASSERT_EQ(MatrixGroupHelper<R2d>::N, 0);
  {
    using LieGroup = R2d;
    LieGroup::LieAlgebra X(0.1, 0.2);
    auto hat_X = LieGroup::hat(X);
    LOGI("hat_X:\n%s", toStr(hat_X).c_str());

    // ambientToMatrix() is not implemented for non-matrix groups,
    // and R^2 is not a matrix group. The call ambientToMatrix() here
    // shuold cause a compile-time error.
    //
    // auto ret_matrix = MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X);
  }

  ASSERT_FALSE(MatrixGroupHelper<R3d>::kIsMatrixGroup);
  ASSERT_EQ(MatrixGroupHelper<R3d>::N, 0);
  {
    using LieGroup = R3d;
    LieGroup::LieAlgebra X(0.1, 0.2, 0.3);
    auto hat_X = LieGroup::hat(X);
    LOGI("hat_X:\n%s", toStr(hat_X).c_str());
    // ambientToMatrix() is not implemented for non-matrix groups,
    // and R^2 is not a matrix group. The call ambientToMatrix() here
    // shuold cause a compile-time error.
    //
    // auto ret_matrix = MatrixGroupHelper<LieGroup>::ambientToMatrix(hat_X);
  }
}

TEST(TestMatrixGroupHelper, SO3MatrixGroupCommonOps) {
  using MatrixGroup = SO3d;
  static constexpr int N = 3;

  MatrixGroup g(MatrixGroup::Identity());

  Eigen::MatrixXd m(N, N - 1);
  m.setRandom();
  LOGI("m:\n%s", toStr(m).c_str());
  LOGI("g * m:\n%s", toStr(g * m).c_str());  // operator*(const Matrix&)
  ASSERT_TRUE(m.isApprox(g * m, 1e-6));
  LOGI(
      "g * m.block:\n%s",
      toStr(g * m.block<N, N - 1>(0, 0)).c_str());  // operator*(const Block&)
  ASSERT_TRUE(
      (m.block<N, N - 1>(0, 0).isApprox(g * m.block<N, N - 1>(0, 0), 1e-6)));
  LOGI(
      "g * (-m):\n%s",
      toStr(g * (-m)).c_str());  // operator*(const CwiseUnaryOp&)
  ASSERT_TRUE((-m).isApprox(g * (-m), 1e-6));
  LOGI(
      "g * (m+m):\n%s",
      toStr(g * (m + m)).c_str());  // operator*(const CwiseBinaryOp&)
  ASSERT_TRUE((m + m).isApprox(g * (m + m), 1e-6));
  LOGI(
      "g * (2*m):\n%s",
      toStr(g * (2 * m)).c_str());  // operator*(const CwiseBinaryOp&)
  ASSERT_TRUE((2 * m).isApprox(g * (2 * m), 1e-6));
  LOGI(
      "g * (m*m^T):\n%s",
      toStr(g * (m * m.transpose())).c_str());  // operator*(const Product&)
  ASSERT_TRUE((m * m.transpose()).isApprox(g * (m * m.transpose()), 1e-6));
  LOGI(
      "g * (2*m*m^T):\n%s",
      toStr(g * (2 * m * m.transpose())).c_str());  // operator*(const Product&)
  ASSERT_TRUE(
      (2 * m * m.transpose()).isApprox(g * (2 * m * m.transpose()), 1e-6));
  LOGI(
      "g * (m*m^T*m):\n%s",
      toStr(g * (m * m.transpose() * m)).c_str());  // operator*(const Product&)
  ASSERT_TRUE(
      (m * m.transpose() * m).isApprox(g * (m * m.transpose() * m), 1e-6));
  LOGI(
      "g * (m*m^T*m + m):\n%s",
      toStr(g * (m * m.transpose() * m + m))
          .c_str());  // operator*(const CwiseBinaryOp&)
  ASSERT_TRUE((m * m.transpose() * m + m)
                  .isApprox(g * (m * m.transpose() * m + m), 1e-6));
  LOGI("g * 2:\n%s", toStr(g * 2).c_str());
  ASSERT_TRUE((g.matrix() * 2).isApprox(g * 2, 1e-6));

  Eigen::MatrixXd m2 = m.transpose();
  LOGI("m2 * g:\n%s", toStr(m2 * g).c_str());
  ASSERT_TRUE(m2.isApprox(m2 * g, 1e-6));
  LOGI("m2.block * g:\n%s", toStr(m2.block<N - 1, N>(0, 0) * g).c_str());
  ASSERT_TRUE(
      (m2.block<N - 1, N>(0, 0).isApprox(m2.block<N - 1, N>(0, 0) * g, 1e-6)));
  LOGI("(-m2) * g:\n%s", toStr((-m2) * g).c_str());
  ASSERT_TRUE((-m2).isApprox((-m2) * g, 1e-6));
  LOGI("(m2+m2) * g:\n%s", toStr((m2 + m2) * g).c_str());
  ASSERT_TRUE((m2 + m2).isApprox((m2 + m2) * g, 1e-6));
  LOGI("(m2^T*m2) * g:\n%s", toStr((m2.transpose() * m2) * g).c_str());
  ASSERT_TRUE((m2.transpose() * m2).isApprox((m2.transpose() * m2) * g, 1e-6));
  LOGI("2 * g:\n%s", toStr(2 * g).c_str());
  ASSERT_TRUE((g.matrix() * 2).isApprox(2 * g, 1e-6));

  Eigen::MatrixXd m3(N, N);
  // m3.block<N,N>(0,0) = g;  // This is not supported yet.
  m3.block<N, N>(0, 0) = g.matrix();  // This is the workround.
  LOGI("m3:\n%s", toStr(m3).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m3.block<N, N>(0, 0), 1e-6));

  MatrixGroup::LieAlgebra tmp_v;
  tmp_v.setRandom();
  m3 = MatrixGroup::Exp(tmp_v);
  g = m3.block<N, N>(0, 0);
  LOGI("new m3:\n%s", toStr(m3).c_str());
  LOGI("new g.matrix():\n%s", toStr(g.matrix()).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m3.block<N, N>(0, 0), 1e-6));

  Eigen::MatrixXd m4;
  m4 = g;
  LOGI("m4:\n%s", toStr(m4).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m4, 1e-6));

  Eigen::MatrixXd m5(g);
  LOGI("m5:\n%s", toStr(m5).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m5, 1e-6));
}

TEST(TestMatrixGroupHelper, Sim3MatrixGroupCommonOps) {
  using MatrixGroup = Sim3d;
  static constexpr int N = 4;

  {
    // Test constructor from matrix
    Eigen::VectorXd v(7);
    v << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7;
    Sim3d g = Sim3d::Exp(v);
    LOGI("g.matrix():\n%s", toStr(g.matrix()).c_str());
    Sim3d g2 = g.matrix();  // construct g2 from matrix
    LOGI("g2.matrix():\n%s", toStr(g2.matrix()).c_str());
    ASSERT(g.matrix().isApprox(g2.matrix(), 1e-6));
  }

  MatrixGroup g(MatrixGroup::Identity());

  Eigen::MatrixXd m(N, N - 1);
  m.setRandom();
  LOGI("m:\n%s", toStr(m).c_str());
  LOGI("g * m:\n%s", toStr(g * m).c_str());  // operator*(const Matrix&)
  ASSERT_TRUE(m.isApprox(g * m, 1e-6));
  LOGI(
      "g * m.block:\n%s",
      toStr(g * m.block<N, N - 1>(0, 0)).c_str());  // operator*(const Block&)
  LOGI("m:\n%s", toStr(m).c_str());
  ASSERT_TRUE(m.isApprox(g * m.block<N, N - 1>(0, 0), 1e-6));
  LOGI(
      "g * m.block<N-1, N-1> (action on R3):\n%s",
      toStr(g * m.block<N - 1, N - 1>(0, 0))
          .c_str());  // operator*(const Block&)
  ASSERT_TRUE((m.block<N - 1, N - 1>(0, 0).isApprox(
      g * m.block<N - 1, N - 1>(0, 0), 1e-6)));
  LOGI(
      "g * (-m):\n%s",
      toStr(g * (-m)).c_str());  // operator*(const CwiseUnaryOp&)
  ASSERT_TRUE((-m).isApprox(g * (-m), 1e-6));
  LOGI(
      "g * (m+m):\n%s",
      toStr(g * (m + m)).c_str());  // operator*(const CwiseBinaryOp&)
  ASSERT_TRUE((m + m).isApprox(g * (m + m), 1e-6));
  LOGI(
      "g * (2*m):\n%s",
      toStr(g * (2 * m)).c_str());  // operator*(const CwiseBinaryOp&)
  ASSERT_TRUE((2 * m).isApprox(g * (2 * m), 1e-6));
  LOGI(
      "g * (m*m^T):\n%s",
      toStr(g * (m * m.transpose())).c_str());  // operator*(const Product&)
  ASSERT_TRUE((m * m.transpose()).isApprox(g * (m * m.transpose()), 1e-6));
  LOGI(
      "g * (2*m*m^T):\n%s",
      toStr(g * (2 * m * m.transpose())).c_str());  // operator*(const Product&)
  ASSERT_TRUE(
      (2 * m * m.transpose()).isApprox(g * (2 * m * m.transpose()), 1e-6));
  LOGI(
      "g * (m*m^T*m):\n%s",
      toStr(g * (m * m.transpose() * m)).c_str());  // operator*(const Product&)
  ASSERT_TRUE(
      (m * m.transpose() * m).isApprox(g * (m * m.transpose() * m), 1e-6));
  LOGI(
      "g * (m*m^T*m + m):\n%s",
      toStr(g * (m * m.transpose() * m + m))
          .c_str());  // operator*(const CwiseBinaryOp&)
  ASSERT_TRUE((m * m.transpose() * m + m)
                  .isApprox(g * (m * m.transpose() * m + m), 1e-6));
  LOGI("g * 2:\n%s", toStr(g * 2).c_str());
  ASSERT_TRUE((g.matrix() * 2).isApprox(g * 2, 1e-6));

  Eigen::MatrixXd m2 = m.transpose();
  LOGI("m2 * g:\n%s", toStr(m2 * g).c_str());
  ASSERT_TRUE(m2.isApprox(m2 * g, 1e-6));
  LOGI("m2.block * g:\n%s", toStr(m2.block<N - 1, N>(0, 0) * g).c_str());
  ASSERT_TRUE(
      (m2.block<N - 1, N>(0, 0).isApprox(m2.block<N - 1, N>(0, 0) * g, 1e-6)));
  LOGI("(-m2) * g:\n%s", toStr((-m2) * g).c_str());
  ASSERT_TRUE((-m2).isApprox((-m2) * g, 1e-6));
  LOGI("(m2+m2) * g:\n%s", toStr((m2 + m2) * g).c_str());
  ASSERT_TRUE((m2 + m2).isApprox((m2 + m2) * g, 1e-6));
  LOGI("(m2^T*m2) * g:\n%s", toStr((m2.transpose() * m2) * g).c_str());
  ASSERT_TRUE((m2.transpose() * m2).isApprox((m2.transpose() * m2) * g, 1e-6));
  LOGI("2 * g:\n%s", toStr(2 * g).c_str());
  ASSERT_TRUE((g.matrix() * 2).isApprox(2 * g, 1e-6));

  Eigen::MatrixXd m3(N, N);
  // m3.block<N,N>(0,0) = g;  // This is not supported yet.
  m3.block<N, N>(0, 0) = g.matrix();  // This is the workround.
  LOGI("m3:\n%s", toStr(m3).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m3.block<N, N>(0, 0), 1e-6));

  MatrixGroup::LieAlgebra tmp_v;
  tmp_v.setRandom();
  m3 = MatrixGroup::Exp(tmp_v);
  g = m3.block<N, N>(0, 0);
  LOGI("new m3:\n%s", toStr(m3).c_str());
  LOGI("new g.matrix():\n%s", toStr(g.matrix()).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m3.block<N, N>(0, 0), 1e-6));

  Eigen::MatrixXd m4;
  m4 = g;
  LOGI("m4:\n%s", toStr(m4).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m4, 1e-6));

  Eigen::MatrixXd m5(g);
  LOGI("m5:\n%s", toStr(m5).c_str());
  ASSERT_TRUE(g.matrix().isApprox(m5, 1e-6));
}

SK4SLAM_UNITTEST_ENTRYPOINT
