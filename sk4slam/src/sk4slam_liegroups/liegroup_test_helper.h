#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_liegroups/bch.h"
#include "sk4slam_liegroups/general_computation.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_math/math.h"
#include "sk4slam_math/random.h"

////////////////////////////////////////////////////////////////////////////////
// Test GroupOpJacobian
////////////////////////////////////////////////////////////////////////////////
#define LIE_GROUP_TEST_GroupOpJacobian(LieGroup, PerturbationName)             \
  {                                                                            \
    LOGI("LIE_GROUP_TEST_GroupOpJacobian: *****************************");     \
    LOGI("### " #LieGroup "::" #PerturbationName)                              \
    const double delta_scale = 1e-4;                                           \
    const double approx_eps = 1e-3;                                            \
    using Perturbation = LieGroup::PerturbationName;                           \
    using LieAlgebra = LieGroup::LieAlgebra;                                   \
    using EndomophismMatrix =                                                  \
        Eigen::Matrix<double, LieGroup::kDim, LieGroup::kDim>;                 \
    auto dist =                                                                \
        MultivariateUniformDistribution<double>::standard(LieGroup::kDim);     \
    const Perturbation* perturbation =                                         \
        RetractionInterface::defaultInstance<Perturbation>();                  \
    LieGroup::LieAlgebra X1 = dist();                                          \
    LieGroup::LieAlgebra X2 = dist();                                          \
    LieGroup g1 = LieGroup::Exp(X1);                                           \
    LieGroup g2 = LieGroup::Exp(X2);                                           \
                                                                               \
    LOGI("X1:\n%s", toStr(X1.transpose()).c_str());                            \
    LOGI("X2:\n%s", toStr(X2.transpose()).c_str());                            \
                                                                               \
    EndomophismMatrix mult_J1, mult_J2;                                        \
    EndomophismMatrix inv_J;                                                   \
    EndomophismMatrix right_delta_J1, right_delta_J2;                          \
    EndomophismMatrix left_delta_J1, left_delta_J2;                            \
                                                                               \
    LieGroup right_delta = g1.inverse() * g2;                                  \
    LieGroup left_delta = g2 * g1.inverse();                                   \
    LieGroup product = g1 * g2;                                                \
    LieGroup inv = g1.inverse();                                               \
    LieGroup test_right_delta, test_left_delta, test_product, test_inv;        \
                                                                               \
    perturbation->Inverse(g1, &test_inv, &inv_J);                              \
    ASSERT_TRUE(test_inv.isApprox(inv, 1e-6));                                 \
    {                                                                          \
      LieGroup::LieAlgebra dx1 = dist() * delta_scale;                         \
      LieGroup new_g1 = (*perturbation)(g1, dx1);                              \
      LieGroup new_inv = new_g1.inverse();                                     \
      LieGroup::LieAlgebra delta_inv = perturbation->section(inv, new_inv);    \
      LieGroup::LieAlgebra approx_delta_inv = inv_J * dx1;                     \
      LOGI(                                                                    \
          "       delta_inv: %s",                                              \
          toStr(delta_inv.transpose(), Precision(8)).c_str());                 \
      LOGI(                                                                    \
          "approx_delta_inv: %s",                                              \
          toStr(approx_delta_inv.transpose(), Precision(8)).c_str());          \
      ASSERT_TRUE(delta_inv.isApprox(approx_delta_inv, approx_eps));           \
    }                                                                          \
                                                                               \
    perturbation->Multiply(g1, g2, &test_product, &mult_J1, &mult_J2);         \
    ASSERT_TRUE(test_product.isApprox(product, 1e-6));                         \
    {                                                                          \
      LieGroup::LieAlgebra dx1 = dist() * delta_scale;                         \
      LieGroup::LieAlgebra dx2 = dist() * delta_scale;                         \
      LieGroup new_g1 = (*perturbation)(g1, dx1);                              \
      LieGroup new_g2 = (*perturbation)(g2, dx2);                              \
      LieGroup new_product = new_g1 * new_g2;                                  \
      LieGroup::LieAlgebra delta_prd =                                         \
          perturbation->section(product, new_product);                         \
      LieGroup::LieAlgebra approx_delta_prd = mult_J1 * dx1 + mult_J2 * dx2;   \
      LOGI(                                                                    \
          "       delta_prd: %s",                                              \
          toStr(delta_prd.transpose(), Precision(8)).c_str());                 \
      LOGI(                                                                    \
          "approx_delta_prd: %s",                                              \
          toStr(approx_delta_prd.transpose(), Precision(8)).c_str());          \
      ASSERT_TRUE(delta_prd.isApprox(approx_delta_prd, approx_eps));           \
    }                                                                          \
    perturbation->RightDelta(                                                  \
        g1, g2, &test_right_delta, &right_delta_J1, &right_delta_J2);          \
    ASSERT_TRUE(test_right_delta.isApprox(right_delta, 1e-6));                 \
    {                                                                          \
      EndomophismMatrix tmp_J1, tmp_J2;                                        \
      LieGroup tmp_prd;                                                        \
      perturbation->Multiply(g1.inverse(), g2, &tmp_prd, &tmp_J1, &tmp_J2);    \
      LOGI(                                                                    \
          "right_delta_J1:\n%s", toStr(right_delta_J1, Precision(8)).c_str()); \
      LOGI("tmp_J1 * inv_J:\n%s", toStr(tmp_J1* inv_J, Precision(8)).c_str()); \
      LOGI(                                                                    \
          "right_delta_J2:\n%s", toStr(right_delta_J2, Precision(8)).c_str()); \
      LOGI("tmp_J2:\n%s", toStr(tmp_J2, Precision(8)).c_str());                \
      ASSERT_TRUE(tmp_prd.isApprox(test_right_delta, 1e-6));                   \
      ASSERT_TRUE(tmp_J2.isApprox(right_delta_J2, 1e-6));                      \
      ASSERT_TRUE((tmp_J1 * inv_J).isApprox(right_delta_J1, 1e-6));            \
                                                                               \
      if (std::is_same_v<Perturbation, LieGroup::RightPerturbation>) {         \
        EndomophismMatrix section_J1, section_J2;                              \
        LieAlgebra delta_lie_alg;                                              \
        perturbation->sectionImpl(                                             \
            g1, g2, &delta_lie_alg, &section_J1, &section_J2);                 \
        ASSERT_TRUE(                                                           \
            LieGroup::Exp(delta_lie_alg).isApprox(test_right_delta, 1e-6));    \
        ASSERT_TRUE(                                                           \
            (EndomophismMatrix(LieGroup::Jr(delta_lie_alg)) * section_J1)      \
                .isApprox(right_delta_J1, 1e-6));                              \
        ASSERT_TRUE(                                                           \
            (EndomophismMatrix(LieGroup::Jr(delta_lie_alg)) * section_J2)      \
                .isApprox(right_delta_J2, 1e-6));                              \
      }                                                                        \
    }                                                                          \
                                                                               \
    perturbation->LeftDelta(                                                   \
        g1, g2, &test_left_delta, &left_delta_J1, &left_delta_J2);             \
    ASSERT_TRUE(test_left_delta.isApprox(left_delta, 1e-6));                   \
    {                                                                          \
      EndomophismMatrix tmp_J1, tmp_J2;                                        \
      LieGroup tmp_prd;                                                        \
      perturbation->Multiply(g2, g1.inverse(), &tmp_prd, &tmp_J2, &tmp_J1);    \
      LOGI("left_delta_J1:\n%s", toStr(left_delta_J1, Precision(8)).c_str());  \
      LOGI("tmp_J1 * inv_J:\n%s", toStr(tmp_J1* inv_J, Precision(8)).c_str()); \
      LOGI("left_delta_J2:\n%s", toStr(left_delta_J2, Precision(8)).c_str());  \
      LOGI("tmp_J2:\n%s", toStr(tmp_J2, Precision(8)).c_str());                \
      ASSERT_TRUE(tmp_prd.isApprox(test_left_delta, 1e-6));                    \
      ASSERT_TRUE((tmp_J1 * inv_J).isApprox(left_delta_J1, 1e-6));             \
      ASSERT_TRUE(tmp_J2.isApprox(left_delta_J2, 1e-6));                       \
                                                                               \
      if (std::is_same_v<Perturbation, LieGroup::LeftPerturbation>) {          \
        EndomophismMatrix section_J1, section_J2;                              \
        LieAlgebra delta_lie_alg;                                              \
        perturbation->sectionImpl(                                             \
            g1, g2, &delta_lie_alg, &section_J1, &section_J2);                 \
        ASSERT_TRUE(                                                           \
            LieGroup::Exp(delta_lie_alg).isApprox(test_left_delta, 1e-6));     \
        ASSERT_TRUE(                                                           \
            (EndomophismMatrix(LieGroup::Jl(delta_lie_alg)) * section_J1)      \
                .isApprox(left_delta_J1, 1e-6));                               \
        ASSERT_TRUE(                                                           \
            (EndomophismMatrix(LieGroup::Jl(delta_lie_alg)) * section_J2)      \
                .isApprox(left_delta_J2, 1e-6));                               \
      }                                                                        \
    }                                                                          \
  }

////////////////////////////////////////////////////////////////////////////////
// Test TransformVectorJacobian
////////////////////////////////////////////////////////////////////////////////
#define LIE_GROUP_TEST_TransformVectorJacobian(LieGroup, PerturbationName)   \
  {                                                                          \
    LOGI("LIE_GROUP_TEST_TransformVectorJacobian: ********************");    \
    LOGI("### " #LieGroup "::" #PerturbationName)                            \
    const double delta_scale = 1e-4;                                         \
    const double approx_eps = 1e-3;                                          \
    using Perturbation = LieGroup::PerturbationName;                         \
    using LieAlgebra = LieGroup::LieAlgebra;                                 \
    static constexpr int N = MatrixGroupHelper<LieGroup>::N;                 \
    static constexpr int kDim = LieGroup::kDim;                              \
    using JacobianMatrix = Eigen::Matrix<double, N, kDim>;                   \
    using VectorN = Vector<N>;                                               \
    const Perturbation* perturbation =                                       \
        RetractionInterface::defaultInstance<Perturbation>();                \
    auto dist =                                                              \
        MultivariateUniformDistribution<double>::standard(LieGroup::kDim);   \
    auto distN = MultivariateUniformDistribution<double>::standard(N);       \
    LieGroup::LieAlgebra X = dist();                                         \
    LieGroup g = LieGroup::Exp(X);                                           \
    VectorN v = distN();                                                     \
    VectorN expected_result = g * v;                                         \
    VectorN result;                                                          \
    JacobianMatrix J;                                                        \
    perturbation->TransformVector(g, v, &result, &J);                        \
    LOGI("              v: %s", toStr(v.transpose()).c_str());               \
    LOGI("expected_result: %s", toStr(expected_result.transpose()).c_str()); \
    LOGI("         result: %s", toStr(result.transpose()).c_str());          \
    LOGI("J:\n%s", toStr(J).c_str());                                        \
    ASSERT_TRUE(result.isApprox(expected_result));                           \
    LieGroup::LieAlgebra delta = delta_scale * dist();                       \
    LieGroup new_g = (*perturbation)(g, delta);                              \
    LOGI("delta: %s", toStr(delta.transpose(), Precision(8)).c_str());       \
    VectorN delta_v = new_g * v - expected_result;                           \
    VectorN approx_delta_v = J * delta;                                      \
    LOGI(                                                                    \
        "approx_delta_v: %s",                                                \
        toStr(approx_delta_v.transpose(), Precision(8)).c_str());            \
    LOGI(                                                                    \
        "       delta_v: %s",                                                \
        toStr(delta_v.transpose(), Precision(8)).c_str());                   \
    ASSERT_TRUE(approx_delta_v.isApprox(delta_v, approx_eps));               \
  }

////////////////////////////////////////////////////////////////////////////////
// Test TransformHomoVectorJacobian
////////////////////////////////////////////////////////////////////////////////
#define LIE_GROUP_TEST_TransformHomoVectorJacobian(LieGroup, PerturbationName) \
  {                                                                            \
    LOGI("LIE_GROUP_TEST_TransformHomoVectorJacobian: ********************");  \
    LOGI("### " #LieGroup "::" #PerturbationName)                              \
    const double delta_scale = 1e-4;                                           \
    const double approx_eps = 1e-3;                                            \
    using Perturbation = LieGroup::PerturbationName;                           \
    using LieAlgebra = LieGroup::LieAlgebra;                                   \
    static constexpr int N = LieGroup::N;                                      \
    static constexpr int kDim = LieGroup::kDim;                                \
    using JacobianMatrix = Eigen::Matrix<double, N, kDim>;                     \
    using VectorN = Vector<N>;                                                 \
    const Perturbation* perturbation =                                         \
        RetractionInterface::defaultInstance<Perturbation>();                  \
    auto dist =                                                                \
        MultivariateUniformDistribution<double>::standard(LieGroup::kDim);     \
    auto distN = MultivariateUniformDistribution<double>::standard(N);         \
    LieGroup::LieAlgebra X = dist();                                           \
    LieGroup g = LieGroup::Exp(X);                                             \
    VectorN v = distN();                                                       \
    VectorN expected_result = g * v;                                           \
    VectorN result;                                                            \
    JacobianMatrix J;                                                          \
    perturbation->TransformHomoVector(g, v, &result, &J);                      \
    LOGI("              v: %s", toStr(v.transpose()).c_str());                 \
    LOGI("expected_result: %s", toStr(expected_result.transpose()).c_str());   \
    LOGI("         result: %s", toStr(result.transpose()).c_str());            \
    LOGI("J:\n%s", toStr(J).c_str());                                          \
    ASSERT_TRUE(result.isApprox(expected_result));                             \
    LieGroup::LieAlgebra delta = delta_scale * dist();                         \
    LieGroup new_g = (*perturbation)(g, delta);                                \
    LOGI("delta: %s", toStr(delta.transpose(), Precision(8)).c_str());         \
    VectorN delta_v = new_g * v - expected_result;                             \
    VectorN approx_delta_v = J * delta;                                        \
    LOGI(                                                                      \
        "approx_delta_v: %s",                                                  \
        toStr(approx_delta_v.transpose(), Precision(8)).c_str());              \
    LOGI(                                                                      \
        "       delta_v: %s",                                                  \
        toStr(delta_v.transpose(), Precision(8)).c_str());                     \
    ASSERT_TRUE(approx_delta_v.isApprox(delta_v, approx_eps));                 \
    {                                                                          \
      using VectorNp1 = Vector<N + 1>;                                         \
      VectorNp1 v1, result1;                                                   \
      v1 << v, 1;                                                              \
      using AugJacobianMatrix = Eigen::Matrix<double, N + 1, kDim>;            \
      AugJacobianMatrix J1;                                                    \
      perturbation->TransformVector(g, v1, &result1, &J1);                     \
      LOGI("expected_result: %s", toStr(expected_result.transpose()).c_str()); \
      LOGI("        result1: %s", toStr(result1.transpose()).c_str());         \
      LOGI("J1:\n%s", toStr(J1).c_str());                                      \
      ASSERT_TRUE(result1.head<N>().isApprox(expected_result));                \
      ASSERT_TRUE(J1.topRows<N>().isApprox(J));                                \
    }                                                                          \
  }
