#pragma once

#include "sk4slam_liegroups/Rp_x_SOn.h"
#include "sk4slam_liegroups/SE2.h"
#include "sk4slam_liegroups/SE3.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/Sim2.h"
#include "sk4slam_liegroups/Sim3.h"

/// Define py-interface for general LieGroup
// clang-format off
#define DEFINE_LIE_GROUP_PY_INTERFACE(LieGroup)                               \
  def(py::init<>(/*Default Constructor*/))                                    \
      .def("inverse", &LieGroup::inverse)                                     \
      .def(                                                                   \
          "__mul__",                                                          \
          static_cast<LieGroup (LieGroup::*)(const LieGroup&) const>(         \
              &LieGroup::operator*))                                          \
      .def("isApprox", &LieGroup::isApprox)                                   \
      .def_static("Identity", &LieGroup::Identity)                            \
      .def_static("Log", &LieGroup::Log)                                      \
      .def_static("Exp", &LieGroup::Exp)                                      \
      .def_static(                                                            \
          "Ad",                                                               \
          static_cast<LieGroup::LieAlgebraEndomorphism(*)(const LieGroup&)>(  \
              &LieGroup::Ad))                                                 \
      .def_static(                                                            \
          "Ad",                                                               \
          static_cast<LieGroup::LieAlgebra(*)(                                \
              const LieGroup&, const LieGroup::LieAlgebra&)>(&LieGroup::Ad))  \
      .def_static("ad", &LieGroup::ad)                                        \
      .def_static("bracket", &LieGroup::bracket)                              \
      .def_static("Jl", &LieGroup::Jl)                                        \
      .def_static("invJl", &LieGroup::invJl)                                  \
      .def_static("Jr", &LieGroup::Jr)                                        \
      .def_static("invJr", &LieGroup::invJr)                                  \
      .def_static("generator", &LieGroup::generator)                          \
      .def_static("hat", &LieGroup::hat)                                      \
      .def_static("vee", &LieGroup::vee)
// clang-format on

/// Define py-interface for Product LieGroup
// clang-format off
#define DEFINE_PRODUCT_LIE_GROUP_PY_INTERFACE(ProductGroup)                 \
  def(py::init<>(/*Default Constructor*/))                                  \
      .def("inverse", &ProductGroup::inverse<ProductGroup>)                 \
      .def(                                                                 \
          "__mul__",                                                        \
          static_cast<ProductGroup (ProductGroup::*)(const ProductGroup&)   \
                          const>(&ProductGroup::operator* <ProductGroup>))  \
      .def("isApprox", &ProductGroup::isApprox<ProductGroup>)               \
      .def_static("Identity", &ProductGroup::Identity<ProductGroup>)        \
      .def_static("Log", &ProductGroup::Log<ProductGroup>)                  \
      .def_static("Exp", &ProductGroup::Exp<ProductGroup>)                  \
      .def_static(                                                          \
          "Ad",                                                             \
          static_cast<ProductGroup::LieAlgebraEndomorphism(*)(              \
              const ProductGroup&)>(&ProductGroup::Ad<ProductGroup>))       \
      .def_static(                                                          \
          "Ad", static_cast<ProductGroup::LieAlgebra(*)(                    \
                    const ProductGroup&, const ProductGroup::LieAlgebra&)>( \
                    &ProductGroup::Ad<ProductGroup>))                       \
      .def_static("ad", &ProductGroup::ad)                                  \
      .def_static("bracket", &ProductGroup::bracket)                        \
      .def_static("Jl", &ProductGroup::Jl)                                  \
      .def_static("invJl", &ProductGroup::invJl)                            \
      .def_static("Jr", &ProductGroup::Jr)                                  \
      .def_static("invJr", &ProductGroup::invJr)                            \
      .def_static("generator", &ProductGroup::generator)                    \
      .def_static("hat", &ProductGroup::hat)                                \
      .def_static("vee", &ProductGroup::vee)
// clang-format on

/// Define additional py-interface for MatrixGroup (based on LieGroup)
#define DEFINE_ADDITIONAL_PY_INTERFACE_FOR_MATRIX_GROUP(MatrixGroup)     \
  def(py::init<const Eigen::MatrixXd&>()) /*Construct from matrix*/      \
      .def("matrix", &MatrixGroup::matrix)                               \
      .def(                                                              \
          "__mul__", /* Multiply by vector from the right */             \
          [](const MatrixGroup& g, const Eigen::VectorXd& mat) {         \
            return Eigen::VectorXd(g * mat);                             \
          },                                                             \
          py::is_operator())                                             \
      .def(                                                              \
          "__mul__", /* Multiply by matrix from the right */             \
          [](const MatrixGroup& g, const Eigen::MatrixXd& mat) {         \
            return g * mat;                                              \
          },                                                             \
          py::is_operator())                                             \
      .def(                                                              \
          "__rmul__", /* Multiply by matrix from the left */             \
          [](const MatrixGroup& g, const Eigen::MatrixXd& mat) {         \
            return mat * g;                                              \
          },                                                             \
          py::is_operator())                                             \
      .def(                                                              \
          "__mul__", /* Multiply by scalar from the right */             \
          [](const MatrixGroup& g, double scalar) {                      \
            return g.matrix() * scalar;                                  \
          },                                                             \
          py::is_operator())                                             \
      .def(                                                              \
          "__rmul__", /* Multiply by scalar from the left */             \
          [](const MatrixGroup& g, double scalar) {                      \
            return scalar * g.matrix();                                  \
          },                                                             \
          py::is_operator())                                             \
      .def_static("JmultVector", &MatrixGroup::JmultVector<MatrixGroup>) \
      .def_static(                                                       \
          "matrixGenerator", &MatrixGroup::matrixGenerator<MatrixGroup>)

/// Define py-interface for MatrixGroup
#define DEFINE_MATRIX_GROUP_PY_INTERFACE(MatrixGroup) \
  DEFINE_LIE_GROUP_PY_INTERFACE(MatrixGroup)          \
      .DEFINE_ADDITIONAL_PY_INTERFACE_FOR_MATRIX_GROUP(MatrixGroup)

/// Define py-interface for ProductMatrixGroup
#define DEFINE_PRODUCT_MATRIX_GROUP_PY_INTERFACE(ProductMatrixGroup) \
  DEFINE_PRODUCT_LIE_GROUP_PY_INTERFACE(ProductMatrixGroup)          \
      .DEFINE_ADDITIONAL_PY_INTERFACE_FOR_MATRIX_GROUP(ProductMatrixGroup)

/// Define additional py-interface for AffineGroup (based on MatrixGroup)
#define DEFINE_ADDITIONAL_PY_INTERFACE_FOR_AFFINE_GROUP(AffineGroup)           \
  def(py::init<                                                                \
          const Eigen::MatrixXd&,                                              \
          const Eigen::VectorXd&>(/*Construct from a linear transformation*/   \
                                  /*and a translation vector*/))               \
      .def(py::init<                                                           \
           const decltype(std::declval<AffineGroup>().linear())&,              \
           const Eigen::VectorXd&>(/*Construct from a linear transformation*/  \
                                   /*and a translation vector*/))              \
      .def(                                                                    \
          "linear", py::overload_cast<>(                                       \
                        &AffineGroup::linear, py::const_)) /* const version */ \
      .def(                                                                    \
          "setLinear",                                                         \
          [](AffineGroup& g,                                                   \
             const decltype(std::declval<AffineGroup>().linear())& linear) {   \
            g.linear() = linear;                                               \
          }) /* non-const version */                                           \
      .def(                                                                    \
          "translation",                                                       \
          py::overload_cast<>(                                                 \
              &AffineGroup::translation, py::const_)) /* const version */      \
      .def("setTranslation", [](AffineGroup& g, const Eigen::MatrixXd& t) {    \
        g.translation() = t;                                                   \
      }) /* non-const version */

/// Define py-interface for AffineGroup
#define DEFINE_AFFINE_GROUP_PY_INTERFACE(AffineGroup) \
  DEFINE_MATRIX_GROUP_PY_INTERFACE(AffineGroup)       \
      .DEFINE_ADDITIONAL_PY_INTERFACE_FOR_AFFINE_GROUP(AffineGroup)

/// Define Rp specific py-interface (Based on MatrixGroup)
#define DEFINE_Rp_SPECIFIC_PY_INTERFACE()                                   \
  def(py::init<double>(/*Construct from a linear transformation*/           \
                       /*and a translation vector*/))                       \
      .def(                                                                 \
          "value",                                                          \
          py::overload_cast<>(&Rpd::value, py::const_)) /* const version */ \
      .def("setValue", [](Rpd& g, double v) {                               \
        g.value() = v;                                                      \
      }) /* non-const version */

/// Define SO2 specific py-interface (Based on MatrixGroup)
#define DEFINE_SO2_SPECIFIC_PY_INTERFACE()                           \
  def("normalize", &SO2d::normalize)                                 \
      .def("normalized", &SO2d::normalized)                          \
      .def("toAngle", [](const SO2d& g) { return SO2d::Log(g)(0); }) \
      .def_static("FromAngle", [](double angle) { return SO2d(angle); })

/// Define SO3 specific py-interface (Based on MatrixGroup)
#define DEFINE_SO3_SPECIFIC_PY_INTERFACE()                                  \
  def("normalize", &SO3d::normalize)                                        \
      .def("normalized", &SO3d::normalized)                                 \
      .def(                                                                 \
          "toQuaternion",                                                   \
          [](const SO3d& g) {                                               \
            const auto& c = Eigen::Quaterniond(g.matrix()).coeffs();        \
            return Eigen::Vector4d(c[3], c[0], c[1], c[2]);                 \
          })                                                                \
      .def_static("FromTwoUnitVectors", &SO3d::FromTwoUnitVectors)          \
      .def_static(                                                          \
          "FromQuaternion",                                                 \
          [](double w, double x, double y, double z) {                      \
            return SO3d(Eigen::Quaterniond(w, x, y, z).toRotationMatrix()); \
          })                                                                \
      .def_static(                                                          \
          "FromAxisAngle", [](const Eigen::Vector3d& axis, double angle) {  \
            return SO3d::Exp(axis.normalized() * angle);                    \
          })

/// Define Rp_x_SOn specific py-interface (Based on ProductMatrixGroup)
#define DEFINE_Rp_x_SOn_SPECIFIC_PY_INTERFACE(Rp_x_SOn)                        \
  def(py::init<double, const Eigen::MatrixXd&>(                                \
          /*Construct from a scale *and a rotation*/))                         \
      .def(py::init<                                                           \
           double, const decltype(std::declval<Rp_x_SOn>().rotation())&>(      \
          /*Construct from a scale *and a rotation*/))                         \
      .def(                                                                    \
          "scale", py::overload_cast<>(                                        \
                       &Rp_x_SOn::scale, py::const_)) /* const version */      \
      .def(                                                                    \
          "setScale", [](Rp_x_SOn& g,                                          \
                         double s) { g.scale() = s; }) /* non-const version */ \
      .def(                                                                    \
          "rotation",                                                          \
          py::overload_cast<>(                                                 \
              &Rp_x_SOn::rotation, py::const_)) /* const version */            \
      .def(                                                                    \
          "setRotation",                                                       \
          [](Rp_x_SOn& g,                                                      \
             const decltype(std::declval<Rp_x_SOn>().rotation())& rot) {       \
            g.rotation() = rot;                                                \
          }) /* non-const version */

/// Define SE(n) specific py-interface (Based on AffineGroup)
#define DEFINE_SEn_SPECIFIC_PY_INTERFACE(SEn)                               \
  def("rotation",                                                           \
      py::overload_cast<>(&SEn::rotation, py::const_) /* const version */)  \
      .def(                                                                 \
          "setRotation",                                                    \
          [](SEn& g, const decltype(std::declval<SEn>().rotation())& rot) { \
            g.rotation() = rot;                                             \
          }) /* non-const version */

/// Define Sim(n) specific py-interface (Based on AffineGroup)
#define DEFINE_SIMn_SPECIFIC_PY_INTERFACE(SIMn)                               \
  def("rotation",                                                             \
      py::overload_cast<>(&SIMn::rotation, py::const_) /* const version */)   \
      .def(                                                                   \
          "setRotation",                                                      \
          [](SIMn& g, const decltype(std::declval<SIMn>().rotation())& rot) { \
            g.rotation() = rot;                                               \
          }) /* non-const version */                                          \
      .def(                                                                   \
          "scale",                                                            \
          py::overload_cast<>(&SIMn::scale, py::const_)) /* const version */  \
      .def("setScale", [](SIMn& g, double s) {                                \
        g.scale() = s;                                                        \
      }) /* non-const version */
