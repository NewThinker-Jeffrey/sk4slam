#include "liegroups_py.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
PYBIND11_MODULE(sk4slam_liegroups, m) {
  using namespace sk4slam;  // NOLINT

  py::class_<Rpd>(m, "Rp")
      .DEFINE_MATRIX_GROUP_PY_INTERFACE(Rpd)
      .DEFINE_Rp_SPECIFIC_PY_INTERFACE();

  py::class_<SO2d>(m, "SO2")
      .DEFINE_MATRIX_GROUP_PY_INTERFACE(SO2d)
      .DEFINE_SO2_SPECIFIC_PY_INTERFACE();

  py::class_<SO3d>(m, "SO3")
      .DEFINE_MATRIX_GROUP_PY_INTERFACE(SO3d)
      .DEFINE_SO3_SPECIFIC_PY_INTERFACE();

  py::class_<SE2d>(m, "SE2")
      .DEFINE_AFFINE_GROUP_PY_INTERFACE(SE2d)
      .DEFINE_SEn_SPECIFIC_PY_INTERFACE(SE2d);

  py::class_<SE3d>(m, "SE3")
      .DEFINE_AFFINE_GROUP_PY_INTERFACE(SE3d)
      .DEFINE_SEn_SPECIFIC_PY_INTERFACE(SE3d);

  py::class_<Rp_x_SO2d>(m, "Rp_x_SO2")
      .DEFINE_PRODUCT_MATRIX_GROUP_PY_INTERFACE(Rp_x_SO2d)
      .DEFINE_Rp_x_SOn_SPECIFIC_PY_INTERFACE(Rp_x_SO2d);

  py::class_<Rp_x_SO3d>(m, "Rp_x_SO3")
      .DEFINE_PRODUCT_MATRIX_GROUP_PY_INTERFACE(Rp_x_SO3d)
      .DEFINE_Rp_x_SOn_SPECIFIC_PY_INTERFACE(Rp_x_SO3d);

  py::class_<Sim2d>(m, "Sim2")
      .DEFINE_AFFINE_GROUP_PY_INTERFACE(Sim2d)
      .DEFINE_SIMn_SPECIFIC_PY_INTERFACE(Sim2d);

  py::class_<Sim3d>(m, "Sim3")
      .DEFINE_AFFINE_GROUP_PY_INTERFACE(Sim3d)
      .DEFINE_SIMn_SPECIFIC_PY_INTERFACE(Sim3d);
}
