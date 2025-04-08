#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_camera/camera_model.h"

namespace sk4slam {

/// @brief Radial Tangential distortion model (Brown Conrady)
/// @tparam _num_distortion_params    the Number of distortion parameters
///
/// The projection parameters are:
/// @f$ f_x, f_y, c_x, c_y @f$
///
/// The distortion parameters are:
/// @f$ k_1, k_2, p_1, p_2, k_3 @f$
///
/// (@f$ k_3 @f$ is only present when @f$ \texttt{num_distortion_params} = 5
/// @f$).
///
/// The distortion model is:
///
/// @f$ x_d = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2
/// x^2) @f$
///
/// @f$ y_d = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) +
/// 2 p_2 x y @f$
///
/// where @f$ r^2 = x^2 + y^2 @f$ and @f$ (x, y) @f$ is the homogeneously
/// normalized coordinate of the 3D point @f$ (X, Y, Z) @f$ (in the camera
/// coordinate system), i.e.
///
/// @f$ x = X / Z @f$
///
/// @f$ y = Y / Z @f$.
///
/// @note This camera model only support FOV smaller than 180°.
template <int _num_distortion_params>
class RadTan : public CameraModel<
                   RadTan<_num_distortion_params>, 4, _num_distortion_params> {
  static_assert(
      _num_distortion_params == 0 || _num_distortion_params == 4 ||
          _num_distortion_params == 5,
      "RadTan camera model only supports 0 (for no distortion), 4 or 5 "
      "distortion parameters");

 public:
  using Base = CameraModel<RadTan, 4, _num_distortion_params>;
  using Base::kNumDistortionParams;
  using Base::kNumProjectionParams;

  using Interface = CameraModelInterface;
  using Interface::Vector2;
  using Interface::Vector3;

  static_assert(
      kNumProjectionParams == 4,
      "RadTan camera model only supports 4 projection parameters");
  static_assert(kNumDistortionParams == _num_distortion_params);

  template <typename... Args>
  explicit RadTan(Args&&... args) : Base(std::forward<Args>(args)...) {}

 public:
  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project2().
  template <typename Scalar>
  static bool project2Impl(
      const Scalar* point, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* pixel) {
    static const Scalar kNum_1 = Scalar(1.);
    static const Scalar kNum_2 = Scalar(2.);

    Scalar x = point[0], y = point[1];
    Scalar xd, yd;

    if constexpr (kNumDistortionParams > 0) {
      Scalar x2 = x * x;
      Scalar y2 = y * y;
      Scalar xy = x * y;
      Scalar r2 = x2 + y2;
      Scalar r4 = r2 * r2;
      Scalar f = kNum_1 + distortion_params[0] * r2 + distortion_params[1] * r4;

      if constexpr (kNumDistortionParams == 5) {
        Scalar r6 = r4 * r2;
        f += distortion_params[4] * r6;
      }

      // radial distortion
      Scalar xf = x * f;
      Scalar yf = y * f;

      // tangential distortion
      Scalar dx = kNum_2 * distortion_params[2] * xy +
                  distortion_params[3] * (r2 + kNum_2 * x2);
      Scalar dy = kNum_2 * distortion_params[3] * xy +
                  distortion_params[2] * (r2 + kNum_2 * y2);

      xd = xf + dx;
      yd = yf + dy;
    } else {
      xd = x;
      yd = y;
    }

    const Scalar& fx = projection_params[0];
    const Scalar& fy = projection_params[1];
    const Scalar& cx = projection_params[2];
    const Scalar& cy = projection_params[3];
    pixel[0] = xd * fx + cx;
    pixel[1] = yd * fy + cy;
    return true;
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project3().
  template <typename Scalar>
  static bool project3Impl(
      const Scalar* point, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* pixel) {
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    if (point[2] < kEps) {
      return false;
    }
    return project2Impl(
        Vector2<Scalar>(point[0] / point[2], point[1] / point[2]).data(),
        projection_params, distortion_params, pixel);
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::backProject2().
  template <typename Scalar>
  static bool backProject2Impl(
      const Scalar* pixel, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* point) {
    using std::abs;
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    static const Scalar kNum_1 = Scalar(1.);
    static const Scalar kNum_2 = Scalar(2.);
    Scalar x = pixel[0], y = pixel[1];
    const Scalar& fx = projection_params[0];
    const Scalar& fy = projection_params[1];
    const Scalar& cx = projection_params[2];
    const Scalar& cy = projection_params[3];
    x = (x - cx) / fx;
    y = (y - cy) / fy;

    if constexpr (kNumDistortionParams > 0) {
      Scalar xo = x;
      Scalar yo = y;

      // Loop until convergence, empirically 10 iterations is enough.
      for (int i = 0; i < 10; i++) {
        Scalar x2 = x * x;
        Scalar y2 = y * y;
        Scalar xy = x * y;
        Scalar r2 = x2 + y2;
        Scalar r4 = r2 * r2;
        Scalar f =
            kNum_1 + distortion_params[0] * r2 + distortion_params[1] * r4;
        if constexpr (kNumDistortionParams == 5) {
          Scalar r6 = r4 * r2;
          f += distortion_params[4] * r6;
        }

        // tangential distortion (approximated)
        Scalar dx = kNum_2 * distortion_params[2] * xy +
                    distortion_params[3] * (r2 + kNum_2 * x2);
        Scalar dy = kNum_2 * distortion_params[3] * xy +
                    distortion_params[2] * (r2 + kNum_2 * y2);
        Scalar new_x = (xo - dx) / f;
        Scalar new_y = (yo - dy) / f;
        if (abs(new_x - x) < kEps && abs(new_y - y) < kEps) {
          break;
        }
        x = new_x;
        y = new_y;
      }
    }

    point[0] = x;
    point[1] = y;
    return true;
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::backProject3().
  template <typename Scalar>
  static bool backProject3Impl(
      const Scalar* pixel, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* point) {
    bool success =
        backProject2Impl(pixel, projection_params, distortion_params, point);
    if (success) {
      static const Scalar kNum_1 = Scalar(1.);
      using std::sqrt;
      Scalar x = point[0];
      Scalar y = point[1];
      Scalar norm = sqrt(x * x + y * y + kNum_1);
      point[0] = x / norm;
      point[1] = y / norm;
      point[2] = kNum_1 / norm;
      return true;
    }
    return false;
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project3AndComputeJacobians().
  static bool project3AndComputeJacobiansImpl(
      const double* point, const double* proj_params, const double* dist_params,
      double* pixel, double* jacobian_wrt_point,
      double* jacobian_wrt_proj_params, double* jacobian_wrt_dist_params) {
    const double x = point[0] / point[2];
    const double y = point[1] / point[2];

    double xd, yd;
    double x2, y2, xy, r2, r4, f;  // for disortion
    double r6;  // r6 is only used if kNumDistortionParams == 5
    if constexpr (kNumDistortionParams > 0) {
      x2 = x * x;
      y2 = y * y;
      xy = x * y;
      r2 = x2 + y2;
      r4 = r2 * r2;
      f = 1.0 + dist_params[0] * r2 + dist_params[1] * r4;

      if constexpr (kNumDistortionParams == 5) {
        r6 = r4 * r2;
        f += dist_params[4] * r6;
      }

      // radial distortion
      const double xf = x * f;
      const double yf = y * f;

      // tangential distortion
      const double dx =
          2.0 * dist_params[2] * xy + dist_params[3] * (r2 + 2.0 * x2);
      const double dy =
          2.0 * dist_params[3] * xy + dist_params[2] * (r2 + 2.0 * y2);

      xd = xf + dx;
      yd = yf + dy;
    } else {
      xd = x;
      yd = y;
    }

    const double& fx = proj_params[0];
    const double& fy = proj_params[1];
    const double& cx = proj_params[2];
    const double& cy = proj_params[3];
    if (pixel) {
      pixel[0] = xd * fx + cx;
      pixel[1] = yd * fy + cy;
    }

    if (jacobian_wrt_proj_params) {
      jacobian_wrt_proj_params[0] = xd;
      jacobian_wrt_proj_params[1] = 0;
      jacobian_wrt_proj_params[2] = 1;
      jacobian_wrt_proj_params[3] = 0;

      jacobian_wrt_proj_params[4] = 0;
      jacobian_wrt_proj_params[5] = yd;
      jacobian_wrt_proj_params[6] = 0;
      jacobian_wrt_proj_params[7] = 1;
    }

    if constexpr (kNumDistortionParams > 0) {
      if (jacobian_wrt_dist_params) {
        const double jacobian_wrt_f_0 = x * fx;
        const double jacobian_wrt_f_1 = y * fy;
        const double jacobian_wrt_dx = fx;
        const double jacobian_wrt_dy = fy;

        jacobian_wrt_dist_params[0] = r2 * jacobian_wrt_f_0;
        jacobian_wrt_dist_params[1] = r4 * jacobian_wrt_f_0;
        jacobian_wrt_dist_params[2] = 2.0 * xy * jacobian_wrt_dx;
        jacobian_wrt_dist_params[3] = (r2 + 2.0 * x2) * jacobian_wrt_dx;
        if constexpr (kNumDistortionParams == 5) {
          jacobian_wrt_dist_params[4] = r6 * jacobian_wrt_f_0;
        }
        jacobian_wrt_dist_params[kNumDistortionParams + 0] =
            r2 * jacobian_wrt_f_1;
        jacobian_wrt_dist_params[kNumDistortionParams + 1] =
            r4 * jacobian_wrt_f_1;
        jacobian_wrt_dist_params[kNumDistortionParams + 2] =
            (r2 + 2.0 * y2) * jacobian_wrt_dy;
        jacobian_wrt_dist_params[kNumDistortionParams + 3] =
            2.0 * xy * jacobian_wrt_dy;
        if constexpr (kNumDistortionParams == 5) {
          jacobian_wrt_dist_params[kNumDistortionParams + 4] =
              r6 * jacobian_wrt_f_1;
        }
      }
    }

    if (jacobian_wrt_point) {
      double cx_px, cx_py, cy_px, cy_py;

      if constexpr (kNumDistortionParams > 0) {
        const double pr2_px = 2.0 * x;
        const double pr2_py = 2.0 * y;
        const double pr4_px = 2.0 * r2 * pr2_px;
        const double pr4_py = 2.0 * r2 * pr2_py;
        double pf_px = dist_params[0] * pr2_px + dist_params[1] * pr4_px;
        double pf_py = dist_params[0] * pr2_py + dist_params[1] * pr4_py;
        if constexpr (kNumDistortionParams == 5) {
          double pr6_px = 3.0 * r4 * pr2_px;
          double pr6_py = 3.0 * r4 * pr2_py;
          pf_px += dist_params[4] * pr6_px;
          pf_py += dist_params[4] * pr6_py;
        }

        // dx = 2.0 * dist_params[2] * xy + dist_params[3] * (r2 + 2.0 * x2);
        // dy = 2.0 * dist_params[3] * xy + dist_params[2] * (r2 + 2.0 * y2);
        const double pdx_px =
            2.0 * dist_params[2] * y + dist_params[3] * (6.0 * x);
        const double pdx_py =
            2.0 * dist_params[2] * x + dist_params[3] * (2.0 * y);
        const double pdy_px =
            2.0 * dist_params[3] * y + dist_params[2] * (2.0 * x);
        const double pdy_py =
            2.0 * dist_params[3] * x + dist_params[2] * (6.0 * y);

        cx_px = (f + x * pf_px + pdx_px) * fx;
        cx_py = (x * pf_py + pdx_py) * fx;
        cy_px = (y * pf_px + pdy_px) * fy;
        cy_py = (f + y * pf_py + pdy_py) * fy;
      } else {
        cx_px = fx;
        cx_py = 0.;
        cy_px = 0.;
        cy_py = fy;
      }

      // double d = point[2];
      // double d2 = d * d;
      // jacobian_wrt_point[0] = cx_px / d;
      // jacobian_wrt_point[1] = cx_py / d;
      // jacobian_wrt_point[2] = -cx_px * point[0] / d2 - cx_py * point[1] / d2;
      // jacobian_wrt_point[3] = cy_px / d;
      // jacobian_wrt_point[4] = cy_py / d;
      // jacobian_wrt_point[5] = -cy_px * point[0] / d2 - cy_py * point[1] / d2;

      double jacobian_wrt_pxpy[4] = {cx_px, cx_py, cy_px, cy_py};
      double hnormalized[2] = {x, y};
      Interface::convertJacobianWrtHnormalizedToJacobianWrtPoint3(
          jacobian_wrt_pxpy, hnormalized, point[2], jacobian_wrt_point);
    }
    return true;
  }
};

using RadTan4 = RadTan<4>;
using RadTan5 = RadTan<5>;

}  // namespace sk4slam
