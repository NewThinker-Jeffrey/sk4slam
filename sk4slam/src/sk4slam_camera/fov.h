#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_camera/camera_model.h"

namespace sk4slam {

/// @brief FOV distortion model (see
/// https://www.researchgate.net/publication/29638148_Straight_Lines_Have_to_Be_Straight_Automatic_Calibration_and_Removal_of_Distortion_from_Scenes_of_Structured_Environments)
///
/// The projection parameters are:
/// @f$ f_x, f_y, c_x, c_y @f$
///
/// The distortion parameters are:
/// @f$ \omega @f$
///
/// The distance between an image point and the principal point is usually
/// roughly proportional to the angle between the corresponding 3D point,
/// the optical center, and the optical axis (Fig. 1), so that the angular
/// resolution is roughly proportional to the image resolution along an image
/// radius.
///
/// @note This camera model only support FOV smaller than 180°.
class Fov : public CameraModel<Fov, 4, 1> {
 public:
  using Base = CameraModel<Fov, 4, 1>;
  using Base::kNumDistortionParams;
  using Base::kNumProjectionParams;

  using Interface = CameraModelInterface;
  using Interface::Vector2;
  using Interface::Vector3;

  static_assert(
      kNumProjectionParams == 4,
      "Fov camera model only supports 4 projection parameters");

  template <typename... Args>
  explicit Fov(Args&&... args) : Base(std::forward<Args>(args)...) {}

 public:
  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project2().
  template <typename Scalar>
  static bool project2Impl(
      const Scalar* point, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* pixel) {
    using std::abs;
    using std::atan;
    using std::sqrt;
    using std::tan;
    static const Scalar kNum_1 = Scalar(1.);
    static const Scalar kNum_2 = Scalar(2.);
    static const Scalar kNum_3 = Scalar(3.);
    static const Scalar kNum_12 = Scalar(12.);
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    static const Scalar kEps2 = kEps * kEps;

    Scalar x = point[0], y = point[1];
    Scalar x2 = x * x;
    Scalar y2 = y * y;
    Scalar r2 = x2 + y2;

    Scalar ru;  //  = sqrt(r2);
    if (r2 < kEps2) {
      // We need to regularize the sqrt() function near 0 to avoid infinite
      // derivatives, which can cause numerical issues in optimization when
      // ceres' auto-differentiation is used.

      // ru = (r2 / kEps2) * kEps;
      ru = r2 / kEps;
    } else {
      ru = sqrt(r2);
    }

    const Scalar& w = distortion_params[0];

    Scalar rd_on_ru;
    if (abs(w) < kEps) {
      rd_on_ru = kNum_1 + (w * w) * (kNum_1 / kNum_12 - r2 / kNum_3);
    } else {
      const Scalar tmp = kNum_2 * tan(w / kNum_2);
      if (ru < kEps) {
        // const Scalar ru_tmp = ru * tmp;
        // rd_on_ru = (tmp / w) * (kNum_1 - ru_tmp * ru_tmp / kNum_3);
        rd_on_ru = tmp / w;
      } else {
        rd_on_ru = atan(ru * tmp) / (ru * w);
        // Note
        //      tan(a) ≈ a + (a^3)/3 = a (1 + (a^2)/3)
        //     atan(a) ≈ a - (a^3)/3 = a (1 - (a^2)/3)
        // when a is small.
        // So, if w is small, we have
        //   tmp = 2 tan(w / 2) ≈ w * (1 + (w^2)/12)
        //   ru * tmp ≈ (w * ru) * (1 + (w^2)/12)
        //   atan(ru * tmp) ≈ (ru * tmp) * (1 - (ru^2 * tmp^2)/3)
        //
        //   atan(ru * tmp) / (ru * w)
        //        ≈ (1 + (w^2)/12) * (1 - (ru^2 * tmp^2)/3)
        //        ≈ 1 + (w^2)/12 - (ru^2 * tmp^2)/3
        //        ≈ 1 + (w^2)/12 - (ru^2 * w^2)/3
        //        ≈ 1 + (w^2) * (1/12 - (ru^2)/3)
      }
    }

    Scalar xd = rd_on_ru * x;
    Scalar yd = rd_on_ru * y;

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
    using std::sqrt;
    using std::tan;
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

    Scalar x2 = x * x;
    Scalar y2 = y * y;
    Scalar r2 = x2 + y2;

    Scalar rd = sqrt(r2);
    Scalar ru_on_rd;
    const Scalar& w = distortion_params[0];

    if (abs(w) < kEps) {
      ru_on_rd = kNum_1;
    } else {
      const Scalar tmp = kNum_2 * tan(w / kNum_2);
      if (rd < kEps) {
        ru_on_rd = w / tmp;
      } else {
        ru_on_rd = tan(w * rd) / (rd * tmp);
      }
    }

    point[0] = x * ru_on_rd;
    point[1] = y * ru_on_rd;
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
    using std::abs;
    using std::atan;
    using std::cos;
    using std::sqrt;
    using std::tan;
    static const double kEps = Eigen::NumTraits<double>::epsilon();
    static const double kEps2 = kEps * kEps;

    const double x = point[0] / point[2];
    const double y = point[1] / point[2];
    double x2 = x * x;
    double y2 = y * y;
    double xy = x * y;
    double r2 = x2 + y2;

    double ru;  //  = sqrt(r2);
    if (r2 < kEps2) {
      // ru = (r2 / kEps2) * kEps;
      ru = r2 / kEps;
    } else {
      ru = sqrt(r2);
    }

    const double& w = dist_params[0];
    const double tan_wby2 = tan(w / 2.);

    double rd_on_ru;
    if (abs(w) < kEps) {
      rd_on_ru = 1. + (w * w) * (1. / 12. - r2 / 3.);
    } else {
      const double tmp = 2. * tan_wby2;
      if (ru < kEps) {
        // const double ru_tmp = ru * tmp;
        // rd_on_ru = (tmp / w) * (1. - ru_tmp * ru_tmp / 3.);
        rd_on_ru = tmp / w;
      } else {
        rd_on_ru = atan(ru * tmp) / (ru * w);
      }
    }

    double xd = rd_on_ru * x;
    double yd = rd_on_ru * y;

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

    double s, u;
    if (jacobian_wrt_dist_params || jacobian_wrt_point) {
      s = 2. * tan_wby2 * ru;
      u = atan(s);
    }

    if (jacobian_wrt_dist_params) {
      double prd_on_ru_pw;
      if (abs(w) < kEps) {
        // rd_on_ru = 1. + (w * w) * (1. / 12.  - r2 / 3.);
        prd_on_ru_pw = 2 * w * (1. / 12. - r2 / 3.);
      } else {
        const double cos_wby2 = cos(w / 2.);
        const double w2 = w * w;
        const double A = 1. / (w * (1 + s * s) * cos_wby2 * cos_wby2);
        double B;
        if (ru < kEps) {
          B = 2. * tan_wby2 / w2;
        } else {
          B = u / (ru * w2);
        }
        prd_on_ru_pw = A - B;
      }

      jacobian_wrt_dist_params[0] = prd_on_ru_pw * x * fx;
      jacobian_wrt_dist_params[1] = prd_on_ru_pw * y * fy;
    }

    if (jacobian_wrt_point) {
      double prd_on_ru_pru;
      if (ru < kEps) {
        prd_on_ru_pru = 0.;
      } else {
        double A, B;
        if (abs(w) < kEps) {
          A = 1. / ((1. + s * s) * ru);
          B = 1. / ru;
        } else {
          A = 2. * tan_wby2 / ((1. + s * s) * w * ru);
          B = u / (r2 * w);
          // When ru->0, (A-B)->0.
        }
        prd_on_ru_pru = A - B;
      }

      double jacobian_wrt_pxpy[4];
      if (ru < kEps) {
        jacobian_wrt_pxpy[0] = rd_on_ru * fx;
        jacobian_wrt_pxpy[1] = 0.;
        jacobian_wrt_pxpy[2] = 0.;
        jacobian_wrt_pxpy[3] = rd_on_ru * fy;
      }
      {
        jacobian_wrt_pxpy[0] = (rd_on_ru + prd_on_ru_pru * x2 / ru) * fx;
        jacobian_wrt_pxpy[1] = (prd_on_ru_pru * xy / ru) * fx;
        jacobian_wrt_pxpy[2] = (prd_on_ru_pru * xy / ru) * fy;
        jacobian_wrt_pxpy[3] = (rd_on_ru + prd_on_ru_pru * y2 / ru) * fy;
      }
      double hnormalized[2] = {x, y};
      Interface::convertJacobianWrtHnormalizedToJacobianWrtPoint3(
          jacobian_wrt_pxpy, hnormalized, point[2], jacobian_wrt_point);
    }
    return true;
  }
};

}  // namespace sk4slam
