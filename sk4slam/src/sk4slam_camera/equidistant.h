#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_camera/camera_model.h"

namespace sk4slam {

/// @brief Equidistant distortion model (Kannala-Brandt8)
///
/// The projection parameters are:
/// @f$ f_x, f_y, c_x, c_y @f$
///
/// The distortion parameters are:
/// @f$ k_1, k_2, k_3, k4 @f$
///
/// The distortion model is:
///
/// @f$ x_d = r_d \cdot (x / r) @f$
///
/// @f$ y_d = r_d \cdot (y / r) @f$
///
/// where @f$ (x,y,z) @f$ is the normalized coordinate of the 3D
/// point @f$ (X,Y,Z) @f$ (in the camera coordinate system) and
///
/// @f$ r = \sqrt{x^2 + y^2} @f$
///
/// @f$ r_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4
/// \theta^8) @f$
///
/// @f$ \theta = \arccos(z) @f$
///
/// (or alternatively, @f$ \theta = \arctan(r / z) @f$ if @f$ z > 0 @f$).
///
/// Note @f$ \theta @f$ is the angle between the bearing vector @f$ (x,y,z) @f$
/// and the optical axis (i.e. the Z axis of the camera coordinate system).
///
/// @note The FOV for this camera model can even be larger than 180° (but
/// smaller than @f$ (360 - \epsilon) @f$° for some small @f$ \epsilon > 0 @f$).
class Equidistant : public CameraModel<Equidistant, 4, 4> {
 public:
  using Base = CameraModel<Equidistant, 4, 4>;
  using Base::kNumDistortionParams;
  using Base::kNumProjectionParams;

  using Interface = CameraModelInterface;
  using Interface::Vector2;
  using Interface::Vector3;

  static_assert(
      kNumProjectionParams == 4,
      "Equidistant camera model only supports 4 projection parameters");

  template <typename... Args>
  explicit Equidistant(Args&&... args) : Base(std::forward<Args>(args)...) {}

 public:
  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project2().
  template <typename Scalar>
  static bool project2Impl(
      const Scalar* point, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* pixel) {
    using std::atan;
    using std::sqrt;
    const Scalar x = point[0], y = point[1];
    const Scalar x2 = x * x;
    const Scalar y2 = y * y;
    const Scalar r2 = x2 + y2;
    const Scalar r = sqrt(r2);
    const Scalar theta = atan(r);
    return projectCommon(
        theta, x, y, r, projection_params, distortion_params, pixel);
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project3().
  template <typename Scalar>
  static bool project3Impl(
      const Scalar* point, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* pixel) {
    using std::acos;
    using std::asin;
    using std::sqrt;
    static const Scalar kNum_1 = Scalar(1.);
    static const Scalar kNum_0p5sqrt2 = Scalar(0.5 * sqrt(2.));
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    static const Scalar kEps2 = kEps * kEps;
    Eigen::Matrix<Scalar, 3, 1> point_eigen(point[0], point[1], point[2]);
    point_eigen.normalize();
    const Scalar x = point_eigen[0];
    const Scalar y = point_eigen[1];
    const Scalar z = point_eigen[2];

    const Scalar x2 = x * x;
    const Scalar y2 = y * y;
    const Scalar r2 = x2 + y2;

    // const Scalar z2 = z * z;
    // const Scalar r2 = kNum_1 - z2;  // lower precision when z is close to 1

    Scalar r;

    // clang-format off
    // static const Scalar kSqrtEps = sqrt(kEps);
    // if (r2 < kEps) {
    //   // r = (r2 / kEps) * kSqrtEps;
    //   r = r2 / kSqrtEps;
    // clang-format on
    if (r2 < kEps2) {
      // We need to regularize the sqrt() function near 0 to avoid infinite
      // derivatives, which can cause numerical issues in optimization when
      // ceres' auto-differentiation is used.

      // r = (r2 / kEps2) * kEps;
      r = r2 / kEps;
    } else {
      r = sqrt(r2);
    }
    Scalar theta;
    if (z > kNum_0p5sqrt2) {
      // LOGA("z > kNum_0p5sqrt2");
      theta = asin(r);
    } else {
      // LOGA("z <= kNum_0p5sqrt2");
      theta = acos(z);
    }
    return projectCommon(
        theta, x, y, r, projection_params, distortion_params, pixel);
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::backProject2().
  template <typename Scalar>
  static bool backProject2Impl(
      const Scalar* pixel, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* point) {
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    Eigen::Matrix<Scalar, 3, 1> point3;
    if (backProject3Impl(
            pixel, projection_params, distortion_params, point3.data())) {
      if (point3[2] > kEps) {
        point[0] = point3[0] / point3[2];
        point[1] = point3[1] / point3[2];
        return true;
      }
    }
    return false;
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::backProject3().
  template <typename Scalar>
  static bool backProject3Impl(
      const Scalar* pixel, const Scalar* projection_params,
      const Scalar* distortion_params, Scalar* point) {
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;
    using std::tan;
    static const Scalar kNum_1 = Scalar(1.);
    static const Scalar kNum_2 = Scalar(2.);
    static const Scalar kNum_3 = Scalar(3.);
    static const Scalar kNum_5 = Scalar(5.);
    static const Scalar kNum_7 = Scalar(7.);
    static const Scalar kNum_9 = Scalar(9.);
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    const Scalar x = pixel[0], y = pixel[1];
    const Scalar& fx = projection_params[0];
    const Scalar& fy = projection_params[1];
    const Scalar& cx = projection_params[2];
    const Scalar& cy = projection_params[3];
    Scalar xd = (x - cx) / fx;
    Scalar yd = (y - cy) / fy;
    Scalar rd = sqrt(xd * xd + yd * yd);
    if (rd < kEps) {
      Scalar norm = sqrt(rd * rd + kNum_1);
      point[0] = xd / norm;
      point[1] = yd / norm;
      point[2] = kNum_1 / norm;
      return true;
    }

    Scalar theta = rd;
    // Loop until convergence, empirically 10 iterations is enough.
    for (int i = 0; i < 10; ++i) {
      Scalar theta2 = theta * theta;
      Scalar theta4 = theta2 * theta2;
      Scalar theta6 = theta4 * theta2;
      Scalar theta8 = theta6 * theta2;
      Scalar residual = theta * (kNum_1 + distortion_params[0] * theta2 +
                                 distortion_params[1] * theta4 +
                                 distortion_params[2] * theta6 +
                                 distortion_params[3] * theta8) -
                        rd;
      // if (abs(residual) < kEps) {
      //   break;
      // }
      Scalar derivative = kNum_1 + kNum_3 * distortion_params[0] * theta2 +
                          kNum_5 * distortion_params[1] * theta4 +
                          kNum_7 * distortion_params[2] * theta6 +
                          kNum_9 * distortion_params[3] * theta8;
      Scalar delta = residual / derivative;
      if (abs(delta) < kEps) {
        break;
      }
      theta -= delta;
    }

    Scalar c = cos(theta);
    Scalar s = sin(theta);
    point[0] = s * xd / rd;
    point[1] = s * yd / rd;
    point[2] = c;
    return true;
  }

  /// @brief       Implementation of the project function template @ref
  /// CameraModel::project3AndComputeJacobians().
  static bool project3AndComputeJacobiansImpl(
      const double* point, const double* proj_params, const double* dist_params,
      double* pixel, double* jacobian_wrt_point,
      double* jacobian_wrt_proj_params, double* jacobian_wrt_dist_params) {
    using std::acos;
    using std::asin;
    using std::sqrt;
    static const double kEps = Eigen::NumTraits<double>::epsilon();
    static const double kEps2 = kEps * kEps;
    static const double kNum_0p5sqrt2 = 0.5 * sqrt(2.);
    const Eigen::Vector3d point_eigen(point[0], point[1], point[2]);
    const double norm = point_eigen.norm();
    const double x = point_eigen[0] / norm;
    const double y = point_eigen[1] / norm;
    const double z = point_eigen[2] / norm;

    // const double x2 = x * x;
    // const double y2 = y * y;
    // const double r2 = x2 + y2;  // = 1 - z * z

    const double z2 = z * z;
    const double r2 = 1. - z2;  // lower precision when z is close to 1

    double r;  // Note r = sin(theta)
    if (r2 < kEps2) {
      r = r2 / kEps;
    } else {
      r = sqrt(r2);
    }

    double tmp_theta = acos(z > 1. ? 1. : (z < -1. ? -1. : z));
    if (z > kNum_0p5sqrt2) {
      // LOGA("z > kNum_0p5sqrt2");
      tmp_theta = asin(r);
    } else {
      // LOGA("z <= kNum_0p5sqrt2");
      tmp_theta = acos(z);
    }

    if (r < kEps) {
      // tmp_theta ≈ 0 or tmp_theta ≈ π
      if (tmp_theta > 1.) {
        // tmp_theta ≈ π
        // We can't project points close to the opposite of the optical axis.
        // (FOV < 360° - ε)
        return false;
      } else {
        // tmp_theta ≈ 0
        tmp_theta = r;
      }
    }

    const double theta = tmp_theta;
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta4 * theta2;
    const double theta8 = theta6 * theta2;
    const double rd =
        theta * (1. + dist_params[0] * theta2 + dist_params[1] * theta4 +
                 dist_params[2] * theta6 + dist_params[3] * theta8);
    double tmp_rd_on_r;
    if (r < kEps) {
      // r = sin (theta) ≈ theta - 1/6 * theta3 = theta * (1 - 1/6 * theta2)
      // rd ≈ theta +  k1 * theta3 = theta * (1 + k1 * theta2)
      // rd / r ≈ (1 + k1 * theta2) / (1 - 1/6 * theta2)
      //        ≈ (1 + k1 * theta2) * (1 + 1/6 * theta2)
      //        ≈  1 + (k1 + 1/6) * theta2

      // tmp_rd_on_r = 1. + dist_params[0] * theta2;
      tmp_rd_on_r = 1. + (dist_params[0] + (1. / 6.)) * theta2;
    } else {
      tmp_rd_on_r = rd / r;
    }
    const double rd_on_r = tmp_rd_on_r;
    const double xd = rd_on_r * x;
    const double yd = rd_on_r * y;

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

    if (jacobian_wrt_dist_params) {
      if (r < kEps) {
        jacobian_wrt_dist_params[0] = 0.0;
        jacobian_wrt_dist_params[1] = 0.0;
        jacobian_wrt_dist_params[2] = 0.0;
        jacobian_wrt_dist_params[3] = 0.0;

        jacobian_wrt_dist_params[4] = 0.0;
        jacobian_wrt_dist_params[5] = 0.0;
        jacobian_wrt_dist_params[6] = 0.0;
        jacobian_wrt_dist_params[7] = 0.0;
      } else {
        double ppixel0_prd = fx * x / r;
        double ppixel1_prd = fy * y / r;
        const double theta3 = theta * theta2;
        const double theta5 = theta3 * theta2;
        const double theta7 = theta5 * theta2;
        const double theta9 = theta7 * theta2;

        jacobian_wrt_dist_params[0] = theta3 * ppixel0_prd;
        jacobian_wrt_dist_params[1] = theta5 * ppixel0_prd;
        jacobian_wrt_dist_params[2] = theta7 * ppixel0_prd;
        jacobian_wrt_dist_params[3] = theta9 * ppixel0_prd;

        jacobian_wrt_dist_params[4] = theta3 * ppixel1_prd;
        jacobian_wrt_dist_params[5] = theta5 * ppixel1_prd;
        jacobian_wrt_dist_params[6] = theta7 * ppixel1_prd;
        jacobian_wrt_dist_params[7] = theta9 * ppixel1_prd;
      }
    }

    if (jacobian_wrt_point) {
      JacobianMatrix<double, 2, 3> jacobian_wrt_normalized;
      double* tmpd = jacobian_wrt_normalized.data();

      double prd_on_r_pz;  // D(rd_on_r) / D(z)
      if (r < kEps) {
        // LOGA("r < kEps");

        // clang-format off
        // z = cos(theta) ≈ 1 - theta^2 / 2      (theta^2 ≈ 2 - 2 * z);
        // rd_on_r ≈ 1 + (k1 + 1/6) * theta2
        //         = 1 + (k1 + 1/6) * （2 - 2 * z）
        //         = 1 + 2 * (k1 + 1/6) - 2 * (k1 + 1/6) * z
        // prd_on_r_pz
        //         = - 2 * (k1 + 1/6)
        // clang-format on

        // prd_on_r_pz = -2 * dist_params[0];
        prd_on_r_pz = -2 * (dist_params[0] + 1. / 6.);
      } else {
        // LOGA("r >= kEps");

        // rd_on_r = rd / r;

        // clang-format off
        // rd = theta * (1. + dist_params[0] * theta2 +
        //                     dist_params[1] * theta4 +
        //                     dist_params[2] * theta6 +
        //                     dist_params[3] * theta8);
        // clang-format on
        double prd_ptheta =
            1. + 3. * dist_params[0] * theta2 + 5. * dist_params[1] * theta4 +
            7. * dist_params[2] * theta6 + 9. * dist_params[3] * theta8;

        // clang-format off
        // theta = acos(z), so
        // ptheta_pz = -1. / sqrt(1. - z * z) = -1. / r
        // prd_pz = prd_ptheta * ptheta_pz;  // = - prd_ptheta / r
        // pr_pz = - z / r;  // since r = sqrt(1-z*z)
        // prd_on_r_pz = (prd_pz * r - rd * pr_pz) / r2
        //             = (- prd_ptheta + rd / r * z) / r2
        // clang-format on
        prd_on_r_pz = (rd_on_r * z - prd_ptheta) / r2;

        // clang-format off
        // when rd is small (r and theta are also small),
        //
        // rd_on_r ≈ 1 + (k1 + 1/6) * theta2
        //
        // z = cos(theta) ≈ 1 - theta^2 / 2      (theta^2 ≈ 2 - 2 * z)
        //
        // rd_on_r * z ≈ 1 + (k1 + 1/6 - 1/2) * theta^2
        //             = 1 + (k1 - 1/3) * theta^2
        //
        // prd_ptheta ≈ 1 + 3 * k1 * theta^2
        //
        // (rd_on_r * z - prd_ptheta) = - (2 * k1 + 1/3) * theta^2
        //
        // prd_on_r_pz ≈ -(2 * k1 + 1/3) * (theta^2 / r2)
        //             ≈ -(2 * k1 + 1/3)
        //             = - 2 * (k1 + 1/6)
        //
        // this coincides with the result from the above `if (r < kEps)` block.
        //
        // clang-format on
      }

      // xd = rd_on_r * x;
      // yd = rd_on_r * y;
      // pixel[0] = xd * fx + cx;
      // pixel[1] = yd * fy + cy;
      tmpd[0] = fx * rd_on_r;
      tmpd[1] = 0.;
      tmpd[2] = fx * x * prd_on_r_pz;
      tmpd[3] = 0.;
      tmpd[4] = fy * rd_on_r;
      tmpd[5] = fy * y * prd_on_r_pz;

      // LOGA(
      //     "jacobian_wrt_normalized: \n%s",
      //     toStr(jacobian_wrt_normalized).c_str());

      double normalized[3] = {x, y, z};
      Interface::convertJacobianWrtNormalized3ToJacobianWrtPoint3(
          tmpd, normalized, norm, jacobian_wrt_point);
    }
    return true;
  }

 private:
  template <typename Scalar>
  static bool projectCommon(
      Scalar tmp_theta, const Scalar& x, const Scalar& y, const Scalar& r,
      const Scalar* proj_params, const Scalar* dist_params, Scalar* pixel) {
    using std::sqrt;
    static const Scalar kNum_1 = Scalar(1.);
    static const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();
    if (r < kEps) {
      // tmp_theta ≈ 0 or tmp_theta ≈ π
      if (tmp_theta > kNum_1) {
        // tmp_theta ≈ π
        // We can't project points close to the opposite of the optical axis.
        // (FOV < 360° - ε)
        return false;
      } else {
        // tmp_theta ≈ 0
        tmp_theta = r;
      }
    }

    const Scalar theta = tmp_theta;
    const Scalar theta2 = theta * theta;
    const Scalar theta4 = theta2 * theta2;
    const Scalar theta6 = theta4 * theta2;
    const Scalar theta8 = theta6 * theta2;
    const Scalar rd =
        theta * (kNum_1 + dist_params[0] * theta2 + dist_params[1] * theta4 +
                 dist_params[2] * theta6 + dist_params[3] * theta8);
    Scalar tmp_rd_on_r;
    if (r < kEps) {
      // r = sin (theta) ≈ theta - 1/6 * theta3 = theta * (1 - 1/6 * theta2)
      // rd ≈ theta +  k1 * theta3 = theta * (1 + k1 * theta2)
      // rd / r ≈ (1 + k1 * theta2) / (1 - 1/6 * theta2)
      //        ≈ (1 + k1 * theta2) * (1 + 1/6 * theta2)
      //        ≈  1 + (k1 + 1/6) * theta2

      // tmp_rd_on_r = kNum_1 + dist_params[0] * theta2;
      tmp_rd_on_r = kNum_1 + (dist_params[0] + Scalar(1. / 6.)) * theta2;
    } else {
      tmp_rd_on_r = rd / r;
    }
    const Scalar rd_on_r = tmp_rd_on_r;
    const Scalar xd = rd_on_r * x;
    const Scalar yd = rd_on_r * y;

    const Scalar& fx = proj_params[0];
    const Scalar& fy = proj_params[1];
    const Scalar& cx = proj_params[2];
    const Scalar& cy = proj_params[3];
    pixel[0] = xd * fx + cx;
    pixel[1] = yd * fy + cy;
    return true;
  }
};

}  // namespace sk4slam
