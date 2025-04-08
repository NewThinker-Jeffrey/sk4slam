#pragma once

#include <Eigen/Core>
#include <memory>
#include <opencv2/core/core.hpp>
#include <vector>

#include "sk4slam_basic/template_helper.h"
#include "sk4slam_camera/auto_jacobian_impl.h"

namespace sk4slam {

/// @brief The interface for camera models.
///
/// @note The image coordinate is in the convention of OpenCV. i.e., the
///       origin is at the center of the first pixel (the upper-left pixel).
///       This convention also coincides with Kalibr.
///       See the discussion in
///       https://github.com/ethz-asl/kalibr/issues/115#issuecomment-315114990
class CameraModelInterface {
 public:
  template <typename Scalar>
  using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vector2d = Vector2<double>;
  using Vector2f = Vector2<float>;
  using Vector2i = Vector2<int>;

  template <typename Scalar>
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vector3d = Vector3<double>;
  using Vector3f = Vector3<float>;

  /// @brief The Jacobian matrix type.
  /// @note The Jacobians are all stored in row-major order, to be compatible
  /// with
  ///       ceres-solver.
  template <
      typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
  using JacobianMatrix = std::conditional_t<
      _cols == 1,
      Eigen::Matrix<Scalar, _rows, 1>,  // Use Eigen::RowMajor when cols == 1
                                        // will cause a compile error.
      Eigen::Matrix<Scalar, _rows, _cols, Eigen::RowMajor>>;

  template <
      typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
  using JacobianMap = Eigen::Map<JacobianMatrix<Scalar, _rows, _cols>>;

  template <
      typename Scalar, int _rows = Eigen::Dynamic, int _cols = Eigen::Dynamic>
  using ConstJacobianMap =
      Eigen::Map<const JacobianMatrix<Scalar, _rows, _cols>>;

 public:
  virtual ~CameraModelInterface() {}

  virtual int numProjectionParams() const = 0;
  virtual int numDistortionParams() const = 0;
  virtual double* projectionParams() = 0;

  virtual const double* projectionParams() const = 0;
  virtual double* distortionParams() = 0;
  virtual const double* distortionParams() const = 0;

  virtual int numIntrinsics() const = 0;
  virtual double* intrinsics() = 0;
  virtual const double* intrinsics() const = 0;

  virtual double fx() const = 0;
  virtual double fy() const = 0;
  virtual double cx() const = 0;
  virtual double cy() const = 0;

  virtual std::unique_ptr<CameraModelInterface> clone() const = 0;

  /// @brief Project a 3D point to a 2D pixel.
  ///
  /// @param point    The input 3D point in camera coordinate
  /// @param pixel    The output 2D pixel in image coordinate. A `2 x 1` vector.
  /// @return  Return true if the projection is successful, false otherwise.
  ///
  virtual bool project3(const double* point, double* pixel) const = 0;
  virtual bool project3(const float* point, float* pixel) const = 0;

  /// @brief Back-project a 2D pixel to a 3D point (on the unit sphere).
  ///
  /// @param pixel    The input 2D pixel in image coordinate
  /// @param point    The output 3D point in camera coordinate
  /// @return  Return true if the back-projection is successful, false
  /// otherwise.
  ///
  virtual bool backProject3(const double* pixel, double* point) const = 0;
  virtual bool backProject3(const float* pixel, float* point) const = 0;

  /// @brief  Project a 2D point on the normalized image plane (z = 1) to a 2D
  /// pixel.
  /// @param point   The input 2D point on the normalized image plane.
  /// @param pixel   The output 2D pixel in image coordinate.
  /// @return  Return true if the projection is successful, false otherwise.
  virtual bool project2(const double* point, double* pixel) const = 0;
  virtual bool project2(const float* point, float* pixel) const = 0;

  /// @brief  Back-project a 2D pixel to a 2D point on the normalized image
  /// plane (z = 1).
  /// @param pixel   The input 2D pixel in image coordinate.
  /// @param point   The output 2D point on the normalized image plane.
  /// @return  Return true if the back-projection is successful, false
  /// otherwise.
  virtual bool backProject2(const double* pixel, double* point) const = 0;
  virtual bool backProject2(const float* pixel, float* point) const = 0;

  /// @brief Project a 3D point to a 2D pixel, and optionally compute the
  /// Jacobian @f$\frac{\partial \text{pixel}}{\partial \text{point}}@f$.
  ///
  /// @param point    The input 3D point in camera coordinate
  /// @param pixel    The output 2D pixel in image coordinate. A `2 x 1` vector.
  ///                 It's computed only if it's not nullptr.
  /// @param jacobian_wrt_point
  ///                 Jacobian of the pixel w.r.t. the 3D point. A `2 x 3`
  ///                 matrix. It's computed only if it's not nullptr.
  /// @return  Return true if the projection is successful, false otherwise.
  ///
  /// @note
  ///   The Jacobians are all stored in row-major order, to be compatible with
  ///   ceres-solver.
  ///
  virtual bool project3AndComputeJacobian(
      const double* point, double* pixel, double* jacobian_wrt_point) const = 0;

  virtual bool project3AndComputeJacobiansWithExternalParameters(
      const double* point, const double* proj_params, const double* dist_params,
      double* pixel, double* jacobian_wrt_point,
      double* jacobian_wrt_proj_params,
      double* jacobian_wrt_dist_params) const = 0;

  virtual bool project3AndComputeJacobiansWithExternalParameters(
      const double* point, const double* intrinsics, double* pixel,
      double* jacobian_wrt_point, double* jacobian_wrt_intrinsics) const = 0;

 public:
  Eigen::VectorXd projectionParamsVector() const {
    return Eigen::Map<const Eigen::VectorXd>(
        projectionParams(), numProjectionParams());
  }
  Eigen::VectorXd distortionParamsVector() const {
    return Eigen::Map<const Eigen::VectorXd>(
        distortionParams(), numDistortionParams());
  }
  Eigen::VectorXd intrinsicsVector() const {
    return Eigen::Map<const Eigen::VectorXd>(intrinsics(), numIntrinsics());
  }
  void setIntrinsicsVector(const Eigen::VectorXd& intrinsics) {
    ASSERT(intrinsics.size() == numIntrinsics());
    Eigen::Map<Eigen::VectorXd> internal(this->intrinsics(), numIntrinsics());
    internal = intrinsics;
  }

  /// @brief Project a 3D point to a 2D pixel.
  ///
  /// @tparam Point3  The type of the input 3D point. It can be either
  /// `cv::Point3_<Scalar>` or `Eigen::Vector3<Scalar>` or `Scalar*`,
  /// where `Scalar` can be `double` or `float` or integers.
  //
  /// @param point    The input 3D point in camera coordinate.
  /// @return         Return a pair of bool and the 2D pixel in image
  /// coordinate.
  ///                 The bool indicates whether the projection is successful.
  ///                 If the projection is successful, the 2D pixel is returned.
  ///                 Otherwise, the 2D pixel is not defined.
  /// @note  The return type is `std::pair<bool, Pixel>`. When @c Point3 is
  ///         `cv::Point3_<Scalar>`, the type `Pixel` is
  ///         `cv::Point_<OutputScalar>`.
  ///        Otherwise, `Pixel` is `Eigen::Vector3<OutputScalar>`. The
  ///        `OutputScalar` is `double` only when the input `Scalar` is
  ///        `double`, otherwise it's `float`.
  template <typename Point3>
  decltype(auto) project3(const Point3& point) const {
    using Scalar = typename PointTypeTraits<Point3>::Scalar;
    if constexpr (PointTypeTraits<Point3>::kIsOpencvPoint) {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        cv::Point_<Scalar> pixel;
        bool success = project3(&point.x, &pixel.x);
        return std::make_pair(success, pixel);
      } else {
        cv::Point3f pointf(point.x, point.y, point.z);
        cv::Point2f pixelf;
        bool success = project3(&pointf.x, &pixelf.x);
        return std::make_pair(success, pixelf);
      }
    } else {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        Vector2<Scalar> pixel;
        bool success = project3(&point[0], pixel.data());
        return std::make_pair(success, pixel);
      } else {
        Vector3f pointf(point[0], point[1], point[2]);
        Vector2f pixelf;
        bool success = project3(pointf.data(), pixelf.data());
        return std::make_pair(success, pixelf);
      }
    }
  }

  /// @brief  Back-project a 2D pixel to a 3D point (on the unit sphere).
  /// @tparam Pixel      The type of the input pixel. It can be either
  /// `cv::Point_<Scalar>` or `Eigen::Vector2<Scalar>` or `Scalar*`.
  /// @param pixel      The input 2D pixel in image coordinate.
  /// @return   Return a pair of bool and the 3D point on the unit sphere.
  ///           The bool indicates whether the back-projection is successful.
  ///           If the back-projection is successful, the 3D point is returned.
  ///           Otherwise, the 3D point is not defined.
  /// @note  The return type is `std::pair<bool, Point3>`. When @c Pixel is
  ///         `cv::Point_<Scalar>`, the type `Point3` is
  ///         `cv::Point3_<OutputScalar>`.
  ///        Otherwise, `Point3` is `Eigen::Vector3<OutputScalar>`. The
  ///        `OutputScalar` is `double` only when the input `Scalar` is
  ///        `double`, otherwise it's `float`.
  template <typename Pixel>
  decltype(auto) backProject3(const Pixel& pixel) const {
    using Scalar = typename PointTypeTraits<Pixel>::Scalar;
    if constexpr (PointTypeTraits<Pixel>::kIsOpencvPoint) {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        cv::Point3_<Scalar> point;
        bool success = backProject3(&pixel.x, &point.x);
        return std::make_pair(success, point);
      } else {
        cv::Point2f pixelf(pixel.x, pixel.y);
        cv::Point3f pointf;
        bool success = backProject3(&pixelf.x, &pointf.x);
        return std::make_pair(success, pointf);
      }
    } else {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        Vector3<Scalar> point;
        bool success = backProject3(&pixel[0], point.data());
        return std::make_pair(success, point);
      } else {
        Vector2f pixelf(pixel[0], pixel[1]);
        Vector3f pointf;
        bool success = backProject3(pixelf.data(), pointf.data());
        return std::make_pair(success, pointf);
      }
    }
  }

  /// @brief    Project a 2D point on the normalized image plane (z = 1) to a 2D
  /// pixel.
  /// @tparam Point2  The type of the input point. It can be either
  /// `cv::Point_<Scalar>` or `Eigen::Vector2<Scalar>` or `Scalar*`.
  /// @param point   The input 2D point on the normalized image plane.
  /// @return  Return a pair of bool and the 2D pixel in image coordinate.
  ///           The bool indicates whether the projection is successful.
  ///           If the projection is successful, the 2D pixel is returned.
  ///           Otherwise, the 2D pixel is not defined.
  /// @note  The return type is `std::pair<bool, Pixel>`. When @c Point2 is
  ///         `cv::Point_<Scalar>`, the type `Pixel` is
  ///         `cv::Point_<OutputScalar>`.
  ///        Otherwise, `Pixel` is `Eigen::Vector2<OutputScalar>`. The
  ///        `OutputScalar` is `double` only when the input `Scalar` is
  ///        `double`, otherwise it's `float`.
  template <typename Point2>
  decltype(auto) project2(const Point2& point) const {
    using Scalar = typename PointTypeTraits<Point2>::Scalar;
    if constexpr (PointTypeTraits<Point2>::kIsOpencvPoint) {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        cv::Point_<Scalar> pixel;
        bool success = project2(&point.x, &pixel.x);
        return std::make_pair(success, pixel);
      } else {
        cv::Point2f pointf(point.x, point.y);
        cv::Point2f pixelf;
        bool success = project2(&pointf.x, &pixelf.x);
        return std::make_pair(success, pixelf);
      }
    } else {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        Vector2<Scalar> pixel;
        bool success = project2(&point[0], pixel.data());
        return std::make_pair(success, pixel);
      } else {
        Vector2f pointf(point[0], point[1]);
        Vector2f pixelf;
        bool success = project2(pointf.data(), pixelf.data());
        return std::make_pair(success, pixelf);
      }
    }
  }

  /// @brief    Back-project a 2D pixel to a 2D point on the normalized image
  /// plane (z = 1).
  /// @tparam Pixel  The type of the input pixel. It can be either
  /// `cv::Point_<Scalar>` or `Eigen::Vector2<Scalar>` or `Scalar*`.
  /// @param pixel   The input 2D pixel in image coordinate.
  /// @return  Return a pair of bool and the 2D point on the normalized image
  /// plane.
  ///           The bool indicates whether the back-projection is successful.
  ///           If the back-projection is successful, the 2D point is returned.
  ///           Otherwise, the 2D point is not defined.
  /// @note  The return type is `std::pair<bool, Point>`. When @c Pixel is
  ///         `cv::Point_<Scalar>`, the type `Point` is
  ///         `cv::Point_<OutputScalar>`.
  ///        Otherwise, `Point` is `Eigen::Vector2<OutputScalar>`. The
  ///        `OutputScalar` is `double` only when the input `Scalar` is
  ///        `double`, otherwise it's `float`.
  template <typename Pixel>
  decltype(auto) backProject2(const Pixel& pixel) const {
    using Scalar = typename PointTypeTraits<Pixel>::Scalar;
    if constexpr (PointTypeTraits<Pixel>::kIsOpencvPoint) {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        cv::Point_<Scalar> point;
        bool success = backProject2(&pixel.x, &point.x);
        return std::make_pair(success, point);
      } else {
        cv::Point2f pixelf(pixel.x, pixel.y);
        cv::Point2f pointf;
        bool success = backProject2(&pixelf.x, &pointf.x);
        return std::make_pair(success, pointf);
      }
    } else {
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        Vector2<Scalar> point;
        bool success = backProject2(&pixel[0], point.data());
        return std::make_pair(success, point);
      } else {
        Vector2f pixelf(pixel[0], pixel[1]);
        Vector2f pointf;
        bool success = backProject2(pixelf.data(), pointf.data());
        return std::make_pair(success, pointf);
      }
    }
  }

  /// @brief  Back-project a 2D pixel to a 2D point on the normalized image
  /// plane (z = 1), and optionally compute the Jacobian @f$\frac{\partial
  /// \text{point}}
  /// {\partial \text{pixel}}@f$.
  /// @param pixel   The input 2D pixel in image coordinate.
  /// @param point   The output 2D point on the normalized image plane (z = 1).
  /// @param jacobian_wrt_pixel
  ///                Jacobian of the 2D point w.r.t. the 2D pixel. A `2 x 2`
  ///                matrix. It's computed only if it's not nullptr.
  /// @return  Return true if the back-projection is successful, false
  /// otherwise.
  virtual bool backProject2AndComputeJacobian(
      const double* pixel, double* point, double* jacobian_wrt_pixel) const {
    Eigen::Vector3d point3;
    bool back_proj_success = backProject2(pixel, point3.data());
    if (!back_proj_success) {
      return false;
    }
    if (point) {
      point[0] = point3[0];
      point[1] = point3[1];
    }
    if (jacobian_wrt_pixel) {
      point3[2] = 1.0;
      JacobianMatrix<double, 2, 3> proj_jacobian;
      if (!project3AndComputeJacobian(
              point3.data(), nullptr, proj_jacobian.data())) {
        return false;
      }
      JacobianMap<double, 2, 2> backproj_jacobian(jacobian_wrt_pixel);
      backproj_jacobian = proj_jacobian.leftCols(2).inverse();
    }
    return true;
  }

 public:
  struct LUT;
  enum class LutFunction : uint64_t {
    PROJECT3 = (1ul << 0),
    PROJECT2 = (1ul << 1),
    BACKPROJECT3 = (1ul << 2),
    BACKPROJECT2 = (1ul << 3),
    PROJECT3_JACOBIAN = (1ul << 4),
    BACKPROJECT2_JACOBIAN = (1ul << 5),

    ALL =
        (PROJECT3 | PROJECT2 | BACKPROJECT3 | BACKPROJECT2 | PROJECT3_JACOBIAN |
         BACKPROJECT2_JACOBIAN)
  };
  std::unique_ptr<LUT> makeLUT(
      int width, int height, bool interpolate_pixel = false,
      bool interpolate_jac = false,
      LutFunction lut_function =
          LutFunction::ALL,  // less functions, less memory needed.
      const float project2_max_fov = 120.0f / 180.0f * M_PI,
      const float project3_max_fov = 240.0f / 180.0f * M_PI) const {
    return std::make_unique<LUT>(
        this, width, height, interpolate_pixel, interpolate_jac, lut_function,
        project2_max_fov, project3_max_fov);
  }

 public:
  template <typename Point>
  struct _IsOpenCVPoint {
    DEFINE_HAS_MEMBER_VARIABLE(x)
    static const bool value = HasMemberVariable_x<Point>;
  };
  template <typename Point>
  static constexpr bool IsOpenCVPoint = _IsOpenCVPoint<Point>::value;

  template <typename Point>
  struct _IsEigenPoint {
    DEFINE_HAS_MEMBER_FUNCTION(x)
    static const bool value = HasMemberFunction_x<Point>;
  };
  template <typename Point>
  static constexpr bool IsEigenPoint = _IsEigenPoint<Point>::value;

  template <
      typename Point, bool _is_opencv_point = false,
      bool _is_eigen_point = false>
  struct PointTypeTraitsImpl {
    static_assert(std::is_pointer_v<Point>, "Point must be a pointer type");
    static const bool kIsOpencvPoint = false;
    using Scalar = std::remove_cv_t<std::remove_pointer_t<Point>>;
  };
  template <typename Point>
  struct PointTypeTraitsImpl<Point, true, false> {
    static const bool kIsOpencvPoint = true;
    using Scalar = typename Point::value_type;
  };
  template <typename Point>
  struct PointTypeTraitsImpl<Point, false, true> {
    static const bool kIsOpencvPoint = false;
    using Scalar = typename Point::Scalar;
  };
  template <typename Point>
  using PointTypeTraits =
      PointTypeTraitsImpl<Point, IsOpenCVPoint<Point>, IsEigenPoint<Point>>;

  /// @brief Get the data pointer of a point, where the Point tyoe can be
  /// either OpenCV Point or Eigen Vector or even a raw pointer (e.g. double*).
  template <typename Point>
  static inline decltype(auto) getPointData(Point&& point) {
    if constexpr (PointTypeTraits<
                      std::remove_reference_t<Point>>::kIsOpencvPoint) {
      return &(point.x);
    } else {
      return &(point[0]);
    }
  }

 protected:
  /// @brief A helper function for project3AndComputeJacobian().
  ///
  /// Usually, the Jacobian w.r.t. the normalized 3D point (on @f$ S^2 @f$) is
  /// easier to compute than that w.r.t. the original 3D point.
  /// This function can convert the Jacobian w.r.t. the normalized 3D point to
  /// that w.r.t. the original 3D point.
  ///
  /// @param jacobian_wrt_normalized
  ///                Jacobian of the 2D pixel w.r.t. the normalized 3D point. A
  ///                `2 x 3` matrix.
  /// @param normalized
  ///                The normalized 3D point.
  /// @param norm    The norm of the original 3D point.
  /// @param jacobian_wrt_point3
  ///                The output Jacobian of the 2D pixel w.r.t. the original 3D
  ///                point. A `2 x 3` matrix.
  ///
  /// @details Note that the normalized 3D point is overparameterized: it has
  ///          3 parameters but only 2 degrees of freedom (it's on @f$ S^2 @f$).
  ///
  ///          Therefore, we need first to convert the normalized 3D point @f$
  ///          P_N\in S^2 @f$ to some minimal parameterization @f$ \xi \in
  ///          \mathbb{R}^2 @f$ and convert the Jacobian w.r.t. @f$ P_N @f$ to
  ///          that w.r.t. @f$ \xi @f$:
  ///
  ///             @f$ \frac{\partial p}{\partial \xi} = \frac{\partial
  ///             p}{\partial P_N} \frac{\partial P_N}{\partial \xi} @f$
  ///
  ///          Then, we can convert the Jacobian w.r.t. @f$ \xi @f$ to that
  ///          w.r.t. the original 3D point $@f$ P\in \mathbb{R}^3 @f$:
  ///
  ///             @f$ \frac{\partial p}{\partial P} = \frac{\partial p}{\partial
  ///             \xi} \frac{\partial \xi}{\partial P} @f$
  ///
  ///          Now the remaining problem is how to choose the minimal
  ///          parameterization. We can choose two unit vectors
  ///          @f$ \mathbf{\alpha}, \mathbf{\beta} \in \mathbb{R}^3 @f$ such
  ///          that @f$ \mathbf{\alpha}, \mathbf{\beta}, P_N @f$ are mutually
  ///          orthogonal. Then for any point @f$ S\in S^2 @f$ near @f$ P_N @f$,
  ///          we can parameterize it as @f$ \xi = (\alpha \cdot S, \beta \cdot
  ///          S) @f$.
  ///
  ///          With this minimal parameterization, we have
  ///
  ///             @f$ \frac{\partial P_N}{\partial \xi} = \begin{bmatrix}
  ///             \mathbf{\alpha} & \mathbf{\beta} \end{bmatrix} @f$
  ///
  ///             @f$ \frac{\partial \xi}{\partial P} = \begin{bmatrix}
  ///             \mathbf{\alpha}^T \\ \mathbf{\beta}^T \end{bmatrix} / |P| @f$
  ///
  ///          And the output Jacobian w.r.t. the original 3D point is:
  ///
  ///             @f$ \text{jacobian_wrt_point3} =
  ///             \text{jacobian_wrt_normalized} * \frac{\partial P_N}{\partial
  ///             P} @f$
  ///
  ///          where
  ///             @f$ \frac{\partial P_N}{\partial P} = \frac{\partial
  ///             P_N}{\partial \xi} \frac{\partial \xi}{\partial P} =
  ///             \begin{bmatrix} \mathbf{\alpha} & \mathbf{\beta} \end{bmatrix}
  ///             \begin{bmatrix} \mathbf{\alpha}^T  \\ \mathbf{\beta}^T
  ///             \end{bmatrix}  / |P| = (I - P_N P_N^T) / |P| @f$
  ///
  ///           Note that
  ///
  ///             @f$  \begin{bmatrix} \mathbf{\alpha} & \mathbf{\beta}
  ///             \end{bmatrix} \begin{bmatrix} \mathbf{\alpha}^T
  ///             \\ \mathbf{\beta}^T \end{bmatrix} = \begin{bmatrix}
  ///             \mathbf{\alpha} & \mathbf{\beta} & P_N \end{bmatrix}
  ///             \begin{bmatrix} \mathbf{\alpha}^T  \\ \mathbf{\beta}^T
  ///             \\ P_N^T \end{bmatrix} - P_N P_N^T = I - P_N P_N^T @f$
  ///
  /// In the implementation, we expand the matrix multiplication to reduce the
  /// number of floating point operations.
  static void convertJacobianWrtNormalized3ToJacobianWrtPoint3(
      const double* jacobian_wrt_normalized, const double* normalized,
      const double& norm, double* jacobian_wrt_point3) {
    ConstJacobianMap<double, 2, 3> jacobian_wrt_normalized_matrix(
        jacobian_wrt_normalized);
    const double& x = normalized[0];
    const double& y = normalized[1];
    const double& z = normalized[2];
    double inv_norm = 1. / norm;
    const JacobianMatrix<double, 2, 3> tmp =
        jacobian_wrt_normalized_matrix * inv_norm;
    // jacobian_wrt_point3 = tmp * (I - normalized * normalized^T)
    //                     = tmp - (tmp * normalized) * normalized^T
    const double* tmpd = tmp.data();
    const double d0 = tmpd[0] * x + tmpd[1] * y + tmpd[2] * z;
    const double d1 = tmpd[3] * x + tmpd[4] * y + tmpd[5] * z;
    jacobian_wrt_point3[0] = tmpd[0] - d0 * x;
    jacobian_wrt_point3[1] = tmpd[1] - d0 * y;
    jacobian_wrt_point3[2] = tmpd[2] - d0 * z;
    jacobian_wrt_point3[3] = tmpd[3] - d1 * x;
    jacobian_wrt_point3[4] = tmpd[4] - d1 * y;
    jacobian_wrt_point3[5] = tmpd[5] - d1 * z;
  }

  /// @brief  A helper function for project3AndComputeJacobian().
  ///
  /// Usually, the Jacobian w.r.t. the Homo-normalized 2D point (on the
  /// normalized image plane @f$ Z=1 @f$) is easier to compute than that
  /// w.r.t. the original 3D point. This function can convert the
  /// Jacobian w.r.t. the Homo-normalized 2D point to that w.r.t. the
  /// original 3D point.
  ///
  /// @param jacobian_wrt_hnormalized
  ///                Jacobian of the 2D pixel w.r.t. the Homo-normalized 2D
  ///                point. A `2 x 2` matrix.
  /// @param hnormalized
  ///                The Homo-normalized 2D point.
  /// @param Z       The Z value for the original 3D point
  /// @param jacobian_wrt_point3
  ///                The output Jacobian of the 2D pixel w.r.t. the original 3D
  ///                point. A `2 x 3` matrix.
  ///
  /// @details We compute @p jacobian_wrt_point3 as:
  ///
  ///  @f$ \text{jacobian_wrt_point3} = \text{jacobian_wrt_hnormalized} * H @f$
  ///
  ///  where @f$ H @f$ is the homogeneous transformation matrix from the
  ///  normalized image plane @f$ Z=1 @f$ to the original 3D point.
  ///
  ///  @f$ H = \begin{bmatrix} \frac{1}{Z} & 0 & -\frac{X}{Z^2} \\ 0 &
  ///  \frac{1}{Z} & -\frac{Y}{Z^2} \end{bmatrix} @f$
  ///
  /// In the implementation, we expand the matrix multiplication to reduce the
  /// number of floating point operations.
  static void convertJacobianWrtHnormalizedToJacobianWrtPoint3(
      const double* jacobian_wrt_hnormalized, const double* hnormalized,
      const double& Z, double* jacobian_wrt_point3) {
    const double X_on_Z2 = hnormalized[0] / Z;
    const double Y_on_Z2 = hnormalized[1] / Z;
    const double* jalias = jacobian_wrt_hnormalized;
    jacobian_wrt_point3[0] = jalias[0] / Z;
    jacobian_wrt_point3[1] = jalias[1] / Z;
    jacobian_wrt_point3[2] = -jalias[0] * X_on_Z2 - jalias[1] * Y_on_Z2;
    jacobian_wrt_point3[3] = jalias[2] / Z;
    jacobian_wrt_point3[4] = jalias[3] / Z;
    jacobian_wrt_point3[5] = -jalias[2] * X_on_Z2 - jalias[3] * Y_on_Z2;
  }
};

/// @brief  The CRTP base class for camera models, which implements the common
/// interfaces.
///
/// @tparam Derived            The derived class.
/// @tparam _num_projection_params  The number of projection parameters.
/// @tparam _num_distortion_params  The number of distortion parameters.
///
template <
    typename Derived, int _num_projection_params, int _num_distortion_params>
class CameraModel : public CameraModelInterface {
 public:
  static constexpr int kNumProjectionParams = _num_projection_params;
  static constexpr int kNumDistortionParams = _num_distortion_params;
  static constexpr int kNumIntrinsics =
      kNumProjectionParams + kNumDistortionParams;

  template <typename Scalar>
  using Intrinsics = Eigen::Matrix<Scalar, kNumIntrinsics, 1>;
  using Intrinsicsd = Intrinsics<double>;
  using Intrinsicsf = Intrinsics<float>;

  using Interface = CameraModelInterface;
  using Interface::PointTypeTraits;
  using Interface::Vector2;
  using Interface::Vector2d;
  using Interface::Vector2f;
  using Interface::Vector3;
  using Interface::Vector3d;
  using Interface::Vector3f;

  using Interface::backProject2;
  using Interface::backProject3;
  using Interface::project2;
  using Interface::project3;

 public:
  CameraModel() {}

  template <typename IntrinsicsLike>
  explicit CameraModel(const IntrinsicsLike& intrinsics)
      : intrinsics_(intrinsics) {}

  template <typename ProjectionParamsVec, typename DistortionParamsVec>
  explicit CameraModel(
      const ProjectionParamsVec& projection_params,
      const DistortionParamsVec& distortion_params) {
    setProjectionParams(projection_params);
    setDistortionParams(distortion_params);
  }

  std::unique_ptr<Derived> cloneDerived() const {
    return std::make_unique<Derived>(intrinsics_);
  }

  std::unique_ptr<CameraModelInterface> clone() const override {
    return cloneDerived();
  }

  template <typename ProjectionParamsVec>
  void setProjectionParams(const ProjectionParamsVec& projection_params) {
    intrinsics_.template head<kNumProjectionParams>() = projection_params;
  }

  template <typename DistortionParamsVec>
  void setDistortionParams(const DistortionParamsVec& distortion_params) {
    if constexpr (kNumDistortionParams > 0) {
      intrinsics_.template tail<kNumDistortionParams>() = distortion_params;
    }
  }

  template <typename IntrinsicsLike>
  void setIntrinsics(const IntrinsicsLike& intrinsics) {
    intrinsics_ = intrinsics;
  }

 public:
  /// @brief       Project a 3D point to a 2D point. A subclass is required to
  /// implement a template function
  ///              `project3Impl()` with the same signature.
  /// @tparam Point3    the type of the 3D point. It can be any type that
  ///                   acts like `Vector3<Scalar>` or `cv::Point3_<Scalar>`
  ///                   or even `Scalar*`, where `Scalar` can be `float` or
  ///                   `double` or any other type that can be used as the
  ///                   scalar type of Eigen, such as `ceres::Jet`.
  /// @tparam ProjectionParams
  ///                   the type of the projection parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @tparam DistortionParams
  ///                   the type of the distortion parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @param point      the 3D point to be projected
  /// @param projection_params
  ///                   the projection parameters
  /// @param distortion_params
  ///                   the distortion parameters
  /// @param pixel      the output 2D pixel
  /// @return           Return true if the projection is successful, false
  /// otherwise.
  template <
      typename Point3, typename ProjectionParams, typename DistortionParams>
  static bool project3(
      const Point3& point, const ProjectionParams& projection_params,
      const DistortionParams& distortion_params,
      typename PointTypeTraits<Point3>::Scalar* pixel) {
    return Derived::template project3Impl(
        getPointData(point), &(projection_params[0]), &(distortion_params[0]),
        pixel);
  }

  template <typename Point3, typename IntrinsicsLike>
  static bool project3(
      const Point3& point, const IntrinsicsLike& intrinsics,
      typename PointTypeTraits<Point3>::Scalar* pixel) {
    using Scalar = typename PointTypeTraits<Point3>::Scalar;
    if constexpr (kNumDistortionParams > 0) {
      return Derived::template project3Impl(
          getPointData(point), &(intrinsics[0]),
          &(intrinsics[0]) + kNumProjectionParams, pixel);
    } else {
      return Derived::template project3Impl(
          getPointData(point), &(intrinsics[0]),
          static_cast<const Scalar*>(nullptr), pixel);
    }
  }

  template <typename Point3>
  bool project3(
      const Point3& point,
      typename PointTypeTraits<Point3>::Scalar* pixel) const {
    // ASSERT(false);
    using Scalar = typename PointTypeTraits<Point3>::Scalar;
    if constexpr (
        std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
      const Scalar* tmpp;
      if constexpr (PointTypeTraits<Point3>::kIsOpencvPoint) {
        tmpp = &point.x;
      } else {
        tmpp = &point[0];
      }
      return project3(tmpp, pixel);
    } else {
      // Scalar is ceres::Jet
      using Scalar = typename PointTypeTraits<Point3>::Scalar;
      Intrinsics<Scalar> intrinsics = intrinsics_.template cast<Scalar>();
      return Derived::template project3Impl(
          getPointData(point), intrinsics.data(),
          intrinsics.data() + kNumProjectionParams, pixel);
    }
  }

  /// @brief       BackProject a 2D point to a 3D point on the unit sphere. A
  /// subclass is required to implement a template function
  ///              `backProject3Impl()` with the same signature.
  /// @tparam Point2    the type of the 2D point. It can be any type that
  ///                   acts like `Vector2<Scalar>` or `cv::Point2_<Scalar>`
  ///                   or even `Scalar*`, where `Scalar` can be `float` or
  ///                   `double`.
  /// @tparam ProjectionParams
  ///                   the type of the projection parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @tparam DistortionParams
  ///                   the type of the distortion parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @param pixel      the 2D point to be back-projected
  /// @param projection_params
  ///                   the projection parameters
  /// @param distortion_params
  ///                   the distortion parameters
  /// @param point      the output 3D point on the unit sphere.
  /// @return           Return true if the back-projection is successful, false
  /// otherwise.
  template <
      typename Point2, typename ProjectionParams, typename DistortionParams>
  static bool backProject3(
      const Point2& pixel, const ProjectionParams& projection_params,
      const DistortionParams& distortion_params,
      typename PointTypeTraits<Point2>::Scalar* point) {
    return Derived::template backProject3Impl(
        getPointData(pixel), &(projection_params[0]), &(distortion_params[0]),
        point);
  }

  /// @brief Project a 3D point to a 2D pixel and compute the Jacobians w.r.t.
  /// the 3D point, the projection parameters, and the distortion parameters.
  /// If a subclass implements `project3AndComputeJacobiansImpl()`, then this
  /// function will call it; otherwise, ceres::Jet will be used to compute the
  /// Jacobians automatically.
  /// @tparam Scalar     the scalar type. It can be `float` or `double`.
  ///
  /// @param point          The 3D point to project.
  /// @param proj_params    The projection parameters.
  /// @param dist_params    The distortion parameters.
  /// @param pixel          The output 2D pixel.
  /// @param jacobian_wrt_point
  ///                       The Jacobian w.r.t. the 3D point. A 2 x 3 matrix.
  /// @param jacobian_wrt_proj_params
  ///                       The Jacobian w.r.t. the projection parameters. A
  ///                       `2 x kNumProjectionParams` matrix.
  /// @param jacobian_wrt_dist_params
  ///                       The Jacobian w.r.t. the distortion parameters.
  ///                       A `2 x kNumDistortionParams` matrix.
  /// @return      Return true if the projection is successful, false otherwise.
  /// @note
  ///   The Jacobians are all row-major, to be consistent with ceres-solver.
  ///
  static bool project3AndComputeJacobians(
      const double* point, const double* proj_params, const double* dist_params,
      double* pixel, double* jacobian_wrt_point,
      double* jacobian_wrt_proj_params, double* jacobian_wrt_dist_params) {
    static constexpr bool kSubClassHasImplementedProjectAndJacobian =
        HasMemberFunction_project3AndComputeJacobiansImpl<
            Derived, const double*, const double*, const double*, double*,
            double*, double*, double*>;
    if constexpr (kSubClassHasImplementedProjectAndJacobian) {
      return Derived::project3AndComputeJacobiansImpl(
          point, proj_params, dist_params, pixel, jacobian_wrt_point,
          jacobian_wrt_proj_params, jacobian_wrt_dist_params);
    } else {
      return project3AndAutoJacobians<Derived>(
          point, proj_params, dist_params, pixel, jacobian_wrt_point,
          jacobian_wrt_proj_params, jacobian_wrt_dist_params);
    }
  }

  bool project3AndComputeJacobian(
      const double* point, double* pixel,
      double* jacobian_wrt_point) const override {
    return project3AndComputeJacobians(
        point, projectionParams(), distortionParams(), pixel,
        jacobian_wrt_point, nullptr, nullptr);
  }

  bool project3AndComputeJacobiansWithExternalParameters(
      const double* point, const double* proj_params, const double* dist_params,
      double* pixel, double* jacobian_wrt_point,
      double* jacobian_wrt_proj_params,
      double* jacobian_wrt_dist_params) const override {
    return project3AndComputeJacobians(
        point, proj_params, dist_params, pixel, jacobian_wrt_point,
        jacobian_wrt_proj_params, jacobian_wrt_dist_params);
  }

  bool project3AndComputeJacobiansWithExternalParameters(
      const double* point, const double* intrinsics, double* pixel,
      double* jacobian_wrt_point,
      double* jacobian_wrt_intrinsics) const override {
    if constexpr (kNumDistortionParams == 0) {
      return project3AndComputeJacobians(
          point, intrinsics, intrinsics + kNumProjectionParams, pixel,
          jacobian_wrt_point, jacobian_wrt_intrinsics, nullptr);
    } else {
      if (jacobian_wrt_intrinsics == nullptr) {
        return project3AndComputeJacobians(
            point, intrinsics, intrinsics + kNumProjectionParams, pixel,
            jacobian_wrt_point, nullptr, nullptr);
      } else {
        JacobianMap<double, 2, kNumIntrinsics> jacobian_wrt_intrinsics_matrix(
            jacobian_wrt_intrinsics);

        JacobianMatrix<double, 2, kNumProjectionParams>
            jacobian_wrt_proj_params;
        JacobianMatrix<double, 2, kNumDistortionParams>
            jacobian_wrt_dist_params;
        bool success = project3AndComputeJacobians(
            point, intrinsics, intrinsics + kNumProjectionParams, pixel,
            jacobian_wrt_point, jacobian_wrt_proj_params.data(),
            jacobian_wrt_dist_params.data());
        if (success) {
          jacobian_wrt_intrinsics_matrix << jacobian_wrt_proj_params,
              jacobian_wrt_dist_params;
        }
        return success;
      }
    }
  }

  /// @brief   Projects a 2D point on the normalized image plane (z = 1) to a 2D
  /// pixel.
  ///          A subclass is required to implement a template function
  ///          `project2Impl()` with the same signature.
  /// @tparam Point2          the type of the 2D point. It can be
  /// `cv::Point2_<Scalar>`
  ///                         or `Eigen::Vector2<Scalar>` or even `Scalar*`,
  ///                         where `Scalar` is `float` or `double`.
  /// @tparam ProjectionParams
  ///                   the type of the projection parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @tparam DistortionParams
  ///                   the type of the distortion parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @param  point          the 2D point on the normalized image plane.
  /// @param  projection_params   the projection parameters.
  /// @param  distortion_params   the distortion parameters.
  /// @param  pixel          the output 2D pixel.
  /// @return  `true` if the projection is successful, `false` otherwise.
  template <
      typename Point2, typename ProjectionParams, typename DistortionParams>
  static bool project2(
      const Point2& point, const ProjectionParams& projection_params,
      const DistortionParams& distortion_params,
      typename PointTypeTraits<Point2>::Scalar* pixel) {
    return Derived::template project2Impl(
        getPointData(point), &(projection_params[0]), &(distortion_params[0]),
        pixel);
  }

  /// @brief   Back projects a 2D pixel to a 2D point on the normalized image
  /// plane (z = 1).
  ///          A subclass is required to implement a template function
  ///          `backProject2Impl()` with the same signature.
  /// @tparam Point2    the type of the 2D point. It can be
  /// `cv::Point2_<Scalar>`
  ///                   or `Eigen::Vector2<Scalar>` or even `Scalar*`, where
  ///                   `Scalar` is `float` or `double`.
  /// @tparam ProjectionParams
  ///                   the type of the projection parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  /// @tparam DistortionParams
  ///                   the type of the distortion parameters. It can be any
  ///                   type that acts like `VectorX<Scalar>` or `Scalar*`.
  ///
  /// @param pixel      the 2D pixel.
  /// @param projection_params
  ///                   the projection parameters.
  /// @param distortion_params
  ///                   the distortion parameters.
  /// @param point
  ///                   the output 2D point on the normalized image plane.
  /// @return   Returns `true` if the back projection is successful, `false`
  /// otherwise.
  template <
      typename Point2, typename ProjectionParams, typename DistortionParams>
  static bool backProject2(
      const Point2& pixel, const ProjectionParams& projection_params,
      const DistortionParams& distortion_params,
      typename PointTypeTraits<Point2>::Scalar* point) {
    return Derived::template backProject2Impl(
        getPointData(pixel), &(projection_params[0]), &(distortion_params[0]),
        point);
  }

 public:
  double* projectionParams() override {
    return intrinsics_.data();
  }

  const double* projectionParams() const override {
    return intrinsics_.data();
  }

  double* distortionParams() override {
    return intrinsics_.data() + kNumProjectionParams;
  }

  const double* distortionParams() const override {
    return intrinsics_.data() + kNumProjectionParams;
  }

  int numProjectionParams() const override {
    return kNumProjectionParams;
  }

  int numDistortionParams() const override {
    return kNumDistortionParams;
  }

  int numIntrinsics() const override {
    return kNumIntrinsics;
  }

  double* intrinsics() override {
    return intrinsics_.data();
  }

  const double* intrinsics() const override {
    return intrinsics_.data();
  }

  /// Derived class can override these methods if their projection parameters
  /// are not stored in the order of fx, fy, cx, cy.
  double fx() const override {
    return intrinsics_(0);
  }

  double fy() const override {
    return intrinsics_(1);
  }

  double cx() const override {
    return intrinsics_(2);
  }

  double cy() const override {
    return intrinsics_(3);
  }

  /// @brief   Projects a 3D point to a 2D pixel. See @ref
  /// CameraModelInterface::project3()
  bool project3(const double* point, double* pixel) const override {
    return project3(point, projectionParams(), distortionParams(), pixel);
  }
  bool project3(const float* point, float* pixel) const override {
    Intrinsicsf intrinsicsf = intrinsics_.template cast<float>();
    return project3(
        point, intrinsicsf.data(), intrinsicsf.data() + kNumProjectionParams,
        pixel);
  }

  /// @brief   Backprojects a 2D pixel to a 3D point. See @ref
  /// CameraModelInterface::backProject()
  bool backProject3(const double* pixel, double* point) const override {
    return backProject3(pixel, projectionParams(), distortionParams(), point);
  }
  bool backProject3(const float* pixel, float* point) const override {
    Intrinsicsf intrinsicsf = intrinsics_.template cast<float>();
    return backProject3(
        pixel, intrinsicsf.data(), intrinsicsf.data() + kNumProjectionParams,
        point);
  }

  /// @brief  Project a 2D point on the normalized image plane (z = 1) to a 2D
  /// pixel.
  ///         See @ref CameraModelInterface::project2()
  bool project2(const double* point, double* pixel) const override {
    return project2(point, projectionParams(), distortionParams(), pixel);
  }
  bool project2(const float* point, float* pixel) const override {
    Intrinsicsf intrinsicsf = intrinsics_.template cast<float>();
    return project2(
        point, intrinsicsf.data(), intrinsicsf.data() + kNumProjectionParams,
        pixel);
  }

  /// @brief  Back-project a 2D pixel to a 2D point on the normalized image
  /// plane (z = 1).
  ///         See @ref CameraModelInterface::backProject2()
  bool backProject2(const double* pixel, double* point) const override {
    return backProject2(pixel, projectionParams(), distortionParams(), point);
  }
  bool backProject2(const float* pixel, float* point) const override {
    Intrinsicsf intrinsicsf = intrinsics_.template cast<float>();
    return backProject2(
        pixel, intrinsicsf.data(), intrinsicsf.data() + kNumProjectionParams,
        point);
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(kNumIntrinsics % 2 == 0);

 private:
  DEFINE_HAS_MEMBER_FUNCTION(project3AndComputeJacobiansImpl)

 private:
  Intrinsicsd intrinsics_;
};

using LutCameraModel = CameraModelInterface::LUT;
struct CameraModelInterface::LUT : public CameraModelInterface {
  using Interface = CameraModelInterface;
  using Interface::backProject2;
  using Interface::backProject3;
  using Interface::project2;
  using Interface::project3;
  using Interface::Vector2;
  using Interface::Vector3;

  // static constexpr int kNumProjectionParams = 0;
  // static constexpr int kNumDistortionParams = 0;
  // static constexpr int kNumIntrinsics = 0;

  int width() const {
    return lut_data_->width_;
  }
  int height() const {
    return lut_data_->height_;
  }

  double fx() const override {
    return lut_data_->fx_;
  }

  double fy() const override {
    return lut_data_->fy_;
  }

  double cx() const override {
    return lut_data_->cx_;
  }

  double cy() const override {
    return lut_data_->cy_;
  }

  LUT(const CameraModelInterface* cam_model, const int width, const int height,
      bool interpolate_pixel, bool interpolate_jac, LutFunction lut_function_in,
      const float project2_max_fov, const float project3_max_fov)
      : lut_data_(std::make_unique<LUTData>(
            cam_model, width, height, lut_function_in, project2_max_fov,
            project3_max_fov)),
        interpolate_pixel_(interpolate_pixel),
        interpolate_jac_(interpolate_jac) {}

  LUT(const LUT& other) = default;
  LUT(LUT&& other) = default;
  ~LUT() {}

  std::unique_ptr<LUT> cloneLUT() const {
    return std::make_unique<LUT>(*this);
  }

  std::unique_ptr<CameraModelInterface> clone() const override {
    return cloneLUT();
  }

  // trivial implementation for some interfaces which are usually used
  // for camera calibration. (LUT can't be used for calibration)
  int numProjectionParams() const override {
    return 0;
  }
  int numDistortionParams() const override {
    return 0;
  }
  double* projectionParams() override {
    return nullptr;
  }
  const double* projectionParams() const override {
    return nullptr;
  }
  double* distortionParams() override {
    return nullptr;
  }
  const double* distortionParams() const override {
    return nullptr;
  }
  int numIntrinsics() const override {
    return 0;
  }
  double* intrinsics() override {
    return nullptr;
  }
  const double* intrinsics() const override {
    return nullptr;
  }
  bool project3AndComputeJacobiansWithExternalParameters(
      const double* point, const double* proj_params, const double* dist_params,
      double* pixel, double* jacobian_wrt_point,
      double* jacobian_wrt_proj_params,
      double* jacobian_wrt_dist_params) const override {
    // return project3AndComputeJacobian(point, pixel, jacobian_wrt_point);
    return false;
  }
  bool project3AndComputeJacobiansWithExternalParameters(
      const double* point, const double* intrinsics, double* pixel,
      double* jacobian_wrt_point,
      double* jacobian_wrt_intrinsics) const override {
    // return project3AndComputeJacobian(point, pixel, jacobian_wrt_point);
    return false;
  }

  /// @brief  Project a 2D point on the normalized image plane (z = 1) to a 2D
  /// pixel.
  /// @param point   The input 2D point on the normalized image plane.
  /// @param pixel   The output 2D pixel in image coordinate.
  /// @param interpolate  Whether to interpolate the pixel value.
  /// @return  Return true if the projection is successful, false otherwise.
  bool project2(const double* point, double* pixel) const override {
    Vector2f pointf(point[0], point[1]);
    Vector2f pixelf;
    bool success = project2(pointf.data(), pixelf.data());
    pixel[0] = pixelf[0];
    pixel[1] = pixelf[1];
    return success;
  }
  bool project2(const float* point, float* pixel) const override {
    const auto& proj2_info_ = lut_data_->proj2_info_;
    const auto& proj2_table_ = lut_data_->proj2_table_;
    const float* p2 = point;
    const float query_x =
        (p2[0] - proj2_info_.startx) / proj2_info_.resolutionx;
    const float query_y =
        (p2[1] - proj2_info_.starty) / proj2_info_.resolutiony;
    if (!checkQuery(
            query_x, query_y, proj2_info_.pwidth, proj2_info_.pheight)) {
      return false;
    }
    Vector2f v = lookup(
        query_x, query_y, proj2_info_.pwidth, proj2_info_.pheight, proj2_table_,
        interpolate_pixel_);
    pixel[0] = v[0];
    pixel[1] = v[1];
    return true;
  }

  /// @brief  Back-project a 2D pixel to a 2D point on the normalized image
  /// plane (z = 1).
  /// @param pixel   The input 2D pixel in image coordinate.
  /// @param point   The output 2D point on the normalized image plane.
  /// @param interpolate  Whether to interpolate the point value.
  /// @return  Return true if the back-projection is successful, false
  /// otherwise.
  bool backProject2(const double* pixel, double* point) const override {
    Vector2f pixelf(pixel[0], pixel[1]);
    Vector2f pointf;
    if (backProject2(pixelf.data(), pointf.data())) {
      point[0] = pointf[0];
      point[1] = pointf[1];
      return true;
    }
    return false;
  }
  bool backProject2(const float* pixel, float* point) const override {
    const int width_ = lut_data_->width_;
    const int height_ = lut_data_->height_;
    const auto& backproj2_table_ = lut_data_->backproj2_table_;
    const float& query_x = pixel[0];
    const float& query_y = pixel[1];
    if (!checkQuery(query_x, query_y, width_, height_)) {
      return false;
    }
    Vector2f v = lookup(
        query_x, query_y, width_, height_, backproj2_table_,
        interpolate_pixel_);
    // if v[0] or v[1] is nan, return false
    if (std::isnan(v[0]) || std::isnan(v[1])) {
      return false;
    }
    point[0] = v[0];
    point[1] = v[1];
    return true;
  }
  bool backProject2(int xi, int yi, float* point) const {
    const int width_ = lut_data_->width_;
    const int height_ = lut_data_->height_;
    const auto& backproj2_table_ = lut_data_->backproj2_table_;
    if (!checkQuery(xi, yi, width_, height_)) {
      return false;
    }
    const Vector2f& v = backproj2_table_[yi * width_ + xi];
    // if v[0] or v[1] is nan, return false
    if (std::isnan(v[0]) || std::isnan(v[1])) {
      return false;
    }
    point[0] = v[0];
    point[1] = v[1];
    return true;
  }

  /// @brief Project a 3D point to a 2D pixel.
  ///
  /// @param point    The input 3D point in camera coordinate
  /// @param pixel    The output 2D pixel in image coordinate. A `2 x 1` vector.
  /// @param interpolate  Whether to interpolate the pixel value.
  /// @return  Return true if the projection is successful, false otherwise.
  ///
  bool project3(const double* point, double* pixel) const override {
    Vector3f pointf(point[0], point[1], point[2]);
    Vector2f pixelf;
    bool success = project3(pointf.data(), pixelf.data());
    pixel[0] = pixelf[0];
    pixel[1] = pixelf[1];
    return success;
  }
  bool project3(const float* point, float* pixel) const override {
    const auto& proj3_info_ = lut_data_->proj3_info_;
    const auto& proj3_table_ = lut_data_->proj3_table_;
    Vector3f p3(point[0], point[1], point[2]);
    p3.normalize();
    const float p3zp1 = p3[2] + 1.0f;
    const float p3cx = p3[0] / p3zp1;
    const float p3cy = p3[1] / p3zp1;
    const float query_x = (p3cx - proj3_info_.startx) / proj3_info_.resolutionx;
    const float query_y = (p3cy - proj3_info_.starty) / proj3_info_.resolutiony;
    if (!checkQuery(
            query_x, query_y, proj3_info_.pwidth, proj3_info_.pheight)) {
      return false;
    }
    Vector2f v = lookup(
        query_x, query_y, proj3_info_.pwidth, proj3_info_.pheight, proj3_table_,
        interpolate_pixel_);
    pixel[0] = v[0];
    pixel[1] = v[1];
    return true;
  }

  /// @brief Back-project a 2D pixel to a 3D point (on the unit sphere).
  ///
  /// @param pixel    The input 2D pixel in image coordinate
  /// @param point    The output 3D point in camera coordinate
  /// @param interpolate  Whether to interpolate the point value.
  /// @return  Return true if the back-projection is successful, false
  /// otherwise.
  ///
  bool backProject3(const double* pixel, double* point) const override {
    Vector2f pixelf(pixel[0], pixel[1]);
    Vector3f pointf;
    bool success = backProject3(pixelf.data(), pointf.data());
    point[0] = pointf[0];
    point[1] = pointf[1];
    point[2] = pointf[2];
    return success;
  }
  bool backProject3(const float* pixel, float* point) const override {
    const int width_ = lut_data_->width_;
    const int height_ = lut_data_->height_;
    const auto& backproj3_table_ = lut_data_->backproj3_table_;
    const float& query_x = pixel[0];
    const float& query_y = pixel[1];
    if (!checkQuery(query_x, query_y, width_, height_)) {
      return false;
    }
    Vector3f v = lookup(
        query_x, query_y, width_, height_, backproj3_table_,
        interpolate_pixel_);
    if (interpolate_pixel_) {
      v.normalize();  // The interpolated point needs normalization.
    }
    point[0] = v[0];
    point[1] = v[1];
    point[2] = v[2];
    return true;
  }
  bool backProject3(int xi, int yi, float* point) const {
    const int width_ = lut_data_->width_;
    const int height_ = lut_data_->height_;
    const auto& backproj3_table_ = lut_data_->backproj3_table_;
    if (!checkQuery(xi, yi, width_, height_)) {
      return false;
    }
    const Vector3f& v = backproj3_table_[yi * width_ + xi];
    point[0] = v[0];
    point[1] = v[1];
    point[2] = v[2];
    return true;
  }

  bool project3AndComputeJacobian(
      const double* point, double* pixel,
      double* jacobian_wrt_point) const override {
    const auto& proj3_info_ = lut_data_->proj3_info_;
    const auto& proj3_table_ = lut_data_->proj3_table_;
    const auto& proj3_jac_table_ = lut_data_->proj3_jac_table_;
    Vector3f p3(point[0], point[1], point[2]);
    const float norm = p3.norm();
    p3 /= norm;
    const float p3zp1 = p3[2] + 1.0f;
    const float p3cx = p3[0] / p3zp1;
    const float p3cy = p3[1] / p3zp1;
    const float query_x = (p3cx - proj3_info_.startx) / proj3_info_.resolutionx;
    const float query_y = (p3cy - proj3_info_.starty) / proj3_info_.resolutiony;
    if (!checkQuery(
            query_x, query_y, proj3_info_.pwidth, proj3_info_.pheight)) {
      return false;
    }

    if (pixel) {
      Vector2f v = lookup(
          query_x, query_y, proj3_info_.pwidth, proj3_info_.pheight,
          proj3_table_, interpolate_pixel_);
      pixel[0] = v[0];
      pixel[1] = v[1];
    }

    if (jacobian_wrt_point) {
      JacobianMatrix<float, 2, 3> j = lookup(
          query_x, query_y, proj3_info_.pwidth, proj3_info_.pheight,
          proj3_jac_table_, interpolate_jac_);
      const float* jd = j.data();
      const double Z = point[2];
      jacobian_wrt_point[0] = jd[0] / norm;
      jacobian_wrt_point[1] = jd[1] / norm;
      jacobian_wrt_point[2] = jd[2] / norm;
      jacobian_wrt_point[3] = jd[3] / norm;
      jacobian_wrt_point[4] = jd[4] / norm;
      jacobian_wrt_point[5] = jd[5] / norm;
    }
    return true;
  }

  bool backProject2AndComputeJacobian(
      const double* pixel, double* point,
      double* jacobian_wrt_pixel) const override {
    const int width_ = lut_data_->width_;
    const int height_ = lut_data_->height_;
    const auto& backproj2_table_ = lut_data_->backproj2_table_;
    const auto& backproj2_jac_table_ = lut_data_->backproj2_jac_table_;
    const float& query_x = pixel[0];
    const float& query_y = pixel[1];
    if (!checkQuery(query_x, query_y, width_, height_)) {
      return false;
    }
    if (point) {
      Vector2f v = lookup(
          query_x, query_y, width_, height_, backproj2_table_,
          interpolate_pixel_);

      // if v[0] or v[1] is nan, return false
      if (std::isnan(v[0]) || std::isnan(v[1])) {
        return false;
      }
      point[0] = v[0];
      point[1] = v[1];
    }

    if (jacobian_wrt_pixel) {
      JacobianMatrix<float, 2, 2> j = lookup(
          query_x, query_y, width_, height_, backproj2_jac_table_,
          interpolate_jac_);

      // if j has nan, return false
      if (j.hasNaN()) {
        return false;
      }
      const float* jd = j.data();
      jacobian_wrt_pixel[0] = jd[0];
      jacobian_wrt_pixel[1] = jd[1];
      jacobian_wrt_pixel[2] = jd[2];
      jacobian_wrt_pixel[3] = jd[3];
    }
    return true;
  }

  template <typename Point3>
  bool project3(
      const Point3& point,
      typename PointTypeTraits<Point3>::Scalar* pixel) const {
    // ASSERT(false);
    const auto& proj3_info_ = lut_data_->proj3_info_;
    const auto& proj3_table_ = lut_data_->proj3_table_;

    using Scalar = typename PointTypeTraits<Point3>::Scalar;
    const Scalar* tmpp;
    if constexpr (PointTypeTraits<Point3>::kIsOpencvPoint) {
      tmpp = &point.x;
    } else {
      tmpp = &point[0];
    }

    if constexpr (
        std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
      return project3(tmpp, pixel);
    } else {
      // Scalar is ceres::Jet
      static const Scalar kNum_1 = Scalar(1.0);
      Vector3<Scalar> p3(tmpp[0], tmpp[1], tmpp[2]);
      p3.normalize();
      const Scalar p3zp1 = p3[2] + kNum_1;
      const Scalar p3cx = p3[0] / p3zp1;
      const Scalar p3cy = p3[1] / p3zp1;
      const Scalar query_x =
          (p3cx - Scalar(proj3_info_.startx)) / Scalar(proj3_info_.resolutionx);
      const Scalar query_y =
          (p3cy - Scalar(proj3_info_.starty)) / Scalar(proj3_info_.resolutiony);

      const auto& table = proj3_table_;
      int table_w = proj3_info_.pwidth;
      int table_h = proj3_info_.pheight;
      if (!checkQuery(query_x.a, query_y.a, table_w, table_h)) {
        return false;
      }

      int x0 = std::floor(query_x.a);
      int y0 = std::floor(query_y.a);
      ASSERT(x0 >= 0 && x0 < table_w && y0 >= 0 && y0 < table_h);
      x0 = std::min(x0, table_w - 1);
      y0 = std::min(y0, table_h - 1);
      int x1 = x0 + 1;
      int y1 = y0 + 1;
      ASSERT(x1 < table_w && y1 < table_h);

      Scalar alpha = query_x - Scalar(x0);
      Scalar beta = query_y - Scalar(y0);
      Vector2<Scalar> v00 = table[y0 * table_w + x0].cast<Scalar>();
      Vector2<Scalar> v01 = table[y0 * table_w + x1].cast<Scalar>();
      Vector2<Scalar> v10 = table[y1 * table_w + x0].cast<Scalar>();
      Vector2<Scalar> v11 = table[y1 * table_w + x1].cast<Scalar>();
      Vector2<Scalar> v = v00 * (kNum_1 - alpha) * (kNum_1 - beta) +
                          v01 * alpha * (kNum_1 - beta) +
                          v10 * (kNum_1 - alpha) * beta + v11 * alpha * beta;
      pixel[0] = v[0];
      pixel[1] = v[1];
      return true;
    }
  }

 private:
  template <typename QueryScalar>
  static bool checkQuery(
      const QueryScalar query_x, const QueryScalar query_y, const int table_w,
      const int table_h) {
    if (query_x < 0 || query_x > table_w - 1 || query_y < 0 ||
        query_y > table_h - 1) {
      return false;
    } else {
      return true;
    }
  }

  template <typename ValueType>
  static ValueType bilinearInterpolate(
      const float alpha,  // weight_X0 / weight_X1 = (1 - alpha) / alpha
      const float beta,   // weight_0X / weight_1X = (1 - beta) / beta
      const ValueType& v00, const ValueType& v01, const ValueType& v10,
      const ValueType& v11) {
    return v00 * (1 - alpha) * (1 - beta) + v01 * alpha * (1 - beta) +
           v10 * (1 - alpha) * beta + v11 * alpha * beta;
  }

  template <typename ValueType>
  static ValueType interpolate(
      const float query_x, const float query_y, const int table_w,
      const int table_h, const std::vector<ValueType>& table,
      bool force_interpolate = false) {
    int x0 = std::floor(query_x);
    int y0 = std::floor(query_y);
    ASSERT(x0 >= 0 && x0 < table_w && y0 >= 0 && y0 < table_h);
    if (!force_interpolate) {
      if (x0 == query_x && y0 == query_y) {
        // x0 = std::min(std::max(x0, 0), table_w - 1);
        // y0 = std::min(std::max(y0, 0), table_h - 1);
        return table[y0 * table_w + x0];
      }
    } else {
      x0 = std::min(x0, table_w - 1);
      y0 = std::min(y0, table_h - 1);
    }

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    ASSERT(x1 < table_w && y1 < table_h);

    float alpha = query_x - x0;
    float beta = query_y - y0;
    const ValueType& v00 = table[y0 * table_w + x0];
    const ValueType& v01 = table[y0 * table_w + x1];
    const ValueType& v10 = table[y1 * table_w + x0];
    const ValueType& v11 = table[y1 * table_w + x1];
    return bilinearInterpolate(alpha, beta, v00, v01, v10, v11);
  }

  template <typename ValueType>
  static ValueType lookup(
      const float query_x, const float query_y, const int table_w,
      const int table_h, const std::vector<ValueType>& table,
      bool do_interpolate) {
    if (do_interpolate) {
      return interpolate(query_x, query_y, table_w, table_h, table);
    } else {
      const int x = std::round(query_x);
      const int y = std::round(query_y);
      return table[y * table_w + x];
    }
  }

 private:
  struct ProjectionTableInfo {
    int pwidth;
    int pheight;
    float resolutionx;
    float resolutiony;
    float startx;
    float starty;
  };

  class LUTData {
   public:
    friend class LUT;
    LUTData(
        const CameraModelInterface* cam_model, const int width,
        const int height, LutFunction lut_function_in,
        const float project2_max_fov, const float project3_max_fov)
        : width_(width),
          height_(height),
          fx_(cam_model->fx()),
          fy_(cam_model->fy()),
          cx_(cam_model->cx()),
          cy_(cam_model->cy()) {
      uint64_t lut_function = static_cast<uint64_t>(lut_function_in);

      if (lut_function & static_cast<uint64_t>(LutFunction::BACKPROJECT2)) {
        backproj2_table_.resize(width * height);
      }
      if (lut_function & static_cast<uint64_t>(LutFunction::BACKPROJECT3)) {
        backproj3_table_.resize(width * height);
      }
      if (lut_function &
          static_cast<uint64_t>(LutFunction::BACKPROJECT2_JACOBIAN)) {
        backproj2_jac_table_.resize(width * height);
      }

      double proj2min_x = std::numeric_limits<double>::max();
      double proj2min_y = std::numeric_limits<double>::max();
      double proj2max_x = -std::numeric_limits<double>::max();
      double proj2max_y = -std::numeric_limits<double>::max();
      double proj3min_x = std::numeric_limits<double>::max();
      double proj3min_y = std::numeric_limits<double>::max();
      double proj3max_x = -std::numeric_limits<double>::max();
      double proj3max_y = -std::numeric_limits<double>::max();
      double fx = -1.0, fy = -1.0;
      for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
          Vector2d pixel_center(x, y);
          // Vector2d pixel_center(x + 0.5, y + 0.5);  // Use this definition if
          // the origin of the image coordinate is at the left-top corner,
          // rather than the center of the first pixel.

          auto bproj3_res = cam_model->backProject3(pixel_center);
          ASSERT(bproj3_res.first);
          const auto& p3 = bproj3_res.second;
          ASSERT(std::abs(p3.squaredNorm() - 1.0) < 1e-8);

          if (lut_function & static_cast<uint64_t>(LutFunction::BACKPROJECT3)) {
            backproj3_table_[y * width + x] = p3.cast<float>();
          }
          const double p3zp1 = p3[2] + 1.0f;
          const double p3cx = p3[0] / p3zp1;
          const double p3cy = p3[1] / p3zp1;
          proj3min_x = std::min(proj3min_x, p3cx);
          proj3min_y = std::min(proj3min_y, p3cy);
          proj3max_x = std::max(proj3max_x, p3cx);
          proj3max_y = std::max(proj3max_y, p3cy);

          Eigen::Vector2d p2;
          JacobianMatrix<double, 2, 2> backproj2_jacobian;
          double* backproj2_jacobian_data = nullptr;
          if (lut_function &
              static_cast<uint64_t>(LutFunction::BACKPROJECT2_JACOBIAN)) {
            backproj2_jacobian_data = backproj2_jacobian.data();
          }
          bool bproj2_success = cam_model->backProject2AndComputeJacobian(
              pixel_center.data(), p2.data(), backproj2_jacobian_data);
          // ASSERT(bproj2_success);  // This assertion will fail if FOV > 180°
          if (bproj2_success) {
            if (lut_function &
                static_cast<uint64_t>(LutFunction::BACKPROJECT2)) {
              backproj2_table_[y * width + x] = p2.cast<float>();
            }
            if (lut_function &
                static_cast<uint64_t>(LutFunction::BACKPROJECT2_JACOBIAN)) {
              backproj2_jac_table_[y * width + x] =
                  backproj2_jacobian.cast<float>();
            }
            proj2min_x = std::min(proj2min_x, p2[0]);
            proj2min_y = std::min(proj2min_y, p2[1]);
            proj2max_x = std::max(proj2max_x, p2[0]);
            proj2max_y = std::max(proj2max_y, p2[1]);

            if (fx < 0.0) {
              ASSERT(fy < 0.0);
              if (p2[0] > 0.0 && p2[1] > 0.0) {
                // If p2 is the first positive point, then it's near the center
                // point of the normalized image plane, so we can use it and its
                // neighbors to estimate the focal length.
                ASSERT(x > 0 && y > 0);
                // Vector2f p2_neighbor =
                //     backproj2_table_[(y - 1) * width + (x - 1)];
                Vector2d p2_neighbor;
                Vector2d pixel_neighbor = pixel_center - Vector2d(1, 1);
                ASSERT(cam_model->backProject2AndComputeJacobian(
                    pixel_neighbor.data(), p2_neighbor.data(), nullptr));

                fx = 1.0 / (p2[0] - p2_neighbor.x());
                fy = 1.0 / (p2[1] - p2_neighbor.y());
              }
            }
          } else {
            // If backProject2 fails, we set the corresponding entry to NaN.
            if (lut_function &
                static_cast<uint64_t>(LutFunction::BACKPROJECT2)) {
              backproj2_table_[y * width + x].setConstant(
                  std::numeric_limits<float>::quiet_NaN());
            }

            if (lut_function &
                static_cast<uint64_t>(LutFunction::BACKPROJECT2_JACOBIAN)) {
              backproj2_jac_table_[y * width + x].setConstant(
                  std::numeric_limits<float>::quiet_NaN());
            }
          }
        }
      }

      ASSERT(fx > 0.0);
      ASSERT(fy > 0.0);

      double proj2_angle_thr = project2_max_fov / 2.0f;
      double proj2_xy_thr = std::tan(proj2_angle_thr);
      proj2min_x = std::max(proj2min_x, -proj2_xy_thr);
      proj2min_y = std::max(proj2min_y, -proj2_xy_thr);
      proj2max_x = std::min(proj2max_x, proj2_xy_thr);
      proj2max_y = std::min(proj2max_y, proj2_xy_thr);

      double proj3_angle_thr = project3_max_fov / 4.0f;
      double proj3_xy_thr = std::tan(proj3_angle_thr);
      proj3min_x = std::max(proj3min_x, -proj3_xy_thr);
      proj3min_y = std::max(proj3min_y, -proj3_xy_thr);
      proj3max_x = std::min(proj3max_x, proj3_xy_thr);
      proj3max_y = std::min(proj3max_y, proj3_xy_thr);

      proj2_info_.resolutionx = 1.0f / fx;
      proj2_info_.resolutiony = 1.0f / fy;
      proj2_info_.startx = proj2min_x;
      proj2_info_.starty = proj2min_y;
      proj2_info_.pwidth =
          std::ceil((proj2max_x - proj2min_x) / proj2_info_.resolutionx) + 1;
      proj2_info_.pheight =
          std::ceil((proj2max_y - proj2min_y) / proj2_info_.resolutiony) + 1;

      if (lut_function & static_cast<uint64_t>(LutFunction::PROJECT2)) {
        proj2_table_.resize(proj2_info_.pwidth * proj2_info_.pheight);
        for (size_t y = 0; y < proj2_info_.pheight; y++) {
          for (size_t x = 0; x < proj2_info_.pwidth; x++) {
            Vector2d p2(
                proj2_info_.startx + x * proj2_info_.resolutionx,
                proj2_info_.starty + y * proj2_info_.resolutiony);
            auto proj2_res = cam_model->project2(p2);
            ASSERT(proj2_res.first);
            proj2_table_[y * proj2_info_.pwidth + x] =
                proj2_res.second.cast<float>();
          }
        }
      }

      proj3_info_.resolutionx = 0.5f / fx;
      proj3_info_.resolutiony = 0.5f / fy;
      proj3_info_.startx = proj3min_x;
      proj3_info_.starty = proj3min_y;
      proj3_info_.pwidth =
          std::ceil((proj3max_x - proj3min_x) / proj3_info_.resolutionx) + 1;
      proj3_info_.pheight =
          std::ceil((proj3max_y - proj3min_y) / proj3_info_.resolutiony) + 1;

      if (lut_function & static_cast<uint64_t>(LutFunction::PROJECT3)) {
        proj3_table_.resize(proj3_info_.pwidth * proj3_info_.pheight);
      }
      if (lut_function &
          static_cast<uint64_t>(LutFunction::PROJECT3_JACOBIAN)) {
        proj3_jac_table_.resize(proj3_info_.pwidth * proj3_info_.pheight);
      }
      for (size_t y = 0; y < proj3_info_.pheight; y++) {
        for (size_t x = 0; x < proj3_info_.pwidth; x++) {
          Vector3d p3c(
              proj3_info_.startx + x * proj3_info_.resolutionx,
              proj3_info_.starty + y * proj3_info_.resolutiony, 1.0);
          p3c.normalize();
          ASSERT(p3c.z() > 0.0);
          Vector3d p3 = 2 * p3c.z() * p3c;
          p3[2] -= 1.0;
          ASSERT(std::abs(p3.squaredNorm() - 1.0) < 1e-8);

          Eigen::Vector2d pixel;
          JacobianMatrix<double, 2, 3> proj3_jacobian;
          double* proj3_jacobian_data = nullptr;
          if (lut_function &
              static_cast<uint64_t>(LutFunction::PROJECT3_JACOBIAN)) {
            proj3_jacobian_data = proj3_jacobian.data();
          }
          bool proj3_succes = cam_model->project3AndComputeJacobian(
              p3.data(), pixel.data(), proj3_jacobian_data);
          ASSERT(proj3_succes);
          if (lut_function & static_cast<uint64_t>(LutFunction::PROJECT3)) {
            proj3_table_[y * proj3_info_.pwidth + x] = pixel.cast<float>();
          }
          if (lut_function &
              static_cast<uint64_t>(LutFunction::PROJECT3_JACOBIAN)) {
            proj3_jac_table_[y * proj3_info_.pwidth + x] =
                proj3_jacobian.cast<float>();
          }
        }
      }
    }

   private:
    int width_;
    int height_;
    double fx_, fy_, cx_, cy_;
    ProjectionTableInfo proj2_info_;
    ProjectionTableInfo proj3_info_;
    std::vector<Vector2f> proj2_table_;
    std::vector<Vector2f> proj3_table_;
    std::vector<Vector2f> backproj2_table_;
    std::vector<Vector3f> backproj3_table_;

    std::vector<JacobianMatrix<float, 2, 3>> proj3_jac_table_;
    std::vector<JacobianMatrix<float, 2, 2>> backproj2_jac_table_;
  };

 private:
  std::shared_ptr<LUTData> lut_data_;
  bool interpolate_pixel_;
  bool interpolate_jac_;
};

}  // namespace sk4slam
