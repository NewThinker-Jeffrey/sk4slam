#pragma once

#include "sk4slam_camera/camera_model_types.h"

namespace sk4slam {

// createCameraModel

struct _CreateCameraModelHelper {
  template <typename CameraModel, typename IntrinsicsLike>
  std::unique_ptr<CameraModelInterface> operator()(
      const IntrinsicsLike& intrinsics) const {
    if constexpr (std::is_same_v<CameraModel, CameraModelInterface::LUT>) {
      LOGE("LUT should be created with CameraModelInterface::makeLUT()!");
      return nullptr;
    } else {
      return std::make_unique<CameraModel>(intrinsics);
    }
  }
};

template <typename IntrinsicsLike>
inline std::unique_ptr<CameraModelInterface> createCameraModel(
    CameraModelType cam_type, const IntrinsicsLike& intrinsics) {
  return processByCameraModelType<true>(
      _CreateCameraModelHelper(), cam_type, intrinsics);
}

template <CameraModelType cam_type, typename IntrinsicsLike>
inline std::unique_ptr<CameraModelInterface> createCameraModel(
    const IntrinsicsLike& intrinsics) {
  if constexpr (cam_type == CameraModelType::LUT) {
    LOGE("LUT should be created with CameraModelInterface::makeLUT()!");
    return nullptr;
  } else {
    return std::make_unique<GetCameraModelClass<cam_type>>(intrinsics);
  }
}

// project3WithCameraModel

struct _Project3WithCameraModelHelper {
  // project with external parameters (Only for optimizable camera models)
  template <typename CameraModel, typename FirstArg, typename... RestArgs>
  bool operator()(FirstArg& first_arg, RestArgs&&... rest_args) const {
    return CameraModel::project3(
        std::forward<FirstArg>(first_arg),
        std::forward<RestArgs>(rest_args)...);
  }

  // project with internal parameters
  template <typename CameraModel, typename... Args>
  bool operator()(const CameraModelInterface* cam, Args&&... args) const {
    auto casted_cam = static_cast<const CameraModel*>(cam);
    return casted_cam->project3(std::forward<Args>(args)...);
  }
};

template <typename Point3, typename ProjectionParams, typename DistortionParams>
inline bool project3WithCameraModel(
    CameraModelType cam_type, const Point3& point,
    const ProjectionParams& projection_params,
    const DistortionParams& distortion_params,
    typename CameraModelInterface::PointTypeTraits<Point3>::Scalar* pixel) {
  return processByCameraModelType<true>(
      _Project3WithCameraModelHelper(), cam_type, point, projection_params,
      distortion_params, pixel);
}

template <typename Point3, typename IntrinsicsLike>
inline bool project3WithCameraModel(
    CameraModelType cam_type, const Point3& point,
    const IntrinsicsLike& intrinsics,
    typename CameraModelInterface::PointTypeTraits<Point3>::Scalar* pixel) {
  return processByCameraModelType<true>(
      _Project3WithCameraModelHelper(), cam_type, point, intrinsics, pixel);
}

template <typename Point3>
inline bool project3WithCameraModel(
    const CameraModelInterface* cam, const Point3& point,
    typename CameraModelInterface::PointTypeTraits<Point3>::Scalar* pixel) {
  return processByCameraModelType<false>(
      _Project3WithCameraModelHelper(), getCameraModelType(cam), cam, point,
      pixel);
}

}  // namespace sk4slam
