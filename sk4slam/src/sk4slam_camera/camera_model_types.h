#pragma once

#include <Eigen/Core>
#include <memory>

#include "sk4slam_basic/logging.h"
#include "sk4slam_camera/camera_model.h"
#include "sk4slam_camera/equidistant.h"
#include "sk4slam_camera/fov.h"
#include "sk4slam_camera/pinhole.h"
#include "sk4slam_camera/radtan.h"

namespace sk4slam {

enum class CameraModelType : uint32_t {
  /// @brief Unknown camera model
  UNKNOWN,

  PINHOLE,  // RadTan<0>

  /// @brief The Radial-Tangential (Brown Conrady) distortion model (with 4
  /// distortion parameters)
  RADTAN4,  // RadTan<4>

  /// @brief The Radial-Tangential (Brown Conrady) distortion model (with 5
  /// distortion parameters)
  RADTAN5,  // RadTan<5>

  // /// @brief The Modified Brown Conrady distortion model (used by RealSense
  // /// cameras)
  // RADTAN5_MODIFIED,  // not implemented

  /// @brief The Equidistant (KannalaBrandt8) distortion model
  EQUIDISTANT,  // Equidistant

  /// @brief The FOV distortion model.
  /// This model has only one distortion parameter, which is the field of view
  /// of the corresponding ideal fish-eye lens, so we called it the FOV model.
  /// See:
  /// https://www.researchgate.net/publication/29638148_Straight_Lines_Have_to_Be_Straight_Automatic_Calibration_and_Removal_of_Distortion_from_Scenes_of_Structured_Environments
  FOV,  // Fov

  /// @brief The model expressed as a look-up table (LUT)
  LUT,  // LUT

  NUM_TYPES
};

}  // namespace sk4slam

#define APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS_EXCEPT_LUT(SUB_MACRO) \
  SUB_MACRO(Pinhole, PINHOLE)                                          \
  SUB_MACRO(RadTan4, RADTAN4)                                          \
  SUB_MACRO(RadTan5, RADTAN5)                                          \
  SUB_MACRO(Equidistant, EQUIDISTANT)                                  \
  SUB_MACRO(Fov, FOV)

#define APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(SUB_MACRO)      \
  APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS_EXCEPT_LUT(SUB_MACRO) \
  SUB_MACRO(LutCameraModel, LUT)

namespace sk4slam {

inline std::string CameraModelTypeToStr(CameraModelType type) {
#define SPECIALIZE_SK4SLAM_CAMERA_MODEL_TO_STR(Class, Type) \
  {CameraModelType::Type, #Type},
  static const std::unordered_map<CameraModelType, std::string>
      cam_model_to_str = {APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(
          SPECIALIZE_SK4SLAM_CAMERA_MODEL_TO_STR){
          CameraModelType::UNKNOWN, "UNKNOWN"}};
  auto it = cam_model_to_str.find(type);
  return it != cam_model_to_str.end() ? it->second : "UNKNOWN";
}

inline CameraModelType CameraModelTypeFromStr(const std::string& str) {
#define SPECIALIZE_SK4SLAM_CAMERA_MODEL_FROM_STR(Class, Type) \
  {#Type, CameraModelType::Type},
  static const std::unordered_map<std::string, CameraModelType>
      cam_model_from_str = {APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(
          SPECIALIZE_SK4SLAM_CAMERA_MODEL_FROM_STR){
          "UNKNOWN", CameraModelType::UNKNOWN}};
  auto it = cam_model_from_str.find(str);
  return it != cam_model_from_str.end() ? it->second : CameraModelType::UNKNOWN;
}

/// @brief  Get the camera_model class from the camera_model type
template <CameraModelType _cam_model>
struct _GetCameraModelClass {
  using Class = void;
};

#define SPECIALIZE_SK4SLAM_GET_CAMERA_MODEL_CLASS(_Class, Type) \
  template <>                                                   \
  struct _GetCameraModelClass<CameraModelType::Type> {          \
    using Class = _Class;                                       \
  };

APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(
    SPECIALIZE_SK4SLAM_GET_CAMERA_MODEL_CLASS)

template <CameraModelType _cam_model>
using GetCameraModelClass = typename _GetCameraModelClass<_cam_model>::Class;

/// @brief Get the camera_model type from the camera_model class
template <typename CameraModelClass>
struct _GetCameraModelType {
  static constexpr CameraModelType type = CameraModelType::UNKNOWN;
};

#define SPECIALIZE_SK4SLAM_GET_CAMERA_MODEL_TYPE(Class, Type)      \
  template <>                                                      \
  struct _GetCameraModelType<Class> {                              \
    static constexpr CameraModelType type = CameraModelType::Type; \
  };

APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(
    SPECIALIZE_SK4SLAM_GET_CAMERA_MODEL_TYPE)

template <typename CameraModelClass>
inline constexpr CameraModelType GetCameraModelType =
    _GetCameraModelType<CameraModelClass>::type;

/// @brief  Get the camera_model type from the camera_model instance
inline CameraModelType getCameraModelType(
    const CameraModelInterface* camera_model) {
  // clang-format off
#define SPECIALIZE_SK4SLAM_GET_CAMERA_MODEL_TYPE2(Class, Type) \
  else if (dynamic_cast<const Class*>(camera_model))        \
  {                                                   \
    return CameraModelType::Type;                          \
  }
  if (false) {}  // NOLINT
  APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(SPECIALIZE_SK4SLAM_GET_CAMERA_MODEL_TYPE2)  // NOLINT
  else {  // NOLINT
    LOGE("Unknown camera_model type!");
    return CameraModelType::UNKNOWN;
  }
  // clang-format on
}

/// @brief Executes different operations based on the specified camera model
/// type.
///
/// This function dispatches the operation (`op`) to the appropriate handler
/// for the given `CameraModelType`. Depending on the `exclude_lut` parameter,
/// the LUT camera model type can be excluded from processing, as it is not
/// considered a regular camera model. Unsupported camera model types will
/// result in an exception.
///
/// @tparam exclude_lut A boolean flag indicating whether the LUT camera model
/// should be excluded.
/// @tparam Operator The type of the operation to be applied.
/// @tparam Args The types of additional arguments passed to the operation.
/// @param op The operation to execute, templated to allow flexibility.
/// @param cam_model The camera model type that determines the operation to
/// execute.
/// @param args Additional arguments to be forwarded to the operation.
/// @return The result of the operation for the specific camera model type.
/// @throws std::runtime_error If the provided camera model type is invalid.
template <bool exclude_lut, typename Operator, typename... Args>
decltype(auto) processByCameraModelType(
    Operator&& op, CameraModelType cam_model, Args&&... args) {
#define CASE_FOR_SK4SLAM_PROCESS_BY_CAMERA_MODEL(Class, Type) \
  case CameraModelType::Type:                                 \
    return op.template operator()<Class>(std::forward<Args>(args)...);

  if constexpr (exclude_lut) {
    switch (cam_model) {
      APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS_EXCEPT_LUT(
          CASE_FOR_SK4SLAM_PROCESS_BY_CAMERA_MODEL)
      default:
        throw std::runtime_error(
            "Invalid camera_model type" +
            toStr(static_cast<uint32_t>(cam_model)) + "!");
    }
  } else {
    switch (cam_model) {
      APPLY_MACRO_ON_ALL_SK4SLAM_CAMERA_MODELS(
          CASE_FOR_SK4SLAM_PROCESS_BY_CAMERA_MODEL)
      default:
        throw std::runtime_error(
            "Invalid camera_model type" +
            toStr(static_cast<uint32_t>(cam_model)) + "!");
    }
  }
}

/// @brief Iterates over all camera model types and applies a specified
/// operation.
///
/// This function attempts to execute an operation (`op`) for each camera model
/// type. If the operation returns `true` for any camera model type, the
/// iteration stops, and the corresponding camera model type is returned. If no
/// camera model type requests termination, the function returns
/// `CameraModelType::UNKNOWN`. The `exclude_lut` parameter allows excluding the
/// LUT camera model type from the iteration.
///
/// @tparam exclude_lut A boolean flag indicating whether the LUT camera model
/// should be excluded.
/// @tparam Operator The type of the operation to be applied.
/// @tparam Args The types of additional arguments passed to the operation.
/// @param op The operation to execute for each camera model type.
/// @param args Additional arguments to be forwarded to the operation.
/// @return The `CameraModelType` that requests termination. If no termination
/// is requested, returns `CameraModelType::UNKNOWN`.
template <bool exclude_lut, typename Operator, typename... Args>
CameraModelType foreachCameraModelType(Operator&& op, const Args&... args) {
  static constexpr uint32_t num_types =
      static_cast<uint32_t>(CameraModelType::NUM_TYPES);
  static constexpr uint32_t lut_index =
      static_cast<uint32_t>(CameraModelType::LUT);

  for (uint32_t i = 1; i < num_types; ++i) {
    if constexpr (exclude_lut) {
      if (i == lut_index)
        continue;
    }
    CameraModelType op_type = static_cast<CameraModelType>(i);
    if (processByCameraModelType<exclude_lut>(op, op_type, args...)) {
      return op_type;
    }
  }
  return CameraModelType::UNKNOWN;
}

}  // namespace sk4slam
