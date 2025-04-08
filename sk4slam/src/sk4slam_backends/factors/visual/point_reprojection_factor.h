#pragma once

#include "sk4slam_backends/factor_base.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_camera/camera_model.h"
#include "sk4slam_pose/pose.h"

namespace sk4slam {

/// @brief This factor calculates the reprojection error of a 3D point.
///
/// The factor assumes that the camera is fixed on a rigid body whose pose
/// needs to be estimated. The body's pose is expressed in a "map" frame, and
/// the system may contain multiple "maps" (for instance, when multiple robots
/// explore the same environment and share their maps, or when a large map is
/// divided into submaps). Different maps can be linked by their poses in the
/// world frame, which is unique and consistently defined.
///
/// Each 3D point is associated with a single map but can be observed
/// from keyframes in different maps. The 3D point can be represented
/// either in the map frame or in the camera frame of the point's
/// "reference keyframe" (when the point is "anchored" to a keyframe).
/// Refer to the @ref PointRepresentation enum for more details.
///
class PointReprojectionFactor : public FactorBase<PointReprojectionFactor> {
  using Base = FactorBase<PointReprojectionFactor>;

 public:
  using Base::JacobianMatrixXd;
  static constexpr int kResidualDim = 2;

  /// @brief The indices of the variables involved in the factor.
  enum VariableIdx {
    MAP_POSE_IN_WORLD = 0,  ///< Pose of the map containing the observer
                            ///< keyframe, expressed in the world frame.
                            ///< If the variable key for this index is
                            ///< set to null, the world frame is assumed
                            ///< to be the same as the map frame.
    BODY_POSE_IN_MAP,  ///< Pose of the observer keyframe (body), expressed in
                       ///< the corresponding map frame.
    CAM_POSE_IN_BODY,  ///< Extrinsics of the observer camera, expressed in its
                       ///< body frame.
    MAPr_POSE_IN_WORLD,  ///< Pose of the map containing the reference keyframe,
                         ///< expressed in the world frame. If the variable
                         ///< key for this index is set to null, then it's
                         ///< assumed that the reference keyframe is in the same
                         ///< map with the observer keyframe.
    BODYr_POSE_IN_MAPr,  ///< Pose of the reference keyframe (body), expressed
                         ///< in the corresponding map frame. If the variable
                         ///< key for this index is set to null, then it's
                         ///< assumed that the reference keyframe is the same as
                         ///< the observer keyframe.
    CAMr_POSE_IN_BODYr,  ///< Extrinsics of the reference camera, expressed in
                         ///< its body frame. If the variable key for this
                         ///< index is set to null, then it's assumed that the
                         ///< reference camera is the same as the observer
                         ///< camera.
    POINT,               ///< 3D point in map or reference camera frame.
    CAM_INTRINSICS,      ///< Intrinsic parameters of the observer camera.
    NUM_VARIABLES        ///< Number of variables.
  };

  static constexpr int kNumVariables =
      static_cast<int>(VariableIdx::NUM_VARIABLES);

  /// @brief Define the types of the variables for each index.
  DECLARE_VARIABLE_TYPES(
      Pose3d,    // MAP_POSE_IN_WORLD
      Pose3d,    // BODY_POSE_IN_MAP
      Pose3d,    // CAM_POSE_IN_BODY
      Pose3d,    // MAPr_POSE_IN_WORLD
      Pose3d,    // BODYr_POSE_IN_MAPr
      Pose3d,    // CAMr_POSE_IN_BODYr
      Vector3d,  // POINT
      VectorXd   // CAM_INTRINSICS
  )

  /// @brief Enum to define how the 3D point is represented.
  enum PointRepresentation {
    MAP_XYZ,     ///< 3D coordinates (X, Y, Z) in the map frame.
    CAMERA_xyZ,  ///< Z-normalized coordinates (x = X/Z, y = Y/Z, Z) in the
                 ///< camera frame.
    CAMERA_INVERSE_DEPTH  ///< Z-normalized, depth-inversed coordinates (x =
                          ///< 1/Z, y = 1/Z, 1/Z) in the camera frame.
  };

 public:
  /// @brief Constructor for the reprojection factor.
  /// @param measurement 2D measurement in the image plane.
  /// @param point_representation Representation of the 3D point.
  /// @param cam Pointer to the camera model.
  /// @param variable_keys Keys of the variables used in the factor.
  PointReprojectionFactor(
      const Vector2d& measurement,
      const PointRepresentation point_representation,
      const CameraModelInterface* cam,
      const std::vector<VariableKey>& variable_keys)
      : Base(resolveAliasKeys(variable_keys, point_representation)),
        point_representation_(point_representation),
        cam_(cam),
        measurement_(measurement) {}

  /// @brief Getter for the 2D measurement.
  const Vector2d& measurement() const {
    return measurement_;
  }

  /// @brief Evaluate the reprojection error and optionally compute Jacobians.
  /// @param T_W_M Pose of the map in the world frame.
  /// @param T_M_B Pose of the body in the map frame.
  /// @param T_B_C Pose of the camera in the body frame.
  /// @param T_W_Mr Pose of the reference map in the world frame.
  /// @param T_Mr_Br Pose of the reference body in the reference map frame.
  /// @param T_Br_Cr Pose of the reference camera in the reference body frame.
  /// @param p 3D point.
  /// @param cam_intrinsics Camera intrinsic parameters.
  /// @param j_T_W_M Optional Jacobian for T_W_M.
  /// @param j_T_M_B Optional Jacobian for T_M_B.
  /// @param j_T_B_C Optional Jacobian for T_B_C.
  /// @param j_T_W_Mr Optional Jacobian for T_W_Mr.
  /// @param j_T_Mr_Br Optional Jacobian for T_Mr_Br.
  /// @param j_T_Br_Cr Optional Jacobian for T_Br_Cr.
  /// @param j_p Optional Jacobian for the 3D point.
  /// @param j_cam_intrinsics Optional Jacobian for the camera intrinsics.
  /// @return The reprojection error as a 2D vector.
  virtual VectorXd evaluateError(
      const Pose3d& T_W_M, const Pose3d& T_M_B, const Pose3d& T_B_C,
      const Pose3d& T_W_Mr, const Pose3d& T_Mr_Br, const Pose3d& T_Br_Cr,
      const Vector3d& p, const VectorXd& cam_intrinsics,
      JacobianMatrixXd* j_T_W_M = nullptr, JacobianMatrixXd* j_T_M_B = nullptr,
      JacobianMatrixXd* j_T_B_C = nullptr, JacobianMatrixXd* j_T_W_Mr = nullptr,
      JacobianMatrixXd* j_T_Mr_Br = nullptr,
      JacobianMatrixXd* j_T_Br_Cr = nullptr, JacobianMatrixXd* j_p = nullptr,
      JacobianMatrixXd* j_cam_intrinsics = nullptr) const;

 protected:
  const VariableKey& key(VariableIdx input) const {
    return Base::getVariableKey(static_cast<int>(input));
  }

  static std::vector<VariableKey> resolveAliasKeys(
      const std::vector<VariableKey>& variable_keys,
      const PointRepresentation point_representation);

  VariableKey getAliasKey(VariableIdx input) const;

  Vector3d getPointInWorld(
      const Pose3d& T_W_Mr, const Pose3d& T_Mr_Br, const Pose3d& T_Br_Cr,
      const Vector3d& p) const;

 private:
  PointRepresentation
      point_representation_;         ///< Representation of the 3D point.
  const CameraModelInterface* cam_;  ///< Pointer to the camera model.
  Vector2d measurement_;             ///< 2D measurement in the image plane.
};

/// @brief  Create a reprojection factor for typical monocular VIO applications.
/// @tparam landmark_representation
/// @param measurement
/// @param cam
/// @param imu_pose_key
/// @param landmark_key
/// @param cam_extrinsics_key
/// @param cam_intrinsics_key
template <
    PointReprojectionFactor::PointRepresentation landmark_representation =
        PointReprojectionFactor::MAP_XYZ>
PointReprojectionFactor CreateMonoVioReprojectionFactor(
    const Vector2d& measurement, const CameraModelInterface* cam,
    VariableKey imu_pose_key, VariableKey landmark_key,
    VariableKey cam_extrinsics_key = null_variable,
    VariableKey cam_intrinsics_key = null_variable,
    VariableKey ref_imu_pose_key = null_variable) {
  return PointReprojectionFactor(
      measurement, landmark_representation, cam,
      {
          null_variable,       // MAP_POSE_IN_WORLD
          imu_pose_key,        // BODY_POSE_IN_MAP
          cam_extrinsics_key,  // CAM_POSE_IN_BODY
          null_variable,       // MAPr_POSE_IN_WORLD
          ref_imu_pose_key,    // BODYr_POSE_IN_MAPr
          null_variable,       // CAMr_POSE_IN_BODYr
          landmark_key,        // POINT
          cam_intrinsics_key,  // CAM_INTRINSICS
      });
}

}  // namespace sk4slam
