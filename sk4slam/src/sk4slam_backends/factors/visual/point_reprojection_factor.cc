
#include "sk4slam_backends/factors/visual/point_reprojection_factor.h"

#include "sk4slam_basic/string_helper.h"

namespace sk4slam {

VectorXd PointReprojectionFactor::evaluateError(
    const Pose3d& T_W_M, const Pose3d& T_M_B, const Pose3d& T_B_C,
    const Pose3d& T_W_Mr, const Pose3d& T_Mr_Br, const Pose3d& T_Br_Cr,
    const Vector3d& p, const VectorXd& cam_intrinsics,
    JacobianMatrixXd* j_T_W_M, JacobianMatrixXd* j_T_M_B,
    JacobianMatrixXd* j_T_B_C, JacobianMatrixXd* j_T_W_Mr,
    JacobianMatrixXd* j_T_Mr_Br, JacobianMatrixXd* j_T_Br_Cr,
    JacobianMatrixXd* j_p, JacobianMatrixXd* j_cam_intrinsics) const {
  static const Pose3d IdentityPose3d = Pose3d::Identity();
  // Helper functions to retrieve poses.
  auto getT_W_Mr = [&]() -> const Pose3d& {
    if (key(MAPr_POSE_IN_WORLD) == null_variable) {
      if (key(MAP_POSE_IN_WORLD) != null_variable) {
        return T_W_M;
      } else {
        return IdentityPose3d;
      }
    } else {
      return T_W_Mr;
    }
  };

  auto getT_Mr_Br = [&]() -> const Pose3d& {
    if (key(BODYr_POSE_IN_MAPr) == null_variable) {
      ASSERT(key(BODY_POSE_IN_MAP) != null_variable);
      return T_M_B;
    } else {
      return T_Mr_Br;
    }
  };

  auto getT_Br_Cr = [&]() -> const Pose3d& {
    if (key(CAMr_POSE_IN_BODYr) == null_variable) {
      if (key(CAM_POSE_IN_BODY) != null_variable) {
        return T_B_C;
      } else {
        return IdentityPose3d;
      }
    } else {
      return T_Br_Cr;
    }
  };

  // Compute the 3D coordinates of the point in the world frame.
  Vector3d t_W_p = getPointInWorld(getT_W_Mr(), getT_Mr_Br(), getT_Br_Cr(), p);
  LOGA("t_W_p: %s", toStr(t_W_p.transpose()).c_str());

  // Transform the point into the camera frame.
  Pose3d T_M_W;
  Vector3d t_M_p;
  if (key(MAP_POSE_IN_WORLD) == null_variable) {
    t_M_p = t_W_p;
  } else {
    T_M_W = T_W_M.inverse();
    t_M_p = T_M_W * t_W_p;
  }
  LOGA("t_M_p: %s", toStr(t_M_p.transpose()).c_str());

  Pose3d T_B_M = T_M_B.inverse();
  Vector3d t_B_p = T_B_M * t_M_p;
  LOGA("t_B_p: %s", toStr(t_B_p.transpose()).c_str());

  Pose3d T_C_B;
  Vector3d t_C_p;  // Coordinates of the point in the camera frame.
  if (key(CAM_POSE_IN_BODY) == null_variable) {
    t_C_p = t_B_p;
    LOGA("t_C_p: %s (no extrinsic)", toStr(t_C_p.transpose()).c_str());
  } else {
    T_C_B = T_B_C.inverse();
    t_C_p = T_C_B * t_B_p;
    LOGA(
        "t_C_p: %s (extrinsic = %s)", toStr(t_C_p.transpose()).c_str(),
        toOneLineStr(T_C_B.matrix()).c_str());
  }

  // Project the point into the image, and compute the Jacobians if requested.
  Vector2d projected_pixel;
  bool project_success = false;
  CameraModelInterface::JacobianMatrix<double> proj_j_wrt_intrinsics;
  CameraModelInterface::JacobianMatrix<double, 2, 3> proj_j_wrt_point;
  double* proj_j_wrt_point_data = nullptr;
  bool pose_jacobian_required = j_T_M_B;
  if (pose_jacobian_required) {
    proj_j_wrt_point_data = proj_j_wrt_point.data();
  }
  if (key(CAM_INTRINSICS) == null_variable) {
    ASSERT(!j_cam_intrinsics);
    project_success = cam_->project3AndComputeJacobian(
        t_C_p.data(), projected_pixel.data(), proj_j_wrt_point_data);
  } else {
    double* proj_j_wrt_intrinsics_data = nullptr;
    if (j_cam_intrinsics) {
      proj_j_wrt_intrinsics.resize(2, cam_intrinsics.size());
      proj_j_wrt_intrinsics_data = proj_j_wrt_intrinsics.data();
    }
    project_success = cam_->project3AndComputeJacobiansWithExternalParameters(
        t_C_p.data(), cam_intrinsics.data(), projected_pixel.data(),
        proj_j_wrt_point.data(), proj_j_wrt_intrinsics_data);
  }

  if (!project_success) {
    // If the projection failed, return an empty vector to indicate that the
    // factor is invalid.
    return VectorXd();
  }

  // Compute the reprojection error.
  Vector2d error = projected_pixel - measurement_;
  LOGA(
      "Reprojection error: %s, projected_pixel: %s, measurement: %s",
      toStr(error.transpose()).c_str(),
      toStr(projected_pixel.transpose()).c_str(),
      toStr(measurement_.transpose()).c_str());

  // Compute Jacobians if requested.
  if (j_cam_intrinsics) {
    *j_cam_intrinsics = proj_j_wrt_intrinsics;
  }

  Eigen::Matrix<double, 2, 3> j_chain =
      proj_j_wrt_point;  // j_chain = (D error) / (D t_C_p)
  if (key(CAM_POSE_IN_BODY) != null_variable) {
    j_chain =
        j_chain * T_C_B.rotation().matrix();  // j_chain = (D error) / (D t_B_p)
  }
  if (j_T_B_C) {
    ASSERT(key(CAM_POSE_IN_BODY) != null_variable);
    const Vector3d& t = T_B_C.translation();
    j_T_B_C->leftCols<3>() = j_chain * SO3d::hat(t_B_p - t);
    j_T_B_C->rightCols<3>() = -j_chain;
  }

  j_chain =
      j_chain * T_B_M.rotation().matrix();  // j_chain = (D error) / (D t_M_p)
  if (j_T_M_B) {
    const Vector3d& t = T_M_B.translation();
    j_T_M_B->leftCols<3>() = j_chain * SO3d::hat(t_M_p - t);
    j_T_M_B->rightCols<3>() = -j_chain;
  }

  if (key(MAP_POSE_IN_WORLD) != null_variable) {
    j_chain =
        j_chain * T_M_W.rotation().matrix();  // j_chain = (D error) / (D t_W_p)
  }
  if (j_T_W_M) {
    ASSERT(key(MAP_POSE_IN_WORLD) != null_variable);
    const Vector3d& t = T_W_M.translation();
    j_T_W_M->leftCols<3>() = j_chain * SO3d::hat(t_W_p - t);
    j_T_W_M->rightCols<3>() = -j_chain;
  }

  if (j_p) {
    if (point_representation_ == PointRepresentation::MAP_XYZ) {
      if (key(MAPr_POSE_IN_WORLD) == null_variable) {
        ASSERT(key(MAP_POSE_IN_WORLD) == null_variable);
        *j_p = j_chain;
      } else {
        if (j_T_W_Mr) {
          const Vector3d& t = T_W_Mr.translation();
          j_T_W_Mr->leftCols<3>() = -j_chain * SO3d::hat(t_W_p - t);
          j_T_W_Mr->rightCols<3>() = j_chain;
        }
        *j_p = j_chain * T_W_Mr.rotation().matrix();
      }
    } else {
      throw std::runtime_error(formatStr(
          "Jacobians for PointRepresentation %d have not been implemented "
          "yet!",
          point_representation_));
    }
  }
  return error;
}

std::vector<VariableKey> PointReprojectionFactor::resolveAliasKeys(
    const std::vector<VariableKey>& variable_keys,
    const PointRepresentation point_representation) {
  std::vector<VariableKey> resolved_keys(variable_keys);
  if (resolved_keys.at(MAPr_POSE_IN_WORLD) ==
      resolved_keys.at(MAP_POSE_IN_WORLD)) {
    // Only one map is involved, so we ignore the world frame.
    resolved_keys.at(MAPr_POSE_IN_WORLD) = null_variable;
    resolved_keys.at(MAP_POSE_IN_WORLD) = null_variable;
  }
  if (resolved_keys.at(BODYr_POSE_IN_MAPr) ==
      resolved_keys.at(BODY_POSE_IN_MAP)) {
    resolved_keys.at(BODYr_POSE_IN_MAPr) = null_variable;
    ASSERT(resolved_keys.at(MAPr_POSE_IN_WORLD) == null_variable);
  }
  if (resolved_keys.at(CAMr_POSE_IN_BODYr) ==
      resolved_keys.at(CAM_POSE_IN_BODY)) {
    resolved_keys.at(CAMr_POSE_IN_BODYr) = null_variable;
  }

  LOGA("variable_keys: %s", Base::formatVariableKeys(variable_keys).c_str());
  LOGA("resolved_keys: %s", Base::formatVariableKeys(resolved_keys).c_str());

  // If the point is not anchored to keyframe, then the variable key
  // for BODYr_POSE_IN_MAPr and CAMr_POSE_IN_BODYr should be null.
  if (point_representation == PointRepresentation::MAP_XYZ) {
    // resolved_keys.at(BODYr_POSE_IN_MAPr) = null_variable;
    // resolved_keys.at(CAMr_POSE_IN_BODYr) = null_variable;

    ASSERT(resolved_keys.at(BODYr_POSE_IN_MAPr) == null_variable);
    ASSERT(resolved_keys.at(CAMr_POSE_IN_BODYr) == null_variable);
  }
  return resolved_keys;
}

VariableKey PointReprojectionFactor::getAliasKey(VariableIdx input) const {
  if (input == MAPr_POSE_IN_WORLD && key(MAPr_POSE_IN_WORLD) == null_variable) {
    return key(MAP_POSE_IN_WORLD);
  } else if (
      input == BODYr_POSE_IN_MAPr && key(BODYr_POSE_IN_MAPr) == null_variable) {
    return key(BODY_POSE_IN_MAP);
  } else if (
      input == CAMr_POSE_IN_BODYr && key(CAMr_POSE_IN_BODYr) == null_variable) {
    return key(CAM_POSE_IN_BODY);
  } else {
    return key(input);
  }
}

Vector3d PointReprojectionFactor::getPointInWorld(
    const Pose3d& T_W_Mr, const Pose3d& T_Mr_Br, const Pose3d& T_Br_Cr,
    const Vector3d& p) const {
  Vector3d t_Mr_p;
  if (point_representation_ == PointRepresentation::MAP_XYZ) {
    t_Mr_p = p;
  } else {
    Vector3d t_Cr_p;
    double z;
    if (point_representation_ == PointRepresentation::CAMERA_xyZ) {
      z = p.z();
    } else {
      ASSERT(
          point_representation_ == PointRepresentation::CAMERA_INVERSE_DEPTH);
      z = 1.0 / p.z();
    }
    t_Cr_p = Vector3d(p.x() * z, p.y() * z, z);
    Vector3d t_Br_p;
    if (getAliasKey(CAMr_POSE_IN_BODYr) == null_variable) {
      // If the extrinsic is not provided, assume it is the identity.
      t_Br_p = t_Cr_p;
    } else {
      // Apply the extrinsic (T_Br_Cr)
      t_Br_p = T_Br_Cr * t_Cr_p;
    }
    t_Mr_p = T_Mr_Br * t_Br_p;
  }
  LOGA("t_Mr_p: %s", toStr(t_Mr_p.transpose()).c_str());

  if (getAliasKey(MAPr_POSE_IN_WORLD) == null_variable) {
    // If the map pose (in world frame) is not provided, assume it is the
    // identity.
    return t_Mr_p;
  } else {
    // Apply the map pose (in world frame)
    return T_W_Mr * t_Mr_p;
  }
}

}  // namespace sk4slam
