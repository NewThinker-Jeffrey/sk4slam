// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_geometry/third_party/colmap/geometry/rigid3.h"
#include "sk4slam_geometry/third_party/colmap/util/types.h"

namespace sk4slam_colmap {

// Solver for the Generalized P3P problem (NP3P or GP3P), based on:
//
//      Lee, Gim Hee, et al. "Minimal solutions for pose estimation of a
//      multi-camera system." Robotics Research. Springer International
//      Publishing, 2016. 521-538.
//
// This class is based on an original implementation by Federico Camposeco.
class GP3PEstimator {
 public:
  // The generalized image observations, which is composed of the relative pose
  // of a camera in the generalized camera and a ray in the camera frame.
  struct X_t {
    Rigid3d cam_from_rig;
    Eigen::Vector3d ray_in_cam;
  };

  // The observed 3D feature points in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The estimated rig_from_world pose of the generalized camera.
  typedef Rigid3d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  enum class ResidualType {
    CosineDistance,
    ReprojectionError,
  };

  // Whether to compute the cosine similarity or the reprojection error.
  // [WARNING] The reprojection error being in normalized coordinates,
  // the unique error threshold of RANSAC corresponds to different pixel values
  // in the different cameras of the rig if they have different intrinsics.
  ResidualType residual_type = ResidualType::CosineDistance;

  // Estimate the most probable solution of the GP3P problem from a set of
  // three 2D-3D point correspondences.
  static std::vector<M_t> Estimate(
      const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D);

  // Calculate the squared cosine distance error between the rays given a set of
  // 2D-3D point correspondences and the rig pose of the generalized camera.
  void Residuals(
      const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D,
      const M_t& rig_from_world, std::vector<double>* residuals);
};

}  // namespace sk4slam_colmap
