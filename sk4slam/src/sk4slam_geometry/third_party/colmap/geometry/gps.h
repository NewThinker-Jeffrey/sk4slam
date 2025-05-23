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

#include "sk4slam_geometry/third_party/colmap/util/types.h"

namespace sk4slam_colmap {

// Transform ellipsoidal GPS coordinates to Cartesian GPS coordinate
// representation and vice versa.
class GPSTransform {
 public:
  enum ELLIPSOID { GRS80, WGS84 };

  explicit GPSTransform(int ellipsoid = GRS80);

  std::vector<Eigen::Vector3d> EllToXYZ(
      const std::vector<Eigen::Vector3d>& ell) const;

  std::vector<Eigen::Vector3d> XYZToEll(
      const std::vector<Eigen::Vector3d>& xyz) const;

  // Convert GPS (lat / lon / alt) to ENU coords. with lat0 and lon0
  // defining the origin of the ENU frame
  std::vector<Eigen::Vector3d> EllToENU(
      const std::vector<Eigen::Vector3d>& ell, double lat0, double lon0) const;

  std::vector<Eigen::Vector3d> XYZToENU(
      const std::vector<Eigen::Vector3d>& xyz, double lat0, double lon0) const;

  std::vector<Eigen::Vector3d> ENUToEll(
      const std::vector<Eigen::Vector3d>& enu, double lat0, double lon0,
      double alt0) const;

  std::vector<Eigen::Vector3d> ENUToXYZ(
      const std::vector<Eigen::Vector3d>& enu, double lat0, double lon0,
      double alt0) const;

 private:
  // Semimajor axis.
  double a_;
  // Semiminor axis.
  double b_;
  // Flattening.
  double f_;
  // Numerical eccentricity.
  double e2_;
};

}  // namespace sk4slam_colmap
