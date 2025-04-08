#pragma once

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

template <typename Scalar, int dim>
using Isometry = Eigen::Transform<Scalar, dim, Eigen::Isometry>;

namespace geometry_private {
DEFINE_HAS_MEMBER_VARIABLE(z)
};

template <
    typename FirstPointType, typename SecondPointType,
    typename FirstCvPointType, typename SecondCvPointType>
void convertCvPointPairsToEigen(
    const std::vector<FirstCvPointType>& cv_Xs,
    const std::vector<SecondCvPointType>& cv_Xprimes,
    std::vector<std::pair<FirstPointType, SecondPointType>>* point_pairs) {
  ASSERT(cv_Xs.size() == cv_Xprimes.size());
  point_pairs->clear();
  point_pairs->resize(cv_Xs.size());

  // for (size_t i=0; i<cv_Xs.size(); ++i) {
  //   auto& pair = point_pairs->at(i);
  //   cv2eigen(cv_Xs[i], pair.first);
  //   cv2eigen(cv_Xprimes[i], pair.second);
  // }

  for (size_t i = 0; i < cv_Xs.size(); ++i) {
    auto& pair = point_pairs->at(i);
    using geometry_private::HasMemberVariable_z;
    if constexpr (HasMemberVariable_z<FirstCvPointType>) {
      pair.first = FirstPointType(cv_Xs[i].x, cv_Xs[i].y, cv_Xs[i].z);
    } else {
      pair.first = FirstPointType(cv_Xs[i].x, cv_Xs[i].y);
    }

    if constexpr (HasMemberVariable_z<SecondCvPointType>) {
      pair.second =
          SecondPointType(cv_Xprimes[i].x, cv_Xprimes[i].y, cv_Xprimes[i].z);
    } else {
      pair.second = SecondPointType(cv_Xprimes[i].x, cv_Xprimes[i].y);
    }
  }
}

template <typename FirstPointType, typename SecondPointType>
void convertPointPairsToXsAndXprimes(
    const std::vector<std::pair<FirstPointType, SecondPointType>>& point_pairs,
    std::vector<FirstPointType>* Xs, std::vector<SecondPointType>* Xprimes) {
  Xs->clear();
  Xprimes->clear();
  Xs->reserve(point_pairs.size());
  Xprimes->reserve(point_pairs.size());

  for (const auto& point_pair : point_pairs) {
    Xs->push_back(point_pair.first);
    Xprimes->push_back(point_pair.second);
  }
}

template <typename FirstPointType, typename SecondPointType>
void convertSelectedPointPairsToXsAndXprimes(
    const std::vector<size_t>& selected_indices,
    const std::vector<std::pair<FirstPointType, SecondPointType>>& point_pairs,
    std::vector<FirstPointType>* Xs, std::vector<SecondPointType>* Xprimes) {
  Xs->clear();
  Xprimes->clear();
  Xs->reserve(selected_indices.size());
  Xprimes->reserve(selected_indices.size());

  for (size_t index : selected_indices) {
    const auto& point_pair = point_pairs[index];
    Xs->push_back(point_pair.first);
    Xprimes->push_back(point_pair.second);
  }
}

template <
    typename FirstPointType, typename SecondPointType,
    typename FirstHomoPointType, typename SecondHomoPointType>
void convertSelectedPointPairsToHomogeneousXsAndXprimes(
    const std::vector<size_t>& selected_indices,
    const std::vector<std::pair<FirstPointType, SecondPointType>>& point_pairs,
    std::vector<FirstHomoPointType>* Xs,
    std::vector<SecondHomoPointType>* Xprimes) {
  Xs->clear();
  Xprimes->clear();
  Xs->reserve(selected_indices.size());
  Xprimes->reserve(selected_indices.size());

  for (size_t index : selected_indices) {
    const auto& point_pair = point_pairs[index];
    Xs->push_back(point_pair.first.homogeneous());
    Xprimes->push_back(point_pair.second.homogeneous());
  }
}

template <typename ScalarType>
Isometry<ScalarType, 3> convertMatrix3x4ToIsometry3(
    const Eigen::Matrix<ScalarType, 3, 4>& matrix) {
  Isometry<ScalarType, 3> isometry;
  isometry.linear() = matrix.template block<3, 3>(0, 0);
  isometry.translation() = matrix.template block<3, 1>(0, 3);
  return isometry;
}

template <typename ScalarType>
Eigen::Matrix<ScalarType, 3, 4> convertIsometry3ToMatrix3x4(
    const Isometry<ScalarType, 3>& isometry) {
  Eigen::Matrix<ScalarType, 3, 4> matrix;
  matrix.template block<3, 3>(0, 0) = isometry.linear();
  matrix.template block<3, 1>(0, 3) = isometry.translation();
  return matrix;
}

template <typename ScalarType>
std::vector<Isometry<ScalarType, 3>> convertMatrix3x4ToIsometry3(
    const std::vector<Eigen::Matrix<ScalarType, 3, 4>>& matricies) {
  std::vector<Isometry<ScalarType, 3>> isometries;
  isometries.reserve(matricies.size());
  for (const auto& matrix : matricies) {
    isometries.push_back(convertMatrix3x4ToIsometry3(matrix));
  }
  return isometries;
}

template <typename ScalarType>
std::vector<Eigen::Matrix<ScalarType, 3, 4>> convertIsometry3ToMatrix3x4(
    const std::vector<Isometry<ScalarType, 3>>& isometries) {
  std::vector<Isometry<ScalarType, 3>> matricies;
  matricies.reserve(isometries.size());
  for (const auto& isometry : isometries) {
    matricies.push_back(convertIsometry3ToMatrix3x4(isometry));
  }
  return matricies;
}

template <typename ScalarType, int _dim>
inline std::vector<Eigen::Matrix<ScalarType, _dim + 1, 1>> convertToHomogeneous(
    const std::vector<Eigen::Matrix<ScalarType, _dim, 1>>& points) {
  std::vector<Eigen::Matrix<ScalarType, _dim + 1, 1>> homogeneous_points;
  homogeneous_points.reserve(points.size());
  for (const auto& point : points) {
    homogeneous_points.push_back(point.homogeneous());
  }
  return homogeneous_points;
}

}  // namespace sk4slam
