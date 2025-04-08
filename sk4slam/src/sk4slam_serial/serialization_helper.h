#pragma once

// #include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/xml_oarchive.hpp>
// #include <boost/archive/xml_iarchive.hpp>

// #include <boost/archive/binary_iarchive.hpp>
// #include <boost/archive/binary_oarchive.hpp>

#include <Eigen/Core>

#include "sk4slam_basic/serializable.h"
#include "sk4slam_basic/unique_id.h"
#include "sk4slam_camera/camera_model_factory.h"
#include "sk4slam_pose/pose.h"
#include "sk4slam_pose/tf.h"

namespace sk4slam {

template <>
class SerializableWrapper<UniqueId>
    : public UniqueId, public Serializable<SerializableWrapper<UniqueId>> {
  friend class Serializable<SerializableWrapper<UniqueId>>;
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    std::string id_str;
    if constexpr (Archive::is_saving::value) {
      id_str = this->hexString();
    }
    serialization::serialize(ar, id_str, nullptr, version);
    if constexpr (!Archive::is_saving::value) {
      this->fromHexString(id_str);
    }
  }
};

template <
    typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
    int _MaxCols>
class SerializableWrapper<
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
    : public Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>,
      public Serializable<SerializableWrapper<
          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>> {
  friend class Serializable<SerializableWrapper<
      Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>>;
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    static constexpr bool kIsDynamicSize =
        (_Rows == Eigen::Dynamic || _Cols == Eigen::Dynamic);
    if (Archive::is_saving::value) {
      if constexpr (_Rows == 1 || _Cols == 1) {
        std::vector<_Scalar> data(this->data(), this->data() + this->size());
        serialization::serialize(ar, data, nullptr, version);
      } else {
        std::vector<std::vector<_Scalar>> data;
        data.resize(this->rows());
        for (int i = 0; i < this->rows(); ++i) {
          auto& row_i_data = data[i];
          row_i_data.resize(this->cols());
          for (int j = 0; j < this->cols(); ++j) {
            row_i_data[j] = (*this)(i, j);
          }
        }
        serialization::serialize(ar, data, nullptr, version);
      }
    } else {
      if constexpr (_Rows == 1 || _Cols == 1) {
        std::vector<_Scalar> data;
        serialization::serialize(ar, data, nullptr, version);
        if (data.empty()) {
          return;
        }
        if constexpr (kIsDynamicSize) {
          this->resize(data.size());
        }
        std::copy(data.begin(), data.end(), this->data());
      } else {
        std::vector<std::vector<_Scalar>> data;
        serialization::serialize(ar, data, nullptr, version);
        if (data.empty()) {
          return;
        }
        if constexpr (kIsDynamicSize) {
          this->resize(data.size(), data[0].size());
        }
        for (int i = 0; i < this->rows(); ++i) {
          auto& row_i_data = data[i];
          for (int j = 0; j < this->cols(); ++j) {
            (*this)(i, j) = row_i_data[j];
          }
        }
      }
    }
  }
};

/// @brief Wrapper for serialization of Rot3d (SO3d)
template <>
class SerializableWrapper<Rot3d>
    : public Rot3d, public Serializable<SerializableWrapper<Rot3d>> {
  friend class Serializable<SerializableWrapper<Rot3d>>;

  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    Eigen::Quaterniond q;
    if (Archive::is_saving::value) {
      q = Eigen::Quaterniond(this->matrix());
    }

    serialization::serialize(ar, q.coeffs(), nullptr, version);

    if (Archive::is_loading::value) {
      static_cast<Rot3d&>(*this) = Rot3d(q.normalized().toRotationMatrix());
    }
  }
};

/// @brief Wrapper for serialization of Pose3d (SE3d)
template <>
class SerializableWrapper<Pose3d>
    : public Pose3d, public Serializable<SerializableWrapper<Pose3d>> {
  friend class Serializable<SerializableWrapper<Pose3d>>;

  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    if (version == 0) {
      // Serialize as a 7x1 vector [qx, qy, qz, qw, tx, ty, tz]
      Vector<7> pose_vec;

      if (Archive::is_saving::value) {
        Eigen::Quaterniond q(this->rotation().matrix());
        pose_vec << q.coeffs(), this->translation();
      }

      serialization::serialize(ar, pose_vec, nullptr, version);

      if (Archive::is_loading::value) {
        Eigen::Quaterniond q(
            pose_vec[3], pose_vec[0], pose_vec[1], pose_vec[2]);
        this->rotation() = q.normalized().toRotationMatrix();
        this->translation() = pose_vec.tail<3>();
      }
    } else if (version == 1) {
      // Serialize as a 4x4 matrix
      Eigen::Matrix4d pose_matrix;

      if (Archive::is_saving::value) {
        pose_matrix = *this;
      }

      serialization::serialize(ar, pose_matrix, nullptr, version);

      if (Archive::is_loading::value) {
        *static_cast<Pose3d*>(this) = Pose3d(pose_matrix);
      }
    } else {
      throw std::runtime_error("Unsupported version for Pose3d serialization");
    }
  }
};

#define SERIALIZE_POSE_AS_MATRIX(archive, data) SERIALIZE_V(archive, data, 1)

template <>
class SerializableWrapper<TimedPose3d>
    : public TimedPose3d,
      public Serializable<SerializableWrapper<TimedPose3d>> {
  friend class Serializable<SerializableWrapper<TimedPose3d>>;

  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    TimedPose3d::Timestamp& timestamp = this->timestamp;
    Pose3d& pose = this->pose;

    SERIALIZE(ar, timestamp);
    SERIALIZE(ar, pose);
  }
};

template <>
class SerializableWrapper<Pose3dBuf>
    : public Pose3dBuf, public Serializable<SerializableWrapper<Pose3dBuf>> {
  friend class Serializable<SerializableWrapper<Pose3dBuf>>;
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    serialization::serialize(ar, this->posebuf_, nullptr, version);
  }
};

template <typename _FrameId, typename _Timestamp>
struct SerializableWrapper<TfTransform_<_FrameId, _Timestamp>>
    : public TfTransform_<_FrameId, _Timestamp>,
      public Serializable<
          SerializableWrapper<TfTransform_<_FrameId, _Timestamp>>> {
  friend class Serializable<
      SerializableWrapper<TfTransform_<_FrameId, _Timestamp>>>;
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    auto& frame_id = this->frame_id;
    auto& child_frame_id = this->child_frame_id;
    auto& is_dynamic = this->is_dynamic;
    auto& pose = this->pose;
    auto& pose_buffer = this->pose_buffer;

    SERIALIZE(ar, frame_id);
    SERIALIZE(ar, child_frame_id);
    SERIALIZE(ar, is_dynamic);
    if (!is_dynamic) {
      SERIALIZE(ar, pose);
    } else {
      SERIALIZE(ar, pose_buffer);
    }
  }
};

template <typename _FrameId, typename _Timestamp>
class SerializableWrapper<Tf_<_FrameId, _Timestamp>>
    : public Tf_<_FrameId, _Timestamp>,
      public Serializable<SerializableWrapper<Tf_<_FrameId, _Timestamp>>> {
  friend class Serializable<SerializableWrapper<Tf_<_FrameId, _Timestamp>>>;
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    auto transforms = this->getAllSharedTfTransforms();
    SERIALIZE(ar, transforms);
    if (Archive::is_loading::value) {
      this->insertTfTransforms(transforms);
    }
  }
};

}  // namespace sk4slam

namespace boost {
namespace serialization {

/// @brief  Boost serialization for Eigen::Matrix
template <
    class Archive, typename _Scalar, int _Rows, int _Cols, int _Options,
    int _MaxRows, int _MaxCols>
void serialize(
    Archive& ar,
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix,
    const unsigned int version) {
  static constexpr bool is_dynamic_size =
      (_MaxRows != Eigen::Dynamic || _MaxCols != Eigen::Dynamic);
  if constexpr (is_dynamic_size) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    ar& rows& cols;
    if (Archive::is_loading::value) {
      matrix.resize(rows, cols);
    }
  }

  std::vector<_Scalar> data(
      matrix.data(), matrix.data() + matrix.rows() * matrix.cols());
  ar& data;

  if (Archive::is_loading::value) {
    std::copy(data.begin(), data.end(), matrix.data());
  }
}

}  // namespace serialization
}  // namespace boost
