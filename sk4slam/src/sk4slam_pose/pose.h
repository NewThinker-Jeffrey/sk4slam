#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_cpp/binary_search.h"
#include "sk4slam_cpp/deque.h"
#include "sk4slam_liegroups/SE3.h"

namespace sk4slam {

template <typename Scalar>
using Pose3 = SE3<Scalar>;

template <typename Scalar>
using Rot3 = SO3<Scalar>;

using Pose3d = Pose3<double>;
using Rot3d = Rot3<double>;

template <typename _Timestamp, typename Scalar>
struct TimedPose3_ {
  using Timestamp = _Timestamp;

  Timestamp timestamp;
  Pose3<Scalar> pose;
};

template <typename _Timestamp, typename Scalar>
class Pose3Buf_ {
 public:
  using Timestamp = _Timestamp;
  using Pose = Pose3<Scalar>;
  using TimedPose = TimedPose3_<_Timestamp, Scalar>;

  /// @brief Constructor for Pose3Buf_.
  ///
  /// @param max_buf_size Maximum buffer size. Negative indicates no limit.
  /// @note Actual buffer size may temporarily exceed the limit as old poses are
  /// only removed when the size exceeds twice the limit. This reduces frequent
  /// erase operations on the underlying vector.
  explicit Pose3Buf_(int max_buf_size = -1) : max_buf_size_(max_buf_size) {}

  /// @brief Updates the pose buffer with a new pose and timestamp.
  ///
  /// The new pose will be added only if its timestamp is strictly greater
  /// than the last timestamp in the buffer. If the buffer size exceeds
  /// twice the specified limit, old poses in the first half of the buffer
  /// will be removed to reduce frequent erase operations.
  ///
  /// @param timestamp The timestamp of the new pose.
  /// @param pose The new pose to add to the buffer.
  /// @return `true` if the update is successful, `false` if the timestamp
  ///         is not strictly greater than the last one in the buffer.
  bool update(const Timestamp& timestamp, const Pose& pose) {
    // Ensure the new timestamp is strictly greater than the last one in the
    // buffer
    if (posebuf_.empty() || posebuf_.back().timestamp < timestamp) {
      // Remove old poses if the buffer size exceeds twice the maximum limit
      if (max_buf_size_ > 0 && posebuf_.size() >= max_buf_size_ * 2) {
        // Batch erase the first half of the buffer to minimize
        // erase operations
        posebuf_.erase(posebuf_.begin(), posebuf_.begin() + max_buf_size_);
        posebuf_.trim_to_optimal();
      }

      // Add the new pose to the buffer
      posebuf_.push_back(TimedPose{timestamp, pose});

      return true;
    } else {
      LOGW(
          "Pose3Buf_: Ignored update with timestamp older than the latest "
          "pose.");
      return false;
    }
  }

  /// @brief Removes all poses in the buffer with timestamps older than the
  /// specified timestamp. The pose with equal or newer timestamp will be kept.
  void removeOlderThan(const Timestamp& timestamp) {
    auto it = std::lower_bound(
        posebuf_.begin(), posebuf_.end(), timestamp,
        [](const TimedPose& tp, const Timestamp& ts) {
          return tp.timestamp < ts;
        });
    posebuf_.erase(posebuf_.begin(), it);
  }

  const TimedPose& getLatest() const {
    ASSERT(!posebuf_.empty());
    return posebuf_.back();
  }

  const TimedPose& getOldest() const {
    ASSERT(!posebuf_.empty());
    return posebuf_.front();
  }

  std::vector<TimedPose> getAll() const {
    std::vector<TimedPose> timedposes;
    timedposes.reserve(posebuf_.size());
    for (const auto& tp : posebuf_) {
      timedposes.push_back(tp);
    }
  }

  std::vector<Timestamp> getAllPoseTimes() const {
    std::vector<Timestamp> timestamps;
    timestamps.reserve(posebuf_.size());
    for (const auto& tp : posebuf_) {
      timestamps.push_back(tp.timestamp);
    }
    return timestamps;
  }

  int size() const {
    return posebuf_.size();
  }

  bool empty() const {
    return posebuf_.empty();
  }

  /// @brief Retrieves the pose at a specified timestamp.
  ///
  /// The pose is retrieved using interpolation or extrapolation between the two
  /// nearest poses in the buffer if the exact timestamp is not found. However,
  /// if the timestamp is far from its nearest neighbors (the distance to its
  /// nearest neighbors exceeds `extrapolation_thr` when extrapolating
  /// or exceeds `interpolation_thr` when interpolating), the function will
  /// return `false`.
  ///
  /// @param timestamp The timestamp at which to retrieve the pose.
  /// @param pose[out] The pose retrieved at the specified timestamp.
  /// @param extrapolation_thr The threshold for extrapolation. If the timestamp
  /// is beyond the last pose or older than the first pose by more than this
  /// threshold, extrapolation will be skipped and `false` will be returned.
  /// Default is 0, meaning no extrapolation is allowed.
  /// @param interpolation_thr The threshold for interpolation. If the distance
  /// between the timestamp and its nearest neighbor is greater than this
  /// threshold, interpolation will be skipped and `false` will be returned.
  /// Default is -1, meaning interpolation is always allowed.
  /// @param left_neighbor[out] The left neighbor of the pose in the buffer. If
  /// interpolation or extrapolation is used, this will be the pose with the
  /// smaller timestamp. If the exact timestamp is found or there is only one
  /// pose in the buffer, this will be the same as the pose.
  /// @param right_neighbor[out] The right neighbor of the pose in the buffer.
  /// If interpolation or extrapolation is used, this will be the pose with the
  /// larger timestamp. If the exact timestamp is found or there is only one
  /// pose in the buffer, this will be the same as the pose.
  /// @return `true` if the pose is successfully retrieved, `false` otherwise.
  bool get(
      const Timestamp& timestamp, Pose* pose,
      const Timestamp& extrapolation_thr = 0,
      const Timestamp& interpolation_thr = -1,
      TimedPose* left_neighbor = nullptr,
      TimedPose* right_neighbor = nullptr) const {
    // // Test get() with multiple timestamps
    // if (!left_neighbor && !right_neighbor) {
    //   auto poses = get({timestamp}, extrapolation_thr, interpolation_thr);
    //   if (poses.size() == 1) {
    //     *pose = poses.begin()->second;
    //     return true;
    //   } else {
    //     return false;
    //   }
    // }

    if (posebuf_.empty() ||
        posebuf_.back().timestamp + extrapolation_thr < timestamp ||
        posebuf_.front().timestamp > timestamp + extrapolation_thr) {
      LOGW(
          "Pose3Buf_: attempt to get with a timestamp out of "
          "range! ignored.");
      return false;
    }
    if (posebuf_.size() == 1) {
      *pose = posebuf_.front()
                  .pose;  // When timestamp is equal to the first timestamp or
                          // within a ±extrapolation_thr range of the first
                          // timestamp, return the first pose
      if (left_neighbor) {
        *left_neighbor = posebuf_.front();
      }
      if (right_neighbor) {
        *right_neighbor = posebuf_.front();
      }
      return true;
    }
    ASSERT(posebuf_.size() >= 2);

    auto search_res = binarySearchLowerBound(
        posebuf_, timestamp, std::less<Timestamp>(),
        [](const TimedPose& tp) { return tp.timestamp; });
    ASSERT(search_res);
    int begin_idx = *search_res;
    return get(
        begin_idx, timestamp, pose, extrapolation_thr, interpolation_thr,
        left_neighbor, right_neighbor);
  }

  std::unordered_map<Timestamp, Pose> get(
      const std::vector<Timestamp>& timestamps,
      const Timestamp& extrapolation_thr = 0,
      const Timestamp& interpolation_thr = -1) const {
    if (timestamps.empty()) {
      return {};
    }
    if (posebuf_.empty()) {
      LOGW("Pose3Buf_: Empty pose buffer! ignored.");
      return {};
    }

    // Find the search range
    auto cmp = [](const TimedPose& tp) { return tp.timestamp; };
    auto [it_min, it_max] =
        std::minmax_element(timestamps.begin(), timestamps.end());
    int begin_idx = 0, end_idx = posebuf_.size();
    {
      auto r0 = binarySearchLowerBound(
          posebuf_, *it_min, std::less<Timestamp>(), cmp);
      if (r0) {
        begin_idx = *r0;
      }
      auto r1 = binarySearchUpperBound(
          posebuf_, *it_max, std::less<Timestamp>(), cmp);
      if (r1) {
        end_idx = *r1;
      }
    }
    auto it_begin = posebuf_.begin() + begin_idx;
    auto it_end = posebuf_.begin() + end_idx;

    // Iterate over the query timestamps and retrieve the poses
    std::unordered_map<Timestamp, Pose> poses;
    poses.rehash(2 * timestamps.size());
    std::unordered_map<size_t, Vector<6, Scalar>> cached_deltas;
    for (const auto& timestamp : timestamps) {
      if (poses.count(timestamp)) {
        continue;
      }
      if (posebuf_.back().timestamp + extrapolation_thr < timestamp ||
          posebuf_.front().timestamp > timestamp + extrapolation_thr) {
        LOGW(
            "Pose3Buf_: attempt to get with a timestamp out of "
            "range! ignored.");
        continue;
      }
      if (posebuf_.size() == 1) {
        poses[timestamp] =
            posebuf_.front()
                .pose;  // When timestamp is equal to the first timestamp or
                        // within a ±extrapolation_thr range of the first
                        // timestamp, return the first pose
        continue;
      }
      ASSERT(posebuf_.size() >= 2);

      auto search_res = binarySearchLowerBound(
          it_begin, it_end, timestamp, std::less<Timestamp>(), cmp);
      ASSERT(search_res);
      int begin_idx = *search_res - posebuf_.begin();
      Pose pose;
      if (get(begin_idx, timestamp, &pose, extrapolation_thr, interpolation_thr,
              nullptr, nullptr, &cached_deltas)) {
        poses[timestamp] = pose;
      }
    }
    return poses;
  }

 protected:
  bool get(
      int begin_idx, const Timestamp& timestamp, Pose* pose,
      const Timestamp& extrapolation_thr, const Timestamp& interpolation_thr,
      TimedPose* left_neighbor, TimedPose* right_neighbor,
      std::unordered_map<size_t, Vector<6, Scalar>>* cached_deltas =
          nullptr) const {
    if (begin_idx >= posebuf_.size()) {
      ASSERT(begin_idx == posebuf_.size());
      // The timestamp is beyond the last timestamp, but within a
      // ±extrapolation_thr range. Do extrapolation with the last two poses.
      interpolate(posebuf_.size() - 2, timestamp, pose, cached_deltas);
      if (left_neighbor) {
        *left_neighbor = posebuf_[posebuf_.size() - 2];
      }
      if (right_neighbor) {
        *right_neighbor = posebuf_[posebuf_.size() - 1];
      }
    } else if (posebuf_[begin_idx].timestamp == timestamp) {
      // Good luck! We found the exact timestamp
      *pose = posebuf_[begin_idx].pose;
      if (left_neighbor) {
        *left_neighbor = posebuf_[begin_idx];
      }
      if (right_neighbor) {
        *right_neighbor = posebuf_[begin_idx];
      }
    } else if (begin_idx == 0) {
      // The timestamp is older than the first timestamp, but within a
      // ±extrapolation_thr range. Do extrapolation with the first two poses.
      interpolate(0, timestamp, pose, cached_deltas);
      if (left_neighbor) {
        *left_neighbor = posebuf_[0];
      }
      if (right_neighbor) {
        *right_neighbor = posebuf_[1];
      }
    } else {
      // Interpolate between the two closest timestamps
      ASSERT(begin_idx > 0 && begin_idx < posebuf_.size());

      // First check if the timestamp is close enough to its nearest neighbors
      // if interpolation_thr is set.
      if (interpolation_thr > 0) {
        Timestamp dt0 = timestamp - posebuf_[begin_idx - 1].timestamp;
        Timestamp dt1 = posebuf_[begin_idx].timestamp - timestamp;
        if (dt0 > interpolation_thr || dt1 > interpolation_thr) {
          LOGW(
              "Pose3Buf_: attempt to get with a timestamp that is too far "
              "from the closest timestamp! target_time = %s, dt0 = %s, dt1 = "
              "%s.",
              toStr(timestamp).c_str(), toStr(dt0).c_str(), toStr(dt1).c_str());
          return false;
        }
      }

      // Do the interpolation
      interpolate(begin_idx - 1, timestamp, pose, cached_deltas);
      if (left_neighbor) {
        *left_neighbor = posebuf_[begin_idx - 1];
      }
      if (right_neighbor) {
        *right_neighbor = posebuf_[begin_idx];
      }
    }
    return true;
  }

  void interpolate(
      size_t idx0, const Timestamp& target_time, Pose* pose,
      std::unordered_map<size_t, Vector<6, Scalar>>* cached_deltas) const {
    // Note alpha may >1 or <0 when we are extrapolating
    double alpha = (target_time - posebuf_[idx0].timestamp) /
                   (posebuf_[idx0 + 1].timestamp - posebuf_[idx0].timestamp);
    using OptimizablePose = typename Pose::SeparateLeftOptimizable;
    OptimizablePose pose0(posebuf_[idx0].pose);
    if (cached_deltas) {
      auto it = cached_deltas->find(idx0);
      if (it == cached_deltas->end()) {
        OptimizablePose pose1(posebuf_[idx0 + 1].pose);
        Vector<6, Scalar> delta01 = pose1 - pose0;
        it = cached_deltas->insert(std::make_pair(idx0, delta01)).first;
      }
      const auto& delta01 = it->second;
      *pose = pose0 + alpha * delta01;
    } else {
      OptimizablePose pose1(posebuf_[idx0 + 1].pose);
      Vector<6, Scalar> delta01 = pose1 - pose0;
      *pose = pose0 + alpha * delta01;
    }
  }

 protected:
  Deque<TimedPose> posebuf_;  ///< Buffer storing timed poses.
  int max_buf_size_;  ///< Maximum buffer size. Negative indicates no limit.
                      ///< Note: Actual buffer size may temporarily exceed
                      ///< the limit as old poses are only removed when the
                      ///< size exceeds twice the limit. This reduces
                      ///< frequent erase operations on the underlying vector.
};

using TimedPose3d = TimedPose3_<double, double>;
using Pose3dBuf = Pose3Buf_<double, double>;

}  // namespace sk4slam
