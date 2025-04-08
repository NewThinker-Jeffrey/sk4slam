#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_cpp/mutex.h"
#include "sk4slam_pose/pose.h"

namespace sk4slam {

template <typename _FrameId, typename _Timestamp = double>
struct TfTransform_ {
  using FrameId = _FrameId;
  using Timestamp = _Timestamp;
  using Pose = Pose3<double>;
  using PoseBuf = Pose3Buf_<Timestamp, double>;

  FrameId frame_id;        ///< The frame id of the parent frame.
  FrameId child_frame_id;  ///< The frame id of the child frame.
  bool is_dynamic;         ///< Whether the transform is dynamic or static.
  Pose pose;  ///< Pose of the child frame in the parent frame. Only valid if
              ///< is_dynamic is false.

  bool updateDynamic(const Timestamp& time, const Pose& pose)
      EXCLUDES(pose_buffer_mutex) {
    UniqueLock lock(pose_buffer_mutex);
    return pose_buffer->update(time, pose);
  }
  bool getDynamic(const Timestamp& time, Pose* pose) const
      EXCLUDES(pose_buffer_mutex) {
    SharedLock lock(pose_buffer_mutex);
    return pose_buffer->get(time, pose);
  }
  void removeOlderThan(const Timestamp& time) EXCLUDES(pose_buffer_mutex) {
    UniqueLock lock(pose_buffer_mutex);
    pose_buffer->removeOlderThan(time);
  }

  TfTransform_() : is_dynamic(false) {}
  TfTransform_(
      const FrameId& frame_id, const FrameId& child_frame_id,
      const bool is_dynamic, const Pose& pose, std::shared_ptr<PoseBuf> buffer)
      : frame_id(frame_id),
        child_frame_id(child_frame_id),
        is_dynamic(is_dynamic),
        pose(pose),
        pose_buffer(std::move(buffer)) {
    if (is_dynamic) {
      pose_buffer = std::make_shared<PoseBuf>();
    }
  }

 protected:
  std::shared_ptr<PoseBuf> pose_buffer
      GUARDED_BY(pose_buffer_mutex);  ///< The pose buffer. Only valid if
                                      ///< is_dynamic is true.
  mutable Mutex pose_buffer_mutex;
};

template <typename _FrameId, typename _Timestamp = double>
struct TfUpdate_ {
  using FrameId = _FrameId;
  using Timestamp = _Timestamp;
  using Pose = Pose3<double>;

  FrameId frame_id;        ///< The frame id of the parent frame.
  FrameId child_frame_id;  ///< The frame id of the child frame.
  bool is_dynamic;         ///< Whether the transform is dynamic or static.
  Pose pose;  ///< Pose of the child frame in the parent frame. Only valid if
              ///< is_dynamic is false.
  Timestamp timestamp;  ///< The timestamp of the pose. Only valid if
                        ///< is_dynamic is true.
};

template <typename _FrameId>
struct TfFrameIdPair_ {
  _FrameId from_frame_id;
  _FrameId to_frame_id;
  bool operator==(const TfFrameIdPair_& other) const {
    return from_frame_id == other.from_frame_id &&
           to_frame_id == other.to_frame_id;
  }
};

template <typename _FrameId, typename _Timestamp = double>
class Tf_ {
 public:
  using Transform = TfTransform_<_FrameId, _Timestamp>;
  using TransformPtr = std::shared_ptr<Transform>;
  using ConstTransformPtr = std::shared_ptr<const Transform>;
  using FrameId = typename Transform::FrameId;
  using Timestamp = typename Transform::Timestamp;
  using Pose = typename Transform::Pose;
  using PoseBuf = typename Transform::PoseBuf;

  using TfUpdate = TfUpdate_<_FrameId, _Timestamp>;
  using TfUpdateCallback =
      std::function<void(const std::shared_ptr<const TfUpdate>&)>;

  /// @brief  Register a calback function to be called when a transform is
  /// updated.
  void registerTfUpdateCallback(const TfUpdateCallback& callback)
      EXCLUDES(callbacks_mutex_) {
    UniqueLock lock(callbacks_mutex_);
    tf_update_callbacks_.push_back(callback);
  }

  /// @brief   Update according to the given tf_update.
  bool update(const TfUpdate& tf_update) {
    if (tf_update.is_dynamic) {
      return updateTransform(
          tf_update.frame_id, tf_update.child_frame_id, tf_update.timestamp,
          tf_update.pose);
    } else {
      return registerStaticTransform(
          tf_update.frame_id, tf_update.child_frame_id, tf_update.pose);
    }
  }

  /// @brief  To check whether a frame is root (has no parent frame so far).
  /// There might be
  ///        multiple root frames.
  /// A frame is root if it is registered but not in the transforms_ map (has no
  /// parent frame)
  bool isRoot(const FrameId& frame_id) const EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    return registered_frames_.find(frame_id) != registered_frames_.end() &&
           transforms_.find(frame_id) == transforms_.end();
  }

  /// Register a static transform between parent_frame_id and child_frame_id.
  bool registerStaticTransform(
      const FrameId& parent_frame_id, const FrameId& child_frame_id,
      const Pose& pose) EXCLUDES(transforms_mutex_) {
    auto new_transform = std::make_shared<Transform>(
        parent_frame_id, child_frame_id, false, pose, nullptr);
    if (!insertTfTransform(new_transform)) {
      LOGW(
          "sk4slam::TF: StaticTransform already exists for frame %s!",
          toStr(child_frame_id).c_str());
      return false;
    }

    // Trigger the callbacks
    runTfUpdateCallbacks(std::make_shared<TfUpdate>(
        TfUpdate{parent_frame_id, child_frame_id, false, pose, -1}));
    return true;
  }

  /// Register a dynamic transform between parent_frame_id and child_frame_id.
  /// If the transform already exists, do nothing. Otherwise, add a new
  /// dynamic transform with an empty pose buffer.
  bool registerDynamicTransform(
      const FrameId& parent_frame_id, const FrameId& child_frame_id)
      EXCLUDES(transforms_mutex_) {
    auto new_transform = std::make_shared<Transform>(
        parent_frame_id, child_frame_id, true, Pose::Identity(),
        std::make_shared<PoseBuf>());
    if (!insertTfTransform(new_transform)) {
      LOGW(
          "sk4slam::TF: DynamicTransform already exists for frame %s!",
          toStr(child_frame_id).c_str());
      return false;
    }
    return true;
  }

  /// @brief  Add a pose with timestamp to the pose buffer of the transform
  /// between
  ///        parent_frame_id and child_frame_id. If the transform is static,
  ///        do nothing.
  bool updateTransform(
      const FrameId& parent_frame_id, const FrameId& child_frame_id,
      const Timestamp& time, const Pose& pose) {
    auto transform = getTfTransform(child_frame_id);
    if (!transform) {
      registerDynamicTransform(parent_frame_id, child_frame_id);
      transform = getTfTransform(child_frame_id);
      ASSERT(transform);
    }
    ASSERT(transform->frame_id == parent_frame_id);
    if (transform->is_dynamic) {
      bool success = transform->updateDynamic(time, pose);
      if (success) {
        // Trigger the callbacks
        runTfUpdateCallbacks(std::make_shared<TfUpdate>(
            TfUpdate{parent_frame_id, child_frame_id, true, pose, time}));
      }
      return success;
    } else {
      LOGE(
          "sk4slam::TF: Transform for frame %s is static! Can't add pose!",
          toStr(child_frame_id).c_str());
      return false;
    }
  }

  bool getStaticTransform(
      const FrameId& from_frame_id, const FrameId& to_frame_id,
      Pose* pose) const {
    if (!checkFrames(from_frame_id, to_frame_id)) {
      return false;
    }

    const auto& [is_static, cached_pose] =
        getCachedStaticTransform(from_frame_id, to_frame_id);
    if (is_static) {
      *pose = *cached_pose;
      return true;
    } else {
      return false;
    }
  }

  const Pose& getStaticTransform(
      const FrameId& from_frame_id, const FrameId& to_frame_id) const {
    if (!checkFrames(from_frame_id, to_frame_id)) {
      std::string err_msg = formatStr(
          "sk4slam::TF: Frame %s or %s not registered!",
          toStr(from_frame_id).c_str(), toStr(to_frame_id).c_str());
      LOGE("%s", err_msg.c_str());
      throw std::runtime_error(err_msg);
    }

    const auto& [is_static, cached_pose] =
        getCachedStaticTransform(from_frame_id, to_frame_id);
    if (is_static) {
      return *cached_pose;
    } else {
      std::string err_msg = formatStr(
          "sk4slam::TF: getStaticTransform() failed since the transform "
          "between frames %s and %s is not static!",
          toStr(from_frame_id).c_str(), toStr(to_frame_id).c_str());
      LOGE("%s", err_msg.c_str());
      throw std::runtime_error(err_msg);
    }
  }

  bool isTransformStatic(
      const FrameId& from_frame_id, const FrameId& to_frame_id) const {
    if (!checkFrames(from_frame_id, to_frame_id)) {
      return false;
    }
    const auto& [is_static, cached_pose] =
        getCachedStaticTransform(from_frame_id, to_frame_id);
    return is_static;
  }

  bool getTransform(
      const FrameId& from_frame_id, const FrameId& to_frame_id,
      const Timestamp& time, Pose* pose) const {
    if (!checkFrames(from_frame_id, to_frame_id)) {
      return false;
    }
    FrameId common_ancestor;
    if (!findCommonAncestor(from_frame_id, to_frame_id, &common_ancestor)) {
      LOGE(
          "sk4slam::TF: No common ancestor found for frames %s and %s!",
          toStr(from_frame_id).c_str(), toStr(to_frame_id).c_str());
      return false;
    }
    return getTransform(
        from_frame_id, to_frame_id, common_ancestor, time, pose);
  }

  void removeOlderThan(const Timestamp& time) {
    auto transforms = getAllTfTransforms();
    for (auto& transform : transforms) {
      transform->removeOlderThan(time);
    }
  }

  std::vector<const Transform*> getAllStaticTransforms() const {
    auto transforms = getAllTfTransforms();
    std::vector<const Transform*> static_transforms;
    for (auto& transform : transforms) {
      if (!transform->is_dynamic) {
        static_transforms.push_back(transform);
      }
    }
    return static_transforms;
  }

 protected:
  bool checkFrames(const FrameId& from_frame_id, const FrameId& to_frame_id)
      const EXCLUDES(transforms_mutex_) {
    bool ret = true;
    {
      SharedLock lock(transforms_mutex_);
      ret = checkFramesInternal(from_frame_id, to_frame_id);
    }
    if (!ret) {
      LOGE(
          "sk4slam::TF: Frame %s or %s not registered!",
          toStr(from_frame_id).c_str(), toStr(to_frame_id).c_str());
    }
    return ret;
  }

  bool checkFramesInternal(
      const FrameId& from_frame_id, const FrameId& to_frame_id) const
      REQUIRES_SHARED(transforms_mutex_) {
    return registered_frames_.find(from_frame_id) != registered_frames_.end() &&
           registered_frames_.find(to_frame_id) != registered_frames_.end();
  }

  bool findCommonAncestor(
      const FrameId& from_frame_id, const FrameId& to_frame_id,
      FrameId* common_ancestor, bool static_only = false) const
      EXCLUDES(transforms_mutex_) {
    if (from_frame_id == to_frame_id) {
      *common_ancestor = from_frame_id;
      return true;
    }

    SharedLock lock(transforms_mutex_);
    std::unordered_set<FrameId> from_ancestors;
    auto it = transforms_.find(from_frame_id);
    while (it != transforms_.end()) {
      const auto& transform = it->second;
      if (static_only && transform->is_dynamic) {
        break;
      }
      if (transform->frame_id == to_frame_id) {
        *common_ancestor = to_frame_id;
        return true;
      }
      from_ancestors.insert(transform->frame_id);
      it = transforms_.find(transform->frame_id);
    }

    auto it2 = transforms_.find(to_frame_id);
    while (it2 != transforms_.end()) {
      const auto& transform2 = it2->second;
      if (static_only && transform2->is_dynamic) {
        break;
      }
      if (from_ancestors.find(transform2->frame_id) != from_ancestors.end()) {
        *common_ancestor = transform2->frame_id;
        return true;
      }
      it2 = transforms_.find(transform2->frame_id);
    }

    return false;
  }

  bool getPoseInAncestorRecursive(
      const FrameId& from_frame_id, const FrameId& ancestor_id,
      const Timestamp& time, Pose* pose) const {
    if (from_frame_id == ancestor_id) {
      *pose = Pose::Identity();
      return true;
    }

    auto transform = getTfTransform(from_frame_id);
    if (!transform) {
      LOGE(
          "sk4slam::TF: No transform found for frame %s!",
          toStr(from_frame_id).c_str());
      return false;
    }

    if (transform->frame_id == ancestor_id) {
      if (transform->is_dynamic) {
        return transform->getDynamic(time, pose);
      } else {
        *pose = transform->pose;
        return true;
      }
    }

    Pose pose_in_parent;
    if (transform->is_dynamic) {
      if (!transform->getDynamic(time, &pose_in_parent)) {
        LOGE(
            "sk4slam::TF: No pose found for frame %s at time %f!",
            toStr(from_frame_id).c_str(), time);
        return false;
      }
    } else {
      pose_in_parent = transform->pose;
    }

    Pose parent_in_ancestor;
    if (!getPoseInAncestorRecursive(
            transform->frame_id, ancestor_id, time, &parent_in_ancestor)) {
      LOGE(
          "sk4slam::TF: No pose found for frame %s at time %f!",
          toStr(transform->frame_id).c_str(), time);
      return false;
    }

    *pose = parent_in_ancestor * pose_in_parent;
    return true;
  }

  bool getTransform(
      const FrameId& from_frame_id, const FrameId& to_frame_id,
      const FrameId& common_ancestor, const Timestamp& time, Pose* pose) const {
    if (to_frame_id == common_ancestor) {
      return getPoseInAncestorRecursive(
          from_frame_id, common_ancestor, time, pose);
    } else if (from_frame_id == common_ancestor) {
      Pose to_frame_in_ancestor;
      bool found = getPoseInAncestorRecursive(
          to_frame_id, common_ancestor, time, &to_frame_in_ancestor);
      if (!found) {
        LOGE(
            "sk4slam::TF: No pose found for frame %s at time %f!",
            toStr(to_frame_id).c_str(), time);
        return false;
      } else {
        *pose = to_frame_in_ancestor.inverse();
        return true;
      }
    } else {
      Pose from_frame_in_ancestor;
      Pose to_frame_in_ancestor;
      bool found_from = getPoseInAncestorRecursive(
          from_frame_id, common_ancestor, time, &from_frame_in_ancestor);
      if (!found_from) {
        LOGE(
            "sk4slam::TF: No pose found for frame %s at time %f!",
            toStr(from_frame_id).c_str(), time);
        return false;
      }
      bool found_to = getPoseInAncestorRecursive(
          to_frame_id, common_ancestor, time, &to_frame_in_ancestor);
      if (!found_to) {
        LOGE(
            "sk4slam::TF: No pose found for frame %s at time %f!",
            toStr(to_frame_id).c_str(), time);
        return false;
      }
      *pose = to_frame_in_ancestor.inverse() * from_frame_in_ancestor;
      return true;
    }
  }

 protected:
  bool insertTfTransform(const TransformPtr& transform)
      EXCLUDES(transforms_mutex_) {
    UniqueLock lock(transforms_mutex_);
    bool ret =
        transforms_.insert({transform->child_frame_id, transform}).second;
    if (ret) {
      registered_frames_.insert(transform->child_frame_id);
      registered_frames_.insert(transform->frame_id);
    }
    return ret;
  }

  template <typename TransformPtrContainer>
  void insertTfTransforms(const TransformPtrContainer& transforms) {
    for (const auto& transform : transforms) {
      insertTfTransform(transform);
    }
  }

  Transform* getTfTransform(const FrameId& child_frame_id)
      EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    auto it = transforms_.find(child_frame_id);
    if (it == transforms_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  const Transform* getTfTransform(const FrameId& child_frame_id) const
      EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    auto it = transforms_.find(child_frame_id);
    if (it == transforms_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  std::vector<Transform*> getAllTfTransforms() EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    std::vector<Transform*> ret;
    for (const auto& pair : transforms_) {
      ret.push_back(pair.second.get());
    }
    return ret;
  }

  std::vector<TransformPtr> getAllSharedTfTransforms()
      EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    std::vector<TransformPtr> ret;
    for (const auto& pair : transforms_) {
      ret.push_back(pair.second);
    }
    return ret;
  }

  std::vector<const Transform*> getAllTfTransforms() const
      EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    std::vector<const Transform*> ret;
    for (const auto& pair : transforms_) {
      ret.push_back(pair.second.get());
    }
    return ret;
  }

  std::vector<ConstTransformPtr> getAllSharedTfTransforms() const
      EXCLUDES(transforms_mutex_) {
    SharedLock lock(transforms_mutex_);
    std::vector<ConstTransformPtr> ret;
    for (const auto& pair : transforms_) {
      ret.push_back(pair.second);
    }
    return ret;
  }

  void runTfUpdateCallbacks(const std::shared_ptr<const TfUpdate>& update)
      EXCLUDES(callbacks_mutex_) {
    // Trigger the callbacks
    std::vector<TfUpdateCallback> tf_update_callbacks;
    {
      SharedLock lock(callbacks_mutex_);
      tf_update_callbacks = tf_update_callbacks_;
    }
    if (!tf_update_callbacks.empty()) {
      for (const auto& callback : tf_update_callbacks) {
        callback(update);
      }
    }
  }

 protected:
  using FrameIdPair = TfFrameIdPair_<FrameId>;

  std::pair<bool, const Pose*> getCachedStaticTransform(
      const FrameId& from_frame_id, const FrameId& to_frame_id) const
      EXCLUDES(cache_mutex_) {
    return getCachedStaticTransform({from_frame_id, to_frame_id});
  }

  std::pair<bool, const Pose*> getCachedStaticTransform(
      const FrameIdPair& pair) const EXCLUDES(cache_mutex_) {
    static const Pose* const nullptr_pose = nullptr;
    SharedLock lock(cache_mutex_);
    auto it = is_transform_static_cache_.find(pair);
    if (it == is_transform_static_cache_.end()) {
      lock.unlock();
      return updateCachedStaticTransform(pair);
    }
    if (it->second) {
      const Pose* pose = &static_transforms_cache_.at(pair);
      return std::make_pair(true, pose);
    } else {
      return std::make_pair(false, nullptr_pose);
    }
  }

  std::pair<bool, const Pose*> updateCachedStaticTransform(
      const FrameIdPair& pair, bool is_static, const Pose& pose = Pose()) const
      EXCLUDES(cache_mutex_) {
    static const Pose* const nullptr_pose = nullptr;

    UniqueLock lock(cache_mutex_);
    is_transform_static_cache_[pair] = is_static;
    if (is_static) {
      return std::make_pair(
          true, &static_transforms_cache_.insert({pair, pose}).first->second);
    } else {
      return std::make_pair(true, nullptr_pose);
    }
  }

  std::pair<bool, const Pose*> updateCachedStaticTransform(
      const FrameIdPair& pair) const {
    const auto& [from_frame_id, to_frame_id] = pair;
    FrameId static_common_ancestor;
    if (!findCommonAncestor(
            from_frame_id, to_frame_id, &static_common_ancestor, true)) {
      LOGE(
          "sk4slam::TF: No common ancestor (static) found for frames %s "
          "and %s!",
          toStr(from_frame_id).c_str(), toStr(to_frame_id).c_str());
      return updateCachedStaticTransform(pair, false);
    }
    Pose pose;
    ASSERT(getTransform(
        from_frame_id, to_frame_id, static_common_ancestor, 0, &pose));
    return updateCachedStaticTransform(pair, true, pose);
  }

 protected:
  // TODO(jeffrey): Use our atomic hash table instead of std::unordered_map and
  // std::unordered_set;

  std::unordered_map<FrameId, TransformPtr> transforms_
      GUARDED_BY(transforms_mutex_);

  std::unordered_set<FrameId> registered_frames_ GUARDED_BY(transforms_mutex_);

  mutable std::unordered_map<FrameIdPair, Pose> static_transforms_cache_
      GUARDED_BY(cache_mutex_);

  mutable std::unordered_map<FrameIdPair, bool> is_transform_static_cache_
      GUARDED_BY(cache_mutex_);

  std::vector<TfUpdateCallback> tf_update_callbacks_
      GUARDED_BY(callbacks_mutex_);

  mutable Mutex transforms_mutex_;
  mutable Mutex cache_mutex_;
  mutable Mutex callbacks_mutex_;
};

}  // namespace sk4slam

namespace std {
template <typename _FrameId>
struct hash<sk4slam::TfFrameIdPair_<_FrameId>> {
  size_t operator()(const sk4slam::TfFrameIdPair_<_FrameId>& pair) const {
    return std::hash<_FrameId>()(pair.from_frame_id) ^
           std::hash<_FrameId>()(pair.to_frame_id);
  }
};
}  // namespace std
