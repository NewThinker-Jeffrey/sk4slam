#include "sk4slam_imu/imu_data_buffer.h"

#include "sk4slam_cpp/binary_search.h"

namespace sk4slam {

void ImuDataBuffer::addImuData(
    ImuDataSeq& imu_data_seq, const ImuData& imu_data) {
  if (imu_data_seq.empty() ||
      imu_data_seq.back().timestamp < imu_data.timestamp) {
    imu_data_seq.push_back(imu_data);
  } else {
    LOGW(
        "ImuDataBuffer::addImuData(): imu_data is out of order! new - "
        "last = %.6f",
        imu_data.timestamp - imu_data_seq.back().timestamp);
  }
}

void ImuDataBuffer::removeOldImuData(
    ImuDataSeq& imu_data_seq, double timestamp /*exclusive*/) {
  if (!imu_data_seq.empty()) {
    static const double time_margin =
        1.0;  // Time margin to prevent overly frequent data removal.

    // If the earliest IMU data timestamp is within the margin, skip removal
    if (imu_data_seq.front().timestamp >= timestamp - time_margin) {
      return;  // No need to remove anything
    }
    auto search_res = binarySearchLowerBound(
        imu_data_seq, timestamp, std::less<double>(),
        [](const ImuData& imu_data) { return imu_data.timestamp; });
    ASSERT(search_res);
    size_t end_idx = *search_res;
    imu_data_seq.erase(imu_data_seq.begin(), imu_data_seq.begin() + end_idx);
#if USE_SK4SLAM_DEQUE_FOR_IMU_DATA_SEQ_IN_VIO_IMU_DATA_BUFFER
    imu_data_seq.trim_to_optimal();
#else
    imu_data_seq.shrink_to_fit();  // This might be slow when the deque is large
#endif
  }
}

std::vector<ImuData> ImuDataBuffer::getImuDataBetween(
    const ImuDataSeq& imu_data_seq, double start_timestamp,
    double end_timestamp) const {
  return ImuData::selectBetween(imu_data_seq, start_timestamp, end_timestamp);
}

std::vector<ImuData> ImuDataBuffer::getRecentImuData(
    const ImuDataSeq& imu_data_seq, double seconds) const {
  return ImuData::selectRecent(imu_data_seq, seconds);
}

void ImuDataBuffer::addImuData(const UniqueId& imu_uid, const ImuData& imu_data)
    EXCLUDES(imu_data_mutex_) {
  UniqueLock lock(imu_data_mutex_);
  addImuData(imu_data_[imu_uid], imu_data);
}

void ImuDataBuffer::addImuData(const ImuData& imu_data)
    EXCLUDES(imu_data_mutex_) {
  UniqueLock lock(imu_data_mutex_);
  if (imu_data_.empty()) {
    // Initialize the imu_data_ map with a random imu_uid
    imu_data_[UniqueId(true)] = ImuDataSeq();
  }

  addImuData(imu_data_.begin()->second, imu_data);
}

void ImuDataBuffer::removeOldImuData(
    const UniqueId& imu_uid, double timestamp /*exclusive*/) {
  UniqueLock lock(imu_data_mutex_);
  auto it = imu_data_.find(imu_uid);
  if (it != imu_data_.end()) {
    removeOldImuData(it->second, timestamp);
  }
}

void ImuDataBuffer::removeOldImuData(double timestamp /*exclusive*/) {
  UniqueLock lock(imu_data_mutex_);
  for (auto& imu_data : imu_data_) {
    removeOldImuData(imu_data.second, timestamp);
  }
}

std::vector<ImuData> ImuDataBuffer::getImuDataBetween(
    const UniqueId& imu_uid, double start_timestamp,
    double end_timestamp) const {
  SharedLock lock(imu_data_mutex_);
  auto it = imu_data_.find(imu_uid);
  if (it != imu_data_.end()) {
    return getImuDataBetween(it->second, start_timestamp, end_timestamp);
  }
  return {};
}

std::vector<ImuData> ImuDataBuffer::getImuDataBetween(
    double start_timestamp, double end_timestamp) const {
  SharedLock lock(imu_data_mutex_);
  if (imu_data_.empty()) {
    return {};
  }
  return getImuDataBetween(
      imu_data_.begin()->second, start_timestamp, end_timestamp);
}

std::vector<ImuData> ImuDataBuffer::getRecentImuData(
    const UniqueId& imu_uid, double seconds) const {
  std::vector<ImuData> ret;
  SharedLock lock(imu_data_mutex_);
  auto it = imu_data_.find(imu_uid);
  if (it != imu_data_.end()) {
    return getRecentImuData(it->second, seconds);
  }
  return {};
}

std::vector<ImuData> ImuDataBuffer::getRecentImuData(double seconds) const {
  std::vector<ImuData> ret;
  SharedLock lock(imu_data_mutex_);
  if (imu_data_.empty()) {
    return {};
  }
  return getRecentImuData(imu_data_.begin()->second, seconds);
}

double ImuDataBuffer::getLatestImuTime(const UniqueId& imu_uid) const {
  SharedLock lock(imu_data_mutex_);
  auto it = imu_data_.find(imu_uid);
  if (it != imu_data_.end() && !it->second.empty()) {
    return it->second.back().timestamp;
  } else {
    return -1;
  }
}

double ImuDataBuffer::getLatestImuTime() const {
  double ret = -1;
  SharedLock lock(imu_data_mutex_);
  for (const auto& [imu_uid, imu_data] : imu_data_) {
    if (imu_data.empty()) {
      continue;
    }
    ret = std::max(ret, imu_data.back().timestamp);
  }
  return ret;
}

}  // namespace sk4slam
