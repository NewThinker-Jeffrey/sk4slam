#include "sk4slam_imu/imu_model.h"

namespace sk4slam {

ImuData ImuData::interpolate(
    const ImuData& imu_1, const ImuData& imu_2, double timestamp) {
  // time-distance lambda
  double lambda =
      (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
  // LOGD("lambda - %d\n", lambda);
  // interpolate between the two times
  ImuData data;
  data.timestamp = timestamp;
  data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
  data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
  return data;
}

std::vector<ImuData> ImuData::fillDataGaps(
    const std::vector<ImuData>& in_data, double max_gap) {
  std::vector<ImuData> out_data;
  if (in_data.empty()) {
    return out_data;
  }

  // If "in_data" is of size 1, then so will be the out_data.
  for (size_t i = 0; i < in_data.size() - 1; i++) {
    out_data.push_back(in_data[i]);
    double t0 = in_data[i].timestamp;
    double t1 = in_data[i + 1].timestamp;
    double gap = t1 - t0;
    if (gap > max_gap) {
      if (gap > 0.1) {
        LOGW(
            YELLOW
            "Imu::fillDataGaps(): LARGE_IMU_GAP!! "
            "We're filling a large imu gap (%.3f) from %.3f to %.3f!!!\n" RESET,
            gap, t0, t1);
      }

      int to_fill = gap / max_gap;
      double interval = gap / (to_fill + 1);
      for (int k = 0; k < to_fill; k++) {
        double t = t0 + (k + 1) * interval;
        ASSERT(t < t1 - 0.5 * interval);
        out_data.push_back(interpolate(in_data[i], in_data[i + 1], t));
      }
    }
  }

  out_data.push_back(in_data.back());
  return out_data;
}

}  // namespace sk4slam
