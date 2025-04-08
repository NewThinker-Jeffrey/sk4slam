#pragma once

#include <Eigen/Core>  // Eigen::NumTraits
#include <cmath>
#include <limits>

namespace sk4slam {

namespace liegroup {

// Some frequently used constants for Lie groups
template <typename Scalar>
struct Constants {
  static inline const Scalar kPI = Scalar(M_PI);

  // static inline const Scalar kEps = std::numeric_limits<Scalar>::epsilon();
  //
  // A pitfall of using std::numeric_limits:
  //    Older versions of ceres doesn't provide a specialization of
  //    std::numeric_limits for Jet types, hence the kEps will be 0 for
  //    Jet types and causes bugs.
  // So we use Eigen::NumTraits instead.
  static inline const Scalar kEps = Eigen::NumTraits<Scalar>::epsilon();

  static inline const Scalar kNum_0 = Scalar(0);
  static inline const Scalar kNum_0p5 = Scalar(0.5);
  static inline const Scalar kNum_1 = Scalar(1);
  static inline const Scalar kNum_neg1 = Scalar(-1);
  static inline const Scalar kNum_2 = Scalar(2);
  static inline const Scalar kNum_3 = Scalar(3);
  static inline const Scalar kNum_4 = Scalar(4);
  static inline const Scalar kNum_12 = Scalar(12);
  static inline const Scalar kNum_720 = Scalar(720);
};

}  // namespace liegroup

}  // namespace sk4slam
