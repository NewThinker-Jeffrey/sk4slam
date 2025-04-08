#pragma once

#include "sk4slam_math/third_party/colmap/math/math.h"

namespace sk4slam {

// Return 1 if number is positive, -1 if negative, and 0 if the number is 0.
template <typename T>
inline int signOfNumber(T val) {
  return sk4slam_colmap::SignOfNumber(val);
}

// Clamp the given value to a low and maximum value.
template <typename T>
inline T clamp(const T& value, const T& low, const T& high) {
  return sk4slam_colmap::Clamp(value, low, high);
}

// Convert angle in degree to radians.
template <typename Float>
inline Float degToRad(Float deg) {
  return sk4slam_colmap::DegToRad(deg);
}

// Convert angle in radians to degree.
template <typename Float>
inline Float radToDeg(Float rad) {
  return sk4slam_colmap::RadToDeg(rad);
}

// Determine median value in vector. Returns NaN for empty vectors.
template <typename T>
inline double median(const std::vector<T>& elems) {
  return sk4slam_colmap::Median(elems);
}

// Determine mean value in a vector.
template <typename T>
inline double mean(const std::vector<T>& elems) {
  return sk4slam_colmap::Mean(elems);
}

// Determine sample variance in a vector.
template <typename T>
inline double variance(const std::vector<T>& elems) {
  return sk4slam_colmap::Variance(elems);
}

// Determine sample standard deviation in a vector.
template <typename T>
inline double stddev(const std::vector<T>& elems) {
  return sk4slam_colmap::StdDev(elems);
}

// Generate N-choose-K combinations.
//
// Note that elements in range [first, last) must be in sorted order,
// according to `std::less`.
template <class Iterator>
inline bool nextCombination(Iterator first, Iterator middle, Iterator last) {
  return sk4slam_colmap::NextCombination(first, middle, last);
}

// Sigmoid function.
template <typename T>
inline T sigmoid(T x, T alpha = 1) {
  return sk4slam_colmap::Sigmoid(x, alpha);
}

// Scale values according to sigmoid transform.
//
//   x \in [0, 1] -> x \in [-x0, x0] -> sigmoid(x, alpha) -> x \in [0, 1]
//
// @param x        Value to be scaled in the range [0, 1].
// @param x0       Spread that determines the range x is scaled to.
// @param alpha    Exponential sigmoid factor.
//
// @return         The scaled value in the range [0, 1].
template <typename T>
inline T scaleSigmoid(T x, T alpha = 1, T x0 = 10) {
  return sk4slam_colmap::ScaleSigmoid(x, alpha, x0);
}

// Binomial coefficient or all combinations, defined as n! / ((n - k)! k!).
inline size_t Cnk(size_t n, size_t k) {
  return sk4slam_colmap::NChooseK(n, k);
}

// Cast value from one type to another and truncate instead of overflow, if the
// input value is out of range of the output data type.
template <typename T1, typename T2>
inline T2 truncateCast(T1 value) {
  return sk4slam_colmap::TruncateCast<T1, T2>(value);
}

// Compute the n-th percentile in the given sequence.
template <typename T>
inline T percentile(const std::vector<T>& elems, double p) {
  return sk4slam_colmap::Percentile(elems, p);
}

}  // namespace sk4slam
