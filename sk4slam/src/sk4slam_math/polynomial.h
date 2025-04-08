#pragma once

#include "sk4slam_math/third_party/colmap/math/polynomial.h"

namespace sk4slam {

// All polynomials are assumed to be the form:
//
//   sum_{i=0}^N polynomial(i) x^{N-i}.
//
// and are given by a vector of coefficients of size N + 1.
//
// The implementation is based on COLMAP's old polynomial functionality and is
// inspired by Ceres-Solver's/Theia's implementation to support complex
// polynomials. The companion matrix implementation is based on NumPy.

// Evaluate the polynomial for the given coefficients at x using the Horner
// scheme. This function is templated such that the polynomial may be evaluated
// at real and/or imaginary points.
template <typename T>
T evaluatePolynomial(const Eigen::VectorXd& coeffs, const T& x) {
  T value = 0.0;
  for (Eigen::VectorXd::Index i = 0; i < coeffs.size(); ++i) {
    value = value * x + coeffs(i);
  }
  return value;
}

// Find the roots of a polynomial.
// The real and/or imaginary variable may be NULL if the output is not needed.
inline bool findRootsForPolynomial(
    const Eigen::VectorXd& coeffs, Eigen::VectorXd* real, Eigen::VectorXd* imag,
    bool use_durand_kerner = false) {
  if (coeffs.size() < 2) {
    return false;
  } else if (coeffs.size() == 2) {
    // Find the root of polynomials of the form: a * x + b = 0.
    return sk4slam_colmap::FindLinearPolynomialRoots(coeffs, real, imag);
  } else if (coeffs.size() == 3) {
    // Find the roots of polynomials of the form: a * x^2 + b * x + c = 0.
    return sk4slam_colmap::FindQuadraticPolynomialRoots(coeffs, real, imag);
  } else if (use_durand_kerner) {
    // Find the roots of a polynomial using the Durand-Kerner method, based on:
    //
    //    https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
    //
    // The Durand-Kerner is comparatively fast but often unstable/inaccurate.
    return sk4slam_colmap::FindPolynomialRootsDurandKerner(coeffs, real, imag);
  } else {
    // Find the roots of a polynomial using the companion matrix method, based
    // on:
    //
    //    R. A. Horn & C. R. Johnson, Matrix Analysis. Cambridge,
    //    UK: Cambridge University Press, 1999, pp. 146-7.
    //
    // Compared to Durand-Kerner, this method is slower but more
    // stable/accurate.
    return sk4slam_colmap::FindPolynomialRootsCompanionMatrix(
        coeffs, real, imag);
  }
}

inline Eigen::VectorXd pickRealRoots(
    const Eigen::VectorXd& real, const Eigen::VectorXd& imag, double eps = 1e-8,
    bool merge_repeated_roots = false) {
  ASSERT(real.size() == imag.size());
  Eigen::VectorXd real_roots(real.size());
  size_t j = 0;
  for (size_t i = 0; i < real.size(); ++i) {
    if (std::abs(imag(i)) < eps) {
      real_roots(j) = real(i);
      ++j;
    }
  }
  real_roots.conservativeResize(j);

  // Sort the real roots in ascending order and then merge repeated real roots
  if (merge_repeated_roots) {
    std::sort(real_roots.data(), real_roots.data() + real_roots.size());
    j = 1;
    for (size_t i = 1; i < real_roots.size(); ++i) {
      if (std::abs(real_roots(i) - real_roots(j - 1)) > eps) {
        real_roots(j) = real_roots(i);
        ++j;
      }
    }
    real_roots.conservativeResize(j);
  }

  return real_roots;
}

inline Eigen::VectorXd findRealRootsForPolynomial(
    const Eigen::VectorXd& coeffs, double eps = 1e-8,
    bool merge_repeated_roots = false, bool use_durand_kerner = false) {
  Eigen::VectorXd real, imag;
  findRootsForPolynomial(coeffs, &real, &imag, use_durand_kerner);
  return pickRealRoots(real, imag, eps, merge_repeated_roots);
}

}  // namespace sk4slam
