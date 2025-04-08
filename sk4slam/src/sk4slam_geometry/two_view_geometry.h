#pragma once

#include <Eigen/Core>

#include "sk4slam_geometry/essential_matrix.h"
#include "sk4slam_geometry/fundamental_matrix.h"
#include "sk4slam_geometry/homography_matrix.h"
#include "sk4slam_math/sac.h"

namespace sk4slam {

enum class TwoViewGeometryType {
  ROTATION,     // Rotation matrix    , DOF=3  (A special case of HOMOGRAPHY)
  ESSENTIAL,    // Essential matrix   , DOF=5
  FUNDAMENTAL,  // Fundamental matrix , DOF=7
  HOMOGRAPHY,   // Homography matrix  , DOF=8
  UNKNOWN
};

static const char* TwoViewGeometryTypeStr[] = {
    "ROTATION", "ESSENTIAL", "FUNDAMENTAL", "HOMOGRAPHY", "UNKNOWN"};

inline std::string toStr(TwoViewGeometryType type) {
  static constexpr int n =
      sizeof(TwoViewGeometryTypeStr) / sizeof(TwoViewGeometryTypeStr[0]);
  ASSERT(static_cast<int>(type) < n);
  return TwoViewGeometryTypeStr[static_cast<int>(type)];
}

// Degeneracy-aware TwoView geometry
struct TwoViewGeometry {
  // The fundamental/essential/homography matrix
  Eigen::Matrix3d matrix;

  // The type of the geometry
  TwoViewGeometryType type{TwoViewGeometryType::UNKNOWN};

  TwoViewGeometry() {}
  TwoViewGeometry(
      TwoViewGeometryType type_in,
      const Eigen::Matrix3d& matrix_in = Eigen::Matrix3d::Identity())
      : type(type_in), matrix(matrix_in) {}

  using PointPair = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

  // If the geometry type is HOMOGRAPHY (including ROTATION) or ESSENTIAL,
  // we can recover the rotation and translation (up to scale) between
  // the two views, assuming that the images are calibrated (i.e., the
  // intrinsics K1 = K2 = I3x3).
  //
  // Note that decomposing a Homography/Essential matrix may yield several
  // possible poses, so we need some points to perform the Cheirality-Check
  // (positive-depth check) and determine the correct one.
  //
  // For other geometry types (such as FUNDAMENTAL), no pose can be recovered,
  // and the function will return false.
  //
  // About the last parameter `H_pure_rotation_thr`, see the member variable
  // `pure_rotation_thr` of the struct
  // `DegeneracyAwareEssentialEstimator::DegeneracyAwareOptions`.
  bool computePose(
      const std::vector<PointPair>& point_pairs, Eigen::Matrix3d* R,
      Eigen::Vector3d* t, const std::vector<size_t>& selected_indices = {},
      const double H_pure_rotation_thr =
          3e-3) const;  // empty means selecting all points

  std::vector<double> computeSquaredAlgebraicErrors(
      const std::vector<PointPair>& point_pairs,
      const std::vector<size_t>& selected_indices = {}) const;

  std::vector<double> computeSquaredSampsonErrors(
      const std::vector<PointPair>& point_pairs,
      const std::vector<size_t>& selected_indices = {}) const;

  double computeSquaredSampsonErrorsSum(
      const std::vector<PointPair>& point_pairs,
      const std::vector<size_t>& selected_indices = {}) const {
    std::vector<double> errs =
        computeSquaredSampsonErrors(point_pairs, selected_indices);
    return std::accumulate(errs.begin(), errs.end(), 0.0);
  }

  double computeSquaredSampsonErrorsAve(
      const std::vector<PointPair>& point_pairs,
      const std::vector<size_t>& selected_indices = {}) const {
    std::vector<double> errs =
        computeSquaredSampsonErrors(point_pairs, selected_indices);
    return std::accumulate(errs.begin(), errs.end(), 0.0) / errs.size();
  }
};

class TwoViewGeometryEstimator {
 public:  // basic properties.
  using DataPoint = TwoViewGeometry::PointPair;

  using Parameter = TwoViewGeometry;

  // Constants used to compute the default RANSAC options
  static constexpr int kDefatultFocal = 400;
  static constexpr int kDefatultFocalSquare = kDefatultFocal * kDefatultFocal;
  static constexpr int kDefaultPointsUsedForInitialH = 5;
  static constexpr int kDefault_H_MaxOptIter = 1;
  static RansacOptions defaultRansacOptions() {
    return RansacOptions(
        (3.0 * 3.0) /
            kDefatultFocalSquare,  // error_thr (for normalized images)
        0.999,                     // confidence
        1000,                      // max_iter
        0.0,                       // initial_inlier_ratio_thr
        0,                         // local_opt_max_iter
                                   // ^ None zero local_opt_max_iter may
                                   // ^ significantly slow down the computation
        1                          // final_opt_max_iter
    );                             // NOLINT
  }

 public:
  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const = 0;

 public:
  virtual ~TwoViewGeometryEstimator() {}

  virtual std::vector<double> errors(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points, const Parameter& model) const;

  virtual bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;

 public:
  std::vector<Parameter> compute2(
      const std::vector<DataPoint>& all_points) const {
    std::vector<size_t> selected_indices;
    selected_indices.resize(all_points.size());
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    return compute(selected_indices, all_points);
  }

  std::vector<double> errors2(
      const std::vector<DataPoint>& all_points, const Parameter& model) const {
    std::vector<size_t> selected_indices;
    selected_indices.resize(all_points.size());
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    return errors(selected_indices, all_points, model);
  }

  bool localOptimize2(
      const std::vector<DataPoint>& all_points, Parameter* param) const {
    std::vector<size_t> inlier_indices;
    inlier_indices.resize(all_points.size());
    std::iota(inlier_indices.begin(), inlier_indices.end(), 0);
    return localOptimize(inlier_indices, all_points, param);
  }

 protected:
  // For convenience.
  static std::vector<Parameter> getParametersFromMatrices(
      TwoViewGeometryType type, const std::vector<Eigen::Matrix3d>& matrices);
};

// How to address the degeneracy problem after estimating a Fundamental
// or Essential matrix by basic methods (7/8-point methods for Fundamental
// and 5/8-point methods for Essential) ?
//
// ** For 7/8 point Fundamental matrix estimators:
//
// If the new F is better than the so-far best, then do the following:
//
// 1. Check H-degenerate sample:
// Check whether the 7/8-points sample is H-degenerate. If NOT, just continue
// the normal RANSAC process.
//
// Note: If we do local optimization on F before checking H-degenerate
//       sample, it should be ensured that the points in the original
//       7/8-points sample are still consistent with the optimized F.
//       The RANSAC framework should be able to guarantee this.
//
// 2. Check dominant H:
// If the sample is H-degenerate, then locally optimize H and check whether
// it's a dominant one, i.e. more than `dominant_h_percent` (say 80%) of the
// F-inliers are consistent with it. This usually happens when the points in
// the original 7/8-points sample are on the same plane or the motion between
// the two views is a pure rotation.
// If the H is NOT a dominant one, then just continue the normal RANSAC
// process.
//
// 3. Re-estimate F:
// Otherwise, separate the points into two groups: for those consistent with
// H (on-H points) and those not (off-H points).
// If there're less than `min_off_h_points` (say 9) off-H points, then just
// output the homography;
// Otherwise, we estimate a Fundamental matrix F with the dominant-H and
// the off-H points by plane-and-parallax method in a inner RANSAC process.
//
//  ^ NOTE:
//    Our tests showed that the DEGENSAC algorithm for Fundamental matrix
//    is highly sensitive to the noise level. In the presence of high noise,
//    there is a significant probability that the off-H points may
//    erroneously fit a spurious F-matrix. Increasing the value of
//    `min_off_h_points` may be beneficial, but it also increases the
//    likelihood of the algorithm disregarding the "good" off-H points and
//    producing a degenerate H matrix even if there are some valid off-H
//    points available.
//
// 4. Continue the normal RANSAC process:
// If we get an F matrix with more than `min_off_h_points` off-H point
// supporters, then coninue the normal RANSAC process with the re-estimated F;
// otherwise, continue the normal RANSAC process with the degenrated H.
//
//
// ** For 5/8-point Essential matrix estimators:
//
// If the new E is better than the so-far best, then do the following:
//
// 1. Check H-degenerate sample:
// Similar to the 7/8-point Fundamental case.
//
// 2. Check dominant H:
// Similar to the 7/8-point Fundamental case.
// If the H is NOT a dominant one, then just continue the normal RANSAC
// process.
//
// 3. Check pure rotation:
// Otherwise, check whether the H is a scaled rotation matrix.
//  ^ NOTE:
//    There is a threold `pure_rotation_thr` used to check whether an H matrix
//    is a rotation matrix. It also depends on the noise level.
//    Increasing the threshold if the noise level is high, but it also increases
//    the likelihood of the algorithm mistaking a motion with small traslation
//    as a pure rotation.
//
// 4. Continue the normal RANSAC process:
// If H is NOT a scaled rotation matrix, then continue the normal RANSAC
// process with E;
// otherwise, continue the normal RANSAC process with the R (=H/scale).
//

struct EpipolarDegensacScoring {
  double degenerate_score_multiplier;

  explicit EpipolarDegensacScoring(double degenerate_score_multiplier_in)
      : degenerate_score_multiplier(degenerate_score_multiplier_in) {}

  struct Score {
    // The number of inliers.
    size_t num_inliers = 0;

    bool degenerated = false;

    // The sum of all inlier errs.
    double err_sum = 0;
  };

  // Compute the score.
  Score evaluate(
      const TwoViewGeometryEstimator& model, const TwoViewGeometry& param,
      const std::vector<TwoViewGeometry::PointPair>& all_points,
      const RansacOptions& sac_options, std::vector<size_t>* inliers) const;

  // Compare the two scores (return true if score2 is strictly better).
  bool compare(const Score& score1, const Score& score2) const;
};

class DegeneracyAwareFundamentalEstimator : public TwoViewGeometryEstimator {
 public:
  struct DegeneracyAwareOptions {
    bool enable_degeneracy_check = false;

    // Ransac options.
    RansacOptions ransac_options;

    // The error threshold `h_degen_err_thr` is used when checking
    // H-degenerate samples.
    //
    // It should be relatively larger than the error threshold used in
    // RANSAC to avoid mistakenly discarding H-degenerate samples.
    //
    // (It's disabled if it's assigned a negative value, and the the error
    //  threshold used in checking H-degenerate samples will be the same
    //  as the one used in RANSAC).
    //
    // (When checking H-degenerate samples, we estimate H from a fixed but
    //  noisy F and 3 extra noisy points. As a result, the estimated H may
    //  deviate significantly from the ground truth H. Therefore, the error
    //  threshold should be relatively larger to account for this potential
    //  deviation)
    double h_degen_err_thr;

    // Max number of iterations for the local optimization of H.
    int h_max_opt_iter;

    // As explained in the above comments.
    double dominant_h_percent;

    // `points_used_for_initial_h` can be  3,4,or 5:
    //     - 3 is much faster but less accurate (and depends on the accuracy of
    //     F)
    //     - 5 is the most accurate but slowest (does not depend on F)
    //     - 4 is a compromise (does not depend on F)
    int points_used_for_initial_h;

    // As explained in the above comments. Only for Fundamental matrix
    // estimation.
    int min_off_h_points;

    // The multiplier for the score of a degenerate model.
    double degenerate_score_multiplier;

    DegeneracyAwareOptions(
        bool enable_degeneracy_check_in = false,
        const RansacOptions& ransac_options_in = defaultRansacOptions(),
        int h_max_opt_iter_in = kDefault_H_MaxOptIter,
        double dominant_h_percent_in = 0.8,
        int points_used_for_initial_h_in = kDefaultPointsUsedForInitialH,
        int min_off_h_points_in = 9,
        double degenerate_score_multiplier_in = 1.05)
        : enable_degeneracy_check(enable_degeneracy_check_in),
          ransac_options(ransac_options_in),
          h_max_opt_iter(h_max_opt_iter_in),
          dominant_h_percent(dominant_h_percent_in),
          points_used_for_initial_h(points_used_for_initial_h_in),
          min_off_h_points(min_off_h_points_in),
          degenerate_score_multiplier(degenerate_score_multiplier_in) {
      if (enable_degeneracy_check) {
        ransac_options.check_final_degeneracy = true;

        // Our tests have indicated that the `check_step_degeneracy` does
        // not significantly impact the results. As a result, we have
        // chosen to disable it in order to reduce computational overhead.
        // However, you can enable it manually if computational overhead is
        // not a concern.

        // ransac_options.check_step_degeneracy = true;
      } else {
        ransac_options.check_step_degeneracy = false;
        ransac_options.check_final_degeneracy = false;
      }

      h_degen_err_thr = -1.0;
      // h_degen_err_thr = (2.0 * 2.0) * ransac_options.error_thr;
    }
  };

  DegeneracyAwareFundamentalEstimator(
      const DegeneracyAwareOptions& degen_params = DegeneracyAwareOptions())
      : degen_params_(degen_params) {}

  // The argument `model` should already contain an initial value that was
  // computed from the selected points, and the function may update `model`
  // if the degeneracy is detected and handled.
  // The argument `initial_inliers` is the set of inliers for the initial
  // model.
  // The function returns true if the `model` is updated, false otherwise.
  virtual bool handleDegeneracy(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points,
      const std::vector<size_t>& initial_inliers, Parameter* model);

 protected:
  DegeneracyAwareOptions degen_params_;
};

class DegeneracyAwareEssentialEstimator : public TwoViewGeometryEstimator {
 public:
  struct DegeneracyAwareOptions {
    bool enable_degeneracy_check = false;

    // Ransac options.
    RansacOptions ransac_options;

    // The error threshold `h_degen_err_thr` is used when checking
    // H-degenerate samples.
    //
    // It should be relatively larger than the error threshold used in RANSAC
    // to avoid mistakenly discarding H-degenerate samples.
    //
    // (It's disabled if it's assigned a negative value, and the the error
    //  threshold used in checking H-degenerate samples will be the same
    //  as the one used in RANSAC).
    //
    // (When checking H-degenerate samples, we estimate H from a fixed but
    //  noisy F and 3 extra noisy points. As a result, the estimated H may
    //  deviate significantly from the ground truth H. Therefore, the error
    //  threshold should be relatively larger to account for this potential
    //  deviation)
    double h_degen_err_thr;

    // Max number of iterations for the local optimization of H.
    int h_max_opt_iter;

    // As explained in the above comments.
    double dominant_h_percent;

    // `points_used_for_initial_h` can be  3,4,or 5:
    //     - 3 is much faster but less accurate (and depends on the accuracy of
    //     F)
    //     - 5 is the most accurate but slowest (does not depend on F)
    //     - 4 is a compromise (does not depend on F)
    //
    // However, If we're using 5-point algorithm, then
    // `points_used_for_initial_h` is forced to be 5 to ensure that the initial
    // H is reliable.
    int points_used_for_initial_h;

    // `pure_rotation_thr` is the threshold used for checking whether H is a
    // scaled rotation matrix. Smaller threshold means more strict checking.
    //
    // If any two column vectors of (the normalized) H are nearly orthogonal,
    // i.e. their dot product is less than this threshold, then H is
    // considered to be a scaled rotation matrix.
    //
    // To be more specific, the underlying check is:
    //
    //        (H^T * H - I).max_abs < pure_rotation_thr
    //
    //    (assuming H is already normalized, i.e. (H^T * H).trace() = 3)
    //
    double pure_rotation_thr;

    // The multiplier for the score of a degenerate model.
    double degenerate_score_multiplier;

    DegeneracyAwareOptions(
        bool enable_degeneracy_check_in = false,
        const RansacOptions& ransac_options_in = defaultRansacOptions(),
        int h_max_opt_iter_in = kDefault_H_MaxOptIter,
        double dominant_h_percent_in = 0.8,
        int points_used_for_initial_h_in = kDefaultPointsUsedForInitialH,
        // double pure_rotation_thr_in = 1e-2,
        double pure_rotation_thr_in = 3e-3,
        double degenerate_score_multiplier_in = 1.1)
        : enable_degeneracy_check(enable_degeneracy_check_in),
          ransac_options(ransac_options_in),
          h_max_opt_iter(h_max_opt_iter_in),
          dominant_h_percent(dominant_h_percent_in),
          points_used_for_initial_h(points_used_for_initial_h_in),
          pure_rotation_thr(pure_rotation_thr_in),
          degenerate_score_multiplier(degenerate_score_multiplier_in) {
      if (enable_degeneracy_check) {
        ransac_options.check_final_degeneracy = true;

        // Our tests have indicated that the `check_step_degeneracy` does
        // not significantly impact the results. As a result, we have
        // chosen to disable it in order to reduce computational overhead.
        // However, you can enable it manually if computational overhead is
        // not a concern.

        // ransac_options.check_step_degeneracy = true;
      } else {
        ransac_options.check_step_degeneracy = false;
        ransac_options.check_final_degeneracy = false;
      }

      h_degen_err_thr = -1.0;
      // h_degen_err_thr = (2.0 * 2.0) * ransac_options.error_thr;
    }
  };

  DegeneracyAwareEssentialEstimator(
      const DegeneracyAwareOptions& degen_params = DegeneracyAwareOptions())
      : degen_params_(degen_params) {}

  // The argument `model` should already contain an initial value that was
  // computed from the selected points, and the function may update `model`
  // if the degeneracy is detected and handled.
  // The argument `initial_inliers` is the set of inliers for the initial
  // model.
  // The function returns true if the `model` is updated, false otherwise.
  virtual bool handleDegeneracy(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points,
      const std::vector<size_t>& initial_inliers, Parameter* model);

 protected:
  DegeneracyAwareOptions degen_params_;
};

// PlaneAndParallaxEstimator is used as a inner estimator in
// DegeneracyAwareFundamentalEstimator when degeneracy is detected.
class PlaneAndParallaxEstimator : public TwoViewGeometryEstimator {
 public:
  static constexpr int kMinimalSampleSize = 2;
  explicit PlaneAndParallaxEstimator(
      const Eigen::Matrix3d& H = Eigen::Matrix3d::Identity())
      : H_(H) {}
  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    return getParametersFromMatrices(
        TwoViewGeometryType::FUNDAMENTAL,
        FundamentalMatrix::solveWithKnownH(selected_indices, all_points, H_));
  }

  virtual bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;

  static Ransac<PlaneAndParallaxEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<PlaneAndParallaxEstimator>(options).solve(point_pairs);
  }

 private:
  Eigen::Matrix3d H_;
};

////////////////////////////////////////////////////////////////////////
// Below we provide some widely used estimators.
////////////////////////////////////////////////////////////////////////

class HomographyEstimator : public TwoViewGeometryEstimator {
 public:
  static constexpr int kMinimalSampleSize = 4;
  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    return getParametersFromMatrices(
        TwoViewGeometryType::HOMOGRAPHY,
        HomographyMatrix::solveDLT(selected_indices, all_points));
  }

  static Ransac<HomographyEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<HomographyEstimator>(options).solve(point_pairs);
  }
};

class Fundamental7PointEstimator : public DegeneracyAwareFundamentalEstimator {
 public:
  static constexpr int kMinimalSampleSize = 7;
  Fundamental7PointEstimator(
      const DegeneracyAwareOptions& degen_params = DegeneracyAwareOptions())
      : DegeneracyAwareFundamentalEstimator(degen_params) {}

  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    return getParametersFromMatrices(
        TwoViewGeometryType::FUNDAMENTAL,
        FundamentalMatrix::solveWith7Points(selected_indices, all_points));
  }

  static Ransac<Fundamental7PointEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<Fundamental7PointEstimator>(options).solve(point_pairs);
  }

  using Scoring = EpipolarDegensacScoring;
  static Ransac<Fundamental7PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& ransac_options = defaultRansacOptions()) {
    DegeneracyAwareOptions options(true, ransac_options);
    return degensac(point_pairs, options);
  }
  static Ransac<Fundamental7PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const DegeneracyAwareOptions& options) {
    return Ransac<Fundamental7PointEstimator, Scoring>(
               options.ransac_options, Fundamental7PointEstimator(options),
               Scoring(options.degenerate_score_multiplier))
        .solve(point_pairs);
  }
};

class Fundamental8PointEstimator : public DegeneracyAwareFundamentalEstimator {
 public:
  static constexpr int kMinimalSampleSize = 8;
  Fundamental8PointEstimator(
      const DegeneracyAwareOptions& degen_params = DegeneracyAwareOptions())
      : DegeneracyAwareFundamentalEstimator(degen_params) {}

  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    return getParametersFromMatrices(
        TwoViewGeometryType::FUNDAMENTAL,
        FundamentalMatrix::solveWith8Points(selected_indices, all_points));
  }

  static Ransac<Fundamental8PointEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<Fundamental8PointEstimator>(options).solve(point_pairs);
  }

  using Scoring = EpipolarDegensacScoring;
  static Ransac<Fundamental8PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& ransac_options = defaultRansacOptions()) {
    DegeneracyAwareOptions options(true, ransac_options);
    return degensac(point_pairs, options);
  }
  static Ransac<Fundamental8PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const DegeneracyAwareOptions& options) {
    return Ransac<Fundamental8PointEstimator, Scoring>(
               options.ransac_options, Fundamental8PointEstimator(options),
               Scoring(options.degenerate_score_multiplier))
        .solve(point_pairs);
  }
};

class Essential5PointEstimator : public DegeneracyAwareEssentialEstimator {
 public:
  static constexpr int kMinimalSampleSize = 5;
  Essential5PointEstimator(
      const DegeneracyAwareOptions& degen_params = DegeneracyAwareOptions())
      : DegeneracyAwareEssentialEstimator(degen_params) {}
  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    return getParametersFromMatrices(
        TwoViewGeometryType::ESSENTIAL,
        EssentialMatrix::solveWith5Points(selected_indices, all_points));
  }

  static Ransac<Essential5PointEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<Essential5PointEstimator>(options).solve(point_pairs);
  }

  using Scoring = EpipolarDegensacScoring;
  static Ransac<Essential5PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& ransac_options = defaultRansacOptions()) {
    DegeneracyAwareOptions options(true, ransac_options);
    return degensac(point_pairs, options);
  }
  static Ransac<Essential5PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const DegeneracyAwareOptions& options) {
    return Ransac<Essential5PointEstimator, Scoring>(
               options.ransac_options, Essential5PointEstimator(options),
               Scoring(options.degenerate_score_multiplier))
        .solve(point_pairs);
  }
};

class Essential8PointEstimator : public DegeneracyAwareEssentialEstimator {
 public:
  static constexpr int kMinimalSampleSize = 8;
  Essential8PointEstimator(
      const DegeneracyAwareOptions& degen_params = DegeneracyAwareOptions())
      : DegeneracyAwareEssentialEstimator(degen_params) {}
  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const {
    return getParametersFromMatrices(
        TwoViewGeometryType::ESSENTIAL,
        EssentialMatrix::solveWith8Points(selected_indices, all_points));
  }

  static Ransac<Essential8PointEstimator>::Report ransac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    return Ransac<Essential8PointEstimator>(options).solve(point_pairs);
  }

  using Scoring = EpipolarDegensacScoring;

  // NOTE:
  // Our tests showed that the degensac of 8-point-essential is not as stable
  // as that of other estimators.
  static Ransac<Essential8PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& ransac_options = defaultRansacOptions()) {
    DegeneracyAwareOptions options(true, ransac_options);
    return degensac(point_pairs, options);
  }

  static Ransac<Essential8PointEstimator, Scoring>::Report degensac(
      const std::vector<DataPoint>& point_pairs,
      // const DegeneracyAwareOptions& options = DegeneracyAwareOptions(true)) {
      DegeneracyAwareOptions options) {
    if (options.ransac_options.local_opt_max_iter == 0) {
      // For 8-point essential, local optimization is recommended.
      options.ransac_options.local_opt_max_iter = 1;
    }
    return Ransac<Essential8PointEstimator, Scoring>(
               options.ransac_options, Essential8PointEstimator(options),
               Scoring(options.degenerate_score_multiplier))
        .solve(point_pairs);
  }
};

class Essential2PointEstimator : public TwoViewGeometryEstimator {
  Eigen::Matrix3d known_rotation_;

 public:
  static constexpr int kMinimalSampleSize = 2;
  explicit Essential2PointEstimator(const Eigen::Matrix3d& known_rotation)
      : known_rotation_(known_rotation) {}

  virtual std::vector<Parameter> compute(
      const std::vector<size_t>& selected_indices,
      const std::vector<DataPoint>& all_points) const;

  virtual bool localOptimize(
      const std::vector<size_t>& inlier_indices,
      const std::vector<DataPoint>& all_points, Parameter* param) const;

  static Ransac<Essential2PointEstimator>::Report ransac(
      const Eigen::Matrix3d& known_rotation,
      const std::vector<DataPoint>& point_pairs,
      const RansacOptions& options = defaultRansacOptions()) {
    Essential2PointEstimator estimator(known_rotation);
    return Ransac<Essential2PointEstimator>(options, estimator)
        .solve(point_pairs);
  }
};

}  // namespace sk4slam
