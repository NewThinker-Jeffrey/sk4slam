#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/liegroup_base.h"
#include "sk4slam_liegroups/matrix_group_helper.h"

namespace sk4slam {

template <typename ScalarType>
class SO3;

using SO3d = SO3<double>;

namespace so3_internal {
struct HorizontalHeadingFixedPerturbation;

template <int _heading_axis>
struct YawFixedPerturbation;
}  // namespace so3_internal

template <typename ScalarType>
class SO3 : public LieGroupBase<SO3<ScalarType>>,
            public MatrixGroupCommonOps<SO3<ScalarType>> {
  using _MatrixGroupCommonOps = MatrixGroupCommonOps<SO3<ScalarType>>;

  using _LieGroupBase = LieGroupBase<SO3<ScalarType>>;

 public:
  using _MatrixGroupCommonOps::operator*;
  using _MatrixGroupCommonOps::operator=;

  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = 3;

  static constexpr int kAmbientDim = 9;

  // Ambient is the vector space that the Lie group is embedded in.
  using Ambient = Eigen::Matrix<Scalar, 3, 3>;

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static SO3 Identity() {
    return SO3(Eigen::Matrix<Scalar, 3, 3>::Identity());
  }

  SO3 operator*(const SO3& other) const {
    return SO3(rotation_matrix_ * other.rotation_matrix_);
  }

  SO3 inverse() const {
    return SO3(rotation_matrix_.transpose());
  }

  bool isApprox(
      const SO3& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return rotation_matrix_.isApprox(other.rotation_matrix_, eps);
  }

  /// @brief      The exponential map on the Lie algebra of SO3.
  /// @param w    The Lie algebra element. @p w is a 3-vector.
  /// @return     The corresponding Lie group element.
  ///
  /// @note       This implements the Rodrigues' rotation formula:
  /// @f[
  /// \exp(w) = \cos(|w|) I + \frac{\sin(|w|)}{|w|} w + (1 - \cos(|w|)) w w^T
  /// @f]
  /// Or in terms of the skew-symmetric matrix @f$ [w]_\times @f$:
  /// @f[
  /// \exp(w) = I + \frac{\sin(|w|)}{|w|} [w]_\times + (1 - \cos(|w|))
  /// [w]_\times^2
  /// @f]
  static SO3 Exp(const LieAlgebra& w) {
    return SO3(expM(w));
  }

  static LieAlgebra Log(const SO3& g) {
    return logM(g.matrix());
  }

  static LieAlgebraEndomorphism Ad(const SO3& g) {
    return g.matrix();
  }

  // Ad(g, w) = Ad(g) * w
  static LieAlgebra Ad(const SO3& g, const LieAlgebra& w) {
    return g.rotation_matrix_ * w;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& w) {
    return hat(w);
  }

  static LieAlgebra bracket(const LieAlgebra& w1, const LieAlgebra& w2) {
    return w1.cross(w2);
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& w) {
    return JlImpl(w);
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& w) {
    return JlImpl(-w);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& w) {
    return invJlImpl(w);
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& w) {
    return invJlImpl(-w);
  }

  static Ambient hat(const LieAlgebra& w) {
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    Eigen::Matrix<Scalar, 3, 3> skew_matrix;
    // // clang-format off
    // skew_matrix <<   kNum_0,  -w(2),   w(1),
    //                    w(2), kNum_0,  -w(0),
    //                   -w(1),   w(0),  kNum_0;
    // // clang-format on
    Scalar* m = skew_matrix.data();
    const Scalar& x = w[0];
    const Scalar& y = w[1];
    const Scalar& z = w[2];
    m[0] = kNum_0;
    m[1] = z;
    m[2] = -y;
    m[3] = -z;
    m[4] = kNum_0;
    m[5] = x;
    m[6] = y;
    m[7] = -x;
    m[8] = kNum_0;
    return skew_matrix;
  }

  static LieAlgebra vee(const Ambient& w_hat) {
    return LieAlgebra(w_hat(2, 1), w_hat(0, 2), w_hat(1, 0));
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    LieAlgebra w = LieAlgebra::Zero();
    w(i) = liegroup::Constants<Scalar>::kNum_1;
    return hat(w);
  }

  // For use with CeresManifoldBlock
  ScalarType* data() {
    return rotation_matrix_.data();
  }

  const ScalarType* data() const {
    return rotation_matrix_.data();
  }

  template <typename _ScalarType>
  SO3<_ScalarType> cast() const {
    return SO3<_ScalarType>(rotation_matrix_.template cast<_ScalarType>());
  }

 public:  // MatrixGroupCommonOps overrides.
  // Override MatrixGroupCommonOps::JmultVector().
  // Return
  //    J = d [Exp(x) * v] / d (x)      (Evaluated at x=0, x ∈ LieAlg, v ∈ R^N)
  template <typename = SO3>
  static Eigen::Matrix<Scalar, 3, kDim> JmultVector(
      const Vector<3, Scalar>& v) {
    return hat(-v);
  }

 public:
  // SO3 specific interfaces

  SO3() : rotation_matrix_(Eigen::Matrix<Scalar, 3, 3>::Identity()) {}

  template <typename MatrixXpr, ENABLE_IF(IsMatrixXpr<MatrixXpr>)>
  SO3(const MatrixXpr& m) {  // NOLINT
    static_assert(
        (MatrixXpr::ColsAtCompileTime == 3 ||
         MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) &&
            (MatrixXpr::RowsAtCompileTime == 3 ||
             MatrixXpr::RowsAtCompileTime == Eigen::Dynamic),
        "Matrix dimension mismatch!");
    if constexpr (MatrixXpr::ColsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.cols() == 3);
    }
    if constexpr (MatrixXpr::RowsAtCompileTime == Eigen::Dynamic) {
      ASSERT(m.rows() == 3);
    }
    rotation_matrix_ = m;
  }

  SO3(const Eigen::Matrix<Scalar, 3, 3>& rotation_matrix)  // NOLINT
  : rotation_matrix_(rotation_matrix) {}

  const Eigen::Matrix<Scalar, 3, 3>& matrix() const {
    return rotation_matrix_;
  }

  static SO3 Perfect(const Eigen::Matrix<Scalar, 3, 3>& R) {
    // Return a perfect rotation matrix.
    // return SO3(expM(logM(*R)));
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return SO3(svd.matrixU() * svd.matrixV().transpose());
  }

  static SO3 Perfect(const SO3& R) {
    return Perfect(R.matrix());
  }

  void normalize() {
    rotation_matrix_ = Perfect(rotation_matrix_).matrix();
  }

  SO3 normalized() const {
    return Perfect(*this);
  }

  static SO3 FromTwoUnitVectors(
      const Eigen::Matrix<Scalar, 3, 1>& from,
      const Eigen::Matrix<Scalar, 3, 1>& to) {
    return SO3::Exp(axisAngleFromTwoUnitVectors(from, to));
  }

  /// Override _LieGroupBase::LeftPerturbation::TransformVector()
  template <typename LieGroup>
  struct LeftPerturbationTemplate
      : public _LieGroupBase::template LeftPerturbationTemplate<LieGroup> {
    using _Base =
        typename _LieGroupBase::template LeftPerturbationTemplate<LieGroup>;
    using _Base::kDof;

    template <
        typename _LieGroup, typename VectorXpr, typename ResultVector,
        typename JacobianWrtG =
            Eigen::Matrix<typename _LieGroup::Scalar, Eigen::Dynamic, kDim>>
    void TransformVector(
        const _LieGroup& g, const VectorXpr& v, ResultVector* result,
        JacobianWrtG* jacobian_wrt_g = nullptr) const {
      static_assert(kDof == _LieGroup::kDim);
      static_assert(
          MatrixGroupHelper<_LieGroup>::kIsMatrixGroup,
          "Non-matrix group can not be multiplied with a vector!");
      if (!jacobian_wrt_g && !result) {
        return;
      }
      auto gv = g * v;
      if (jacobian_wrt_g) {
        // The default implementation in _LieGroupBase::LeftPerturbation is:
        //     *jacobian_wrt_g = g.matrix() * _LieGroup::JmultVector(v) *
        //                 EndomophismMatrix(_LieGroup::Ad(g.inverse()));
        // which is equivalent to (a more efficient implementation):
        *jacobian_wrt_g = hat(-gv);
      }
      if (result) {
        *result = gv;
      }
    }
  };

  /// Nothing to override for RightPerturbation.
  template <typename LieGroup>
  using RightPerturbationTemplate =
      typename _LieGroupBase::template RightPerturbationTemplate<LieGroup>;

  DEFINE_LIE_PERTURBATIONS(SO3);
  DEFINE_LIE_OPTIMIZABLES(SO3);

  // SO3 specific perturbations
  using YawOnlyPerturbation = SubLeftPerturbation<SubSpaceByAxes<2>>;

  template <int _heading_axis>
  using YawFixedPerturbation =
      so3_internal::YawFixedPerturbation<_heading_axis>;

  using YawOnly = SubLeftOptimizable<SubSpaceByAxes<2>>;

  template <int _heading_axis>
  using YawFixed =
      OptimizableManifold<SO3, YawFixedPerturbation<_heading_axis>>;

 public:
  static Eigen::Matrix<Scalar, 3, 3> expM(
      const Eigen::Matrix<Scalar, 3, 1>& w) {
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;

    using std::cos;
    using std::sin;

    const Scalar theta = w.norm();
    Eigen::Matrix<Scalar, 3, 3> R;

    // ceres::Jet supports compare operators (e.g. <=) and numeric_limits,
    // see ceres/jet.h for:
    //     CERES_DEFINE_JET_COMPARISON_OPERATOR(<=)
    //     numeric_limits<ceres::Jet<T, N>>
    // COMMENTS-UPDATE:
    // std::numeric_limits is only specialized for Jet in newer versions of
    // ceres, see the comments for liegroup::Constants::kEps in constants.h.
    if (theta > eps) {
      const Scalar costheta = cos(theta);
      const Scalar sintheta = sin(theta);

      // auto w_on_theta = w / theta;
      // R = Eigen::Matrix<Scalar, 3, 3>::Identity() * costheta +
      //     hat(w_on_theta) * sintheta +
      //     (w_on_theta * w_on_theta.transpose()) * (kNum_1 - cos(theta));

      const Scalar wx = w[0] / theta;
      const Scalar wy = w[1] / theta;
      const Scalar wz = w[2] / theta;

      const Scalar coeff_diag = costheta;
      const Scalar coeff_symmetry = kNum_1 - costheta;
      const Scalar coeff_skew = sintheta;

      Scalar* Rd = R.data();

      // clang-format off
      Rd[0] /*= R(0, 0)*/ = coeff_diag + wx * wx * coeff_symmetry;
      Rd[1] /*= R(1, 0)*/ = wz * coeff_skew + wx * wy * coeff_symmetry;
      Rd[2] /*= R(2, 0)*/ = -wy * coeff_skew + wx * wz * coeff_symmetry;
      Rd[3] /*= R(0, 1)*/ = -wz * coeff_skew + wx * wy * coeff_symmetry;
      Rd[4] /*= R(1, 1)*/ = coeff_diag + wy * wy * coeff_symmetry;
      Rd[5] /*= R(2, 1)*/ = wx * coeff_skew + wy * wz * coeff_symmetry;
      Rd[6] /*= R(0, 2)*/ = wy * coeff_skew + wx * wz * coeff_symmetry;
      Rd[7] /*= R(1, 2)*/ = -wx * coeff_skew + wy * wz * coeff_symmetry;
      Rd[8] /*= R(2, 2)*/ = coeff_diag + wz * wz * coeff_symmetry;
      // clang-format on
    } else {
      // When w is small enough, we can use the first
      // order approximation.

      // R = Eigen::Matrix<Scalar, 3, 3>::Identity() + hat(w);
      Scalar* Rd = R.data();
      // clang-format off
      Rd[0] /*= R(0, 0)*/ = kNum_1;
      Rd[1] /*= R(1, 0)*/ = w[2];
      Rd[2] /*= R(2, 0)*/ = -w[1];
      Rd[3] /*= R(0, 1)*/ = -w[2];
      Rd[4] /*= R(1, 1)*/ = kNum_1;
      Rd[5] /*= R(2, 1)*/ = w[0];
      Rd[6] /*= R(0, 2)*/ = w[1];
      Rd[7] /*= R(1, 2)*/ = -w[0];
      Rd[8] /*= R(2, 2)*/ = kNum_1;
      // clang-format on
    }
    return R;
  }

  static Eigen::Matrix<Scalar, 3, 1> logM(
      const Eigen::Matrix<Scalar, 3, 3>& R) {
    using liegroup::Constants;
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    const Scalar& kNum_neg1 = liegroup::Constants<Scalar>::kNum_neg1;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;

    using std::acos;
    using std::sin;
    using std::sqrt;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

    Vector3 w;
    const Scalar* Rd = R.data();
    Scalar &x = w[0], &y = w[1], &z = w[2];
    x = Rd[5] - Rd[7];  // R(2, 1) - R(1, 2);
    y = Rd[6] - Rd[2];  // R(0, 2) - R(2, 0);
    z = Rd[1] - Rd[3];  // R(1, 0) - R(0, 1);
    Scalar costheta = (R.trace() - kNum_1) / kNum_2;
    costheta = costheta > kNum_1 ? kNum_1 : costheta;
    costheta = costheta < kNum_neg1 ? kNum_neg1 : costheta;
    const Scalar theta = acos(costheta);
    const Scalar sintheta = sin(theta);

    if (sintheta < eps) {
      // Theta ≈ 0  or Theta ≈ π.
      //
      // When theta is 0, then R = I and w = 0;
      //
      // When theta is π, we get the Identity again if applying
      // the R twice (rotation 2π around any axis).
      // So, R * R = I, R is the inverse (also transpose) of itself,
      // i.e. R = R^T.

      if (costheta < kNum_0) {
        // Theta ≈ π.
        //
        // If a base vector v is perpendicular to the axis of rotation,
        // then v' dot v = -1, where v' is the rotated version of v, i.e.
        // v and v' are antiparallel. If v is not perpendicular to the
        // axis of rotation, then v and v' are not antiparallel and
        // v' dot v > -1.
        //
        // There's at least one base vector that is NOT perpendicular
        // to the axis of rotation, i.e. at least one of the diagonal
        // elements of R is greater than -1.
        LOGA("SO3::logM: Theta is close to 180 degrees!");
        // const Scalar& PI = Scalar(M_PI);
        const Scalar& kAntiParallelDotThr = Scalar(1e-4) - kNum_1;
        if (R(2, 2) > kAntiParallelDotThr) {
          w = (theta / sqrt(kNum_2 + kNum_2 * R(2, 2))) *
              Vector3(R(0, 2), R(1, 2), kNum_1 + R(2, 2));
        } else if (R(1, 1) > kAntiParallelDotThr) {
          w = (theta / sqrt(kNum_2 + kNum_2 * R(1, 1))) *
              Vector3(R(0, 1), kNum_1 + R(1, 1), R(2, 1));
        } else {
          ASSERT(R(0, 0) > kAntiParallelDotThr);
          w = (theta / sqrt(kNum_2 + kNum_2 * R(0, 0))) *
              Vector3(kNum_1 + R(0, 0), R(1, 0), R(2, 0));
        }
        return w;
      } else {
        // Theta ≈ 0.
        w *= kNum_0p5;
        return w;
      }
    }

    const Scalar theta_on_sintheta = theta / sintheta;
    const Scalar half_theta_on_sintheta = kNum_0p5 * theta_on_sintheta;
    w *= half_theta_on_sintheta;
    return w;
  }

  static Eigen::Matrix<Scalar, 3, 1> axisAngleFromTwoUnitVectors(
      const Eigen::Matrix<Scalar, 3, 1>& from,
      const Eigen::Matrix<Scalar, 3, 1>& to) {
    using std::abs;
    using std::asin;
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kEps = liegroup::Constants<Scalar>::kEps;
    const Scalar& kPI = liegroup::Constants<Scalar>::kPI;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;

    const Eigen::Matrix<Scalar, 3, 1> cross = from.cross(to);
    Scalar dot = from.dot(to);
    Scalar sintheta = cross.norm();
    if (sintheta < kEps) {
      if (dot < kNum_0) {
        // theta is close to 180 degrees, so the axis can be any
        // vector orthogonal to `from`.
        static const auto X = Eigen::Matrix<Scalar, 3, 1>::UnitX();
        static const auto Y = Eigen::Matrix<Scalar, 3, 1>::UnitY();
        Eigen::Matrix<Scalar, 3, 1> w;
        if (abs(from.x()) > abs(from.y())) {
          w = from.cross(Y);
        } else {
          w = from.cross(X);
        }
        w.normalize();
        return w * kPI;
      } else {
        // theta is small
        return cross;
      }
    } else {
      Scalar theta_on_sintheta;
      if (dot < kNum_0) {
        theta_on_sintheta = (kPI - asin(sintheta)) / sintheta;
      } else {
        theta_on_sintheta = asin(sintheta) / sintheta;
      }
      return cross * theta_on_sintheta;
    }
  }

 protected:
  static LieAlgebraEndomorphism JlImpl(const LieAlgebra& w) {
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar theta = w.norm();

    using std::cos;
    using std::sin;

    LieAlgebraEndomorphism J;
    if (theta > eps) {
      const Scalar sintheta_on_theta = sin(theta) / theta;

      // LieAlgebra a = w / theta;
      // LieAlgebraEndomorphism J =
      //     sintheta_on_theta * LieAlgebraEndomorphism::Identity() +
      //     (kNum_1 - sintheta_on_theta) * a * a.transpose() +
      //     ((kNum_1 - cos(theta)) / theta) * hat(a);

      const Scalar wx = w[0] / theta;
      const Scalar wy = w[1] / theta;
      const Scalar wz = w[2] / theta;

      const Scalar coeff_diag = sintheta_on_theta;
      const Scalar coeff_symmetry = kNum_1 - sintheta_on_theta;
      const Scalar coeff_skew = (kNum_1 - cos(theta)) / theta;

      Scalar* Jd = J.data();

      // clang-format off
      Jd[0] /*= J(0, 0)*/ = coeff_diag + wx * wx * coeff_symmetry;
      Jd[1] /*= J(1, 0)*/ = wz * coeff_skew + wx * wy * coeff_symmetry;
      Jd[2] /*= J(2, 0)*/ = -wy * coeff_skew + wx * wz * coeff_symmetry;
      Jd[3] /*= J(0, 1)*/ = -wz * coeff_skew + wx * wy * coeff_symmetry;
      Jd[4] /*= J(1, 1)*/ = coeff_diag + wy * wy * coeff_symmetry;
      Jd[5] /*= J(2, 1)*/ = wx * coeff_skew + wy * wz * coeff_symmetry;
      Jd[6] /*= J(0, 2)*/ = wy * coeff_skew + wx * wz * coeff_symmetry;
      Jd[7] /*= J(1, 2)*/ = -wx * coeff_skew + wy * wz * coeff_symmetry;
      Jd[8] /*= J(2, 2)*/ = coeff_diag + wz * wz * coeff_symmetry;
      // clang-format on
    } else {
      // When w is small enough, we can use the first
      // order approximation.

      // J = LieAlgebraEndomorphism::Identity() +
      // liegroup::Constants<Scalar>::kNum_0p5 * hat(w);
      const Scalar kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
      Scalar* Jd = J.data();
      // clang-format off
      Jd[0] /*= J(0, 0)*/ = kNum_1;
      Jd[1] /*= J(1, 0)*/ = kNum_0p5 * w[2];
      Jd[2] /*= J(2, 0)*/ = -kNum_0p5 * w[1];
      Jd[3] /*= J(0, 1)*/ = -kNum_0p5 * w[2];
      Jd[4] /*= J(1, 1)*/ = kNum_1;
      Jd[5] /*= J(2, 1)*/ = kNum_0p5 * w[0];
      Jd[6] /*= J(0, 2)*/ = kNum_0p5 * w[1];
      Jd[7] /*= J(1, 2)*/ = -kNum_0p5 * w[0];
      Jd[8] /*= J(2, 2)*/ = kNum_1;
      // clang-format on
    }

    return J;
  }

  static LieAlgebraEndomorphism invJlImpl(const LieAlgebra& w) {
    // return JlImpl(w).inverse();

    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;
    const Scalar theta = w.norm();

    using std::cos;
    using std::sin;

    LieAlgebraEndomorphism invJ;
    if (theta > eps) {
      const Scalar theta_on_sintheta = theta / sin(theta);

      // LieAlgebra a = w / theta;
      // LieAlgebraEndomorphism invJ =
      //     (kNum_0p5 * (cos(theta) + kNum_1) * theta_on_sintheta) *
      //              LieAlgebraEndomorphism::Identity() +
      //     (kNum_1 - kNum_0p5 * (cos(theta) + kNum_1) * theta_on_sintheta) *
      //              a * a.transpose() +
      //     ((kNum_1 - cos(theta)) / theta) * hat(a);

      const Scalar wx = w[0] / theta;
      const Scalar wy = w[1] / theta;
      const Scalar wz = w[2] / theta;

      const Scalar coeff_diag =
          kNum_0p5 * (cos(theta) + kNum_1) * theta_on_sintheta;
      const Scalar coeff_symmetry = kNum_1 - coeff_diag;
      const Scalar coeff_skew = -kNum_0p5 * theta;

      Scalar* invJd = invJ.data();

      // clang-format off
      invJd[0] /*= invJ(0, 0)*/ = coeff_diag + wx * wx * coeff_symmetry;
      invJd[1] /*= invJ(1, 0)*/ = wz * coeff_skew + wx * wy * coeff_symmetry;
      invJd[2] /*= invJ(2, 0)*/ = -wy * coeff_skew + wx * wz * coeff_symmetry;
      invJd[3] /*= invJ(0, 1)*/ = -wz * coeff_skew + wx * wy * coeff_symmetry;
      invJd[4] /*= invJ(1, 1)*/ = coeff_diag + wy * wy * coeff_symmetry;
      invJd[5] /*= invJ(2, 1)*/ = wx * coeff_skew + wy * wz * coeff_symmetry;
      invJd[6] /*= invJ(0, 2)*/ = wy * coeff_skew + wx * wz * coeff_symmetry;
      invJd[7] /*= invJ(1, 2)*/ = -wx * coeff_skew + wy * wz * coeff_symmetry;
      invJd[8] /*= invJ(2, 2)*/ = coeff_diag + wz * wz * coeff_symmetry;
      // clang-format on
    } else {
      // When w is small enough, we can use the first
      // order approximation.

      // J^{-1} = LieAlgebraEndomorphism::Identity() -
      // liegroup::Constants<Scalar>::kNum_0p5 * hat(w);
      const Scalar kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
      Scalar* invJd = invJ.data();
      // clang-format off
      invJd[0] /*= invJ(0, 0)*/ = kNum_1;
      invJd[1] /*= invJ(1, 0)*/ = -kNum_0p5 * w[2];
      invJd[2] /*= invJ(2, 0)*/ = kNum_0p5 * w[1];
      invJd[3] /*= invJ(0, 1)*/ = kNum_0p5 * w[2];
      invJd[4] /*= invJ(1, 1)*/ = kNum_1;
      invJd[5] /*= invJ(2, 1)*/ = -kNum_0p5 * w[0];
      invJd[6] /*= invJ(0, 2)*/ = -kNum_0p5 * w[1];
      invJd[7] /*= invJ(1, 2)*/ = kNum_0p5 * w[0];
      invJd[8] /*= invJ(2, 2)*/ = kNum_1;
      // clang-format on
    }

    return invJ;
  }

 private:
  Ambient rotation_matrix_;
};

namespace so3_internal {
/// @brief   A perturbation that fixes the horizontal heading of a rotation.
///          This is particularly useful in VIO when a keyframe is selected
///          to be the reference frame (i.e. the origin of the map), where
///          we want to fix the yaw of the reference frame but still allow
///          the pitch and roll to be optimized.
///
/// In this perturbation, we assume the heading direction of the body frame
/// is always perpendicular to a global 'pitch axis' (usually the global Y
/// axis), thus the heading direction is constrained to a vertical plane
/// (usually the global XOZ plane).
///
/// The perturbation has 2 degrees of freedom:
///    - pitch (global): the rotation around the global pitch axis
///    - roll  (local): the rotation around the local heading direction.
template <typename Derived>
class HorizontalHeadingFixedPerturbation_
    : public RetractionBase<Derived, SO3d> {
  static constexpr bool kCheckHeading = true;
  using _PerturbationInterface = RetractionBase<Derived, SO3d>;
  using LeftPerturbation = typename SO3d::LeftPerturbation;
  using RightPerturbation = typename SO3d::RightPerturbation;
  using RetractionInterface::DeclareTransformJacobianTypes;

 public:
  using _PerturbationInterface::section;
  using _PerturbationInterface::transformJacobian;
  using typename _PerturbationInterface::Manifold;
  using LieGroup = SO3d;
  using Scalar = double;
  static_assert(std::is_same_v<LieGroup, Manifold>);
  static constexpr int kDof = 2;
  static constexpr int kAmbientDim = 9;

 public:
  bool operator==(const Derived& other) const override {
    return derived()->equals(other);
  }

  template <typename _Tangent, typename _SO3>
  _SO3 operator()(const _SO3& g, const _Tangent& local) const {
    using Scalar = typename _SO3::Scalar;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar& kTolerance = Tolerance<Scalar>;
    const Vector3& local_heading =
        derived()->localHeading().template cast<Scalar>();
    const Vector3& global_pitch_axis =
        derived()->globalPitchAxis().template cast<Scalar>();
    Vector3 global_pitch_rot_vec = local[0] * global_pitch_axis;
    Vector3 local_roll_rot_vec = local[1] * local_heading;
    _SO3 ret =
        _SO3::Exp(global_pitch_rot_vec) * g * _SO3::Exp(local_roll_rot_vec);

    if constexpr (kCheckHeading) {
      checkHeading(g);

      // Force the constraint be satisfied (i.e. the heading direction
      // perpendicular to the pitch axis).
      Vector3 global_heading = ret * local_heading;
      Vector3 fixed_global_heading =
          global_pitch_axis.cross(global_heading.cross(global_pitch_axis));
      fixed_global_heading.normalize();

      _SO3 deltaR =
          // _SO3::FromTwoUnitVectors(global_heading, fixed_global_heading);
          _SO3::Exp(global_heading.cross(
              fixed_global_heading));  // Approximation of the above when deltaR
                                       // is small.
      ret = deltaR * ret;
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        fixed_global_heading = ret * local_heading;
        LOGA(
            "HorizontalHeadingFixedPerturbation: global_heading before fix: "
            "%s, after fix: %s;  constraint error before fix: %.20f, "
            "after fix: %.20f",
            toStr(global_heading.transpose()).c_str(),
            toStr(fixed_global_heading.transpose()).c_str(),
            global_heading.dot(global_pitch_axis),
            fixed_global_heading.dot(global_pitch_axis));
      }
    }

    return ret;
  }

  template <typename _Tangent, typename _SO3>
  void sectionImpl(const _SO3& g, const _SO3& g2, _Tangent* delta) const {
    using Scalar = typename _SO3::Scalar;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar& kTolerance = Tolerance<Scalar>;
    const Vector3& local_heading =
        derived()->localHeading().template cast<Scalar>();
    const Vector3& global_pitch_axis =
        derived()->globalPitchAxis().template cast<Scalar>();

    Vector3 global_heading = g * local_heading;
    Vector3 global_heading2 = g2 * local_heading;

    if constexpr (kCheckHeading) {
      using std::abs;
      // Check that the heading direction is perpendicular to the pitch axis.
      Scalar dot = global_heading.dot(
          derived()->globalPitchAxis().template cast<Scalar>());
      Scalar dot2 = global_heading2.dot(
          derived()->globalPitchAxis().template cast<Scalar>());
      if (abs(dot) > kTolerance || abs(dot2) > kTolerance) {
        std::string msg =
            "HorizontalHeadingFixedPerturbation: Heading direction is not "
            "perpendicular to the pitch axis!";
        if constexpr (
            std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
          msg += " dot = " + std::to_string(dot) +
                 ", dot2 = " + std::to_string(dot2);
        }
        throw std::runtime_error(msg);
      }
    }

    Vector3 global_pitch_delta, local_roll_delta;
    {
      // Matrix3 tmpR, tmpR2;
      // tmpR << global_heading, global_pitch_axis,
      //     global_heading.cross(global_pitch_axis);
      // tmpR2 << global_heading2, global_pitch_axis,
      //     global_heading2.cross(global_pitch_axis);
      // _SO3 tmpDeltaR(tmpR2 * tmpR.inverse());
      // global_pitch_delta = _SO3::Log(tmpDeltaR);

      global_pitch_delta =
          _SO3::axisAngleFromTwoUnitVectors(global_heading, global_heading2);
      _SO3 tmpDeltaR = _SO3::Exp(global_pitch_delta);
      // tmpDeltaR shuold satisfies
      //    (tmpDeltaR * g) * local_heading = g2 * local_heading,
      // i.e.
      //    tmpDeltaR * global_heading = global_heading2

      _SO3 tmpg = tmpDeltaR * g;
      local_roll_delta = _SO3::Log(tmpg.inverse() * g2);
    }

    if constexpr (
        std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
      LOGA(
          "HorizontalHeadingFixedPerturbation: "
          "global_pitch_delta = %s, local_roll_delta = %s",
          toStr(global_pitch_delta.transpose()).c_str(),
          toStr(local_roll_delta.transpose()).c_str());
    }

    if constexpr (kCheckHeading) {
      // Check that the global_pitch_delta is parallel to the pitch axis and
      // local_roll_delta is parallel to the local heading.
      Vector3 pitch_cross = global_pitch_delta.cross(global_pitch_axis);
      Vector3 local_cross = local_roll_delta.cross(local_heading);
      if (pitch_cross.norm() > kTolerance || local_cross.norm() > kTolerance) {
        std::string msg =
            "HorizontalHeadingFixedPerturbation: "
            "global_pitch_delta is not parallel to the pitch axis or "
            "local_roll_delta "
            "is not parallel to the local heading!";
        if constexpr (
            std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
          msg += " pitch_cross = " + toStr(pitch_cross.transpose()) +
                 ", local_cross = " + toStr(local_cross.transpose());
        }
        throw std::runtime_error(msg);
      }
    }
    (*delta)[0] = global_pitch_delta.dot(global_pitch_axis);
    (*delta)[1] = local_roll_delta.dot(local_heading);
  }

  template <typename ToPerturbation, typename _Tangent>
  typename LieGroup::LieAlgebra convertPerturbation(
      const _Tangent& tangent, const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    using LeftPerturbation = typename LieGroup::LeftPerturbation;
    using RightPerturbation = typename LieGroup::RightPerturbation;
    static_assert(
        std::is_same_v<ToPerturbation, LeftPerturbation> ||
            std::is_same_v<ToPerturbation, RightPerturbation>,
        "ToPerturbation must be LeftPerturbation or RightPerturbation!");
    LieGroup g2 = (*this)(g, tangent);
    if constexpr (std::is_same_v<ToPerturbation, LeftPerturbation>) {
      return LieGroup::Log(g2 * g.inverse());
    } else {
      static_assert(std::is_same_v<ToPerturbation, RightPerturbation>);
      return LieGroup::Log(g.inverse() * g2);
    }
  }

  template <typename ToPerturbation>
  Eigen::Matrix<Scalar, 3, 2> convertPerturbation(
      const LieGroup& g,
      const ToPerturbation& to_perturbation = ToPerturbation()) const {
    using LeftPerturbation = typename LieGroup::LeftPerturbation;
    using RightPerturbation = typename LieGroup::RightPerturbation;
    static_assert(
        std::is_same_v<ToPerturbation, LeftPerturbation> ||
            std::is_same_v<ToPerturbation, RightPerturbation>,
        "ToPerturbation must be LeftPerturbation or RightPerturbation!");
    if constexpr (kCheckHeading) {
      checkHeading(g);
    }

    Eigen::Matrix<Scalar, 3, 2> J;
    if constexpr (std::is_same_v<ToPerturbation, LeftPerturbation>) {
      J << derived()->globalPitchAxis().template cast<Scalar>(),
          g * derived()->localHeading().template cast<Scalar>();
    } else {
      static_assert(std::is_same_v<ToPerturbation, RightPerturbation>);
      J << g.inverse() * derived()->globalPitchAxis().template cast<Scalar>(),
          derived()->localHeading().template cast<Scalar>();
    }
    return J;
  }

  using TransformJacobianTypes =
      DeclareTransformJacobianTypes<LeftPerturbation, RightPerturbation>;

  template <
      typename SrcPerturbation, typename JacobianMatrixWrtSrcPerturbation,
      typename JacobianMatrixWrtThisPerturbation>
  void transformJacobianImpl(
      const JacobianMatrixWrtSrcPerturbation& jacobian_under_src_perturbation,
      const LieGroup& g,
      JacobianMatrixWrtThisPerturbation* jacobian_under_this_perturbation,
      const SrcPerturbation& src_perturbation = SrcPerturbation()) const {
    // static_assert(
    //         std::is_same_v<SrcPerturbation, LeftPerturbation> ||
    //         std::is_same_v<SrcPerturbation, RightPerturbation>,
    //     "SrcPerturbation must be LeftPerturbation or RightPerturbation!");
    if constexpr (kCheckHeading) {
      checkHeading(g);
    }

    Eigen::Matrix<Scalar, 3, 2> J;
    if constexpr (std::is_same_v<SrcPerturbation, LeftPerturbation>) {
      J << derived()->globalPitchAxis().template cast<Scalar>(),
          g * derived()->localHeading().template cast<Scalar>();
    } else if constexpr (std::is_same_v<SrcPerturbation, RightPerturbation>) {
      J << g.inverse() * derived()->globalPitchAxis().template cast<Scalar>(),
          derived()->localHeading().template cast<Scalar>();
    } else {
      throw std::runtime_error(
          std::string("SrcPerturbation must be LeftPerturbation or "
                      "RightPerturbation! but it is ") +
          classname<SrcPerturbation>());
    }
    *jacobian_under_this_perturbation = jacobian_under_src_perturbation * J;
  }

 private:
  template <typename Scalar>
  static inline const Scalar Tolerance = Scalar(1e-6);

  template <typename _SO3>
  void checkHeading(const _SO3& g) const {
    using std::abs;
    using Scalar = typename _SO3::Scalar;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    const Scalar& kTolerance = Tolerance<Scalar>;
    const Vector3& local_heading =
        derived()->localHeading().template cast<Scalar>();
    const Vector3& global_pitch_axis =
        derived()->globalPitchAxis().template cast<Scalar>();
    // Check that the heading direction is perpendicular to the pitch axis.
    Vector3 global_heading = g * local_heading;
    Scalar dot = global_heading.dot(global_pitch_axis);
    if (abs(dot) > kTolerance) {
      std::string msg =
          "HorizontalHeadingFixedPerturbation: Heading direction is not "
          "perpendicular to the pitch axis!";
      if constexpr (
          std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        msg += " dot = " + std::to_string(dot);
      }
      throw std::runtime_error(msg);
    }
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
};

class HorizontalHeadingFixedPerturbation
    : public HorizontalHeadingFixedPerturbation_<
          HorizontalHeadingFixedPerturbation> {
 public:
  HorizontalHeadingFixedPerturbation(
      const Eigen::Vector3d& local_heading = Eigen::Vector3d::UnitX(),
      const Eigen::Vector3d& global_pitch_axis = Eigen::Vector3d::UnitY())
      : local_heading_(local_heading.normalized()),
        global_pitch_axis_(global_pitch_axis.normalized()) {}

 private:
  // using _Base =
  //     HorizontalHeadingFixedPerturbation_<HorizontalHeadingFixedPerturbation>;
  friend class HorizontalHeadingFixedPerturbation_<
      HorizontalHeadingFixedPerturbation>;

  const Eigen::Vector3d& localHeading() const {
    return local_heading_;
  }

  const Eigen::Vector3d& globalPitchAxis() const {
    return global_pitch_axis_;
  }

  bool equals(const HorizontalHeadingFixedPerturbation& other) const {
    if (this == &other) {
      return true;
    }
    return globalPitchAxis().isApprox(other.globalPitchAxis()) &&
           localHeading().isApprox(other.localHeading());
  }

 private:
  Eigen::Vector3d local_heading_;
  Eigen::Vector3d global_pitch_axis_;
};

template <int _heading_axis>
class YawFixedPerturbation : public HorizontalHeadingFixedPerturbation_<
                                 YawFixedPerturbation<_heading_axis>> {
  static_assert(
      _heading_axis == 0 || _heading_axis == 1 || _heading_axis == 2,
      "Invalid heading axis!");

  // using _Base =
  //     HorizontalHeadingFixedPerturbation_<YawFixedPerturbation<_heading_axis>>;
  friend class HorizontalHeadingFixedPerturbation_<
      YawFixedPerturbation<_heading_axis>>;

  const Eigen::Vector3d& localHeading() const {
    static const Eigen::Vector3d kHeadings[] = {
        Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ()};
    return kHeadings[_heading_axis];
  }

  const Eigen::Vector3d& globalPitchAxis() const {
    static const Eigen::Vector3d kPitchAxis = Eigen::Vector3d::UnitY();
    return kPitchAxis;
  }

  bool equals(const YawFixedPerturbation& other) const {
    return true;
  }
};

}  // namespace so3_internal

template <typename Scalar>
SO3<Scalar>* asSO3(Eigen::Matrix<Scalar, 3, 3>* rotation_matrix) {
  return reinterpret_cast<SO3<Scalar>*>(rotation_matrix);
}

template <typename Scalar>
const SO3<Scalar>* asSO3(const Eigen::Matrix<Scalar, 3, 3>* rotation_matrix) {
  return reinterpret_cast<const SO3<Scalar>*>(rotation_matrix);
}

template <typename Scalar>
SO3<Scalar>* asSO3(Scalar* rotation_matrix_data) {
  return reinterpret_cast<SO3<Scalar>*>(rotation_matrix_data);
}

template <typename Scalar>
const SO3<Scalar>* asSO3(const Scalar* rotation_matrix_data) {
  return reinterpret_cast<const SO3<Scalar>*>(rotation_matrix_data);
}

}  // namespace sk4slam
