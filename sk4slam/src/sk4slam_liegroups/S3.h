#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/SO3.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/liegroup_base.h"

namespace sk4slam {

template <typename ScalarType>
class S3 : public LieGroupBase<S3<ScalarType>> {
  using _LieGroupBase = LieGroupBase<S3<ScalarType>>;

 public:
  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = 3;

  static constexpr int kAmbientDim = 4;

  // Ambient is the vector space that the Lie group is embedded in.
  struct Ambient : public Eigen::Quaternion<Scalar> {
    // Import the constructors from the base class
    using Eigen::Quaternion<Scalar>::Quaternion;

    // Let Ambient be a vector space.
    static Ambient Zero() {
      const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
      return Ambient(kNum_0, kNum_0, kNum_0, kNum_0);
    }
    bool isZero(const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
      return this->coeffs().isZero(eps);
    }
    bool isApprox(
        const Ambient& other,
        const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
      return this->coeffs().isApprox(other.coeffs(), eps);
    }

    Ambient operator+(const Ambient& other) const {
      return Ambient(
          this->w() + other.w(), this->x() + other.x(), this->y() + other.y(),
          this->z() + other.z());
    }

    Ambient operator-(const Ambient& other) const {
      return Ambient(
          this->w() - other.w(), this->x() - other.x(), this->y() - other.y(),
          this->z() - other.z());
    }

    Ambient operator*(const Scalar& scalar) const {
      return Ambient(
          this->w() * scalar, this->x() * scalar, this->y() * scalar,
          this->z() * scalar);
    }

    Ambient operator/(const Scalar& scalar) const {
      return Ambient(
          this->w() / scalar, this->x() / scalar, this->y() / scalar,
          this->z() / scalar);
    }

    Ambient operator-() const {
      return Ambient(-this->w(), -this->x(), -this->y(), -this->z());
    }

    Ambient& operator+=(const Ambient& other) {
      *this = *this + other;
      return *this;
    }

    Ambient& operator-=(const Ambient& other) {
      *this = *this - other;
      return *this;
    }

    Ambient& operator*=(const Scalar& scalar) {
      *this = *this * scalar;
      return *this;
    }

    Ambient& operator/=(const Scalar& scalar) {
      *this = *this / scalar;
      return *this;
    }

    // Let Ambient be an algebra.
    static Ambient Identity() {
      return Ambient(Eigen::Quaternion<Scalar>::Identity());
    }

    Ambient operator*(const Ambient& other) const {
      using Base = Eigen::Quaternion<Scalar>;
      return Ambient(this->Base::operator*(other));
    }

    Ambient& operator*=(const Ambient& other) {
      *this = *this * other;
      return *this;
    }

    // opration on R3
    Eigen::Matrix<Scalar, 3, 1> operator*(
        const Eigen::Matrix<Scalar, 3, 1>& v) const {
      using Base = Eigen::Quaternion<Scalar>;
      return this->Base::operator*(v);
    }
  };

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static S3 Identity() {
    return S3(Eigen::Quaternion<Scalar>::Identity());
  }

  S3 operator*(const S3& other) const {
    return S3(q_ * other.q_);
  }

  S3 inverse() const {
    return S3(q_.conjugate());
  }

  bool isApprox(
      const S3& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return q_.isApprox(other.q_, eps);
  }

  static S3 Exp(const LieAlgebra& half_w) {
    using std::cos;
    using std::sin;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    Scalar theta = half_w.norm();
    Scalar sintheta_on_theta;
    if (theta < liegroup::Constants<Scalar>::kEps) {
      return S3(Eigen::Quaternion<Scalar>(
          kNum_1, half_w.x(), half_w.y(), half_w.z()));
    } else {
      Scalar sintheta_on_theta = sin(theta) / theta;
      LieAlgebra v = sintheta_on_theta * half_w;
      return S3(Eigen::Quaternion<Scalar>(cos(theta), v.x(), v.y(), v.z()));
    }
  }

  static LieAlgebra Log(const S3& g) {
    using std::abs;
    using std::acos;
    using std::sin;

    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    const Scalar& kNum_0p5 = liegroup::Constants<Scalar>::kNum_0p5;
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    const Scalar& kNum_neg1 = liegroup::Constants<Scalar>::kNum_neg1;
    const Scalar& eps = liegroup::Constants<Scalar>::kEps;

    const Eigen::Quaternion<Scalar>& q = g.q_;
    Scalar costheta = q.w();
    costheta = costheta > kNum_1 ? kNum_1 : costheta;
    costheta = costheta < kNum_neg1 ? kNum_neg1 : costheta;
    Scalar theta = acos(costheta);
    Scalar sintheta = sin(theta);
    if (abs(sintheta) < eps) {
      if (costheta > kNum_0) {
        return LieAlgebra(q.x(), q.y(), q.z());
      } else {
        // The axis is arbitrary, so we choose the x-axis.
        return LieAlgebra(
            liegroup::Constants<Scalar>::kPI,
            liegroup::Constants<Scalar>::kNum_0,
            liegroup::Constants<Scalar>::kNum_0);
      }
    }

    Scalar theta_on_sintheta = theta / sintheta;
    LieAlgebra v(q.x(), q.y(), q.z());
    return theta_on_sintheta * v;
  }

  static LieAlgebraEndomorphism Ad(const S3& g) {
    return g.q_.toRotationMatrix();
  }

  // Ad(g, X) = Ad(g) * X
  static LieAlgebra Ad(const S3& g, const LieAlgebra& half_w) {
    return g.q_ * half_w;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& half_w) {
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    return SO3<Scalar>::hat(kNum_2 * half_w);
  }

  static LieAlgebra bracket(
      const LieAlgebra& half_w1, const LieAlgebra& half_w2) {
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    return half_w1.cross(kNum_2 * half_w2);
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& half_w) {
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    return SO3<Scalar>::Jl(kNum_2 * half_w);
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& half_w) {
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    return SO3<Scalar>::Jr(kNum_2 * half_w);
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& half_w) {
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    return SO3<Scalar>::invJl(kNum_2 * half_w);
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& half_w) {
    const Scalar& kNum_2 = liegroup::Constants<Scalar>::kNum_2;
    return SO3<Scalar>::invJr(kNum_2 * half_w);
  }

  static Ambient hat(const LieAlgebra& half_w) {
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    return Ambient(kNum_0, half_w.x(), half_w.y(), half_w.z());
  }

  static LieAlgebra vee(const Ambient& half_w_hat) {
    return LieAlgebra(half_w_hat.x(), half_w_hat.y(), half_w_hat.z());
  }

  static Ambient generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    LieAlgebra half_w = LieAlgebra::Zero();
    half_w(i) = liegroup::Constants<Scalar>::kNum_1;
    return hat(half_w);
  }

  // For use with CeresManifoldBlock
  ScalarType* data() {
    return q_.coeffs().data();
  }

  const ScalarType* data() const {
    return q_.coeffs().data();
  }

  template <typename _ScalarType>
  S3<_ScalarType> cast() const {
    return S3<_ScalarType>(q_.template cast<_ScalarType>());
  }

 public:
  // S3 specific interfaces

  S3() : q_(Eigen::Quaternion<Scalar>::Identity()) {}

  explicit S3(const Eigen::Quaternion<Scalar>& q) : q_(q) {}

  const Ambient& q() const {
    return q_;
  }

  using YawOnly =
      typename _LieGroupBase::template SubLeftOptimizable<SubSpaceByAxes<2>>;

 private:
  Ambient q_;
};

template <typename Scalar>
S3<Scalar>* asS3(Eigen::Quaternion<Scalar>* quaternion) {
  return reinterpret_cast<S3<Scalar>*>(quaternion);
}

template <typename Scalar>
const S3<Scalar>* asS3(const Eigen::Quaternion<Scalar>* quaternion) {
  return reinterpret_cast<const S3<Scalar>*>(quaternion);
}

template <typename Scalar>
S3<Scalar>* asS3(
    Scalar* quaternion_data) {  // quaternion_data should be in xyzw order.
  return reinterpret_cast<S3<Scalar>*>(quaternion_data);
}

template <typename Scalar>
const S3<Scalar>* asS3(const Scalar* quaternion_data) {
  return reinterpret_cast<const S3<Scalar>*>(quaternion_data);
}

using S3d = S3<double>;

}  // namespace sk4slam
