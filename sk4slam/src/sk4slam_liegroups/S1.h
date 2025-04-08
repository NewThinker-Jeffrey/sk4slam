#pragma once

#include <Eigen/Core>

#include "sk4slam_basic/logging.h"
#include "sk4slam_liegroups/SO2.h"
#include "sk4slam_liegroups/constants.h"
#include "sk4slam_liegroups/liegroup_base.h"

namespace sk4slam {

template <typename ScalarType>
class S1 : public LieGroupBase<S1<ScalarType>> {
 public:
  // Standard LieGroup interfaces

  using Scalar = ScalarType;

  static constexpr int kDim = 1;

  static constexpr int kComplexDim = 2;

  // Complex is the vector space that the Lie group is embedded in.
  using ComplexBase = Eigen::Matrix<Scalar, 2, 1>;
  struct Complex : public ComplexBase {
    // Import the constructors from the base class
    using ComplexBase::ComplexBase;

    // Make Complex be an algebra.
    static Complex Identity() {
      const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
      const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
      return Complex(kNum_1, kNum_0);
    }

    Complex operator*(const Complex& other) const {
      const Scalar& x = this->x();
      const Scalar& y = this->y();
      const Scalar& ox = other.x();
      const Scalar& oy = other.y();
      return Complex(x * ox - y * oy, x * oy + y * ox);
    }

    Complex& operator*=(const Complex& other) {
      *this = *this * other;
      return *this;
    }

    // Operation on R2
    Eigen::Matrix<Scalar, 2, 1> operator*(
        const Eigen::Matrix<Scalar, 2, 1>& v) const {
      const Scalar& c = this->x();
      const Scalar& s = this->y();
      const Scalar& x = v.x();
      const Scalar& y = v.y();
      return Eigen::Matrix<Scalar, 2, 1>(c * x - s * y, c * y + s * x);
    }

    Eigen::Matrix<Scalar, 2, 2> toRotationMatrix() {
      Eigen::Matrix<Scalar, 2, 2> R;
      Scalar* Rd;
      const Scalar& c = this->x();
      const Scalar& s = this->y();
      Rd[0] = c;
      Rd[1] = s;
      Rd[2] = -s;
      Rd[3] = c;
      return R;
    }

    Complex conjugate() const {
      return Complex(this->x(), -this->y());
    }
  };

  using Ambient = Complex;

  using LieAlgebra = Eigen::Matrix<Scalar, kDim, 1>;

  using LieAlgebraEndomorphism = Eigen::Matrix<Scalar, kDim, kDim>;

  static S1 Identity() {
    return S1(Complex::Identity());  // 1, 0
  }

  S1 operator*(const S1& other) const {
    return S1(c_ * other.c_);
  }

  S1 inverse() const {
    return S1(c_.conjugate());
  }

  bool isApprox(
      const S1& other,
      const Scalar& eps = liegroup::Constants<Scalar>::kEps) const {
    return c_.isApprox(other.c_, eps);
  }

  static S1 Exp(const LieAlgebra& w) {
    using std::cos;
    using std::sin;
    const Scalar& theta = w[0];
    return S1(Complex(cos(theta), sin(theta)));
  }

  static LieAlgebra Log(const S1& g) {
    using std::atan2;
    return LieAlgebra(atan2(g.c_.y(), g.c_.x()));
  }

  // S1 is an abelian group, so Ad and ad are trivial.
  static LieAlgebraEndomorphism Ad(const S1& g) {
    return LieAlgebraEndomorphism::Identity();
  }

  // Ad(g, w) = Ad(g) * w
  static LieAlgebra Ad(const S1& g, const LieAlgebra& w) {
    return w;
  }

  static LieAlgebraEndomorphism ad(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Zero();
  }

  static LieAlgebra bracket(const LieAlgebra& w1, const LieAlgebra& w2) {
    return LieAlgebra::Zero();
  }

  // \exp(X+\delta) = \exp(Jl(X) \delta) \exp(X)
  static LieAlgebraEndomorphism Jl(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  // \exp(X+\delta) = \exp(X) \exp(Jr(X) \delta)
  static LieAlgebraEndomorphism Jr(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  // inverse of Jl
  static LieAlgebraEndomorphism invJl(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  // inverse of Jr
  static LieAlgebraEndomorphism invJr(const LieAlgebra& w) {
    return LieAlgebraEndomorphism::Identity();
  }

  static Ambient hat(const LieAlgebra& w) {
    const Scalar& kNum_0 = liegroup::Constants<Scalar>::kNum_0;
    return Ambient(kNum_0, w[0]);
  }

  static LieAlgebra vee(const Ambient& w_hat) {
    return LieAlgebra(w_hat.y());
  }

  static Complex generator(int i) {
    ASSERT(i >= 0 && i < kDim);
    const Scalar& kNum_1 = liegroup::Constants<Scalar>::kNum_1;
    LieAlgebra w(kNum_1);
    return hat(w);
  }

  // For use with CeresManifoldBlock
  ScalarType* data() {
    return c_.data();
  }

  const ScalarType* data() const {
    return c_.data();
  }

  template <typename _ScalarType>
  S1<_ScalarType> cast() const {
    return S1<_ScalarType>(c_.template cast<_ScalarType>());
  }

 public:
  // S1 specific interfaces

  S1() : c_(ComplexBase::Identity()) {}

  explicit S1(const ComplexBase& c) : c_(c) {}

  explicit S1(const Scalar& real, const Scalar& imag) : c_(real, imag) {}

  const Complex& c() const {
    return c_;
  }

 private:
  Complex c_;
};

template <typename Scalar>
S1<Scalar>* asS1(Eigen::Matrix<Scalar, 2, 1>* complex) {
  return reinterpret_cast<S1<Scalar>*>(complex);
}

template <typename Scalar>
const S1<Scalar>* asS1(const Eigen::Matrix<Scalar, 2, 1>* complex) {
  return reinterpret_cast<const S1<Scalar>*>(complex);
}

template <typename Scalar>
S1<Scalar>* asS1(Scalar* complex_data) {
  return reinterpret_cast<S1<Scalar>*>(complex_data);
}

template <typename Scalar>
const S1<Scalar>* asS1(const Scalar* complex_data) {
  return reinterpret_cast<const S1<Scalar>*>(complex_data);
}

using S1d = S1<double>;

}  // namespace sk4slam
