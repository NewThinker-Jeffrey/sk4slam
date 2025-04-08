#include "sk4slam_basic/reflection.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

namespace my_ns {
class A {};
class B : public A {};

class C {
 public:
  virtual ~C() {}
};
class D : public C {
 public:
  static int bar(bool x, double y) {
    return 0;
  }
  int foo(bool x, double y) {
    return 0;
  }
};
}  // namespace my_ns

TEST(TestReflection, classname) {
  using namespace my_ns;  // NOLINT
  using sk4slam::classname;

  LOGI("classname<int>(): %s", classname<int>());  // "int"
  LOGI("classname<A>(): %s", classname<A>());      // "my_ns::A"

  A* b = new B();
  C* d = new D();
  LOGI("type of b: %s", classname(b).c_str());  // "my_ns::A*"
  LOGI("type of d: %s", classname(d).c_str());  // "my_ns::C*"
  LOGI(
      "type of *b: %s",
      classname(*b).c_str());  // "my_ns::A" (since A has no vtable)
  LOGI(
      "type of *d: %s",
      classname(*d).c_str());  // "my_ns::D" (since C has a vtable)
  LOGI(
      "type of (D::bar): %s",
      classname(D::bar)
          .c_str());  // "int (bool, double)"   (static member function)
  LOGI(
      "type of (&D::bar): %s",
      classname(&D::bar).c_str());  // "int (*)(bool, double)"  (pointer to
                                    // static member function)
  LOGI(
      "type of (&D::foo): %s",
      classname(&D::foo).c_str());  // "int (my_ns::D::*)(bool, double)"
                                    // (pointer to member function)
  delete b;
  delete d;
}

SK4SLAM_UNITTEST_ENTRYPOINT
