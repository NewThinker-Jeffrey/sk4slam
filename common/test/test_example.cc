#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sk4slam_common/entrypoint.h>

// Doc for gtest: https://google.github.io/googletest/

///////////////////////////////////////////////
// TEST(TestCaseName, TestName)
///////////////////////////////////////////////

// TEST(TestSuiteName, TestName1) {  // for newer version of gtest
TEST(TestCaseName, TestName1) {
  std::cout << "TestName1" << std::endl;
}

// TEST(TestSuiteName, TestName2) {  // for newer version of gtest
TEST(TestCaseName, TestName2) {
  std::cout << "TestName2" << std::endl;
}

///////////////////////////////////////////////
// Assertions supported by gtest:
//    ASSERT_*,  EXPECT_*
// For a full list of assertions reference:
// https://google.github.io/googletest/reference/assertions.html#exceptions
///////////////////////////////////////////////

// ASSERT_* versions generate fatal failures when they fail,
// and abort the current function.
//
// EXPECT_* versions generate nonfatal failures, which don’t
// abort the current function.
//
// Usually EXPECT_* are preferred, as they allow more than one
// failure to be reported in a test.
//
// However, you should use ASSERT_* if it doesn’t make sense to
// continue when the assertion in question fails.

TEST(TestAssertions, TestEQ) {
  int a = 1;
  int b = 1;

  // ASSERT_*
  ASSERT_EQ(a, b);
  ASSERT_EQ(a, b) << "Print something if the check failed.";

  // EXPECT_*
  EXPECT_EQ(a, b);
  EXPECT_EQ(a, b) << "Print something if the check failed.";
}

TEST(TestAssertions, TestOtherOperators) {
  int a = 2;
  int b = 1;

  // Here we only test the EXPECT_* versions and
  // don't print anything.

  EXPECT_EQ(a, 2);  // a == 2
  EXPECT_NE(a, b);  // a != b
  EXPECT_GE(a, b);  // a >= b
  EXPECT_LE(b, a);  // b <= a
  EXPECT_GT(a, b);  // a > b
  EXPECT_LT(b, a);  // b < a
}

///////////////////////////////////////////////
// Test fixture
// https://google.github.io/googletest/primer.html#same-data-multiple-tests
///////////////////////////////////////////////

class TestFixtureClass : public testing::Test {
 protected:
  virtual void SetUp() {
    test_data = {1, 2, 3, 4};
  }

  virtual void TearDown() {
    test_data.clear();
  }

  void printTestData() {
    for (const auto item : test_data) {
      std::cout << item << " ";
    }
    std::cout << std::endl;
  }

  std::vector<int> test_data;
};

// Use TEST_F(...) instead of TEST(...)
TEST_F(TestFixtureClass, FixtureTestName1) {
  // We can access the members of TestFixtureClass.
  EXPECT_GE(test_data.size(), 0);
  printTestData();
}

///////////////////////////////////////////////
// TYPED_TEST: Run tests for multiple types.
// https://google.github.io/googletest/reference/testing.html#TYPED_TEST_SUITE
///////////////////////////////////////////////

// Define the templated test fixture.
template <class ScalarType>
class TypedTestFixtureClass : public testing::Test {
 protected:
  virtual void SetUp() {
    test_data = {1, 2, 3, 4};
  }

  virtual void TearDown() {
    test_data.clear();
  }

  void printTestData() {
    for (const auto item : test_data) {
      std::cout << item << " ";
    }
    std::cout << std::endl;
  }

  std::vector<ScalarType> test_data;
};

// Register the types to test.
using MyTypes = ::testing::Types<char, int, unsigned int, float, double>;
// NOTE: use TYPED_TEST_SUITE instead for newer versions of gtest
TYPED_TEST_CASE(TypedTestFixtureClass, MyTypes);

// Use TYPED_TEST(...) instead of TEST(...)
TYPED_TEST(TypedTestFixtureClass, TypedTestName1) {
  // We can access the members of TestFixtureClass.
  EXPECT_GE(this->test_data.size(), 0);
  this->printTestData();
}

SK4SLAM_UNITTEST_ENTRYPOINT
