#include "sk4slam_cpp/binary_search.h"

#include <cmath>
#include <deque>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

using namespace sk4slam;  // NOLINT

class BinarySearchTest : public ::testing::Test {
 protected:
  std::deque<double> arr{1, 2, 2, 3, 4, 5, 5, 5, 6};

  std::deque<double> empty_arr;

  void SetUp() override {}

  void TearDown() override {}
};

/////////////// Test index search functions  /////////////////////

TEST_F(BinarySearchTest, SearchAnyIndex) {
  auto result = binarySearchAny(arr, 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 5);

  result = binarySearchAny(arr, 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchFirstIndex) {
  auto result = binarySearchFirst(arr, 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 5);

  result = binarySearchFirst(arr, 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 1);

  result = binarySearchFirst(arr, 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchLastIndex) {
  auto result = binarySearchLast(arr, 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 7);

  result = binarySearchLast(arr, 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 2);

  result = binarySearchLast(arr, 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchFirstAndLastIndex) {
  auto result = binarySearchFirstAndLast(arr, 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->first, 5);
  EXPECT_EQ(result->second, 7);

  result = binarySearchFirstAndLast(arr, 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->first, 1);
  EXPECT_EQ(result->second, 2);

  result = binarySearchFirstAndLast(arr, 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchLowerBoundIndex) {
  auto result = binarySearchLowerBound(empty_arr, 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 0);

  result = binarySearchLowerBound(arr, 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 3);

  result = binarySearchLowerBound(arr, 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 5);

  result = binarySearchLowerBound(arr, 4.5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 5);

  result = binarySearchLowerBound(arr, 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 1);

  result = binarySearchLowerBound(arr, 7);
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, arr.size());

  result = binarySearchLowerBound(arr, 0);
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 0);
}

TEST_F(BinarySearchTest, SearchUpperBoundIndex) {
  auto result = binarySearchUpperBound(empty_arr, 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 0);

  result = binarySearchUpperBound(arr, 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 3 + 1);

  result = binarySearchUpperBound(arr, 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 7 + 1);

  result = binarySearchLowerBound(arr, 4.5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 5);

  result = binarySearchUpperBound(arr, 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 2 + 1);

  result = binarySearchUpperBound(arr, 7);
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, arr.size());

  result = binarySearchUpperBound(arr, 0);
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 0);
}

/////////////// Test iterator search functions  /////////////////////

TEST_F(BinarySearchTest, SearchAnyIterator) {
  auto result = binarySearchAny(arr.begin(), arr.end(), 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 5);

  result = binarySearchAny(arr.begin(), arr.end(), 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchFirstIterator) {
  auto result = binarySearchFirst(arr.begin(), arr.end(), 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 5);

  result = binarySearchFirst(arr.begin(), arr.end(), 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 1);

  result = binarySearchFirst(arr.begin(), arr.end(), 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchLastIterator) {
  auto result = binarySearchLast(arr.begin(), arr.end(), 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 7);

  result = binarySearchLast(arr.begin(), arr.end(), 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 2);

  result = binarySearchLast(arr.begin(), arr.end(), 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchFirstAndLastIterator) {
  auto result = binarySearchFirstAndLast(arr.begin(), arr.end(), 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), result->first), 5);
  EXPECT_EQ(std::distance(arr.begin(), result->second), 7);

  result = binarySearchFirstAndLast(arr.begin(), arr.end(), 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), result->first), 1);
  EXPECT_EQ(std::distance(arr.begin(), result->second), 2);

  result = binarySearchFirstAndLast(arr.begin(), arr.end(), 7);
  // EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(result);
}

TEST_F(BinarySearchTest, SearchLowerBoundIterator) {
  auto result = binarySearchLowerBound(empty_arr.begin(), empty_arr.end(), 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, empty_arr.begin());

  result = binarySearchLowerBound(arr.begin(), arr.end(), 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 3);

  result = binarySearchLowerBound(arr.begin(), arr.end(), 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 5);

  result = binarySearchLowerBound(arr.begin(), arr.end(), 4.5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 5);

  result = binarySearchLowerBound(arr.begin(), arr.end(), 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 1);

  result = binarySearchLowerBound(arr.begin(), arr.end(), 7);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, arr.end());

  result = binarySearchLowerBound(arr.begin(), arr.end(), 0);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, arr.begin());
}

TEST_F(BinarySearchTest, SearchUpperBoundIterator) {
  auto result = binarySearchLowerBound(empty_arr.begin(), empty_arr.end(), 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, empty_arr.begin());

  result = binarySearchUpperBound(arr.begin(), arr.end(), 3);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 3 + 1);

  result = binarySearchUpperBound(arr.begin(), arr.end(), 5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 7 + 1);

  result = binarySearchUpperBound(arr.begin(), arr.end(), 4.5);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 5);

  result = binarySearchUpperBound(arr.begin(), arr.end(), 2);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(std::distance(arr.begin(), *result), 2 + 1);

  result = binarySearchUpperBound(arr.begin(), arr.end(), 7);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(result, arr.end());

  result = binarySearchUpperBound(arr.begin(), arr.end(), 0);
  // ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, arr.begin());
}

SK4SLAM_UNITTEST_ENTRYPOINT
