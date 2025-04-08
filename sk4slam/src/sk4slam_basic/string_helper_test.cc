#include "sk4slam_basic/string_helper.h"

#include <set>
#include <unordered_set>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestStringHelper, replaceOnce) {
  std::string str = "hello world hello world";
  std::cout << "str before replace: " << str << std::endl;
  int replaced = sk4slam::replaceOnce(&str, "world", "WORLD");
  std::cout << "replaced: " << replaced << std::endl;
  std::cout << "str after replace: " << str << std::endl;
  ASSERT_EQ(replaced, 1);
  ASSERT_EQ(str, "hello WORLD hello world");
}

TEST(TestStringHelper, replaceAll) {
  std::string str = "hello world hello world";
  std::cout << "str before replace: " << str << std::endl;
  int replaced = sk4slam::replaceAll(&str, "world", "WORLD");
  std::cout << "replaced: " << replaced << std::endl;
  std::cout << "str after replace: " << str << std::endl;
  ASSERT_EQ(replaced, 2);
  ASSERT_EQ(str, "hello WORLD hello WORLD");
}

TEST(TestStringHelper, formatStr) {
  std::string str = sk4slam::formatStr("%s %d", "hello", 123);
  std::cout << "formatted: " << str << std::endl;
  ASSERT_EQ(str, "hello 123");
}

TEST(TestStringHelper, Oss) {
  double a = 1.23;
  double b = 4.56789;
  sk4slam::Oss oss;
  oss << a << " " << b;
  oss.format(", %.3f %.3f", a, b);
  std::cout << "oss: " << oss.str() << std::endl;
  ASSERT_EQ(oss.str(), "1.2300 4.5679, 1.230 4.568");
}

TEST(StringFunctionsTest, SplitTest) {
  std::string text = "Hello world this is a test";
  std::vector<std::string> expected = {"Hello", "world", "this",
                                       "is",    "a",     "test"};
  auto split1 = sk4slam::split(text, ' ');
  auto split2 = sk4slam::split(text, " ");

  LOGI("text: %s", text.c_str());
  LOGI("split1: %s", sk4slam::toStr(split1).c_str());
  LOGI("split2: %s", sk4slam::toStr(split2).c_str());
  ASSERT_EQ(expected, split1);
  ASSERT_EQ(expected, split2);

  std::string text2 = "Hello world \n \t this \t is \t a \t test";
  auto split3 = sk4slam::split(text2);
  LOGI("text2: %s", text2.c_str());
  LOGI("split3: %s", sk4slam::toStr(split3).c_str());
  ASSERT_EQ(expected, split3);
}

TEST(StringFunctionsTest, JoinTest) {
  std::vector<std::string> words = {"Hello", "world", "this",
                                    "is",    "a",     "test"};
  std::string text = "Hello world this is a test";
  auto join1 = sk4slam::join(words, ' ');
  auto join2 = sk4slam::join(words, " ");

  LOGI("words: %s", sk4slam::toStr(words).c_str());
  LOGI("join1: %s", join1.c_str());
  LOGI("join2: %s", join2.c_str());
  ASSERT_EQ(text, join1);
  ASSERT_EQ(text, join2);
}

TEST(StringFunctionsTest, StripTest) {
  std::string text = "  Hello world  \t\n\r\n ";
  std::string expected = "Hello world";
  auto stripped = sk4slam::strip(text);
  LOGI("text: %s", text.c_str());
  LOGI("stripped: %s", stripped.c_str());
  ASSERT_EQ(expected, stripped);
}

SK4SLAM_UNITTEST_ENTRYPOINT
