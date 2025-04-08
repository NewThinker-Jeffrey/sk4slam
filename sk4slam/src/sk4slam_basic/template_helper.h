#pragma once

#include <memory>
#include <string>
#include <type_traits>

// clang-format off
// Note that `enable_if` must depend on a template parameter that is deduced.
// Example:
//   template <int a, ENABLE_IF(a!=0)>
//   class XXX {...}
//
// Also note that if your condition expression includes "...<...>...",
// you may need an extra pair of parentheses around the whole expression,
// like this:
//   template <typename Scalar, ENABLE_IF((std::is_same_v<Scalar, int>))>
//   class XXX {...}                      ^                           ^
//                                          Note the extra parentheses.
// Without the extra parentheses the compiler may report errors when
// expanding the macro.
// clang-format on
#define ENABLE_IF(cond) \
  typename std::enable_if<(cond), int>::type _dummy_template_parameter = 0

// ENABLE_IF_N(...) can be used when forward declaring a template defined with
// an ENABLE_IF(...) in the template parameter list.
// For example:
//   template <typename T, ENABLE_IF_N((std::is_same_v<T, int>))>
//   class XXX {...};  // only declaration
#define ENABLE_IF_N(cond) \
  typename std::enable_if<(cond), int>::type _dummy_template_parameter

// Get the raw type
template <typename T, bool _remove_pointer = false>
struct RawTypeT {
  using type = std::remove_reference_t<std::remove_cv_t<T>>;
};

template <typename T>
struct RawTypeT<T, true> {
  using type = std::remove_reference_t<
      std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<T>>>>;
};

template <typename T, bool _remove_pointer = false>
using RawType = typename RawTypeT<T, _remove_pointer>::type;

// Primary template: By default, T is not an STL container
template <typename T, typename = void>
struct _is_stl_container : std::false_type {};

// Specialization: Check if T has begin(), end(), and value_type
template <typename T>
struct _is_stl_container<
    T,
    std::void_t<
        typename T::value_type,               // Ensure value_type exists
        decltype(std::declval<T>().begin()),  // Ensure begin() exists
        decltype(std::declval<T>().end())>>   // Ensure end() exists
    : std::true_type {};

// Helper variable template for _is_stl_container
template <typename T>
constexpr bool is_stl_container_v =
    !std::is_base_of_v<std::string, T> && _is_stl_container<T>::value;

// Primary template: By default, T is not an STL map
template <typename T, typename = void>
struct _is_stl_map : std::false_type {};

// Specialization: Check if T has key_type and mapped_type (traits of maps)
template <typename T>
struct _is_stl_map<
    T,
    std::void_t<
        typename T::key_type,      // Ensure key_type exists
        typename T::mapped_type>>  // Ensure mapped_type exists
    : std::true_type {};

// Helper variable template for _is_stl_map
template <typename T>
constexpr bool is_stl_map_v = is_stl_container_v<T>&& _is_stl_map<T>::value;

// Primary template: Default case, not a std::pair
template <typename T>
struct is_std_pair : std::false_type {};

// Specialization: When T is a std::pair
template <typename T1, typename T2>
struct is_std_pair<std::pair<T1, T2>> : std::true_type {};

// Helper variable template
template <typename T>
constexpr bool is_std_pair_v = is_std_pair<T>::value;

// Primary template: Default case, not a std::tuple
template <typename T>
struct is_std_tuple : std::false_type {};

// Specialization: When T is a std::tuple
template <typename... Args>
struct is_std_tuple<std::tuple<Args...>> : std::true_type {};

// Helper variable template
template <typename T>
constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

// Detect shared_ptr
template <typename T>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename T>
constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

// Detect unique_ptr
template <typename T>
struct is_unique_ptr : std::false_type {};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

template <typename T>
constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;

// clang-format off
#define DEFINE_HAS_MEMBER_FUNCTION(FunctionName)                \
  template <typename T, typename... Args>                       \
  class _HasMemberFunction_##FunctionName                       \
  {                                                             \
    template <typename C>                                       \
    static constexpr auto Check(int _)                          \
        -> decltype(std::declval<C>().FunctionName(             \
                        std::declval<Args>()...),               \
                    std::true_type());                          \
    template <typename C>                                       \
    static constexpr std::false_type Check(...);                \
    template <typename C>                                       \
    static constexpr auto Call(int _)                           \
        -> decltype(std::declval<C>().FunctionName(             \
            std::declval<Args>()...));                          \
    template <typename C>                                       \
    static constexpr void Call(...);                            \
                                                                \
   public:                                                      \
    static constexpr bool value = decltype(Check<T>(0))::value; \
    using type = decltype(Call<T>(0));                          \
  };                                                            \
  template <typename T, typename... Args>                       \
  static constexpr bool HasMemberFunction_##FunctionName =      \
      _HasMemberFunction_##FunctionName<T, Args...>::value;     \
  template <typename T, typename... Args>                       \
  using TypeOfMemberFunction_##FunctionName =                   \
      typename _HasMemberFunction_##FunctionName<T, Args...>::type;


#define DEFINE_HAS_MEMBER_OPERATOR(OperatorName, op)            \
  template <typename T, typename... Args>                       \
  class _HasMemberOperator_##OperatorName                       \
  {                                                             \
    template <typename C>                                       \
    static constexpr auto Check(int _)                          \
        -> decltype(std::declval<C>().operator op(              \
                        std::declval<Args>()...),               \
                    std::true_type());                          \
    template <typename C>                                       \
    static constexpr std::false_type Check(...);                \
    template <typename C>                                       \
    static constexpr auto Call(int _)                           \
        -> decltype(std::declval<C>().operator op(              \
            std::declval<Args>()...));                          \
    template <typename C>                                       \
    static constexpr void Call(...);                            \
                                                                \
   public:                                                      \
    static constexpr bool value = decltype(Check<T>(0))::value; \
    using type = decltype(Call<T>(0));                          \
  };                                                            \
  template <typename T, typename... Args>                       \
  static constexpr bool HasMemberOperator_##OperatorName =      \
      _HasMemberOperator_##OperatorName<T, Args...>::value;     \
  template <typename T, typename... Args>                       \
  using TypeOfMemberOperator_##OperatorName =                   \
      typename _HasMemberOperator_##OperatorName<T, Args...>::type;


#define DEFINE_HAS_MEMBER_VARIABLE(VariableName)                       \
  template <typename T>                                                \
  class _HasMemberVariable_##VariableName                              \
  {                                                                    \
    template <typename C>                                              \
    static constexpr auto Check(int _)                                 \
        -> decltype(std::declval<C>().VariableName, std::true_type()); \
    template <typename>                                                \
    static constexpr std::false_type Check(...);                       \
    template <typename C>                                              \
    static constexpr auto Call(int _)                                  \
        -> decltype(std::declval<C>().VariableName);                   \
    template <typename>                                                \
    static constexpr void Call(...);                                   \
                                                                       \
   public:                                                             \
    static constexpr bool value = decltype(Check<T>(0))::value;        \
    using type = decltype(Call<T>(0));                                 \
  };                                                                   \
  template <typename T>                                                \
  static constexpr bool HasMemberVariable_##VariableName =             \
      _HasMemberVariable_##VariableName<T>::value;                     \
  template <typename T>                                                \
  using TypeOfMemberVariable_##VariableName =                          \
      typename _HasMemberVariable_##VariableName<T>::type;
// clang-format on
