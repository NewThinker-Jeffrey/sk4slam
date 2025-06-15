#pragma once

#include <Eigen/Core>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

// The CRTP base class for all configurable classes.
template <typename Derived>
struct Configurable;

/// @brief  Checks if a type is a configurable class.
template <typename T>
inline constexpr bool IsConfigurable =
    std::is_base_of_v<Configurable<RawType<T>>, RawType<T>>;

// clang-format off
/// @brief   The CRTP base class for all configurable classes.
///          Derived class should implement the following two methods:
///
///             1. bool _has_impl(const std::string& config_key) const {},
///
///               _has_impl() should return true if the config key exists,
///               otherwise false.
///
///             2. template <typename T>
///                T _get_impl(
///                    const std::string& config_key,
///                    bool required,
///                    const T* default_value) const {}
///
///               _get_impl() should return the value of the config key. If
///               the config key does not exist, the behavior depends on the
///               @p required parameter:
///               - If @p required is false, the @p default_value will be
///                 returned.
///               - Otherwise, an exception will be thrown (when T is not
///                 @c Derived ) or an empty sub-config will be returned
///                 (when T is @c Derived ).
///
// clang-format on
template <typename Derived>
struct Configurable {
  template <typename T, typename... Args>
  class has_constructor {
   private:
    template <typename U, typename = decltype(U(std::declval<Args>()...))>
    static std::true_type test(int);

    template <typename>
    static std::false_type test(...);

   public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };

  template <typename T, typename... Args>
  static inline constexpr bool has_constructor_v =
      has_constructor<T, Args...>::value;

 public:
  /// @param config_key  The key of the config item. Note @p config_key can
  ///                    contain '.' to specify nested config items and '['' and
  ///                    ']' to specify list items. For example,
  ///                    "foo.bar[0].baz" specifies the "baz" item of the first
  ///                    item of the "bar" list in the "foo" config item.
  bool has(const std::string& config_key) const {
    return derived()->_has_impl(config_key);
  }

  /// @brief  Sub classes should implement the get() method templates.
  /// @tparam T   Type of the value to be returned. It can be any of the
  ///             following types: fundamental types (int/float/double, etc.),
  ///             sub config (Derived), Eigen::Matrix/Vector, and STL
  ///             containers, e.g. std::vector<*>,
  ///             std::unordered_map<std::string, *>,
  ///             std::map<std::string, *>, etc.
  /// If the @p config_key does not exist, the default_value will be returned.
  template <typename _T, Verbose verbose = Verbose::SILENT>
  RawType<_T> get(
      const std::string& config_key, const RawType<_T>& default_value) const {
    using T = RawType<_T>;
    static constexpr bool kConstructFromSubconfig =
        !IsConfigurable<T> && has_constructor_v<T, const Derived&>;
    // TODO(jeffrey): Check why kConstructFromSubconfig doesn't work
    if constexpr (/*kConstructFromSubconfig*/ false) {
      if (!has(config_key)) {
        return default_value;
      } else {
        return T(get(config_key));
      }
    } else {
      T v = derived()->template _get_impl<T>(config_key, false, &default_value);
      if constexpr (verbose != Verbose::SILENT) {
        this->template log<T, verbose>(config_key, v);
      }
      return v;
    }
  }

  /// If the config_key does not exist, an empty sub-config will be
  //  returned (when T is Derived) or an exception will be thrown (when T is
  /// other types)
  template <typename _T = Derived, Verbose verbose = Verbose::SILENT>
  RawType<_T> get(const std::string& config_key) const {
    using T = RawType<_T>;
    static constexpr bool kConstructFromSubconfig =
        !IsConfigurable<T> && has_constructor_v<T, const Derived&>;
    // TODO(jeffrey): Check why kConstructFromSubconfig doesn't work
    if constexpr (/*kConstructFromSubconfig*/ false) {
      if (!has(config_key)) {
        throw std::runtime_error(formatStr(
            "%s::get(%s) failed: key not found!", classname<Derived>(),
            config_key.c_str()));
      } else {
        return T(get(config_key));
      }
    } else {
      T v = derived()->template _get_impl<T>(config_key, true, nullptr);
      if constexpr (verbose != Verbose::SILENT) {
        this->template log<T, verbose>(config_key, v);
      }
      return v;
    }
  }

  // getX() with different verbose levels
  template <typename T>
  auto getA(const std::string& config_key, const T& default_value) const {
    return this->template get<T, Verbose::ALL>(config_key, default_value);
  }
  template <typename T = Derived>
  auto getA(const std::string& config_key) const {
    return this->template get<T, Verbose::ALL>(config_key);
  }

  template <typename T>
  auto getD(const std::string& config_key, const T& default_value) const {
    return this->template get<T, Verbose::DEBUG>(config_key, default_value);
  }
  template <typename T = Derived>
  auto getD(const std::string& config_key) const {
    return this->template get<T, Verbose::DEBUG>(config_key);
  }

  template <typename T>
  auto getI(const std::string& config_key, const T& default_value) const {
    return this->template get<T, Verbose::INFO>(config_key, default_value);
  }
  template <typename T = Derived>
  auto getI(const std::string& config_key) const {
    return this->template get<T, Verbose::INFO>(config_key);
  }

  template <typename T>
  auto getW(const std::string& config_key, const T& default_value) const {
    return this->template get<T, Verbose::WARNING>(config_key, default_value);
  }
  template <typename T = Derived>
  auto getW(const std::string& config_key) const {
    return this->template get<T, Verbose::WARNING>(config_key);
  }

  template <typename T>
  auto getE(const std::string& config_key, const T& default_value) const {
    return this->template get<T, Verbose::ERROR>(config_key, default_value);
  }
  template <typename T = Derived>
  auto getE(const std::string& config_key) const {
    return this->template get<T, Verbose::ERROR>(config_key);
  }

 private:
  const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }
  template <typename T, Verbose verbose>
  void log(const std::string& config_key, const T& v) const {
    static_assert(verbose != Verbose::SILENT);
    bool found = has(config_key);
    std::string log_str =
        found ? "GET_CONFIG-found    : " : "GET_CONFIG-not_found: ";
    log_str += config_key + " = " + toStr(v);
    if constexpr (verbose == Verbose::ALL) {
      LOGA("%s", log_str.c_str());
    } else if constexpr (verbose == Verbose::DEBUG) {
      LOGD("%s", log_str.c_str());
    } else if constexpr (verbose == Verbose::INFO) {
      LOGI("%s", log_str.c_str());
    } else if constexpr (verbose == Verbose::WARNING) {
      LOGW("%s", log_str.c_str());
    } else if constexpr (verbose == Verbose::ERROR) {
      LOGE("%s", log_str.c_str());
    } else {
      throw std::runtime_error("Invalid verbose level");
    }
  }
};

}  // namespace sk4slam

// clang-format off

// Macros used to load the value of a variable from the config. If the variable
// doesn't exist in the config, exception is thrown.

#define CONFIG_LOAD  (config, variable)   \
    (variable) = (config).get <decltype(variable)>(#variable)
#define CONFIG_LOAD_A(config, variable)   \
    (variable) = (config).getA<decltype(variable)>(#variable)
#define CONFIG_LOAD_D(config, variable)   \
    (variable) = (config).getD<decltype(variable)>(#variable)
#define CONFIG_LOAD_I(config, variable)   \
    (variable) = (config).getI<decltype(variable)>(#variable)
#define CONFIG_LOAD_W(config, variable)   \
    (variable) = (config).getW<decltype(variable)>(#variable)
#define CONFIG_LOAD_E(config, variable)   \
    (variable) = (config).getE<decltype(variable)>(#variable)


// Macros used to update the value of a variable if it exists in the config.
#define CONFIG_UPDT(config, variable)   \
    (variable) = (config).get(#variable, variable)
#define CONFIG_UPDT_A(config, variable)   \
    (variable) = (config).getA(#variable, variable)
#define CONFIG_UPDT_D(config, variable)   \
    (variable) = (config).getD(#variable, variable)
#define CONFIG_UPDT_I(config, variable)   \
    (variable) = (config).getI(#variable, variable)
#define CONFIG_UPDT_W(config, variable)   \
    (variable) = (config).getW(#variable, variable)
#define CONFIG_UPDT_E(config, variable)   \
    (variable) = (config).getE(#variable, variable)

// clang-format on

// Macros below are used to read / update a configuration variable only once.
// They're useful when initializing a local static variable.

#define CONFIG_UPDT_ONCE(config, variable) \
  {                                        \
    static bool _tmp_already_read = false; \
    if (!_tmp_already_read) {              \
      CONFIG_UPDT(config, variable);       \
      _tmp_already_read = true;            \
    }                                      \
  }

#define CONFIG_UPDT_ONCE_A(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_UPDT_A(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_UPDT_ONCE_D(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_UPDT_D(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_UPDT_ONCE_I(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_UPDT_I(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_UPDT_ONCE_W(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_UPDT_W(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_UPDT_ONCE_E(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_UPDT_E(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_LOAD_ONCE(config, variable) \
  {                                        \
    static bool _tmp_already_read = false; \
    if (!_tmp_already_read) {              \
      CONFIG_LOAD(config, variable);       \
      _tmp_already_read = true;            \
    }                                      \
  }

#define CONFIG_LOAD_ONCE_A(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_LOAD_A(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_LOAD_ONCE_D(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_LOAD_D(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_LOAD_ONCE_I(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_LOAD_I(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_LOAD_ONCE_W(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_LOAD_W(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

#define CONFIG_LOAD_ONCE_E(config, variable) \
  {                                          \
    static bool _tmp_already_read = false;   \
    if (!_tmp_already_read) {                \
      CONFIG_LOAD_E(config, variable);       \
      _tmp_already_read = true;              \
    }                                        \
  }

/// @brief  A macro to initialize the members of a struct from a configuration
/// object.
///
/// This macro helps construct an object of type `StructName` and initializes
/// its members using the provided configuration object `config`. You can use
/// the macro `CONFIG_OPTIONAL_MEM` to update a member’s value if it is
/// present in the configuration object. Alternatively, you can use
/// `CONFIG_REQUIRED_MEM` to enforce loading a value from the configuration
/// object. If the value is not found, `CONFIG_REQUIRED_MEM` will throw an
/// error.
///
/// @tparam Config The type of the configuration object, which must be derived
/// from `Configurable`.
/// @param config The configuration object that contains the values to
/// initialize the struct members.
/// @param StructName The type of the struct which contains the members to be
/// configured.
///
/// Example usage:
/// @code
/// struct Options {
///   std::string member1{"default_value1"};
///   std::string member2;
///   Options() {}
///
///   CONFIG_MEMBERS(Options) {
///     // `member1` has a default value, so use `CONFIG_OPTIONAL_MEM`
///     // which makes loading optional.
///     CONFIG_OPTIONAL_MEM(member1);
///
///     // `member2` does not have a default value, so use `CONFIG_REQUIRED_MEM`
///     // to enforce loading, and it will throw an error if not found.
///     CONFIG_REQUIRED_MEM(config, member2);
///   }
/// };
/// @endcode
///
/// In this example, the members of the `Options` struct will be initialized
/// with the corresponding values from the `config` object. If `member2` is
/// missing in the configuration, the program will throw an error.
#define CONFIG_MEMBERS(StructName)                                         \
  template <typename Config, ENABLE_IF((sk4slam::IsConfigurable<Config>))> \
  explicit StructName(const Config& config)

#define CONFIG_BASE_MEMBERS(BaseStructName) BaseStructName(config)

#define CONFIG_OPTIONAL_MEM(member) CONFIG_UPDT_I(config, member)

#define CONFIG_REQUIRED_MEM(member) CONFIG_LOAD_I(config, member)

#define CONFIG_OPTIONAL_SUB_MEM(member) \
  member = decltype(member)(config.get(#member))

#define CONFIG_REQUIRED_SUB_MEM(member)                              \
  {                                                                  \
    if (!config.has(#member)) {                                      \
      throw std::runtime_error("Missing required member: " #member); \
    } else {                                                         \
      member = decltype(member)(config.get(#member));                \
    }                                                                \
  }
