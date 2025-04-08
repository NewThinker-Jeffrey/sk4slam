#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <string>

// clang-format off
/// @brief Declare a dynamic factory for a base class.
/// @param BaseClass The name of the base class.
/// @param ... The types of the constructor arguments for the derived classes.
///
/// This macro declares a dynamic factory for the specified base class.
/// The factory provides static methods to register and create derived class
/// instances dynamically based on a string type identifier.
///
/// Derived classes must provide a "factory constructor" that takes the same
/// arguments as the `FactoryFunction`, as defined by the "..." macro arguments.
///
/// Usage:
/// ```cpp
/// class Base {
///   DECLARE_DYNAMIC_FACTORY(Base, int, std::string)
/// };
///
/// // Example: Define a Derived class
/// class Derived : public Base {
/// public:
///   Derived(int a, const std::string& b) {}
/// };
///
/// // Example: Register the derived class with the factory
/// Base::registerFactory("Derived", [](int a, const std::string& b) {
///   return new Derived(a, b);
/// });
///
/// // Example: Create an instance dynamically
/// Base* instance = Base::create("Derived", 42, "example");
/// if (instance) {
///   // Use the instance
///   delete instance;
/// }
/// ```
#define DECLARE_DYNAMIC_FACTORY(BaseClass, ...)                               \
 public:                                                                      \
  using FactoryFunction = std::function<BaseClass*(__VA_ARGS__)>;             \
                                                                              \
  template <typename... Args>                                                 \
  static BaseClass* create(const std::string& type, Args&&... args) {         \
    FactoryFunction func;                                                     \
    {                                                                         \
      std::lock_guard<std::mutex> lock(factory_registry_mutex_);              \
      auto it = factory_registry_.find(type);                                 \
      if (it == factory_registry_.end()) {                                    \
        return nullptr;                                                       \
      }                                                                       \
      func = it->second;                                                      \
    }                                                                         \
    return func(std::forward<Args>(args)...);                                 \
  }                                                                           \
                                                                              \
  static void registerFactory(                                                \
      const std::string& type, FactoryFunction func) {                        \
    std::lock_guard<std::mutex> lock(factory_registry_mutex_);                \
    if (factory_registry_.find(type) != factory_registry_.end()) {            \
      throw std::runtime_error("Class already registered: " + type);          \
    }                                                                         \
    factory_registry_[type] = func;                                           \
  }                                                                           \
                                                                              \
 private:                                                                     \
  static inline std::map<std::string, FactoryFunction> factory_registry_;     \
  static inline std::mutex factory_registry_mutex_;                           \
                                                                              \
  template <typename T>                                                       \
  using remove_cvref_t =                                                      \
      typename std::remove_cv<typename std::remove_reference<T>::type>::type; \
                                                                              \
  template <typename T, typename = void>                                      \
  struct HasStaticTypeMethod : std::false_type {};                            \
                                                                              \
  template <typename T>                                                       \
  struct HasStaticTypeMethod<                                                 \
      T,                                                                      \
      std::void_t<                                                            \
          decltype(std::declval<remove_cvref_t<decltype(T::type())>>()        \
                    == std::declval<std::string>())>>                         \
      : std::true_type {};                                                    \
                                                                              \
 protected:                                                                   \
  template <typename DerivedClass>                                            \
  static std::string getTypeStringFor##BaseClass(                             \
      const std::string& class_name) {                                        \
    if constexpr (HasStaticTypeMethod<DerivedClass>::value) {                 \
      return DerivedClass::type();                                            \
    } else {                                                                  \
      return class_name;                                                      \
    }                                                                         \
  }                                                                           \
                                                                              \
  template <typename DerivedClass>                                            \
  struct BaseClass##AutoRegistry {                                            \
    BaseClass##AutoRegistry(const std::string& type) {                        \
      BaseClass::registerFactory(type, [](auto... args) {                     \
        return new DerivedClass(std::forward<decltype(args)>(args)...);       \
      });                                                                     \
    }                                                                         \
  };
// clang-format on

/// @brief Automatically register a derived class to a base class factory.
/// @param BaseClass The name of the base class.
/// @param DerivedClass The name of the derived class.
///
/// This macro registers a derived class with a base class factory, allowing
/// instances of the derived class to be created dynamically using the factory.
/// The type identifier used for registration is determined as follows:
/// - If the derived class defines a static `type()` function, the return value
///   of `type()` is used as the type identifier.
/// - If the derived class does not define a static `type()` function, the class
///   name (`#DerivedClass`) is used as the default type identifier.
///
/// Usage:
/// ```cpp
/// // Example 1: Derived class with a static type() function
/// class DerivedWithType : public Base {
///   AUTO_REGISTER_FACTORY(Base, DerivedWithType)
/// public:
///   static std::string type() { return "CustomDerived"; }
/// };
///
/// // Example 2: Derived class without a static type() function
/// class DerivedWithoutType : public Base {
///   AUTO_REGISTER_FACTORY(Base, DerivedWithoutType)
/// public:
///   ...
/// };
///
/// // Creating instances dynamically
/// Base* instance1 = Base::create("CustomDerived");
/// Base* instance2 = Base::create("DerivedWithoutType");
///
/// if (instance1) {
///   // Use instance1
///   delete instance1;
/// }
///
/// if (instance2) {
///   // Use instance2
///   delete instance2;
/// }
/// ```
#define AUTO_REGISTER_FACTORY(BaseClass, DerivedClass)      \
  static inline const BaseClass##AutoRegistry<DerivedClass> \
      k##DerivedClass##AutoRegistry_{                       \
          getTypeStringFor##BaseClass<DerivedClass>(#DerivedClass)};
