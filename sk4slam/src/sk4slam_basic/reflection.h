#pragma once

#include <cxxabi.h>
#include <memory>
#include <string>

namespace sk4slam {

/// @brief   Get the demangled type name of a type or an object.
///
/// Usage:
/// @code
///   std::cout << classname<int>() << std::endl;  // "int"
///   std::cout << classname<MyClass>() << std::endl;  // "MyClass"
///   MyClass obj;
///   std::cout << classname(obj) << std::endl;  // "MyClass"
/// @endcode
template <typename T>
const char* classname();

template <typename T>
std::string classname(const T& obj);

////////////// implementation //////////////

namespace reflection_internal {

inline std::string _demangle(const char* name) {
#if defined(__clang__) || defined(__GNUC__)
  // g++ version of demangle
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : std::string(name);
#elif defined(_MSC_VER)
  char demangled_name[1024];
  DWORD demangled_size = sizeof(demangled_name);
  if (UnDecorateSymbolName(
          name, demangled_name, demangled_size, UNDNAME_COMPLETE)) {
    return std::string(demangled_name);
  } else {
    return std::string(name);
  }
#else
  // unsupported platform
  return std::string(name);
#endif
}

}  // namespace reflection_internal

template <typename T>
const char* classname() {
  // Returns a const char pointer representing the demangled type name of the
  // static type T. For the same type T, the returned pointer will always remain
  // the same due to the use of a static std::string to store the demangled type
  // name.
  static std::string name = reflection_internal::_demangle(typeid(T).name());
  return name.c_str();
}

template <typename T>
std::string classname(const T& obj) {
  // Returns a std::string representing the demangled **runtime** type name of
  // the given object.
  //
  // Unlike the static version above, this function provides the actual runtime
  // type name of the object (e.g., for objects of polymorphic classes).
  // However, it returns a std::string instead of a persistent const char
  // pointer.
  //
  // While returning a persistent const char pointer for the runtime type would
  // be convenient, this function does not implement that behavior to avoid
  // complexities. Handling runtime polymorphism would require additional logic,
  // such as caching demangled names in a map with std::type_index as the key
  // and the demangled type name as the value. However, such an implementation
  // would introduce memory overhead and require careful management to ensure
  // thread-safety, which is beyond the scope of this function.
  return reflection_internal::_demangle(typeid(obj).name());
}

}  // namespace sk4slam
