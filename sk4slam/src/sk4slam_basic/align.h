#pragma once

namespace sk4slam {
template <size_t alignment = 8>
inline void* alignedMalloc(size_t size) {
  static_assert(
      alignment > 0 && (alignment & (alignment - 1)) == 0,
      "Alignment must be a positive power of 2");
  void* original = ::operator new(size + alignment);
  void* aligned = reinterpret_cast<void*>(
      (reinterpret_cast<size_t>(original) & ~size_t(alignment - 1)) +
      alignment);
  *(reinterpret_cast<void**>(aligned) - 1) = original;
  return aligned;
}

inline void alignedFree(void* aligned) {
  ::operator delete(*(reinterpret_cast<void**>(aligned) - 1));
}
}  // namespace sk4slam

#define DEFINE_ALIGNED_NEW_OPERATORS(cls)                    \
  void* operator new(std::size_t size) {                     \
    return sk4slam::alignedMalloc<alignof(cls)>(size);       \
  }                                                          \
  void operator delete(void* ptr) {                          \
    sk4slam::alignedFree(ptr);                               \
  }                                                          \
  void* operator new[](std::size_t size) {                   \
    return sk4slam::alignedMalloc<alignof(cls)>(size);       \
  }                                                          \
  void operator delete[](void* ptr) {                        \
    sk4slam::alignedFree(ptr);                               \
  }                                                          \
  /* in-place new and delete. there is no actual    */       \
  /* memory allocated we can safely let the default */       \
  /*implementation handle this particular case.*/            \
  static void* operator new(std::size_t size, void* ptr) {   \
    return ::operator new(size, ptr);                        \
  }                                                          \
  static void* operator new[](std::size_t size, void* ptr) { \
    return ::operator new[](size, ptr);                      \
  }                                                          \
  void operator delete(void* memory, void* ptr) {            \
    return ::operator delete(memory, ptr);                   \
  }                                                          \
  void operator delete[](void* memory, void* ptr) {          \
    return ::operator delete[](memory, ptr);                 \
  }
