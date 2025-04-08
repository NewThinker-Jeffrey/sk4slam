#pragma once

// The purpose of this header file is to wrap some stl types (such as
// std::unique_lock) to make use of Clang Thread Safety Analysis (CTSA).

// To enable CTSA, we need to add the flag '-Wthread-safety' or
// '-Werror=thread-safety' when compiling:
//    ```clang -c -Wthread-safety example.cpp```
//    ```clang -c -Werror=thread-safety example.cpp```
//                  # ↑ this reports errors instead of warnings
// When working with cmake we should set cmake-args:
//      -DCMAKE_CXX_COMPILER=clang++   # use clang++ as the compiler
//      -DCMAKE_CXX_FLAGS="-Werror=thread-safety"
// See: https://clang.llvm.org/docs/ThreadSafetyAnalysis.html

// Usually CTSA needs a suitably annotated mutex.h that declares which methods
// perform locking, unlocking, and so on (by adding some particular
// __attribute__ to the classes and their member methods).
//
// Though developers have added (in libc++: llvm-project/libcxx) clang thread
// safety annotations to std::mutex and std::lock_guard (see
// https://reviews.llvm.org/D14731, so code using these types can use these
// types directly instead of having to wrap the types to provide annotations),
// another frequently used type std::unique_lock has not been adapted yet.
// So we still need to do some wrapping for the stl types.

// The following code are modified from the template mutex.h:
//   - https://clang.llvm.org/docs/ThreadSafetyAnalysis.html#mutexheader
// Note the Mutex class in the template mutex.h is designed to be used as
// read-write-lock, however the standard implementation of read-write-lock, i.e.
// std::shared_mutex, is not present until C++17.

#if __cplusplus >= 201703L
#include <shared_mutex>
#define USE_READ_WRITE_BASE_MUTEX 1
#else
#include <mutex>
#define USE_READ_WRITE_BASE_MUTEX 0
#endif

#define USE_CTSA 1
// #define USE_CTSA 0

#include <condition_variable>

///////////////////////////////////////////////////////////////////////////////////////////

// Enable thread safety attributes only with clang.
// The attributes can be safely erased when compiling with other compilers.

#if USE_CTSA
#if defined(__clang__) && (!defined(SWIG))
#define THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define THREAD_ANNOTATION_ATTRIBUTE__(x)  // no-op
#endif  // defined(__clang__) && (!defined(SWIG))
#else
#define THREAD_ANNOTATION_ATTRIBUTE__(x)  // no-op
#endif                                    // USE_CTSA

#define CAPABILITY(x) THREAD_ANNOTATION_ATTRIBUTE__(capability(x))

#define SCOPED_CAPABILITY THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

#define GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))

#define PT_GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))

#define ACQUIRED_BEFORE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define ACQUIRED_AFTER(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define REQUIRES(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(requires_capability(__VA_ARGS__))

#define REQUIRES_SHARED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))

#define ACQUIRE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define ACQUIRE_SHARED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define RELEASE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

#define RELEASE_SHARED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(release_shared_capability(__VA_ARGS__))

#define RELEASE_GENERIC(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(release_generic_capability(__VA_ARGS__))

#define TRY_ACQUIRE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))

#define TRY_ACQUIRE_SHARED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))

#define EXCLUDES(...) THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define ASSERT_CAPABILITY(x) THREAD_ANNOTATION_ATTRIBUTE__(assert_capability(x))

#define ASSERT_SHARED_CAPABILITY(x) \
  THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_capability(x))

#define RETURN_CAPABILITY(x) THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#define NO_THREAD_SAFETY_ANALYSIS \
  THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

#if !USE_CTSA

namespace sk4slam {

#if !USE_READ_WRITE_BASE_MUTEX
using Mutex = std::mutex;
using UniqueLock = std::unique_lock<Mutex>;
using SharedLock = UniqueLock;
using ConditionVariable = std::condition_variable;
#else
using Mutex = std::shared_mutex;  // C++ 17
using UniqueLock = std::unique_lock<Mutex>;
using SharedLock = std::shared_lock<Mutex>;  // C++ 14
using ConditionVariable = std::condition_variable_any;
#endif

}  // namespace sk4slam

#else

namespace sk4slam {

#if USE_READ_WRITE_BASE_MUTEX

// Note:
// std::shared_lock was introduced in C++14 as an RAII class for
// std::shared_timed_mutex, while std::shared_mutex is added later in C++17.
// See:
//   https://stackoverflow.com/questions/40207171/why-shared-timed-mutex-is-defined-in-c14-but-shared-mutex-in-c17

using BaseMutex = std::shared_mutex;  // C++17

using BaseUniqueLock = std::unique_lock<BaseMutex>;

using BaseSharedLock = std::shared_lock<BaseMutex>;  // C++14

// Class std::condition_variable provides a condition variable that
// can only wait on an object of type unique_lock<mutex>, allowing
// maximum effciency on some platforms.
//
// The std::condition_variable_any class is a generalization of
// std::condition_variable. Whereas std::condition_variable works only
// on std::unique_lock<std::mutex>, std::condition_variable_any can
// operate on any lock that meets the BasicLockable requirements.
//
// See:
//   https://en.cppreference.com/w/cpp/thread/condition_variable_any
using BaseConditionVariable = std::condition_variable_any;

#else

using BaseMutex = std::mutex;

using BaseUniqueLock = std::unique_lock<BaseMutex>;

using BaseSharedLock = BaseUniqueLock;

using BaseConditionVariable = std::condition_variable;

#endif

// Defines an annotated interface for mutexes.
// These methods can be implemented to use any internal mutex implementation.
class CAPABILITY("mutex") Mutex : public BaseMutex {
 public:
  Mutex() : BaseMutex() {}

  // Acquire/lock this mutex exclusively.  Only one thread can have exclusive
  // access at any one time.  Write operations to guarded data require an
  // exclusive lock.
  void lock() ACQUIRE() {
    BaseMutex::lock();
  }

  // Release/unlock an exclusive mutex.
  void unlock() RELEASE() {
    BaseMutex::unlock();
  }

  // Try to acquire the mutex.  Returns true on success, and false on failure.
  bool try_lock() TRY_ACQUIRE(true) {
    return BaseMutex::try_lock();
  }

  // Acquire/lock this mutex for read operations, which require only a shared
  // lock.  This assumes a multiple-reader, single writer semantics.  Multiple
  // threads may acquire the mutex simultaneously as readers, but a writer
  // must wait for all of them to release the mutex before it can acquire it
  // exclusively.
  void lock_shared() ACQUIRE_SHARED() {
#if USE_READ_WRITE_BASE_MUTEX
    BaseMutex::lock_shared();
#else
    BaseMutex::lock();
#endif
  }

  // Release/unlock a shared mutex.
  void unlock_shared() RELEASE_SHARED() {
#if USE_READ_WRITE_BASE_MUTEX
    BaseMutex::unlock_shared();
#else
    BaseMutex::unlock();
#endif
  }

  // Try to acquire the mutex for read operations.
  bool try_lock_shared() TRY_ACQUIRE_SHARED(true) {
#if USE_READ_WRITE_BASE_MUTEX
    return BaseMutex::try_lock_shared();
#else
    return BaseMutex::try_lock();
#endif
  }

  BaseMutex& base() {
    return *this;
  }
};

// UniqueLock is an RAII class that acquires a mutex in its constructor, and
// releases it in its destructor.
class SCOPED_CAPABILITY UniqueLock : public BaseUniqueLock {
 public:
  explicit UniqueLock(Mutex& mu) ACQUIRE(mu) : BaseUniqueLock(mu) {}
  UniqueLock(Mutex& mu, std::defer_lock_t t) ACQUIRE(mu)
      : BaseUniqueLock(mu, t) {}
  UniqueLock(Mutex& mu, std::try_to_lock_t t) ACQUIRE(mu)
      : BaseUniqueLock(mu, t) {}
  UniqueLock(Mutex& mu, std::adopt_lock_t t) ACQUIRE(mu)
      : BaseUniqueLock(mu, t) {}
  // ~UniqueLock() RELEASE() {}
  ~UniqueLock() RELEASE_GENERIC() {}

  // Acquire all associated mutexes exclusively.
  void lock() ACQUIRE() {
    BaseUniqueLock::lock();
  }

  // Release all associated mutexes.
  void unlock() RELEASE() {
    BaseUniqueLock::unlock();
  }

  // Try to acquire all associated mutexes exclusively.
  bool try_lock() TRY_ACQUIRE(true) {
    return BaseUniqueLock::try_lock();
  }

  BaseUniqueLock& base() {
    return *this;
  }
};

class SCOPED_CAPABILITY SharedLock : public BaseSharedLock {
 public:
  explicit SharedLock(Mutex& mu) ACQUIRE_SHARED(mu) : BaseSharedLock(mu) {}
  SharedLock(Mutex& mu, std::defer_lock_t t) ACQUIRE_SHARED(mu)
      : BaseSharedLock(mu, t) {}
  SharedLock(Mutex& mu, std::try_to_lock_t t) ACQUIRE_SHARED(mu)
      : BaseSharedLock(mu, t) {}
  SharedLock(Mutex& mu, std::adopt_lock_t t) ACQUIRE_SHARED(mu)
      : BaseSharedLock(mu, t) {}

  // Acquire all associated mutexes exclusively.
  void lock() ACQUIRE_SHARED() {
    BaseSharedLock::lock();
  }

  // Release all associated mutexes.
  // Why is this not RELEASE_SHARED()? See
  // - https://github.com/llvm/llvm-project/issues/32851
  // void unlock() RELEASE_SHARED() {
  void unlock() RELEASE_GENERIC() {
    BaseSharedLock::unlock();
  }

  // ~SharedLock() RELEASE_SHARED() {}
  ~SharedLock() RELEASE_GENERIC() {}

  // Try to acquire all associated mutexes exclusively.
  bool try_lock() TRY_ACQUIRE_SHARED(true) {
    return BaseSharedLock::try_lock();
  }

  BaseSharedLock& base() {
    return *this;
  }
};

class ConditionVariable : public BaseConditionVariable {
 public:
  template <class Lock>
  void wait(Lock& lock) {
    BaseConditionVariable::wait(lock.base());
  }

  template <class Lock, class Predicate>
  void wait(Lock& lock, Predicate stop_waiting) {
    BaseConditionVariable::wait(lock.base(), stop_waiting);
  }

  template <class Lock, class Rep, class Period>
  std::cv_status wait_for(
      Lock& lock, const std::chrono::duration<Rep, Period>& rel_time) {
    return BaseConditionVariable::wait_for(lock.base(), rel_time);
  }

  template <class Lock, class Rep, class Period, class Predicate>
  bool wait_for(
      Lock& lock, const std::chrono::duration<Rep, Period>& rel_time,
      Predicate stop_waiting) {
    return BaseConditionVariable::wait_for(lock.base(), rel_time, stop_waiting);
  }

  template <class Lock, class Clock, class Duration>
  std::cv_status wait_until(
      Lock& lock,
      const std::chrono::time_point<Clock, Duration>& timeout_time) {
    return BaseConditionVariable::wait_until(lock.base(), timeout_time);
  }

  template <class Lock, class Clock, class Duration, class Predicate>
  bool wait_until(
      Lock& lock, const std::chrono::time_point<Clock, Duration>& timeout_time,
      Predicate stop_waiting) {
    return BaseConditionVariable::wait_until(
        lock.base(), timeout_time, stop_waiting);
  }
};

}  // namespace sk4slam

#endif  // USE_CTSA

#ifdef USE_LOCK_STYLE_THREAD_SAFETY_ATTRIBUTES
// The original version of thread safety analysis the following attribute
// definitions.  These use a lock-based terminology.  They are still in use
// by existing thread safety code, and will continue to be supported.

// Deprecated.
#define PT_GUARDED_VAR THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_var)

// Deprecated.
#define GUARDED_VAR THREAD_ANNOTATION_ATTRIBUTE__(guarded_var)

// Replaced by REQUIRES
#define EXCLUSIVE_LOCKS_REQUIRED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(exclusive_locks_required(__VA_ARGS__))

// Replaced by REQUIRES_SHARED
#define SHARED_LOCKS_REQUIRED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(shared_locks_required(__VA_ARGS__))

// Replaced by CAPABILITY
#define LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(lockable)

// Replaced by SCOPED_CAPABILITY
#define SCOPED_LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

// Replaced by ACQUIRE
#define EXCLUSIVE_LOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(exclusive_lock_function(__VA_ARGS__))

// Replaced by ACQUIRE_SHARED
#define SHARED_LOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(shared_lock_function(__VA_ARGS__))

// Replaced by RELEASE and RELEASE_SHARED
#define UNLOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(unlock_function(__VA_ARGS__))

// Replaced by TRY_ACQUIRE
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(exclusive_trylock_function(__VA_ARGS__))

// Replaced by TRY_ACQUIRE_SHARED
#define SHARED_TRYLOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(shared_trylock_function(__VA_ARGS__))

// Replaced by ASSERT_CAPABILITY
#define ASSERT_EXCLUSIVE_LOCK(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(assert_exclusive_lock(__VA_ARGS__))

// Replaced by ASSERT_SHARED_CAPABILITY
#define ASSERT_SHARED_LOCK(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_lock(__VA_ARGS__))

// Replaced by EXCLUDE_CAPABILITY.
#define LOCKS_EXCLUDED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

// Replaced by RETURN_CAPABILITY
#define LOCK_RETURNED(x) THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#endif  // USE_LOCK_STYLE_THREAD_SAFETY_ATTRIBUTES
