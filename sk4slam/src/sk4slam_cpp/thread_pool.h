#pragma once

#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "sk4slam_basic/string_helper.h"
#include "sk4slam_cpp/mutex.h"

namespace sk4slam {

class Task;
class TaskManager;
class ThreadPool;
class ThreadPoolGroup;

// using TaskID = size_t;
// static const TaskID INVALID_TASK = reinterpret_cast<size_t>(nullptr);
// using TaskID = const Task*;
// static constexpr TaskID INVALID_TASK = nullptr;
// using ThreadPoolID = const ThreadPool*;
// static constexpr ThreadPoolID INVALID_THREAD_POOL = nullptr;

struct TaskID {
  explicit TaskID(size_t set_id = 0) : id(set_id) {}
  bool operator==(const TaskID& other) const {
    return id == other.id;
  }
  bool operator!=(const TaskID& other) const {
    return id != other.id;
  }
  bool operator<(const TaskID& other) const {
    return id < other.id;
  }
  operator bool() {
    return id != 0;
  }
  size_t id;
};
static const TaskID INVALID_TASK(0);

struct ThreadPoolID {
  explicit ThreadPoolID(size_t set_id = 0) : id(set_id) {}
  bool operator==(const ThreadPoolID& other) const {
    return id == other.id;
  }
  bool operator!=(const ThreadPoolID& other) const {
    return id != other.id;
  }
  bool operator<(const ThreadPoolID& other) const {
    return id < other.id;
  }
  operator bool() {
    return id != 0;
  }
  size_t id;
};
static const ThreadPoolID INVALID_THREAD_POOL(0);

class ThreadPool {
 public:
  // Create an anonymous thread pool.
  static std::shared_ptr<ThreadPool> create(int num_threads = 1);

  // Create/Remove a named thread pool.
  // Keep in mind that for every named thread pool there wiil be a hidden
  // shared_ptr referencing it after creation, hence it won't be destructed
  // before removeNamed() is called.
  static std::shared_ptr<ThreadPool> createNamed(
      const std::string& thread_name, int num_threads = 1);
  static void removeNamed(
      const std::string& thread_name, bool wait_all_works_done = true);

  // Get a shared_ptr to a named thread pool.
  // If the pool is not created yet, it will be auto created with num_threads=1.
  // If you need a named thread pool containing more than one threads, you have
  // to call createNamed() explicitly beforehand.
  static std::shared_ptr<ThreadPool> getNamed(const std::string& thread_name);

 public:
  // Schedule a task
  TaskID schedule(
      std::function<void()> work_item,
      const std::set<TaskID>& dependencies = std::set<TaskID>())
      EXCLUDES(mutex_);

  // Return false if the task is already done (or unregistered) before
  // calling wait();
  // Otherwise return true.
  bool wait(TaskID task);

  // Return false if any of the tasks is already done (or unregistered) before
  // calling waitTasks();
  template <typename TaskIDIterator>
  bool waitTasks(TaskIDIterator begin, TaskIDIterator end) {
    bool ret = true;
    for (auto it = begin; it != end; ++it) {
      // ret = ret && wait(*it);  // This will skip wait() if ret is already
      // false!
      ret = wait(*it) && ret;  // This will wait() on every task.
    }
    return ret;
  }

  void waitUntilAllTasksDone();

  // After calling freeze(), new tasks can't be scheduled anymore until
  // unfreeze() is called;
  void freeze() EXCLUDES(mutex_);
  void unfreeze() EXCLUDES(mutex_);
  bool isFrozen() const EXCLUDES(mutex_);

  // num_threads
  inline int numThreads() const {
    return pool_.size();
  }

  // id
  inline ThreadPoolID id() const {
    return id_;
  }
  ~ThreadPool();

  // A task group is a set of tasks that need to be scheduled and waited
  // together.
  struct TaskGroup {
   public:
    explicit TaskGroup(std::shared_ptr<ThreadPool> thread_pool)
        : thread_pool_(thread_pool) {}

    // Schedule a task
    TaskID schedule(
        std::function<void()> work_item,
        const std::set<TaskID>& dependencies = std::set<TaskID>()) {
      TaskID ret = thread_pool_->schedule(work_item, dependencies);
      if (ret != INVALID_TASK) {
        tasks_.push_back(ret);
      }
      return ret;
    }

    // Wait until all tasks in the group are done.
    bool wait() {
      std::vector<TaskID> tasks;
      tasks.swap(tasks_);
      if (tasks.empty()) {
        return true;
      }
      return thread_pool_->waitTasks(tasks.begin(), tasks.end());
    }

    ~TaskGroup() {
      wait();
    }

   private:
    std::vector<TaskID> tasks_;
    std::shared_ptr<ThreadPool> thread_pool_;
  };

  static TaskGroup createTaskGroupForNamed(const std::string& thread_name) {
    return TaskGroup(getNamed(thread_name));
  }

 private:
  friend class ThreadPoolGroup;
  explicit ThreadPool(
      int num_threads, const std::string& thread_name,
      std::shared_ptr<TaskManager> task_manager);

  // Stop the threads
  void stop(bool wait_all_tasks_done = true) EXCLUDES(mutex_);

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  void initThreads(int num_threads);
  void addPendingTask(TaskID task) EXCLUDES(mutex_);
  void addPendingTaskNoLock(TaskID task) REQUIRES(mutex_);
  TaskID waitForNextTask(bool* exit) EXCLUDES(mutex_);
  void doWork() EXCLUDES(mutex_);

 private:
  std::shared_ptr<TaskManager> task_manager_;

  mutable Mutex mutex_;
  ConditionVariable cond_ GUARDED_BY(mutex_);
  std::deque<TaskID> pending_tasks_ GUARDED_BY(mutex_);
  bool frozen_ GUARDED_BY(mutex_) = false;
  bool stop_request_ GUARDED_BY(mutex_) = false;
  bool stopped_ GUARDED_BY(mutex_) = false;
  bool wait_all_tasks_done_before_stop_ GUARDED_BY(mutex_) = true;
  std::vector<std::thread> pool_;
  const std::string thread_name_;

  ThreadPoolID id_;

  // to make stop() thread-safe.
  ConditionVariable stop_cond_ GUARDED_BY(mutex_);
};

// Thread pools in a group will share the same task_manager, with which
// it's possible to make a task depend on tasks running in other thread pools.
class ThreadPoolGroup {
 public:
  ThreadPoolGroup();
  ~ThreadPoolGroup();

  // Create an anonymous thread pool.
  std::shared_ptr<ThreadPool> create(int num_threads = 1) EXCLUDES(mutex_);

  // Create/Remove a named thread pool.
  // Keep in mind that for every named thread pool there wiil be a hidden
  // shared_ptr referencing it after creation, hence it won't be destructed
  // before removeNamed() is called.
  std::shared_ptr<ThreadPool> createNamed(
      const std::string& thread_name, int num_threads = 1) EXCLUDES(mutex_);
  void removeNamed(
      const std::string& thread_name, bool wait_all_works_done = true)
      EXCLUDES(mutex_);

  // Get a shared_ptr to a named thread pool.
  // If the pool is not created yet, it will be auto created with num_threads=1.
  // If you need a named thread pool containing more than one threads, you have
  // to call createNamed() explicitly beforehand.
  std::shared_ptr<ThreadPool> getNamed(const std::string& thread_name)
      EXCLUDES(mutex_);

  // Remove all thread pools (including named and anonymous ones).
  void clear(bool wait_all_works_done = true) EXCLUDES(mutex_);

  ThreadPool::TaskGroup createTaskGroupForNamed(
      const std::string& thread_name) {
    return ThreadPool::TaskGroup(getNamed(thread_name));
  }

 public:
  static std::shared_ptr<ThreadPoolGroup> defaultInstance();

 private:
  void removeAnonymous(
      ThreadPoolID thread_pool_id, bool wait_all_works_done = true)
      EXCLUDES(mutex_);

  std::shared_ptr<ThreadPool> createGeneric(
      const std::string& thread_name, int num_threads) EXCLUDES(mutex_);

  void removeExpired() EXCLUDES(mutex_);

 private:
  std::shared_ptr<TaskManager> task_manager_;
  Mutex mutex_;
  std::map<ThreadPoolID, std::weak_ptr<ThreadPool>> anonymous_thread_pools_
      GUARDED_BY(mutex_);
  std::map<std::string, std::shared_ptr<ThreadPool>> named_thread_pools_
      GUARDED_BY(mutex_);
  bool clearing_ GUARDED_BY(mutex_) = false;
};

//// A thread-safe wrapper for an object of type T.
//// - Use set() to set the object to wrap;
//// - Use get() to get a reference to the object;
//// - Use destroy() to effectively & safely destroy the object.
////   (This function would make sure the destructor been called and all the
////    references to the object been released before calling the destructor)
////   NOTE: Make sure there's no reference living in the current thread before
////         calling destroy(), which will lead to an endless waiting.
//// - Use preDestroy() before calling destroy() to prevent new references.
//// - Use setDestroyWaitingPeriod() to set the waiting period in destroy() when
////   waiting for other references to be released and a callback function which
////   would be called once in every period.
template <class T>
class ThreadSafeWrapper {
 public:
  std::shared_ptr<T> get() EXCLUDES(mutex_) {
    UniqueLock locker(mutex_);
    if (!ready_) {
      return nullptr;
    } else {
      return ptr_;
    }
  }

  void set(std::unique_ptr<T>&& ptr) EXCLUDES(mutex_) {
    UniqueLock locker(mutex_);
    if (ptr && !ptr_) {
      ready_ = true;
      ptr_ = std::move(ptr);
    }
  }

  void preDestroy() EXCLUDES(mutex_) {
    UniqueLock locker(mutex_);
    ready_ = false;
  }

  void destroy() EXCLUDES(mutex_) {
    int period_ms;
    std::function<void()> period_cb;
    {
      UniqueLock locker(mutex_);
      if (!ptr_) {
        return;
      }
      period_ms = period_ms_;
      period_cb = period_cb_;
    }

    preDestroy();
    while (true) {
      {
        UniqueLock lock(mutex_);
        if (!ptr_ || ptr_.use_count() <= 1) {
          break;
        }
      }

      // Other threads are using the object which ptr_ points to.
      if (period_cb) {
        period_cb();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }

    std::shared_ptr<T> destroy_copy;
    {
      UniqueLock lock(mutex_);
      destroy_copy = ptr_;
      ptr_.reset();
    }
    printf("destroy_copy.use_count() == %ld\n", destroy_copy.use_count());
    ASSERT(destroy_copy.use_count() == 1);
    destroy_copy.reset();  // destructor is called here.
  }

  ~ThreadSafeWrapper() {
    destroy();
  }

  void setDestroyWaitingPeriod(int period_ms, std::function<void()> period_cb)
      EXCLUDES(mutex_) {
    UniqueLock lock(mutex_);
    period_cb_ = period_cb;
    period_ms_ = period_ms;
  }

 private:
  Mutex mutex_;
  std::function<void()> period_cb_ GUARDED_BY(mutex_);
  int period_ms_ GUARDED_BY(mutex_) = 20;
  bool ready_ GUARDED_BY(mutex_) = false;
  std::shared_ptr<T> ptr_ GUARDED_BY(mutex_);
};

}  // namespace sk4slam
