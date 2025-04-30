#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_cpp/deque.h"
#include "sk4slam_cpp/mutex.h"

// C++17 is required since we need the syntax "if constexpr".

namespace sk4slam {

enum class WorkQueueMode : int64_t {
  // All jobs are independent.
  NO_ORDER,

  // First-in-first-out mode.
  // A job might start at anytime, but its output callback wouldn't
  // be invoked before all the earlier (enqueued) jobs' callbacks finished.
  FIFO_ORDER
};

template <typename WorkOutputType>
struct WorkQueueOutputCallbackT : public std::function<void(WorkOutputType)> {
  template <typename F>
  WorkQueueOutputCallbackT(F cb) : std::function<void(WorkOutputType)>(cb) {}
};

template <>
struct WorkQueueOutputCallbackT<void> {};

// NOTE:
//    If `WorkInput` is raw pointer (e.g. MyStruct*), then we assume the
//    input pointers are always valid before they've been processed.
//
//    If `WorkInput` is normal type (e.g. MyStruct)， then we'll make a
//    copy of the input.
//
// NOTE:
//    `WorkOutput` is only needed in FIFO_ORDER mode (where we need to
//    set a output callback)
template <
    typename WorkInput, WorkQueueMode mode_ = WorkQueueMode::NO_ORDER,
    typename WorkOutput = void>
class WorkQueue {
  static_assert(
      mode_ == WorkQueueMode::NO_ORDER || mode_ == WorkQueueMode::FIFO_ORDER,
      "Invalid mode!");
  static_assert(
      mode_ == WorkQueueMode::NO_ORDER || !std::is_void<WorkOutput>::value,
      "WorkOutput must be specified (since we need output callbacks) in "
      "FIFO_ORDER mode!");
  static_assert(
      std::is_move_constructible<WorkInput>::value,
      "WorkInput must be move constructible!"
      " (since we'll move the input to the worker thread)");

 public:
  using WorkFunction = std::function<WorkOutput(WorkInput)>;
  using OutputCallback = WorkQueueOutputCallbackT<WorkOutput>;

  template <
      WorkQueueMode mode = mode_, ENABLE_IF(mode == WorkQueueMode::NO_ORDER)>
  // We define this constructor as a template because `enable_if` must depend on
  // a template parameter that is deduced.
  WorkQueue(
      WorkFunction work_function, const std::string& queue_name = "",
      size_t num_threads = 1, int max_queue_size = -1,
      bool remove_old_jobs_on_full_queue = false)
      : work_function_(work_function),
        queue_name_(queue_name),
        max_queue_size_(max_queue_size),
        remove_old_jobs_on_full_queue_(remove_old_jobs_on_full_queue) {
    static_assert(
        mode_ == mode, "WorkQueueMode must be NO_ORDER for this constructor!");
    for (size_t i = 0; i < num_threads; ++i) {
      worker_threads_.emplace_back(
          &WorkQueue::workerThreadFunc, this, queue_name_);
    }
  }

  template <
      WorkQueueMode mode = mode_, ENABLE_IF(mode == WorkQueueMode::FIFO_ORDER)>
  WorkQueue(
      WorkFunction work_function, OutputCallback output_callback,
      const std::string& queue_name = "", size_t num_threads = 1,
      int max_queue_size = -1, bool remove_old_jobs_on_full_queue = false)
      : work_function_(work_function),
        output_callback_(output_callback),
        queue_name_(queue_name),
        max_queue_size_(max_queue_size),
        remove_old_jobs_on_full_queue_(remove_old_jobs_on_full_queue) {
    static_assert(
        mode_ == mode,
        "WorkQueueMode must be FIFO_ORDER for this constructor!");
    ASSERT(output_callback_);
    for (size_t i = 0; i < num_threads; ++i) {
      worker_threads_.emplace_back(
          &WorkQueue::workerThreadFunc, this, queue_name_);
    }
  }

  ~WorkQueue() {
    stop();
  }

  bool enqueue(WorkInput work_input, bool log_drop = false) EXCLUDES(mutex_) {
    UniqueLock lock(mutex_);
    if (stop_request_) {
      return false;
    }
    if (!addWorkItem(std::move(work_input), log_drop)) {
      return false;
    }
    is_empty_ = false;
    cv_.notify_one();
    return true;
  }

  void waitUntilAllJobsDone() const EXCLUDES(mutex_) {
    UniqueLock lock(mutex_);
    while (!is_empty_) {
      cv_empty_.wait(lock);
    }
  }

  void stop(bool wait_until_all_jobs_done = true) EXCLUDES(mutex_) {
    bool is_first_call = false;
    {
      UniqueLock lock(mutex_);
      if (!stop_request_) {
        is_first_call = true;
        stop_request_ = true;
        wait_until_all_jobs_done_ = wait_until_all_jobs_done;
        cv_.notify_all();
      }
    }

    if (is_first_call) {
      for (auto& thread : worker_threads_) {
        thread.join();
      }

      {
        UniqueLock lock(mutex_);
        is_empty_ = true;
        cv_empty_.notify_all();
        stopped_ = true;
        cv_stop_.notify_all();
      }
    } else {
      UniqueLock lock(mutex_);
      cv_stop_.wait(lock, [this]() REQUIRES(mutex_) { return stopped_; });
    }
  }

 protected:
  struct WorkItem {
    size_t enqueued_idx;
    WorkInput work_input;
    std::shared_ptr<WorkOutput> work_output;

    explicit WorkItem(WorkInput input, size_t idx)
        : work_input(std::move(input)),
          enqueued_idx(idx),
          work_output(nullptr) {}
  };
  using WorkItemPtr = std::shared_ptr<WorkItem>;

  bool addWorkItem(WorkInput input, bool log_drop = false) REQUIRES(mutex_) {
    size_t idx = num_enqueued_++;
    bool need_trim = false;
    if (max_queue_size_ > 0 && queue_.size() >= max_queue_size_) {
      if (remove_old_jobs_on_full_queue_) {
        unfinished_jobs_.erase(queue_.front()->enqueued_idx);
        queue_.pop_front();
        need_trim = true;
        if (log_drop) {
          LOGW(
              "WorkQueue %s: queue full! dropped the oldest job!",
              queue_name_.c_str());
        }
      } else {
        if (log_drop) {
          LOGW(
              "WorkQueue %s: queue full! dropped the job!",
              queue_name_.c_str());
        }
        return false;
      }
    }
    ASSERT(queue_.size() < max_queue_size_);
    queue_.emplace_back(new WorkItem(std::move(input), idx));
    if (need_trim) {
      queue_.trim_to_optimal();  // queue_.shrink_to_fit();
    }
    unfinished_jobs_.insert(idx);
    return true;
  }

  void workerThreadFunc(const std::string& thread_name) EXCLUDES(mutex_) {
    if (!thread_name.empty()) {
      pthread_setname_np(pthread_self(), thread_name.substr(0, 15).c_str());
    }

    while (true) {
      WorkItemPtr work_item = nullptr;
      {
        UniqueLock lock(mutex_);
        cv_.wait(lock, [this]() REQUIRES(mutex_) {
          return stop_request_ || !queue_.empty();
        });

        if (stop_request_) {
          if (!wait_until_all_jobs_done_) {
            // queue_.clear();
            queue_ = Deque<WorkItemPtr>();
            if constexpr (mode_ == WorkQueueMode::FIFO_ORDER) {
              ready_outputs_.clear();
              // unfinished_jobs_.clear();
            }
            return;
          } else if (queue_.empty()) {
            return;
          }
        }
        work_item = queue_.front();
        queue_.pop_front();
        queue_.trim_to_optimal();  // queue_.shrink_to_fit();
      }

      if constexpr (mode_ == WorkQueueMode::FIFO_ORDER) {
        work_item->work_output.reset(
            new WorkOutput(work_function_(std::move(work_item->work_input))));

        UniqueLock lock(mutex_);
        ready_outputs_[work_item->enqueued_idx] = work_item;
        while (!ready_outputs_.empty() &&
               ready_outputs_.begin()->first <= *unfinished_jobs_.begin()) {
          LOGA(
              "before erase: output_ready (size=%d, next=%d), unfinished_jobs_ "
              "(size=%d, next=%d)",
              ready_outputs_.size(), ready_outputs_.begin()->first,
              unfinished_jobs_.size(), *unfinished_jobs_.begin());
          auto work_output = ready_outputs_.begin()->second->work_output;
          auto job_to_remove = ready_outputs_.begin()->first;
          ready_outputs_.erase(ready_outputs_.begin());

          lock.unlock();  // unlock to run the callback.
          output_callback_(std::move(*work_output));
          lock.lock();  // relock after the callback.
          unfinished_jobs_.erase(job_to_remove);
        }

        if (queue_.empty() && unfinished_jobs_.empty()) {
          is_empty_ = true;
          cv_empty_.notify_all();
        }
      } else {
        work_function_(std::move(work_item->work_input));

        UniqueLock lock(mutex_);
        unfinished_jobs_.erase(work_item->enqueued_idx);
        if (queue_.empty() && unfinished_jobs_.empty()) {
          is_empty_ = true;
          cv_empty_.notify_all();
        }
      }
    }
  }

 protected:
  const int max_queue_size_;
  const bool remove_old_jobs_on_full_queue_;
  const std::string queue_name_;

  std::vector<std::thread> worker_threads_;
  WorkFunction work_function_;  // The function that does the work.

  mutable Mutex mutex_;
  ConditionVariable cv_ GUARDED_BY(mutex_);
  Deque<WorkItemPtr> queue_ GUARDED_BY(mutex_);
  size_t num_enqueued_ GUARDED_BY(mutex_) = 0;
  std::set<size_t> unfinished_jobs_ GUARDED_BY(mutex_);

  bool stop_request_ GUARDED_BY(mutex_) = false;
  bool stopped_ GUARDED_BY(mutex_) = false;
  bool wait_until_all_jobs_done_ GUARDED_BY(mutex_) = true;

  mutable ConditionVariable cv_empty_ GUARDED_BY(mutex_);
  bool is_empty_ GUARDED_BY(mutex_) = true;

  // Only needed in FIFO_ORDER mode.
  OutputCallback output_callback_;  // The function that handles the
                                    // output in FIFO_ORDER mode.
  std::map<size_t, WorkItemPtr> ready_outputs_ GUARDED_BY(mutex_);

  // to make stop() thread-safe
  ConditionVariable cv_stop_ GUARDED_BY(mutex_);
};

template <typename WorkInput, typename WorkOutput>
using WorkQueueFIFO =
    WorkQueue<WorkInput, WorkQueueMode::FIFO_ORDER, WorkOutput>;

class TaskQueue
    : public WorkQueue<std::function<void()>, WorkQueueMode::NO_ORDER, void> {
 public:
  using Base = WorkQueue<std::function<void()>, WorkQueueMode::NO_ORDER, void>;
  TaskQueue(
      const std::string& queue_name = "", size_t num_threads = 1,
      int max_queue_size = -1, bool remove_old_jobs_on_full_queue = false)
      : Base(
            [](std::function<void()> job) { job(); }, queue_name, num_threads,
            max_queue_size, remove_old_jobs_on_full_queue) {}
};

}  // namespace sk4slam
