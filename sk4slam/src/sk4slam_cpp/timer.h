#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "sk4slam_basic/string_helper.h"
#include "sk4slam_cpp/mutex.h"

namespace sk4slam {

class TimerTask;
using TimerTaskID = const TimerTask*;
static constexpr TimerTaskID INVALID_TIMER_TASK = nullptr;

class Timer {
 public:
  // Create an anonymous timer.
  static std::shared_ptr<Timer> create(int num_threads = 1);

  // Create/Remove a named timer.
  // Keep in mind that for every named timer there wiil be a hidden
  // shared_ptr referencing it after creation, hence it won't be destructed
  // before removeNamed() is called.
  static std::shared_ptr<Timer> createNamed(
      const std::string& timer_name, int num_threads = 1);
  static void removeNamed(const std::string& timer_name);

  // Get a shared_ptr to a named timer.
  // If the timer is not created yet, it will be auto created with
  // num_threads=1. If you need a named timer employing more than one threads,
  // you have to call createNamed() explicitly beforehand.
  static std::shared_ptr<Timer> getNamed(const std::string& timer_name);

  // the <work> should return a boolean value. and when it returns true,
  // it will be scheduled for the next run after <period_ms> ms;
  // otherwise, it will be removed and thus won't run anymore.
  TimerTaskID schedule(
      const std::function<bool()>& work, double period_ms,
      double first_delay_ms = 0.0, bool precise = true) EXCLUDES(mutex_);

  TimerTaskID scheduleOnce(
      const std::function<void()>& work, double delay_ms = 0.0)
      EXCLUDES(mutex_);

  void cancel(TimerTaskID task) EXCLUDES(mutex_);

 public:
  using TimePoint = std::chrono::system_clock::time_point;
  static TimePoint now();
  TimerTaskID scheduleOnce(
      const std::function<void()>& work, const TimePoint& time_point)
      EXCLUDES(mutex_);

  // num_threads
  inline int numThreads() const {
    return threads_.size();
  }

  ~Timer();

 private:
  explicit Timer(const std::string& thread_name = "", int num_threads = 1);
  void stop() EXCLUDES(mutex_);
  void doWork() EXCLUDES(mutex_);
  bool addTask(const TimePoint& time_point, std::shared_ptr<TimerTask> task)
      REQUIRES(mutex_);
  void removeTask(std::shared_ptr<TimerTask> task) REQUIRES(mutex_);
  bool scheduleOnce(
      std::shared_ptr<TimerTask> task, const TimePoint& time_point)
      EXCLUDES(mutex_);
  bool scheduleOnce(std::shared_ptr<TimerTask> task, double delay_ms)
      EXCLUDES(mutex_);
  std::shared_ptr<TimerTask> makePeriodHandler(
      double period_ms, std::function<bool()> work);
  std::shared_ptr<TimerTask> makePrecisePeriodHandler(
      const TimePoint& cur_time_point, double period_ms,
      std::function<bool()> work);

 private:
  Mutex mutex_;
  ConditionVariable cond_ GUARDED_BY(mutex_);
  using ConcurrentTasks = std::map<TimerTaskID, std::shared_ptr<TimerTask>>;
  std::map<TimePoint, ConcurrentTasks> tasks_ GUARDED_BY(mutex_);
  std::map<TimerTaskID, TimePoint> task_to_time_point_ GUARDED_BY(mutex_);
  std::map<TimerTaskID, std::shared_ptr<TimerTask>> running_tasks_
      GUARDED_BY(mutex_);
  std::vector<std::shared_ptr<std::thread>> threads_;
  bool stop_ GUARDED_BY(mutex_) = false;
};

}  // namespace sk4slam
