
#include "sk4slam_cpp/timer.h"

#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/time.h"

namespace sk4slam {

class TimerTask {
 public:
  ~TimerTask() {
    stop();
  }

  inline TimerTaskID id() const {
    return this;
  }

 private:
  friend class Timer;
  void execute() EXCLUDES(mutex_);
  void stop() EXCLUDES(mutex_);
  bool isStopped() EXCLUDES(mutex_);
  void setWorkItem(const std::function<void()>& work_item);

 private:
  std::function<void()> work_item_;
  Mutex mutex_;
  bool executing_ GUARDED_BY(mutex_) = false;
  bool stop_ GUARDED_BY(mutex_) = false;
  ConditionVariable cond_ GUARDED_BY(mutex_);
};

void TimerTask::setWorkItem(const std::function<void()>& work_item) {
  work_item_ = work_item;
}

void TimerTask::execute() {
  {
    UniqueLock locker(mutex_);
    if (!work_item_ || stop_) {
      return;
    }
    executing_ = true;
  }
  work_item_();

  UniqueLock locker(mutex_);
  executing_ = false;
  cond_.notify_all();
}

void TimerTask::stop() {
  const auto predicate = [&]() REQUIRES(mutex_) { return !executing_; };

  {
    UniqueLock locker(mutex_);
    stop_ = true;
    cond_.wait(locker, predicate);
  }
}

bool TimerTask::isStopped() {
  UniqueLock locker(mutex_);
  return stop_;
}

Timer::Timer(const std::string& thread_name, int num_threads) {
  for (int i = 0; i != num_threads; ++i) {
    threads_.emplace_back(new std::thread([this, thread_name]() {
      if (!thread_name.empty()) {
        pthread_setname_np(pthread_self(), thread_name.substr(0, 15).c_str());
      }
      doWork();
    }));
  }
}

Timer::~Timer() {
  stop();
}

Timer::TimePoint Timer::now() {
  return std::chrono::system_clock::now();
}

void Timer::stop() {
  {
    UniqueLock locker(mutex_);
    stop_ = true;
    cond_.notify_all();
  }

  for (auto thread : threads_) {
    thread->join();
  }
}

void Timer::doWork() {
  while (true) {
    std::shared_ptr<TimerTask> pending_task;
    {
      UniqueLock locker(mutex_);
      cond_.wait(locker, [this]() REQUIRES(mutex_) {
        return !tasks_.empty() || stop_;
      });

      if (stop_) {
        return;
      }

      auto begin = tasks_.begin();
      if (cond_.wait_until(locker, begin->first) == std::cv_status::timeout) {
        if (begin ==
            tasks_.begin()) {  // make sure the first task is unchanged.
          pending_task = begin->second.begin()->second;
          removeTask(pending_task);
          running_tasks_[pending_task->id()] = pending_task;
          cond_.notify_one();
        } else {
          LOGI(
              "Timer::doWork():  the beginning task changed during waiting. "
              "(the earlier beginning task might be JUST removed, or a new "
              "task might "
              "be JUST inserted at the beginnig)");
        }
      }
    }

    if (pending_task) {
      pending_task->execute();

      UniqueLock locker(mutex_);
      running_tasks_.erase(pending_task->id());
    }
  }
}

bool Timer::addTask(
    const TimePoint& time_point, std::shared_ptr<TimerTask> task) {
  if (stop_) {
    LOGW("Timer::addTask(): Failed to add task because the timer is stopped.");
    return false;
  }

  if (task_to_time_point_.count(task->id())) {
    LOGW(
        "Timer::addTask(): Failed to add task because the task is already "
        "scheduled!");
    return false;
  }

  if (task->isStopped()) {
    LOGW(
        "Timer::addTask(): Failed to add task because the task "
        "has already been stopped!");
    return false;
  }

  auto task_id = task->id();
  if (tasks_.count(time_point) == 0) {
    tasks_[time_point] = ConcurrentTasks();
  }
  tasks_[time_point][task_id] = task;
  task_to_time_point_[task_id] = time_point;
  return true;
}

void Timer::removeTask(std::shared_ptr<TimerTask> task) {
  auto task_id = task->id();
  if (task_to_time_point_.count(task_id)) {
    auto time_point = task_to_time_point_[task_id];
    task_to_time_point_.erase(task_id);
    tasks_[time_point].erase(task_id);
    if (tasks_[time_point].empty()) {
      tasks_.erase(time_point);
    }
  }
}

bool Timer::scheduleOnce(
    std::shared_ptr<TimerTask> task, const TimePoint& time_point) {
  UniqueLock locker(mutex_);
  bool ret = addTask(time_point, task);
  if (ret) {
    cond_.notify_one();
  }
  return ret;
}

bool Timer::scheduleOnce(std::shared_ptr<TimerTask> task, double delay_ms) {
  // return scheduleOnce(task, now() + std::chrono::milliseconds(delay_ms));
  return scheduleOnce(
      task, now() + std::chrono::nanoseconds(int64_t(delay_ms * 1e6)));
}

TimerTaskID Timer::scheduleOnce(
    const std::function<void()>& work, const TimePoint& time_point) {
  auto task = std::make_shared<TimerTask>();
  task->setWorkItem(work);
  if (scheduleOnce(task, time_point)) {
    return task->id();
  } else {
    LOGW("Timer::scheduleOnce Failed!");
    return INVALID_TIMER_TASK;
  }
}

TimerTaskID Timer::scheduleOnce(
    const std::function<void()>& work, double delay_ms) {
  auto task = std::make_shared<TimerTask>();
  task->setWorkItem(work);
  if (scheduleOnce(task, delay_ms)) {
    return task->id();
  } else {
    LOGW("Timer::scheduleOnce Failed!");
    return INVALID_TIMER_TASK;
  }
}

std::shared_ptr<TimerTask> Timer::makePeriodHandler(
    double period_ms, std::function<bool()> work) {
  auto task = std::make_shared<TimerTask>();
  std::weak_ptr<TimerTask> weak_task = task;
  task->setWorkItem([this, period_ms, work, weak_task]() {
    if (work()) {
      // schedule for next run
      std::shared_ptr<TimerTask> task = weak_task.lock();
      if (task) {
        scheduleOnce(task, period_ms);
      }
    }
  });
  return task;
}

std::shared_ptr<TimerTask> Timer::makePrecisePeriodHandler(
    const TimePoint& cur_time_point, double period_ms,
    std::function<bool()> work) {
  std::shared_ptr<TimePoint> cur_time_point_ptr(new TimePoint(cur_time_point));
  auto task = std::make_shared<TimerTask>();
  std::weak_ptr<TimerTask> weak_task = task;
  task->setWorkItem([this, cur_time_point_ptr, period_ms, work, weak_task]() {
    if (work()) {
      // schedule for next run
      auto now = this->now();
      int64_t ns = (now - *cur_time_point_ptr) / std::chrono::nanoseconds(1);
      int64_t interval_ns = int64_t(period_ms * 1e6);
      auto next_time_point =
          *cur_time_point_ptr + std::chrono::nanoseconds(
                                    // round up to next interval
                                    ((ns / interval_ns) + 1) * interval_ns);
      std::shared_ptr<TimerTask> task = weak_task.lock();
      if (task) {
        *cur_time_point_ptr = next_time_point;
        scheduleOnce(task, next_time_point);
      }
    }
  });
  return task;
}

TimerTaskID Timer::schedule(
    const std::function<bool()>& work, double period_ms, double first_delay_ms,
    bool precise) {
  ASSERT(period_ms > 0 && first_delay_ms >= 0);

  if (precise) {
    auto begin_time =
        now() + std::chrono::nanoseconds(int64_t(first_delay_ms * 1e6));
    auto task = makePrecisePeriodHandler(begin_time, period_ms, work);
    if (scheduleOnce(task, begin_time)) {
      return task->id();
    } else {
      LOGW("Timer::schedule Failed!");
      return INVALID_TIMER_TASK;
    }
  } else {
    auto task = makePeriodHandler(period_ms, work);
    if (scheduleOnce(task, first_delay_ms)) {
      return task->id();
    } else {
      LOGW("Timer::schedule Failed!");
      return INVALID_TIMER_TASK;
    }
  }
}

void Timer::cancel(TimerTaskID task_id) {
  std::shared_ptr<TimerTask> task;
  {
    UniqueLock locker(mutex_);
    if (running_tasks_.count(task_id)) {
      task = running_tasks_.at(task_id);
    } else if (task_to_time_point_.count(task_id)) {
      auto time_point = task_to_time_point_.at(task_id);
      ASSERT(tasks_.count(time_point));
      ASSERT(tasks_.at(time_point).count(task_id));
      task = tasks_.at(time_point).at(task_id);
    } else {
      // task not found
      return;
    }
    ASSERT(task);
  }

  task->stop();
  ASSERT(task->isStopped());

  {
    UniqueLock locker(mutex_);
    removeTask(task);
    cond_.notify_one();
  }
}

namespace {
Mutex static_mutex_;
std::map<std::string, std::shared_ptr<Timer>> named_timers_
    GUARDED_BY(static_mutex_);
}  // namespace

std::shared_ptr<Timer> Timer::create(int num_threads) {
  std::shared_ptr<Timer> timer(new Timer("", num_threads));
  return timer;
}

std::shared_ptr<Timer> Timer::createNamed(
    const std::string& timer_name, int num_threads) {
  {
    SharedLock locker(static_mutex_);
    auto it = named_timers_.find(timer_name);
    if (it != named_timers_.end()) {
      LOGW(
          "Timer::createNamed(): attempting to re-create the named "
          "timer '%s' (ignored)",
          timer_name.c_str());
      return it->second;
    }
  }

  std::shared_ptr<Timer> timer(new Timer(timer_name, num_threads));

  {
    UniqueLock locker(static_mutex_);

    // recheck the conditions
    auto it = named_timers_.find(timer_name);
    if (it != named_timers_.end()) {
      LOGW(
          "Timer::createNamed(): attempting to re-create the named "
          "timer '%s' (ignored)",
          timer_name.c_str());
      return it->second;
    }
    named_timers_[timer_name] = timer;
  }

  return timer;
}

std::shared_ptr<Timer> Timer::getNamed(const std::string& timer_name) {
  {
    SharedLock locker(static_mutex_);
    auto it = named_timers_.find(timer_name);
    if (it != named_timers_.end()) {
      return it->second;
    }
  }
  return createNamed(timer_name, 1);
}

void Timer::removeNamed(const std::string& timer_name) {
  std::shared_ptr<Timer> timer = nullptr;
  {
    UniqueLock locker(static_mutex_);
    auto it = named_timers_.find(timer_name);
    if (it != named_timers_.end()) {
      timer = it->second;
      named_timers_.erase(it);
    }
  }

  if (timer) {
    timer->stop();
  }
}

}  // namespace sk4slam
