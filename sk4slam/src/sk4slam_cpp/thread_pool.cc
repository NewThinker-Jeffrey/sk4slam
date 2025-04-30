
#include "sk4slam_cpp/thread_pool.h"

#include <atomic>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/time.h"

namespace sk4slam {

class Task {
 public:
  explicit Task(const std::function<void()>& work_item)
      : work_item_(work_item) {
    static std::atomic<size_t> last_id(0);
    id_.id = ++last_id;
  }
  void execute() {
    if (work_item_) {
      work_item_();
    }
  }
  inline TaskID id() const {
    return id_;
  }

 private:
  std::function<void()> work_item_;
  std::string debug_label_;
  TimeCounter tc_;
  TaskID id_;
  friend class TaskManager;
  friend class ThreadPool;
};

class TaskManager {
 public:
  void bindThreadPool(std::shared_ptr<ThreadPool> pool) EXCLUDES(mutex_);

  // TaskManager will take the unique ownership of 'task'
  TaskID addTask(
      std::unique_ptr<Task>&& task, ThreadPoolID pool, bool* pending,
      const std::set<TaskID>& dependencies = std::set<TaskID>())
      EXCLUDES(mutex_);

  std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>> execute(
      TaskID task) EXCLUDES(mutex_);

  bool wait(TaskID task) EXCLUDES(mutex_);

  void waitUntilThreadIdle(ThreadPoolID pool) EXCLUDES(mutex_);

  // return the number of tasks that depend on the current task.
  // (these tasks would be removed as well)
  int removeTask(TaskID task) EXCLUDES(mutex_);

  int numUnfinishedTasksForThreadPool(ThreadPoolID pool) EXCLUDES(mutex_);

  // return the number of tasks in OTHER thread-pools which depend on some task
  // in the current pool. (these tasks would be removed as well)
  int removeAllUnfinishedTasksForThreadPool(ThreadPoolID pool) EXCLUDES(mutex_);

 private:
  void dispatch(TaskID task, ThreadPoolID pool) REQUIRES(mutex_);

  void addDependencies(TaskID task, const std::set<TaskID>& dependencies)
      REQUIRES(mutex_);

  // If the task is ready to run, return the thread pool in which
  // the task will run. Othrewise, return nullptr.
  ThreadPoolID checkDependenciesForTask(TaskID task) REQUIRES(mutex_);

  void removeTaskNoLock(TaskID task) REQUIRES(mutex_);

  std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>> finishTask(
      TaskID task) EXCLUDES(mutex_);

 private:
  Mutex mutex_;

  // For wait();
  std::map<TaskID, std::shared_ptr<ConditionVariable>> wait_task_conds_
      GUARDED_BY(mutex_);
  inline void notifyTask(TaskID task) REQUIRES(mutex_) {
    auto it = wait_task_conds_.find(task);
    if (it != wait_task_conds_.end()) {
      it->second->notify_all();
    }
  }

  std::map<ThreadPoolID, std::shared_ptr<ConditionVariable>> wait_pool_conds_
      GUARDED_BY(mutex_);
  inline void notifyPool(ThreadPoolID pool) REQUIRES(mutex_) {
    auto it = wait_pool_conds_.find(pool);
    if (it != wait_pool_conds_.end()) {
      it->second->notify_all();
    }
  }

  // TaskManager must take the ownership of all the tasks.
  std::map<TaskID, std::unique_ptr<Task>> tasks_ GUARDED_BY(mutex_);

  // tasks that are not ready to run (still have unfinished dependencies);
  // mapping a task to the thread pool in which the task will run.
  std::map<TaskID, ThreadPoolID> unready_task_to_thread_pool_
      GUARDED_BY(mutex_);

  // tasks that are ready to run;
  std::map<TaskID, ThreadPoolID> pending_task_to_thread_pool_
      GUARDED_BY(mutex_);

  // the inverse map for "unready_task_to_thread_pool_" and
  // "pending_task_to_thread_pool_".
  std::map<ThreadPoolID, std::set<TaskID>> thread_pool_to_unready_task_
      GUARDED_BY(mutex_);
  std::map<ThreadPoolID, std::set<TaskID>> thread_pool_to_pending_task_
      GUARDED_BY(mutex_);

  std::map<TaskID, std::set<TaskID>> dependencies_ GUARDED_BY(mutex_);
  std::map<TaskID, std::set<TaskID>> dependent_tasks_ GUARDED_BY(mutex_);

  std::map<ThreadPoolID, std::weak_ptr<ThreadPool>> binded_pools_
      GUARDED_BY(mutex_);
};

void TaskManager::bindThreadPool(std::shared_ptr<ThreadPool> pool) {
  UniqueLock locker(mutex_);

  // first remove expired pools
  for (auto it = binded_pools_.begin(); it != binded_pools_.end();) {
    if (it->second.expired()) {
      it = binded_pools_.erase(it);
    } else {
      ++it;
    }
  }

  // then add the new one
  binded_pools_[pool->id()] = pool;
}

TaskID TaskManager::addTask(
    std::unique_ptr<Task>&& unique_task, ThreadPoolID pool, bool* pending,
    const std::set<TaskID>& dependencies) {
  TaskID task = unique_task->id();
  ThreadPoolID thread_pool = INVALID_THREAD_POOL;
  {
    UniqueLock locker(mutex_);
    tasks_[task] = std::move(unique_task);
    dependencies_[task] = std::set<TaskID>();
    dependent_tasks_[task] = std::set<TaskID>();
    addDependencies(task, dependencies);
    dispatch(task, pool);
    thread_pool = checkDependenciesForTask(task);
  }

  if (pending) {
    if (thread_pool) {
      *pending = true;
    } else {
      *pending = false;
    }
  }

  return task;
}

void TaskManager::addDependencies(
    TaskID task, const std::set<TaskID>& dependencies) {
  //  if (tasks_.count(task) == 0) {
  //    LOGW("TaskManager::addDependencies(): Unregistered task!!");
  //    return;
  //  }

  // Do NOT add dependencies for an already dispatched task
  // if (unready_task_to_thread_pool_.count(task) != 0) {
  //   LOGW("TaskManager::addDependencies(): Attempt to add dependencies "
  //         "for an already dispatched task!!");
  //   return;
  // }

  for (TaskID dependency : dependencies) {
    if (tasks_.count(dependency) != 0) {
      dependencies_[task].insert(dependency);
      dependent_tasks_[dependency].insert(task);
    } else {
      // // When the dependency task is expired (i.e. completed),
      // // no longer need to add it as a dependency.
      // LOGW(
      //     "TaskManager::addDependencies(): Unregistered or expired "
      //     "dependency!!");
    }
  }
}

void TaskManager::dispatch(TaskID task, ThreadPoolID pool) {
  // if (unready_task_to_thread_pool_.count(task) != 0) {
  //   LOGW("TaskManager::dispatch(): Attempt to dispatch a task twice!!");
  //   return;
  // }

  // if (tasks_.count(task) == 0) {
  //   LOGW("TaskManager::dispatch(): Unregistered task!!");
  //   return;
  // }

  unready_task_to_thread_pool_[task] = pool;
  if (thread_pool_to_unready_task_.count(pool) == 0) {
    thread_pool_to_unready_task_[pool] = std::set<TaskID>();
  }
  thread_pool_to_unready_task_[pool].insert(task);
}

ThreadPoolID TaskManager::checkDependenciesForTask(TaskID task) {
  ThreadPoolID thread_pool = INVALID_THREAD_POOL;
  if (dependencies_.count(task) != 0) {
    // printf("DEBUG: checkDependenciesForTask dependencies_[task].size=%d \n",
    //        dependencies_[task].size());
    if (dependencies_[task].empty() &&
        unready_task_to_thread_pool_.count(task) != 0) {
      // all dependencies ready.
      thread_pool = unready_task_to_thread_pool_[task];

      unready_task_to_thread_pool_.erase(task);
      thread_pool_to_unready_task_[thread_pool].erase(task);
      if (thread_pool_to_unready_task_[thread_pool].empty()) {
        thread_pool_to_unready_task_.erase(thread_pool);
      }

      pending_task_to_thread_pool_[task] = thread_pool;
      if (thread_pool_to_pending_task_.count(thread_pool) == 0) {
        thread_pool_to_pending_task_[thread_pool] = std::set<TaskID>();
      }
      thread_pool_to_pending_task_[thread_pool].insert(task);
    }
  }

  return thread_pool;
}

bool TaskManager::wait(TaskID task) {
  UniqueLock locker(mutex_);
  if (!tasks_.count(task)) {
    return false;
  }

  if (!wait_task_conds_.count(task)) {
    wait_task_conds_[task].reset(new ConditionVariable());
  }
  auto cond = wait_task_conds_.at(task);
  cond->wait(locker);

  if (wait_task_conds_.count(task)) {
    wait_task_conds_.erase(task);
  }
  return true;
}

void TaskManager::waitUntilThreadIdle(ThreadPoolID pool) {
  UniqueLock locker(mutex_);
  if (!thread_pool_to_pending_task_.count(pool) &&
      !thread_pool_to_unready_task_.count(pool)) {
    return;
  }

  if (!wait_pool_conds_.count(pool)) {
    wait_pool_conds_[pool].reset(new ConditionVariable());
  }
  auto cond = wait_pool_conds_.at(pool);
  cond->wait(locker);

  if (wait_pool_conds_.count(pool)) {
    wait_pool_conds_.erase(pool);
  }
}

std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>>
TaskManager::finishTask(TaskID task) {
  UniqueLock locker(mutex_);
  std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>> new_pending_tasks;
  if (tasks_.count(task) != 0) {
    // the unique pointer of a finished task should be already reset.
    ASSERT(tasks_[task] == nullptr);
    ASSERT(dependent_tasks_.count(task) != 0);
    ASSERT(dependencies_.count(task) != 0 && dependencies_[task].empty());
    for (TaskID dependent_task : dependent_tasks_[task]) {
      dependencies_[dependent_task].erase(task);
      ThreadPoolID thread_pool = checkDependenciesForTask(dependent_task);
      if (thread_pool) {
        auto it = binded_pools_.find(thread_pool);
        ASSERT(it != binded_pools_.end());
        auto shared_pool = it->second.lock();
        if (shared_pool) {
          new_pending_tasks.push_back(
              std::make_pair(shared_pool, dependent_task));
        }
      }
    }
    dependencies_.erase(task);
    dependent_tasks_.erase(task);
    tasks_.erase(task);

    notifyTask(task);

    ThreadPoolID thread_pool = pending_task_to_thread_pool_[task];
    pending_task_to_thread_pool_.erase(task);
    thread_pool_to_pending_task_[thread_pool].erase(task);
    if (thread_pool_to_pending_task_[thread_pool].empty()) {
      thread_pool_to_pending_task_.erase(thread_pool);
      if (!thread_pool_to_unready_task_.count(thread_pool)) {
        // If there's neither pending nor unready task belonging to pool,
        // the pool is emptyed.
        notifyPool(thread_pool);
      }
    }
    ASSERT(unready_task_to_thread_pool_.count(task) == 0);
  }
  return new_pending_tasks;
}

std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>>
TaskManager::execute(TaskID task) {
  std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>> new_pending_tasks;
  std::unique_ptr<Task> tmp_task(nullptr);
  {
    UniqueLock locker(mutex_);
    if (tasks_.count(task) != 0) {
      if (pending_task_to_thread_pool_.count(task) == 0) {
        LOGW(
            "TaskManager::execute: CHECK failed!!! --> "
            "Task in not pending!");
        return new_pending_tasks;
      }

      tmp_task = std::move(tasks_[task]);
      tasks_[task].reset();
      if (!tmp_task) {
        // Shouldn't happen.
        LOGE(
            "TaskManager::execute(): BUG: Another thread is running the task!");
        ASSERT(tmp_task);  // report a bug.
        return new_pending_tasks;
      }
    }
  }

  if (tmp_task) {
    tmp_task->execute();
    new_pending_tasks = finishTask(task);
    return new_pending_tasks;
  } else {
    // skip expired task.
    LOGW("TaskManager::execute: skip expired task");
    return new_pending_tasks;
  }
}

int TaskManager::removeTask(TaskID task) {
  UniqueLock locker(mutex_);
  int tmp = tasks_.size();
  removeTaskNoLock(task);
  int total_removed = tmp - tasks_.size();
  return total_removed > 0 ? total_removed - 1 : 0;
}

void TaskManager::removeTaskNoLock(TaskID task) {
  if (tasks_.count(task) != 0) {
    tasks_.erase(task);

    if (unready_task_to_thread_pool_.count(task) != 0) {
      ThreadPoolID pool = unready_task_to_thread_pool_[task];
      unready_task_to_thread_pool_.erase(task);
      thread_pool_to_unready_task_[pool].erase(task);
      if (thread_pool_to_unready_task_[pool].empty()) {
        thread_pool_to_unready_task_.erase(pool);
        if (!thread_pool_to_pending_task_.count(pool)) {
          // If there's neither pending nor unready task belonging to pool,
          // the pool is emptyed.
          notifyPool(pool);
        }
      }
    } else {
      ASSERT(pending_task_to_thread_pool_.count(task) != 0);
      ThreadPoolID pool = pending_task_to_thread_pool_[task];
      pending_task_to_thread_pool_.erase(task);
      thread_pool_to_pending_task_[pool].erase(task);
      if (thread_pool_to_pending_task_[pool].empty()) {
        thread_pool_to_pending_task_.erase(pool);
        if (!thread_pool_to_unready_task_.count(pool)) {
          // If there's neither pending nor unready task belonging to pool,
          // the pool is emptyed.
          notifyPool(pool);
        }
      }
    }

    notifyTask(task);

    for (TaskID dependency : dependencies_[task]) {
      dependent_tasks_[dependency].erase(task);
    }
    dependencies_.erase(task);

    for (TaskID dependent_task : dependent_tasks_[task]) {
      removeTaskNoLock(dependent_task);
    }
    ASSERT(dependent_tasks_[task].empty());
    dependent_tasks_.erase(task);
  }
}

int TaskManager::numUnfinishedTasksForThreadPool(ThreadPoolID pool) {
  int unready = 0;
  int pending = 0;

  SharedLock locker(mutex_);
  if (thread_pool_to_unready_task_.count(pool) != 0) {
    ASSERT(!thread_pool_to_unready_task_.at(pool).empty());
    unready = thread_pool_to_unready_task_.at(pool).size();
  }
  if (thread_pool_to_pending_task_.count(pool) != 0) {
    ASSERT(!thread_pool_to_pending_task_.at(pool).empty());
    pending = thread_pool_to_pending_task_.at(pool).size();
  }

  return unready + pending;
}

int TaskManager::removeAllUnfinishedTasksForThreadPool(ThreadPoolID pool) {
  int tasks_in_cur_pool = 0;

  UniqueLock locker(mutex_);
  int tmp = tasks_.size();

  if (thread_pool_to_unready_task_.count(pool) != 0) {
    ASSERT(!thread_pool_to_unready_task_[pool].empty());
    std::set<TaskID> unready_tasks_copy(thread_pool_to_unready_task_[pool]);
    tasks_in_cur_pool += unready_tasks_copy.size();
    for (TaskID task : unready_tasks_copy) {
      removeTaskNoLock(task);
    }
  }

  if (thread_pool_to_pending_task_.count(pool) != 0) {
    ASSERT(!thread_pool_to_pending_task_[pool].empty());
    std::set<TaskID> pending_tasks_copy(thread_pool_to_pending_task_[pool]);
    tasks_in_cur_pool += pending_tasks_copy.size();
    for (TaskID task : pending_tasks_copy) {
      removeTaskNoLock(task);
    }
  }

  ASSERT(thread_pool_to_unready_task_.count(pool) == 0);
  ASSERT(thread_pool_to_pending_task_.count(pool) == 0);

  int total_removed = tmp - tasks_.size();
  ASSERT(total_removed >= tasks_in_cur_pool);
  int tasks_in_other_pools = total_removed - tasks_in_cur_pool;
  return tasks_in_other_pools;
}

////////////////////////////////////////////

ThreadPool::ThreadPool(
    int num_threads, const std::string& thread_name,
    std::shared_ptr<TaskManager> task_manager)
    : thread_name_(thread_name), task_manager_(task_manager) {
  static std::atomic<size_t> last_id(0);
  id_.id = ++last_id;
  initThreads(num_threads);
}

ThreadPool::~ThreadPool() {
  stop(true);
  task_manager_.reset();
}

void ThreadPool::initThreads(int num_threads) {
  UniqueLock locker(mutex_);
  for (int i = 0; i != num_threads; ++i) {
    pool_.emplace_back([this]() {
      if (!thread_name_.empty()) {
        pthread_setname_np(pthread_self(), thread_name_.substr(0, 15).c_str());
      }
      doWork();
    });
  }
}

TaskID ThreadPool::schedule(
    std::function<void()> work_item, const std::set<TaskID>& dependencies) {
  std::unique_ptr<Task> unique_task(new Task(work_item));

  UniqueLock locker(mutex_);
  bool pending = false;
  if (!frozen_) {
    TaskID task = task_manager_->addTask(
        std::move(unique_task), id(), &pending, dependencies);
    if (pending) {
      addPendingTaskNoLock(task);
    }
    return task;
  }

  LOGW("ThreadPool::schedule Failed!");
  return INVALID_TASK;
}

void ThreadPool::freeze() {
  UniqueLock locker(mutex_);
  frozen_ = true;
}

void ThreadPool::unfreeze() {
  UniqueLock locker(mutex_);
  if (stop_request_) {
    LOGW(YELLOW
         "ThreadPool::unfreeze() called ater requesting stop! "
         "(will be ignored)" RESET);
    return;
  }
  frozen_ = false;
}

bool ThreadPool::isFrozen() const {
  SharedLock locker(mutex_);
  return frozen_;
}

bool ThreadPool::wait(TaskID task) {
  if (task == INVALID_TASK) {
    return false;
  }
  return task_manager_->wait(task);
}

void ThreadPool::waitUntilAllTasksDone() {
  task_manager_->waitUntilThreadIdle(id());
}

void ThreadPool::stop(bool wait_all_tasks_done) {
  // It's possible that more than one threads call stop() meanwhile.
  // Only the first call effectively determines the value of
  // 'wait_all_works_done'.

  bool is_first_call_to_stop = false;

  {
    UniqueLock locker(mutex_);
    if (stopped_) {
      // already stopped_.
      return;
    }

    if (!stop_request_) {
      // it's the first call
      is_first_call_to_stop = true;
      stop_request_ = true;

      frozen_ = true;
      wait_all_tasks_done_before_stop_ = wait_all_tasks_done;

      if (!wait_all_tasks_done_before_stop_) {
        int tasks_in_cur_pool =
            task_manager_->numUnfinishedTasksForThreadPool(id());
        int removed_tasks_in_other_pools =
            task_manager_->removeAllUnfinishedTasksForThreadPool(id());
        if (tasks_in_cur_pool == 0) {
          ASSERT(removed_tasks_in_other_pools == 0);
        }

        pending_tasks_.clear();
        LOGW(
            "ThreadPool::clear(): [at most %d] tasks in the current pool and "
            "[exact %d] tasks in other pools have been abandoned!",
            tasks_in_cur_pool, removed_tasks_in_other_pools);
      }

      cond_.notify_all();
    }
  }

  if (is_first_call_to_stop) {
    for (std::thread& thread : pool_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    UniqueLock locker(mutex_);
    stopped_ = true;
    stop_cond_.notify_all();
  } else {
    UniqueLock locker(mutex_);
    stop_cond_.wait(locker, [this]() REQUIRES(mutex_) { return stopped_; });
  }

  {
    SharedLock locker(mutex_);
    ASSERT(task_manager_->numUnfinishedTasksForThreadPool(id()) == 0);
    ASSERT(pending_tasks_.empty());
    ASSERT(stopped_);
  }
}

void ThreadPool::addPendingTask(TaskID task) {
  UniqueLock locker(mutex_);
  addPendingTaskNoLock(task);
}

void ThreadPool::addPendingTaskNoLock(TaskID task) {
  if (!stop_request_ || wait_all_tasks_done_before_stop_) {
    pending_tasks_.push_back(task);
    cond_.notify_one();
  }
}

TaskID ThreadPool::waitForNextTask(bool* p_exit) {
  bool exit = false;
  TaskID task = INVALID_TASK;

  const auto predicate = [&]() REQUIRES(mutex_) {
    if (stop_request_ && !wait_all_tasks_done_before_stop_) {
      // exit current thread immediately when
      // (!wait_all_tasks_done_before_stop_).
      exit = true;
      return true;
    } else if (!pending_tasks_.empty()) {
      task = pending_tasks_.front();
      pending_tasks_.pop_front();
      pending_tasks_.trim_to_optimal();  // pending_tasks_.shrink_to_fit();
      return true;
    } else if (stop_request_) {  // wait_all_tasks_done_before_stop_
      if (0 == task_manager_->numUnfinishedTasksForThreadPool(id())) {
        // exit current thread when all tasks finished.
        exit = true;
        // other threads might be still waiting for the unfinished tasks.
        // wake them up and let them exit.
        cond_.notify_all();  // wake up all in one-shot.
                             // redundant calls may be made.
        // cond_.notify_one();  // wake up one by one. maybe better?
        return true;
      } else {
        return false;  // wait for the unfinished tasks.
      }
    } else {
      return false;
    }
  };

  UniqueLock locker(mutex_);
  cond_.wait(locker, predicate);

  *p_exit = exit;
  return task;
}

void ThreadPool::doWork() {
  for (;;) {
    bool exit;
    TaskID task = waitForNextTask(&exit);
    if (exit) {
      return;
    }

    if (task != INVALID_TASK) {
      // std::vector<std::pair<std::shared_ptr<ThreadPool>, TaskID>>
      // new_pending_tasks;
      auto new_pending_tasks = task_manager_->execute(task);
      for (auto pending_task_pair : new_pending_tasks) {
        // todo: thread_pool should be a weak ptr?
        std::shared_ptr<ThreadPool> thread_pool = pending_task_pair.first;
        TaskID pending_task = pending_task_pair.second;
        thread_pool->addPendingTask(pending_task);
      }
    } else {
      // expired task
      LOGW("ThreadPool::doWork(): Expired task!");
    }
  }
}

std::shared_ptr<ThreadPool> ThreadPool::create(int num_threads) {
  return ThreadPoolGroup::defaultInstance()->create(num_threads);
}

std::shared_ptr<ThreadPool> ThreadPool::createNamed(
    const std::string& thread_name, int num_threads) {
  return ThreadPoolGroup::defaultInstance()->createNamed(
      thread_name, num_threads);
}

std::shared_ptr<ThreadPool> ThreadPool::getNamed(
    const std::string& thread_name) {
  return ThreadPoolGroup::defaultInstance()->getNamed(thread_name);
}

void ThreadPool::removeNamed(
    const std::string& thread_name, bool wait_all_works_done) {
  return ThreadPoolGroup::defaultInstance()->removeNamed(
      thread_name, wait_all_works_done);
}

std::shared_ptr<ThreadPoolGroup> ThreadPoolGroup::defaultInstance() {
  static std::mutex mtx;
  static std::shared_ptr<ThreadPoolGroup> default_thread_pool_group = nullptr;
  if (!default_thread_pool_group) {
    std::lock_guard<std::mutex> locker(mtx);
    if (!default_thread_pool_group) {
      default_thread_pool_group = std::make_shared<ThreadPoolGroup>();
    }
  }
  ASSERT(default_thread_pool_group);
  return default_thread_pool_group;
}

ThreadPoolGroup::ThreadPoolGroup() : task_manager_(new TaskManager()) {}

ThreadPoolGroup::~ThreadPoolGroup() {
  clear(true);
  task_manager_.reset();
}

std::shared_ptr<ThreadPool> ThreadPoolGroup::create(int num_threads) {
  return createGeneric("", num_threads);
}

std::shared_ptr<ThreadPool> ThreadPoolGroup::createNamed(
    const std::string& thread_name, int num_threads) {
  return createGeneric(thread_name, num_threads);
}

std::shared_ptr<ThreadPool> ThreadPoolGroup::getNamed(
    const std::string& thread_name) {
  {
    SharedLock locker(mutex_);
    if (named_thread_pools_.count(thread_name) == 0) {
      // LOGW(
      //     "ThreadPoolGroup::get(): unknown thread pool '%s'!",
      //     thread_name.c_str());
      // return nullptr;
    } else {
      return named_thread_pools_.at(thread_name);
    }
  }

  return createNamed(thread_name, 1);
}

void ThreadPoolGroup::removeNamed(
    const std::string& thread_name, bool wait_all_works_done) {
  std::shared_ptr<ThreadPool> pool;
  {
    SharedLock locker(mutex_);
    if (named_thread_pools_.count(thread_name) == 0) {
      return;
    } else {
      pool = named_thread_pools_[thread_name];
    }
  }

  pool->stop(wait_all_works_done);  // this might be time-consuming.

  UniqueLock locker(mutex_);
  if (named_thread_pools_.count(thread_name)) {
    named_thread_pools_.erase(thread_name);
  }
}

void ThreadPoolGroup::removeAnonymous(
    ThreadPoolID thread_pool_id, bool wait_all_works_done) {
  std::shared_ptr<ThreadPool> pool;
  {
    SharedLock locker(mutex_);
    if (anonymous_thread_pools_.count(thread_pool_id) == 0) {
      return;
    } else {
      pool = anonymous_thread_pools_.at(thread_pool_id).lock();
    }
  }

  if (pool) {
    pool->stop(wait_all_works_done);  // this might be time-consuming.
  }

  UniqueLock locker(mutex_);
  if (anonymous_thread_pools_.count(thread_pool_id)) {
    anonymous_thread_pools_.erase(thread_pool_id);
  }
}

void ThreadPoolGroup::removeExpired() {
  UniqueLock locker(mutex_);
  std::vector<ThreadPoolID> to_remove;
  for (auto& it : anonymous_thread_pools_) {
    if (it.second.expired()) {
      to_remove.push_back(it.first);
    }
  }
  for (auto thread_pool_id : to_remove) {
    anonymous_thread_pools_.erase(thread_pool_id);
  }
}

std::shared_ptr<ThreadPool> ThreadPoolGroup::createGeneric(
    const std::string& thread_name, int num_threads) {
  removeExpired();

  std::shared_ptr<TaskManager> tmp_task_manager = nullptr;
  {
    SharedLock locker(mutex_);
    if (clearing_) {
      LOGW(
          "ThreadPoolGroup::createGeneric(): "
          "Cant't create new thread-pool when clearing!");
      return nullptr;
    }
    if (!thread_name.empty() && named_thread_pools_.count(thread_name)) {
      LOGW(
          "ThreadPoolGroup::createGeneric(): attempting to re-create the named "
          "thread pool '%s' (ignored)",
          thread_name.c_str());
      return named_thread_pools_.at(thread_name);
    }
    tmp_task_manager = task_manager_;
  }

  std::shared_ptr<ThreadPool> thread_pool(
      new ThreadPool(num_threads, thread_name, tmp_task_manager));

  {
    UniqueLock locker(mutex_);

    // recheck the conditions
    if (clearing_) {
      LOGW(
          "ThreadPoolGroup::createGeneric(): "
          "Cant't create new thread-pool when clearing!");
      return nullptr;
    }
    if (!thread_name.empty() && named_thread_pools_.count(thread_name)) {
      LOGW(
          "ThreadPoolGroup::createGeneric(): attempting to re-create the named "
          "thread pool '%s' (ignored)",
          thread_name.c_str());
      return named_thread_pools_.at(thread_name);
    }

    // register the thread pool
    if (thread_name.empty()) {
      anonymous_thread_pools_[thread_pool->id()] = thread_pool;
    } else {
      named_thread_pools_[thread_name] = thread_pool;
    }
    task_manager_->bindThreadPool(thread_pool);
    return thread_pool;
  }
}

void ThreadPoolGroup::clear(bool wait_all_works_done) {
  std::vector<std::string> all_named;
  std::vector<ThreadPoolID> all_anonymous;
  {
    UniqueLock locker(mutex_);
    clearing_ = true;
    for (const auto& pair : named_thread_pools_) {
      all_named.push_back(pair.first);
    }
    for (const auto& pair : anonymous_thread_pools_) {
      all_anonymous.push_back(pair.first);
    }
  }

  std::vector<std::shared_ptr<std::thread>> join_threads;

  for (std::string thread_name : all_named) {
    auto join_thread = std::make_shared<std::thread>(
        [this, thread_name, wait_all_works_done]() {
          removeNamed(thread_name, wait_all_works_done);
        });
    join_threads.push_back(join_thread);
  }
  for (ThreadPoolID thread_pool_id : all_anonymous) {
    auto join_thread = std::make_shared<std::thread>(
        [this, thread_pool_id, wait_all_works_done]() {
          removeAnonymous(thread_pool_id, wait_all_works_done);
        });
    join_threads.push_back(join_thread);
  }

  for (auto join_thread : join_threads) {
    join_thread->join();
  }

  {
    SharedLock locker(mutex_);
    ASSERT(named_thread_pools_.empty());
  }
}

}  // namespace sk4slam
