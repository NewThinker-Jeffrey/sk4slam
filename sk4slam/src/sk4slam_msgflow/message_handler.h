#pragma once

#include "sk4slam_basic/unique_id.h"
#include "sk4slam_cpp/hashable_pair.h"
#include "sk4slam_cpp/work_queue.h"
#include "sk4slam_msgflow/message-topic-registration.h"
#include "sk4slam_msgflow/msgflow.h"

namespace sk4slam {

class MessageHandler {
 public:
  MessageHandler() {}

  void attachToMessageFlow(
      MessageFlow* msgflow, const std::string& kMessageFlowSubscriberName,
      const std::string& kSubscriberThreadName);

  virtual ~MessageHandler();

  virtual void registerPublishers(MessageFlow* msgflow);

  virtual void registerSubscribers(
      MessageFlow* msgflow, const std::string& kMessageFlowSubscriberName) {}

#define FORWARD_MESSAGE_HELPER(ForwardFunction, ...)                 \
  for (auto& base_handler : other_handlers) {                        \
    if (auto handler = dynamic_cast<decltype(this)>(base_handler)) { \
      if (async) {                                                   \
        enqueueTask([handler, __VA_ARGS__]() {                       \
          handler->ForwardFunction(__VA_ARGS__);                     \
        });                                                          \
      } else {                                                       \
        handler->ForwardFunction(__VA_ARGS__);                       \
      }                                                              \
    }                                                                \
  }

 protected:
  virtual void processExampleMassage(int msg_param1, int msg_param2) {}

  void forwardExampleMessage(
      std::vector<MessageHandler*>& other_handlers, int msg_param1,
      int msg_param2, bool async = false) {
    FORWARD_MESSAGE_HELPER(processExampleMassage, msg_param1, msg_param2)
  }

 protected:
  void logToMessageFlow(const std::string& msg);
  void enqueueTask(std::function<void()>&& task);
  void stopTaskQueue();

  /// @name Message drop policies
  /// @{
 protected:
  using MessageSourceKey = hashable_pair<std::string, UniqueId>;
  struct DropPolicyInterface {
    using MessageQueue = Deque<std::shared_ptr<const void>>;
    virtual ~DropPolicyInterface() = default;
    void pushMessage(const std::shared_ptr<const void>& msg) EXCLUDES(mutex_) {
      UniqueLock lock(mutex_);
      unprocessed_msgs_.push_back(msg);
    }
    void popMessage(const std::shared_ptr<const void>& msg) EXCLUDES(mutex_) {
      UniqueLock lock(mutex_);
      ASSERT(msg == unprocessed_msgs_.front());
      unprocessed_msgs_.pop_front();
    }
    bool shouldDrop(const std::shared_ptr<const void>& msg) const
        EXCLUDES(mutex_) {
      SharedLock lock(mutex_);
      return shouldDrop(msg, unprocessed_msgs_);
    }

   protected:
    virtual bool shouldDrop(
        const std::shared_ptr<const void>& msg,
        const MessageQueue& unprocessed_msgs) const REQUIRES_SHARED(mutex_) = 0;

   private:
    mutable Mutex mutex_;
    MessageQueue unprocessed_msgs_;
  };

  struct KeepLatestN : public DropPolicyInterface {
    KeepLatestN() {}
    explicit KeepLatestN(int keep_latest_n) : keep_latest_n_(keep_latest_n) {}

   protected:
    bool shouldDrop(
        const std::shared_ptr<const void>& msg,
        const MessageQueue& unprocessed_msgs) const override {
      ASSERT(msg == unprocessed_msgs.front());
      return unprocessed_msgs.size() > keep_latest_n_;
    }

   protected:
    int keep_latest_n_ =
        1;  ///< Drop the front message if the number of remaining unprocessed
            ///< messages is greater than this value. 1 means only the latest
            ///< message will be processed.
  };

  /// @brief  Register a drop policy for a given message source.
  /// @warning Drop policies should be registered before the message handler
  ///          been attached to a message-flow. Calling this function after
  ///          attaching can result in threadsafety issues.
  void registerDropPolicy(
      const MessageSourceKey& key,
      std::shared_ptr<DropPolicyInterface> drop_policy) {
    drop_policies_[key] = std::move(drop_policy);
  }

  std::unordered_map<MessageSourceKey, std::shared_ptr<DropPolicyInterface>>
      drop_policies_;  ///< Drop policies for each message source.
  /// @}

  /// Helper macros for using drop policies.
#define REGISTER_DROP_POLICY(name, uid, drop_policy) \
  {                                                  \
    MessageSourceKey key(#name, uid);                \
    registerDropPolicy(key, std::move(drop_policy)); \
  }

#define PUSH_MSG_TO_DROP_POLICY(name, uid, msg) \
  {                                             \
    MessageSourceKey key(#name, uid);           \
    auto it = drop_policies_.find(key);         \
    if (it != drop_policies_.end()) {           \
      it->second->pushMessage(msg);             \
    }                                           \
  }

#define CHECK_AND_POP_MSG_FROM_DROP_POLICY(name, uid, msg) \
  {                                                        \
    MessageSourceKey key(#name, uid);                      \
    auto it = drop_policies_.find(key);                    \
    bool should_drop = false;                              \
    if (it != drop_policies_.end()) {                      \
      should_drop = it->second->shouldDrop(msg);           \
      it->second->popMessage(msg);                         \
    }                                                      \
    if (should_drop) {                                     \
      return;                                              \
    }                                                      \
  }

 private:
  std::unique_ptr<TaskQueue> task_queue_;
  std::function<void(std::shared_ptr<const std::string>)> pub_handler_log_;
};

}  // namespace sk4slam

MESSAGE_FLOW_TOPIC(
    MESSAGE_HANDLER_LOG_TOPIC, std::shared_ptr<const std::string>);
