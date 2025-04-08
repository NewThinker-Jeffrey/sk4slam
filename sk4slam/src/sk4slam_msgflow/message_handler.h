#pragma once

#include "sk4slam_cpp/work_queue.h"
#include "sk4slam_msgflow/msgflow.h"

namespace sk4slam {

class MessageHandler {
 public:
  MessageHandler() {}

  void attachToMessageFlow(
      MessageFlow* msgflow, const std::string& kMessageFlowSubscriberName,
      const std::string& kSubscriberThreadName);

  virtual ~MessageHandler();

  virtual void registerPublishers(MessageFlow* msgflow) {}

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
  void enqueueTask(std::function<void()>&& task);
  void stopTaskQueue();

 private:
  std::unique_ptr<TaskQueue> task_queue_;
};

}  // namespace sk4slam
