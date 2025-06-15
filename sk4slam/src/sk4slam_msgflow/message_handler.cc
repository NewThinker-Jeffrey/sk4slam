#include "sk4slam_msgflow/message_handler.h"

namespace sk4slam {

MessageHandler::~MessageHandler() {
  stopTaskQueue();
}

void MessageHandler::stopTaskQueue() {
  if (task_queue_) {
    task_queue_->stop();
  }
  task_queue_.reset();
}

void MessageHandler::enqueueTask(std::function<void()>&& task) {
  if (task_queue_) {
    ASSERT(task_queue_->enqueue(std::move(task)));
  }
}

void MessageHandler::attachToMessageFlow(
    MessageFlow* msgflow, const std::string& kMessageFlowSubscriberName,
    const std::string& kSubscriberThreadName) {
  ASSERT(!task_queue_);
  task_queue_.reset(new TaskQueue(kSubscriberThreadName));
  registerSubscribers(msgflow, kMessageFlowSubscriberName);
  registerPublishers(msgflow);
}

void MessageHandler::logToMessageFlow(const std::string& log_str) {
  if (pub_handler_log_) {
    auto msg = std::make_shared<std::string>(log_str);
    pub_handler_log_(msg);
  }
}

void MessageHandler::registerPublishers(MessageFlow* msgflow) {
  if (!pub_handler_log_) {  // Check if already registered
    pub_handler_log_ = msgflow->registerPublisher<
        sk4slam_msgflow_topics::MESSAGE_HANDLER_LOG_TOPIC>();
  }
}

}  // namespace sk4slam
