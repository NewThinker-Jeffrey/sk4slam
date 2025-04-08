#pragma once

#include <atomic>

#include "sk4slam_msgflow/maplab-message-flow/message-delivery-queue.h"

namespace sk4slam_msgflow {
// The message dispatcher works on a list of queues and invokes the callbacks
// according to its policy.
class MessageDispatcher {
 public:
  MessageDispatcher() {}
  virtual ~MessageDispatcher() {}

  // Signals the dispatcher that a new message is available for delivery. Must
  // be called for each incoming message.
  virtual void newMessageInQueue(const MessageDeliveryQueueBasePtr& queue) = 0;
  virtual void shutdown() = 0;
  virtual void waitUntilIdle() const = 0;
};
typedef std::shared_ptr<MessageDispatcher> MessageDispatcherPtr;
}  // namespace sk4slam_msgflow
