#pragma once

#include <memory>
#include <mutex>

#include "sk4slam_msgflow/maplab-message-flow/subscriber-list.h"

namespace sk4slam_msgflow {

template <typename MessageType>
class Publisher {
 public:
  explicit Publisher(
      const std::weak_ptr<SubscriberList<MessageType>>& topic_subscribers)
      : topic_subscribers_(topic_subscribers) {}

  // Publish the message to all subscribers.
  void publish(const MessageType& message) {
    SubscriberListPtr<MessageType> topic_subscribers =
        topic_subscribers_.lock();
    if (topic_subscribers) {
      topic_subscribers->publishToAllSubscribersBlocking(message);
    }
  }

 private:
  std::weak_ptr<SubscriberList<MessageType>> topic_subscribers_;
};
}  // namespace sk4slam_msgflow
