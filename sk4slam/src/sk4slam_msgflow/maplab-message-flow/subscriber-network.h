#pragma once

#include <string>
#include <unordered_map>
#include <utility>

#include "sk4slam_basic/logging.h"
#include "sk4slam_msgflow/maplab-message-flow/callback-types.h"
#include "sk4slam_msgflow/maplab-message-flow/subscriber-list.h"

namespace sk4slam_msgflow {
// Stores the subscribing functions for each topic. Publisher are not stored.
class SubscriberNetwork {
 public:
  template <typename MessageTopicDefinition>
  void addSubscriber(
      const SubscriberCallback<MessageTopicDefinition>& callback) {
    ASSERT(callback);
    typedef typename MessageTopicDefinition::message_type MessageType;
    SubscriberListPtr<MessageType> subscriber_list =
        getSubscriberListAndAllocateIfNecessary<MessageTopicDefinition>();
    ASSERT(subscriber_list);
    subscriber_list->addSubscriber(callback);
  }

  template <typename MessageTopicDefinition>
  SubscriberListPtr<typename MessageTopicDefinition::message_type>
  getSubscriberListAndAllocateIfNecessary() {
    typedef typename MessageTopicDefinition::message_type MessageType;
    const std::string& topic_name = MessageTopicDefinition::kMessageTopic;

    ASSERT(!topic_name.empty());
    SubscriberListBasePtr& subscriber_list = subscriber_lists_[topic_name];
    if (subscriber_list == nullptr) {
      subscriber_list.reset(new SubscriberList<MessageType>());
    }
    return std::static_pointer_cast<SubscriberList<MessageType> >(
        subscriber_list);
  }

  void unregisterAllSubscribers() {
    for (std::pair<const std::string, SubscriberListBasePtr>& list :
         subscriber_lists_) {
      list.second->clear();
    }
  }

 private:
  // Map from message topic to the list of its subscribers.
  std::mutex m_subscriber_lists_;
  std::unordered_map<std::string, SubscriberListBasePtr> subscriber_lists_;
};
}  // namespace sk4slam_msgflow
