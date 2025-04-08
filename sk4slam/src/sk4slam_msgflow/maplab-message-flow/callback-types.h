#pragma once

#include <functional>

namespace sk4slam_msgflow {
template <typename MessageTopicDefinition>
using MessageCallback =
    std::function<void(const typename MessageTopicDefinition::message_type&)>;
template <typename MessageTopicDefinition>
using PublisherFunction = MessageCallback<MessageTopicDefinition>;
template <typename MessageTopicDefinition>
using SubscriberCallback = MessageCallback<MessageTopicDefinition>;
}  // namespace sk4slam_msgflow
