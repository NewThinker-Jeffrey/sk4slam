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
  finalizeMessageIngressSubscriptions(msgflow, kMessageFlowSubscriberName);
  registerPublishers(msgflow);
}

void MessageHandler::addMessageIngressRoute(
    const MessageIngressDescriptor& descriptor,
    std::function<void(const ErasedMessage&)> callback) {
  if (message_ingress_subscriptions_finalized_) {
    throw std::logic_error(
        "Message ingress route registered after handler attachment");
  }
  if (descriptor.topic_name.empty() || !descriptor.message_type_id ||
      !descriptor.subscribe || !callback) {
    throw std::invalid_argument("Invalid message ingress route");
  }

  auto& route = message_ingress_routes_[descriptor.topic_name];
  if (!route) {
    route = std::make_shared<MessageIngressRoute>();
    route->subscription_descriptor = &descriptor;
    route->message_type_id = descriptor.message_type_id;
    route->subscribe = descriptor.subscribe;
  } else if (
      route->message_type_id != descriptor.message_type_id ||
      route->subscribe != descriptor.subscribe) {
    throw std::logic_error(
        "Conflicting message ingress descriptors for topic: " +
        descriptor.topic_name);
  }
  route->callbacks.emplace_back(std::move(callback));
}

void MessageHandler::finalizeMessageIngressSubscriptions(
    MessageFlow* flow, const std::string& subscriber_name) {
  ASSERT(flow);
  ASSERT(!subscriber_name.empty());
  ASSERT(!message_ingress_subscriptions_finalized_);
  message_ingress_subscriptions_finalized_ = true;

  for (const auto& [topic_name, route] : message_ingress_routes_) {
    ASSERT(route);
    ASSERT(route->subscription_descriptor);
    ASSERT(topic_name == route->subscription_descriptor->topic_name);
    ASSERT(route->subscribe);
    ASSERT(!route->callbacks.empty());
    std::shared_ptr<MessageIngressRoute> route_ptr = route;
    const MessageIngressDescriptor* subscription_descriptor =
        route_ptr->subscription_descriptor;
    const MessageIngressDescriptor::SubscribeFunction subscribe =
        route_ptr->subscribe;
    subscribe(
        *subscription_descriptor, flow, subscriber_name,
        [route_ptr = std::move(route_ptr)](
            const MessageIngressDescriptor&, const ErasedMessage& message) {
          for (const auto& callback : route_ptr->callbacks) {
            callback(message);
          }
        });
  }
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
