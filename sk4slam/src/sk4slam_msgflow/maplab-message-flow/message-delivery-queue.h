#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_cpp/deque.h"

namespace sk4slam_msgflow {
// UNIQUE_ID_DEFINE_ID(MessageDeliveryQueueId);

using MessageDeliveryQueueId = size_t;
void generateMessageDeliveryQueueId(MessageDeliveryQueueId* queue_id);

struct DeliveryOptions {
  DeliveryOptions() : exclusivity_group_id(-1) {}
  // Ensures the exclusive execution of deliveries across all subscribers with
  // the same group id. With a FIFO message dispatcher, this will expand the
  // delivery order guarantees across multiple subscribers. I.e. not only all
  // messages on individual subscribers will be delivered in publishing order,
  // but also all messages to subscribers with the same exclusivity id will be
  // delivered in the publishing order.
  // A negative value means no exclusivity is enforced.
  int exclusivity_group_id;
};

class MessageDeliveryQueueBase {
 public:
  virtual ~MessageDeliveryQueueBase() {}
  virtual void deliverOldestMessage() = 0;
  virtual std::string getTopicName() const = 0;
  virtual const DeliveryOptions& getDeliveryOptions() const = 0;
  virtual size_t size() const = 0;
};
typedef std::shared_ptr<MessageDeliveryQueueBase> MessageDeliveryQueueBasePtr;

// Maintains a list of messages scheduled for delivery to a specific subscriber.
template <typename MessageTopicDefinition>
class MessageDeliveryQueue : public MessageDeliveryQueueBase {
 public:
  typedef typename MessageTopicDefinition::message_type MessageType;
  typedef std::function<void(const MessageType&)> SubscriberCallback;

  MessageDeliveryQueue(
      const SubscriberCallback& subscriber_callback,
      const DeliveryOptions& delivery_options)
      : delivery_options_(delivery_options),
        subscriber_callback_(subscriber_callback) {
    ASSERT(subscriber_callback);
  }
  virtual ~MessageDeliveryQueue() {}

  void queueMessageForDelivery(const MessageType& message) {
    std::lock_guard<std::mutex> lock(m_message_queue_);
    message_queue_.emplace_back(message);
  }

  void deliverOldestMessage() final {
    MessageType message;
    {
      std::lock_guard<std::mutex> lock(m_message_queue_);
      ASSERT(!message_queue_.empty());
      message = message_queue_.front();
      message_queue_.pop_front();
      message_queue_.trim_to_optimal();  // message_queue_.shrink_to_fit();
    }

    // Run the subscriber callback; the lock ensures only one callback can be
    // run simultaneously.
    std::lock_guard<std::mutex> lock_subscriber(m_subscriber_execution_);
    subscriber_callback_(message);
  }

  std::string getTopicName() const final {
    return MessageTopicDefinition::kMessageTopic;
  }

  const DeliveryOptions& getDeliveryOptions() const final {
    return delivery_options_;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m_message_queue_);
    return message_queue_.empty();
  }

  virtual size_t size() const {
    std::lock_guard<std::mutex> lock(m_message_queue_);
    return message_queue_.size();
  }

 private:
  const DeliveryOptions delivery_options_;

  // Protects the callback to prevent concurrent calls to the subscriber
  // callback.
  std::mutex m_subscriber_execution_;
  const SubscriberCallback subscriber_callback_;

  mutable std::mutex m_message_queue_;
  sk4slam::Deque<MessageType> message_queue_;
};
}  // namespace sk4slam_msgflow
// UNIQUE_ID_DEFINE_ID_HASH(sk4slam_msgflow::MessageDeliveryQueueId);
