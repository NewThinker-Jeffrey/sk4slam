#include "sk4slam_msgflow/maplab-message-flow/message-flow.h"

#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

#include "sk4slam_basic/logging.h"
// #include <maplab-common/accessors.h>

#include "sk4slam_msgflow/maplab-message-flow/message-dispatcher.h"
#include "sk4slam_msgflow/maplab-message-flow/subscriber-network.h"

namespace sk4slam_msgflow {

void generateMessageDeliveryQueueId(MessageDeliveryQueueId* queue_id) {
  static std::atomic<MessageDeliveryQueueId> counter(0);
  *queue_id = counter.fetch_add(1);
}

MessageFlow::MessageFlow(const MessageDispatcherPtr& dispatcher)
    : message_dispatcher_(dispatcher) {
  ASSERT(dispatcher);
}

MessageFlow::~MessageFlow() {
  shutdown();
}

void MessageFlow::shutdown() {
  // First clear all subscribers such that published messages get rejected
  // from now on. Then signal the dispatcher to shutdown. An application could
  // then call to WaitUntilIdle() for a clean shutdown where all remaining
  // tasks can execute until the end.
  std::lock_guard<std::mutex> lock(mutex_network_and_maps_);
  subscriber_network_.unregisterAllSubscribers();
  message_dispatcher_->shutdown();
}

void MessageFlow::waitUntilIdle() const {
  message_dispatcher_->waitUntilIdle();
}

std::string MessageFlow::printDeliveryQueueStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_network_and_maps_);

  std::stringstream output;
  constexpr size_t kNumAlignment = 30u;
  output << "Message delivery queues:" << std::endl;
  output << std::setiosflags(std::ios::left) << std::setw(kNumAlignment)
         << "subscriber-node" << std::setw(kNumAlignment) << "queue-topic"
         << std::setw(kNumAlignment) << "queue-id" << std::setw(kNumAlignment)
         << "num elements" << std::endl;

  for (const MessageDeliveryQueueMap::value_type& value :
       subscriber_message_queues_) {
    const MessageDeliveryQueueId& queue_id = value.first;
    const MessageDeliveryQueueBasePtr& queue = value.second;
    // const std::string& subscriber_node_name =
    //     common::getChecked(subscriber_node_names_, queue_id);
    const std::string& subscriber_node_name =
        subscriber_node_names_.at(queue_id);
    ASSERT(queue);
    output << std::setiosflags(std::ios::left) << std::setw(kNumAlignment)
           << subscriber_node_name << std::setw(kNumAlignment)
           << queue->getTopicName() << std::setw(kNumAlignment) << queue_id
           << std::setw(kNumAlignment) << queue->size() << std::endl;
  }
  return output.str();
}
}  // namespace sk4slam_msgflow
