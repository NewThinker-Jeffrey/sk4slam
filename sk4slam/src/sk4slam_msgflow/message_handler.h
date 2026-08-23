#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/template_helper.h"
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

 public:
  using ErasedMessage = std::shared_ptr<const void>;

  /// Opaque process-local message-type identity obtained from classname<T>().
  /// It is compared by pointer identity and must not be reconstructed from its
  /// string content.
  using MessageTypeId = const char*;

  template <typename Message>
  static MessageTypeId messageTypeId() {
    return classname<Message>();
  }

  /// Returns a non-owning typed view without constructing another shared_ptr.
  /// The returned reference must not outlive erased_message.
  template <typename Message>
  static const Message& castErasedMessage(const ErasedMessage& erased_message) {
    ASSERT(erased_message);
    return *static_cast<const Message*>(erased_message.get());
  }

  /// Type-safe facade over a type-erased callback table.
  ///
  /// Example:
  /// @code
  /// registry.registerHandler<Message>(default_handler);
  /// registry.registerHandler<Message>("topic_a", topic_a_handler);
  /// registry.dispatch<Message>("topic_a", message);  // topic_a_handler
  /// registry.dispatch<Message>("topic_b", message);  // default_handler
  /// registry.dispatch<Message>(message);             // default_handler
  /// @endcode
  class TypedHandlerRegistry {
   public:
    using ErasedHandler = std::function<void(const ErasedMessage&)>;
    using HandlerKey = hashable_pair<MessageTypeId, std::string>;

    /// Registers the default handler for Message, independent of topic.
    /// This is equivalent to registerHandler<Message>("", handler). A
    /// topic-specific lookup falls back to this handler when no exact handler
    /// is registered for that topic.
    template <typename Message, typename Handler>
    void registerHandler(Handler&& handler) {
      registerHandler<Message>("", std::forward<Handler>(handler));
    }

    /// Registers a handler for the exact (Message, topic_name) pair.
    /// A non-empty topic-specific handler takes precedence over the default
    /// handler. Registering the same exact pair again replaces the previously
    /// registered handler, allowing a derived class or later configuration to
    /// override a default callback. A default handler and any number of
    /// distinct topic-specific handlers may coexist.
    template <typename Message, typename Handler>
    void registerHandler(const std::string& topic_name, Handler&& handler) {
      const MessageTypeId message_type_id = messageTypeId<Message>();
      using RegisteredHandler = std::decay_t<Handler>;
      const char* registered_handler_type = classname<RegisteredHandler>();
      ErasedHandler erased_handler =
          [handler = std::forward<Handler>(handler)](
              const ErasedMessage& erased_message) mutable {
            handler(std::static_pointer_cast<const Message>(erased_message));
          };

      const HandlerKey key(message_type_id, topic_name);
      const auto existing = handlers_.find(key);
      if (existing != handlers_.end()) {
        LOGD(
            "TypedHandlerRegistry: overriding handler for message %s, topic "
            "'%s': %s -> %s",
            message_type_id, topic_name.c_str(),
            existing->second.registered_handler_type, registered_handler_type);
      }
      handlers_.insert_or_assign(
          key,
          HandlerEntry{std::move(erased_handler), registered_handler_type});
    }

    /// Checks only the default handler registered without a topic.
    template <typename Message>
    bool hasHandler() const {
      return hasHandler(messageTypeId<Message>());
    }

    /// Returns true when an exact topic-specific handler exists or a default
    /// handler for Message is available as fallback.
    template <typename Message>
    bool hasHandler(const std::string& topic_name) const {
      return hasHandler(messageTypeId<Message>(), topic_name);
    }

    /// Runtime-type counterpart of hasHandler<Message>().
    bool hasHandler(MessageTypeId message_type_id) const {
      return findHandler(message_type_id, "") != nullptr;
    }

    /// Runtime-type counterpart of hasHandler<Message>(topic_name).
    bool hasHandler(
        MessageTypeId message_type_id, const std::string& topic_name) const {
      return findHandler(message_type_id, topic_name) != nullptr;
    }

    /// Returns true only for an exact (message type, topic name)
    /// registration. Unlike hasHandler(message_type_id, topic_name), this
    /// query does not fall back to the message type's default handler.
    bool hasExactHandler(
        MessageTypeId message_type_id, const std::string& topic_name) const {
      return handlers_.find(HandlerKey(message_type_id, topic_name)) !=
             handlers_.end();
    }

    /// Dispatches only to the default handler registered without a topic.
    /// Topic-specific handlers are not considered because no topic identity is
    /// available on this call path.
    template <typename Message>
    bool dispatch(const std::shared_ptr<const Message>& message) const {
      return dispatch(
          messageTypeId<Message>(),
          std::static_pointer_cast<const void>(message));
    }

    /// Dispatches to the exact topic-specific handler when present, otherwise
    /// to the default handler registered without a topic. At most one handler
    /// is invoked.
    template <typename Message>
    bool dispatch(
        const std::string& topic_name,
        const std::shared_ptr<const Message>& message) const {
      return dispatch(
          messageTypeId<Message>(), topic_name,
          std::static_pointer_cast<const void>(message));
    }

    /// Runtime-type dispatch without a topic. Only the default handler is
    /// eligible, matching dispatch<Message>(message).
    bool dispatch(
        MessageTypeId message_type_id, const ErasedMessage& message) const {
      return dispatch(message_type_id, "", message);
    }

    /// Runtime-type counterpart of dispatch<Message>(topic_name, message).
    /// An empty topic_name only checks the default handler; it cannot select a
    /// topic-specific handler.
    bool dispatch(
        MessageTypeId message_type_id, const std::string& topic_name,
        const ErasedMessage& message) const {
      const ErasedHandler* handler = findHandler(message_type_id, topic_name);
      if (!handler) {
        return false;
      }
      (*handler)(message);
      return true;
    }

    /// Returns the number of exact (message type, topic name) registrations.
    /// The default handler, when present, counts as one registration.
    size_t size() const {
      return handlers_.size();
    }

   private:
    const ErasedHandler* findHandler(
        MessageTypeId message_type_id, const std::string& topic_name) const {
      auto it = handlers_.find(HandlerKey(message_type_id, topic_name));
      if (it != handlers_.end()) {
        return &it->second.handler;
      }
      if (!topic_name.empty()) {
        it = handlers_.find(HandlerKey(message_type_id, ""));
        if (it != handlers_.end()) {
          return &it->second.handler;
        }
      }
      return nullptr;
    }

    struct HandlerEntry {
      ErasedHandler handler;
      const char* registered_handler_type{nullptr};
    };

    std::unordered_map<HandlerKey, HandlerEntry> handlers_;
  };

  /// @name Message drop policies
  /// @{
 public:
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

  DEFINE_HAS_MEMBER_FUNCTION(getDropPolicyUID)

  /// Default customization point for extracting the source discriminator used
  /// by a message's drop policy. DECLARE_DROP_POLICY_UID normally supplies the
  /// detected member function. A message without that declaration returns a
  /// null UID and therefore does not participate in drop-policy processing.
  /// The extraction rule is a property of Message and must not vary by topic.
  template <typename Message>
  struct MessageDropTraits {
    static constexpr bool kHasDropPolicyUID =
        HasMemberFunction_getDropPolicyUID<const Message>;

    static UniqueId dropPolicyUID(const Message& message) {
      if constexpr (kHasDropPolicyUID) {
        return message.getDropPolicyUID();
      } else {
        return UniqueId::null;
      }
    }
  };

  /// Type-erased description shared by registry-driven subscriptions.
  struct MessageIngressDescriptor {
    using ErasedMessage = MessageHandler::ErasedMessage;
    using Sink = std::function<void(
        const MessageIngressDescriptor&, const ErasedMessage&)>;
    using SubscribeFunction = void (*)(
        const MessageIngressDescriptor&, MessageFlow*, const std::string&,
        const Sink&);
    using DropPolicyUidFunction = UniqueId (*)(const ErasedMessage&);

    std::string topic_name;
    MessageTypeId message_type_id{nullptr};
    SubscribeFunction subscribe{nullptr};
    /// Optional. A missing extractor or an extractor returning UniqueId::null
    /// disables drop-policy handling without disabling message delivery.
    DropPolicyUidFunction drop_policy_uid{nullptr};
  };

  /// Process-wide ingress registry isolated by Category. Descriptor may
  /// extend MessageIngressDescriptor with category-specific extractors.
  /// Different Category types own independent registries even when they use
  /// the same topic names or message types. Registration is expected to finish
  /// before handlers attach to MessageFlow.
  template <typename Category, typename Descriptor = MessageIngressDescriptor>
  class MessageIngressRegistry {
   public:
    using DescriptorType = Descriptor;

    static MessageIngressRegistry& instance() {
      static MessageIngressRegistry registry;
      return registry;
    }

    /// Registers one descriptor per topic name within this Category. Reusing a
    /// message type on several topics is allowed; handler selection can still
    /// distinguish them by topic name.
    void registerDescriptor(Descriptor descriptor) {
      if (descriptor.topic_name.empty()) {
        throw std::invalid_argument("Message ingress topic name is empty");
      }
      if (!descriptor.subscribe) {
        throw std::invalid_argument("Message ingress subscriber is empty");
      }
      if (!descriptor.message_type_id) {
        throw std::invalid_argument("Message ingress type id is empty");
      }
      const std::string topic_name = descriptor.topic_name;
      const auto [it, inserted] =
          descriptors_.emplace(topic_name, std::move(descriptor));
      if (!inserted) {
        throw std::logic_error(
            "Duplicate message ingress topic name: " + topic_name);
      }
    }

    const Descriptor* find(const std::string& topic_name) const {
      const auto it = descriptors_.find(topic_name);
      return it == descriptors_.end() ? nullptr : &it->second;
    }

    std::vector<const Descriptor*> descriptors() const {
      std::vector<const Descriptor*> result;
      result.reserve(descriptors_.size());
      for (const auto& [topic_name, descriptor] : descriptors_) {
        result.push_back(&descriptor);
      }
      return result;
    }

   private:
    std::map<std::string, Descriptor> descriptors_;
  };

  template <typename Registry>
  class MessageIngressRegistrar {
   public:
    using Descriptor = typename Registry::DescriptorType;

    explicit MessageIngressRegistrar(Descriptor descriptor) {
      Registry::instance().registerDescriptor(std::move(descriptor));
    }
  };

  template <typename Topic>
  static void subscribeMessageTopicToErasedSink(
      const MessageIngressDescriptor& descriptor, MessageFlow* flow,
      const std::string& subscriber_name,
      const MessageIngressDescriptor::Sink& sink) {
    using MessagePtr = typename Topic::message_type;
    flow->registerSubscriber<Topic>(
        subscriber_name, [&descriptor, sink](const MessagePtr& message) {
          sink(descriptor, std::static_pointer_cast<const void>(message));
        });
  }

  /// Creates an ingress descriptor whose drop-policy UID is extracted through
  /// Traits::dropPolicyUID(const Message&). Topic supplies the strongly typed
  /// subscription; topic_name explicitly identifies that subscription in the
  /// ingress registry and drop-policy key.
  template <
      typename Message, typename Topic,
      typename Traits = MessageDropTraits<Message>>
  static MessageIngressDescriptor makeMessageIngressDescriptor(
      const std::string& topic_name) {
    return MessageIngressDescriptor{
        topic_name, messageTypeId<Message>(),
        &subscribeMessageTopicToErasedSink<Topic>,
        [](const ErasedMessage& erased_message) {
          const auto& message = castErasedMessage<Message>(erased_message);
          return Traits::dropPolicyUID(message);
        }};
  }

  /// Creates an ingress descriptor that never participates in drop-policy
  /// processing. Messages are still subscribed, queued, and dispatched.
  template <typename Message, typename Topic>
  static MessageIngressDescriptor makeMessageIngressDescriptorWithoutDropPolicy(
      const std::string& topic_name) {
    return MessageIngressDescriptor{
        topic_name, messageTypeId<Message>(),
        &subscribeMessageTopicToErasedSink<Topic>, nullptr};
  }

  /// Adds every descriptor currently registered in Registry to this handler's
  /// pending ingress routes. Routes from different categories that name the
  /// same MessageFlow topic are merged into one physical subscription; their
  /// callbacks run synchronously in route-registration order.
  ///
  /// The physical subscriptions are created by attachToMessageFlow() after
  /// registerSubscribers() returns. Registries populated after attachment are
  /// not subscribed retroactively.
  template <typename Registry, typename Callback>
  void registerMessageIngressRoutes(Callback&& callback) {
    for (const auto* descriptor : Registry::instance().descriptors()) {
      ASSERT(descriptor);
      ASSERT(descriptor->subscribe);
      addMessageIngressRoute(
          *descriptor,
          [descriptor, callback](const ErasedMessage& message) mutable {
            callback(*descriptor, message);
          });
    }
  }

 protected:
  using MessageSourceKey = hashable_pair<std::string, UniqueId>;

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
// Exposes an existing UniqueId expression as the source discriminator used by
// a message's drop policy. SourceExpression must return either a persistent
// UniqueId lvalue or const UniqueId&; it must not produce a temporary value.
#define DECLARE_DROP_POLICY_UID(SourceExpression)     \
  const sk4slam::UniqueId& getDropPolicyUID() const { \
    return SourceExpression;                          \
  }

// The named variants accept a runtime string expression. The legacy variants
// below preserve their original behavior by stringizing the name token.
#define REGISTER_NAMED_DROP_POLICY(name, uid, drop_policy)         \
  {                                                                \
    const auto sk4slam_drop_policy_uid_value = (uid);              \
    if (!sk4slam_drop_policy_uid_value.isNull()) {                 \
      MessageSourceKey key((name), sk4slam_drop_policy_uid_value); \
      registerDropPolicy(key, std::move(drop_policy));             \
    }                                                              \
  }

#define PUSH_MSG_TO_NAMED_DROP_POLICY(name, uid, msg)              \
  {                                                                \
    const auto sk4slam_drop_policy_uid_value = (uid);              \
    if (!sk4slam_drop_policy_uid_value.isNull()) {                 \
      MessageSourceKey key((name), sk4slam_drop_policy_uid_value); \
      auto it = drop_policies_.find(key);                          \
      if (it != drop_policies_.end()) {                            \
        it->second->pushMessage(msg);                              \
      }                                                            \
    }                                                              \
  }

#define CHECK_AND_POP_MSG_FROM_NAMED_DROP_POLICY(name, uid, msg)   \
  {                                                                \
    bool should_drop = false;                                      \
    const auto sk4slam_drop_policy_uid_value = (uid);              \
    if (!sk4slam_drop_policy_uid_value.isNull()) {                 \
      MessageSourceKey key((name), sk4slam_drop_policy_uid_value); \
      auto it = drop_policies_.find(key);                          \
      if (it != drop_policies_.end()) {                            \
        should_drop = it->second->shouldDrop(msg);                 \
        it->second->popMessage(msg);                               \
      }                                                            \
    }                                                              \
    if (should_drop) {                                             \
      return;                                                      \
    }                                                              \
  }

#define REGISTER_DROP_POLICY(name, uid, drop_policy) \
  REGISTER_NAMED_DROP_POLICY(#name, uid, drop_policy)

#define PUSH_MSG_TO_DROP_POLICY(name, uid, msg) \
  PUSH_MSG_TO_NAMED_DROP_POLICY(#name, uid, msg)

#define CHECK_AND_POP_MSG_FROM_DROP_POLICY(name, uid, msg) \
  CHECK_AND_POP_MSG_FROM_NAMED_DROP_POLICY(#name, uid, msg)

 protected:
  /// Invokes callback for every handler that is a TargetHandler.
  ///
  /// With async == false, callbacks execute immediately in the caller's
  /// thread. With async == true, callbacks are queued on *this* forwarding
  /// handler's task queue and execute later in that queue's thread. They are
  /// deliberately not queued on the target handlers: child handlers normally
  /// do not attach to MessageFlow and therefore do not own active task queues.
  /// Consequently, the forwarding handler must have been attached to a
  /// MessageFlow before async forwarding is used.
  template <typename TargetHandler, typename Callback>
  void forwardToHandlers(
      const std::vector<MessageHandler*>& handlers, Callback callback,
      bool async = false) {
    for (MessageHandler* base_handler : handlers) {
      auto* target_handler = dynamic_cast<TargetHandler*>(base_handler);
      if (!target_handler) {
        continue;
      }
      if (async) {
        enqueueTask(
            [target_handler, callback]() mutable { callback(target_handler); });
      } else {
        callback(target_handler);
      }
    }
  }

  /// Legacy helper macro for forwarding messages to other handlers.
  ///
  /// Kept for source compatibility with existing users and may be deprecated
  /// in the future. New code should prefer the type-safe forwardToHandlers()
  /// helper below, which does not depend on macro-specific local variable
  /// names.
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
  void logToMessageFlow(const std::string& msg);
  void enqueueTask(std::function<void()>&& task);
  void stopTaskQueue();

 private:
  struct MessageIngressRoute {
    const MessageIngressDescriptor* subscription_descriptor{nullptr};
    MessageTypeId message_type_id{nullptr};
    MessageIngressDescriptor::SubscribeFunction subscribe{nullptr};
    std::vector<std::function<void(const ErasedMessage&)>> callbacks;
  };

  void addMessageIngressRoute(
      const MessageIngressDescriptor& descriptor,
      std::function<void(const ErasedMessage&)> callback);
  void finalizeMessageIngressSubscriptions(
      MessageFlow* flow, const std::string& subscriber_name);

  std::map<std::string, std::shared_ptr<MessageIngressRoute>>
      message_ingress_routes_;
  bool message_ingress_subscriptions_finalized_{false};
  std::unique_ptr<TaskQueue> task_queue_;
  std::function<void(std::shared_ptr<const std::string>)> pub_handler_log_;
};

}  // namespace sk4slam

MESSAGE_FLOW_TOPIC(
    MESSAGE_HANDLER_LOG_TOPIC, std::shared_ptr<const std::string>);
