#pragma once

#define MESSAGE_FLOW_TOPIC(NAME, MESSAGE_TYPE)          \
  namespace sk4slam_msgflow_topics {                    \
  struct NAME {                                         \
    static constexpr const char* kMessageTopic = #NAME; \
    typedef MESSAGE_TYPE message_type;                  \
  };                                                    \
  }
