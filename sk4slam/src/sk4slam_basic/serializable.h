#pragma once

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {
// CRTP (Curiously Recurring Template Pattern) base class for all serializable
// classes. Classes inheriting from this should implement `serialize_impl`
// to define their serialization logic.
template <typename Derived>
struct Serializable;

/// @brief Checks if a type is serializable.
/// @tparam T The type to check.
/// @note A type is considered serializable if it inherits from Serializable.
template <typename T>
inline constexpr bool IsSerializable =
    std::is_base_of_v<Serializable<RawType<T>>, RawType<T>>;

/// @brief Wrapper template for enabling serialization support for user-defined
///        types.
/// @tparam Data The user-defined type.
///
/// @details To enable serialization for a user-defined type without modifying
///          the type itself, specialize this template. The specialization must
///          inherit from both the user-defined type and
///          `Serializable<SerializableWrapper<Data>>`. The specialized class
///          should implement the `serialize_impl` method for the serialization
///          logic.
///
/// Example:
/// @code
/// struct MyData {
///   int a;
///   float b;
/// };
///
/// template <>
/// class SerializableWrapper<MyData> :
///       public MyData,
///       public Serializable<SerializableWrapper<MyData>> {
///   friend class Serializable<SerializableWrapper<MyData>>;
///   void serialize_impl(
///       Archive& ar, const char* name, const unsigned int version) {
///     SERIALIZE(ar, a);
///     SERIALIZE(ar, b);
///   }
/// };
/// @endcode
template <typename Data>
struct SerializableWrapper;

/// @brief Checks if a type can be wrapped as a serializable type.
/// @tparam Data The type to check.
/// @note A type can be wrapped if the `SerializableWrapper<Data>` class is
///       specialized and inherits both from the user-defined type and
///       `Serializable<SerializableWrapper<Data>>`.
template <typename Data>
inline constexpr bool HasSerializableWrapper = std::is_base_of_v<
    Serializable<SerializableWrapper<Data>>, SerializableWrapper<Data>>&&
    std::is_base_of_v<Data, SerializableWrapper<Data>>;

namespace serialization {

/// @brief Serializes a serializable object using a given archive.
/// @tparam Archive The archive type.
/// @tparam Data The type of the object to serialize.
/// @param ar The archive instance for serialization.
/// @param data The object to serialize.
/// @param name (Optional) The name of the serialized data in the archive.
/// @param version (Optional) The serialization version.
///
/// @details This function handles three cases:
/// 1. If the type is directly serializable (inherits from `Serializable`),
///    it calls the archive's `serialize` method for the type.
/// 2. If the type can be wrapped as serializable using `SerializableWrapper`,
///    the wrapper is used for serialization.
/// 3. Otherwise, it assumes the type is a basic type and directly serializes
///    it.
///
/// Example of an archive interface:
/// @code
/// struct Archive {
///   template <class T>
///   void serialize(T& serializable_data, const char* name = nullptr,
///                  const unsigned int version = 0);
/// };
/// @endcode
template <typename Archive, typename Data>
void serialize(
    Archive& ar, Data& data, const char* name, const unsigned int version) {
  if constexpr (IsSerializable<Data>) {
    ar.serialize(data, name, version);
  } else if constexpr (HasSerializableWrapper<Data>) {
    static_assert(
        !IsSerializable<Data>,
        "Error: Attempted to wrap a serializable class in SerializableWrapper. "
        "This is unnecessary because the type is already serializable. Check "
        "the definition of SerializableWrapper<YourType> or remove it if "
        "redundant.");
    SerializableWrapper<Data>& wrapped =
        static_cast<SerializableWrapper<Data>&>(data);
    ar.serialize(wrapped, name, version);
  } else {
    ar.serialize(data, name, version);
  }
}

}  // namespace serialization

/// @brief  A wrapper class for Boost archive types to provide a unified
/// interface for serialization.
/// @tparam BoostArchive The Boost archive type.
template <class BoostArchive>
struct BoostArchiveWrapper : public BoostArchive {
  /// @brief Serializes data using the Boost archive.
  /// @tparam Serializable The serializable object type.
  /// @param data The object to serialize.
  /// @param name The name of the data being serialized.
  /// @param version The version of the serialization (default is 0).
  template <typename Serializable>
  void serialize(
      Serializable& data, const char* name = nullptr,
      const unsigned int version = 0) {
    boost() & data;
  }

 private:
  /// @brief Provides access to the underlying Boost archive.
  BoostArchive& boost() {
    return static_cast<BoostArchive&>(*this);
  }
};

/// @brief  Helper struct to handle serialization for a specific archive type.
/// @tparam Archive The archive type.
template <class Archive>
struct SerializeToArchive {
  DEFINE_HAS_MEMBER_FUNCTION(serialize)
  DEFINE_HAS_MEMBER_OPERATOR(boost_streaming_operator, &)

  template <class _Archive>
  static constexpr bool IsBoostArchive =
      HasMemberOperator_boost_streaming_operator<_Archive, double&> &&
      !HasMemberFunction_serialize<
          _Archive, double&, const char*, const unsigned int>;

  /// @brief Serializes a serializable object to the specified archive.
  /// @tparam Serializable The type of the object to serialize.
  /// @param data The object to serialize.
  /// @param ar The archive to serialize to.
  /// @param version The version of the serialization (default is 0).
  template <typename Serializable>
  static void op(Serializable& data, Archive& ar, const unsigned int version) {
    if constexpr (IsBoostArchive<Archive>) {
      BoostArchiveWrapper<Archive>& wrapper =
          static_cast<BoostArchiveWrapper<Archive>&>(ar);
      data.serialize_impl(wrapper, version);
    } else {
      data.serialize_impl(ar, version);
    }
  }
};

/// @brief CRTP base class for all serializable classes.
/// @tparam Derived The derived class.
/// @details A derived class should implement the `serialize_impl()` method.
///          Example usage:
///          @code
///          class MySerializableClass : public
///          Serializable<MySerializableClass> {
///            friend class Serializable<MySerializableClass>;
///            template <typename Archive>
///            void serialize_impl(Archive& ar, const unsigned int version) {
///              SERIALIZE(ar, member1);
///              SERIALIZE(ar, member2);
///              SERIALIZE_V(ar, member3, version);
///            }
///          private:
///            int member1;
///            double member2;
///            OtherSerializableClass member3;
///          };
///          @endcode
///          To serialize a derived class:
///          @code
///          MySerializableClass obj;
///          boost::archive::text_oarchive oa;
///          oa << obj;  // or obj.serialize(oa);
///          @endcode
///          We have already provide support for boost archive types,
///          other archive types can be supported by specializing the
///          `SerializeToArchive` template for them.
template <typename Derived>
class Serializable {
 public:
  /// @brief Serializes the object to the specified archive.
  /// @tparam Archive The archive type.
  /// @param ar The archive to serialize to.
  /// @param version The version of the serialization (default is 0).
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0) {
    SerializeToArchive<Archive>::op(*this, ar, version);
  }

 private:
  template <class Archive>
  friend struct SerializeToArchive;

  /// @brief Helper method to invoke the derived class's `serialize_impl`
  /// method.
  /// @tparam Archive The archive type.
  /// @param ar The archive to serialize to.
  /// @param version The version of the serialization.
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    derived().serialize_impl(ar, version);
  }

  /// @brief Accesses the derived class.
  /// @return A reference to the derived class.
  Derived& derived() {
    return static_cast<Derived&>(*this);
  }
};

template <typename Data>
class SerializableWrapper : public Serializable<SerializableWrapper<Data>> {
  friend class Serializable<SerializableWrapper<Data>>;
  template <class Archive>
  void serialize_impl(Archive& ar, const unsigned int version = 0) {
    static_assert(
        std::is_same_v<Data, void>,
        "SerializableWrapper must be specialized for specific types!");
  }
};
}  // namespace sk4slam

/// @brief Helper macro for serialization.
#define SERIALIZE_DETAIL(ar, data, name) \
  sk4slam::serialization::serialize(ar, data, name, 0)

#define SERIALIZE(ar, member) SERIALIZE_DETAIL(ar, member, #member)

/// @brief Helper macro for serialization with versioning, this is useful
///        when member itself is versioned, e.g. when member itself is a
///        Serializable derived class with versioning.
#define SERIALIZE_DETAIL_V(ar, data, name, version) \
  sk4slam::serialization::serialize(ar, data, name, version)

#define SERIALIZE_V(ar, member, version) \
  SERIALIZE_DETAIL_V(ar, member, #member, version)
