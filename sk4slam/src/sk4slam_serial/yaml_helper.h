#pragma once

#include <Eigen/Core>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "sk4slam_basic/configurable.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/serializable.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"
#include "sk4slam_serial/serialization_helper.h"

// #define ENABLE_SK4SLAM_MATRIX_YAML_CONVERT

namespace sk4slam {
using YamlObj = YAML::Node;

// Helper struct to check if a class is derived from Serializable
template <typename Derived>
struct is_serializable
    : std::is_base_of<sk4slam::Serializable<Derived>, Derived> {};

//////////// matrix <--> yaml //////////////

template <typename Matrix>
YamlObj matrixToYaml(const Matrix& rhs);

template <typename Matrix>
bool yamlToMatrix(const YamlObj& node, Matrix& rhs);

template <typename Matrix = Eigen::MatrixXd>
Matrix yamlToMatrix(const YamlObj& node);

}  // namespace sk4slam

namespace YAML {

// Define the YAML::convert for STL containers and Eigen matrices.
//
// For Eigen matrices, if the matrix only has one row or one column (at
// compile time), it will be treated as a 1d array. Otherwise, it will
// be treated as a 2d array.
//
// For STL containers, since the converts for std::map and std::vector
// are already implemented in yaml-cpp, we only implement that for
// other frequently used STL containers (including std::unordered_map,
// std::set, and std::unordered_set)

#ifdef ENABLE_SK4SLAM_MATRIX_YAML_CONVERT
// The definition of the convert for Eigen::Matrix may cause redefinition
// errors when compiling sk4slam together with other libraries (e.g.,
// maplab), so we provide a macro here to enable or disable it.

template <
    typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
    int _MaxCols>
struct convert<
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  using Matrix =
      Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

  static Node encode(const Matrix& rhs) {
    return sk4slam::matrixToYaml(rhs);
  }

  static bool decode(const Node& node, Matrix& rhs) {
    return sk4slam::yamlToMatrix(node, rhs);
  }
};

#endif  // ENABLE_SK4SLAM_MATRIX_YAML_CONVERT

// std::unordered_map
template <typename K, typename V>
struct convert<std::unordered_map<K, V>> {
  static Node encode(const std::unordered_map<K, V>& rhs) {
    Node node(NodeType::Map);
    for (typename std::unordered_map<K, V>::const_iterator it = rhs.begin();
         it != rhs.end(); ++it) {
      node.force_insert(it->first, it->second);
    }
    return node;
  }

  static bool decode(const Node& node, std::unordered_map<K, V>& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    rhs.clear();
    for (const_iterator it = node.begin(); it != node.end(); ++it) {
      rhs[it->first.template as<K>()] = it->second.template as<V>();
    }
    return true;
  }
};

// std::unordered_set
template <typename T>
struct convert<std::unordered_set<T>> {
  static Node encode(const std::unordered_set<T>& rhs) {
    Node node(NodeType::Sequence);
    for (typename std::unordered_set<T>::const_iterator it = rhs.begin();
         it != rhs.end(); ++it) {
      node.push_back(*it);
    }
    return node;
  }

  static bool decode(const Node& node, std::unordered_set<T>& rhs) {
    if (!node.IsSequence()) {
      return false;
    }

    rhs.clear();
    for (const_iterator it = node.begin(); it != node.end(); ++it) {
      auto res = rhs.insert(it->template as<T>());
      if (!res.second) {
        // TODO(jeffrey): toStr() (ostream operator <<) may not support type T.
        LOGW(
            "Duplicate key found in YAML: %s",
            sk4slam::toStr(it->template as<T>()).c_str());
      }
    }
    return true;
  }
};

// std::set
template <typename T>
struct convert<std::set<T>> {
  static Node encode(const std::set<T>& rhs) {
    Node node(NodeType::Sequence);
    for (typename std::set<T>::const_iterator it = rhs.begin(); it != rhs.end();
         ++it) {
      node.push_back(*it);
    }
    return node;
  }

  static bool decode(const Node& node, std::set<T>& rhs) {
    if (!node.IsSequence()) {
      return false;
    }

    rhs.clear();
    for (const_iterator it = node.begin(); it != node.end(); ++it) {
      auto res = rhs.insert(it->template as<T>());
      if (!res.second) {
        // TODO(jeffrey): toStr() (ostream operator <<) may not support type T.
        LOGW(
            "Duplicate key found in YAML: %s",
            sk4slam::toStr(it->template as<T>()).c_str());
      }
    }
    return true;
  }
};

}  // namespace YAML

namespace sk4slam {

std::unique_ptr<YamlObj> parseYaml(const std::string& yaml_str);

std::unique_ptr<YamlObj> loadYaml(const std::string& file);

std::string formatYaml(const YamlObj& yaml);

void saveYaml(const std::string& file, const YamlObj& yaml);

template <typename Serializable>
void saveAsYaml(
    const Serializable& obj, const std::string& file, unsigned int version = 0);

template <typename Serializable>
void loadFromYaml(
    Serializable& obj, const std::string& file, unsigned int version = 0);

template <typename Serializable>
Serializable loadFromYaml(const std::string& file, unsigned int version = 0) {
  Serializable obj;
  loadFromYaml(obj, file, version);
  return obj;
}

/// @brief Used to serialize data into a YAML document.
class YamlOutArchive {
 public:
  YamlOutArchive() = default;

  using is_saving = std::true_type;
  using is_loading = std::false_type;
  static_assert(is_saving::value);
  static_assert(!is_loading::value);

  /// @brief Gets the serialized YAML node.
  /// @return The root YAML node containing serialized data.
  YAML::Node getNode() const {
    return root_node_;
  }

  void save(const std::string& file) const {
    saveYaml(file, root_node_);
  }

  /// @brief Serializes a single member into the YAML node.
  /// Specializes behavior for Serializable-derived classes.
  template <typename T>
  void serialize(
      const T& const_value, const char* name = nullptr,
      const unsigned int version = 0) {
    T& value = const_cast<T&>(
        const_value);  // This is safe because we will not modify the object.
    if constexpr (IsSerializable<T>) {
      // Handle Serializable-derived objects
      YamlOutArchive sub_archive;
      value.serialize(sub_archive, version);
      if (name) {
        root_node_[name] = sub_archive.getNode();
      } else {
        root_node_ = sub_archive.getNode();
      }
    } else if constexpr (is_stl_map_v<T>) {
      // Handle STL map-like containers
      using KeyType = typename T::key_type;
      using MappedType = typename T::mapped_type;

      // Static check: Ensure key type is not a pointer
      static_assert(
          !std::is_pointer_v<KeyType>, "Map key type cannot be a pointer.");

      YAML::Node map_node;
      for (auto& [key, val] : value) {
        if constexpr (std::is_same_v<KeyType, std::string>) {
          // Store as a YAML map if the key is a string
          YamlOutArchive val_archive;
          serialization::serialize(
              val_archive, val, nullptr,
              version);  // Serialize value recursively
          map_node[key] = val_archive.getNode();
        } else {
          // Store as a list of {"key": key, "value": value}
          YAML::Node entry;
          YamlOutArchive key_archive, val_archive;
          serialization::serialize(
              key_archive, key, nullptr, version);  // Serialize key recursively
          serialization::serialize(
              val_archive, val, nullptr,
              version);  // Serialize value recursively
          entry["key"] = key_archive.getNode();
          entry["value"] = val_archive.getNode();
          map_node.push_back(entry);
        }
      }
      if (name) {
        root_node_[name] = map_node;
      } else {
        root_node_ = map_node;
      }
    } else if constexpr (is_stl_container_v<T>) {
      // Handle generic STL containers
      YAML::Node container_node;
      for (auto& item : value) {
        YamlOutArchive sub_archive;
        serialization::serialize(
            sub_archive, item, nullptr,
            version);  // Serialize each element recursively
        container_node.push_back(sub_archive.getNode());
      }
      if (name) {
        root_node_[name] = container_node;
      } else {
        root_node_ = container_node;
      }
    } else if constexpr (is_shared_ptr_v<T> || is_unique_ptr_v<T>) {
      // Handle shared_ptr and unique_ptr
      if (value) {
        YamlOutArchive sub_archive;
        serialization::serialize(
            sub_archive, *value, nullptr,
            version);  // Dereference and serialize
        if (name) {
          root_node_[name] = sub_archive.getNode();
        } else {
          root_node_ = sub_archive.getNode();
        }
      } else {
        if (name) {
          root_node_[name] = YAML::Null;
        } else {
          root_node_ = YAML::Null;
        }
      }
    } else if constexpr (is_std_pair_v<T>) {
      // Handle std::pair
      YAML::Node pair_node;
      YamlOutArchive first_archive, second_archive;
      serialization::serialize(
          first_archive, value.first, nullptr,
          version);  // Serialize first element
      serialization::serialize(
          second_archive, value.second, nullptr,
          version);  // Serialize second element
      pair_node["first"] = first_archive.getNode();
      pair_node["second"] = second_archive.getNode();
      // pair_node.push_back(first_archive.getNode());
      // pair_node.push_back(second_archive.getNode());
      if (name) {
        root_node_[name] = pair_node;
      } else {
        root_node_ = pair_node;
      }
    } else if constexpr (is_std_tuple_v<T>) {
      // Handle std::tuple
      YAML::Node tuple_node;
      serialize_tuple(
          value, tuple_node, version);  // Recursively serialize tuple elements
      if (name) {
        root_node_[name] = tuple_node;
      } else {
        root_node_ = tuple_node;
      }
    } else {
      // Handle primitive types and others
      if (name) {
        root_node_[name] = value;
      } else {
        root_node_ = value;
      }
    }
  }

  template <typename T>
  YamlOutArchive& operator<<(const T& value) {
    serialize(value);
    return *this;
  }

 protected:
  template <size_t I, typename Tuple>
  void serialize_tuple_element(
      const Tuple& tuple, YAML::Node& node, const unsigned int version) {
    YamlOutArchive sub_archive;
    serialization::serialize(sub_archive, std::get<I>(tuple), nullptr, version);
    node.push_back(sub_archive.getNode());
  }

  template <typename Tuple, size_t... Indices>
  void serialize_tuple_impl(
      const Tuple& tuple, YAML::Node& node, const unsigned int version,
      std::index_sequence<Indices...>) {
    (..., serialize_tuple_element<Indices>(tuple, node, version));
  }

  template <typename Tuple>
  void serialize_tuple(
      const Tuple& tuple, YAML::Node& node, const unsigned int version) {
    constexpr size_t tuple_size = std::tuple_size_v<Tuple>;
    serialize_tuple_impl(
        tuple, node, version, std::make_index_sequence<tuple_size>{});
  }

 private:
  YAML::Node root_node_;  ///< Root YAML node containing all serialized data.
};

/// @brief Used to deserialize data from a YAML document.
class YamlInArchive {
 public:
  /// @brief Constructor to initialize with a YAML document.
  /// @param root_node The root YAML node containing the data.
  /// @param throw_if_not_found If true, throws an exception if a key is not
  /// found.
  explicit YamlInArchive(
      const YAML::Node& root_node, bool throw_if_not_found = false)
      : root_node_(root_node), throw_if_not_found_(throw_if_not_found) {}

  /// @brief Constructor to load a YAML document from a file.
  explicit YamlInArchive(
      const std::string& filename, bool throw_if_not_found = false)
      : YamlInArchive(*loadYaml(filename), throw_if_not_found) {}

  using is_saving = std::false_type;
  using is_loading = std::true_type;
  static_assert(is_loading::value);
  static_assert(!is_saving::value);

  /// @brief Deserializes a single member from the YAML node.
  /// Specializes behavior for Serializable-derived classes.
  template <typename T>
  void serialize(
      T& value, const char* name = nullptr, const unsigned int version = 0) {
    YAML::Node node = name ? root_node_[name] : root_node_;

    if (!node) {
      if (throw_if_not_found_) {
        LOGE(
            "YamlInArchive::serialize(): Key '%s' not found in YAML node!",
            name ? name : "root");
        throw std::runtime_error(
            "Key '" + std::string(name ? name : "root") +
            "' not found in YAML node");
      }
      return;
    }

    if constexpr (IsSerializable<T>) {
      // Handle Serializable-derived objects
      YamlInArchive sub_archive(node, throw_if_not_found_);
      value.serialize(sub_archive, version);
    } else if constexpr (is_stl_map_v<T>) {
      // Handle STL map-like containers
      using KeyType = typename T::key_type;
      using MappedType = typename T::mapped_type;

      // Static check: Ensure key type is not a pointer
      static_assert(
          !std::is_pointer_v<KeyType>, "Map key type cannot be a pointer.");

      value.clear();
      if constexpr (std::is_same_v<KeyType, std::string>) {
        // If the key is a string, treat as a standard YAML map
        for (auto it = node.begin(); it != node.end(); ++it) {
          MappedType val;
          YamlInArchive val_archive(it->second, throw_if_not_found_);
          serialization::serialize(
              val_archive, val, nullptr,
              version);  // Deserialize value recursively
          value[it->first.as<KeyType>()] = val;
        }
      } else {
        // If the key is not a string, treat as a list of {"key": key, "value":
        // value}
        for (auto it = node.begin(); it != node.end(); ++it) {
          KeyType key;
          MappedType val;
          YamlInArchive key_archive((*it)["key"], throw_if_not_found_);
          YamlInArchive val_archive((*it)["value"], throw_if_not_found_);
          serialization::serialize(
              key_archive, key, nullptr,
              version);  // Deserialize key recursively
          serialization::serialize(
              val_archive, val, nullptr,
              version);  // Deserialize value recursively
          value[key] = val;
        }
      }
    } else if constexpr (is_stl_container_v<T>) {
      // Handle generic STL containers
      value.clear();
      for (auto it = node.begin(); it != node.end(); ++it) {
        typename T::value_type element;
        YamlInArchive sub_archive(*it, throw_if_not_found_);
        serialization::serialize(
            sub_archive, element, nullptr,
            version);  // Deserialize each element recursively
        value.insert(value.end(), element);
      }
    } else if constexpr (is_shared_ptr_v<T> || is_unique_ptr_v<T>) {
      // Handle shared_ptr and unique_ptr
      if (value) {
        throw std::runtime_error(
            "Pointer must be null before deserialization.");
      }
      if (!node.IsNull()) {
        if constexpr (is_shared_ptr_v<T>) {
          value = std::make_shared<typename T::element_type>();
        } else if constexpr (is_unique_ptr_v<T>) {
          value = std::make_unique<typename T::element_type>();
        }
        YamlInArchive sub_archive(node, throw_if_not_found_);
        serialization::serialize(sub_archive, *value, nullptr, version);
      }
    } else if constexpr (is_std_pair_v<T>) {
      // Handle std::pair
      YAML::Node first_node = node["first"];
      YAML::Node second_node = node["second"];
      if (!first_node || !second_node) {
        throw std::runtime_error(
            "YamlInArchive::serialize(): Missing 'first' or 'second' field for "
            "std::pair.");
      }
      YamlInArchive first_archive(first_node, throw_if_not_found_);
      YamlInArchive second_archive(second_node, throw_if_not_found_);
      serialization::serialize(
          first_archive, value.first, nullptr,
          version);  // Deserialize first element
      serialization::serialize(
          second_archive, value.second, nullptr,
          version);  // Deserialize second element
    } else if constexpr (is_std_tuple_v<T>) {
      deserialize_tuple(
          value, node, version);  // Recursively deserialize tuple elements
    } else {
      // Handle primitive types and others
      value = node.as<T>();
    }
  }

  template <typename T>
  YamlInArchive& operator>>(T& value) {
    serialize(value);
    return *this;
  }

 protected:
  template <typename Tuple, size_t... Indices>
  void deserialize_tuple_impl(
      Tuple& tuple, const YAML::Node& node, const unsigned int version,
      std::index_sequence<Indices...>) {
    (..., (serialization::serialize(
              YamlInArchive{node[Indices]}, std::get<Indices>(tuple), nullptr,
              version)));
  }

  template <typename Tuple>
  void deserialize_tuple(
      Tuple& tuple, const YAML::Node& node, const unsigned int version) {
    constexpr size_t tuple_size = std::tuple_size_v<Tuple>;
    deserialize_tuple_impl(
        tuple, node, version, std::make_index_sequence<tuple_size>{});
  }

 private:
  YAML::Node root_node_;  ///< Root YAML node containing all deserialized data.
  bool throw_if_not_found_;  ///< Flag to determine behavior when a key is not
                             ///< found.
};

template <typename Serializable>
void saveAsYaml(
    const Serializable& obj, const std::string& file, unsigned int version) {
  YamlOutArchive archive;
  serialization::serialize(archive, obj, nullptr, version);
  archive.save(file);
}

template <typename Serializable>
void loadFromYaml(
    Serializable& obj, const std::string& file, unsigned int version) {
  YamlInArchive archive(file);
  serialization::serialize(archive, obj, nullptr, version);
}

namespace yaml_config_internal {
template <typename T>
inline constexpr bool IsEigenMatrix = false;
template <
    typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
inline constexpr bool IsEigenMatrix<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> = true;
}  // namespace yaml_config_internal

class YamlConfig : public Configurable<YamlConfig> {
 public:
  YamlConfig() {}

  explicit YamlConfig(const std::string& file) : yaml_(*loadYaml(file)) {}

  /// @note A shallow copy of @p yaml is created.
  explicit YamlConfig(const YamlObj& yaml) : yaml_(yaml) {}

  YamlConfig(const YamlConfig& other) : YamlConfig(other.yaml_) {}

  YamlConfig& operator=(const YamlConfig& other) {
    yaml_.reset(other.yaml_);
    return *this;
  }

  void load(const std::string& file) {
    yaml_.reset(*loadYaml(file));
  }

 public:
  /// You can use the environment variable 'SK4SLAM_GLOBAL_CFG'
  /// to specify you root config file (.yaml);
  /// If the environment variable is not set, we'll try the
  /// 'sk4slam_root_cfg.yaml' file in the current working directory.
  static const YamlConfig& root();

  static bool registerNamed(const std::string& name, const YamlConfig& cfg);

  static bool registerNamed(
      const std::string& name, const std::string& cfg_file);

  static const YamlConfig& getNamed(const std::string& name);

 public:  // Implementation of the interfaces of the base class Configurable.
  /// @brief  Implementation of @ref Configurable interface
  ///         @ref Configurable::has().
  ///         _has_impl() should return true if the config key exists,
  ///         otherwise false.
  bool _has_impl(const std::string& config_key) const {
    return getSubConfig(yaml_, config_key) != nullptr;
  }

  /// @brief  Implementation of  @ref Configurable interface
  ///         @ref Configurable::get().
  ///         _get_impl() should return the value of the config key. If
  ///         the config key does not exist, the behavior depends on the
  ///         @p required parameter:
  ///         - If @p required is false, the @p default_value will be
  ///           returned.
  ///         - Otherwise, an exception will be thrown (when T is not
  ///           @c Derived ) or an empty sub-config will be returned
  ///           (when T is @c Derived ).
  template <typename T>
  T _get_impl(
      const std::string& config_key, bool required,
      const T* default_value) const {
    auto yaml_ptr = getSubConfig(yaml_, config_key);
    if (yaml_ptr) {
      auto& yaml = *yaml_ptr;
      if constexpr (std::is_same_v<T, YamlConfig>) {
        return YamlConfig(yaml);
      } else if constexpr (yaml_config_internal::IsEigenMatrix<T>) {
        return yamlToMatrix<T>(yaml);
      } else {
        return yaml.as<T>();
      }
    } else {
      if constexpr (std::is_same_v<T, YamlConfig>) {
        if (default_value) {
          return *default_value;
        } else {
          return YamlConfig();  // Return an empty sub config
        }
      } else {
        if (required) {
          std::string error_msg = formatStr(
              "YamlConfig: Key '%s' is required but not found in the config!",
              config_key.c_str());
          LOGE("%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        } else {
          return *default_value;
        }
      }
    }
  }

  const YAML::Node& yaml() const {
    return yaml_;
  }

 public:  // Functions for testing or debugging.
  /// @brief   Return the underlying YAML::Node object. This function is
  ///          intended to be used only for testing or debugging purposes.
  ///
  /// The YamlConfig class is designed to be a read-only wrapper of the
  /// internal configuration, so we encourage users to use this class in a
  /// read-only style and no convenience functions to modify the internal
  /// YAML::Node object are provided.
  ///
  /// However, in some testing or debugging scenarios, it might be helpful
  /// if users can modify the internal configuration data. Therefore,
  /// we provide this function to allow users to get a reference to the
  /// internal YAML::Node object and modify it directly.
  /// For example, you can use the following code to change the value of
  /// a config item:
  ///
  ///   cfg.mutableYaml()["foo"]["bar"] = 123;
  ///
  /// @warning
  ///           There might be other YamlConfig objects sharing the same
  ///           internal data with this object (The copy constructor and
  ///           assignment operator of YamlConfig will create a new YamlConfig
  ///           object with the same internal data), so modifying the returned
  ///           YAML::Node object may affect other YamlConfig objects, which
  ///           may lead to unexpected behavior. To avoid this, you may need to
  ///           call the @ref cloneInternal() function before getting the
  ///           mutable YAML::Node object. For example:
  ///   cfg.cloneInternal();
  ///   cfg.mutableYaml()["foo"]["bar"] = 123;
  ///
  YamlObj& mutableYaml() {
    return yaml_;
  }

  /// @brief    Clone the internal YAML::Node object.
  ///
  /// This function might be useful when you need to modify the internal
  /// YAML::Node (by calling @ref mutableYaml() ) without affecting other
  /// YamlConfig objects sharing the same data.
  /// As @ref mutableYaml() , this function is intended to be used only for
  /// testing or debugging purposes.
  void cloneInternal() {
    yaml_.reset(YAML::Clone(yaml_));
  }

 protected:
  static std::unique_ptr<YamlObj> getSubConfig(
      const YamlObj& yaml, const std::string& config_key);

 private:
  YamlObj yaml_;
};

inline const YamlConfig& rootCfg() {
  return YamlConfig::root();
}

inline const YamlConfig& getNamedCfg(const std::string& name) {
  return YamlConfig::getNamed(name);
}

inline bool registerNamedCfg(const std::string& name, const YamlConfig& cfg) {
  return YamlConfig::registerNamed(name, cfg);
}

inline bool registerNamedCfg(
    const std::string& name, const std::string& cfg_file) {
  return YamlConfig::registerNamed(name, cfg_file);
}

//////////// matrix <--> yaml //////////////

template <typename Matrix>
Matrix yamlToMatrix(const YamlObj& node) {
  Matrix rhs;
  yamlToMatrix(node, rhs);
  return rhs;
}

template <typename Vector>
YamlObj vectorToYaml(const Vector& rhs) {
  static_assert(
      Vector::RowsAtCompileTime == 1 || Vector::ColsAtCompileTime == 1);
  return sk4slam::matrixToYaml(rhs);
}

template <typename Vector = Eigen::VectorXd>
Vector yamlToVector(const YamlObj& node) {
  static_assert(
      Vector::RowsAtCompileTime == 1 || Vector::ColsAtCompileTime == 1);
  Vector rhs;
  yamlToMatrix(node, rhs);
  return rhs;
}

template <typename Matrix>
YamlObj matrixToYaml(const Matrix& rhs) {
  YamlObj node(YAML::NodeType::Sequence);
  if constexpr (
      Matrix::RowsAtCompileTime == 1 || Matrix::ColsAtCompileTime == 1) {
    // Eigen::Vector (1d array)
    for (int i = 0; i < rhs.size(); ++i) {
      node.push_back(rhs(i));
    }
  } else {
    // General Eigen::Matrix (2d array)
    for (int i = 0; i < rhs.rows(); ++i) {
      YamlObj row(YAML::NodeType::Sequence);
      for (int j = 0; j < rhs.cols(); ++j) {
        row.push_back(rhs(i, j));
      }
      node.push_back(row);
    }
  }
  return node;
}

template <typename Matrix>
bool yamlToMatrix(const YamlObj& node, Matrix& rhs) {
  using Scalar = typename Matrix::Scalar;
  if (!node.IsSequence()) {
    LOGW("Non-Sequece yaml node can not be converted to Matrix!");
    return false;
  }

  if constexpr (
      Matrix::RowsAtCompileTime == 1 || Matrix::ColsAtCompileTime == 1) {
    // Eigen::Vector (1d array)
    size_t size = node.size();
    if constexpr (
        Matrix::RowsAtCompileTime == Eigen::Dynamic ||
        Matrix::ColsAtCompileTime == Eigen::Dynamic) {
      rhs.resize(size);
    } else {
      if (size != rhs.size()) {
        LOGW(
            "Fail to converted yaml node to Matrix since the size of the "
            "yaml node is not equal to that of the matrix!");
        return false;
      }
    }
    for (size_t i = 0; i < size; ++i) {
      rhs(i) = node[i].as<double>();
    }
  } else {
    // General Eigen::Matrix (2d array)
    int rows = node.size();
    if (rows == 0) {
      LOGW("Empty yaml node can not be converted to Matrix!");
      return false;
    }
    if (!node[0].IsSequence()) {
      LOGW(
          "Non-Sequece yaml node can not be converted to a row of Matrix! "
          "(row = %d)",
          0);
      return false;
    }
    int cols = node[0].size();
    if (cols == 0) {
      LOGW(
          "Empty yaml node can not be converted to a row of Matrix! (row = "
          "%d)",
          0);
      return false;
    }
    if constexpr (Matrix::RowsAtCompileTime != Eigen::Dynamic) {
      if (rows != rhs.rows()) {
        LOGW(
            "Fail to converted yaml node to Matrix since the number of rows "
            "of the yaml node is not equal to that of the matrix!");
        return false;
      }
    }
    if constexpr (Matrix::ColsAtCompileTime != Eigen::Dynamic) {
      if (cols != rhs.cols()) {
        LOGW(
            "Fail to converted yaml node to Matrix since the number of "
            "columns of the yaml node is not equal to that of the matrix!");
        return false;
      }
    }
    if constexpr (
        Matrix::RowsAtCompileTime == Eigen::Dynamic ||
        Matrix::ColsAtCompileTime == Eigen::Dynamic) {
      rhs.resize(rows, cols);
    }
    for (int i = 0; i < rows; ++i) {
      if (!node[i].IsSequence()) {
        LOGW(
            "Non-Sequece yaml node can not be converted to a row of Matrix! "
            "(row = %d)",
            i);
        return false;
      }
      if (node[i].size() != cols) {
        LOGW(
            "Fail to converted yaml node to Matrix since the number of "
            "columns of row %d is not equal to that of row 0!",
            i, cols);
        return false;
      }
      for (int j = 0; j < cols; ++j) {
        rhs(i, j) = node[i][j].as<double>();
      }
    }
  }
  return true;
}

}  // namespace sk4slam

namespace YAML {
template <>
struct convert<sk4slam::YamlConfig> {
  static Node encode(const sk4slam::YamlConfig& rhs) {
    return rhs.yaml();
  }

  static bool decode(const Node& node, sk4slam::YamlConfig& rhs) {
    rhs.mutableYaml() = node;
    return true;
  }
};
}  // namespace YAML

namespace std {
inline ostream& operator<<(ostream& os, const sk4slam::YamlConfig& rhs) {
  os << rhs.yaml();
  return os;
}
}  // namespace std
