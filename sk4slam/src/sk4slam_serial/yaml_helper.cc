#include "sk4slam_serial/yaml_helper.h"

#include <filesystem>
#include <fstream>

#include "sk4slam_cpp/mutex.h"

namespace sk4slam {

namespace {

std::unique_ptr<YamlConfig> root_cfg;
std::unordered_map<std::string, std::unique_ptr<YamlConfig>> named_cfgs;
Mutex root_cfg_mutex;

}  // namespace

std::unique_ptr<YamlObj> parseYaml(const std::string& yaml_str) {
  std::unique_ptr<YamlObj> yaml(new YamlObj);
  *yaml = YAML::Load(yaml_str);
  return yaml;
}

std::unique_ptr<YamlObj> loadYaml(const std::string& file) {
  std::unique_ptr<YamlObj> yaml(new YamlObj);
  *yaml = YAML::LoadFile(file);
  return yaml;
}

namespace {

std::string insertIndents(
    const std::string& str, int indent, bool indent_first_line = false) {
  std::string res;
  std::string indent_str = std::string(indent, ' ');
  auto lines = split(strip(str), '\n');
  for (size_t i = 0; i < lines.size(); ++i) {
    if (indent_first_line || i > 0) {
      res += indent_str;
    }
    res += lines[i];
    if (i < lines.size() - 1) {
      res += '\n';
    }
  }
  return res;
}

bool isScalarList(const YamlObj& yaml) {
  if (yaml.IsSequence() && yaml[0].IsScalar()) {
    return true;
  } else {
    return false;
  }
}

bool isEmptyList(const YamlObj& yaml) {
  return yaml.IsSequence() && yaml.size() == 0;
}

bool isEmptyMap(const YamlObj& yaml) {
  return yaml.IsMap() && yaml.size() == 0;
}

bool shouldPutInOneLine(const YamlObj& yaml) {
  if (yaml.IsScalar() || isScalarList(yaml) || isEmptyList(yaml) ||
      isEmptyMap(yaml)) {
    return true;
  } else {
    return false;
  }
}

std::string formatIntoOneLine(const YamlObj& yaml) {
  if (yaml.IsScalar()) {
    return yaml.as<std::string>();
  } else if (isEmptyList(yaml)) {
    return "[]";
  } else if (isEmptyMap(yaml)) {
    return "{}";
  } else {
    ASSERT(isScalarList(yaml));
    Oss oss;
    oss << "[";
    for (size_t i = 0; i < yaml.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << yaml[i];
    }
    oss << "]";
    return oss.str();
  }
}

void formatYaml(const YamlObj& yaml, Oss& oss, int cur_indent = 0) {
  static const int indent_step = 2;
  if (!yaml.IsDefined()) {
    LOGE("Unknown yaml node type: %d", yaml.Type());
    throw std::runtime_error("Unknown yaml node type in sk4slam::formatYaml()");
  }

  if (yaml.IsNull()) {
    return;
  }

  if (shouldPutInOneLine(yaml)) {
    oss << std::string(cur_indent, ' ') << formatIntoOneLine(yaml) << std::endl;
  } else {
    if (yaml.IsSequence()) {
      for (const auto& item : yaml) {
        bool one_line = shouldPutInOneLine(yaml[0]);
        if (one_line) {
          oss << std::string(cur_indent, ' ')
              << "-" + std::string(indent_step - 1, ' ')
              << formatIntoOneLine(item) << std::endl;
        } else {
          oss << std::string(cur_indent, ' ')
              << "-" + std::string(indent_step - 1, ' ')
              << insertIndents(
                     sk4slam::formatYaml(item), cur_indent + indent_step,
                     false);
          oss << std::endl;
        }
      }
    } else {
      ASSERT(yaml.IsMap());
      for (const auto& item : yaml) {
        bool one_line = shouldPutInOneLine(item.second);
        if (one_line) {
          oss << std::string(cur_indent, ' ') << item.first
              << ":" + std::string(indent_step - 1, ' ')
              << formatIntoOneLine(item.second) << std::endl;
        } else {
          oss << std::string(cur_indent, ' ') << item.first << ":" << std::endl
              << insertIndents(
                     sk4slam::formatYaml(item.second), cur_indent + indent_step,
                     true);
          oss << std::endl;
        }
      }
    }
  }
}

std::unique_ptr<YamlObj> getImmediateSubConfig(
    const YamlObj& yaml, const std::string& maybe_indexed_key) {
  auto brackt_pos = maybe_indexed_key.find('[');
  if (brackt_pos != std::string::npos) {
    std::string key = maybe_indexed_key.substr(0, brackt_pos);
    int idx = std::stoi(maybe_indexed_key.substr(
        brackt_pos + 1, maybe_indexed_key.size() - brackt_pos - 2));
    // LOGA("getImmediateSubConfig: key = %s, idx = %d", key.c_str(), idx);
    auto yaml_at_key = yaml[key];
    if (yaml_at_key && yaml_at_key.IsSequence() && idx < yaml_at_key.size()) {
      return std::make_unique<YamlObj>(yaml_at_key[idx]);
    } else {
      return nullptr;
    }
  } else {
    auto yaml_at_key = yaml[maybe_indexed_key];
    if (yaml_at_key) {
      return std::make_unique<YamlObj>(yaml_at_key);
    } else {
      return nullptr;
    }
  }
}

}  // namespace

std::string formatYaml(const YamlObj& yaml) {
  Oss oss;
  formatYaml(yaml, oss);
  return strip(oss.str());
}

void saveYaml(const std::string& file, const YamlObj& yaml) {
  std::ofstream fout(file);
  if (!fout.is_open()) {
    LOGW(YELLOW "saveYaml(): Can't open file: %s!" RESET, file.c_str());
  }
  // fout << yaml << std::endl;
  fout << formatYaml(yaml) << std::endl;
}

const YamlConfig& YamlConfig::root() {
  if (!root_cfg) {
    UniqueLock lock(root_cfg_mutex);
    if (!root_cfg) {
      const char* env_value = std::getenv("SK4SLAM_ROOT_CFG");
      std::string file = "";
      if (env_value) {
        file = env_value;
      } else {
        file = "./sk4slam_root_cfg.yaml";
      }
      if (std::filesystem::exists(file)) {
        LOGI("YamlConfig::root(): Using root config file: %s", file.c_str());
        root_cfg = std::make_unique<YamlConfig>(file);
      } else {
        LOGW(
            "YamlConfig::root(): Can't find root config file: %s!",
            file.c_str());
        root_cfg = std::make_unique<YamlConfig>();
      }
    }
  }
  ASSERT(root_cfg);
  return *root_cfg;
}

bool YamlConfig::registerNamed(const std::string& name, const YamlConfig& cfg) {
  UniqueLock lock(root_cfg_mutex);
  if (named_cfgs.count(name)) {
    LOGW(
        "YamlConfig::RegisterNamed(): Named config '%s' already exists!",
        name.c_str());
    return false;
  }
  named_cfgs[name] = std::make_unique<YamlConfig>(cfg);
  return true;
}

bool YamlConfig::registerNamed(
    const std::string& name, const std::string& cfg_file) {
  return registerNamed(name, YamlConfig(cfg_file));
}

const YamlConfig& YamlConfig::getNamed(const std::string& name) {
  static YamlConfig null_cfg;
  UniqueLock lock(root_cfg_mutex);
  auto iter = named_cfgs.find(name);
  if (iter == named_cfgs.end()) {
    LOGE(
        "YamlConfig::GetNamed(): Named config '%s' doesn't exist!",
        name.c_str());
    return null_cfg;
  }
  return *(iter->second);
}

std::unique_ptr<YamlObj> YamlConfig::getSubConfig(
    const YamlObj& top_yaml, const std::string& config_key) {
  std::vector<std::string> keys = split(config_key, '.');
  auto yaml = getImmediateSubConfig(top_yaml, keys[0]);
  if (keys.size() > 1) {
    for (int i = 1; yaml && i < keys.size(); i++) {
      yaml = getImmediateSubConfig(*yaml, keys[i]);
    }
  }
  return yaml;
}

}  // namespace sk4slam
