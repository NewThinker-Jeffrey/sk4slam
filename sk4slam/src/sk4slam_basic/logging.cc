#include "sk4slam_basic/logging.h"

#include <atomic>
#include <mutex>
#include <thread>

#include "sk4slam_basic/time.h"

namespace sk4slam {

Verbose Logging::current_verbose_level = Verbose::INFO;

namespace {
// To ensure each call to log() won't be interrupt by other calls.
static std::mutex log_mutex;

bool force_flush = false;
bool log_lock_enabled = false;
// static std::atomic<bool> log_lock_enabled(true);
}  // namespace

void Logging::enableLogLock() {
  log_lock_enabled = true;
}

void Logging::disableLogLock() {
  log_lock_enabled = false;
}

void Logging::enableForceFlush() {
  force_flush = true;
}

void Logging::disableForceFlush() {
  force_flush = false;
}

void Logging::setVerbose(const std::string& level) {
  if (level == "ALL") {
    setVerbose(Verbose::ALL);
  } else if (level == "DEBUG") {
    setVerbose(Verbose::DEBUG);
  } else if (level == "INFO") {
    setVerbose(Verbose::INFO);
  } else if (level == "WARNING") {
    setVerbose(Verbose::WARNING);
  } else if (level == "ERROR") {
    setVerbose(Verbose::ERROR);
  } else if (level == "SILENT") {
    setVerbose(Verbose::SILENT);
  } else {
    std::cout << "Invalid verbose level requested: " << level << std::endl;
    std::cout
        << "Valid verbose levels are: ALL, DEBUG, INFO, WARNING, ERROR, SILENT"
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void Logging::setVerbose(Verbose level) {
  current_verbose_level = level;
  std::cout << "Setting verbose level to: ";
  switch (current_verbose_level) {
    case Verbose::ALL:
      std::cout << "ALL";
      break;
    case Verbose::DEBUG:
      std::cout << "DEBUG";
      break;
    case Verbose::INFO:
      std::cout << "INFO";
      break;
    case Verbose::WARNING:
      std::cout << "WARNING";
      break;
    case Verbose::ERROR:
      std::cout << "ERROR";
      break;
    case Verbose::SILENT:
      std::cout << "SILENT";
      break;
    default:
      std::cout << std::endl;
      std::cout << "Invalid verbose level requested: "
                << static_cast<uint32_t>(level) << std::endl;
      std::cout << "Valid levels are: ALL, DEBUG, INFO, WARNING, ERROR, SILENT"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  std::cout << std::endl;
}

void Logging::log(
    const char location[], const char line[], const char* format, ...) {
  // To ensure each call to log() won't be interrupt by other calls.
  bool lock_enabled = log_lock_enabled;
  if (lock_enabled) {
    log_mutex.lock();
  }

  // print time first.
  // printf("%s", Time::now().dateStr("%Z %Y-%m-%d %H:%M:%S.$9 ").c_str());
  printf("%s", Time::now().dateStr("%Z %Y-%m-%d %H:%M:%S.$6 ").c_str());
  // printf("%s", Time::now().dateStr("%Z %Y-%m-%d %H:%M:%S.$3 ").c_str());

  printf("%u ", std::this_thread::get_id());

  // Print the location info
  std::string path(location);
  std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
  // Truncate the filename if it's too long.
  static constexpr size_t MAX_FILE_PATH_LEGTH = 64;
  if (base_filename.size() > MAX_FILE_PATH_LEGTH) {
    printf(
        "%s", base_filename
                  .substr(
                      base_filename.size() - MAX_FILE_PATH_LEGTH,
                      base_filename.size())
                  .c_str());
  } else {
    printf("%s", base_filename.c_str());
  }
  printf(":%s ", line);

  // Print the rest of the args
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);

  if (force_flush) {
    fflush(stdout);
  }

  if (lock_enabled) {
    log_mutex.unlock();
  }
}

}  // namespace sk4slam
