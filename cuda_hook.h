#pragma once

#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::string &name) : lib_name_(name) {
    int32_t flags{RTLD_LAZY};
    handle_ = dlopen(lib_name_.c_str(), flags);
    if (handle_ == nullptr) {
      std::string errorStr = " due to " + std::string{dlerror()};
      throw std::runtime_error("Unable to open library: " + lib_name_ +
                               errorStr);
    }
  }

  void *SymbolAddress(const char *name) {
    if (handle_ == nullptr) {
      throw std::runtime_error("Handle to library is nullptr");
    }
    void *ret = dlsym(handle_, name);
    if (ret == nullptr) {
      const std::string kERROR_MSG(
          lib_name_ + ": error loading symbol: " + std::string(name));
      throw std::invalid_argument(kERROR_MSG);
    }
    return ret;
  }

  ~DynamicLibrary() {
    try {
      dlclose(handle_);
    } catch (...) {
      std::cerr << "Unable to close library: " << lib_name_ << std::endl;
    }
  }

private:
  DynamicLibrary(const DynamicLibrary &) = delete;
  DynamicLibrary &operator=(const DynamicLibrary &) = delete;

  std::string lib_name_{};
  void *handle_{};
};
