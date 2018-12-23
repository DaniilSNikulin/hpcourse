#pragma once

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <chrono>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iomanip>

#include <exception>
#include <stdexcept>


template <class Period = std::ratio<1>>
class Timer
{
  public:
    Timer() : beg_(clock_::now())
    {}

    void reset() {
      beg_ = clock_::now();
    }

    double elapsed() const {
      return std::chrono::duration_cast<duration_>
        (clock_::now() - beg_).count();
    }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, Period> duration_;
    std::chrono::time_point<clock_> beg_;
};



namespace cl_helper {

inline void _setKernelParameters(cl::Kernel &k,int i){}

template<typename T, typename... Args>
inline void _setKernelParameters(cl::Kernel &kernel,int i, const T &firstParameter, const Args& ...restOfParameters) {
  kernel.setArg(i, firstParameter);
  _setKernelParameters(kernel,i+1,restOfParameters...);
}

template<typename... Args>
inline void setKernelParameters(cl::Kernel &kernel, const Args& ...args) {
  _setKernelParameters(kernel, 0, args...);//first number of parameter is 0
}



template<class T, class CoreFunction>
bool check(std::vector<T> const & a,
           std::vector<T> const & b,
           std::vector<T> const & c,
           CoreFunction f)
{
  const size_t min_size = std::min(a.size(), b.size());
  if (c.size() < min_size) {
    return false;
  }
  for (size_t i = 0; i < c.size(); ++i) {
    if (c[i] != f(a, b, i)) {
      return false;
    }
  }
  return true;
}

template<class T, class CoreFunction>
bool check(std::vector<T> const & a,
           std::vector<T> const & c,
           CoreFunction f,
           size_t size)
{
  const double eps = 1e-8;
  if (c.size() < size) {
    return false;
  }
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(c[i] - f(a, i)) > eps) {
      return false;
    }
  }
  return true;
}




struct empty_platform_exception : public std::runtime_error {
  empty_platform_exception()
    : std::runtime_error(" No platforms found. Check OpenCL installation!\n")
  {}
};
struct empty_device_exception : public std::runtime_error {
  empty_device_exception()
    : std::runtime_error(" No gpu devices found. Check OpenCL installation!\n")
  {}
};
struct program_build_exception : public std::runtime_error {
  std::string log_str;

  program_build_exception(cl::Error const & e,
                          cl::Program const & program,
                          cl::Device const & device)
    : std::runtime_error("")
  {
    std::stringstream ss;
    ss << "error in build:"
      << e.what() << " : " << e.err()
      << "\n"
      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
      << "\n";
    log_str = ss.str();
  }

  const char * what() const noexcept override {
    return log_str.c_str();
  }
};



using std::vector;
using std::string;
using std::function;


struct default_platform_getter {
  cl::Platform operator()(vector<cl::Platform> const & all_platforms) {
    return all_platforms[0];
  }
};
struct default_device_getter {
  cl::Device operator()(vector<cl::Device> const & all_devices) {
    return all_devices[0];
  }
};


class Initializer
{
  public:
    using PlatformGetter = std::function<cl::Platform(vector<cl::Platform> const &)>;
    using DeviceGetter = std::function<cl::Device(vector<cl::Device> const &)>;

    Initializer(vector<string> const & file_names,
                bool debug = true,
                PlatformGetter platform_getter = default_platform_getter(),
                DeviceGetter device_getter = default_device_getter())
    {
      vector<cl::Platform> all_platforms;
      cl::Platform::get(&all_platforms);
      if (all_platforms.size()==0) {
        throw empty_platform_exception();
      }

      if (debug) {
        std::cout << "list all platforms:\n";
        for (const cl::Platform & platform : all_platforms) {
          std::cout << "\t" << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        }
      }

      platform = platform_getter(all_platforms);
      if (debug) {
        std::cout << "Using platform: "
          << platform.getInfo<CL_PLATFORM_NAME>()
          << "\n";
      }

      vector<cl::Device> all_devices;
      platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
      if (all_devices.size() == 0) {
        throw empty_device_exception();
      }

      if (debug) {
        std::cout << "list all devices:\n";
        for (const cl::Device & device : all_devices) {
          std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << "\n";
        }
      }

      device = device_getter(all_devices);
      if (debug) {
        std::cout << "Using device: "
          << device.getInfo<CL_DEVICE_NAME>()
          << "\n";
      }

      vector<string> texts = read_sources(file_names);
      cl::Program::Sources sources;
      std::transform(texts.begin(), texts.end(), std::back_inserter(sources),
                     [](string const & text) {
                     return std::make_pair(text.c_str(), text.length() + 1); });

      context = cl::Context({device});
      program = cl::Program(context, sources);
      queue = cl::CommandQueue(context, device);


      try {
        program.build({device});
      } catch (cl::Error const & e) {
        throw program_build_exception(e, program, device);
      }

    }

  public:
    static vector<string> read_sources(vector<string> const & file_names) {
      vector<string> sources;
      sources.reserve(file_names.size());
      for (string const & file_name : file_names) {
        std::ifstream cl_file(file_name);
        std::string cl_string(std::istreambuf_iterator<char>(cl_file),
                              (std::istreambuf_iterator<char>()));
        sources.push_back(cl_string);
      }
      return sources;
    }

  public:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

}


