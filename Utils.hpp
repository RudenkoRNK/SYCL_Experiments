#pragma once

#include <cassert>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <CL/sycl.hpp>

#include <Utility/Misc.hpp>

static int GetIntArgument(int argc, char *argv[], int defaultValue = 0,
                          size_t nArg = 0) {
  auto index = nArg + 1;
  if (argc <= index)
    return defaultValue;
  auto arg = std::string(argv[index]);
  return std::stoi(arg);
}

static std::vector<cl::sycl::cl_int> GetRandomVector(size_t size) {
  auto rd = std::random_device{};
  auto gen = std::mt19937{static_cast<unsigned>(rd())};
  auto vec = std::vector<cl::sycl::cl_int>(size);
  auto rand = std::uniform_int_distribution<>();
  for (auto i = size_t{0}; i < size; ++i)
    vec[i] = rand(gen);
  return vec;
}

static std::string ToString(cl::sycl::info::device_type deviceType) {
  using namespace cl::sycl;
  switch (deviceType) {
  case info::device_type::cpu:
    return "cpu";
  case info::device_type::gpu:
    return "gpu";
  case info::device_type::accelerator:
    return "accelerator";
  case info::device_type::all:
    return "all";
  default:
    assert(false);
  }
}

static void PrintInfo(cl::sycl::queue const &queue, std::ostream &os) {
  using namespace cl::sycl;
  auto device = queue.get_device();
  os << device.get_info<info::device::name>() << std::endl;
  os << "Driver version: " << device.get_info<info::device::driver_version>()
     << std::endl;
  os << "Compute units: " << device.get_info<info::device::max_compute_units>()
     << std::endl;
  auto wgSize = device.get_info<info::device::max_work_group_size>();
  auto localMem = device.get_info<info::device::local_mem_size>();
  os << "Work group size: " << wgSize << std::endl;
  os << "Local memory size: " << localMem / 1024 << "KiB"
     << " (" << localMem / wgSize << "B per work item)" << std::endl;
  os << "Max clock frequency: "
     << device.get_info<info::device::max_clock_frequency>() << "MHz\n";
  os << "Total memory : "
     << (device.get_info<info::device::global_mem_size>() / (1024 * 1024))
     << "MiB" << std::endl;
  os << "Addressing model: " << device.get_info<info::device::address_bits>()
     << " bit" << std::endl;
  os << "OpenCL profile: " << device.get_info<info::device::profile>()
     << std::endl;
  os << "OpenCL version: " << device.get_info<info::device::opencl_c_version>()
     << std::endl;
  auto extensions = device.get_info<info::device::extensions>();
  os << "Supported extensions: " << std::endl;
  for (auto const &ext : extensions)
    os << "  * " << ext << std::endl;

  os << std::endl;
}

static void WarmUp(cl::sycl::queue &queue) {
  auto time = Utility::Benchmark([&]() {
    auto x = std::vector<cl::sycl::cl_int>(8);
    auto buf = cl::sycl::buffer{x};
    auto range1d = cl::sycl::range<1>{buf.get_count()};
    queue.submit([&](cl::sycl::handler &h) {
      auto access =
          buf.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<class WarmUpKernel>(
          range1d, [access](cl::sycl::id<1> id) { access[id[0]] = id[0]; });
    });
  });
  std::cout << "GPU warm up time: " << (time.count() / 1000) << " microseconds"
            << std::endl;
}

template <typename T, typename Competitor>
static void _RunCompetitor(std::vector<T> &vec, std::string_view description,
                           Competitor &&competitor) {
  auto time = Utility::Benchmark([&]() { competitor(vec); });
  std::cout << description << " time: " << (time.count() / 1000)
            << " microseconds" << std::endl;
}

template <typename T, typename Competitor, typename... Competitors>
static void
_Check(std::vector<T> const &previousResult, std::vector<T> const &vec,
       std::string_view previousDescription, std::string_view description,
       Competitor &&competitor, Competitors &&... competitors) {
  auto size = vec.size();
  auto result = vec;

  _RunCompetitor(result, description, competitor);

  for (auto i = size_t{0}; i < size; ++i) {
    if (result[i] == previousResult[i])
      continue;
    auto message = std::stringstream{};
    message << std::endl;
    message << "Results from \"" << previousDescription << "\"";
    message << " and \"" << description << "\"";
    message << " do not match at pos " << i << std::endl;
    message << previousDescription << " value: " << previousResult[i]
            << std::endl;
    message << description << " value: " << result[i] << std::endl;
    throw std::runtime_error{message.str()};
  }

  if constexpr (sizeof...(competitors) > 0)
    _Check(previousResult, vec, description,
           std::forward<Competitors>(competitors)...);
}

template <typename T, typename Competitor, typename... Competitors>
static void Check(std::vector<T> const &vec, std::string_view description,
                  Competitor &&competitor, Competitors &&... competitors) {
  static_assert(sizeof...(competitors) % 2 == 0);
  std::cout << "Running benchmark on vector of " << vec.size() << " elements..."
            << std::endl;

  auto result = vec;
  _RunCompetitor(result, description, competitor);
  if constexpr (sizeof...(competitors) > 0)
    _Check(result, vec, description, std::forward<Competitors>(competitors)...);

  std::cout << std::endl;
}

// Copy-paste https://stackoverflow.com/a/14880868/8099151
// The 'i' is for int, there is a log2 for double in stdclib
static unsigned int log2i(unsigned int x) {
#if defined(__GNUC__) || defined(__clang__)
  return 31 - __builtin_clz(x);
#else
  unsigned int log2Val = 0;
  // Count push off bits to right until 0
  // 101 => 10 => 1 => 0
  // which means hibit was 3rd bit, its value is 2^3
  while (x >>= 1)
    log2Val++; // div by 2 until find log2.  log_2(63)=5.97, so
  // take that as 5, (this is a traditional integer function!)
  // eg x=63 (111111), log2Val=5 (last one isn't counted by the while loop)
  return log2Val;
#endif
}
