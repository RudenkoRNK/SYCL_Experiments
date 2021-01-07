#pragma once

#include <CL/sycl.hpp>

#include <cassert>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

static void PrintInfo(cl::sycl::queue const &queue, std::ostream &os) {
  auto device = queue.get_device();
  os << device.get_info<cl::sycl::info::device::name>() << "\n";
  os << "Driver version: "
     << device.get_info<cl::sycl::info::device::driver_version>() << "\n";
  os << device.get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
}

static int GetIntArgument(int argc, char *argv[], int defaultValue = 0) {
  if (argc <= 1)
    return defaultValue;
  auto arg = std::string(argv[1]);
  return std::stoi(arg);
}

static cl::sycl::queue GetDefaultQueue() {
  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};
  PrintInfo(queue, std::cout);
  return queue;
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
    message << "Results from " << previousDescription;
    message << "and " << description;
    message << "do not match at pos " << i << std::endl;
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
  auto result = vec;
  _RunCompetitor(result, description, competitor);
  if constexpr (sizeof...(competitors) > 0)
    _Check(result, vec, description, std::forward<Competitors>(competitors)...);
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
