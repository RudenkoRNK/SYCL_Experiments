#include <CL/sycl.hpp>
#include <cassert>
#include <random>
#include <sstream>
#include <string>

static void PrintInfo(cl::sycl::queue const &queue, std::ostream &os) {
  auto device = queue.get_device();
  os << device.get_info<cl::sycl::info::device::name>() << "\n";
  os << "Driver version: "
     << device.get_info<cl::sycl::info::device::driver_version>() << "\n";
  os << device.get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
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

template <typename T, typename CPUFunction, typename GPUFunction>
static void Check(std::vector<T> const &vec, CPUFunction &&CPU,
                  GPUFunction &&GPU) {
  auto size = vec.size();
  auto antiDCE = 0;
  auto gpuVec = vec;
  auto cpuVec = vec;

  auto gpu_time = Utility::Benchmark([&]() {
    GPU(gpuVec);
    antiDCE += gpuVec[size - 1];
  });
  auto cpu_time = Utility::Benchmark([&]() {
    CPU(cpuVec);
    antiDCE += cpuVec[size - 1];
  });

  for (auto i = size_t{0}; i < size; ++i) {
    if (gpuVec[i] == cpuVec[i])
      continue;
    auto message = std::stringstream{};
    message << "Results from CPU and GPU do not match at pos " << i
            << std::endl;
    message << "GPU value: " << gpuVec[i] << std::endl;
    message << "CPU value : " << cpuVec[i] << std::endl;
    throw std::runtime_error{message.str()};
  }
  std::cout << "Successfully computed " << size << " elements!" << std::endl;
  std::cout << "GPU time: " << (gpu_time.count() / 1000) << " microseconds"
            << std::endl;
  std::cout << "CPU time: " << (cpu_time.count() / 1000) << " microseconds"
            << std::endl;

  std::cout << "Anti-DCE: " << antiDCE << std::endl;
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
