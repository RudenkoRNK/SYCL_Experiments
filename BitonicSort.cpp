#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <Utility/Misc.hpp>
#include <cassert>
#include <random>
#include <string>

void PrintInfo(cl::sycl::queue const &queue, std::ostream &os) {
  auto device = queue.get_device();
  os << device.get_info<cl::sycl::info::device::name>() << "\n";
  os << "Driver version: "
     << device.get_info<cl::sycl::info::device::driver_version>() << "\n";
  os << device.get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
}

std::vector<cl::sycl::cl_int> GetRandomVector(size_t size) {
  auto rd = std::random_device{};
  auto gen = std::mt19937{static_cast<unsigned>(rd())};
  auto vec = std::vector<cl::sycl::cl_int>(size);
  auto rand = std::uniform_int_distribution<>();
  for (auto i = size_t{0}; i < size; ++i)
    vec[i] = rand(gen);
  return vec;
}

// Copy-paste https://stackoverflow.com/a/14880868/8099151
// The 'i' is for int, there is a log2 for double in stdclib
unsigned int log2i(unsigned int x) {
  unsigned int log2Val = 0;
  // Count push off bits to right until 0
  // 101 => 10 => 1 => 0
  // which means hibit was 3rd bit, its value is 2^3
  while (x >>= 1)
    log2Val++; // div by 2 until find log2.  log_2(63)=5.97, so
  // take that as 5, (this is a traditional integer function!)
  // eg x=63 (111111), log2Val=5 (last one isn't counted by the while loop)
  return log2Val;
}

template <typename T>
void BitonicSort(cl::sycl::queue &queue, std::vector<T> &vec) {
  auto size = vec.size();
  auto nLargeSteps = static_cast<cl::sycl::cl_int>(log2i(size));
  auto buf = cl::sycl::buffer{vec};
  auto range1d = cl::sycl::range<1>{buf.get_count() / 2};

  for (auto i = cl::sycl::cl_int{0}; i != nLargeSteps; ++i)
    for (auto j = cl::sycl::cl_int{0}; j != i + 1; ++j)
      queue.submit([&](cl::sycl::handler &h) {
        auto access =
            buf.template get_access<cl::sycl::access::mode::read_write>(h);
        // Executing kernel
        h.parallel_for<class Bitonic>(range1d, [access, i,
                                                j](cl::sycl::id<1> id_) {
          auto id = id_[0];
          auto boxSize = 2 << (i - j);
          auto isSortPhase = static_cast<bool>(j);
          auto id0 = ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
          auto id1 =
              (id0 + boxSize / 2) +
              (isSortPhase - 1) * ((2 * (id0 % boxSize)) - boxSize / 2 + 1);
          if (access[id0] > access[id1])
            std::swap(access[id0], access[id1]);
        });
      });
}

int main(int argc, char *argv[]) {
  auto size = size_t{4096};
  if (argc > 1) {
    auto arg = std::string(argv[1]);
    auto pow = std::stoi(arg);
    size = 1 << pow;
  }

  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};
  PrintInfo(queue, std::cout);
  auto antiDCE = 0;
  auto vec = GetRandomVector(size);
  auto gpuVec = vec;
  auto cpuVec = vec;

  auto gpu_time = Utility::Benchmark([&]() {
    BitonicSort(queue, gpuVec);
    antiDCE += gpuVec[size - 1];
  });
  auto cpu_time = Utility::Benchmark([&]() {
    std::sort(cpuVec.begin(), cpuVec.end());
    antiDCE += cpuVec[size - 1];
  });

  for (auto i = size_t{0}; i < size; ++i) {
    if (gpuVec[i] != cpuVec[i]) {
      std::cout << "Failed to sort at pos " << i << std::endl;
      std::cout << "Parallel: " << gpuVec[i] << ". Serial: " << cpuVec[i] << "."
                << std::endl;
      return 1;
    }
  }
  std::cout << "Successfully sorted " << size << " elements!" << std::endl;
  std::cout << "GPU time: " << (gpu_time.count() / 1000) << " microseconds"
            << std::endl;
  std::cout << "CPU time: " << (cpu_time.count() / 1000) << " microseconds"
            << std::endl;

  std::cout << "Anti-DCE: " << antiDCE << std::endl;
}
