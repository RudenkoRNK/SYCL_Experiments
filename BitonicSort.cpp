#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <Utility/Misc.hpp>
#include <cassert>
#include <random>
#include <string>

#include "Utils.hpp"

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
