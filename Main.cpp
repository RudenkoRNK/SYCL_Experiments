#include "BitonicSortLocal.hpp"
#include "BitonicSortNaive.hpp"
#include "Utils.hpp"

int main(int argc, char *argv[]) {
  auto pow = GetIntArgument(argc, argv, 12);
  auto size = static_cast<size_t>(1 << pow);

  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};
  PrintInfo(queue, std::cout);

  auto vec = GetRandomVector(size);

  WarmUp(queue);

  Check(
      vec, "CPU", [&](auto &v) { std::sort(v.begin(), v.end()); }, "GPU naive",
      [&](auto &v) { BitonicSortNaive(queue, v); }, "GPU with local memory",
      [&](auto &v) { BitonicSortLocal(queue, v); });
}
