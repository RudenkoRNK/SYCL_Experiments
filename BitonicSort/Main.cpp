#ifdef ESIMDVER
#include "BitonicSortESIMD.hpp"
#else
#include "BitonicSortHier.hpp"
#include "BitonicSortLocal.hpp"
#include "BitonicSortNaive.hpp"
#endif
#include "../Utils.hpp"

int main(int argc, char *argv[]) {
  auto pow = GetIntArgument(argc, argv, 12);
  auto size = static_cast<size_t>(1 << pow);

  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};
  PrintInfo(queue, std::cout);

  auto vec = GetRandomVector(size);

  WarmUp(queue);

  Check(
      vec, "CPU", [&](auto &v) { std::sort(v.begin(), v.end()); }
#ifdef ESIMDVER
      ,
      "GPU with ESIMD", [&](auto &v) { BitonicSortESIMD(queue, v); }
#else
      ,
      "GPU naive", [&](auto &v) { BitonicSortNaive(queue, v); },
      "GPU with local memory", [&](auto &v) { BitonicSortLocal(queue, v); },
      "GPU with PFWI", [&](auto &v) { BitonicSortHier(queue, v); }
#endif
  );
}
