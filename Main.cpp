#include "BitonicSortLocal.hpp"
#include "BitonicSortNaive.hpp"
#include "Utils.hpp"

int main(int argc, char *argv[]) {
  auto pow = GetIntArgument(argc, argv, 12);
  auto size = static_cast<size_t>(1 << pow);

  auto queue = GetDefaultQueue();
  auto vec = GetRandomVector(size);

  Check(
      vec, "Warming up GPU", [&](auto &v) { BitonicSortNaive(queue, v); },
      "CPU", [&](auto &v) { std::sort(v.begin(), v.end()); }, "GPU naive",
      [&](auto &v) { BitonicSortNaive(queue, v); }, "GPU with local memory",
      [&](auto &v) { BitonicSortLocal(queue, v); });
}
