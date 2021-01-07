#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <Utility/Misc.hpp>
#include <cassert>
#include <random>
#include <string>

#include "BitonicSortNaive.hpp"
#include "Utils.hpp"

int main(int argc, char *argv[]) {
  auto pow = GetIntArgument(argc, argv, 12);
  auto size = static_cast<size_t>(1 << pow);

  auto queue = GetDefaultQueue();
  auto vec = GetRandomVector(size);

  Check(
      vec, "CPU", [&](auto &v) { std::sort(v.begin(), v.end()); }, "GPU",
      [&](auto &v) { BitonicSortNaive(queue, v); });
}
