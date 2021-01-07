#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

#include "Utils.hpp"

template <typename T>
static void BitonicSortNaive(cl::sycl::queue &queue, std::vector<T> &vec) {
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
        h.parallel_for<class BitonicNaive>(range1d, [access, i,
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
