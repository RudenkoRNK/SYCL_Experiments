#pragma once

#include <CL/sycl.hpp>

#include "Utils.hpp"

template <typename T>
static void BitonicSortNaive(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  auto size = vec.size();
  auto nLargeSteps = static_cast<cl_int>(log2i(size));
  auto buf = buffer{vec};
  auto range1d = range<1>{buf.get_count() / 2};

  for (auto i = 0; i != nLargeSteps; ++i)
    for (auto j = 0; j != i + 1; ++j)
      queue.submit([&](handler &h) {
        auto access = buf.template get_access<access::mode::read_write>(h);
        // Executing kernel
        h.parallel_for<class BitonicNaiveKernel>(range1d, [access, i,
                                                           j](id<1> id_) {
          auto id = id_[0];
          auto boxSize = 2 << (i - j);
          auto id0 = ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
#ifdef ALTERNATIVE
          // See
          // https://en.wikipedia.org/wiki/Bitonic_sorter#Alternative_representation
          auto isSortPhase = static_cast<bool>(j);
          auto id1 =
              (id0 + boxSize / 2) +
              (isSortPhase - 1) * ((2 * (id0 % boxSize)) - boxSize / 2 + 1);
          if (access[id0] > access[id1])
            std::swap(access[id0], access[id1]);
#else  // faster
          auto id1 = id0 + boxSize / 2;
          auto bigBoxSize = 2 << i;
          if (((id0 / bigBoxSize) % 2) == (access[id0] < access[id1]))
            std::swap(access[id0], access[id1]);
#endif // ALTERNATIVE
        });
      });
  queue.wait();
  buf.template get_access<access::mode::read_write>();
}
