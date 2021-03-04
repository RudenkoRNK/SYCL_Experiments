#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

#include <algorithm>

#include "../utils.hpp"

template <typename T>
static void BitonicSortESIMD(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto constexpr SIMDSize = unsigned{16};
  auto constexpr TSize = sizeof(T);
  auto size = vec.size();
  if (size <= SIMDSize) {
    std::sort(vec.begin(), vec.end());
    return;
  }
  auto original = vec;

  auto nLargeSteps = log2i(size);
  auto buf = buffer{vec};
  for (auto i = 0; i != nLargeSteps; ++i)
    for (auto j = 0; j != i + 1; ++j)
      queue.submit([&](handler &h) {
        auto access = buf.template get_access<access::mode::read_write>(h);
        // Executing kernel
        h.parallel_for<class BitonicESIMDKernel>(
            range<1>{size / SIMDSize / 2}, [=](id<1> id_) SYCL_ESIMD_KERNEL {
              auto id = id_[0] * SIMDSize;
              auto boxSize = 2 << (i - j);
              auto halfBoxSize = boxSize / 2;
              auto bigBoxSize = 2 << i;
              auto isSortPhase = static_cast<bool>(j);
              auto ids = simd<unsigned, SIMDSize>(id, 1);
              auto id0s = ((ids / halfBoxSize) * boxSize) +
                          (ids - (ids / halfBoxSize) * halfBoxSize);
              auto id1s = id0s + halfBoxSize;
              auto data0 = gather<T, SIMDSize>(access, id0s);
              auto data1 = gather<T, SIMDSize>(access, id1s);
              auto id0sparity = id0s / bigBoxSize;
              id0sparity = id0sparity - (id0sparity / 2) * 2;
              auto conds = (id0sparity == 0) == (data0 > data1);
              scatter<T, SIMDSize>(access, data1, id0s, 0, conds);
              scatter<T, SIMDSize>(access, data0, id1s, 0, conds);
            });
      });
  queue.wait();
  buf.template get_access<access::mode::read_write>();
}
