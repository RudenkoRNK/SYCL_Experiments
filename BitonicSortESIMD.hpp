#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

#include <algorithm>

#include "Utils.hpp"

template <typename T>
static void BitonicSortESIMD(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto constexpr SIMDSize = unsigned{32};
  auto constexpr TSize = sizeof(T);
  auto size = vec.size();
  if (size < SIMDSize) {
    std::sort(vec.begin(), vec.end());
    return;
  }
  auto original = vec;

  auto nLargeSteps = static_cast<cl_int>(log2i(size));
  auto buf = buffer{vec};
  for (auto i = 0; i != nLargeSteps; ++i)
    for (auto j = 0; j != i + 1; ++j)
      queue.submit([&](handler &h) {
        auto access = buf.template get_access<access::mode::read_write>(h);
        auto ptr = access.get_pointer().get();
        // Executing kernel
        h.parallel_for<class BitonicESIMDKernel>(
            range<1>{size / SIMDSize / 2},
            [ptr, i, j](id<1> id_) SYCL_ESIMD_KERNEL {
              auto id = id_[0] * SIMDSize;
              auto boxSize = 2 << (i - j);
              auto halfBoxSize = boxSize / 2;
              auto bigBoxSize = 2 << i;
              auto isSortPhase = static_cast<bool>(j);
              auto ids = simd<unsigned, SIMDSize>(id, 1);
              auto id0s = ((ids / halfBoxSize) * boxSize) +
                          (ids - (ids / halfBoxSize) * halfBoxSize);
              auto id1s = id0s + halfBoxSize;
              auto offsets0 = id0s * TSize;
              auto offsets1 = id1s * TSize;
              auto data0 = gather<T, SIMDSize>(ptr, offsets0);
              auto data1 = gather<T, SIMDSize>(ptr, offsets1);
              auto tmp = data0;
              auto conds = id0s / bigBoxSize;
              conds = conds - (conds / 2) * 2;
              data0.merge(data1, conds == 1);
              data1.merge(tmp, conds == 0);
              scatter<T, SIMDSize>(ptr, data0, offsets0);
              scatter<T, SIMDSize>(ptr, data1, offsets1);
            });
      });
  queue.wait();
  buf.template get_access<access::mode::read_write>();

  for (auto i = 0; i != std::min(size_t{8}, size); ++i)
    std::cout << "Original: " << original[i] << "; sorted: " << vec[i]
              << std::endl;
}
