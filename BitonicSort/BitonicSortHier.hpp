#pragma once

#include <CL/sycl.hpp>

#include "../Utils.hpp"

template <typename T>
static void BitonicSortHier(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  auto SIMDSize = unsigned{32};
  if (vec.size() <= SIMDSize)
    SIMDSize = vec.size() / 2;

  auto size = vec.size();
  auto nLargeSteps = static_cast<cl_int>(log2i(size));
  auto buf = buffer{vec};
  auto range1d = range<1>{buf.get_count() / 2};

  for (auto i = 0; i != nLargeSteps; ++i)
    for (auto j = 0; j != i + 1; ++j)
      queue.submit([&](handler &h) {
        auto access = buf.template get_access<access::mode::read_write>(h);
        h.parallel_for_work_group<class BitonicHierKernel>(
            range<1>{size / SIMDSize / 2}, range<1>{SIMDSize}, [=](group<1> g) {
              g.parallel_for_work_item([=](h_item<1> it) {
                auto id = it.get_global_id(0);
                auto boxSize = 2 << (i - j);
                auto isSortPhase = static_cast<bool>(j);
                auto id0 =
                    ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
                auto id1 = id0 + boxSize / 2;
                auto bigBoxSize = 2 << i;
                if (((id0 / bigBoxSize) % 2) == (access[id0] < access[id1]))
                  std::swap(access[id0], access[id1]);
              });
            });
      });
  queue.wait();
  buf.template get_access<access::mode::read_write>();
}
