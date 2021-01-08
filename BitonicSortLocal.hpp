#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

#include <algorithm>

#include "Utils.hpp"

template <typename T>
static void BitonicSortLocal(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  using LocalAccess =
      accessor<T, 1, access::mode::read_write, access::target::local>;
  using GlobalAccess =
      accessor<T, 1, access::mode::read_write, access::target::global_buffer>;
  auto size = vec.size();
  if (size <= 1)
    return;

  auto nLargeSteps = log2i(size);
  auto buf = buffer{vec};

  auto workGroupSizeRaw =
      queue.get_device().get_info<info::device::max_work_group_size>();
  auto localMem = queue.get_device().get_info<info::device::local_mem_size>();
  auto memPerWorkItem = localMem / workGroupSizeRaw;
  auto nElementsPerWorkItem = ClosestPowerOf2(
      memPerWorkItem / sizeof(T) - /*reserve for other variables*/ 16);
  assert(nElementsPerWorkItem >= 2);
  if (size < nElementsPerWorkItem)
    nElementsPerWorkItem = 2;
  // Actually, multiple elements per work item slow down perfomance, but it will
  // be helpful later with SIMD
  nElementsPerWorkItem = 2;

  auto nOpsPerWorkItem = nElementsPerWorkItem / 2;
  auto WGSize = std::min(static_cast<size_t>(ClosestPowerOf2(workGroupSizeRaw)),
                         size / nElementsPerWorkItem);
  auto WGElements = WGSize * nElementsPerWorkItem;
  auto nWorkGroups = size / WGElements;
  auto nWGLargeSteps = log2i(WGElements);
  std::cout << "size " << size << std::endl;
  std::cout << "nWorkGroups " << nWorkGroups << std::endl;
  std::cout << "nWGLargeSteps " << nWGLargeSteps << std::endl;
  std::cout << "WGElements " << WGElements << std::endl;
  std::cout << "WGSize " << WGSize << std::endl;
  std::cout << "nElementsPerWorkItem " << nElementsPerWorkItem << std::endl;

  auto cmp = vec;

  queue.submit([&](handler &h) {
    auto global = buf.template get_access<access::mode::read_write>(h);
    auto local = LocalAccess(range<1>{WGElements}, h);
    auto localBitonicSort = [=](group<1> g, int startIndex,
                                int firstLargeStep = 0) {
      g.parallel_for_work_item([=](h_item<1> it) {
        auto localStart = it.get_local_id()[0] * nElementsPerWorkItem;
        for (auto i = 0; i != nElementsPerWorkItem; ++i) {
          auto localIndex = localStart + i;
          local[localIndex] = global[startIndex + localIndex];
        }
      });

      for (auto i = firstLargeStep; i != nWGLargeSteps; ++i)
        for (auto j = 0; j != i + 1; ++j)
          for (auto el = 0; el != nOpsPerWorkItem; ++el) {
            auto start = el * WGSize;
            g.parallel_for_work_item([=](h_item<1> it) {
              auto id = start + it.get_local_id()[0];
              auto boxSize = 2 << (i - j);
              auto id0 =
                  ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
              auto id1 = id0 + boxSize / 2;
              auto bigBoxSize = 2 << i;
              if (((id0 / bigBoxSize) % 2) == (local[id0] < local[id1]))
                std::swap(local[id0], local[id1]);
            });
          }

      g.parallel_for_work_item([=](h_item<1> it) {
        auto localStart = it.get_local_id()[0] * nElementsPerWorkItem;
        for (auto i = 0; i != nElementsPerWorkItem; ++i) {
          auto localIndex = localStart + i;
          global[startIndex + localIndex] = local[localIndex];
        }
      });
    };

    h.parallel_for_work_group<class BitonicSortLocal>(
        range<1>{nWorkGroups}, range<1>{WGSize},
        [=](group<1> g) { localBitonicSort(g, 0); });
  });
  queue.wait();

  buf.template get_access<cl::sycl::access::mode::read_write>();
  for (auto i = 0; i != std::min(size, size_t{8}); ++i)
    std::cout << "Orig: " << cmp[i] << "; Sorted: " << vec[i] << std::endl;
}
