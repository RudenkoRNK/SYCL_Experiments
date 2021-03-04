#pragma once

#include <CL/sycl.hpp>

#include <algorithm>

#include "../utils.hpp"

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

  auto buf = buffer{vec};

  // Determine how much elements we can load into local memory
  // This also determines how much elements one work item will handle
  auto workGroupSizeRaw =
      queue.get_device().get_info<info::device::max_work_group_size>();
  auto localMem = queue.get_device().get_info<info::device::local_mem_size>();
  auto memPerWorkItem = localMem / workGroupSizeRaw;
  auto nElementsPerWorkItem = ClosestPowerOf2(
      memPerWorkItem / sizeof(T) - /*reserve for other variables*/ 16);
  assert(nElementsPerWorkItem >= 2);
  // Corner case when array is smaller than one work item can handle
  if (size < nElementsPerWorkItem)
    nElementsPerWorkItem = 2;

  // Bitonic sort have "large steps" and "small steps"
  // Large steps are matched to big boxes of green and blue color
  // Small steps are matched to small boxes of orange color
  // See https://en.wikipedia.org/wiki/Bitonic_sorter#How_the_algorithm_works
  auto nLargeSteps = log2i(size);
  auto nOpsPerWorkItem =
      nElementsPerWorkItem / /*number of arguments of swap operation*/ 2;
  auto WGSize = ClosestPowerOf2(workGroupSizeRaw);
  // Corner case when total work items needed
  // is smaller than one work group have
  WGSize = std::min(WGSize,
                    static_cast<decltype(WGSize)>(size / nElementsPerWorkItem));
  auto WGElements = WGSize * nElementsPerWorkItem;
  auto nWorkGroups = size / WGElements;
  // Since one work group can handle no more than WGElements it has its own
  // internal large steps limit
  auto nWGLargeSteps = log2i(WGElements);

#if 0
  std::cout << "size " << size << std::endl;
  std::cout << "nLargeSteps " << nLargeSteps << std::endl;
  std::cout << "nWorkGroups " << nWorkGroups << std::endl;
  std::cout << "nWGLargeSteps " << nWGLargeSteps << std::endl;
  std::cout << "WGElements " << WGElements << std::endl;
  std::cout << "WGSize " << WGSize << std::endl;
  std::cout << "nElementsPerWorkItem " << nElementsPerWorkItem << std::endl;
#endif

  // The algorithm is divided into two parts:
  // "Local" part divides the whole range into chunks of WGElements size
  // and sorts each of them individually with use of local memory
  // "Global" part perfoms sorting of higher order, since
  // it works on larger chuncks which do not fit in local memory
  auto LocalSort = [&](int iLargeStep = 0) {
    assert(iLargeStep == 0 || iLargeStep >= nWGLargeSteps);
    auto firstLargeStep = iLargeStep == 0 ? 0 : nWGLargeSteps - 1;
    queue.submit([&](handler &h) {
      auto global = buf.template get_access<access::mode::read_write>(h);
      auto local = LocalAccess(range<1>{WGElements}, h);
      h.parallel_for_work_group<class BitonicSortLocalKernel>(
          range<1>{nWorkGroups}, range<1>{WGSize}, [=](group<1> g) {
            auto startIndex = g.get_id(0) * WGElements;
            // Load items from global memory
            g.parallel_for_work_item([=](h_item<1> it) {
              auto localStart = it.get_local_id()[0] * nElementsPerWorkItem;
              for (auto i = 0; i != nElementsPerWorkItem; ++i) {
                auto localIndex = localStart + i;
                local[localIndex] = global[startIndex + localIndex];
              }
            });

            // Sort
            for (auto i = firstLargeStep; i != nWGLargeSteps; ++i) {
              auto bigBoxSize = (iLargeStep == 0) ? 2 << i : 2 << iLargeStep;
              for (auto j = 0; j != i + 1; ++j)
                g.parallel_for_work_item([=](h_item<1> it) {
                  auto start = it.get_local_id()[0] * nOpsPerWorkItem;
                  auto boxSize = 2 << (i - j);
                  for (auto el = 0; el != nOpsPerWorkItem; ++el) {
                    auto id = start + el;
                    auto id0 =
                        ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
                    auto id1 = id0 + boxSize / 2;
                    if ((((id0 + startIndex) / bigBoxSize) % 2) ==
                        (local[id0] < local[id1]))
                      std::swap(local[id0], local[id1]);
                  }
                });
            }

            // Load items to global memory
            g.parallel_for_work_item([=](h_item<1> it) {
              auto localStart = it.get_local_id()[0] * nElementsPerWorkItem;
              for (auto i = 0; i != nElementsPerWorkItem; ++i) {
                auto localIndex = localStart + i;
                global[startIndex + localIndex] = local[localIndex];
              }
            });
          });
    });
  };
  auto GlobalSort = [&](int iLargeStep) {
    auto lastSmallStep = iLargeStep - log2i(WGElements);
    auto bigBoxSize = 2 << iLargeStep;
    for (auto j = 0; j != lastSmallStep + 1; ++j)
      queue.submit([&](cl::sycl::handler &h) {
        auto global =
            buf.template get_access<cl::sycl::access::mode::read_write>(h);
        // Executing kernel
        h.parallel_for<class BitonicPartGlobalKernel>(
            range<1>{size / 2}, [=](cl::sycl::id<1> id_) {
              auto id = id_[0];
              auto boxSize = 2 << (iLargeStep - j);
              auto isSortPhase = static_cast<bool>(j);
              auto id0 =
                  ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
              auto id1 = id0 + boxSize / 2;
              if (((id0 / bigBoxSize) % 2) == (global[id0] < global[id1]))
                std::swap(global[id0], global[id1]);
            });
      });
  };

  LocalSort();
  for (auto i = nWGLargeSteps; i != nLargeSteps; ++i) {
    GlobalSort(i);
    LocalSort(i);
  }

  queue.wait();
  buf.template get_access<cl::sycl::access::mode::read_write>();
}
