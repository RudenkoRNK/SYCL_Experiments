#include <numeric>
#include <algorithm>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  auto constexpr size = size_t{16};
  auto constexpr arrSize = size_t{67108864}; // 2^25

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto *shared = malloc_shared<int>(size, q);
  std::fill(shared, shared + size, 0);

  q.submit([&](handler &h) {
    h.parallel_for<class SimplestKernel>(
        range<1>{size}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            unsigned x[arrSize];
            for (auto i = size_t{0}; i != arrSize; ++i)
                x[i] = (i + id[0]) % arrSize;
            auto idx = unsigned{0};
            for (auto i = 0; i != arrSize; ++i)
                idx = x[idx];
            // At this point idx always equals to 0
            idx += id[0];
            shared[id[0]] = idx;
        });
  });
  q.wait();

  for (auto i = 0; i != size; ++i)
    std::cout << "Index: " << i << "; Result: " << shared[i] << std::endl;
  free(shared, q);
}
