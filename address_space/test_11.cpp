#include <numeric>
#include <algorithm>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto constexpr size = size_t{128};
  auto constexpr SIMDSize = size_t{16};

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto *shared = malloc_shared<int>(size, q);
  std::fill(shared, shared + size, 0);

  q.submit([&](handler &h) {
    h.parallel_for<class SimplestKernel>(
        range<1>{size/SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            unsigned x[size];
            for (auto i = size_t{0}; i != size; ++i)
                x[i] = (i + id[0]) % size;
            auto idx = unsigned{0};
            for (auto i = 0; i != size; ++i)
                idx = x[idx];
            auto inds = simd<unsigned, SIMDSize>(id * SIMDSize, 1);
            inds = (inds * 3 + 17)*idx + inds;
            auto conds = inds;
            auto data = simd<int, SIMDSize>(id * SIMDSize, 1);
            scatter<int, SIMDSize>(shared, data, inds * sizeof(int), conds);
        });
  });
  q.wait();

  auto maxLength = size_t{31};
  auto step = size/maxLength + 1;
  for (auto i = 0; i < size; i+=step)
    std::cout << "Index: " << i << "; Result: " << shared[i] << std::endl;
  free(shared, q);
}
