#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto size = size_t{128};
  auto constexpr SIMDSize = unsigned{16};

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto *shared = malloc_shared<int>(size, q);

  q.submit([&](handler &h) {
    h.parallel_for<class SimplestKernel>(
        range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
          auto offsets = simd<unsigned, SIMDSize>(id * SIMDSize * sizeof(int),
                                                  sizeof(int));
          auto data = simd<int, SIMDSize>(id * SIMDSize, 1);
          scatter<int, SIMDSize>(shared, data, offsets);
        });
  });
  q.wait();

  for (auto i = 0; i != size; ++i)
    std::cout << "Index: " << i << "; Result: " << shared[i] << std::endl;
  free(shared, q);
}
