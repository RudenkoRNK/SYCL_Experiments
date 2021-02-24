#include <numeric>
#include <algorithm>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  auto constexpr size = size_t{512};

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto *shared = malloc_shared<int>(size, q);
  std::fill(shared, shared + size, 0);

  q.submit([&](handler &h) {
    h.parallel_for<class SimplestKernel>(
        range<1>{size}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            int x = shared[id[0]] + 1;
            auto *ptr = x > 0 ? shared : &x;
            auto idx = *ptr + id[0];
            shared[id[0]] = idx;
        });
  });
  q.wait();

  auto maxLength = size_t{32};
  auto step = size/maxLength + 1;
  for (auto i = 0; i < size; i+=step)
    std::cout << "Index: " << i << "; Result: " << shared[i] << std::endl;
  free(shared, q);
}
