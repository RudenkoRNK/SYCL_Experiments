#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  auto size = size_t{128};

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto *shared = malloc_shared<int>(size, q);

  q.submit([&](handler &h) {
    h.parallel_for<class SimplestKernel>(
        range<1>{size}, [=](id<1> id) SYCL_ESIMD_KERNEL { shared[id[0]] = id[0]; });
  });
  q.wait();

  for (auto i = 0; i != size; ++i)
    std::cout << "Index: " << i << "; Result: " << shared[i] << std::endl;
  free(shared, q);
}
