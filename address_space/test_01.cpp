#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  auto size = size_t{128};
  auto vec = std::vector<int>(size);

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto buf = buffer{vec};

  q.submit([&](handler &h) {
    auto access = buf.template get_access<access::mode::read_write>(h);
    h.parallel_for<class SimplestKernel>(
        range<1>{size},
        [=](id<1> id) SYCL_ESIMD_KERNEL { access[id[0]] = id[0]; });
  });
  q.wait();
  buf.template get_access<access::mode::read_write>();

  for (auto i = 0; i != size; ++i)
    std::cout << "Index: " << i << "; Result: " << vec[i] << std::endl;
}
