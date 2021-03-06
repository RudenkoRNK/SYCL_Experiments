#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto size = size_t{16};
  auto constexpr SIMDSize = unsigned{16};

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto vec = std::vector<int>(size);
  auto buf = buffer{vec};

  q.submit([&](handler &h) {
    auto access = buf.template get_access<access::mode::read_write>(h);
    h.parallel_for<class SimplestKernel>(
        range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
          auto data = simd<int, SIMDSize>(id * SIMDSize, 1);
          block_store<int, SIMDSize>(access, id * SIMDSize, data);
        });
  });
  buf.template get_access<access::mode::read_write>();

  for (auto i = 0; i != size; ++i)
    std::cout << "Index: " << i << "; Result: " << vec[i] << std::endl;
}
