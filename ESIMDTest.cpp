#include <algorithm>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

template <typename T>
void FillESIMD(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto constexpr SIMDSize = unsigned{8};
  auto constexpr TSize = sizeof(T);

  auto buf = buffer{vec};
  queue.submit([&](handler &h) {
    auto access = buf.template get_access<access::mode::read_write>(h);
    // Executing kernel
    h.parallel_for<class ESIMDTestKernel>(
        range<1>{vec.size() / SIMDSize}, [=](id<1> id_) SYCL_ESIMD_KERNEL {
          auto id = id_[0] * SIMDSize;
          // Fill vector at indices [id*SIMDSize, (id+1)*SIMDSize)
          // data == {id, id+1, id+2, ..., id+SIMDSize-1}
          auto data = simd<T, SIMDSize>(id, 1);
          auto offsets = simd<uint32_t, SIMDSize>(id, 1) * TSize;
          scatter<T, SIMDSize>(access, data, offsets);
        });
  });
  queue.wait();
  buf.template get_access<access::mode::read_write>();
}

int main() {
  auto size = 16;
  auto vec = std::vector<int>(size);

  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};

  FillESIMD(queue, vec);

  for (auto i = 0; i != size; ++i)
    std::cout << "Expected: " << i << "; Computed with ESIMD: " << vec[i]
              << std::endl;
}
