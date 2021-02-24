#include <algorithm>
#include <limits>
#include <numeric>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

template <typename T> void Test(cl::sycl::queue &queue, std::vector<T> &vec) {
  using namespace cl::sycl;
  using namespace sycl::INTEL::gpu;
  auto constexpr SIMDSize = unsigned{16};
  auto constexpr TSize = sizeof(T);
  auto size = vec.size();
  auto *sharedData = malloc_shared<T>(size, queue);
  std::copy(vec.begin(), vec.end(), sharedData);

  queue.submit([&](handler &h) {
    h.parallel_for<class AddressSpaceTestKernel>(
        range<1>{size}, [=](id<1> id_) SYCL_ESIMD_KERNEL {
          auto id = static_cast<T>(id_[0]);
          auto fizz = static_cast<T>(size) + id;
          auto *ptr = (sharedData[id] % 2 == 0) ? &fizz : sharedData + id;
          sharedData[id] = reinterpret_cast<long>(ptr)%std::numeric_limits<T>::max();
        });
  });
  queue.wait();
  std::copy(sharedData, sharedData + size, vec.begin());
  free(sharedData, queue);
}

int main() {
  auto size = 16;
  auto vec = std::vector<int>(size);
  std::iota(vec.begin(), vec.end(), 0);

  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};

  Test(queue, vec);

  for (auto i = 0; i != size; ++i)
    std::cout << "Index: " << i << "; Result: " << vec[i] << std::endl;
}
