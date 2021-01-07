#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <cassert>
#include <random>

//using namespace sycl::INTEL::gpu;
//using namespace cl::sycl;
//void kernel(accessor<int, 1, access::mode::read_write,
//                     access::target::global_buffer> &buf)
//    __attribute__((sycl_device)) {
//  simd<uint32_t, 32> offsets(0, 1);
//  simd<int, 32> v1(0, 1);
//
//  auto v0 = gather<int, 32>(buf.get_pointer(), offsets);
//
//  v0 = v0 + v1;
//
//  scatter<int, 32>(buf.get_pointer(), v0, offsets);
//}

static std::vector<cl::sycl::cl_int> GetRandomVector(size_t size) {
  auto rd = std::random_device{};
  auto gen = std::mt19937{static_cast<unsigned>(rd())};
  auto vec = std::vector<cl::sycl::cl_int>(size);
  auto rand = std::uniform_int_distribution<>(0, 1000);
  for (auto i = size_t{0}; i < size; ++i)
    vec[i] = rand(gen);
  return vec;
}

// Copy-paste https://stackoverflow.com/a/14880868/8099151
// The 'i' is for int, there is a log2 for double in stdclib
static unsigned int log2i(unsigned int x) {
  unsigned int log2Val = 0;
  // Count push off bits to right until 0
  // 101 => 10 => 1 => 0
  // which means hibit was 3rd bit, its value is 2^3
  while (x >>= 1)
    log2Val++; // div by 2 until find log2.  log_2(63)=5.97, so
  // take that as 5, (this is a traditional integer function!)
  // eg x=63 (111111), log2Val=5 (last one isn't counted by the while loop)
  return log2Val;
}

template <typename T> void BitonicSort(std::vector<T> &vec) {
  auto size = vec.size();
  auto nLargeSteps = static_cast<cl::sycl::cl_int>(log2i(size));
  auto buf = cl::sycl::buffer{vec};
  auto queue = cl::sycl::queue{};
  auto range1d = cl::sycl::range<1>{buf.get_count() / 2};

  for (auto i = cl::sycl::cl_int{0}; i != nLargeSteps; ++i)
    for (auto j = cl::sycl::cl_int{0}; j != i + 1; ++j)
      queue.submit([&](cl::sycl::handler &h) {
        auto access =
            buf.template get_access<cl::sycl::access::mode::read_write>(h);
        // Executing kernel
        h.parallel_for<class Bitonic>(range1d, [access, i,
                                                j](cl::sycl::id<1> id_) {
          auto id = id_[0];
          auto boxSize = 2 << (i - j);
          auto isSortPhase = static_cast<bool>(j);
          auto id0 = ((id / (boxSize / 2)) * boxSize) + (id % (boxSize / 2));
          auto id1 =
              (id0 + boxSize / 2) +
              (isSortPhase - 1) * ((2 * (id0 % boxSize)) - boxSize / 2 + 1);
          if (access[id0] > access[id1])
            std::swap(access[id0], access[id1]);
        });
      });
}

int main() {
  auto size = size_t{4096};
  auto vec = GetRandomVector(size);
  auto sortedVec = vec;

  std::sort(sortedVec.begin(), sortedVec.end());
  BitonicSort(vec);

  for (auto i = size_t{0}; i < size; ++i) {
    if (vec[i] != sortedVec[i]) {
      std::cout << "Failed to sort at pos " << i << std::endl;
      std::cout << "Parallel: " << vec[i] << ". Serial: " << sortedVec[i] << "."
                << std::endl;
      return 1;
    }
  }
  std::cout << "Sorted successfully!" << std::endl;
}
