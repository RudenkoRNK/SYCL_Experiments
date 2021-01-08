#include <CL/sycl.hpp>

void Fill(std::vector<int> &vec, size_t workGroupSize, bool useLambda) {
  using namespace cl::sycl;
  using GlobalAccess =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  auto GPUSelector = gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};
  auto buf = buffer{vec};
  auto nWorkGroups = vec.size() / workGroupSize;

  queue.submit([&](handler &h) {
    auto global = buf.template get_access<access::mode::read_write>(h);
    auto lambda = [=](GlobalAccess global, group<1> g) {
      g.parallel_for_work_item([=](h_item<1> it) {
        global[it.get_global_id()[0]] = it.get_local_id()[0];
      });
    };

    if (useLambda) {
      h.parallel_for_work_group<class LambdaTestKernel>(
          range<1>{nWorkGroups}, range<1>{workGroupSize},
          [=](group<1> g) { lambda(global, g); });
    } else {
      h.parallel_for_work_group<class WithoutLambdaTestKernel>(
          range<1>{nWorkGroups}, range<1>{workGroupSize}, [=](group<1> g) {
            g.parallel_for_work_item([=](h_item<1> it) {
              global[it.get_global_id()[0]] = it.get_local_id()[0];
            });
          });
    }
  });
  queue.wait();
  buf.template get_access<cl::sycl::access::mode::read_write>();
  std::sort(vec.begin(), vec.end());
}

int main() {
  auto size = 16;
  auto workGroupSize = 4;
  auto nWorkGroups = size / workGroupSize;
  auto emptyVec = std::vector<int>(size);
  auto lambdaVec = emptyVec;
  auto withoutLambdaVec = emptyVec;

  Fill(lambdaVec, workGroupSize, true);
  Fill(withoutLambdaVec, workGroupSize, false);
  for (auto i = 0; i != size; ++i)
    std::cout << "Expected: " << i / nWorkGroups
              << "; Computed without lambda: " << withoutLambdaVec[i]
              << "; Computed with lambda: " << lambdaVec[i] << std::endl;
}
