#include <CL/sycl.hpp>

void FillLambda(cl::sycl::queue &queue, std::vector<int> &vec,
                size_t workGroupSize) {
  using namespace cl::sycl;
  auto buf = buffer{vec};
  auto nWorkGroups = vec.size() / workGroupSize;

  queue.submit([&](handler &h) {
    auto global = buf.template get_access<access::mode::read_write>(h);
    auto lambda = [=](group<1> g) {
      g.parallel_for_work_item(range<1>{4}, [=](h_item<1> it) {
        global[it.get_global_id()[0]] = it.get_local_id()[0];
      });
    };

    h.parallel_for_work_group<class LambdaTestKernel>(
        range<1>{nWorkGroups}, range<1>{workGroupSize},
        [=](group<1> g) { lambda(g); });
  });

  queue.wait();
  buf.template get_access<cl::sycl::access::mode::read_write>();
}

void FillWithoutLambda(cl::sycl::queue &queue, std::vector<int> &vec,
                       size_t workGroupSize) {
  using namespace cl::sycl;
  auto buf = buffer{vec};
  auto nWorkGroups = vec.size() / workGroupSize;

  queue.submit([&](handler &h) {
    auto global = buf.template get_access<access::mode::read_write>(h);
    h.parallel_for_work_group<class WithoutLambdaTestKernel>(
        range<1>{nWorkGroups}, range<1>{workGroupSize}, [=](group<1> g) {
          g.parallel_for_work_item([=](h_item<1> it) {
            global[it.get_global_id()[0]] = it.get_local_id()[0];
          });
        });
  });
  queue.wait();
  buf.template get_access<cl::sycl::access::mode::read_write>();
}

int main() {
  auto size = 16;
  auto workGroupSize = 4;
  auto nWorkGroups = size / workGroupSize;
  auto emptyVec = std::vector<int>(size);
  auto lambdaVec = emptyVec;
  auto withoutLambdaVec = emptyVec;

  auto GPUSelector = cl::sycl::gpu_selector{};
  auto queue = cl::sycl::queue{GPUSelector};

  FillLambda(queue, lambdaVec, workGroupSize);
  FillWithoutLambda(queue, withoutLambdaVec, workGroupSize);

  std::sort(lambdaVec.begin(), lambdaVec.end());
  std::sort(withoutLambdaVec.begin(), withoutLambdaVec.end());
  for (auto i = 0; i != size; ++i)
    std::cout << "Expected: " << i / nWorkGroups
              << "; Computed without lambda: " << withoutLambdaVec[i]
              << "; Computed with lambda: " << lambdaVec[i] << std::endl;
}
