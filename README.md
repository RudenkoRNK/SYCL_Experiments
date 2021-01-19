```
clang++ -O3 -fsycl -I $SYCL_EXPERIMENTS/Utility/Utility/include/ $SYCL_EXPERIMENTS/Main.cpp
./a.out 20
```

```
clang++ -O3 -fsycl -fsycl-explicit-simd -DESIMDVER -I $SYCL_EXPERIMENTS/Utility/Utility/include/ $SYCL_EXPERIMENTS/Main.cpp
SYCL_PROGRAM_COMPILE_OPTIONS="-vc-codegen" ./a.out 20
```
