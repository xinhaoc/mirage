/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "epilogue.cuh"
#include "gemm_ws.cuh"
#include "kernel_traits.cuh"
#include "mma_tma_ws_mainloop.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename CollectiveMainloop, typename CollectiveEpilogue>
__global__ __launch_bounds__(256, 1) void linear_kernel_hopper_cute_wrapper(
    CUTE_GRID_CONSTANT
    typename CollectiveMainloop::Params const mainloop_params,
    CUTE_GRID_CONSTANT
    typename CollectiveEpilogue::Params const epilogue_params) {
  kernel::gemm_kernel_tma_warp_specialized<CollectiveMainloop,
                                           CollectiveEpilogue>(mainloop_params,
                                                               epilogue_params);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_hopper_cute(void *input_ptr,
                               void *weight_ptr,
                               void *residual_ptr,
                               void *output_ptr) {

  using namespace cute;
  auto problem_shape =
      Shape<Int<BATCH_SIZE>, Int<OUTPUT_SIZE>, Int<REDUCTION_SIZE>>{};

  using KernelTraits =
      kernel::MMAKernelTraits<T,
                              BATCH_SIZE,
                              OUTPUT_SIZE,
                              REDUCTION_SIZE,
                              cutlass::layout::RowMajor, // GmemLayoutATag
                              cutlass::layout::RowMajor, // GmemLayoutBTag
                              cutlass::layout::RowMajor, // GmemLayoutCTag
                              cutlass::layout::RowMajor, // GmemLayoutDTag
                              8,                         // NUM_WARPS
                              64,                        // M
                              16,                        // N
                              64,                        // K
                              decltype(problem_shape),
                              OUTPUT_SIZE, // O_STRIDE
                              4>;          // NUM_STAGES

  using Mainloop = kernel::CollectiveMainloop<KernelTraits>;
  using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;

  using StrideA = typename KernelTraits::StrideA;
  using StrideB = typename KernelTraits::StrideB;
  using StrideC = typename KernelTraits::StrideC;
  //   using StrideD = typename KernelTraits::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, {KernelTraits::BATCH_SIZE, KernelTraits::REDUCTION_SIZE, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, {KernelTraits::OUTPUT_SIZE, KernelTraits::REDUCTION_SIZE, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, {KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});
  //   StrideD stride_D = cutlass::make_cute_packed_stride(
  //       StrideD{}, {KernelTraits::M, KernelTraits::N, 1});

  typename Mainloop::Arguments mainloop_args{
      static_cast<T const *>(input_ptr),  // ptr_A
      stride_A,                           // dA
      static_cast<T const *>(weight_ptr), // ptr_B
      stride_B,                           // dB
  };

  typename Epilogue::Arguments epilogue_args{
      static_cast<T const *>(residual_ptr), // ptr_C
      stride_C,                             // dC
      static_cast<T *>(output_ptr),         // ptr_D
      stride_C,                             // dD
      {1.0f, 1.0f}                          // alpha and beta
  };

  typename Mainloop::Params mainloop_params =
      Mainloop::to_underlying_arguments(problem_shape, mainloop_args);
  typename Epilogue::Params epilogue_params =
      Epilogue::to_underlying_arguments(problem_shape, epilogue_args);

  dim3 grid(1);
  dim3 block(256);

  size_t shared_mem_size = 100000;
  cudaFuncSetAttribute(linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_mem_size);
  // linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
  //     <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int WARMUP_RUNS = 16;
  constexpr int BENCHMARK_RUNS = 1000;

  printf("=== Kernel Performance Profiling ===\n");

  for (int i = 0; i < WARMUP_RUNS; i++) {
    linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
        <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);
  }
  cudaDeviceSynchronize(); // Wait for all warmup runs to complete

  printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  float *iteration_times = new float[BENCHMARK_RUNS];
  float total_time_ms = 0.0f;
  float min_time_ms = FLT_MAX;
  float max_time_ms = 0.0f;

  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    cudaEventRecord(start);
    linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
        <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iteration_time_ms;
    cudaEventElapsedTime(&iteration_time_ms, start, stop);

    total_time_ms += iteration_time_ms;
  }

  float avg_time_ms = total_time_ms / BENCHMARK_RUNS;

  printf("\n=== Performance Results ===\n");
  printf("Configuration:\n");
  printf("  BATCH_SIZE=%d, OUTPUT_SIZE=%d, REDUCTION_SIZE=%d\n",
         BATCH_SIZE,
         OUTPUT_SIZE,
         REDUCTION_SIZE);
  printf("  Average: %.3f ms\n", avg_time_ms);

  printf("===============================\n");

  delete[] iteration_times;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void linear_kernel(torch::Tensor input,
                   torch::Tensor weight,
                   torch::Tensor residual,
                   torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  launch_linear_hopper_cute<cutlass::bfloat16_t, 64, 16, 4096>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear_kernel, "Linear kernel");
}