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
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "bfloat16.h"
#include "hopper/matmul_demo_hopper.cuh"
// create tma
using kernel::linear_kernel_hopper;
using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int HIDDEN_SIZE,
          int Kstages,
          typename TMA_A,
          typename TMA_B,
          typename TMA_OUT>
__global__ void
    linear_kernel_hopper_wrapper(void *output_ptr,
                                 const __grid_constant__ TMA_A tma_a,
                                 const __grid_constant__ TMA_B tma_b,
                                 const __grid_constant__ TMA_OUT tma_out) {
  linear_kernel_hopper<T,
                       BATCH_SIZE,
                       HIDDEN_SIZE,
                       Kstages,
                       TMA_A,
                       TMA_B,
                       TMA_OUT>(output_ptr, tma_a, tma_b, tma_out);
}

void linear_kernel(torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output) {
  

  void  *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();


  using TMA_A = kernel::tma::tma<bfloat16, 3, 4, 3, 64, 4096, 64, 64, true>;
  using TMA_B = kernel::tma::tma<bfloat16, 3, 4, 3, 4096, 64, 64, 64, true>;

  using TMA_OUT = kernel::tma::tma<bfloat16, 3, 4, 3, 64, 64, 64, 64, true>;

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  size_t smem_size = 88888;
  cudaFuncSetAttribute(linear_kernel_hopper_wrapper<bfloat16,
                                                    64,
                                                    4096,
                                                    2,
                                                    TMA_A,
                                                    TMA_B,
                                                    TMA_OUT>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  linear_kernel_hopper_wrapper<bfloat16, 64, 4096, 2, TMA_A, TMA_B, TMA_OUT>
      <<<grid_dim, block_dim>>>(
          output_ptr, TMA_A(input_ptr), TMA_B(weight_ptr), TMA_OUT(output_ptr));

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear_kernel, "Linear kernel");
}