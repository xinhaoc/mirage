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
#include "bfloat16.h"
#include "hopper/linear_hopper.cuh"
#include "hopper/norm_linear_hopper.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
// create tma
using kernel::linear_kernel_hopper;
using kernel::norm_linear_kernel_hopper;
using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          typename TMA_A,
          typename TMA_B,
          typename TMA_RESIDUAL,
          typename TMA_OUT,
          int Kstages = 2>
__global__ __launch_bounds__(256, 1) void linear_kernel_hopper_wrapper(
    void *output_ptr,
    const __grid_constant__ TMA_A tma_a,
    const __grid_constant__ TMA_B tma_b,
    const __grid_constant__ TMA_RESIDUAL tma_residual,
    const __grid_constant__ TMA_OUT tma_out) {
  linear_kernel_hopper<T,
                       BATCH_SIZE,
                       OUTPUT_SIZE,
                       REDUCTION_SIZE,
                       Kstages,
                       TMA_A,
                       TMA_B,
                       TMA_OUT>(
      output_ptr, tma_a, tma_b, tma_residual, tma_out);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_hopper(void *input_ptr,
                          void *weight_ptr,
                          void *residual_ptr,
                          void *output_ptr) {

  constexpr int B = 3;
  constexpr int M = 4;
  constexpr int S = 3;

  constexpr int TILE_SIZE = 64;

  using TMA_A = kernel::tma::tma<bfloat16,
                                 B,
                                 M,
                                 S,
                                 BATCH_SIZE,
                                 REDUCTION_SIZE,
                                 BATCH_SIZE,
                                 TILE_SIZE,
                                 true>;
  using TMA_B = kernel::tma::tma<bfloat16,
                                 B,
                                 M,
                                 S,
                                 OUTPUT_SIZE,
                                 REDUCTION_SIZE,
                                 OUTPUT_SIZE,
                                 TILE_SIZE,
                                 true>;
  using TMA_RESIDUAL = kernel::tma::tma<bfloat16,
                                        0,
                                        0,
                                        0,
                                        BATCH_SIZE,
                                        OUTPUT_SIZE,
                                        BATCH_SIZE,
                                        OUTPUT_SIZE,
                                        true>;

  using TMA_OUT = kernel::tma::tma<bfloat16,
                                   0,
                                   0,
                                   0,
                                   BATCH_SIZE,
                                   OUTPUT_SIZE,
                                   BATCH_SIZE,
                                   OUTPUT_SIZE,
                                   true>;

  TMA_A tma_a(input_ptr);
  TMA_B tma_b(weight_ptr);
  TMA_RESIDUAL tma_residual(residual_ptr);
  TMA_OUT tma_out(output_ptr);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  size_t smem_size = 88888;
  cudaFuncSetAttribute(linear_kernel_hopper_wrapper<T,
                                                    BATCH_SIZE,
                                                    OUTPUT_SIZE,
                                                    REDUCTION_SIZE,
                                                    TMA_A,
                                                    TMA_B,
                                                    TMA_RESIDUAL,
                                                    TMA_OUT>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  linear_kernel_hopper_wrapper<T,
                               BATCH_SIZE,
                               OUTPUT_SIZE,
                               REDUCTION_SIZE,
                               TMA_A,
                               TMA_B,
                               TMA_RESIDUAL,
                               TMA_OUT><<<grid_dim, block_dim, smem_size>>>(
      output_ptr, tma_a, tma_b, tma_residual, tma_out);
}

#define DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(                            \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)                                   \
  case REDUCTION_SIZE:                                                         \
    launch_linear_hopper<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(   \
        input_ptr, weight_ptr, residual_ptr, output_ptr);                      \
    break;

#define DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)         \
  switch (input.size(1)) {                                                     \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 128)   \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 256)   \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 512)   \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 3072)  \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 4096)  \
    default:                                                                   \
      printf("Unsupported reduction size in test: %zu\n", input.size(1));      \
      break;                                                                   \
  }

#define DISPATCH_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE)       \
  case OUTPUT_SIZE:                                                            \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)             \
    break;

#define DISPATCH_LINEAR_HOPPER_OUTPUT_SIZE(BATCH_SIZE)                         \
  switch (output.size(1)) {                                                    \
    DISPATCH_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 16)                    \
    DISPATCH_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 32)                    \
    DISPATCH_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 64)                    \
    default:                                                                   \
      printf("Unsupported output size in test: %zu\n", output.size(1));        \
      break;                                                                   \
  }

#define DISPATCH_LINEAR_HOPPER_BATCH_SIZE_CASE(BATCH_SIZE)                     \
  case BATCH_SIZE:                                                             \
    DISPATCH_LINEAR_HOPPER_OUTPUT_SIZE(BATCH_SIZE)                             \
    break;

void linear_kernel(torch::Tensor input,
                   torch::Tensor weight,
                   torch::Tensor residual,
                   torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (input.size(0)) {
    DISPATCH_LINEAR_HOPPER_BATCH_SIZE_CASE(64)
    default:
      printf("Unsupported output size in test: %zu\n", output.size(0));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}



// norm linear
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          typename TMA_INPUT,
          typename TMA_NORM_WEIGHT,
          typename TMA_LINEAR_WEIGHT,
          typename TMA_OUT,
          int Kstages = 2>
__global__ __launch_bounds__(256, 1) void norm_linear_kernel_hopper_wrapper(
    void *output_ptr,
    const __grid_constant__ TMA_INPUT tma_input,
    const __grid_constant__ TMA_NORM_WEIGHT tma_norm_weight,
    const __grid_constant__ TMA_LINEAR_WEIGHT tma_linear_weight,
    const __grid_constant__ TMA_OUT tma_out,
    float eps) {
  norm_linear_kernel_hopper<T,
                            BATCH_SIZE,
                            OUTPUT_SIZE,
                            REDUCTION_SIZE,
                            Kstages,
                            TMA_INPUT,
                            TMA_NORM_WEIGHT,
                            TMA_LINEAR_WEIGHT,
                            TMA_OUT>(output_ptr,
                                     tma_input,
                                     tma_norm_weight,
                                     tma_linear_weight,
                                     tma_out,
                                     eps);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_norm_linear_hopper(void *input_ptr,
                               void *norm_weight_ptr,
                               void *weight_ptr,
                               void *output_ptr,
                               float eps) {

  constexpr int B = 3;
  constexpr int M = 4;
  constexpr int S = 3;

  constexpr int TILE_SIZE = 64;

  using TMA_INPUT = kernel::tma::tma<bfloat16,
                                     B,
                                     M,
                                     S,
                                     BATCH_SIZE,
                                     REDUCTION_SIZE,
                                     BATCH_SIZE,
                                     TILE_SIZE,
                                     true>;
  using TMA_NORM_WEIGHT = kernel::tma::tma<bfloat16,
                                           B,
                                           M,
                                           S,
                                           BATCH_SIZE,
                                           REDUCTION_SIZE,
                                           BATCH_SIZE,
                                           TILE_SIZE,
                                           true>;
  using TMA_LINEAR_WEIGHT = kernel::tma::tma<bfloat16,
                                             B,
                                             M,
                                             S,
                                             OUTPUT_SIZE,
                                             REDUCTION_SIZE,
                                             OUTPUT_SIZE,
                                             TILE_SIZE,
                                             true>;

  using TMA_OUT = kernel::tma::tma<bfloat16,
                                   0,
                                   0,
                                   0,
                                   BATCH_SIZE,
                                   OUTPUT_SIZE,
                                   BATCH_SIZE,
                                   TILE_SIZE,
                                   true>;

  TMA_INPUT tma_input(input_ptr);
  TMA_NORM_WEIGHT tma_norm_weight(norm_weight_ptr);
  TMA_LINEAR_WEIGHT tma_linear_weight(weight_ptr);
  TMA_OUT tma_out(output_ptr);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  size_t smem_size = 88888;
  cudaFuncSetAttribute(norm_linear_kernel_hopper_wrapper<T,
                                                         BATCH_SIZE,
                                                         OUTPUT_SIZE,
                                                         REDUCTION_SIZE,
                                                         TMA_INPUT,
                                                         TMA_NORM_WEIGHT,
                                                         TMA_LINEAR_WEIGHT,
                                                         TMA_OUT>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  norm_linear_kernel_hopper_wrapper<T,
                                    BATCH_SIZE,
                                    OUTPUT_SIZE,
                                    REDUCTION_SIZE,
                                    TMA_INPUT,
                                    TMA_NORM_WEIGHT,
                                    TMA_LINEAR_WEIGHT,
                                    TMA_OUT><<<grid_dim, block_dim, smem_size>>>(
      output_ptr, tma_input, tma_norm_weight, tma_linear_weight, tma_out, eps);
}

#define DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(                                   \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)                                               \
  case REDUCTION_SIZE:                                                                     \
    launch_norm_linear_hopper<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(          \
        input_ptr, norm_weight_ptr, weight_ptr, output_ptr, eps);                 \
    break;

#define DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)                \
  switch (input.size(1)) {                                                                 \
    /*DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 128)          \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 256)          \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 512)          \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 3072)  */       \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 4096)         \
    default:                                                                               \
      printf("Unsupported reduction size in test: %zu\n", input.size(1));                 \
      break;                                                                               \
  }

#define DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE)              \
  case OUTPUT_SIZE:                                                                        \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)                    \
    break;

#define DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE(BATCH_SIZE)                                \
  switch (output.size(1)) {                                                                \
    DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 64)                           \
    default:                                                                               \
      printf("Unsupported output size in test: %zu\n", output.size(1));                  \
      break;                                                                               \
  }

#define DISPATCH_NORM_LINEAR_HOPPER_BATCH_SIZE_CASE(BATCH_SIZE)                            \
  case BATCH_SIZE:                                                                         \
    DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE(BATCH_SIZE)                                    \
    break;

void norm_linear_kernel(torch::Tensor input,
                        torch::Tensor norm_weight,
                        torch::Tensor weight,
                        torch::Tensor output,
                        float eps) {

  void *input_ptr = input.data_ptr();
  void *norm_weight_ptr = norm_weight.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (input.size(0)) {
    DISPATCH_NORM_LINEAR_HOPPER_BATCH_SIZE_CASE(64)
    default:
      printf("Unsupported output size in test: %zu\n", output.size(0));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear_kernel, "Linear kernel");
  m.def("norm_linear", &norm_linear_kernel, "NormLinear kernel");
}