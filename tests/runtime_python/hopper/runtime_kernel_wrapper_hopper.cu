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
#include "hopper/multitoken_paged_attention_hopper.cuh"
#include "hopper/norm_linear_hopper.cuh"
//  #include "hopper/ref.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
// create tma
using kernel::linear_kernel_hopper;
using kernel::multitoken_paged_attention_hopper_impl;
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
                       TMA_OUT>(tma_a, tma_b, tma_residual, tma_out);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_hopper(void *input_ptr,
                          void *weight_ptr,
                          void *residual_ptr,
                          void *output_ptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE = 64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE = 128; // we should modify this param if we want larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr int OUTPUT_REPEAT_COL =
      (OUTPUT_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  constexpr int OUTPUT_TMA_CP_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;

  //  using TMA_A = kernel::tma::tma<bfloat16,
  //                                 B,
  //                                 M,
  //                                 S,
  //                                 BATCH_SIZE,
  //                                 REDUCTION_SIZE,
  //                                 BATCH_SIZE,
  //                                 TILE_SIZE,
  //                                 true>;
  //  using TMA_B = kernel::tma::tma<bfloat16,
  //                                 B,
  //                                 M,
  //                                 S,
  //                                 OUTPUT_SIZE,
  //                                 REDUCTION_SIZE,
  //                                 OUTPUT_SIZE,
  //                                 TILE_SIZE,
  //                                 true>;
  using TMA_A = kernel::tma::tma_general<bfloat16,
                                         B,
                                         M,
                                         S,
                                         BATCH_SIZE,
                                         REDUCTION_SIZE,
                                         BATCH_SIZE,
                                         TMA_CP_ASYNC_SIZE,
                                         1,
                                         TMA_CP_ASYNC_REPEAT_COL,
                                         true>;
  using TMA_B = kernel::tma::tma_general<bfloat16,
                                         B,
                                         M,
                                         S,
                                         OUTPUT_SIZE,
                                         REDUCTION_SIZE,
                                         OUTPUT_SIZE,
                                         TMA_CP_ASYNC_SIZE,
                                         1,
                                         TMA_CP_ASYNC_REPEAT_COL,
                                         true>;
  using TMA_RESIDUAL = kernel::tma::tma_general<bfloat16,
                                        0,
                                        0,
                                        0,
                                        BATCH_SIZE,
                                        OUTPUT_SIZE,
                                        BATCH_SIZE,
                                        OUTPUT_TMA_CP_SIZE,
                                        1,
                                        OUTPUT_REPEAT_COL,
                                        true>;

  using TMA_OUT = kernel::tma::tma_general<bfloat16,
                                   0,
                                   0,
                                   0,
                                   BATCH_SIZE,
                                   OUTPUT_SIZE,
                                   BATCH_SIZE,
                                   OUTPUT_TMA_CP_SIZE,
                                   1,
                                   OUTPUT_REPEAT_COL,
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

#ifndef MIRAGE_PROFILE_HOPPER
  linear_kernel_hopper_wrapper<T,
                               BATCH_SIZE,
                               OUTPUT_SIZE,
                               REDUCTION_SIZE,
                               TMA_A,
                               TMA_B,
                               TMA_RESIDUAL,
                               TMA_OUT>
      <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_residual, tma_out);
#else

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int WARMUP_RUNS = 16;
  constexpr int BENCHMARK_RUNS = 1000;

  printf("=== Kernel Performance Profiling ===\n");

  for (int i = 0; i < WARMUP_RUNS; i++) {
    linear_kernel_hopper_wrapper<T,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 REDUCTION_SIZE,
                                 TMA_A,
                                 TMA_B,
                                 TMA_RESIDUAL,
                                 TMA_OUT><<<grid_dim, block_dim, smem_size>>>(
        tma_a, tma_b, tma_residual, tma_out);
  }
  cudaDeviceSynchronize(); // Wait for all warmup runs to complete

  printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  float *iteration_times = new float[BENCHMARK_RUNS];
  float total_time_ms = 0.0f;
  float min_time_ms = FLT_MAX;
  float max_time_ms = 0.0f;

  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    cudaEventRecord(start);
    linear_kernel_hopper_wrapper<T,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 REDUCTION_SIZE,
                                 TMA_A,
                                 TMA_B,
                                 TMA_RESIDUAL,
                                 TMA_OUT><<<grid_dim, block_dim, smem_size>>>(
        tma_a, tma_b, tma_residual, tma_out);
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
  printf(" TILE SIZE: %d\n", TILE_SIZE);
  printf("  Average: %.3f ms\n", avg_time_ms);

  printf("===============================\n");

  delete[] iteration_times;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
#endif

#define DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(                            \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)                                   \
  case REDUCTION_SIZE:                                                         \
    launch_linear_hopper<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(   \
        input_ptr, weight_ptr, residual_ptr, output_ptr);                      \
    break;

#define DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)         \
  switch (input.size(1)) {                                                     \
    DISPATCH_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 64)    \
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
      //  DISPATCH_LINEAR_HOPPER_BATCH_SIZE_CASE(16)
      DISPATCH_LINEAR_HOPPER_BATCH_SIZE_CASE(64)
      default:
        printf("Unsupported batch size in test: %zu\n", output.size(0));
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
                              TMA_OUT>(
        tma_input, tma_norm_weight, tma_linear_weight, tma_out, eps);
  }

  template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
  void launch_norm_linear_hopper(void *input_ptr,
                                 void *norm_weight_ptr,
                                 void *weight_ptr,
                                 void *output_ptr,
                                 float eps) {

    constexpr int B = 3;
    constexpr int M = 3;
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
                                     OUTPUT_SIZE,
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
                                      TMA_OUT>
        <<<grid_dim, block_dim, smem_size>>>(
            tma_input, tma_norm_weight, tma_linear_weight, tma_out, eps);
  }

#define DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(                       \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)                                   \
  case REDUCTION_SIZE:                                                         \
    launch_norm_linear_hopper<bfloat16,                                        \
                              BATCH_SIZE,                                      \
                              OUTPUT_SIZE,                                     \
                              REDUCTION_SIZE>(                                 \
        input_ptr, norm_weight_ptr, weight_ptr, output_ptr, eps);              \
    break;

#define DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)    \
  switch (input.size(1)) {                                                     \
    /*DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, \
    128) DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE,                   \
    OUTPUT_SIZE, 256)                                                          \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE,   \
    512) DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(BATCH_SIZE,                   \
    OUTPUT_SIZE, 3072)  */                                                     \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE_CASE(                           \
        BATCH_SIZE, OUTPUT_SIZE, 4096)                                         \
    default:                                                                   \
      printf("Unsupported reduction size in test: %zu\n", input.size(1));      \
      break;                                                                   \
  }

#define DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE)  \
  case OUTPUT_SIZE:                                                            \
    DISPATCH_NORM_LINEAR_HOPPER_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)        \
    break;

#define DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE(BATCH_SIZE)                    \
  switch (output.size(1)) {                                                    \
    DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 16)               \
    DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 32)               \
    DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE_CASE(BATCH_SIZE, 64)               \
    default:                                                                   \
      printf("Unsupported output size in test: %zu\n", output.size(1));        \
      break;                                                                   \
  }

#define DISPATCH_NORM_LINEAR_HOPPER_BATCH_SIZE_CASE(BATCH_SIZE)                \
  case BATCH_SIZE:                                                             \
    DISPATCH_NORM_LINEAR_HOPPER_OUTPUT_SIZE(BATCH_SIZE)                        \
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

  // Multitoken Paged Attention
  template <typename T,
            int NUM_QO_HEADS,
            int NUM_KV_HEADS,
            int KV_CACHE_STRIDE,
            int QKV_STRIDE,
            int O_STRIDE,
            int HEAD_DIM,
            int MAX_SEQ_LEN,
            int PAGE_SIZE,
            typename TMA_Q,
            typename TMA_KV,
            typename TMA_PAGED_KV,
            typename TMA_OUTPUT,
            int MAX_TOKENS = 8>
  __global__ void multitoken_paged_attention_wrapper_hopper(
      const __grid_constant__ TMA_Q tma_q,
      const __grid_constant__ TMA_KV tma_k,
      const __grid_constant__ TMA_KV tma_v,
      const __grid_constant__ TMA_PAGED_KV tma_paged_k_cache,
      const __grid_constant__ TMA_PAGED_KV tma_paged_v_cache,
      const __grid_constant__ TMA_OUTPUT tma_output,
      // old arguments
      void *qkv_ptr,
      void *paged_k_cache_ptr,
      void *paged_v_cache_ptr,
      void *output_ptr,
      int const *qo_indptr_buffer_ptr,
      int const *paged_kv_indptr_buffer_ptr,
      int const *paged_kv_indices_buffer_ptr,
      int const *paged_kv_last_page_len_buffer_ptr,
      int request_id,
      bool qk_norm,
      bool rope,
      void const *q_norm_weight_ptr,
      void const *k_norm_weight_ptr,
      void const *cos_ptr,
      void const *sin_ptr,
      float q_eps,
      float k_eps) {

    multitoken_paged_attention_hopper_impl<T,
                                           NUM_QO_HEADS,
                                           NUM_KV_HEADS,
                                           KV_CACHE_STRIDE,
                                           QKV_STRIDE,
                                           O_STRIDE,
                                           HEAD_DIM,
                                           MAX_SEQ_LEN,
                                           PAGE_SIZE,
                                           TMA_Q,
                                           TMA_KV,
                                           TMA_PAGED_KV,
                                           TMA_OUTPUT,
                                           MAX_TOKENS>(
        tma_q,
        tma_k,
        tma_v,
        tma_paged_k_cache,
        tma_paged_v_cache,
        tma_output,
        // old arguments
        qkv_ptr,
        paged_k_cache_ptr,
        paged_v_cache_ptr,
        output_ptr,
        qo_indptr_buffer_ptr,
        paged_kv_indptr_buffer_ptr,
        paged_kv_indices_buffer_ptr,
        paged_kv_last_page_len_buffer_ptr,
        request_id,
        qk_norm,
        rope,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_eps,
        k_eps);
  }

  template <typename T,
            int NUM_QO_HEADS,
            int NUM_KV_HEADS,
            int KV_CACHE_STRIDE,
            int QKV_STRIDE,
            int O_STRIDE,
            int HEAD_DIM,
            int MAX_SEQ_LEN,
            int PAGE_SIZE,
            int MAX_TOKENS = 8>
  void launch_multitoken_paged_attention_hopper(
      void *qkv_ptr,
      void *paged_k_cache_ptr,
      void *paged_v_cache_ptr,
      void *output_ptr,
      int const *qo_indptr_buffer_ptr,
      int const *paged_kv_indptr_buffer_ptr,
      int const *paged_kv_indices_buffer_ptr,
      int const *paged_kv_last_page_len_buffer_ptr,
      int request_id,
      bool qk_norm,
      bool rope,
      void const *q_norm_weight_ptr,
      void const *k_norm_weight_ptr,
      void const *cos_ptr,
      void const *sin_ptr,
      float q_eps,
      float k_eps) {
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 224 * 1024;

    constexpr int KV_TILE_SIZE = 64;
    constexpr int prompt_len = 8;
    constexpr int num_tokens = 4;

    // printf("on wrapper, num_tokens*NUM_QO_HEADS: %d\n", num_tokens *
    // NUM_QO_HEADS);

    //  using TMA_Q = kernel::tma::tma<bfloat16, 3, 3, 3, num_tokens *
    //  NUM_QO_HEADS, HEAD_DIM, num_tokens * NUM_QO_HEADS, KV_TILE_SIZE, true>;
    using TMA_Q =
        kernel::tma::tma_general<bfloat16,
                                 3,
                                 3,
                                 3,
                                 num_tokens * NUM_QO_HEADS,
                                 HEAD_DIM,
                                 num_tokens * NUM_QO_HEADS,
                                 KV_TILE_SIZE,
                                 1,
                                 (HEAD_DIM + KV_TILE_SIZE - 1) / KV_TILE_SIZE,
                                 true>;
    using TMA_KV =
        kernel::tma::tma_general<bfloat16,
                                 3,
                                 3,
                                 3,
                                 num_tokens * NUM_KV_HEADS,
                                 HEAD_DIM,
                                 num_tokens * NUM_KV_HEADS,
                                 KV_TILE_SIZE,
                                 1,
                                 (HEAD_DIM + KV_TILE_SIZE - 1) / KV_TILE_SIZE,
                                 true>;

    using TMA_PAGED_KV_CACHE = kernel::tma::tma<bfloat16,
                                                3,
                                                3,
                                                3,
                                                KV_TILE_SIZE,
                                                HEAD_DIM,
                                                KV_TILE_SIZE,
                                                HEAD_DIM,
                                                true>;
    using TMA_OUTPUT = kernel::tma::tma<bfloat16,
                                        3,
                                        3,
                                        3,
                                        MAX_TOKENS * NUM_QO_HEADS,
                                        HEAD_DIM,
                                        MAX_TOKENS * NUM_QO_HEADS,
                                        HEAD_DIM,
                                        true>;

    bfloat16 *__restrict__ qkv_ptr_bf16 = static_cast<bfloat16 *>(qkv_ptr);

    TMA_Q tma_q(reinterpret_cast<void *>(qkv_ptr_bf16));
    TMA_KV tma_k(reinterpret_cast<void *>(
        qkv_ptr_bf16 + num_tokens * NUM_QO_HEADS * HEAD_DIM));
    TMA_KV tma_v(reinterpret_cast<void *>(
        qkv_ptr_bf16 + num_tokens * (NUM_QO_HEADS + NUM_KV_HEADS) * HEAD_DIM));
    TMA_PAGED_KV_CACHE tma_paged_k_cache(paged_k_cache_ptr);
    TMA_PAGED_KV_CACHE tma_paged_v_cache(paged_v_cache_ptr);
    TMA_OUTPUT tma_output(output_ptr);

    cudaFuncSetAttribute(
        multitoken_paged_attention_wrapper_hopper<T,
                                                  NUM_QO_HEADS,
                                                  NUM_KV_HEADS,
                                                  KV_CACHE_STRIDE,
                                                  QKV_STRIDE,
                                                  O_STRIDE,
                                                  HEAD_DIM,
                                                  MAX_SEQ_LEN,
                                                  PAGE_SIZE,
                                                  TMA_Q,
                                                  TMA_KV,
                                                  TMA_PAGED_KV_CACHE,
                                                  TMA_OUTPUT,
                                                  MAX_TOKENS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    multitoken_paged_attention_wrapper_hopper<T,
                                              NUM_QO_HEADS,
                                              NUM_KV_HEADS,
                                              KV_CACHE_STRIDE,
                                              QKV_STRIDE,
                                              O_STRIDE,
                                              HEAD_DIM,
                                              MAX_SEQ_LEN,
                                              PAGE_SIZE,
                                              TMA_Q,
                                              TMA_KV,
                                              TMA_PAGED_KV_CACHE,
                                              TMA_OUTPUT,
                                              MAX_TOKENS>
        <<<grid_dim, block_dim, smem_size>>>(tma_q,
                                             tma_k,
                                             tma_v,
                                             tma_paged_k_cache,
                                             tma_paged_v_cache,
                                             tma_output,
                                             // old arguments
                                             qkv_ptr,
                                             paged_k_cache_ptr,
                                             paged_v_cache_ptr,
                                             output_ptr,
                                             qo_indptr_buffer_ptr,
                                             paged_kv_indptr_buffer_ptr,
                                             paged_kv_indices_buffer_ptr,
                                             paged_kv_last_page_len_buffer_ptr,
                                             request_id,
                                             qk_norm,
                                             rope,
                                             q_norm_weight_ptr,
                                             k_norm_weight_ptr,
                                             cos_ptr,
                                             sin_ptr,
                                             q_eps,
                                             k_eps);
  }

  void multitoken_paged_attention_hopper(
      torch::Tensor qkv,
      torch::Tensor paged_k_cache,
      torch::Tensor paged_v_cache,
      torch::Tensor output,
      torch::Tensor qo_indptr_buffer,
      torch::Tensor paged_kv_indptr_buffer,
      torch::Tensor paged_kv_indices_buffer,
      torch::Tensor paged_kv_last_page_len_buffer,
      int request_id,
      bool qk_norm,
      bool rope,
      torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
      torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
      torch::optional<torch::Tensor> cos = torch::nullopt,
      torch::optional<torch::Tensor> sin = torch::nullopt,
      float q_eps = 0.0f,
      float k_eps = 0.0f) {
    void *qkv_ptr = qkv.data_ptr();
    void *paged_k_cache_ptr = paged_k_cache.data_ptr();
    void *paged_v_cache_ptr = paged_v_cache.data_ptr();
    void *output_ptr = output.data_ptr();
    int const *qo_indptr_buffer_ptr = qo_indptr_buffer.data_ptr<int>();
    int const *paged_kv_indptr_buffer_ptr =
        paged_kv_indptr_buffer.data_ptr<int>();
    int const *paged_kv_indices_buffer_ptr =
        paged_kv_indices_buffer.data_ptr<int>();
    int const *paged_kv_last_page_len_buffer_ptr =
        paged_kv_last_page_len_buffer.data_ptr<int>();

    void const *q_norm_weight_ptr =
        qk_norm ? q_norm_weight->data_ptr() : nullptr;
    void const *k_norm_weight_ptr =
        qk_norm ? k_norm_weight->data_ptr() : nullptr;
    void const *cos_ptr = rope ? cos->data_ptr() : nullptr;
    void const *sin_ptr = rope ? sin->data_ptr() : nullptr;
    int const qo_heads = 4;
    int const kv_heads = 1;
    int const head_dim = 128;
    int const qkv_stride = (qo_heads + 2 * kv_heads) * head_dim;
    assert(qkv_stride == qkv.stride(0));
    int const kv_stride = head_dim * kv_heads;
    assert(kv_stride == paged_k_cache.stride(1));
    int const o_stride = head_dim * qo_heads;
    int const page_size = 4096;
    int const max_seq_len = 512;

    launch_multitoken_paged_attention_hopper<bfloat16,
                                             qo_heads,
                                             kv_heads,
                                             kv_stride,
                                             qkv_stride,
                                             o_stride,
                                             head_dim,
                                             max_seq_len,
                                             page_size>(
        qkv_ptr,
        paged_k_cache_ptr,
        paged_v_cache_ptr,
        output_ptr,
        qo_indptr_buffer_ptr,
        paged_kv_indptr_buffer_ptr,
        paged_kv_indices_buffer_ptr,
        paged_kv_last_page_len_buffer_ptr,
        request_id,
        qk_norm,
        rope,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_eps,
        k_eps);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear_kernel, "Linear kernel");
    m.def("norm_linear", &norm_linear_kernel, "NormLinear kernel");
    m.def("multitoken_paged_attention",
          &multitoken_paged_attention_hopper,
          "Multitoken paged attention for Grace Hopper GPU");
  }