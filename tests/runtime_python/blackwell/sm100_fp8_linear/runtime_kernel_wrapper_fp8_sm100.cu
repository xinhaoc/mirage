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
#include "blackwell/task_header.cuh"
#include "hopper/tma_2d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

using bfloat16 = cute::bfloat16_t;
using fp8_e4m3 = cute::float_e4m3_t;

// FP8 linear kernel wrapper
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int MMA_M = 128,
          int MMA_N = 16,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__global__
    __launch_bounds__(256,
                      1) void fp8_linear_sm100_mpk_wrapper(void *tma_a_desc_ptr,
                                                            void *tma_b_desc_ptr,
                                                            void *tma_out_desc_ptr,
                                                            const uint8_t *sfa_ptr,
                                                            const uint8_t *sfb_ptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE = 128; // 128 FP8 elements = 128 bytes
  constexpr int TILE_SIZE = 128;         // bK for FP8
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  using TMA_B =
      kernel::tma::tma_2d<fp8_e4m3,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_N,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<fp8_e4m3,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,                /*GMEM_ROW_*/
                          REDUCTION_SIZE,             /*GMEM_COL_*/
                          MMA_M,                      /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,          /*SMEM_COL_*/
                          REDUCTION_SIZE,             /*GMEM_STRIDE_ROW_*/
                          1,                          /*GMEM_STRIDE_COL_*/
                          1,                          /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,    /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE,  /*SMEM_STRIDE_*/
                          true>;

  using TMA_OUT =
      kernel::tma::tma_2d<bfloat16,
                          0,
                          M,
                          S,
                          BATCH_SIZE,    /*GMEM_ROW_*/
                          OUTPUT_SIZE,   /*GMEM_COL_*/
                          MMA_N,         /*SMEM_ROW_*/
                          MMA_M,         /*SMEM_COL_*/
                          OUTPUT_SIZE,   /*GMEM_STRIDE_ROW_*/
                          1,             /*GMEM_STRIDE_COL_*/
                          1,             /*SMEM_REPEAT_ROW_*/
                          1,             /*SMEM_REPEAT_COL_*/
                          MMA_N * MMA_M, /*SMEM_STRIDE_*/
                          true>;

  TMA_A tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::fp8_linear_sm100_mpk_task_impl<fp8_e4m3,
                                          TMA_A,
                                          TMA_B,
                                          TMA_OUT,
                                          MMA_M,
                                          MMA_N,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          NUM_AB_STAGE,
                                          NUM_ACC_STAGE,
                                          NUM_C_STAGE>(tma_a, tma_b, tma_out,
                                                       sfa_ptr, sfb_ptr);
}

template <int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_fp8_linear_sm100_mpk(void *input_ptr,
                                  void *weight_ptr,
                                  void *output_ptr,
                                  const uint8_t *sfa_ptr,
                                  const uint8_t *sfb_ptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;

  constexpr int TMA_CP_ASYNC_SIZE = 128;
  constexpr int TILE_SIZE = 128;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  CUtensorMap host_i_desc;
  CUtensorMap host_w_desc;
  CUtensorMap host_o_desc;
  CUtensorMap *desc_i_ptr;
  CUtensorMap *desc_w_ptr;
  CUtensorMap *desc_o_ptr;

  // TMA_INPUT (B): [BATCH_SIZE, REDUCTION_SIZE] in FP8
  uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                               static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                               static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t i_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<uint8_t, B, M, S, 2>(
      &host_i_desc,
      static_cast<uint8_t *>(input_ptr),
      i_gmem_shape,
      i_gmem_stride,
      i_smem_shape,
      1,
      i_smem_repeat_col);

  // TMA_WEIGHT (A): [OUTPUT_SIZE, REDUCTION_SIZE] in FP8
  uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUTPUT_SIZE),
                               static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                               static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t w_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<uint8_t, B, M, S, 2>(
      &host_w_desc,
      static_cast<uint8_t *>(weight_ptr),
      w_gmem_shape,
      w_gmem_stride,
      w_smem_shape,
      1,
      w_smem_repeat_col);

  // TMA_OUT: [BATCH_SIZE, OUTPUT_SIZE] in BF16
  uint64_t o_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                               static_cast<uint64_t>(OUTPUT_SIZE)};
  uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(OUTPUT_SIZE)};
  uint32_t o_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                               static_cast<uint32_t>(MMA_M)};
  size_t o_smem_repeat_col = 1;
  mirage::runtime::fill_tma_desc<bfloat16, 0, M, S, 2>(
      &host_o_desc,
      static_cast<bfloat16 *>(output_ptr),
      o_gmem_shape,
      o_gmem_stride,
      o_smem_shape,
      1,
      o_smem_repeat_col);

  cudaMalloc(&desc_i_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_o_ptr, sizeof(CUtensorMap));

  cudaMemcpy(
      desc_i_ptr, &host_i_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(
      desc_w_ptr, &host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(
      desc_o_ptr, &host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 224 * 1024;

  auto *kernel_ptr = &fp8_linear_sm100_mpk_wrapper<BATCH_SIZE,
                                                     OUTPUT_SIZE,
                                                     REDUCTION_SIZE>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
  cutlass::ClusterLaunchParams params = {
      grid_dim, block_dim, cluster_dim, smemBytes};
  cutlass::Status status =
      cutlass::launch_kernel_on_cluster(params,
                                        (void const *)kernel_ptr,
                                        (void *)desc_w_ptr,
                                        (void *)desc_i_ptr,
                                        (void *)desc_o_ptr,
                                        sfa_ptr,
                                        sfb_ptr);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }

  cudaFree(desc_i_ptr);
  cudaFree(desc_w_ptr);
  cudaFree(desc_o_ptr);
}

void fp8_linear_sm100_mpk_kernel(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor sfa,
                                  torch::Tensor sfb,
                                  torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  const uint8_t *sfa_ptr = static_cast<const uint8_t *>(sfa.data_ptr());
  const uint8_t *sfb_ptr = static_cast<const uint8_t *>(sfb.data_ptr());
  void *output_ptr = output.data_ptr();

  constexpr int BATCH_SIZE = 1;
  constexpr int OUTPUT_SIZE = 128;
  constexpr int REDUCTION_SIZE = 768;

  assert(input.size(0) == BATCH_SIZE);
  assert(input.size(1) == REDUCTION_SIZE);
  assert(weight.size(0) == OUTPUT_SIZE);
  assert(weight.size(1) == REDUCTION_SIZE);

  launch_fp8_linear_sm100_mpk<BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, weight_ptr, output_ptr, sfa_ptr, sfb_ptr);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_linear_sm100_mpk",
        &fp8_linear_sm100_mpk_kernel,
        "FP8 Linear kernel SM100 MPK");
}
