
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

#pragma once
#include "../common.h"
#include "../dmem_layout.cuh"
#include "../element_binary.cuh"
#include "../element_unary.cuh"
#include "../reduction.cuh"
#include "../smem_layout.cuh"
#include "../utils.cuh"
#include "tma.cuh"
#include "utils.cuh"
#include "wgmma.cuh"
namespace kernel {

using namespace tma;
using bfloat16 = type::bfloat16_t;

// a 64X64X4K kernel for reference
template <typename T,
          int BATCH_SIZE,
          int HIDDEN_SIZE,
          int Kstages,
          typename TMA_A,
          typename TMA_B,
          typename TMA_OUT>
__device__ __forceinline__ void linear_kernel_hopper(void *output_ptr,
                                                     const TMA_A &tma_a,
                                                     const TMA_B &tma_b,
                                                     const TMA_OUT &tma_out) {
  constexpr int chunk_size = 16 / sizeof(T);
  constexpr int NUM_THREADS = 128;
  constexpr int CONSUMER_WARPGROUPS = 1;
  constexpr int PRODUCER_WARPGROUPS = 1;
  constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int num_n = HIDDEN_SIZE / 64;
  constexpr int num_m = BATCH_SIZE / 64;
  constexpr int num_k = HIDDEN_SIZE / 16;
  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;
  int warpgroup_id = warp_idx / WARPGROUP_WARPS;

  T __restrict__ *d_output = static_cast<T *>(output_ptr) + blockIdx.x * 64 * 1;

  dmem_row<T, 64, 64, 64> output_dmem(d_output);

  extern __shared__ char smem[];

  // copy input
  T *shared_input = (T *)(smem + 128);
  // copy weight
  T *shared_weight = (T *)(smem + 16512);
  // intermidiate
  T *mm_output = (T *)(smem + 128);
  // out

  // define the swizzle mode
  using InputSmem = smem_row<T, 0, 0, 0, 64, 64, 64>;
  InputSmem input_smem(shared_input);
  InputSmem input_smem_buffer(shared_input);

  using WeightSmem = smem_row<T, 0, 0, 0, 64, 64, 64>;
  WeightSmem input_weight_smem(shared_weight);
  WeightSmem input_weight_smem_buffer(shared_weight);

  using A_DESC = wgmma::mma_descriptor<InputSmem>;
  using B_DESC = wgmma::mma_descriptor<WeightSmem>;

  smem_row<T, 3, 4, 3, 64, 64, 64> mm_output_smem(mm_output);
  float s_frag[32];
  // define barries
  Barrier *input_barrier = reinterpret_cast<Barrier *>(smem + 50000);
  Barrier *weight_barrier = reinterpret_cast<Barrier *>(smem + 60000);
  Barrier *compute_done = reinterpret_cast<Barrier *>(smem + 70000);

  // init the barriers and launch the first group of copy
  if (threadIdx.x == 0) {
    for (int i = 0; i < Kstages; i++) {
      initialize_barrier(input_barrier[i], 1);
      initialize_barrier(weight_barrier[i], 1);
      initialize_barrier(compute_done[i], 1);
    }

    // launch the first group of copy
    for (int i = 0; i < Kstages - 1; i++) {
      int4 tma_coords = {0, 0, 0, 0};
      set_barrier_transaction_bytes(input_barrier[i], 8192);
      tma_a.tma_cp_async(input_barrier[i], input_smem_buffer(0, 0), tma_coords);
      set_barrier_transaction_bytes(weight_barrier[i], 8192);
      tma_b.tma_cp_async(
          weight_barrier[i], input_weight_smem_buffer(0, 0), tma_coords);
      // mv smem ptr to next buffer
      input_smem_buffer.set_ptr(shared_input + 8192);
      input_weight_smem_buffer.set_ptr(shared_weight + 8192);
    }
  }

  __syncthreads();

  // warp specialization data movement warpgroup
  if (warpgroup_id == NUM_WARPGROUPS - 1) {

    // wg_decrease_regs<32>();
    if (lane_id() == 0 && warp_idx == (NUM_WARPGROUPS * WARPGROUP_WARPS - 4)) {
      for (int i = Kstages - 2; i < 63; i++) {
        // get cord, copy
        int4 tma_coords = {0, 0, 0, 0};
        set_barrier_transaction_bytes(input_barrier[(i + 1) % Kstages], 8192);
        tma_a.tma_cp_async(input_barrier[(i + 1) % Kstages],
                           input_smem_buffer(0, 0),
                           tma_coords);
        set_barrier_transaction_bytes(weight_barrier[(i + 1) % Kstages], 8192);

        tma_b.tma_cp_async(weight_barrier[(i + 1) % Kstages],
                           input_weight_smem_buffer(0, 0),
                           tma_coords);

        wait(compute_done[(i) % Kstages], (i / Kstages) % 2);

        input_smem_buffer.set_ptr(shared_input +
                                  ((i + Kstages) % Kstages) * 8192);
        input_weight_smem_buffer.set_ptr(shared_weight +
                                         ((i + Kstages) % Kstages) * 8192);
      }
    }
  } else {
    // warp specialization compute warpgroup
    // wg_increase_regs<160>();
    for (int i = 0; i < 64; i++) {
      // wait input, weight
      wait(input_barrier[i % Kstages], ((i / Kstages) % 2));
      wait(weight_barrier[i % Kstages], ((i / Kstages) % 2));

      A_DESC a_desc(input_smem(0, 0));
      B_DESC b_desc(input_weight_smem(0, 0));

      wgmma::warpgroup_arrive();
      // wgmma
      wgmma::mma<bfloat16,
                 64,
                 64,
                 16,
                 InputSmem,
                 WeightSmem,
                 A_DESC,
                 B_DESC,
                 false,
                 true>(s_frag, a_desc, b_desc);
      wgmma::mma_commit_group();
      wgmma::mma_async_wait();

      // flip compute done
      if (threadIdx.x == 0) {
        arrive(compute_done[i % Kstages], 1);
      }
    }
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16:~:text=The%20layout%20of%20the%20fragments%20held%20by%20different%20threads%20is%20shown%20in%20Figure%20149.
    // write back to shared memory

#pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
      int row = (warp_idx % 4) * 16 + (i % 2) * 8 + idx_in_warp / 4;
      int col = (i / 2) * 8 + (idx_in_warp % 4) * 2;
      mm_output_smem.at(row, col) = bfloat16(s_frag[i * 2]);
      mm_output_smem.at(row, col + 1) = bfloat16(s_frag[i * 2 + 1]);
    }

    wg_sync<NUM_THREADS * CONSUMER_WARPGROUPS>(8);
    // copy back to dmem
    if (warp_idx % 4 == 0) {
      tma_out.tma_store_async(mm_output_smem(0, 0), {0, 0});
      if (lane_id == 0) {
        store_commit_group();
        __syncwarp();
      }
    }
    if (lane_id == 0) {
      store_async_wait();
      __syncwarp();
    }
  }
}

} // namespace kernel