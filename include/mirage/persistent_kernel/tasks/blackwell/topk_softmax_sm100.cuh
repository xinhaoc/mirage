#pragma once
#include <cstdio>
#include <iostream>

// Use Thrust to handle host/device allocations
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

// topk_reduce includes
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

// mirage includes
#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "utils.cuh"

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k. 3) This implementation assumes 8 warps are being used.
*/

namespace kernel {

static constexpr int WARP_SIZE = 32;

// ====================== Fused TopK softmax kernel
// =============================== This kernel fuses the softmax, max and argmax
// into a single kernel. Block size is strictly 256 (8 warps): dim3
// block(WARP_SIZE*WARPS_PER_CTA, 1, 1)
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __noinline__ void topk_softmax_task_impl(
    T const *__restrict__ input, // [num_rows, NUM_EXPERTS]
    bool const *__restrict__ finished,
    float *__restrict__ output, // [num_rows, k]
    int const num_rows,
    int const k,
    int *__restrict__ mpk_routing_indices, // [NUM_EXPERTS, num_rows] laid out
                                           // as expert-major: expert * num_rows
                                           // + row
    int *__restrict__ mpk_expert_mask,     // [NUM_EXPERTS]
    int const start_expert,
    int const end_expert,
    bool const renormalize) {
  // Compile-time checks
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW =
      ELTS_PER_ROW / VPT; // subgroup size in a warp
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");

  // Work partitioning
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP =
      ELTS_PER_WARP / ELTS_PER_ROW; // rows each warp processes
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  int const warp_idx = threadIdx.x / WARP_SIZE;
  int const lane_idx = threadIdx.x % WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;

  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  int const thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row < num_rows) {

    bool const row_is_active = finished ? !finished[thread_row] : true;

    // Compute per-thread read pointers
    T const *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    int const thread_group_idx = lane_idx % THREADS_PER_ROW;
    int const first_elt_read_by_thread =
        thread_group_idx * (BYTES_PER_LDG / sizeof(T));
    T const *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;
    T row_chunk_temp[VPT];
    AccessType *row_chunk_vec_ptr =
        reinterpret_cast<AccessType *>(&row_chunk_temp);
    AccessType const *vec_thread_read_ptr =
        reinterpret_cast<AccessType const *>(thread_read_ptr);

    // Vectorized loads across the row
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    cutlass::NumericConverter<float, T> converter;

    float row_chunk[VPT];
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = converter(row_chunk_temp[ii]);
    }

    // Max reduction within subgroup
    float thread_max = row_chunk[0];
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = max(thread_max, row_chunk[ii]);
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other =
          __shfl_xor_sync(0xffffffff, thread_max, mask, THREADS_PER_ROW);
      thread_max = max(thread_max, other);
    }

    // Softmax numerator and sum within subgroup
    float row_sum = 0.f;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = expf(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask, THREADS_PER_ROW);
    }

    float const inv_row_sum = 1.f / row_sum;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = row_chunk[ii] * inv_row_sum;
    }

    // Fused Top-K selection within subgroup
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float row_sum_for_renormalize = 0.f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      float max_val = row_chunk[0];
      int expert = start_col;
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

      // Argmax reduce across subgroup with index tie-breaker (prefer lower
      // index)
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max =
            __shfl_xor_sync(0xffffffff, max_val, mask, THREADS_PER_ROW);
        int other_expert =
            __shfl_xor_sync(0xffffffff, expert, mask, THREADS_PER_ROW);
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Write out the selected top-k value/index (one thread per subgroup
      // writes)
      if (thread_group_idx == 0) {
        bool const node_uses_expert =
            expert >= start_expert && expert < end_expert;
        bool const should_process_row = row_is_active && node_uses_expert;
        int const out_idx = k * thread_row + k_idx;
        output[out_idx] = max_val;
        // indices[out_idx] =
        //     should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        row_sum_for_renormalize += max_val;
        // Optionally populate MPK routing structures
        if (should_process_row && mpk_routing_indices != nullptr &&
            mpk_expert_mask != nullptr) {
          int const local_expert = expert - start_expert;
          // Write 1-based rank into routing indices; stride by num_rows per
          // expert
          mpk_routing_indices[local_expert * num_rows + thread_row] = k_idx + 1;
          // // Mark expert as active (idempotent). Atomic to avoid races across
          // rows. atomicExch(&mpk_expert_mask[local_expert], 1); // race
          // condition might be okay here
          mpk_expert_mask[local_expert] = 1;
        }
      }

      // Blank out the winning value for the next iteration
      if (k_idx + 1 < k) {
        int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        int const thread_to_clear_in_group =
            (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
        if (thread_group_idx == thread_to_clear_in_group) {
          int const offset_for_expert = expert % ELTS_PER_LDG;
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
              -10000.f;
        }
      }
    }

    // Optional renormalization of top-k weights
    if (renormalize && thread_group_idx == 0) {
      float inv = 1.f / row_sum_for_renormalize;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        output[out_idx] = output[out_idx] * inv;
      }
    }
  }
  __syncthreads();
}

namespace detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                "");
  static constexpr int
      VECs_PER_THREAD = (EXPERTS / (ELTS_PER_LDG * WARP_SIZE)) > 0
                            ? (EXPERTS / (ELTS_PER_LDG * WARP_SIZE))
                            : 1;
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / (EXPERTS / VPT);
};
} // namespace detail

} // namespace kernel
