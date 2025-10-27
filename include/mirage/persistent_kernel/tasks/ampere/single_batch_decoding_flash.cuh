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
#include "common.h"
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "norm.cuh"
#include "reduction.cuh"
#include "rotary_embedding.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"
namespace kernel {

// kernel Input: 9X128, K_Cache: 4KX128, V_Cache:4KX128
// Load Q = 8 X 128, K = 1 X 128, V = 1 X 128
// load K into K_Cache, V into V_cache
template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int WEIGHT_STRIDE>
__device__ __forceinline__ void
    single_batch_decoding_kernel(void const *qkv_ptr,
                                 void *k_cache_ptr,
                                 void *v_cache_ptr,
                                 void *output_ptr,
                                 size_t seq_len,
                                 bool qk_norm,
                                 bool rotary_emd,
                                 void const *qnorm_weight_ptr,
                                 void const *knorm_weight_ptr,
                                 void const *cos_ptr,
                                 void const *sin_ptr,
                                 float q_eps,
                                 float k_eps) {
  constexpr size_t MAX_SEQ_LEN = 512;
  constexpr size_t KV_CHUNK_SIZE = 64;
  float const sm_scale = (1.f / sqrt((float)HEAD_DIM));

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  size_t num_iterations = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(seq_len, KV_CHUNK_SIZE);
  int cp_finished_seq_len = curr_iter_len;
  int last_seq_len = curr_iter_len;

  __restrict__ T const *d_q = static_cast<T const *>(qkv_ptr);
  __restrict__ T const *d_k =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * NUM_Q_HEADS;
  __restrict__ T const *d_v =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * (NUM_Q_HEADS + NUM_KV_HEADS);
  T __restrict__ *d_k_cache = static_cast<T *>(k_cache_ptr);
  T __restrict__ *d_v_cache = static_cast<T *>(v_cache_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  dmem_row_const<T, NUM_Q_HEADS, 128, 128> q_dmem(d_q);
  dmem_row_const<T, 1, 128, 128> k_dmem(d_k);
  dmem_row_const<T, 1, 128, 128> v_dmem(d_v);
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> k_cache_dmem(d_k_cache);
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> v_cache_dmem(d_v_cache);
  dmem_row<T, NUM_Q_HEADS, 128, 128> output_dmem(d_output);
  extern __shared__ char smem[];

  // copy input
  // T *shared_q = (T *)(smem + 128);
  // copy weight
  T *shared_k = (T *)(smem + 1920);
  T *shared_k_buffer = (T *)(smem + 18304);

  T *shared_v = (T *)(smem + 34688);
  T *shared_v_buffer = (T *)(smem + 51072);
  // intermidiate
  T *shared_output = (T *)(smem + 128);
  T *zero_buf = (T *)(smem);

  // flashattn metadata
  float *d_smem = (float *)(smem + 67456);
  float *max_smem = (float *)(smem + 67968);
  float *o_smem = (float *)(smem + 68480);

  float *qnorm_sum = (float *)(smem + 84864);
  float *knorm_sum = (float *)(smem + 84880);
  // define the swizzle mode

  // zero buffer
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);

  // using KSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using VSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using OSmem = smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128>;

  // KSmem k_cache_smem(shared_k);
  KSmem k_cache_smem_buffer(shared_k_buffer);
  VSmem v_cache_smem(shared_v);
  VSmem v_cache_smem_buffer(shared_v_buffer);
  OSmem output_smem(shared_output);

  // smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128> output_smem(shared_output);

  // todo, add a chunk assigned function
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // difference from here
  constexpr uint32_t vec_size = 8;
  constexpr uint32_t x_partition =
      HEAD_DIM / vec_size; // 16, partition along HEAD_DIM
  constexpr uint32_t y_partition =
      NUM_Q_HEADS / NUM_KV_HEADS; // 4, partition along NUM_Q_HEADS
  static_assert(y_partition ==
                4); // it is hard to partition when NUM_Q_HEADS is odd
  constexpr uint32_t z_partition =
      NUM_THREADS /
      (x_partition * y_partition); // 2, partition along SEQ_LENGTH

  constexpr uint32_t PIPELINE_STAGE = 2;

  size_t num_iterations = (seq_len + z_partition - 1) / z_partition;

  // K

  vector_t<T, vec_size> q_vec;

  uint32_t q_head_idx = (threadIdx.x / (x_partition)) % y_partition;
  uint32_t token_idx = threadIdx.x / (x_partition * y_partition);
  uint32_t headdim_idx = threadIDx.x % x_partition;

  uint32_t curren_token_idx = 0;

  // load q
  // Q: 4 * 128 -> (8 * 16) * 4 heads
  q_vec.load(d_q + q_head_idx * HEAD_DIM + headdim_idx * vec_size);

// preload K and V
// K,2 * 128 -> first 64threads, each thread ( 8 * 16 ), repeat 4 times
#pragma unroll
  for (uint32_t i = 0; i < (PIPELINE_STAGE - 1); ++i) {
    // load k
    load_smem(shared_k +
                  ((i * z_partition + token_idx) * NUM_Q_HEADS + q_head_idx) *
                      HEAD_DIM +
                  headdim_idx * vec_size,
              d_k + token_idx * HEAD_DIM + headdim_idx * vec_size);
    cp_async_fence();
    // load v
    load_smem(shared_v +
                  ((i * z_partition + token_idx) * NUM_Q_HEADS + q_head_idx) *
                      HEAD_DIM +
                  headdim_idx * vec_size,
              d_v + token_idx * HEAD_DIM + headdim_idx * vec_size);
    cp_async_fence();
  }

  // main loop
  float m = -inf;
  float d = 1.f;
  vector_t<float, vec_size> o;
  o.fill(0.0f);

  float s[y_partition];

  int stage_idx = 0;

  for (uint32_t i = 0; i < num_iterations; i += z_partition) {
    // wait for k finished
    cp_async_wait<2 * PIPELINE_STAGE - 1>();
    // compute qk

    float m_prev = m;
    __syncthreads();
    // load k
    for (uint32_t j = 0; j < y_partition; j++) {
      vector_t<T, vec_size> k_vec;
      k_vec.load(shared_k + (j * bdx + tx) * vec_size);
      s[j] = 0.0f;

      // local qk
#pragma unroll
      for (uint32_t k = 0; k < vec_size; k++) {
        s[j] += q_vec[k] * k_vec[k];
      }
      // reduction across same head(local 16 threads)
#pragma unroll
      for (uint32_t offset = (x_partition / 2); offset > 0; offset /= 2) {
        s[j] += math::shfl_xor_sync(s[j], offset);
      }

      // sm_scale
      s[j] *= sm_scale;

      // mask
      s[j] = (curren_token_idx + token_idx) <= seq_len ? s[j] : -inf;
      m = max(m, s[j]);

      // update o and d
      float o_scale = expf(m_prev - m);
      d *= o_scale;
#pragma unroll
      for (uint32_t j = 0; j < y_partition; ++j) {
        s[j] = expf(s[j] - m);
        d += s[j];
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        o[i] = o[i] * o_scale;
      }
    }
    __syncthreads();
    // load next k
    load_smem(
        shared_k +
            ((stage_idx * z_partition + token_idx) * NUM_Q_HEADS + q_head_idx) *
                HEAD_DIM +
            headdim_idx * vec_size,
        d_k + (curren_token_idx + z_partition + token_idx) * HEAD_DIM +
            headdim_idx * vec_size);
    cp_async_fence();

    cp_async_wait<2 * PIPELINE_STAGE - 1>();

// do v proj
#pragma unroll
    for (uint32_t j = 0; j < y_partition; ++j) {
      vector_t<T, vec_size> v_vec;
      v_vec.load(shared_v + (j * bdx + tx) * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        o[i] = o[i] + s[j] * v_vec[i];
      }
    }

    // load next v
    load_smem(
        shared_k +
            ((stage_idx * z_partition + token_idx) * NUM_Q_HEADS + q_head_idx) *
                HEAD_DIM +
            headdim_idx * vec_size,
        d_k + (curren_token_idx + z_partition + token_idx) * HEAD_DIM +
            headdim_idx * vec_size);
    cp_async_fence();


    stage_idx = (stage_idx + 1) % PIPELINE_STAGE;
    curren_token_idx += y_partition;
  }
  cp_async_wait<0>();
  __syncthreads();
  // final update
    o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      __syncthreads();

      //clean
      o.fill(0.0f);
      m = -inf;
      d = 1.f

#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vector_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }

    //store
    st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
}

} // namespace kernel