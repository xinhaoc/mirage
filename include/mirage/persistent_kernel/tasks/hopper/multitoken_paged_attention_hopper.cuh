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
 #include "../copy_sm80.cuh"
 #include "../dmem_layout.cuh"
 #include "../element_binary.cuh"
 #include "../element_unary.cuh"
 #include "../mma.cuh"
 #include "../norm.cuh"
 #include "../reduction.cuh"
 #include "../rotary_embedding.cuh"
 #include "../smem_layout.cuh"
 #include "../utils.cuh"
 #include "rotary_embedding_wg.cuh"
 #include "norm_wg.cuh"
 #include "smem_layout_tma.cuh"
 #include "tma_3d.cuh"
 namespace kernel {
 
 // NOTE(Jinchen): this task implements the paged attention where a causal mask
 // is applied. In each task, we process one request with one or more tokens
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
           typename TMA_PAGED_KV_CACHE,
           typename TMA_OUTPUT,
           int MAX_TOKENS = 8>
 __device__ __forceinline__ void multitoken_paged_attention_hopper_impl(
     const TMA_Q &tma_q,
     const TMA_KV &tma_k,
     const TMA_KV &tma_v,
     const TMA_PAGED_KV_CACHE &tma_paged_k_cache,
     const TMA_PAGED_KV_CACHE &tma_paged_v_cache,
     const TMA_OUTPUT &tma_output,
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
   constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;
 
   // NOTE(Jinchen): The input is a packed QKV tensor, which may contain
   // multiple tokens. The shape of the packed QKV tensor is
   // [num_tokens, head_dim * (num_qo_heads + num_kv_heads * 2)]
   // NOTE(Jinchen): assume the layout of KV Cache is NHD,
   // i.e., the shape of KV Cache is
   // [max_num_pages, page_size, num_kv_heads, head_dim]
 
   constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
   constexpr int KV_TILE_SIZE = 64;
   constexpr int MAX_PAGES_PER_REQUEST =
       (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
   constexpr int THREADS_PER_WARPGROUP = 128;
   constexpr int CONSUMER_WARPGROUPS = 1;
   constexpr int PRODUCER_WARPGROUPS = 1;
   constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
   constexpr int Kstages = 2;
   // NOTE(Jinchen): we use m16n16k16 mma to compute matrix multiplication
  //  constexpr int MMA_ITERS_M = (MAX_TOKENS * NUM_QO_PER_KV + 15) / 16;
  constexpr int MMA_ITERS_M = (MAX_TOKENS * NUM_QO_PER_KV + 63) / 64;
  constexpr int QSMEM_ROW = 64;
 
   // the scale factor for normalization in softmax
   float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
 
   int warp_idx = warp_id();
   int lane_idx = lane_id();
   int warpgroup_id = warp_idx / WARPGROUP_WARPS;
 
   int const first_token_pos = qo_indptr_buffer_ptr[request_id];
   int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
   // Exit the current task is number of query tokens is zero
   if (first_token_pos == last_token_pos) {
     return;
   }
   int const num_tokens = last_token_pos - first_token_pos;
   if (threadIdx.x == 128) {
     printf("num_tokens: %d\n", num_tokens);
   }
 
   // NOTE(Jinchen): to simplify the implementation, we assume that the metadata
   // of the paged KV cache includes the new tokens, i.e., spaces are allocated
   // before the request is processed, while the real data is copied into the
   // corresponding pages after that
   int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
   int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
   int const num_pages = last_page_pos - first_page_pos;
   int const seq_len = (num_pages - 1) * PAGE_SIZE +
                       paged_kv_last_page_len_buffer_ptr[request_id];
   // valid_lens = [seq_len - num_tokens + 1 + i for i in range(num_tokens)]
 
   // Load the paged KV indices into shared memory
   __shared__ int page_indices[MAX_PAGES_PER_REQUEST];
 #pragma unroll
   for (int i = threadIdx.x; i < num_pages * sizeof(int) / 16;
        i += NUM_THREADS) {
     __uint128_t const *src_ptr =
         reinterpret_cast<__uint128_t const *>(paged_kv_indices_buffer_ptr) + i;
     __uint128_t *dst_ptr = reinterpret_cast<__uint128_t *>(page_indices) + i;
     *dst_ptr = *src_ptr;
   }
   if (num_pages % (16 / sizeof(int)) != 0) {
     int tail_pages = num_pages % (16 / sizeof(int));
     int tail_offset = num_pages - tail_pages;
     for (int i = threadIdx.x; i < tail_pages; i += NUM_THREADS) {
       page_indices[tail_offset + i] =
           paged_kv_indices_buffer_ptr[first_page_pos + tail_offset + i];
     }
   }
   __syncthreads();
 
   T const *__restrict__ d_q =
       reinterpret_cast<T const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
   T const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
   T const *__restrict__ d_v = d_k + HEAD_DIM;
   T *__restrict__ d_paged_k_cache = reinterpret_cast<T *>(paged_k_cache_ptr);
   T *__restrict__ d_paged_v_cache = reinterpret_cast<T *>(paged_v_cache_ptr);
   T *__restrict__ d_output =
       reinterpret_cast<T *>(output_ptr) + first_token_pos * O_STRIDE;
 
   //  DTensors' layouts
   using QDmem =
       dmem_row_const<T, MAX_TOKENS, HEAD_DIM * NUM_QO_PER_KV, QKV_STRIDE>;
   using KVDmem = dmem_row_const<T, MAX_TOKENS, HEAD_DIM, QKV_STRIDE>;
   using KVCacheDmem = dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_CACHE_STRIDE>;
   using ODmem = dmem_row<T, MAX_TOKENS, HEAD_DIM * NUM_QO_PER_KV, O_STRIDE>;
 
   QDmem q_dmem(d_q);
   KVDmem k_dmem(d_k), v_dmem(d_v);
   KVCacheDmem paged_k_cache_dmem(d_paged_k_cache),
       paged_v_cache_dmem(d_paged_v_cache);
   ODmem o_dmem(d_output);
 
   // STensors' offsets and sizes
   constexpr size_t ZERO_BUFFER_OFFSET = 0;
   constexpr size_t ZERO_BUFFER_SIZE = sizeof(T) * 8;
 
   constexpr size_t S_Q_OFFSET = (ZERO_BUFFER_OFFSET + ZERO_BUFFER_SIZE + 127) / 128 * 128;
   constexpr size_t S_Q_SIZE = sizeof(T) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;
 
   constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
   constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;
 
   constexpr size_t S_K_BUFFER_OFFSET = S_K_OFFSET + S_K_SIZE;
   constexpr size_t S_K_BUFFER_SIZE = S_K_SIZE;
 
   constexpr size_t S_V_OFFSET = S_K_BUFFER_OFFSET + S_K_BUFFER_SIZE;
   constexpr size_t S_V_SIZE = S_K_SIZE;
 
   constexpr size_t S_V_BUFFER_OFFSET = S_V_OFFSET + S_V_SIZE;
   constexpr size_t S_V_BUFFER_SIZE = S_K_SIZE;
 
   constexpr size_t S_O_OFFSET = S_V_BUFFER_OFFSET + S_V_BUFFER_SIZE;
   constexpr size_t S_O_SIZE = S_Q_SIZE;
 
   // align to size of float
   constexpr size_t S_Q_NORM_SUM_OFFSET =
       ((S_O_OFFSET + S_O_SIZE + sizeof(float) - 1) &
        ~size_t(sizeof(float) - 1));
   constexpr size_t S_Q_NORM_SUM_SIZE =
       sizeof(float) * 4; // 4 floats for 4 warps
 
   constexpr size_t S_K_NORM_SUM_OFFSET =
       S_Q_NORM_SUM_OFFSET + S_Q_NORM_SUM_SIZE;
   constexpr size_t S_K_NORM_SUM_SIZE = sizeof(float) * 4;
 
   constexpr size_t S_M_BUFFER_OFFSET = S_K_NORM_SUM_OFFSET + S_K_NORM_SUM_SIZE;
   constexpr size_t S_M_BUFFER_SIZE =
       sizeof(float) * MMA_ITERS_M * NUM_THREADS * 2;
 
   constexpr size_t S_D_BUFFER_OFFSET = S_M_BUFFER_OFFSET + S_M_BUFFER_SIZE;
   constexpr size_t S_D_BUFFER_SIZE =
       sizeof(float) * MMA_ITERS_M * NUM_THREADS * 2;
 
   constexpr size_t S_O_BUFFER_OFFSET = S_D_BUFFER_OFFSET + S_D_BUFFER_SIZE;
   constexpr size_t S_O_BUFFER_SIZE =
       sizeof(float) * MMA_ITERS_M * NUM_THREADS * 64;
   constexpr size_t S_TOTAL_OFFSET = S_O_BUFFER_OFFSET + S_O_BUFFER_SIZE;
   static_assert(S_TOTAL_OFFSET <= 224 * 1024);
 
   extern __shared__ char smem_ptr[];
 
   T *zero_buf = reinterpret_cast<T *>(smem_ptr + ZERO_BUFFER_OFFSET);
   clear_smem_buffer<T, 8>(zero_buf);
   uintptr_t smem = (reinterpret_cast<uintptr_t>(zero_buf) + 127) / 128 * 128;

   T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET); 
   T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
   T *s_k_buffer = reinterpret_cast<T *>(smem + S_K_BUFFER_OFFSET);
   T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
   T *s_v_buffer = reinterpret_cast<T *>(smem + S_V_BUFFER_OFFSET);
   T *s_o = reinterpret_cast<T *>(smem + S_O_OFFSET);
   float *s_q_norm_sum = reinterpret_cast<float *>(smem + S_Q_NORM_SUM_OFFSET);
   float *s_k_norm_sum = reinterpret_cast<float *>(smem + S_K_NORM_SUM_OFFSET);
   float *s_m_buffer = reinterpret_cast<float *>(smem + S_M_BUFFER_OFFSET);
   float *s_d_buffer = reinterpret_cast<float *>(smem + S_D_BUFFER_OFFSET);
   float *s_o_buffer = reinterpret_cast<float *>(smem + S_O_BUFFER_OFFSET);
 
   // STensors' layouts
  //  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  //  using QOSmem = smem_row<T, 0, 0, 0, 64, HEAD_DIM, HEAD_DIM>;
  //  using KVSmem = smem_row<T, 0, 0, 0, KV_TILE_SIZE, HEAD_DIM, HEAD_DIM>;
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using QOSmem =
      smem_tma<T, 0, 0, 0, MAX_TOKENS * NUM_QO_PER_KV, 64, (HEAD_DIM + 63) / 64>;
  using KVSmem = smem_tma<T, 0, 0, 0, KV_TILE_SIZE, 64, (HEAD_DIM + 63) / 64>;
  
  using Q_DESC = wgmma::mma_descriptor<QOSmem>;
  using KV_DESC = wgmma::mma_descriptor<KVSmem>;

   ZeroBufferSmem zero_buffer(zero_buf);
   QOSmem q_smem(s_q), o_smem(s_o);
   if (threadIdx.x == 0) {
    printf("s_q::ROW, INNER_COL, OUTER_COL: %llu, %llu, %llu\n", QOSmem::ROW, QOSmem::INNER_COL, QOSmem::OUTER_COL);
    printf("s_q::STRIDE_ROW, STRIDE_OUTER_COL: %llu, %llu\n", QOSmem::STRIDE_ROW, QOSmem::STRIDE_OUTER_COL);
   }
   KVSmem k_smem(s_k), v_smem(s_v);
   KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);
 
   int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
   int curr_iter_len = min(seq_len, KV_TILE_SIZE);
   int cp_finished_seq_len = 0;
   // assert no leafover to be handled when loading qkv
   static_assert(HEAD_DIM % CP_CHUNK_SIZE == 0);
 
   // Currently assume that PAGE_SIZE is a multiplier of KV_TILE_SIZE
   // so that we access a single page in one iteration
   static_assert(PAGE_SIZE % KV_TILE_SIZE == 0);
 
   // 16*128 = 2048
 
   const int TMA_TRANS_BYTES_Q = num_tokens * NUM_QO_PER_KV * HEAD_DIM * sizeof(T);
   const int TMA_TRANS_BYTES_KV = KV_TILE_SIZE * HEAD_DIM * sizeof(T);
 
   //  define barries
   Barrier *q_barrier = reinterpret_cast<Barrier *>(smem + 170000);
   Barrier *kv_barrier = reinterpret_cast<Barrier *>(smem + 180000);
   Barrier *compute_done = reinterpret_cast<Barrier *>(smem + 200000);
 
   // init barrier
   if (threadIdx.x == 0) {
     for (int i = 0; i < Kstages; i++) {
       initialize_barrier(q_barrier[i], 1);
       initialize_barrier(kv_barrier[i], 1);
       initialize_barrier(compute_done[i], 1);
     }
   }
   __syncthreads();
 
   if (warpgroup_id == NUM_WARPGROUPS - 1) {
 // prefetch
 // load q

    if (lane_idx == 0 && warp_idx % 4 == 0) {
      set_barrier_transaction_bytes(q_barrier[0], TMA_TRANS_BYTES_Q);
      printf("q_smem(0, 0) = %p\n", q_smem(0, 0));
      for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
          int g_row = token_idx * (NUM_QO_PER_KV + 2 * NUM_KV_HEADS);
          // qsmem (4, 0) gmem(0, )
          // 4 * 128 * 2
          printf("q_smem(token_idx * NUM_QO_PER_KV, 0) = %p\n", q_smem(token_idx * NUM_QO_PER_KV, 0));
          tma_q.tma_cp_async(q_barrier[0], q_smem(token_idx * NUM_QO_PER_KV, 0), {0, g_row});
      }
    }
 
     wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);
 
     // load k and v
     int page_idx_0 = page_indices[0];

     if (lane_idx == 0 && warp_idx % 4 == 0) {
      int begin = cp_finished_seq_len;
      int end = begin + curr_iter_len;
      int boundary = seq_len - num_tokens;
      int cache_rows =
        (begin >= boundary) ? 0 :
        (end   <= boundary) ? (end - begin)
                            : (boundary - begin);

      int qkv_rows   = next_iter_len - cache_rows;

      if (cache_rows > 0) {
        int page     = page_indices[ begin / PAGE_SIZE ];
        int row_in   = begin % PAGE_SIZE;
        int coords[3]= {0, row_in, page};
        tma_paged_k_cache.tma_cp_async(kv_barrier[slot], k_smem(0,0), coords);
        tma_paged_v_cache.tma_cp_async(kv_barrier[slot], v_smem(0,0), coords);
      }

      if (qkv_rows > 0) {
        int src_row   = begin + cache_rows - boundary;
        int coords2D[2] = {0, src_row};
        tma_k.tma_cp_async(kv_barrier[slot], k_smem(cache_rows, 0), coords2D);
        tma_v.tma_cp_async(kv_barrier[slot], v_smem(cache_rows, 0), coords2D);
      }

     }
//  #pragma unroll
//      for (int chunk_idx = threadIdx.x;
//           chunk_idx < curr_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
//           chunk_idx += NUM_THREADS) {
//        int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
//        int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
//        if (dst_row + cp_finished_seq_len < seq_len - num_tokens) {
//          // load from KV Cache
//          // int page_idx = page_indices[(dst_row + cp_finished_seq_len) /
//          // PAGE_SIZE];
//          int page_offset = (dst_row + cp_finished_seq_len) % PAGE_SIZE;
//          int src_row = page_idx_0 * PAGE_SIZE + page_offset;
//          load_smem(k_buffer_smem(dst_row, col),
//                    paged_k_cache_dmem(src_row, col));
//          load_smem(v_buffer_smem(dst_row, col),
//                    paged_v_cache_dmem(src_row, col));
//        } else {
//          // load from QKV
//          int src_row = dst_row + cp_finished_seq_len - (seq_len - num_tokens);
//          load_smem(k_buffer_smem(dst_row, col), k_dmem(src_row, col));
//          load_smem(v_buffer_smem(dst_row, col), v_dmem(src_row, col));
//        }
//      }

     cp_async_fence();
     cp_finished_seq_len += curr_iter_len;
     if (threadIdx.x == 128) {
       printf("finish preloading k and v\n");
     }
 
     for (int iter = 0; iter < num_iters; iter++) {
       int phase = (iter / Kstages) % 2;
       int slot = iter % Kstages;
       // wait(compute_done[0], phase);
       if (threadIdx.x == 128) {
         printf("start loading iter %d\n", iter);
       }
 
       int next_iter_len = iter + 1 < num_iters
                               ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                               : 0;
       if (next_iter_len > 0) {
         int page_idx = page_indices[cp_finished_seq_len / PAGE_SIZE];
 #pragma unroll
         for (int chunk_idx = threadIdx.x;
              chunk_idx < curr_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
              chunk_idx += NUM_THREADS) {
           int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
           int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
           if (dst_row + cp_finished_seq_len < seq_len - num_tokens) {
             // load from KV Cache
             // int page_idx =
             //    page_indices[(dst_row + cp_finished_seq_len) / PAGE_SIZE];
             int page_offset = (dst_row + cp_finished_seq_len) % PAGE_SIZE;
             int src_row = page_idx * PAGE_SIZE + page_offset;
             load_smem(k_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
             load_smem(v_smem(dst_row, col), paged_v_cache_dmem(src_row, col));
           } else {
             // load from QKV
             int src_row =
                 dst_row + cp_finished_seq_len - (seq_len - num_tokens);
             load_smem(k_smem(dst_row, col), k_dmem(src_row, col));
             load_smem(v_smem(dst_row, col), v_dmem(src_row, col));
           }
         }
         cp_async_fence();
         cp_async_wait<1>();
         cp_finished_seq_len += next_iter_len;
 
         // if (warp_idx == num_warpgroups - 4 && lane_idx == 0) {
         //   arrive(kv_barrier[0], 1);
         // }
       } else {
         cp_async_wait<0>();
         // if (warp_idx == num_warpgroups - 4 && lane_idx == 0) {
         //   arrive(kv_barrier[0], 1);
         // }
       }
 
       if (threadIdx.x == 128) {
         printf("finish loading iter %d\n", iter);
         printf("start waiting for compute iter,phase is %d\n", phase);
       }
 
       // arrive barrier
       if (lane_idx == 0 && warp_idx % 4 == 0) {
         arrive(kv_barrier[slot], 1);
       }
 
       wait(compute_done[slot], phase);
       if (threadIdx.x == 128) {
         printf("finish waiting for compute iter, phase is %d\n", phase);
       }
 
       // rotate the buffers
       if ((iter & 0x1) == 0) {
         k_smem.set_ptr(s_k_buffer);
         k_buffer_smem.set_ptr(s_k);
         v_smem.set_ptr(s_v_buffer);
         v_buffer_smem.set_ptr(s_v);
       } else {
         k_smem.set_ptr(s_k);
         k_buffer_smem.set_ptr(s_k_buffer);
         v_smem.set_ptr(s_v);
         v_buffer_smem.set_ptr(s_v_buffer);
       }
       wg_sync<THREADS_PER_WARPGROUP * PRODUCER_WARPGROUPS>(8);
     }
 
   } else {
 
     float m_local[MMA_ITERS_M][2];
 #pragma unroll
     for (int m = 0; m < MMA_ITERS_M; m++) {
       m_local[m][0] = -inf;
       m_local[m][1] = -inf;
     }
     float d[MMA_ITERS_M][2];
 #pragma unroll
     for (int m = 0; m < MMA_ITERS_M; m++) {
       d[m][0] = 1.f;
       d[m][1] = 1.f;
     }
     float o[MMA_ITERS_M][HEAD_DIM / 16][8];
 #pragma unroll
     for (int m = 0; m < MMA_ITERS_M; m++) {
 #pragma unroll
       for (int n = 0; n < HEAD_DIM / 16; n++) {
         clear_8_floats(o[m][n]);
       }
     }
 
     wait(q_barrier[0], 0);

#if 0
     if (threadIdx.x == 0) {
      for (int i = 0; i < num_tokens * NUM_QO_PER_KV * HEAD_DIM; i++) {
        if (i % 64 == 0) {
          printf("\n i / 64 = %d\n", i / 64);
        }
          printf("%f ", (float)q_smem.at(i));
      }
      printf("\n\nviewed as 16x128 matrix\n");
      for (int i = 0; i < num_tokens * NUM_QO_PER_KV; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            printf("%f ", (float)q_smem.at(i, j));
          
        }
        printf("\n");
      }
     }
#endif
     for (int iter = 0; iter < num_iters; iter++) {
       
       int next_iter_len = iter + 1 < num_iters
       ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
       : 0;
       
       int phase = (iter / Kstages) % 2;
       int slot = iter % Kstages;
       if (threadIdx.x == 0) {
         printf("start compute iter %d\n", iter);
         printf("consumer wait for kv barrier, phase is %d\n", phase);
       }
       wait(kv_barrier[slot], phase);

        // rotate the buffers
        if ((iter & 0x1) == 0) {
          k_smem.set_ptr(s_k_buffer);
          k_buffer_smem.set_ptr(s_k);
          v_smem.set_ptr(s_v_buffer);
          v_buffer_smem.set_ptr(s_v);
        } else {
          k_smem.set_ptr(s_k);
          k_buffer_smem.set_ptr(s_k_buffer);
          v_smem.set_ptr(s_v);
          v_buffer_smem.set_ptr(s_v_buffer);
        }
 
 
       int kv_tokens_to_process = min(
           curr_iter_len,
           max(iter * KV_TILE_SIZE + curr_iter_len - (seq_len - num_tokens), 0));
       int first_kv_token_to_process =
           iter * KV_TILE_SIZE + curr_iter_len - kv_tokens_to_process;
       if (qk_norm) {
         // Q norm
         if (iter == 0) {
           rms_norm_wg<T, QOSmem, NUM_QO_PER_KV, HEAD_DIM, THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
               q_smem,
               static_cast<T const *>(q_norm_weight_ptr),
               s_q_norm_sum,
               q_eps,
               num_tokens /*window_size*/,
               0 /*token_offset*/,
               rope,
               static_cast<T const *>(cos_ptr) +
                   (seq_len - num_tokens) * HEAD_DIM,
               static_cast<T const *>(sin_ptr) +
                   (seq_len - num_tokens) * HEAD_DIM,
               9);
         }
         // K norm
         if (kv_tokens_to_process > 0) {
           rms_norm_wg<T, KVSmem, 1, HEAD_DIM, THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
               k_smem,
               static_cast<T const *>(k_norm_weight_ptr),
               s_k_norm_sum,
               k_eps,
               kv_tokens_to_process /*window_size*/,
               curr_iter_len - kv_tokens_to_process,
               rope,
               static_cast<T const *>(cos_ptr) +
                   first_kv_token_to_process * HEAD_DIM,
               static_cast<T const *>(sin_ptr) +
                   first_kv_token_to_process * HEAD_DIM,
               9);
         }
       } else if (rope) {
         if (iter == 0) {
 #pragma unroll
           for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
             // q rope
             rotary_embedding_wg<T, QOSmem, NUM_QO_PER_KV, HEAD_DIM, THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
                 q_smem,
                 static_cast<T const *>(cos_ptr) +
                     (token_idx + seq_len - num_tokens) * HEAD_DIM,
                 static_cast<T const *>(sin_ptr) +
                     (token_idx + seq_len - num_tokens) * HEAD_DIM,
                 token_idx * NUM_QO_PER_KV,
                 9);
           }
         }
         if (kv_tokens_to_process > 0) {
           for (int token_idx = 0; token_idx < kv_tokens_to_process;
                token_idx++) {
             // k rope
             rotary_embedding_wg<T, KVSmem, 1, HEAD_DIM, THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
                 k_smem,
                 static_cast<T const *>(cos_ptr) +
                     (token_idx + first_kv_token_to_process) * HEAD_DIM,
                 static_cast<T const *>(sin_ptr) +
                     (token_idx + first_kv_token_to_process) * HEAD_DIM,
                 token_idx + curr_iter_len - kv_tokens_to_process,
                 9);
           }
         }
       }
 
       // printf("finish update k and v\n");
 
       wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);
 
       // update the KV Cache
       if (kv_tokens_to_process > 0) {
         int page_idx = page_indices[first_kv_token_to_process / PAGE_SIZE];
         for (int elem_idx = threadIdx.x;
              elem_idx < kv_tokens_to_process * HEAD_DIM;
              elem_idx += NUM_THREADS) {
           int token_idx = elem_idx / HEAD_DIM;
           int col = elem_idx % HEAD_DIM;
           // int page_idx = page_indices[(token_idx + first_kv_token_to_process)
           // / PAGE_SIZE];
           int page_offset = (token_idx + first_kv_token_to_process) % PAGE_SIZE;
           int src_row = (token_idx + first_kv_token_to_process) % KV_TILE_SIZE;
           int dst_row = page_idx * PAGE_SIZE + page_offset;
           paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
           paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
         }
       }
       // printf("finish update KV Cache\n");
 
       // printf("start compute X = QK^T\n");
       // compute X = QK^T
       // NOTE(Jinchen): we use m16n16k16 mma, and let warp layout be
       // 1x4x1, so mma iterates over m and k dimensions
      //  float x_frag_f[MMA_ITERS_M][8];
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          clear_8_floats(x_frag_f[m]);
//        }
//        uint32_t q_frag[4], kt_frag[4];
 
//        int kt_col = (warp_idx << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          int q_row = (m << 4) + (lane_idx & 0xF);
//  #pragma unroll
//          for (int k = 0; k < HEAD_DIM / 16; k++) {
//            int q_col = (k << 4) + ((lane_idx >> 4) << 3);
//            int kt_row = (k << 4) + (((lane_idx & 0xF) >> 3) << 3);
//            T *src_ptr_Q = q_row < num_tokens * NUM_QO_PER_KV
//                               ? q_smem(q_row, q_col)
//                               : zero_buffer(0, 0);
//            T *src_ptr_KT = kt_col < curr_iter_len ? k_smem(kt_col, kt_row)
//                                                   : zero_buffer(0, 0);
//            ldsm(src_ptr_Q, q_frag);
//            ldsm(src_ptr_KT, kt_frag);
//            mma_m16n16k16_bf16bf16bf32(x_frag_f[m], q_frag, kt_frag, x_frag_f[m]);
//          }
//        }

// float x_frag_f[MMA_ITERS_M][8];
// #pragma unroll
//     for (int m = 0; m < MMA_ITERS_M; m++) {
//        clear_8_floats(x_frag_f[m]);
//     }
//     for (int m = 0; m < MMA_ITERS_M; m++) {
//       for (int k = 0; k < HEAD_DIM / 64; k++) {
//         Q_DESC q_desc(q_smem(m * 64, k * 64));
//         KV_DESC k_desc(k_smem(0, k * 64));

//         wgmma::warpgroup_arrive();
//         wgmma::mma<T, 64, 64, 16, QOSmem, KVSmem, Q_DESC, KV_DESC, false, false>(x_frag_f[m], q_desc, k_desc);
//         wgmma::mma_commit_group();
//         wgmma::mma_async_wait();
//       }
//     }

       //  __syncthreads();
//        wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);
 
//        // update m_local: get partial max
//        // NOTE(Jinchen): each thread maintains MMA_ITERS_M * 2 partial max
//        // values. For a given m, the first value is the maximum of
//        // x_frag_f[m][0, 1, 4, 5], and the second value is the maximum of
//        // x_frag_f[m][2, 3, 6, 7]
//        float m_prev[MMA_ITERS_M][2];
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          m_prev[m][0] = m_local[m][0];
//          m_prev[m][1] = m_local[m][1];
//  #pragma unroll
//          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
//            // row_base = (m * 16) + (lane_idx / 4)
//            // col_base = (warp_idx * 16) + ((lane_idx % 4) * 2)
//            // row_offset = ((frag_idx % 4) / 2) * 8
//            // col_offset = ((frag_idx / 4) * 8) + (frag_idx % 2)
//            int row = (m << 4) + (lane_idx >> 2) + (((frag_idx & 0x3) >> 1) << 3);
//            int col = (warp_idx << 4) + ((lane_idx & 0x3) << 1) +
//                      ((frag_idx >> 2) << 3) + (frag_idx & 0x1);
//            int token_idx = row / NUM_QO_PER_KV;
//            bool is_valid =
//                (row < num_tokens * NUM_QO_PER_KV) &&
//                (col + iter * KV_TILE_SIZE <= token_idx + seq_len - num_tokens);
//            x_frag_f[m][frag_idx] = is_valid ? x_frag_f[m][frag_idx] : -inf;
//            m_local[m][(frag_idx & 0x3) >> 1] =
//                max(m_local[m][(frag_idx & 0x3) >> 1], x_frag_f[m][frag_idx]);
//          }
//        }
 // printf("start update m_local\n");
 // update m_local: get local max across 4 threads in a row
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x1));
//          m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x2));
//          m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x1));
//          m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x2));
//        }
 
//        float rescale[MMA_ITERS_M][2];
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          rescale[m][0] =
//              expf(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
//          rescale[m][1] =
//              expf(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
//        }
 
       // update d: get partial sum
//        float d_partial[MMA_ITERS_M][2];
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          d_partial[m][0] = 0.f;
//          d_partial[m][1] = 0.f;
//  #pragma unroll
//          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
//            x_frag_f[m][frag_idx] =
//                x_frag_f[m][frag_idx] != -inf
//                    ? expf(x_frag_f[m][frag_idx] * sm_scale -
//                           m_local[m][(frag_idx & 0x3) >> 1] * sm_scale)
//                    : 0.f;
//            d_partial[m][(frag_idx & 0x3) >> 1] += x_frag_f[m][frag_idx];
//          }
//        }
//        // update d: get local sum across 4 threads in a row
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x1);
//          d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x2);
//          d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x1);
//          d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x2);
//          d[m][0] *= rescale[m][0];
//          d[m][1] *= rescale[m][1];
//          d[m][0] += d_partial[m][0];
//          d[m][1] += d_partial[m][1];
//        }
 
//        // update o: rescale
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//  #pragma unroll
//          for (int n = 0; n < HEAD_DIM / 16; n++) {
//  #pragma unroll
//            for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
//              o[m][n][frag_idx] *= rescale[m][(frag_idx & 0x3) >> 1];
//            }
//          }
//        }
 
       // printf("start compute o\n");
 
       // update o: compute O = exp(X - m) * V and accumulate
       // use m16n16k16 mma to compute and let warp layout be 1x1x4
//        uint32_t x_frag[MMA_ITERS_M][4], v_frag[4];
//  #pragma unroll
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//          convert_f32_to_bf16_uint32(x_frag_f[m], x_frag[m]);
//          int v_row = (warp_idx << 4) + (lane_idx & 0xF);
//  #pragma unroll
//          for (int n = 0; n < HEAD_DIM / 16; n++) {
//            int v_col = (n << 4) + ((lane_idx >> 4) << 3);
//            T *src_ptr_V =
//                v_row < curr_iter_len ? v_smem(v_row, v_col) : zero_buffer(0, 0);
//            ldsm_t(src_ptr_V, v_frag);
//            mma_m16n16k16_bf16bf16bf32(o[m][n], x_frag[m], v_frag, o[m][n]);
//          }
//        }

//        uint32_t x_frag[MMA_ITERS_M][4], v_frag[4];
//        for (int m = 0; m < MMA_ITERS_M; m++) {
//         for (int n = 0; n < HEAD_DIM / 64; n++) {
//           KV_DESC v_desc(v_smem(0, n * 64));
//           wgmma::warpgroup_arrive();
//           wgmma::mma<T, 64, 64, 16, QOSmem, KVSmem, Q_DESC, KV_DESC, false, false>(o[m][n], x_frag_f[m], v_desc);
//           wgmma::mma_commit_group();
//           wgmma::mma_async_wait();
//         }
//        }

//        //  __syncthreads();
//        wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);
 
//        curr_iter_len = next_iter_len;
       printf("arrive compute_done\n");
       if (warp_idx == 0 && lane_idx == 0) {
         arrive(compute_done[slot], 1);
       }
 

       if (threadIdx.x == 0) {
         printf("finish compute, arrive compute_done\n");
       }
//      }
 
//      // write intermediate results to buffer in shared memory
//  #pragma unroll
//      for (int m = 0; m < MMA_ITERS_M; m++) {
//        m_local[m][0] *= m_local[m][0] != -inf ? sm_scale : 1.f;
//        m_local[m][1] *= m_local[m][1] != -inf ? sm_scale : 1.f;
//        s_m_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2] = m_local[m][0];
//        s_m_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2 + 1] = m_local[m][1];
//        s_d_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2] = d[m][0];
//        s_d_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2 + 1] = d[m][1];
//        for (int n = 0; n < HEAD_DIM / 16; n++) {
//  #pragma unroll
//          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
//            s_o_buffer[m * NUM_THREADS * 64 + threadIdx.x * 64 + n * 8 +
//                       frag_idx] = o[m][n][frag_idx];
//                      }
//                    }
//                  }
//                  wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);
                 
//                  if (threadIdx.x == 0) {
//                    printf("s_m_buffer[%d] = %f\n", threadIdx.x, s_m_buffer[0]);
//                  }
//      // get global m, d, and o
//      // each thread handles an element in o in each iteration
//      for (int elem_idx = threadIdx.x;
//           elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
//           elem_idx += NUM_THREADS) {
//        int row = elem_idx / HEAD_DIM;
//        int col = elem_idx % HEAD_DIM;
//        int t_idx = (row % 8) * 4 + (col % 8) / 2;
//        int mma_iter_n = col / 16;
//        /* The fragment layout is as follows:
//         *
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
//         */
//        int frag_idx = ((col % 16) / 8) * 4 + ((row % 16) / 8) * 2 + (col % 2);
 
//        float m_global = -inf;
//        float d_global = 1.f;
//        float o_global = 0.f;
//        // 4 local values per row
//  #pragma unroll
//        for (int local_idx = 0; local_idx < 4; local_idx++) {
//          // access the shared memory buffer
//          int md_smem_offset = (row / 16) * NUM_THREADS * 2 // mma iter m
//                               + local_idx * 32 * 2  // 32 threads per local value
//                               + t_idx * 2           // corresponding thread
//                               + (frag_idx % 4) / 2; // first half or second half
//          float m_prev = m_global,
//                d_prev = d_global; // save previous values
//          float other_m = s_m_buffer[md_smem_offset],
//                other_d = s_d_buffer[md_smem_offset];
//          m_global = max(m_prev, other_m);
//          d_global = d_prev * expf(m_prev - m_global) +
//                     other_d * expf(other_m - m_global);
//          // accumulate o
//          float other_o =
//              s_o_buffer[(row / 16) * NUM_THREADS * 64 // mma iter m
//                         + local_idx * 32 * 64 // 32 threads per local value
//                         + t_idx * 64          // corresponding thread
//                         + mma_iter_n * 8      // mma iter n
//                         + frag_idx];
//          o_global = o_global * expf(m_prev - m_global) +
//                     other_o * expf(other_m - m_global);
//        }
//        o_smem.at(row, col) = bfloat16(o_global / d_global);
//        // if (threadIdx.x == 0) {
//        //   printf("o_global: %f, d_global: %f\n", o_global, d_global);
//        // }
//      }
//      // __syncthreads();
//      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);
 
//      // store the output
//      for (int elem_idx = threadIdx.x;
//           elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
//           elem_idx += NUM_THREADS) {
//        int src_row = elem_idx / HEAD_DIM;
//        int src_col = elem_idx % HEAD_DIM;
//        int dst_row = src_row / NUM_QO_PER_KV;
//        int dst_col = src_col + (src_row % NUM_QO_PER_KV) * HEAD_DIM;
//        o_dmem.at(dst_row, dst_col) = o_smem.at(src_row, src_col);
     }
 
 
   }
 }
 
 // if (warp_idx == 0 && lane_idx == 0) {
 //   set_barrier_transaction_bytes(q_barrier[0], TMA_TRANS_BYTES_Q);
 //   tma_q.tma_cp_async(q_barrier[0], q_smem(0, 0), {0, 0});
 // }
 
 // wait(q_barrier[0], 0);
 // __syncthreads();
 
 } // namespace kernel