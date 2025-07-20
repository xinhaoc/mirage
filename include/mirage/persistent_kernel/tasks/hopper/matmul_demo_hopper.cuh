
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
 #include "tma.cuh"
 #include "../dmem_layout.cuh"
 #include "../element_binary.cuh"
 #include "../element_unary.cuh"
 #include "../reduction.cuh"
 #include "../smem_layout.cuh"
 #include "utils.cuh"
#include "../utils.cuh"
  #include "wgmma.cuh"
 namespace kernel {
 
 using namespace tma;
 using bfloat16 = type::bfloat16_t;
 
 // a 64X64X4K kernel for reference
 template <typename T, int BATCH_SIZE, int HIDDEN_SIZE, int Kstages, typename TMA_A, typename TMA_B>
 __device__ __forceinline__ void linear_kernel_hopper(void *output_ptr,
                                                  const TMA_A &tma_a,
                                                const TMA_B &tma_b) {
   constexpr int chunk_size = 16 / sizeof(T);
   constexpr int NUM_THREADS = 128;
   constexpr int CONSUMER_WARPGROUPS = 1; 
   constexpr int PRODUCER_WARPGROUPS = 1; 
   constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS; 
    
   constexpr int num_chunks = HIDDEN_SIZE / chunk_size;
    
   // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
   constexpr int num_n = HIDDEN_SIZE / 64;
   constexpr int num_m = BATCH_SIZE / 64;
   constexpr int num_k = HIDDEN_SIZE / 16;
   int warp_idx = warp_id();
   int idx_in_warp = threadIdx.x % 32;
   int warpgroup_id = warp_idx / WARPGROUP_WARPS;
 
   T __restrict__ *d_output = static_cast<T *>(output_ptr) + blockIdx.x * 64 * 1;
 
   dmem_row<T, 64, 64, 64> output_dmem(d_output);
 
   extern __shared__ T smem[];
 
   // copy input
   T *shared_input = (T *)(smem + 2176);
   T *shared_input_buffer = (T *)(smem + 4224);
   // copy weight
   T *shared_weight = (T *)(smem + 6272);
   T *shared_weight_buffer = (T *)(smem + 14464);
   // intermidiate
   T *mm_output = (T *)(smem + 2176);
   T *element_unary_output = (T *)(smem + 128);
   T *reduction_output = (T *)(smem + 4224);
   // out
   T *shared_output = (T *)(smem + 128);
 
   // define the swizzle mode
   using InputSmem = smem_row<T, 3, 4, 3, 64, 64, 64>;
   InputSmem input_smem(shared_input);
   InputSmem input_smem_buffer(shared_input);

  //  smem_row<T, 3, 4, 3, 64, 64, 64> input_smem_buffer(shared_input_buffer);
 
   using WeightSmem =smem_row<T, 3, 4, 3, 64, 64, 64>;
   WeightSmem input_weight_smem(shared_weight);
   WeightSmem input_weight_smem_buffer(shared_weight);

   using A_DESC = wgmma::mma_descriptor<InputSmem>;
   using B_DESC = wgmma::mma_descriptor<WeightSmem>;

  //  smem_row<T, 3, 4, 3, 64, 64, 64> input_weight_smem_buffer(
  //      shared_weight_buffer);

   smem_row<T, 3, 4, 3, 64, 64, 64> mm_output_smem(mm_output);

   float s_frag[32];

   //define barries
   __shared__ Barrier input_barrier[Kstages], weight_barrier[Kstages], compute_done[Kstages];

  //init the barriers and launch the first group of copy
   if(threadIdx.x == 0){
    for(int i = 0; i < Kstages; i++){
      initialize_barrier(input_barrier[i], 1);
      initialize_barrier(weight_barrier[i], 1);
    }

     //launch the first group of copy
     #pragma unroll
     for(int i = 0; i < Kstages - 1; i++){
      
      int2 tma_coords = {0, i};
      set_barrier_transaction_bytes(input_barrier[i], 8192);

      tma_a.tma_cp_async(input_barrier[i], input_smem_buffer(0, 0), tma_coords);
      set_barrier_transaction_bytes(weight_barrier[i], 8192);
      tma_b.tma_cp_async(weight_barrier[i], input_weight_smem_buffer(0, 0), tma_coords);
      //mv smem ptr to next buffer
      input_smem_buffer.set_ptr(shared_input + i * 8192);
      input_weight_smem_buffer.set_ptr(shared_weight + i * 8192);
    }
   }

   __syncthreads();

   //warp specialization data movement warpgroup
   if(warpgroup_id == NUM_WARPGROUPS - 1){
      wg_decrease_regs<32>();  
      if(lane_id() == 0 && warp_id == 0){
        for(int i = 0; i < (64 - Kstages + 1); i++){
          //get cord, copy
          int2 tma_coords = {0, i + Kstages - 1};
          set_barrier_transaction_bytes(input_barrier[i], 8192);
          tma_a.tma_cp_async(input_barrier[i], input_smem_buffer(0, 0), tma_coords);
          set_barrier_transaction_bytes(input_barrier[i], 8192);
          tma_b.tma_cp_async(weight_barrier[i], input_weight_smem_buffer(0, 0), tma_coords);
          wait(compute_done[(i)%Kstages], (i/Kstages)%2);
          input_smem_buffer.set_ptr(shared_input + ((i + Kstages) % Kstages) * 8192);
          input_weight_smem_buffer.set_ptr(shared_weight + ((i + Kstages) % Kstages) * 8192);
      }
    }
   }else{
     //warp specialization compute warpgroup
     wg_increase_regs<160>();  
     for(int i = 0; i < 64; i++){
      //wait input, weight
      wait(input_barrier[i % Kstages], ((i / Kstages) % 2));
      wait(weight_barrier[i % Kstages], ((i / Kstages) % 2));

      A_DESC a_desc(input_smem(0, 0));
      B_DESC b_desc(input_weight_smem(0, 0));
      
      wgmma::warpgroup_arrive();
      //wgmma
      wgmma::mma<bfloat16, 64, 64, 16, InputSmem, WeightSmem, A_DESC, B_DESC, false, true>(s_frag,a_desc, b_desc);
      wgmma::mma_commit_group();
      wgmma::mma_async_wait();

      //flip compute done
      if(lane_id() == 0) {
        arrive(compute_done[i%Kstages], 1);
      }
     }

   }

  
  
 }
 
 } // namespace kernel