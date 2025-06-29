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

struct Barrier {
    private:
    uint64_t value;
};

 /*
 mbarrier related functions
  */
 __device__ static inline void initialize_barrier(Barrier& smem_barrier,                 // 64 bits user-manged barrier in smem
                   int thread_count = 1)                   // Thread count expected to arrive/wait on this barrier
{
#if defined(MIRAGE_GRACE_HOPPER)
uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_barrier));
//   uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(thread_count));
#elif defined(__CUDA_ARCH__)
       asm volatile ("brkpt;\n" ::);
#endif
}

__device__ static inline void set_barrier_transaction_bytes(Barrier& smem_barrier,      // 64 bits user-manged barrier in smem
                              uint32_t bytes)              // Number of bytes transfered by per TMA transaction
{
#if defined(MIRAGE_GRACE_HOPPER)
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_barrier));
//   cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(bytes));
#elif defined(__CUDA_ARCH__)
       asm volatile ("brkpt;\n" ::);
#endif
}


__device__ static inline bool try_wait(Barrier& const smem_barrier, uint32_t phase) {
#if defined(MIRAGE_GRACE_HOPPER)
    // uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_barrier));
    cutlass::arch::synclog_emit_cluster_barrier_try_wait(__LINE__, smem_addr, phase);
    uint32_t waitComplete;
    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase));

    return static_cast<bool>(waitComplete);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
    return 0;
  }
