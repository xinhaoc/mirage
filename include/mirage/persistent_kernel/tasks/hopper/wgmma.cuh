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


//reference
//https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
namespace kernel{

    // from  kitten include/ops/group/wgmma/base/base.cuh

    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

    template<kittens::ducks::st::all _ST, int transpose>
    struct wgmma_descriptor {
        using identifier = ducks::wgmma::descriptor::identifier;
        using ST = _ST;
        static constexpr int height = ST::height;
        static constexpr int width  = ST::width;
        using T = ST::T;
        uint64_t base_desc;
        __device__ inline descriptor(const ST &tile) {
            base_desc = matrix_descriptor_encode((uint64_t)(&tile.data[0]));
            if constexpr (transpose) { // transpose mode
                if constexpr (ST::width%4 == 0) {
                    base_desc |= matrix_descriptor_encode((uint64_t)2048*ST::height) << 16;
                    base_desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
                    base_desc |= 1llu << 62; // set wgmma_swizzle mode
                }
                else if constexpr (ST::width%2 == 0) {
                    base_desc |= matrix_descriptor_encode((uint64_t)1024*ST::height) << 16;
                    base_desc |= matrix_descriptor_encode((uint64_t)512) << 32;
                    base_desc |= 2llu << 62; // set wgmma_swizzle mode
                }
                else {
                    base_desc |= matrix_descriptor_encode((uint64_t)512*ST::height) << 16;
                    base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
                    base_desc |= 3llu << 62; // set wgmma_swizzle mode
                }
            }
            else { // normal mode
                if constexpr (ST::width%4 == 0) {
                    base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;   // this line doesn't matter
                    base_desc |= matrix_descriptor_encode((uint64_t)1024) << 32; // 128 byte swizzle x 8 for core matrix rows
                    base_desc |= 1llu << 62; // set wgmma_swizzle mode
                }
                else if constexpr (ST::width%2 == 0) {
                    base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;  // this line doesn't matter
                    base_desc |= matrix_descriptor_encode((uint64_t)512) << 32; // 64 byte swizzle x 8 for core matrix rows
                    base_desc |= 2llu << 62; // set wgmma_swizzle mode
                }
                else {
                    base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;  // this line doesn't matter
                    base_desc |= matrix_descriptor_encode((uint64_t)256) << 32; // 32 byte swizzle x 8 for core matrix rows
                    base_desc |= 3llu << 62; // set wgmma_swizzle mode
                }
            }
        }
        __device__ inline descriptor(const descriptor<ST, transpose> &other) : base_desc(other.base_desc) {} // copy constructor
        __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
            if constexpr (transpose) { // transpose mode
                if constexpr (ST::width%4 == 0) {
                    return base_desc + matrix_descriptor_encode(chunk_idx*2048);
                }
                else if constexpr (ST::width%2 == 0) {
                    return base_desc + matrix_descriptor_encode(chunk_idx*1024);
                }
                else {
                    return base_desc + matrix_descriptor_encode(chunk_idx*512);
                }
            }
            else { // normal mode
                if constexpr (ST::width%4 == 0) {
                    return base_desc + matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*ST::height*2048);
                }
                else if constexpr (ST::width%2 == 0) {
                    return base_desc + matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*ST::height*1024);
                }
                else {
                    return base_desc + matrix_descriptor_encode(chunk_idx*ST::height*512);
                }
            }
        }
    };



__device__ static inline void warpgroup_commit_batch()
{
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  cutlass::arch::synclog_emit_warpgroup_commit_batch(__LINE__);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Attempting to use wgmma.commit_group without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
}

__device__ static inline void wgmma_m64n64k16_bf16bf16bf32(uint64_t const& desc_a,
      uint64_t const& desc_b,
      float         & d00, float         & d01, float         & d02, float         & d03,
      float         & d04, float         & d05, float         & d06, float         & d07,
      float         & d08, float         & d09, float         & d10, float         & d11,
      float         & d12, float         & d13, float         & d14, float         & d15,
      float         & d16, float         & d17, float         & d18, float         & d19,
      float         & d20, float         & d21, float         & d22, float         & d23,
      float         & d24, float         & d25, float         & d26, float         & d27,
      float         & d28, float         & d29, float         & d30, float         & d31,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_smem_smem(__LINE__, desc_a, desc_b);
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32,"
      " %33,"
      " p,   %35, %36, %37, %38;\n"
    "}\n"
      : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
        "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
        "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
        "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
        "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
        "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
        "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27),
        "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_64x64x16_F32BF16BF16_SS without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
}
