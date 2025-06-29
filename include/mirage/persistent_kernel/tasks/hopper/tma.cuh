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
#include <cuda.h>

 ///home/xinhaoc/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp

__host__ static inline void create_tma_desc(){
    using dtype = typename ST::dtype;
    static_assert(axis==0 || axis==1 || axis==2, "axis must be 0, 1, or 2");
    
    constexpr uint32_t  tma_dim = 5; // Always use all 5D
    void *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        std::is_same_v<dtype, fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
        CU_TENSOR_MAP_SWIZZLE_NONE
    );

    uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
    uint64_t gmem_stride[4] = {0, 0, 0, 0};
    uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols;

    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);

    if constexpr (axis == 2) {
        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)rows;
        gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
        gmem_shape[3] = (uint64_t)depth;
        gmem_shape[4] = (uint64_t)batch;

        gmem_stride[0] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[1] = ST::swizzle_bytes;
        gmem_stride[2] = (uint64_t)rows * cols * sizeof(dtype);
        gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);
    }
    else if constexpr (axis == 1) {
        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)depth;
        gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
        gmem_shape[3] = (uint64_t)rows;
        gmem_shape[4] = (uint64_t)batch;

        gmem_stride[0] = (uint64_t)rows * cols * sizeof(dtype);
        gmem_stride[1] = ST::swizzle_bytes;
        gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);

    }
    else {
        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)batch;
        gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
        gmem_shape[3] = (uint64_t)rows;
        gmem_shape[4] = (uint64_t)depth;

        gmem_stride[0] = (uint64_t)depth * rows * cols * sizeof(dtype);
        gmem_stride[1] = ST::swizzle_bytes;
        gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[3] = (uint64_t)rows * cols * sizeof(dtype);
    }
    smem_shape[0] = swizzle_elements;
    smem_shape[1] = shared_tile_height;
    smem_shape[2] = shared_tile_width / swizzle_elements;
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(gmem_stride[0] % 16 == 0); // gmem_stride[0] elements must be a multiple of 16B
    assert(gmem_stride[1] % 16 == 0); // gmem_stride[1] elements must be a multiple of 16B
    assert(gmem_stride[2] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B
    assert(gmem_stride[3] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256
    assert(smem_shape[1] <= 256); // smem_shape[1] elements must be <= 256
    assert(smem_shape[2] <= 256); // smem_shape[2] elements must be <= 256

    assert((smem_shape[0]*sizeof(dtype)) % 16 == 0); // if wgmma_interleave is none, then smem_shape[0] * sizeof(dtype) must be a multiple of 16B

    assert(smem_stride[0] <= 8); // smem_stride[0] must be less <= 8
    assert(smem_stride[1] <= 8); // smem_stride[1] must be less <= 8
    assert(smem_stride[2] <= 8); // smem_stride[2] must be less <= 8
    assert(smem_stride[3] <= 8); // smem_stride[3] must be less <= 8
    assert(smem_stride[4] <= 8); // smem_stride[3] must be less <= 8

    assert(smem_stride[0] == 1); // smem_stride[0] is ignored when wgmma_interleave is none

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        assert(smem_shape[0] * sizeof(dtype) <= ST::swizzle_bytes);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        tma_swizzle,
        tma_l2Promotion,
        tma_oobFill);

    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Error in tile TMA descriptor creation: " << error_string << std::endl;
    }
}