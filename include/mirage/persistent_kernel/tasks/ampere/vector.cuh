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
template <size_t vec_size>
struct vector_t<cutlass::bfloat16_t, vec_size> {
  static_assert(vec_size % 8 == 0, "invalid vector size");
  uint4 data[vec_size / 8];  // uint4 = 16 bytes = 8 bfloat16s

  __device__ __forceinline__ cutlass::bfloat16_t& operator[](size_t i) { 
    return ((cutlass::bfloat16_t*)(data))[i]; 
  }
  
  __device__ __forceinline__ const cutlass::bfloat16_t& operator[](size_t i) const { 
    return ((const cutlass::bfloat16_t*)(data))[i]; 
  }
  
  __device__ __forceinline__ cutlass::bfloat16_t* ptr() { 
    return reinterpret_cast<cutlass::bfloat16_t*>(&data); 
  }
  
  __device__ __forceinline__ void fill(cutlass::bfloat16_t val) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      (*this)[i] = val;
    }
  }
  
  __device__ __forceinline__ void load(const cutlass::bfloat16_t* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
};

template <size_t vec_size>
struct vector_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];

  __device__ __forceinline__ float& operator[](size_t i) { return ((float*)(data))[i]; }
  __device__ __forceinline__ const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  __device__ __forceinline__ float* ptr() { return reinterpret_cast<float*>(&data); }
  __device__ __forceinline__ void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  __device__ __forceinline__ void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  __device__ __forceinline__ void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
};