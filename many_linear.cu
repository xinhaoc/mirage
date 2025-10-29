// #include "include/mirage/persistent_kernel/tasks/linear.cuh"
#define MEASURE 0
#include "include/mirage/persistent_kernel/tasks/linear_cutlass.cuh"
#include "include/mirage/persistent_kernel/tasks/linear_cutlass_split.cuh"
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cuda.h>

static constexpr int SINGLE_KERNEL_THREADS = 128;
static constexpr int MAX_SHARE_MEMORY_SIZE = 160 * 1024;
static constexpr size_t NUM_LAYERS = 30;
static constexpr size_t SM_COUNT = 96;
static constexpr size_t OUTPUT_SIZE = 64;
static constexpr size_t REDUCTION_SIZE = 1024;
static constexpr size_t BATCH_SIZE = 16;
static constexpr bool USE_PIPELINE = true;
static constexpr size_t NUM_TRIALS = 1;
static constexpr size_t NUM_WARMUP_TRIALS = 5;
using bfloat16 = type::bfloat16_t;        // kernel::linear_prefetch<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT>(input_ptr_next, weight_ptr_next, smem_next);

__global__ void main_kernel(void *d_input, void *d_weight, void *d_output, size_t *clock_cycles_mem, size_t *clock_cycles_compute) {
    extern __shared__ char smem[];
  
    if constexpr (USE_PIPELINE) {
      size_t time_start_prefetch = clock64();
      kernel::linear_prefetch<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT>(d_input, d_weight, smem);
      size_t time_end_prefetch = clock64();

      for (size_t layer_num = 0; layer_num < NUM_LAYERS; layer_num++) {
        char * shared_mem_start = smem + (MAX_SHARE_MEMORY_SIZE / 2) * (layer_num % 2);

        void * input_ptr = (bfloat16 *)d_input + (layer_num * BATCH_SIZE * REDUCTION_SIZE);
        void * weight_ptr = (bfloat16 *)d_weight + (layer_num * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);
        void * output_ptr = (bfloat16 *)d_output + (layer_num * BATCH_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);

        char * smem_next = smem + (MAX_SHARE_MEMORY_SIZE / 2) * ((layer_num + 1) % 2);
        void * input_ptr_next = (bfloat16 *)d_input + ((layer_num + 1) * BATCH_SIZE * REDUCTION_SIZE);
        void * weight_ptr_next = (bfloat16 *)d_weight + ((layer_num + 1) * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);

        kernel::linear_main<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT>(
        input_ptr,
        weight_ptr,
        nullptr,
        output_ptr,
        BATCH_SIZE,
        false,
        shared_mem_start,
        layer_num < NUM_LAYERS - 1,
        smem_next,
        input_ptr_next,
        weight_ptr_next
        );

      }
    }

    else {
      for (size_t layer_num = 0; layer_num < NUM_LAYERS; layer_num++) {
        void * input_ptr = (bfloat16 *)d_input + (layer_num * BATCH_SIZE * REDUCTION_SIZE);
        void * weight_ptr = (bfloat16 *)d_weight + (layer_num * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);
        void * output_ptr = (bfloat16 *)d_output + (layer_num * BATCH_SIZE * OUTPUT_SIZE * SM_COUNT) + (blockIdx.x * OUTPUT_SIZE);

        kernel::linear_kernel<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT>(
        input_ptr,
        weight_ptr,
        nullptr,
        output_ptr,
        BATCH_SIZE,
        false
        );
      }
    }

}


int main() {

  // Create synthetic inputs and weight tensors, cudaMemcpy to device memory


  // Launch the main kernel and start the timer

  int device;
  cudaGetDevice(&device);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  printf("a single persistent kernel\n");

  // CUdevice cu_device;
  // CUcontext cu_context;
  // CUmodule mod;
  // CUfunction main_kernel;
  // cuInit(0);
  // cuDeviceGet(&cu_device, 0);
  // cuCtxCreate(&cu_context, 0, cu_device);


  // auto result = cuModuleLoad(&mod, "many_linear.ptx");
  // if (result != CUDA_SUCCESS) {
  //   printf("Error loading module: %d\n", result);
  //   return 1;
  // }
  // auto result2 = cuModuleGetFunction(&main_kernel, mod, "_Z11main_kernelPvS_S_PmS0_");
  // if (result2 != CUDA_SUCCESS) {
  //   printf("Error getting function: %d\n", result2);
  //   return 1;
  // }


  // Allocate device memory for d_input, d_weight, d_output and fill with ones

  size_t input_size = NUM_LAYERS * BATCH_SIZE * REDUCTION_SIZE * sizeof(bfloat16);
  size_t weight_size = NUM_LAYERS * REDUCTION_SIZE * OUTPUT_SIZE * SM_COUNT * sizeof(bfloat16);
  size_t output_size = NUM_LAYERS * BATCH_SIZE * OUTPUT_SIZE * SM_COUNT * sizeof(bfloat16);

  bfloat16 *d_input = nullptr;
  bfloat16 *d_weight = nullptr;
  bfloat16 *d_output = nullptr;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_weight, weight_size);
  cudaMalloc(&d_output, output_size);

  // Fill with ones
  // Allocate host buffers
  bfloat16 *h_input = (bfloat16*)malloc(input_size);
  bfloat16 *h_weight = (bfloat16*)malloc(weight_size);
  bfloat16 *h_output = (bfloat16*)malloc(output_size);

  for (size_t i = 0; i < input_size / sizeof(bfloat16); ++i) {
      h_input[i] = bfloat16(1.0f);
  }
  for (size_t i = 0; i < weight_size / sizeof(bfloat16); ++i) {
      h_weight[i] = bfloat16(1.0f);
  }
  for (size_t i = 0; i < output_size / sizeof(bfloat16); ++i) {
      h_output[i] = bfloat16(1.0f);
  }

  cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, output_size, cudaMemcpyHostToDevice);

  free(h_input);
  free(h_weight);

  // Allocate device memory for clock_cycles_mem and clock_cycles_compute
  size_t *d_clock_cycles_mem = nullptr;
  size_t *d_clock_cycles_compute = nullptr;
  cudaMalloc(&d_clock_cycles_mem, NUM_LAYERS * (REDUCTION_SIZE / 128) * sizeof(size_t));
  cudaMalloc(&d_clock_cycles_compute, NUM_LAYERS * (REDUCTION_SIZE / 128) * sizeof(size_t));
    
  // Launcher persistent kernel
  // cudaFuncSetAttribute(main_kernel,
  //                       cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                       MAX_SHARE_MEMORY_SIZE);
  
  bfloat16 **d_input_ptr;
  bfloat16 **d_weight_ptr;
  bfloat16 **d_output_ptr;
  size_t **d_clock_cycles_mem_ptr;
  size_t **d_clock_cycles_compute_ptr;
  cudaMalloc(&d_input_ptr, sizeof(bfloat16*));
  cudaMalloc(&d_weight_ptr, sizeof(bfloat16*));
  cudaMalloc(&d_output_ptr, sizeof(bfloat16*));
  cudaMalloc(&d_clock_cycles_mem_ptr, sizeof(size_t*));
  cudaMalloc(&d_clock_cycles_compute_ptr, sizeof(size_t*));
  cudaMemcpy(d_input_ptr, &d_input, sizeof(bfloat16*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight_ptr, &d_weight, sizeof(bfloat16*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_ptr, &d_output, sizeof(bfloat16*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_clock_cycles_mem_ptr, &d_clock_cycles_mem, sizeof(size_t*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_clock_cycles_compute_ptr, &d_clock_cycles_compute, sizeof(size_t*), cudaMemcpyHostToDevice);


  void* args[] = {&d_input_ptr, &d_weight_ptr, &d_output_ptr, &d_clock_cycles_mem_ptr, &d_clock_cycles_compute_ptr};
  // void* args[] = {&d_input, &d_weight, &d_output, &d_clock_cycles_mem, &d_clock_cycles_compute, nullptr};
  for (size_t i = 0; i < NUM_WARMUP_TRIALS; ++i) {
    // auto result = cuLaunchKernel(main_kernel, sm_count, 1, 1, SINGLE_KERNEL_THREADS, 1, 1, 48 * 1024, 0, args, nullptr);
    // if (result != CUDA_SUCCESS) {
    //   printf("Error launching kernel: %d\n", result);
    //   return 1;
    // }
    main_kernel<<<dim3(sm_count, 1, 1),
                      dim3(SINGLE_KERNEL_THREADS, 1, 1),
                      MAX_SHARE_MEMORY_SIZE /*smem*/>>>(d_input, d_weight, d_output, d_clock_cycles_mem, d_clock_cycles_compute);
  }
  std::array<float, NUM_TRIALS> all_elapsed_ms;
  for (size_t i = 0; i < NUM_TRIALS; ++i) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    auto start_time = std::chrono::high_resolution_clock::now();
    main_kernel<<<dim3(sm_count, 1, 1),
                      dim3(SINGLE_KERNEL_THREADS, 1, 1),
                      MAX_SHARE_MEMORY_SIZE /*smem*/>>>(d_input, d_weight, d_output, d_clock_cycles_mem, d_clock_cycles_compute);
    // auto result = cuLaunchKernel(main_kernel, sm_count, 1, 1, SINGLE_KERNEL_THREADS, 1, 1, 48 * 1024, 0, args, nullptr);
    // if (result != CUDA_SUCCESS) {
    //   printf("Error launching kernel: %d\n", result);
    //   return 1;
    // }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    all_elapsed_ms[i] = duration.count();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Process the elapsed times
  float min_elapsed_ms = *std::min_element(all_elapsed_ms.begin(), all_elapsed_ms.end());
  float max_elapsed_ms = *std::max_element(all_elapsed_ms.begin(), all_elapsed_ms.end());
  float average_elapsed_ms = std::accumulate(all_elapsed_ms.begin(), all_elapsed_ms.end(), 0.0f) / NUM_TRIALS;
  float std_elapsed_ms = std::sqrt(std::accumulate(all_elapsed_ms.begin(), all_elapsed_ms.end(), 0.0f, [average_elapsed_ms](float acc, float x) { return acc + (x - average_elapsed_ms) * (x - average_elapsed_ms); }) / NUM_TRIALS);
  printf("Min elapsed time: %f ms\n", min_elapsed_ms);
  printf("Max elapsed time: %f ms\n", max_elapsed_ms);
  printf("Average elapsed time: %f ms\n", average_elapsed_ms);
  printf("Standard deviation: %f ms (%.2f%%)\n", std_elapsed_ms, std_elapsed_ms / average_elapsed_ms * 100);
  // print all the elapsed times
  for (size_t i = 0; i < NUM_TRIALS; ++i) {
    printf("Elapsed time %zu: %f ms, ", i, all_elapsed_ms[i]);
  }

  // Output the output tensors to a file for verification
  cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < output_size / sizeof(bfloat16); ++i) {
    if (h_output[i] != static_cast<bfloat16>(REDUCTION_SIZE)) {
      printf("Error: h_output[%zu] = %f\n", i, float(h_output[i]));
      return 1;
    }
  }

  // Write the clock_cycles_mem and clock_cycles_compute to a file
  size_t *h_clock_cycles_mem = (size_t*)malloc(NUM_LAYERS * (REDUCTION_SIZE / 128) * sizeof(size_t));
  size_t *h_clock_cycles_compute = (size_t*)malloc(NUM_LAYERS * (REDUCTION_SIZE / 128) * sizeof(size_t));
  cudaMemcpy(h_clock_cycles_mem, d_clock_cycles_mem, NUM_LAYERS * (REDUCTION_SIZE / 128) * sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_clock_cycles_compute, d_clock_cycles_compute, NUM_LAYERS * (REDUCTION_SIZE / 128) * sizeof(size_t), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < NUM_LAYERS * (REDUCTION_SIZE / 128); ++i) {
    printf("clock_cycles_mem[%zu] = %zu\n", i, h_clock_cycles_mem[i]);
    printf("clock_cycles_compute[%zu] = %zu\n", i, h_clock_cycles_compute[i]);
  }
  free(h_clock_cycles_mem);
  free(h_clock_cycles_compute);


  // cuCtxDestroy(cu_context);
}