#include "include/mirage/persistent_kernel/tasks/linear.cuh"
#include "include/mirage/persistent_kernel/tasks/linear_split.cuh"

static constexpr int SINGLE_KERNEL_THREADS = 128;
static constexpr int MAX_SHARE_MEMORY_SIZE = 160 * 1024;
static constexpr size_t NUM_LAYERS = 45;
static constexpr size_t SM_COUNT = 96;
static constexpr size_t OUTPUT_SIZE = 64;
static constexpr size_t REDUCTION_SIZE = 4096;
static constexpr size_t BATCH_SIZE = 16;
static constexpr bool USE_PIPELINE = true;
using bfloat16 = type::bfloat16_t;

__global__ void main_kernel(void *d_input, void *d_weight, void *d_output) {
    extern __shared__ char smem[];
  
    if constexpr (USE_PIPELINE) {
      kernel::linear_prefetch<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, OUTPUT_SIZE * SM_COUNT>(d_input, d_weight, smem);
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
        input_ptr_next,
        weight_ptr_next,
        smem_next);
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
        false);
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
    
  // Launcher persistent kernel
  cudaFuncSetAttribute(main_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        MAX_SHARE_MEMORY_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  main_kernel<<<dim3(sm_count, 1, 1),
                      dim3(SINGLE_KERNEL_THREADS, 1, 1),
                      MAX_SHARE_MEMORY_SIZE /*smem*/>>>(d_input, d_weight, d_output);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Stop the timer and print the time
  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  printf("Time taken: %f ms\n", elapsed_ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Output the output tensors to a file for verification
  cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < output_size / sizeof(bfloat16); ++i) {
    if (h_output[i] != static_cast<bfloat16>(REDUCTION_SIZE)) {
      printf("Error: h_output[%zu] = %f\n", i, float(h_output[i]));
    }
  }


}