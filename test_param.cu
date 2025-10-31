#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

static constexpr int SINGLE_KERNEL_THREADS = 128;
static constexpr size_t SM_COUNT = 96;
static constexpr size_t ARRAY_SIZE = 1024 * 1024; // 1M elements

// Device helper function that adds two numbers
__device__ __noinline__ float add_two_numbers(float a, float b) {
    return a + b;
}

// Kernel that takes 2 input pointers and 1 output pointer
// Delegates all work to the device helper function
// __global__ void main_kernel(float *input1, float *input2, float *output) {
//     // Calculate global thread index
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t stride = gridDim.x * blockDim.x;
    
//     // Grid-stride loop to handle all elements
//     for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
//         // Delegate the addition to the device helper function
//         output[i] = add_two_numbers(input1[i], input2[i]);
//     }
// }

#define CUDA_CHECK(err) do { if (err != CUDA_SUCCESS) { printf("CUDA error: %d\n", err); return 1; } } while (0)

int main() {
    CUdevice cu_device;
    CUcontext cu_context;
    CUmodule mod;
    CUfunction main_kernel;
    cuInit(0);
    cuDeviceGet(&cu_device, 0);
    cuCtxCreate(&cu_context, 0, cu_device);


    auto result = cuModuleLoad(&mod, "test_param.ptx");
    if (result != CUDA_SUCCESS) {
        printf("Error loading module: %d\n", result);
        return 1;
    }
    auto result2 = cuModuleGetFunction(&main_kernel, mod, "_Z11main_kernelPfS_S_");
    if (result2 != CUDA_SUCCESS) {
        printf("Error getting function: %d\n", result2);
        return 1;
    }



    // int device;
    // cudaGetDevice(&device);
    // int sm_count;
    // cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    // printf("Running kernel with %d SMs\n", sm_count);
    // printf("Array size: %zu elements\n", ARRAY_SIZE);

    // Allocate device memory for inputs and output
    CUdeviceptr d_input1 = NULL;
    CUdeviceptr d_input2 = NULL;
    CUdeviceptr d_output = NULL;

    size_t array_bytes = ARRAY_SIZE * sizeof(float);

    CUDA_CHECK(cuMemAlloc(&d_input1, array_bytes));
    CUDA_CHECK(cuMemAlloc(&d_input2, array_bytes));
    CUDA_CHECK(cuMemAlloc(&d_output, array_bytes));

    // Allocate and initialize host buffers
    float *h_input1 = (float*)malloc(array_bytes);
    float *h_input2 = (float*)malloc(array_bytes);
    float *h_output = (float*)malloc(array_bytes);

    // Fill input arrays with test values
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        h_input1[i] = 1.0f;
        h_input2[i] = 2.0f;
    }

    // Copy inputs to device
    // cudaMemcpy(d_input1, h_input1, array_bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_input2, h_input2, array_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(cuMemcpyHtoD(d_input1, h_input1, array_bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_input2, h_input2, array_bytes));

    CUstream stream;
    auto stream_result = cuStreamCreate(&stream, 0);
    if (stream_result != CUDA_SUCCESS) {
        printf("Error creating stream: %d\n", stream_result);
        return 1;
    }


    // Launch kernel with same grid and block dimensions as original
    // Grid: (sm_count, 1, 1) = (96, 1, 1)
    // Block: (SINGLE_KERNEL_THREADS, 1, 1) = (128, 1, 1)
    // main_kernel<<<dim3(sm_count, 1, 1), 
    //                dim3(SINGLE_KERNEL_THREADS, 1, 1)>>>(
    //     d_input1, d_input2, d_output
    // );
    void *params[3] = {&d_input1, &d_input2, &d_output};
    CUDA_CHECK(cuFuncSetAttribute(main_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 160 * 1024));
    auto launch_result = cuLaunchKernel(main_kernel, 1, 1, 1, 1, 1, 1, 160 * 1024, NULL, (void**) params, NULL);
    if (launch_result != CUDA_SUCCESS) {
        printf("Error launching kernel: %d\n", launch_result);
        return 1;
    }


    // Check for launch errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    // cudaMemcpy(h_output, d_output, array_bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cuMemcpyDtoH(h_output, d_output, array_bytes));

    // Verify results (1.0 + 2.0 = 3.0)
    bool success = true;
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        if (h_output[i] != 3.0f) {
            printf("Error: h_output[%zu] = %f (expected 3.0)\n", i, h_output[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Success! All results are correct.\n");
    }

    // Cleanup
    free(h_input1);
    free(h_input2);
    free(h_output);
    CUDA_CHECK(cuMemFree(d_input1));
    CUDA_CHECK(cuMemFree(d_input2));
    CUDA_CHECK(cuMemFree(d_output));
    // cudaFree(d_input2);
    // cudaFree(d_output);

    return success ? 0 : 1;
}
