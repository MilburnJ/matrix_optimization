#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_fp16.h> // For half-precision support
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Utility to check CUDA errors
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        const cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << __FILE__ << ":" << __LINE__        \
                      << ", " << cudaGetErrorString(error) << std::endl;      \
            exit(1);                                                          \
        }                                                                     \
    }

// Utility to check cuBLAS errors
#define CUBLAS_CHECK(call)                                                    \
    {                                                                         \
        const cublasStatus_t status = call;                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS Error: " << __FILE__ << ":" << __LINE__      \
                      << ", Status Code: " << status << std::endl;            \
            exit(1);                                                          \
        }                                                                     \
    }

int main() {
    int n = 1000; // Matrix dimensions (1000x1000)

    // Track overall execution time
    auto total_start = std::chrono::high_resolution_clock::now();

    // Allocate and initialize host memory in FP16
    std::vector<half> h_A(n * n, __float2half(1.0f)); // Initialize A with 1.0
    std::vector<half> h_B(n * n, __float2half(1.0f)); // Initialize B with 1.0
    std::vector<half> h_C(n * n, __float2half(0.0f)); // Initialize C with 0.0

    // Allocate device memory
    half *d_A, *d_B, *d_C, *d_intermediate;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_intermediate, n * n * sizeof(half)));

    // Copy matrices A and B to the device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n * n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), n * n * sizeof(half), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Enable Tensor Core acceleration
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Perform first matrix multiplication: d_intermediate = d_A * d_B
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for A and B
        n, n, n,                 // Matrix dimensions
        &alpha,
        d_A, CUDA_R_16F, n,
        d_B, CUDA_R_16F, n,
        &beta,
        d_intermediate, CUDA_R_16F, n,
        CUDA_R_32F, // Compute in FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Perform second matrix multiplication: d_C = d_intermediate * d_B
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_intermediate, CUDA_R_16F, n,
        d_B, CUDA_R_16F, n,
        &beta,
        d_C, CUDA_R_16F, n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, n * n * sizeof(half), cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_intermediate);
    cublasDestroy(handle);

    // Track total execution time
    auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "Total execution time: "
              << std::chrono::duration<double>(total_end - total_start).count()
              << " seconds" << std::endl;

    return 0;
}
