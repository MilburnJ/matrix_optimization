#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    // Matrix dimensions
    int n = 1000; // Example size for square matrices

    // Track overall execution time
    auto total_start = std::chrono::high_resolution_clock::now();

    // Allocate and initialize host memory
    // Change Double -> Float
    std::vector<float> A(n * n, 1.0); // Matrix A
    std::vector<float> B(n * n, 1.0); // Matrix B
    std::vector<float> C(n * n, 0.0); // Result matrix

    // Allocate device memory
    double *d_A, *d_B, *d_C, *d_intermediate;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_intermediate, n * n * sizeof(double)));

    // Copy matrices A and B to the device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Perform first matrix multiplication: d_intermediate = d_A * d_B
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_A, n,
        d_B, n,
        &beta,
        d_intermediate, n
    ));

    // Perform second matrix multiplication: d_C = d_intermediate * d_B
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_intermediate, n,
        d_B, n,
        &beta,
        d_C, n
    ));

    // Copy final result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost));

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
