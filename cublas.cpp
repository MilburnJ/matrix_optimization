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
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "     \
                      << cudaGetErrorString(error) << std::endl;              \
            exit(1);                                                          \
        }                                                                     \
    }

// Utility to check cuBLAS errors
#define CUBLAS_CHECK(call)                                                    \
    {                                                                         \
        const cublasStatus_t status = call;                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error: " << __FILE__ << ":" << __LINE__      \
                      << ", " << status << std::endl;                         \
            exit(1);                                                          \
        }                                                                     \
    }

// Function to flatten a nested vector into a flat array
std::vector<double> flatten(const std::vector<std::vector<double>>& nestedVector) {
    std::vector<double> flatVector;
    for (const auto& row : nestedVector) {
        flatVector.insert(flatVector.end(), row.begin(), row.end());
    }
    return flatVector;
}

// GPU matrix multiplication using cuBLAS
std::vector<std::vector<double>> gpuMatrixMultiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    int rowsA, int colsA, int colsB) {

    // Flatten input matrices
    std::vector<double> A_flat = flatten(A);
    std::vector<double> B_flat = flatten(B);
    std::vector<double> C_flat(rowsA * colsB, 0.0);

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, rowsA * colsA * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, colsA * colsB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, rowsA * colsB * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_flat.data(), colsA * colsB * sizeof(double), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        rowsA, colsB, colsA,
        &alpha,
        d_A, rowsA,
        d_B, colsA,
        &beta,
        d_C, rowsA));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_flat.data(), d_C, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Convert flat result back to nested vector
    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i][j] = C_flat[i * colsB + j];
        }
    }

    return C;
}

int main() {
    int n = 1000; // Example matrix size (1000x1000)

    // Initialize matrices A, B, and C with random values
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> B(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 1.0));

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // First, multiply A and B
    std::vector<std::vector<double>> AB = gpuMatrixMultiply(A, B, n, n, n);

    // Then, multiply the result with C
    std::vector<std::vector<double>> ABC = gpuMatrixMultiply(AB, C, n, n, n);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken to multiply three matrices using GPU: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
