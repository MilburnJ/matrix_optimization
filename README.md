# CISC-662 Project: Optimizing Matrix Multiplication

Authors: Giorgi Gvalia, Jakeb Milburn.

It can be said that studying computer architecture involves unwrapping
multiple layers of abstraction that are typically hidden from anyone
who's not closely working with hardware. In modern computers we are
afforded much faster execution speeds not only due to having more
capable processors, but also via bigger caches, hardware optimizations
such as prefetching and faster accelerators such as GPUs. In order to
gain more hands-on experience with these layers of optimizations, we
decided to take a simple algorithm---matrix multiplication---and run
it through a series of optimizations for meeting our goals:

- Goal 1: A <50 ms runtime for multiplying three 1000x1000 matrices.
- Goal 2: A <1 ms (sub-millisecond) runtime for the above.

By doing this, we hoped to learn more about:

- How to work better with cache in modern processors.
- How to optimize code running on the GPU.
- What are BLAS libraries and how to use them.
- How far we could drive the execution time down.

*To see a summary of results, scroll down until the end.*

We started with a naive multiplication algorithm that used a triple
for loop:

```cpp
for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsB; ++j) {
        for (int k = 0; k < colsA; ++k) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

## Enter the GPUs:

Switching to GPUs was a game-changer. Using NVIDIA’s CuBLAS library on Bridges-2 (a high-performance computing (HPC) platform), equipped with 4 powerful GPUs per node, we unlocked new speed. Our iterative optimizations included:

Using CuBLAS with 4 GPUs:
By simply using the CuBLAS library with 4 GPUs, we achieved an initial runtime of 37.653 ms. This provided a strong starting point for further optimizations.

```cpp
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
```

Reducing Memory Transfers:
Memory transfer between the GPU and the host (CPU) can be a bottleneck. By minimizing these transfers, we dropped the runtime to 29.417 ms.

```cpp
    // Copy matrices A and B to the device (Store intermediate result on the GPU)
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
```

Utilizing Tensor Cores:
Tensor cores are specialized hardware on NVIDIA GPUs designed for matrix operations. While not as impactful for smaller matrices, they showed their true power when working with larger datasets.

```cpp
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

```

Switching Precision:
Moving from FP64 (double precision) to FP32 (single precision) significantly reduced computational overhead. This final tweak achieved our best runtime of 27.318 ms for multiplying three 1000x1000 matrices.

```cpp
    // Allocate and initialize host memory
    // Change Double -> Float
    std::vector<float> A(n * n, 1.0); // Matrix A
    std::vector<float> B(n * n, 1.0); // Matrix B
    std::vector<float> C(n * n, 0.0); // Result matrix
```

## Summary of Results

### CPU

| **Method**                    | **Time**   | **Notes**                                                  |
|-------------------------------|------------|------------------------------------------------------------|
| Naive                         | 17.7763 s  |                                                            |
| OpenMP parallel for           | 2.89815 s  | Parallelized across 6 cores via “#pragma omp parallel for” |
| Transposing the second matrix | 2.6083 s   | Helps achieve better temporal locality                     |
| -O3 (without transposition)   | 268.257 ms |                                                            |
| -O3 (with transposition)      | 137.566 ms |                                                            |
| OpenBLAS                      | 55.0175 ms |                                                            |
| OpenBLAS with -O3             | 43.2692 ms |                                                            |

### GPU

| **Method**                              | **Time**   | **Notes**                                                          |
|-----------------------------------------|------------|--------------------------------------------------------------------|
| CuBLAS With 4 V100 GPUs                 | 37.6527 ms |                                                                    |
| CuBLAS With Transfer Optimization       | 29.4177 ms | Reduce overhead by eliminating some transfers between GPU and Host |
| CuBLAS With Tensor Cores                | 31.3216 ms |                                                                    |
| CuBLAS With F32 & Transfer Optimization | 27.3187 ms |                                                                    |
