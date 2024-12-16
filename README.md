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

For this project we used Fangjun Zhou’s
[video](https://www.youtube.com/watch?v=QGYvbsHDPxo)
([source](https://github.com/fangjunzhou/blas-playground)) as
guidance, though we optimized a slightly different algorithm. We
started with a naive multiplication algorithm for two matrices that
used a triple for loop:

```cpp
for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsB; ++j) {
        for (int k = 0; k < colsA; ++k) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

This algorithm was the slowest and least efficient. One of the
quickest ways to get better performance was to parallelize it using
OpenMP, so that's what we did:

```cpp
#pragma omp parallel for
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```

This dramatically reduced the execution time. It was very surprising
to us that most of the inefficiency in the naive algorithm was
removable with such ease. However, the result was still far from our
goals, so next we tried an interesting optimization: transposing the
second matrix.

```cpp
// Transpose matrix B
std::vector<std::vector<int>> BT(colsB, std::vector<int>(rowsA));
for (int i = 0; i < colsB; ++i) {
    for (int j = 0; j < rowsA; ++j) {
        BT[i][j] = B[j][i];
    }
}

// The rest of the code is the same as the example above.
```

As we've learned, transposing the second matrix improves temporal
locality by changing the data access pattern. By doing so, entries
that are in the cache are useful for a longer time. This has given us
a marginal improvement at this point.

Next we tried to leverage GCC's built-in optimizations by compiling the
existing code with `-O3`. This gave us another dramatic (~10x) decrease
in execution time. Looking at the list of optimizations using the
`-fopt-info` flag, we saw that most of the optimizations performed
were:

- Inlining
- Vectorization
- Loop unrolling

However, using the option `f-opt-info-missed` showed us that there's
still room to grow in terms of compiler optimizations.

Next we combined `-O3` with matrix transposition and this time we got
a ~2x speedup again. It seems that we were able to remove other
factors of inefficiency with `-O3` which let the cache optimizations
shine through.

Finally, we tried outsourcing matrix multiplication to the OpenBLAS
library. Since we used nested `std::vector` instances to store our
data and `cblas_dgemm` --- our matrix multiplication function ---
required pointers to 1D arrays, we had to take a performance hit and
transform the 2D structures down into 1D. Then, our code for invoking
the multiplication function looked like this:

```cpp
cblas_dgemm(
    CblasRowMajor,  // Row-major storage
    CblasNoTrans,   // No transpose for A
    CblasNoTrans,   // No transpose for B
    rowsA,          // Number of rows in A and C
    colsB,          // Number of columns in B and C
    colsA,          // Number of columns in A and rows in B
    alpha,          // Scalar alpha
    A_flat.data(),  // Matrix A
    rowsA,          // Leading dimension of A
    B_flat.data(),  // Matrix B
    rowsB,          // Leading dimension of B
    beta,           // Scalar beta
    C_flat.data(),  // Result matrix C
    rowsA           // Leading dimension of C
);
```

Where `A_flat`, `B_flat`, and `C_flat` are flattened vectors
containing our data.

Despite the data transformation overhead, we still saw a 2x
improvement, which was further enhanced with using `-O3`, leading us
to the final result of 43.2692 ms.

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
Moving from FP64 (double precision) to FP32 (single precision) significantly reduced computational overhead. This final tweak achieved our best runtime of **27.318 ms** for multiplying three 1000x1000 matrices.

```cpp
    // Allocate and initialize host memory
    // Change Double -> Float
    std::vector<float> A(n * n, 1.0); // Matrix A
    std::vector<float> B(n * n, 1.0); // Matrix B
    std::vector<float> C(n * n, 0.0); // Result matrix
```

## Takeaways
In summary, our exploration of matrix multiplication optimization demonstrated the power of systematically improving performance across different layers of the hardware and software stack. Starting with a naive implementation, we saw how straightforward techniques like OpenMP parallelization and matrix transposition could significantly improve CPU performance by enhancing parallelism and cache efficiency. Compiler optimizations with -O3 further reduced execution times by applying advanced techniques like inlining and vectorization, but even these improvements had their limits on CPUs. Transitioning to GPUs on Bridges-2 brought a dramatic shift in performance, with CuBLAS providing an excellent starting point and additional optimizations—such as minimizing memory transfers, utilizing Tensor Cores, and switching precision to FP32—further reducing runtimes. This project underscored the critical importance of memory efficiency, hardware-specific optimizations, and leveraging accelerators like GPUs for parallel tasks. By carefully tuning each layer, we not only achieved our primary goal but also gained a deeper understanding of modern computer architecture and performance optimization.

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
