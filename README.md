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
