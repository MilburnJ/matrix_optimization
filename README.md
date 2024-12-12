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

## State-of-the-art Solutions

## Approach
