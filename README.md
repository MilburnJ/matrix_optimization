8-9 minutes + 1-2 mins Q&A

# Goal set during prelim presentation (2 bullets)

- <50 ms for multiplying three 1000x1000 matrices.
- Optionally, <1ms (sub-millisecond) runtime.

# Project/Problem Motivation

Exploration of all the different layers.

# State of the Art and findings about the problem(who else has worked on this problem?)

https://www.youtube.com/watch?v=QGYvbsHDPxo

https://github.com/fangjunzhou/blas-playground

https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

# Approach

## Step By Step Optimization

Starting from the naive code, increasing software and hardware resources.

## CPU, then GPU

We strive to optimize for the CPU first just to see what our
performance ceiling may be and then switch to the GPU.

# Struggles

# Experimental Set up including hardware/software information

## Hardware Choices

Jakeb's Desktop PC, Giorgi's Desktop PC, bridges2.

## Software Stack

- g++
- OpenBLAS - https://github.com/OpenMathLib/OpenBLAS
- CuBlas - https://developer.nvidia.com/cublas

# Results

## Baseline - Naive Algorithm

This is the bare-bones approach.

```
g++ basic_multiply.cpp -o basic
```

`n` = 1000

- My Desktop PC: 17.7763 seconds
- Jakeb's Desktop PC: 13 seconds

## OpenMP Parallel for

We prepend the main for loop with `#pragma omp parallel for`

```
g++ basic_multiply.cpp -o basic -fopenmp
```

2.89815 seconds

## Transposing the second matrix

Just put the transposition code before the multiplication code.

This achieves a better cache hit rate.

2.6083 seconds

## -O3 without transposition

```
g++ basic_multiply.cpp -o basic -fopenmp -O3
```

268.257 ms

## -O3 with transposition

```
g++ transpose.cpp -o trans -fopenmp -O3
```

137.566 ms

## OpenBLAS

```
g++ openblas.cpp -o obl -lopenblas
```

55.0175 ms

## OpenBLAS with -O3

43.2692 ms

## CuBlas with 4 GPUS on Bridges 2

37.6527 ms

# Takeaways

## OpenMP Makes Simple Parallelization very easy

## SIMD is cool!

## GPUs are very efficient for Matrix Multiplication

# Did we meet the goals set during prelim presentation?

As a reminder, our goals were:

- <50 ms for multiplying three 1000x1000 matrices.
- Optionally, <1ms (sub-millisecond) runtime.

The first goal was met!

# What would you do differently the next time?

No ragrets.
