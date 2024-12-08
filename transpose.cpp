#include <iostream>
#include <vector>
#include <chrono>

// Function to multiply two matrices
// std::vector<std::vector<int>> multiplyMatricesOld(
//     const std::vector<std::vector<int>> &A,
//     const std::vector<std::vector<int>> &B) {

//     int rowsA = A.size();
//     int colsA = A[0].size();
//     int colsB = B[0].size();

//     // Initialize result matrix with zeros
//     std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

//     // Naive multiplication (triple nested loop)
// #pragma omp parallel for
//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             for (int k = 0; k < colsA; ++k) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
//     return C;
// }

std::vector<std::vector<int>> multiplyMatrices(
    const std::vector<std::vector<int>> &A,
    const std::vector<std::vector<int>> &B) {

    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

    // Transpose matrix B
    std::vector<std::vector<int>> BT(colsB, std::vector<int>(rowsA));
    for (int i = 0; i < colsB; ++i) {
        for (int j = 0; j < rowsA; ++j) {
            BT[i][j] = B[j][i];
        }
    }

    // Initialize result matrix with zeros
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    // Naive multiplication (triple nested loop)
#pragma omp parallel for
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * BT[j][k];
            }
        }
    }
    return C;
}

int main() {
    int n = 1000; // Example matrix size (500x500)

    // Initialize matrices A, B, and C with random values
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 1));

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // First, multiply A and B
    std::vector<std::vector<int>> AB = multiplyMatrices(A, B);

    // Then, multiply the result with C
    std::vector<std::vector<int>> ABC = multiplyMatrices(AB, C);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken to multiply three matrices: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
