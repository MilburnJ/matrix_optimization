#include <iostream>
#include <vector>
#include <chrono>
#include <openblas/cblas.h>

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

// Function to flatten a nested vector into a flat array
std::vector<double> flatten(const std::vector<std::vector<double>>& nestedVector) {
    std::vector<double> flatVector;
    for (const auto& row : nestedVector) {
        flatVector.insert(flatVector.end(), row.begin(), row.end());
    }
    return flatVector;
}

std::vector<std::vector<double>> multiplyMatrices(
    const std::vector<std::vector<double>> &A,
    const std::vector<std::vector<double>> &B) {

    int rowsA = A.size();
    int rowsB = B.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

//     // Transpose matrix B
//     std::vector<std::vector<int>> BT(colsB, std::vector<int>(rowsA));
//     for (int i = 0; i < colsB; ++i) {
//         for (int j = 0; j < rowsA; ++j) {
//             BT[i][j] = B[j][i];
//         }
//     }

    // Initialize result matrix with zeros
    std::vector<double> C_flat(rowsA * colsB, 0.0);

    // C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;

    std::vector<double> A_flat = flatten(A);
    std::vector<double> B_flat = flatten(B);

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


//     // Naive multiplication (triple nested loop)
// #pragma omp parallel for
//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             for (int k = 0; k < colsA; ++k) {
//                 C[i][j] += A[i][k] * BT[j][k];
//             }
//         }
//     }

    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i][j] = C_flat[i * colsB + j];
        }
    }

    return C;

}

int main() {
    int n = 1000; // Example matrix size (500x500)

    // Initialize matrices A, B, and C with random values
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> B(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 1.0));

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // First, multiply A and B
    std::vector<std::vector<double>> AB = multiplyMatrices(A, B);

    // Then, multiply the result with C
    std::vector<std::vector<double>> ABC = multiplyMatrices(AB, C);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken to multiply three matrices: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
