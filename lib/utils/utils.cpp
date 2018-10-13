#include <iostream>

void printMatrix(double *matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            std::cout << matrix[i * m + j] << " ";
        std::cout << std::endl;
    }
}