/**
 *  CUDA PARALLEL PROGRAMMING: matrix_ops.cu
 *  Purpose: Matrix Operations using CUDA C/C++
 *  @author Prabhsimran Singh
 *  @version 1.0 17/09/18
 *
 *  Build using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_ops.so matrix_ops.cu
 */

#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "utils/devices.cu"
#include "utils/utils.cpp"

#define BLOCK_SIZE 256

/**
* Calculates element-wise sum of two matrices (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double pointer to second input array
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void matSum(double *a, double *b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = a[row * n + col] + b[row * n + col];
}

/**
* Calculates element-wise product of two matrices (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double pointer to second input array
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void matProd(double *a, double *b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = a[row * n + col] * b[row * n + col];
}

/**
* Calculates dot-product of two matrices (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double pointer to second input array
* @param c the double pointer to output array
* @param m the no. rows in a(m x n) and c(m x k)
* @param n the no. cols in a(m x n) and rows in b(n x k)
* @param k the no. cols in b(n x k) and c(m x k)
* @return void
*/
__global__ void matMul(double *a, double *b, double *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0;

    if (row < m && col < k) {
        for (int i = 0; i < n; i++)
            sum += a[row * n + i] * b[i * k + col];
        c[row * k + col] = sum;
    }
}

extern "C" {

    void cuda_mat_sum(double *a, double *b, double *c, int m, int n) {
        double *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_b, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matSum<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_mat_prod(double *a, double *b, double *c, int m, int n) {
        double *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_b, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matProd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
 
    void cuda_mat_mul(double *a, double *b, double *c, int m, int n, int k) {
        double *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_b, (n * k) * sizeof(double));
        cudaMallocManaged(&d_c, (m * k) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (n * k) * sizeof(double), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
        cudaDeviceSynchronize();
    
        cudaMemcpy(c, d_c, (m * k) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
 }