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
#include "utils/devices.cu"
#include "utils/utils.cpp"

#define BLOCK_SIZE 256

/**
* Calculates dot-product of two matrices (using parallel threads on CUDA capable device)
*
* @param a the float pointer to first input array
* @param b the float pointer to second input array
* @param c the float pointer to output array
* @param m the no. rows in a(m x n) and c(m x k)
* @param n the no. cols in a(m x n) and rows in b(n x k)
* @param k the no. cols in b(n x k) and c(m x k)
* @return void
*/
__global__ void matmul(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    if (row < m && col < k) {
        for (int i = 0; i < n; i++)
            sum += a[row * n + i] * b[i * k + col];
        c[row * k + col] = sum;
    }
}

/**
* Calculates element-wise sum of two matrices (using parallel threads on CUDA capable device)
*
* @param a the float pointer to first input array
* @param b the float pointer to second input array
* @param c the float pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void matsum(float *a, float *b, float *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = a[row * n + col] + b[row * n + col];
}

/**
* Calculates element-wise product of two matrices (using parallel threads on CUDA capable device)
*
* @param a the float pointer to first input array
* @param b the float pointer to second input array
* @param c the float pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void matprod(float *a, float *b, float *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = a[row * n + col] * b[row * n + col];
}

/**
* Calculates element-wise sum of a matrix with a value (using parallel threads on CUDA capable device)
*
* @param a the float pointer to first input array
* @param b the float value to add the array with
* @param c the float pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void elemwise_sum(float *a, float b, float *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = a[row * n + col] + b;
}

/**
* Calculates element-wise product of a matrix with a value (using parallel threads on CUDA capable device)
*
* @param a the float pointer to first input array
* @param b the float value to multiply the array with
* @param c the float pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void elemwise_prod(float *a, float b, float *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = a[row * n + col] * b;
}

/**
* Calculates element-wise maximum of a matrix with a value (using parallel threads on CUDA capable device)
*
* @param a the float pointer to first input array
* @param b the float value to check maximum against
* @param c the float pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void elemwise_max(float *a, float b, float *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        c[row * n + col] = (a[row * n + col] > b) ? a[row * n + col] : b;
}

extern "C" {

    void cuda_matmul(float *a, float *b, float *c, int m, int n, int k) {
        float *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_b, (n * k) * sizeof(float));
        cudaMallocManaged(&d_c, (m * k) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (n * k) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
        cudaDeviceSynchronize();
    
        cudaMemcpy(c, d_c, (m * k) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_matsum(float *a, float *b, float *c, int m, int n) {
        float *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_b, (m * n) * sizeof(float));
        cudaMallocManaged(&d_c, (m * n) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (m * n) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matsum<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_matprod(float *a, float *b, float *c, int m, int n) {
        float *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_b, (m * n) * sizeof(float));
        cudaMallocManaged(&d_c, (m * n) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (m * n) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matprod<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_elemwise_sum(float *a, float b, float *c, int m, int n) {
        float *d_a, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_c, (m * n) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        elemwise_sum<<<dimGrid, dimBlock>>>(d_a, b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }

    void cuda_elemwise_prod(float *a, float b, float *c, int m, int n) {
        float *d_a, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_c, (m * n) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        elemwise_prod<<<dimGrid, dimBlock>>>(d_a, b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }
 
    void cuda_elemwise_max(float *a, float b, float *c, int m, int n) {
        float *d_a, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_c, (m * n) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        elemwise_max<<<dimGrid, dimBlock>>>(d_a, b, d_c, m, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * n) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }
 }