//Author: Adriel Kim
//6-27-2020
/*
Desc: Basic 2D matrix operations such as element-wise addition, subtraction, multiplication, and division.
In addition, slightly more complex operations such as dot product.

Challenge: Learning how to represent 2D matices in C++ and mapping 2D indices to kernel
To do:
- Double check if all memory is freed
- Optimize by eliminating redundant calculations
- Test code on department servers
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

//define imin(a,b)  (a<b?a:b)//example of ternary operator in c++
#define N (100)//# of elements in matrices
const int threadsPerBlock = 128;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = 128;//imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//this will be our output array size for sumKernel.

using namespace std;

//any advantages with mapping directly to strucutre of matrix? We're just representing 2D matrix using 1D array...
//it would be difficult to do the above since we want the operations to occur over abitrarily large matrices
//this can definitely be optimzied by elminating redundant calculations
__global__ void matrixAddKernel(int *c, const int *a, const int *b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        //adds total number of running threads to tid, the current index.
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void matrixSubtractKernel(int* c, const int* a, const int* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] - b[tid];
        //adds total number of running threads to tid, the current index.
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void matrixMultiplyKernel(int* c, const int* a, const int* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void matrixDivideKernel(int* c, const int* a, const int* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = (a[tid]/b[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

void printMatrix(int* arr, int dims) {
    for (int i = 0;i < dims; i++) {
        for (int k = 0;k < dims; k++) {
            cout << (arr[k + i * dims]);
        }
        cout << endl;
    }
}

int main()
{
    //figure out how to dynamically allocate
    const int rows = 10;
    const int cols = 10;
    int inc = 0;
    int outputs[rows * cols];
    int intMatrix[rows*cols];
    int intMatrix2[rows*cols];
    int operation = 0;
    cout << "Enter which operation (1 = add, 2 = subtract, 3 = multiply, 4 = divide)" << endl;
    cin >> operation;
    //populated 2D array with data
    for (int i = 0;i<rows;i++) {
        for (int k = 0;k < cols;k++) {
            intMatrix[k+i*rows] = inc;
            intMatrix2[k+i*rows] = inc;
            outputs[k + i * rows] = 0;
           // cout << (intMatrix[i][k]);
            inc++;
        }
        //cout << endl;
    }
    cout << "Matrix 1:" << endl;
    printMatrix(intMatrix, rows);
    cout << "Matrix 2:" << endl;
    printMatrix(intMatrix2, rows);
    cudaError_t cudaStatus = matrixOperation(outputs, intMatrix, intMatrix2, N, operation);
    cout << "Resulting Matrix:" << endl;
    printMatrix(outputs, rows);

    return 0;
}
cudaError_t matrixOperation(int* c, const int* a, const int* b, unsigned int arrSize, int operation) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    switch (operation) {
        case 1:
            matrixAddKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;
        case 2:
            matrixSubtractKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;
        case 3:
            matrixMultiplyKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;
        case 4:
            matrixDivideKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;

    }
    //copies result to host so we can use it.
    cudaStatus = cudaMemcpy(c, dev_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;


}

