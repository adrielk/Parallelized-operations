//By: Adriel Kim, a learning soldification mini-project
//Statistical measure calculator
//4-18-2020
//Incorporating chapters 1-6 from CUDA by Example (with the exception of constant memory)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>  

void die(const char* message);
void checkError();


//Description: This program will calculate the follow statistical measures of a large array of numbers: Mean, Max, Min, and Standard Deviation
//This program will be able to accommodate any array size. It will simply take more time if the array is too large. (We are fixing #of blocks and threads)

//Mean: The GPU will be used to sum all numbers in an array in parallel. CPU will do the final division
//Max: The GPU will find max using shared memory and binary reduction. We will have CPU take care of the last step.
//Min: Computed the same way as max
//Standard Deviation: The GPU kernel will take in an array of numbers and the mean of the array to do std calculations on every number in parallel

//To do later: Add in file reader for user-inputed data.
//Add in a timer. (DONE)
//Implement a CPU version of the measures in order to compare results
//***Work on optimizing your reductions, like Elaheh did with hers (eliminating unnecessary array accesses when array size is not a mutliple of threadsPerBlock

#define imin(a,b)  (a<b?a:b)//example of ternary operator in c++

const int N(33 * 1024);//Array size
const int threadsPerBlock = 256;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//this will be our output array size for sumKernel.

//kernel call is as follows: 
// kernel_name << <blocksPerGrid, threadsPerBlock >> > (params);


//Standard dev implementation: bassically sumKernel with some modifications to the elements we're summing
//With reduciton, we're able to significantly reduce time by allowing many summings to be done in parallel.
//A CPU implementation would take linear time, since we'd have to do computations for each element and the sum them all up.
//After this kernel is done, the CPU will take care of the last parts of the sum, divide by N, and take the square root of the result***
__global__ void stdKernel(float* in, float* out, float mean) {
    __shared__ float cache[threadsPerBlock];
    int block_id = blockIdx.x + gridDim.x * blockIdx.y;
    int thread_id = blockDim.x * block_id + threadIdx.x;
    int cacheIndex = threadIdx.x;
    
    if (thread_id < N)//loading data corresponding to each thread onto shared memory
        cache[cacheIndex] = in[thread_id];
    __syncthreads();

    int i = blockDim.x / 2;//our reduction de binary
    while (i != 0) {
        if (cacheIndex < i) {
            float temp = cache[cacheIndex] - mean;
            float temp2 = cache[cacheIndex + i] - mean;
            temp2 *= temp2;
            cache[cacheIndex] = temp * temp;

            cache[cacheIndex] += temp2;
        }
        __syncthreads();
        i /= 2;
    }


    if (cacheIndex == 0) {
        out[block_id] = cache[0];
    }

}

__global__ void sumKernel(float* in, float* out, const int N)
{
    __shared__ float cache[threadsPerBlock];//a shared array for all threads in a singular block.
    int block_id = blockIdx.x + gridDim.x * blockIdx.y;//NOTE: this block_id goes up to the # of blocks per grid. NOTE2: grimDim.x would be 0 for our purposes???
    int thread_id = blockDim.x * block_id + threadIdx.x;
    int cacheIndex = threadIdx.x;

    if (thread_id < N)//loading data corresponding to each thread onto shared memory
        cache[cacheIndex] = in[thread_id];
    __syncthreads();

    int i = blockDim.x / 2;//our reduction de binary
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];//current cache becomes to the sum of itself and the other "interwoven" thread element
        }
        __syncthreads();
        i /= 2;
    }


    if (cacheIndex == 0) {
        out[block_id] = cache[0];
    }
}

//Max kernel works similarly to sumKernel in that it uses binary reduction. It's slightly modified to find max
__global__ void maxKernel(float* in, float* out, int N) {

    // Determine the "flattened" block id and thread id
    __shared__ float cache[threadsPerBlock];

    int block_id = blockIdx.x + gridDim.x * blockIdx.y;
    int thread_id = blockDim.x * block_id + threadIdx.x;
    int cacheIndex = threadIdx.x;
    //basically, load all values into the cache first, in parallel.
    //wait for all threads to sync before moving on. Only make threadIdx.x == 0 calculate max, using the cache array instead of in;)

    //printf("blockid: %d ; cacheIndex: %d\n", block_id, cacheIndex);
    if (thread_id < N)
        cache[cacheIndex] = in[thread_id];
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            if (cache[cacheIndex] < cache[cacheIndex + i]) {
                cache[cacheIndex] = cache[cacheIndex + i];
            }
        }
        __syncthreads();
        i /= 2;
    }


    if (cacheIndex == 0) {
        out[block_id] = cache[0];
    }
}

//exactly like maxKernel but less than sign changed to greater than 
__global__ void minKernel(float* in, float* out, int N) {

    // Determine the "flattened" block id and thread id
    __shared__ float cache[threadsPerBlock];

    int block_id = blockIdx.x + gridDim.x * blockIdx.y;
    int thread_id = blockDim.x * block_id + threadIdx.x;
    int cacheIndex = threadIdx.x;
    //basically, load all values into the cache first, in parallel.
    //wait for all threads to sync before moving on. Only make threadIdx.x == 0 calculate max, using the cache array instead of in;)

    //printf("blockid: %d ; cacheIndex: %d\n", block_id, cacheIndex);
    if (thread_id < N)
        cache[cacheIndex] = in[thread_id];
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            if (cache[cacheIndex] > cache[cacheIndex + i]) {
                cache[cacheIndex] = cache[cacheIndex + i];
            }
        }
        __syncthreads();
        i /= 2;
    }


    if (cacheIndex == 0) {
        out[block_id] = cache[0];
    }
}

// Prints the specified message and quits
void die(const char* message) {
    printf("%s\n", message);
    exit(1);
}

void checkError() {
    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error) {
        char message[256];
        sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
        die(message);
    }
}

int main()
{
   // double* a, * b, c, * partial_c;
   // double* dev_a, * dev_b, * dev_partial_c;

    float* in, * out;
    float* dev_in, * dev_out;

    in = (float*)malloc(N * sizeof(float));
    out = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //filling up the array with random data to test
    for (int i = 0;i < N;i++) {
        in[i] = rand() % 1000 + 1;//assigns random number between 1 and 1000
    }

    cudaEventRecord(start, 0);

    //Mean implementation: We sum by binary reduction. Our sum will be the at the beginning of the array. CPU will do final step of the sum and division.
    if (cudaMemcpy(dev_in, in, sizeof(float) * N, cudaMemcpyHostToDevice) != cudaSuccess) die("Error allocating memory to GPU");
    if(cudaMemcpy(dev_out, out, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice)!=cudaSuccess) die("Error allocating memory to GPU");

    sumKernel << <blocksPerGrid, threadsPerBlock >> > (dev_in, dev_out, N);

    if(cudaMemcpy(out, dev_in, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost)!=cudaSuccess) die("Error allocating memory to CPU");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to calculate average: %3.1f ms\n", elapsedTime);

    //final step of mean
    float avg = 0;
    for (int i = 0;i < blocksPerGrid;i++) {//maybe inefficient to go through entire block length for non multiples of 256, but whatever...
        avg += out[i];
    }
    avg = avg / N;
 
    printf("Average: %f\n", avg);
    /*------------------------------------------------------------------*/
    //finding MAX

    cudaEventRecord(start, 0);

    maxKernel << <blocksPerGrid, threadsPerBlock >> > (dev_in, dev_out, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to calculate max: %3.1f ms\n", elapsedTime);

    if (cudaMemcpy(dev_out, out, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost) != cudaSuccess) die("Error allocating memory to CPU");

    //final step of max;
    float max = out[0];
    for (int i = 1;i < blocksPerGrid;i++) {
        if (out[i] > max) {
            max = out[i];
        }
    }
    printf("Max Value: %f\n", max);
    /*------------------------------------------------------------------*/
    //finding MIN

    cudaEventRecord(start, 0);

    minKernel << <blocksPerGrid, threadsPerBlock >> > (dev_in, dev_out, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to calculate min: %3.1f ms\n", elapsedTime);

    if (cudaMemcpy(dev_out, out, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost) != cudaSuccess) die("Error allocating memory to CPU");

    //final step of min;
    float min = out[0];
    for (int i = 1;i < blocksPerGrid;i++) {
        if (out[i] < min) {
            min = out[i];
        }
    }
    printf("Min Value: %f\n", min);

    /*------------------------------------------------------------------*/
    //calculating standard deviation

    cudaEventRecord(start, 0);
    
    stdKernel << <blocksPerGrid, threadsPerBlock >> > (dev_in, dev_out, N);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to calculate standard deviation: %3.1f ms\n", elapsedTime);

    if (cudaMemcpy(dev_out, out, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost) != cudaSuccess) die("Error allocating memory to CPU");

    //final step of std;

    float std = 0;
    for (int i = 0;i < threadsPerBlock;i++) {
        std += out[i];
    }
    std /= N;
    std = sqrt(std);

    printf("Standard Deviation: %f\n", std);

    /*------------------------------------------------------------------fin*/ 
    free(in);
    free(out);
    cudaFree(dev_in);
    cudaFree(dev_out);
   
    return 0;

}


