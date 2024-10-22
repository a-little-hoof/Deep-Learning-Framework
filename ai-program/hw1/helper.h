#ifndef HELPER_H
#define HELPER_H

#include <cuda_runtime.h>

#define kCudaThreadsNum 512

inline int CudaGetBlocks(const int N){
    return (N + kCudaThreadsNum -1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n)                       \
    for(int i = blockIdx.x*blockDim.x + threadIdx.x; \
    i < (n);                                         \
    i += blockDim.x *gridDim.x)


__global__ void relu(float* in, float* out);

__global__ void relu_backward(float* in, float* out);

__global__ void sigmoid(float* in, float* out);

__global__ void sigmoid_backward(float* in, float* out);


void print_array(const float* arr, int num, const char* name);


#endif