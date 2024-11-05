// --------------------------------------------------------
// Modified from Octree-based Sparse Convolutional Neural Networks
// Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
// --------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(condition)                                        \
  do {                                                               \
    cudaError_t error = condition;                                   \
    if (error != cudaSuccess) {                                      \
      printf("CHECK failed: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  } while (0)


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                       \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: number of threads per block
constexpr int kCudaThreadsNum = 512;

// CUDA: number of blocks for threads.
inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}