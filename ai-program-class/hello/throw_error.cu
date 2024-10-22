#include "stdio.h"
#include "cuda_runtime.h"

__global__ void hello( )
{
   printf("GPU!");
   printf("blockIdx.x=%d/%d blocks, threadIdx.x=%d/%d threads\n",
                        blockIdx.x,  gridDim.x,
                        threadIdx.x, blockDim.x);
}

int main()
{
   // hello<<< 1, 1025 >>>( );    // Error !!!
   hello<<< 1, 256 >>>( );    // Correct !!!
   cudaError_t err = cudaGetLastError();        // Get error code

   if ( err != cudaSuccess )
   {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
   }

   printf("I am the CPU: Hello World ! \n");

   cudaDeviceSynchronize();

   return 0;
}
