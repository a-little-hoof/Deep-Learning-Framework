#include <iostream>
#include "helper.h"
#include "tensor.h"
#include <stdio.h>
#include <cuda.h>
using namespace std;

int main(){
    vector<int> shape = {2,8};
    Tensor tensor(shape, "CPU");

    // Initialize tensor data on CPU
    for (int i = 0; i < tensor.get_size(); ++i) {
        tensor.data[i] = i - 8;  // Some values will be negative, some positive
    }

    print_array(tensor.data, tensor.get_size(), "Initial Tensor (CPU)");

    // Move tensor to GPU
    tensor.gpu();

    // Allocate output tensor on GPU
    Tensor output_tensor(shape, "GPU");

    // Launch ReLU kernel
    int size = tensor.get_size();
    relu<<<CudaGetBlocks(kCudaThreadsNum), kCudaThreadsNum>>>(tensor.data, output_tensor.data);
    cudaDeviceSynchronize();

    // Move result back to CPU
    output_tensor.cpu();

    // Print result
    print_array(output_tensor.data, output_tensor.get_size(), "ReLU Output (CPU)");
    return 0;
}