#include <stdio.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <helper.h>
#include <tensor.h>

using namespace std;

Tensor::Tensor(vector<int> s, string d): shape(s), device(d), data(nullptr){
    int size = Tensor::get_size();
    if (device=="CPU"){
        data = (float*)malloc(size * sizeof(float));
    }
    else{
        cudaMalloc(&data, size*sizeof(float));
    }
}

Tensor::~Tensor(){
    if (device=="CPU"){
        free(data);
    }
    else{
        cudaFree(data);
    }
}

void Tensor::cpu(){
    if (device=="GPU"){
        float* data_cpu = nullptr;
        int size = get_size();
        data_cpu = (float*)malloc(size*sizeof(float));
        cudaMemcpy(data_cpu, data, size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(data);
        data = data_cpu;
        device = "CPU";
    }
}

void Tensor::gpu(){
    if (device=="CPU"){
        float* data_gpu = nullptr;
        int size = get_size();
        cudaMalloc(&data_gpu, size*sizeof(float));
        cudaMemcpy(data_gpu, data, size*sizeof(float), cudaMemcpyHostToDevice);
        free(data);
        data = data_gpu;
        // cudaMemcpy(data, data_gpu, size*sizeof(float), cudaMemcpyDeviceToHost);
        device = "GPU";
    }
}

int Tensor::get_size(){
    int size = 1;
    for (int i=0; i<shape.size(); ++i){
        size *= shape[i];
    }
    return size;
}