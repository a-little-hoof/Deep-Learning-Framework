#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

class Tensor {
public:
    std::string device;
    std::vector<int> shape;
    int size;
    float* number;
    float* grad;

    void cpu() {
        device = "cpu";
    }

    void gpu() {
        device = "gpu";
    }

    Tensor(const std::vector<int>& shape, const std::string& device)
        : shape(shape), device(device) {
        size = 1;
        for (int dim : shape) {
            size *= dim;
        }

  
        number = new float[size]();
        grad = new float[size]();
    }

    ~Tensor() {
        delete[] number;
        delete[] grad;
    }
};


__global__ void ReLUForwardKernel(float* d_in, float* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] > 0 ? d_in[idx] : 0.0f;
    }
}


__global__ void ReLUBackwardKernel(float* d_in, float* d_out_grad, float* d_in_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_in_grad[idx] = (d_in[idx] > 0) ? d_out_grad[idx] : 0.0f;
    }
}
__global__ void SigmoidForwardKernel(float* d_in,float* d_out,int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] =1.0f/(1.0f+expf(-d_in[idx]));
    }

}
__global__ void SigmoidBackwardKernel(float* d_out,float* d_out_grad,float* d_in_grad,int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size){
        d_in_grad[idx] = d_out_grad[idx] * d_out[idx] * (1 - d_out[idx]);
    }
}

void ReLUForward(Tensor& input, Tensor& output) {
    float *d_in, *d_out;
    int size = input.size * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, input.number, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (input.size + threadsPerBlock - 1) / threadsPerBlock;
    ReLUForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, input.size);
    cudaMemcpy(output.number, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
void SigmoidForward(Tensor& input,Tensor& output){
    float *d_in, *d_out;
    int size = input.size * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, input.number, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (input.size + threadsPerBlock - 1) / threadsPerBlock;
    SigmoidForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, input.size);
    cudaMemcpy(output.number, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

}

void ReLUBackward(Tensor& input, Tensor& output) {
    float *d_in, *d_in_grad, *d_out_grad;
    int size = input.size * sizeof(float);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_in_grad, size);
    cudaMalloc(&d_out_grad, size);

    cudaMemcpy(d_in, input.number, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_grad, output.grad, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (input.size + threadsPerBlock - 1) / threadsPerBlock;
    ReLUBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out_grad, d_in_grad, input.size);

    cudaMemcpy(input.grad, d_in_grad, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_in_grad);
    cudaFree(d_out_grad);
}
void SigmoidBackward(Tensor& input, Tensor& output) {
    float *d_out, *d_out_grad, *d_in_grad;
    int size = input.size * sizeof(float);

    cudaMalloc(&d_out, size);
    cudaMalloc(&d_in_grad, size);
    cudaMalloc(&d_out_grad, size);
    cudaMemcpy(d_out, output.number, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_grad, output.grad, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (input.size + threadsPerBlock - 1) / threadsPerBlock;
    SigmoidBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_out_grad, d_in_grad, input.size);
    cudaMemcpy(input.grad, d_in_grad, size, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in_grad);
    cudaFree(d_out_grad);
}


int main() {
    std::vector<int> shape = {2, 2};


    Tensor input(shape, "cpu");
    Tensor output(shape, "cpu");

    input.number[0] = -1.0f;
    input.number[1] = 0.0f;
    input.number[2] = 1.0f;
    input.number[3] = 2.0f;


    output.grad[0] = 1.0f;
    output.grad[1] = 2.0f;
    output.grad[2] = 3.0f;
    output.grad[3] = 4.0f;


    ReLUForward(input, output);
    ReLUBackward(input, output);

    
    std::cout << "ReLU forward output:" << std::endl;
    for (int i = 0; i < output.size; i++) {
        std::cout << output.number[i] << " ";
    }
    std::cout << std::endl;


    std::cout << "ReLU backward input gradients:" << std::endl;
    for (int i = 0; i < input.size; i++) {
        std::cout << input.grad[i] << " ";
    }
    std::cout << std::endl;

    SigmoidForward(input, output);
    SigmoidBackward(input, output);

    std::cout << "Sigmoid forward output:" << std::endl;
    for (int i = 0; i < output.size; i++) {
        std::cout << output.number[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Sigmoid backward input gradients:" << std::endl;
    for (int i = 0; i < input.size; i++) {
        std::cout << input.grad[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
