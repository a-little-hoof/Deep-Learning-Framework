#include <iostream>
#include "helper.h"
using namespace std;

__global__ void relu(float* in, float* out){
    int i = threadIdx.x;
    out[i] = in[i]>0?in[i]:0;
    // printf("Hello, CUDA!\n");
}

__global__ void relu_backward(float* in_gradient, float* input, float* out_gradient){
    int i = threadIdx.x;
    out_gradient[i] = input[i]>0 ? in_gradient[i]:0;
}

__global__ void sigmoid(float* in, float* out){
    int i = threadIdx.x;
    out[i] = 1.0f/(1.0f+expf(-in[i]));
}

__global__ void sigmoid_backward(float* in_gradient, float* input, float* out_gradient){
    int i = threadIdx.x;
    float y = 1.0f/(1.0f+expf(-input[i]));
    out_gradient[i] = in_gradient[i]*y*(1-y);
}


void print_array(const float* arr, int num, const char* name){
    printf("\n\n%s:\n", name);
    for (int i=0; i<num; ++i){
        if(i%8==0){
            printf("\n");
        }
        printf("%8.1f", arr[i]);
    }
    printf("\n");
}