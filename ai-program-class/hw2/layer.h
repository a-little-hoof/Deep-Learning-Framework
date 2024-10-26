#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <cublas_v2.h>

void fc_forward(const Tensor& X, const Tensor& W, const Tensor& b,Tensor& Y);
void fc_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX);

//conv
void conv_forward(const Tensor& X, const Tensor& W, Tensor& Y);
void conv_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX);
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    float* data_col);

//maxpool
void maxpool_forward(const Tensor& X, Tensor& Y, Tensor& mask);
void maxpool_backward(const Tensor& dY, const Tensor& mask, Tensor& dX);

//softmax
void softmax_forward(const Tensor& X, Tensor& Y);
void softmax_backward(const Tensor& dY, const Tensor& Y, Tensor& dX);

//cross entropy loss
void cross_entropy_forward(const Tensor& X, const Tensor& label, Tensor& loss);
void cross_entropy_backward(const Tensor& X, const Tensor& label, Tensor& dX);

//other functions
void gemm_gpu(cublasOperation_t trans_A, cublasOperation_t trans_B, const int m, const int k, const int n, 
const float alpha, const float *A, const float *B, const float beta, float *C);

#endif
