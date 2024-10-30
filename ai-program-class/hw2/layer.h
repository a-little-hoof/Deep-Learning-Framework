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
    const int height, const int width, 
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    float* data_col);
__global__ void col2im_gpu_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    float* data_im);

//maxpool
void maxpool_forward(const Tensor& X, Tensor& Y, Tensor& mask);
void maxpool_backward(const Tensor& dY, const Tensor& mask, Tensor& dX);
__global__ void max_pool_forward_kernel(int nthreads, float* in_data,
    int channels, int in_h, int in_w, int out_h, int out_w, 
    int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
    float* out_data, float* out_mask);
__global__ void max_pool_backward_kernel(int nthreads, float* in_data,
    float* mask, int channels, int in_h, int in_w, int out_h, int out_w, 
    int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
    float* out_data);

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
