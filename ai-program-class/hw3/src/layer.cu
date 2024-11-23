#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <string>
#include "tensor.h"
#include "layer.h"
#include "utils.h"
#include <float.h>
#include <curand.h>

// Fill the matrix with random numbers on GPU
void matrix_init(Tensor& X) {

    float* data = X.data;
    int size = X.get_size();
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, data, size);
    curandDestroyGenerator(prng);
}

// fc layer forward and backward
// X: [N, C_in], Y: [N, C_out], W: [C_in, C_out], b: [C_out]
// backward function takes partial L / partial y and outputs parital L / partial W or partial L / partial X

void fc_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y){
    int batch_size = X.shape[0];
    int in_features = X.shape[1];
    int out_features = W.shape[1];

    // matrix product with gemm
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, out_features, in_features,  
        1.0, X.data, W.data, 0.0, Y.data);

    // add bias
    Tensor ones_(std::vector<int>{batch_size, 1}, "GPU");
    ones_.fill_(1.0);
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, out_features, 1,
        1.0, ones_.data, b.data, 1.0, Y.data);
}

// dY: [N, C_out], X: [N, C_in], W: [C_in, C_out]
// dW: [C_in, C_out], db: [C_out], dX: [N, C_in]
void fc_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX){

    int batch_size = X.shape[0];
    int in_features = X.shape[1];
    int out_features = W.shape[1];

    // dW = X^T * dY
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, in_features, out_features, batch_size,   
        1.0, X.data, dY.data, 0.0, dW.data);

    // dX = dY * W^T
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, batch_size, in_features, out_features,  
        1.0, dY.data, W.data, 0.0, dX.data);
    

}

//conv_layer forward and backward
// X: [N, C_in, H, W], Y: [N, C_out, H_out, W_out], W: [C_out, C_in, 3, 3], b: [C_out]
// backward function takes partial L / partial y and outputs parital L / partial W or partial L / partial X
void conv_forward(const Tensor& X, const Tensor& W, Tensor& Y){

    int batch_size = X.shape[0];
    int C_in = X.shape[1];
    int height = X.shape[2];
    int width = X.shape[3];
    int C_out = Y.shape[1];
    int H_out = Y.shape[2];
    int W_out = Y.shape[3];
    int len = C_out * H_out * W_out;
    
    //iterate over batch, in each image of the batch, it is converted into a matrix
    //then we do a matrix product with transformed W
    //W is transformed through flatten: [C_out, C_in, 3, 3] -> [C_out, C_in * 3 * 3]
    for(int i = 0; i < batch_size; i++){

        //transform X to X_hat: [C_in, H_out, W_out] -> [C_in * 3 * 3, H_out * W_out]
        Tensor X_hat(std::vector<int>{C_in*3*3, H_out*W_out}, "GPU");

        //calculate the number of kernels we are going to launch,
        //each kernel is responsible for copying a single-channel kernel_w*kernel_h pixels
        //and formulate them into a column, or precisely, 1/3 column
        int num_kernels = C_in * height * width;
    
        im2col_gpu_kernel<<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
            num_kernels, X.data + i * C_in * height * width, 
            height, width, 3, 3, 1, 1,
            X_hat.data);
        cudaDeviceSynchronize();

        // matrix product with gemm
        
        gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, C_out, H_out*W_out, C_in*3*3, 
            1.0, W.data, X_hat.data, 0.0, Y.data + i * len);
    }
    cudaDeviceSynchronize();
}

//code adapted from caffe
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    float* data_col) {

    //each index stands for a single-channel pixel in the output matrix
    CUDA_KERNEL_LOOP(index, n){
        //transform the index to the corresponding pixel in the col matrix: [C_in*kernel_h*kernel_w, h_out*w_out]
    
        const int h_index = index / width; //we first calculate the height index
        const int h_col = h_index % height; //because there are multiple channels, we take % to get the height index(h_col) in the current h*w area
        const int w_col = index % width; //calculate the width index of the img
        
        const int c_im = index / (height * width); //calculate the channel index of the img
        
        const int c_col = c_im * kernel_h * kernel_w; //calculate the channel index of the output matrix
        
        const int h_offset = h_col - pad_h;
        const int w_offset = w_col - pad_w; 
        //calculate the offset of the height and width in the input matrix
        //it should be similar to h_col and w_col, but we need to consider the padding and relocate the pointer to the correct position
        //which is the left-top corner of the kernel

        //data_col_ptr points to the current pixel in the output matrix
        float* data_col_ptr = data_col;
        data_col_ptr += (c_col * height + h_col) * width + w_col;

        //data_im_ptr points to the current pixel in the input matrix
        const float* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        //iterate over the kernel, copy the pixel value to the output matrix
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                data_im_ptr[i * width + j] : 0;
                data_col_ptr += height * width; //becaues we are formulating a column, we need to move the pointer to the next pixel in the column
            }
        }
    }
}

// dY: [N, C_out, H_out, W_out], X: [N, C_in, H, W], W: [C_out, C_in, 3, 3]
// dW: [C_out, C_in, 3, 3], db: [C_out], dX: [N, C_in, H, W]
void conv_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX){

    int batch_size = X.shape[0];
    int C_in = X.shape[1];
    int height = X.shape[2];
    int width = X.shape[3];
    int C_out = dY.shape[1];

    // dW = dY * X_hat^T
    // dW: [C_out, C_in, 3, 3], dY: [N, C_out, H_out, W_out], X_hat: [C_in * 3 * 3, H_out * W_out]
    //transform X to X_hat: [C_in, H_out, W_out] -> [C_in * 3 * 3, H_out * W_out]
    Tensor X_hat(std::vector<int>{batch_size, C_in*3*3, height*width}, "GPU");

    int num_kernels = C_in * height * width;
    im2col_gpu_kernel<<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
            num_kernels, X.data, height, width, 3, 3, 1, 1,
            X_hat.data);
    cudaDeviceSynchronize();

    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, C_out, C_in*3*3, height*width,  
        1.0, dY.data, X_hat.data, 0.0, dW.data);

    // dX_hat = W^T * dY
    // dX_hat: [C_in * 3 * 3, H_out * W_out], W: [C_out, C_in, 3, 3], dY: [C_out, H_out, W_out]
    // here we iterate over batch, each image in the batch is used to calculate a gradient of the input image in X_hat
    Tensor dX_hat(std::vector<int>{batch_size, C_in*3*3, height*width}, "GPU");
    for (int i = 0; i < batch_size; i++){
        gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, C_in*3*3, height*width, C_out,  
            1.0, W.data, dY.data + i * C_out * height * width, 0.0, dX_hat.data + i * C_in * 3 * 3 * height * width);

        // dX_hat: [C_in * 3 * 3, H_out * W_out] -> dX: [C_in, H, W]
        col2im_gpu_kernel<<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
            num_kernels, dX_hat.data + i * C_in * 3 * 3 * height * width, height, width, C_in, 3, 3, 1, 1, dX.data + i * C_in * height * width);
        cudaDeviceSynchronize();
    }
    
}

//code adapted from caffe
__global__ void col2im_gpu_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    float* data_im) {
    //each index stands for a single-channel pixel in the input img
    CUDA_KERNEL_LOOP(index, n) {
        float val = 0;
        
        // map the index to the corresponding pixel in the input img
        // include padding!!!
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);

        // compute the start and end of X_hat
        // this part does not include padding
        const int w_col_start = (w_im < kernel_w) ? 0 : (w_im - kernel_w) + 1;
        const int w_col_end = min(w_im + 1, width);
        const int h_col_start =(h_im < kernel_h) ? 0 : (h_im - kernel_h) + 1;
        const int h_col_end = min(h_im + 1, height);
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col);
                int w_k = (w_im - w_col);
                // postition in the kernel
                int position_in_the_kernel = h_k * kernel_w + w_k;
                int data_col_index = c_im*kernel_h*kernel_w*height*width + (position_in_the_kernel * height + h_col) * width + w_col;
                val += data_col[data_col_index];
            }
        }

        data_im[index] = val;
    }
}



void gemm_gpu(cublasOperation_t trans_A, cublasOperation_t trans_B, const int m, const int n, const int k,  
const float alpha, const float *A, const float *B, const float beta, float *C)
{
    int lda = k, ldb = n, ldc = n;
    if (trans_A == CUBLAS_OP_T)
        lda = m;
    if (trans_B == CUBLAS_OP_T)
        ldb = k;

    // printf("m=%d, k=%d, n=%d\n", m, k, n);
    // printf("lda=%d, ldb=%d, ldc=%d\n", lda, ldb, ldc);

    // Create a handle for CUBLAS
    cublasHandle_t handle; 
    cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, trans_B, trans_A, n, m, k, &alpha, 
    B, ldb, A, lda, &beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);

    cudaDeviceSynchronize();
}

//maxpool
// X: [N, C, H, W], Y: [N, C, H//2, W//2], mask: [N, C, H//2, W//2]
void maxpool_forward(const Tensor& X, Tensor& Y, Tensor& mask){
    int batch_size = X.shape[0];
    int channels = X.shape[1];
    int in_h = X.shape[2];
    int in_w = X.shape[3];
    int out_h = Y.shape[2];
    int out_w = Y.shape[3];
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int pad_h = 0;
    int pad_w = 0;
    int len = batch_size * channels * out_h * out_w;

    max_pool_forward_kernel<<<CudaGetBlocks(len), kCudaThreadsNum>>>(
        len, X.data, channels, in_h, in_w, out_h, out_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, Y.data, mask.data);
    cudaDeviceSynchronize();
}

// each thread is responsible for a local window
__global__ void max_pool_forward_kernel(int nthreads, float* in_data,
    int channels, int in_h, int in_w, int out_h, int out_w, 
    int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
    float* out_data, float* out_mask) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / out_w / out_h / channels; //batch index
        int c = (index / out_w / out_h) % channels; //channel index
        int ph = (index / out_w) % out_h; //height index
        int pw = index % out_w; //width index
        // implement max pooling for each local window,
        // store the max value and mask to out_data[index] & out_mask[index]

        //here we doesn't count the padding
        //which means [hstart, hend) & [wstart, wend) is all in the area of the input img
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, in_h);
        int wend = min(wstart + kernel_w, in_w);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);

        float maxval = -FLT_MAX;
        int maxidx = -1;
        const float* in_index = in_data + (n * channels + c) * in_h * in_w;
        
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (in_index[h * in_w + w] > maxval) {
                    maxidx = h * in_w + w;
                    maxval = in_index[maxidx];
                }
            }
        }

        out_data[index] = maxval;
        out_mask[index] = maxidx;
        
        
    }
}

// dY: [N, C, H//2, W//2], mask: [N, C, H//2, W//2], dX: [N, C, H, W]
void maxpool_backward(const Tensor& dY, const Tensor& mask, Tensor& dX){
    int batch_size = dY.shape[0];
    int channels = dY.shape[1];
    int out_h = dX.shape[2];
    int out_w = dX.shape[3];
    int in_h = dY.shape[2];
    int in_w = dY.shape[3];
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int pad_h = 0;
    int pad_w = 0;
    int len = batch_size * channels * out_h * out_w;

    max_pool_backward_kernel<<<CudaGetBlocks(len), kCudaThreadsNum>>>(
        len, dY.data, mask.data, channels, in_h, in_w, out_h, out_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dX.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
}

__global__ void max_pool_backward_kernel(int nthreads, float* in_data,
    float* mask, int channels, int in_h, int in_w, int out_h, int out_w, 
    int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
    float* out_data) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / out_w / out_h / channels; //batch index
        int c = (index / out_w / out_h) % channels; //channel index
        int h = (index / out_w) % out_h; //height index
        int w = index % out_w; //width index


        int hstart = (h + pad_h < kernel_h) ? 0 : (h - kernel_h + pad_h) / stride_h + 1;
        int wstart = (w + pad_w < kernel_w) ? 0 : (w - kernel_w + pad_w) / stride_w + 1;

        // plus one is because there is a less than sign in the follow up for loop
        int hend = min((h + pad_h) / stride_h + 1, in_h);
        int wend = min((w + pad_w) / stride_w + 1, in_w);

        float gradient = 0;
        //index of the max value in the mask
        const float* mask_index = mask + (n * channels + c) * in_h * in_w;
        // index of the max value in the input gradient
        const float* in_index = in_data + (n * channels + c) * in_h * in_w;

        for (int ph = hstart; ph < hend; ++ph) {
            for (int pw = wstart; pw < wend; ++pw) {
                gradient += (mask_index[ph * in_w + pw] == h * out_w + w) ? in_index[ph * in_w + pw] : 0;
            }
        }
        out_data[index] = gradient;
    }
}

//softmax
// X: [N, C], Y: [N, C]
void softmax_forward(const Tensor& X, Tensor& Y){
    int batch_size = X.shape[0];
    int num_classes = X.shape[1];
    int len = batch_size * num_classes;

    // printf("I'm here\n\n");
    //take max for each batch
    Tensor max_val(std::vector<int>{batch_size}, "GPU");
    max_kernel<<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(X.data, batch_size, num_classes, max_val.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // printf("max_val\n");
    // max_val.print();

    // printf("I'm here 0\n\n");

    Tensor unnormalized(std::vector<int>{batch_size, num_classes}, "GPU");
    exp_kernel<<<CudaGetBlocks(len), kCudaThreadsNum>>>(X.data, len, num_classes, max_val.data, unnormalized.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // printf("unnormalized\n");
    // unnormalized.print();

    // printf("I'm here 1\n\n");

    Tensor sum_val(std::vector<int>{batch_size}, "GPU");
    sum_kernel<<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(unnormalized.data, batch_size, num_classes, sum_val.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // printf("I'm here 2\n\n");


    div_kernel<<<CudaGetBlocks(len), kCudaThreadsNum>>>(unnormalized.data, len, num_classes, sum_val.data, Y.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // printf("I'm here 3\n\n");
}

__global__ void max_kernel(const float* in_data, int batch_size, int num_classes, float* max_val){

    CUDA_KERNEL_LOOP(index, batch_size){
        float max = -FLT_MAX;
        for (int i = 0; i < num_classes; i++){
            if(in_data[index*num_classes+i] > max){
                max = in_data[index*num_classes+i];
            }
        }
        max_val[index] = max;
    }
}

// for N*C elements, each substracts the max value in the batch and takes the exponential
__global__ void exp_kernel(const float* in_data, int len, int num_classes, const float* max_val, float* unnormalized){
    CUDA_KERNEL_LOOP(index, len){
        int batch = index / num_classes;
        unnormalized[index] = exp(in_data[index] - max_val[batch]);
    }
}

__global__ void sum_kernel(const float* in_data, int batch_size, int num_classes, float* sum_val){
    CUDA_KERNEL_LOOP(index, batch_size){
        sum_val[index] = 0;
        for (int i = 0; i < num_classes; i++){
            sum_val[index] += in_data[index*num_classes+i];
        }
    }
}

__global__ void div_kernel(const float* in_data, int len, int num_classes, const float* sum_val, float* y){
    CUDA_KERNEL_LOOP(index, len){
        int batch = index / num_classes;
        y[index] = in_data[index] / sum_val[batch];
    }
}



//cross entropy
// Y: [N, C], target: [N, C], loss: [N]
void cross_entropy_forward(const Tensor& X, const Tensor& target, Tensor& loss){
    int batch_size = X.shape[0];
    int num_classes = X.shape[1];

    // printf("didn't take mean\n\n");

    Tensor probablity(std::vector<int>{batch_size, num_classes}, "GPU");
    softmax_forward(X, probablity);
    // printf("I'm here\n\n");
    Tensor single_CE(std::vector<int>{batch_size}, "GPU");
    cross_entropy_forward_kernel<<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(batch_size, num_classes, probablity.data, target.data, single_CE.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // printf("I'm here 1\n\n");

    cross_entropy_forward_sum_kernel<<<CudaGetBlocks(1), kCudaThreadsNum>>>(1, batch_size, single_CE.data, loss.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    // printf("I'm here 2\n\n");
}

__global__ void cross_entropy_forward_kernel(int nthreads, int num_classes, const float* in_data, const float* target, float* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        out_data[index] = 0;
        for (int i = 0; i < num_classes; i++){
            out_data[index] += -log(in_data[index*num_classes+i])*target[index*num_classes+i];
        }
    }
}

__global__ void cross_entropy_forward_sum_kernel(int nthreads, int batch_size, const float* in_data, float* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        for (int i = 0; i < batch_size; i++){
            out_data[0] += in_data[i];
        }
    }
}

//cross entropy with softmax backward
// L: [1], dX: [N, C], label: [N, C] one-hot
void cross_entropy_with_softmax_backward(const Tensor& L, const Tensor& X, const Tensor& label, Tensor& dX){
    int batch_size = X.shape[0];
    int num_classes = X.shape[1];
    int len = batch_size*num_classes;

    Tensor probablity(std::vector<int>{batch_size, num_classes}, "GPU");
    softmax_forward(X, probablity);

    minus_kernel<<<CudaGetBlocks(len), kCudaThreadsNum>>>(probablity.data, len, label.data, dX.data);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
}

__global__ void minus_kernel(const float* in_data, int len, const float* target, float* out_data){
    CUDA_KERNEL_LOOP(index, len){
        out_data[index] = in_data[index] - target[index];
    }
}