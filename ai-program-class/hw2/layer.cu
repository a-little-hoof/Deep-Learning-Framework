#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <tensor.h>
#include <layer.h>
#include <utils.h>

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
            num_kernels, X.data, 
            height, width, 3, 3, 1, 1,
            X_hat.data);

        // matrix product with gemm
        
        gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, C_out, H_out*W_out, C_in*3*3, 
            1.0, W.data, X_hat.data, 0.0, Y.data + i * len);
    }
    cudaDeviceSynchronize();
}

//function adapted from caffe
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

void conv_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX){

}

// __global__ void col2im_gpu_kernel(const int n, const float* data_col,
//     const int height, const int width, const int channels,
//     const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w,
//     const int stride_h, const int stride_w,
//     const int dilation_h, const int dilation_w,
//     const int height_col, const int width_col,
//     float* data_im) {
//   CUDA_KERNEL_LOOP(index, n) {
//     float val = 0;
//     const int w_im = index % width + pad_w;
//     const int h_im = (index / width) % height + pad_h;
//     const int c_im = index / (width * height);
//     int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
//     int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
//     // compute the start and end of the output
//     const int w_col_start =
//         (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
//     const int w_col_end = min(w_im / stride_w + 1, width_col);
//     const int h_col_start =
//         (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
//     const int h_col_end = min(h_im / stride_h + 1, height_col);
//     // TODO: use LCM of stride and dilation to avoid unnecessary loops
//     for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
//       for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
//         int h_k = (h_im - h_col * stride_h);
//         int w_k = (w_im - w_col * stride_w);
//         if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
//           h_k /= dilation_h;
//           w_k /= dilation_w;
//           int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
//                                 height_col + h_col) * width_col + w_col;
//           val += data_col[data_col_index];
//         }
//       }
//     }
//     data_im[index] = val;
//   }
// }

// void col2im_gpu(const float* data_col, const int channels,
//     const int height, const int width, const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w, const int stride_h,
//     const int stride_w, const int dilation_h, const int dilation_w,
//     float* data_im) {
//   int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
//       stride_h + 1;
//   int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
//       stride_w + 1;
//   int num_kernels = channels * height * width;
//   // To avoid involving atomic operations, we will launch one kernel per
//   // bottom dimension, and then in the kernel add up the top dimensions.
//   // NOLINT_NEXT_LINE(whitespace/operators)
//   col2im_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels),
//                              CAFFE_CUDA_NUM_THREADS>>>(
//       num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
//       pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
//       height_col, width_col, data_im);
//   CUDA_POST_KERNEL_CHECK;
// }



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