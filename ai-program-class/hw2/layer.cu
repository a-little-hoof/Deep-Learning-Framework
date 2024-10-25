#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <tensor.h>
#include <layer.h>

// fc layer forward and backward
// X: [N, C_in], Y: [N, C_out], W: [C_in, C_out], b: [C_out]
// backward function takes partial L / partial y and outputs parital L / partial W or partial L / partial X

void fc_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y)
{
    int batch_size = X.shape[0];
    int in_features = X.shape[1];
    int out_features = W.shape[1];

    // matrix product with gemm
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, in_features, out_features, 
        1.0, X.data, W.data, 0.0, Y.data);

    // add bias
    Tensor ones_(std::vector<int>{batch_size, 1}, "GPU");
    ones_.fill_(1.0);
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, 1, out_features,
        1.0, ones_.data, b.data, 1.0, Y.data);
}

// dY: [N, C_out], X: [N, C_in], W: [C_in, C_out]
// dW: [C_in, C_out], db: [C_out], dX: [N, C_in]
void fc_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX){

    int batch_size = X.shape[0];
    int in_features = X.shape[1];
    int out_features = W.shape[1];

    // dW = X^T * dY
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, in_features, batch_size, out_features,  
        1.0, X.data, dY.data, 0.0, dW.data);

    // dX = dY * W^T
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, batch_size, out_features, in_features,  
        1.0, dY.data, W.data, 0.0, dX.data);
    

}

//conv_layer forward and backward
// X: [N, C_in, H, W], Y: [N, C_out, H_out, W_out], W: [C_out, C_in, K, K], b: [C_out]
// backward function takes partial L / partial y and outputs parital L / partial W or partial L / partial X
void conv_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y){

}
void conv_backward(const Tensor& dY, const Tensor& X, const Tensor& W, Tensor& dW, Tensor& dX){

}




// m=batch_size, k=in_features, n=out_features, C行C列内积维度
// C(m,n) = \alpha A(m,k) * B(k,n) + \beta C(m,n)
void gemm_gpu(cublasOperation_t trans_A, cublasOperation_t trans_B, const int m, const int k, const int n, 
const float alpha, const float *A, const float *B, const float beta, float *C)
{
    int lda = m, ldb = k, ldc = m;
    if (trans_A == CUBLAS_OP_T)
        lda = k;
    if (trans_B == CUBLAS_OP_T)
        ldb = n;

    // Create a handle for CUBLAS
    cublasHandle_t handle; 
    cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, trans_A, trans_B, m, n, k, &alpha, 
    A, lda, B, ldb, &beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);

    cudaDeviceSynchronize();
}