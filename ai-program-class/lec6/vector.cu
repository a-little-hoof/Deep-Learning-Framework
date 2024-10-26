#include <cublas_v2.h>
#include <vector>

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

int main(){
    std::vector<int> s1 = {1,2,3,4,5,6};
    std::vector<int> s2 = {7,8,9,10,11,12};
    int *C = new int[20];
    // Create a handle for CUBLAS
    cublasHandle_t handle; 
    cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, trans_A, trans_B, m, n, k, &alpha, 
    A, lda, B, ldb, &beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);

    cudaDeviceSynchronize();

    return 0;
}