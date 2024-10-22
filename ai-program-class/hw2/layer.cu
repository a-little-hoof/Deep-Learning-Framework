#include <cublas_v2.h>

// C(m,n) = A(m,k) * B(k,n)
void gemm_gpu(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
    int lda = m, ldb = k, ldc = m;
    const float alf = 1, bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle; cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, 
    A, lda, B, ldb, beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}