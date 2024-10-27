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
    float *A = new float[2];
    float *B = new float[6];
    //create two lists
    for (int i=0; i<2; i++){
        A[i] = i;
        printf("%f\n", A[i]);
    }
    for (int i=2; i<8; i++){
        B[i] = i;
        printf("%f\n", B[i]);
    }

    float *C = new float[20];
    for (int i=0; i<1; i++){
        for (int j=0; j<3; j++){
            printf("%f ", C[i*3+j]);
        }
        printf("\n");
    }

    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_T, 1, 2, 3, 1.0, A, B, 0.0, C);
    for (int i=0; i<1; i++){
        for (int j=0; j<3; j++){
            printf("%f ", C[i*3+j]);
        }
        printf("\n");
    }
    for (int i=0; i<20; i++){
        printf("%f ", C[i]);
    }


    return 0;
}