#include <stdio.h>
// #include <cuda.h>

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

// CPU ver
float relu_cpu(float x){
    return x>0 ? x:0;
}

// GPU ver
__global__ void relu(float* in, float* out){
    int i = threadIdx.x;
    out[i] = in[i]>0?in[i]:0;
    // printf("Hello, CUDA!\n");
}

__global__ void helloCUDA(){
    printf("Hello, CUDA!\n");
}

int main(){
    const int N=64;
    const int size = N*sizeof(float);

    //allocate memory on CPU
    float* h_in = (float*) malloc(size);
    float* h_out = (float*) malloc(size);

    
    //initialize input array
    for(int i=0; i<N; ++i){
        h_in[i] = (i-32)*0.1;
    }

    //relu on CPU
    for(int i=0; i<N; ++i){
        h_out[i] = relu_cpu(h_in[i]);
    }

    //print results
    print_array(h_in, N, "in");
    print_array(h_out, N, "out-cpu");
    memset(h_out, 0, size); //reset output array

    //1.allocate memory on GPU
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    //要分配内存！！

    //2.copy data from CPU to GPU
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    //3.launch the kernel
    relu<<<1, N>>>(d_in, d_out);

    //4.copy data from GPU to CPU
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    print_array(h_out, N, "out-gpu");

    //free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;

}
