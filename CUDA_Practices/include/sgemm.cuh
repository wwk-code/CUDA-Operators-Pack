#include "common.cuh"

// CPU矩阵乘函数
void matrix_multiply(int M, int N, int K, DTYPE *A, DTYPE *B, DTYPE *C, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            DTYPE sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // sum += A[i * K + k] * B[k * N + j];
                sum += A[i  + k * M] * B[k + j * N];
            }
            C[i + M * j] = alpha * sum + beta * C[i + M * j];
        }
    }
}


// naive kernel
__global__ __launch_bounds__(1024)
void sgemm_kernel_1(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;   // 纵向
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;   // 横向
    if(x < M && y < N) {
        float temp = 0.0f;
        for(int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}


void test_sgemm_kernel_1(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    cudaDeviceSynchronize();
    dim3 blockDim(32,32,1);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(K,32),1);
    sgemm_kernel_1<<<gridDim,blockDim>>>(M,N,K,A,B,C,alpha,beta);
    cudaDeviceSynchronize();
}



// coalescing GMem access
// __global__ __launch_bounds__(1024)
__global__ 
void sgemm_kernel_2(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta, int BLOCK_SIZE) {
    const uint x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    if(x < M && y < N) {
        float temp = 0.0f;
        for(int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}


void test_sgemm_kernel_2(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    cudaDeviceSynchronize();
    int BLOCK_SIZE = 32 * 32;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    
    sgemm_kernel_2<<<gridDim,blockDim>>>(M,N,K,A,B,C,alpha,beta,BLOCK_SIZE);
    cudaDeviceSynchronize();
}


void test_cublas_sgemm(cublasHandle_t handle, INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    cudaDeviceSynchronize();
}


void test_kernel(int kernel_num,int M,int K,int N,DTYPE alpha,DTYPE *A,DTYPE *B,DTYPE beta,DTYPE *C, cublasHandle_t handle){
    switch (kernel_num){    
        case 0: matrix_multiply(M,N,K,A,B,C,alpha,beta); break;
        case 1: test_sgemm_kernel_1(M,N,K,A,B,C,alpha,beta); break;
        case 2: test_sgemm_kernel_2(M,N,K,A,B,C,alpha,beta); break;


        default: break;
    }
}

