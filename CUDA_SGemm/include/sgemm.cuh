#include "common.cuh"

// CPU矩阵乘函数(列优先版)
void matrix_multiply(int M, int N, int K, DTYPE *A, DTYPE *B, DTYPE *C, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            DTYPE sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}


// naive kernel
__global__ __launch_bounds__(1024)
void sgemm_kernel_naive(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;   // 纵向
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;   // 横向
    if(x < M && y < N) {
        float temp = 0.0;
        for(int i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}


void test_sgemm_kernel_naive(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    cudaDeviceSynchronize();
    dim3 blockDim(32,32,1);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(K,32),1);
    sgemm_kernel_naive<<<gridDim,blockDim>>>(M,N,K,A,B,C,alpha,beta);
    cudaDeviceSynchronize();
}



#define BM 64    
#define BN 64    
#define BK 8     
#define TM 8     
#define TN 8     
__global__ void sgemm_optimized(int M, int N, int K, float* A, float *B, float *C, float alpha, float beta) {

    __shared__ float As[BM][BK];  
    __shared__ float Bs[BK][BN];  
    
    float c[TM][TN] = {{0.0f}};
    
    // 块索引计算
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    // 线程索引计算（每个线程处理8x8子矩阵）
    int thread_row = threadIdx.y * TM;  
    int thread_col = threadIdx.x * TN;  
    
    // 动态指针偏移
    A += block_row * K;  
    B += block_col;      
    C += block_row * N + block_col;

    // 主循环处理K维度（分阶段加载数据）
    for(int t = 0; t < K; t += BK) {
        #pragma unroll
        for(int i=0; i<BM; i+=blockDim.y) { // 一个线程负责 BM / TM 个 C元素复制
            if(block_row+i < M && t+threadIdx.x < K) 
                As[i+threadIdx.y][threadIdx.x] = A[(i+threadIdx.y)*K + t + threadIdx.x];
        }
        #pragma unroll
        for(int j=0; j<BN; j+=blockDim.x) {
            if(block_col+j < N && t+threadIdx.y < K)
                Bs[threadIdx.y][j+threadIdx.x] = B[(t+threadIdx.y)*N + j+threadIdx.x];
        }
        __syncthreads();

        // 核心计算部分（循环展开优化）
        #pragma unroll
        for(int k=0; k<BK; ++k) {

            float a_reg[TM], b_reg[TN];
            
            // 寄存器预加载（减少Shared Memory访问次数）
            #pragma unroll
            for(int i=0; i<TM; ++i) 
                a_reg[i] = As[thread_row+i][k];
                
            #pragma unroll
            for(int j=0; j<TN; ++j) 
                b_reg[j] = Bs[k][thread_col+j];
            
            // 外积计算（8x8子矩阵乘加）
            #pragma unroll
            for(int i=0; i<TM; ++i) {
                #pragma unroll
                for(int j=0; j<TN; ++j) {
                    c[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // 结果写回全局内存（边界处理）
    #pragma unroll
    for(int i=0; i<TM; ++i) {
        #pragma unroll
        for(int j=0; j<TN; ++j) {
            int row = thread_row + i;
            int col = thread_col + j;
            if(row < M && col < N) {
                C[row*N + col] = alpha * c[i][j] + beta * C[row*N + col];
            }
        }
    }
}


void launch_sgemm_shared(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    dim3 block(BN/TN , BM/TM);  // (64/8)x(64/8)=8x8=64线程/block
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_optimized<<<grid, block>>>(M, N, K, A, B, C, alpha, beta);
}


void test_kernel(int kernel_num,int M,int K,int N,DTYPE alpha,DTYPE *A,DTYPE *B,DTYPE beta,DTYPE *C, cublasHandle_t handle){

    switch (kernel_num){    
        
        case 0: matrix_multiply(M,N,K,A,B,C,alpha,beta); break;
        case 1: test_sgemm_kernel_naive(M,N,K,A,B,C,alpha,beta); break;
        case 2: launch_sgemm_shared(M,N,K,A,B,C,alpha,beta); break;

        default: break;
    }

}

