#include <stdio.h>
#include <stdlib.h>

// CUBLAS库默认以列优先顺序处理一维数组，所以我们在这里为了API接口的统一，也采用和CUBLAS库一样的访问顺序设计
#define A(i,j) A[i + j*lda]
#define B(i,j) B[i + j*ldb]
#define C(i,j) C[i + j*ldc]

// naive version
__global__ __launch_bounds__(1024)  //__launch_bounds__(1024)在提示编译器，线程块最大为1024
void mysgemm_v1(int M,int N,int K,float alpha,float *A,float *B, float beta,float *C) {
    int lda = M, ldb = N, ldc = M;   // 其实限定了　
    int tx = threadIdx.x, ty = threadIdx.y; 
    int bx = blockIdx.x, by = blockIdx.y;

    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));

    float temp = 0.;   
    for(int k_count = 0; k_count < K; k_count++) {
        temp += A(tx,k_count) * B(k_count,ty);
    }
    C(tx,ty) = alpha * temp + beta*C(tx,ty);   // 就是 SGEMM 的公式
}


