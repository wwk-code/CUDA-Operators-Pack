#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa(i,j) sa[((j)<<5) + (i)]   // 相较于 kernel3 唯一的变化，之类增强了并行性访问,看最内层for循环处
#define sb(i,j) sb[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// save one living register ty.
__global__  __launch_bounds__(1024)
void mysgemm_v4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa(row,col)=A(row,col);
        sb(col,row)=B(row,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            // 尽管同一线程的不同迭代之间的访问丧失了局部性，但线程并行性访问带来的性能提升
            tmp += sa(row,inner_k_count) * sb(col,inner_k_count);   
        }
        __syncthreads();
    }
    C(row,col) = alpha * tmp + beta*C(row,col);
}