#include <stdio.h>
#include <stdlib.h>

#define A(i,j) A[i + lda*j]
#define B(i,j) B[i + ldb*j]
#define C(i,j) C[i + ldc*j]


// 由于这里的sa和sb仅仅只是在线程块内使用，不需要暴露给外部接口API，所以这里我们无需再做和CUBLAS库API的统一处理，所以采用行优先顺序
#define sa(i,j) sa[(i * KS + j)]
#define sb(i,j) sb[(i * NS + j)]

#define MS 32
#define NS 32
#define KS 32

// Naive Loop Tiling Optimization. 这里虽然做了 loop tiling优化，但内层for循环的存在引出了 synchronize()，反而导致kernel2的性能低于kernel1,但后续的优化是基于kernel2的，这便是其存在的意义
__global__ __launch_bounds__(1024)
void mysgemm_v2(int M, int N, int K,float alpha, float *A,float *B,float beta, float *C) {
    int lda = M, ldb = N, ldc = K;  // leading axis
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // 整个项目中都预先限定了 TB 的大小为 32 * 32
    A = &A((bx<<5),0);   // 尤其注意宏参数中如果这种复杂的表达式，应该要用括号括起来
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));

    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];
    float tmp = 0.;
    for(int k_count = 0; k_count < K; k_count += KS) {
        sa(tx,ty) = A(tx,ty);
        sb(ty,tx) = B(tx,ty);   // caching block using shared memory
        // sb(tx,ty) = B(tx,ty);    // 仔细想想，如果以tx作为 sb 的行轴，底下的内层for循环中对于sb来说,inner_k_count就无法作为其列轴进行访问，没有利用局部性原理
        A += lda * NS;  // 跳过A已经处理了的列(仔细思考，A依旧是按列优先顺序存放的)
        B += 32;   //  跳过A已经处理了的行(32行，因为这里设置的一个块就是32*32)
        __syncthreads();   // 同步当前TB中的所有 threads，确保 sa、sb中的数据是可用的
        for(int inner_k_count = 0; inner_k_count < KS; inner_k_count++) {
            tmp += sa(tx,inner_k_count) * sb(ty,inner_k_count);
        }
        __syncthreads();  // 确保 tmp 已经更新完成
    }
    C(tx,ty) = alpha * tmp + beta * C(tx,ty);
}




