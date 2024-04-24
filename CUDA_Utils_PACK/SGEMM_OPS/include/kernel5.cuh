#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa4(i,j) sa4[((j)<<5) + (i)]   
#define sb4(i,j) sb4[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// save one living register ty.
// increse the workloads of per thread,and decrese the times of kernel boost
__global__  __launch_bounds__(256)
void mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    //int row = tx&31, col = tx>>5;
    // (tx & 7) << 2 ,  * 4 的目的是一共有4个rowx,所以每一个tx的不同都会导致新产生4个row，所以需要 * 4
    int row1 = (tx & 7) << 2, row2 = row1+1, row3 = row2+1, row4 = row3+1, col = tx >> 3;

    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));

    __shared__ float sa4[MS*KS];
    __shared__ float sb4[KS*NS];

    float Cres[4] = {0.,0.,0.,0.};    
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa4(row1,col) = A(row1,col);
        sa4(row2,col) = A(row2,col);
        sa4(row3,col) = A(row3,col);
        sa4(row4,col) = A(row4,col);
        sb4(col,row1) = B(row1,col);
        sb4(col,row2) = B(row2,col);
        sb4(col,row3) = B(row3,col);
        sb4(col,row4) = B(row4,col);

        A+=(lda<<5); B+=32;

        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            b00 = sb4(col,inner_k_count);
            Cres[0] += sa4(row1,inner_k_count) * b00;
            Cres[1] += sa4(row2,inner_k_count) * b00;
            Cres[2] += sa4(row3,inner_k_count) * b00;
            Cres[3] += sa4(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    C(row1,col) = alpha * Cres[0] + beta*C(row1,col);
    C(row2,col) = alpha * Cres[1] + beta*C(row2,col);
    C(row3,col) = alpha * Cres[2] + beta*C(row3,col);
    C(row4,col) = alpha * Cres[3] + beta*C(row4,col);
}