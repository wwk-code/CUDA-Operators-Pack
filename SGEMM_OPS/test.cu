#include "helper_string.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include "utils.cuh"


#define MYSGEMM mysgemm_naive  // slect the kernel here


int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Please select a kernel (range 0 - 11, here 0 is for NVIDIA cuBLAS).\n");
        exit(-1);
    }
    int SIZE[24];
    for(int i = 0; i<24;i++) SIZE[i] = (i+1) << 8;  // 256 - xxx
    int kernel_num = atoi(argv[1]);
    if(kernel_num < 0 || kernel_num > 11) {
        printf("Please enter a valid kernel number (0-11).\n");
        exit(-2);
    }
    int m,n,k,max_size;
    int n_count,N=10,upper_limit = 8;
    if(kernel_num<=4 && kernel_num!=0) upper_limit = 8;
    else upper_limit = (sizeof(SIZE)/sizeof(int));
    max_size = SIZE[upper_limit-1];   // 方阵中 行/列的最大元素数量
    FLOAT *A = NULL,*B = NULL, *C = NULL, *C_ref = NULL;
    FLOAT *dA = NULL,*dB = NULL, *dC = NULL, *dC_ref = NULL;
    FLOAT alpha = 1.0, beta = 0.;  // two arbitary input parameters
    float elapsed_time;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t begin,end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    A = (FLOAT*)malloc(sizeof(FLOAT) * max_size * max_size);
    B = (FLOAT*)malloc(sizeof(FLOAT) * max_size * max_size);
    C = (FLOAT*)malloc(sizeof(FLOAT) * max_size * max_size);
    C_ref = (FLOAT*)malloc(sizeof(FLOAT) * max_size * max_size);
    randomize_matrix(A,max_size*max_size);
    randomize_matrix(B,max_size*max_size);
    randomize_matrix(C,max_size*max_size);
    copy_matrix(C,C_ref,max_size*max_size);
    CUDA_CALLER(cudaMalloc((void**) &dA,sizeof(FLOAT)*max_size*max_size));
    CUDA_CALLER(cudaMalloc((void**) &dB,sizeof(FLOAT)*max_size*max_size));
    CUDA_CALLER(cudaMalloc((void**) &dC,sizeof(FLOAT)*max_size*max_size));
    CUDA_CALLER(cudaMalloc((void**) &dC_ref, sizeof(FLOAT)*max_size*max_size));
    CUDA_CALLER(cudaMemcpy(dA,A,sizeof(FLOAT)*max_size*max_size,cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dB,B,sizeof(FLOAT)*max_size*max_size,cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC,C,sizeof(FLOAT)*max_size*max_size,cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC_ref,C_ref,sizeof(FLOAT)*max_size*max_size,cudaMemcpyHostToDevice));

    for(int i_count = 0; i_count < upper_limit; i_count++) {
        m=n=k=SIZE[i_count];
        printf("\nM=N=K=%d:\n",m);
        if(kernel_num != 0) {
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,dA,m,dB,k,&beta,dC_ref,m);
            test_kernel(kernel_num,m,n,k,alpha,dA,dB,beta,dC);
            cudaDeviceSynchronize();
            cudaMemcpy(C,dC,sizeof(FLOAT)*n*n,cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref,dC_ref,sizeof(FLOAT)*n*n,cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            if(!verify_matrix(C_ref,C,m*n)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(-3);
            }
        }

        cudaEventRecord(begin);
        if (kernel_num != 0){
            for (n_count=0;n_count<N;n_count++){
                test_kernel(kernel_num,m,n,k,alpha,dA,dB,beta,dC);
            }
        }else{
            for (n_count=0;n_count<N;n_count++){
                test_kernel(kernel_num,m,n,k,alpha,dA,dB,beta,dC, handle);
            }
        }
        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time,begin,end);
        elapsed_time /= 1000.;
        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*1e-9*N*m*n*k/elapsed_time);
        fflush(stdout);
        copy_matrix(C_ref,C,m*n); //sync C with cuBLAS to prepare for the next run，avoid of diff cumulation error
    }

    cudaDeviceSynchronize();
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cudaDeviceSynchronize();

    return 0;

}





