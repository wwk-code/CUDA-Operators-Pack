#include "sgemm.cuh"


int main(int argc, char **argv) {

    if(argc != 2) {
        printf("Please select a kernel (range 0 - 11, here 0 is for NVIDIA cuBLAS).\n");
        exit(-1);
    }
    int SIZE[24];
    for(int i = 0; i<24;i++) SIZE[i] = (i+1) << 8;  // 256 - xxx
    int kernel_seq = atoi(argv[1]);

    if(kernel_seq < 0 || kernel_seq > 11) {
        printf("Please enter a valid kernel number (0-11).\n");
        exit(-2);
    }
    int edgeLen = 32;
    int m = edgeLen,n = edgeLen,k = edgeLen;
    int n_count,N=10,upper_limit = 4;
    DTYPE *A,*B,*C,*C_ref;
    DTYPE *dA,*dB,*dC,*dC_ref;
    FLOAT alpha = 1.0, beta = 0.;  // two arbitary input parameters
    float elapsed_time = 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle); 
    cudaEvent_t begin,end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    A = (DTYPE*)malloc(sizeof(DTYPE) * m * k);
    B = (DTYPE*)malloc(sizeof(DTYPE) * k * n);
    C = (DTYPE*)malloc(sizeof(DTYPE) * m * n);
    C_ref = (DTYPE*)malloc(sizeof(DTYPE) * m * n);
    // matrix_init(A,m * n);
    // matrix_init(B,m * n);
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);

    matrix_init(C,m * n);
    matrix_init(C_ref,m * n);

    
    CUDA_CALLER(cudaMalloc((void**)&dA,sizeof(DTYPE) * m * k));
    CUDA_CALLER(cudaMalloc((void**)&dB,sizeof(DTYPE) * k * n));
    CUDA_CALLER(cudaMalloc((void**)&dC,sizeof(DTYPE) * m * n));
    CUDA_CALLER(cudaMemcpy(dA,A,sizeof(DTYPE) * m * k,cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dB,B,sizeof(DTYPE) * k * n,cudaMemcpyHostToDevice));
    
    for(int i_count = 0; i_count < upper_limit; i_count++) {
        // m=n=k=SIZE[i_count];
        printf("\nM=N=K=%d:\n",m);
        if(kernel_seq != 0) {
            test_kernel(0,m,n,k,alpha,A,B,beta,C_ref,cublas_handle);  // CPU MM
            test_kernel(kernel_seq,m,n,k,alpha,dA,dB,beta,dC,cublas_handle);
            CUDA_CALLER(cudaDeviceSynchronize());
            CUDA_CALLER(cudaMemcpy(C,dC,sizeof(DTYPE) * m * k,cudaMemcpyDeviceToHost));
            CUDA_CALLER(cudaDeviceSynchronize());
            if(!verify_matrix(C,C_ref,m*k)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(-3);
            }
        }
        int temp = 1.0;
        // cudaEventRecord(begin);
        // if (kernel_seq != 0){
        //     for (n_count=0;n_count<N;n_count++){
        //         test_kernel(kernel_num,m,n,k,alpha,dA,dB,beta,dC);
        //     }
        // }else{
        //     for (n_count=0;n_count<N;n_count++){
        //         test_kernel(kernel_seq,m,n,k,alpha,dA,dB,beta,dC, handle);
        //     }
        // }
        // cudaEventRecord(end);
        // cudaEventSynchronize(begin);
        // cudaEventSynchronize(end);
        // cudaEventElapsedTime(&elapsed_time,begin,end);
        // elapsed_time /= 1000.;
        // printf("Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*1e-9*N*m*n*k/elapsed_time);
        // fflush(stdout);
        // copy_matrix(C_ref,C,m*n); //sync C with cuBLAS to prepare for the next run，avoid of diff cumulation error
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


