#include "sgemm.cuh"


#define CUBLAS_CHECK(err) \
    if((err) != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

inline void cublas_sgemm(
    cublasHandle_t handle,
    int m, int n, int k,
    const FLOAT* A,  // device ptr, size m×k
    const FLOAT* B,  // device ptr, size k×n
    FLOAT*       C,  // device ptr, size m×n
    FLOAT        alpha,
    FLOAT        beta
) {
    // cuBLAS 默认按 column-major 存储，
    // 这里我们直接假设输入是 column-major。
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            A, m,    // leading dimension = m
            B, k,    // leading dimension = k
            &beta,
            C, m     // leading dimension = m
        )
    );
}




int main(int argc, char **argv) {

    if(argc != 3) {
        printf("Please select a kernel (range 0 - 11, here 0 is for NVIDIA cuBLAS).\n");
        exit(-1);
    }
    
    int edgeLen = 16;

    int kernel_seq = atoi(argv[1]);

    edgeLen = atoi(argv[2]);

    if(kernel_seq < 0 || kernel_seq > 11) {
        printf("Please enter a valid kernel number (0-11).\n");
        exit(-2);
    }
    
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

    if(kernel_seq == 0) {   // custom-cublas-kernel
        cudaEventRecord(begin);
        for (n_count=0;n_count<N;n_count++){
            cublas_sgemm(cublas_handle, m, n, k, dA, dB, dC, alpha, beta);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time,begin,end);
        elapsed_time /= 1000.;
        printf("Average elasped time: %.2f ms, performance: %f GFLOPS.\n", 1000 * elapsed_time/N,2.*1e-9*N*m*n*k/elapsed_time);
        fflush(stdout);
        copy_matrix(C_ref,C,m*n); //sync C with cuBLAS to prepare for the next run，avoid of diff cumulation error
        cudaDeviceSynchronize();
    }
    else {   // custom-cuda-kernel
        cudaEventRecord(begin);
        for (n_count=0;n_count<N;n_count++){
            test_kernel(kernel_seq,m,n,k,alpha,dA,dB,beta,dC,cublas_handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time,begin,end);
        elapsed_time /= 1000.;
        printf("Average elasped time: %.2f ms, performance: %f GFLOPS.\n", 1000 * elapsed_time/N,2.*1e-9*N*m*n*k/elapsed_time);
        fflush(stdout);
        copy_matrix(C_ref,C,m*n); //sync C with cuBLAS to prepare for the next run，avoid of diff cumulation error
        cudaDeviceSynchronize();

    }

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


