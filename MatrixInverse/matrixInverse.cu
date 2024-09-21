#include <iostream>
#include <cuda_runtime.h>



// 使用高斯消元法求矩阵的逆
__global__ void matReverse_kernel(float *A,float *I, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    for(int i = 0;i < N; i++) {
        I[idx * N + i] = idx == i ? 1.0f : 0.0f;
    }
    __syncthreads();

    for(int i = 0;i < N; i++) {
        float diagValue = A[i * N + i];

        if(idx == i) {
            for(int j = 0;j < N; j++) {
                A[idx * N + j] /= diagValue;
                I[idx * N + j] /= diagValue;
            }
        }
        __syncthreads();
        if(idx != i) { 
            float rowValue = A[idx * N + i];
            for(int j = 0;j < N; j++) {
                // 下面这个其实就是行相减(注意:此时A的对角线元已经变为了 1 )
                A[idx * N + j] -= A[i * N + j] * rowValue;
                I[idx * N + j] -= I[i * N + j] * rowValue;
            }
        }
        __syncthreads();
    }
}



void matrixInverse(float *A, float *I, int N) {
    float *d_A, *d_I;
    size_t size = N * N * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_I, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 blockSize = 256;
    dim3 gridSize = (N + blockSize.x - 1) / blockSize.x;
    matReverse_kernel<<<gridSize,blockSize>>>(d_A,d_I,N);

    cudaMemcpy(I, d_I, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_I);
}

int main() {
    const int N = 3;
    float A[N * N] = {1, 3, 5,-1, 2, 3,4, 0, 6};
    float I[N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    matrixInverse(A, I, N);

    std::cout << "Inverse matrix:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << I[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

