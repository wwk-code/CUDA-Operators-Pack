#include <iostream>
#include <cuda_runtime.h>
#include <utility>

#define CEIL_DIV(a,b) ((a) + (b) - 1) / (b)

template<const int TILE_SIZE>
__global__ void transpose_kernel(int m,int n,float *mat,float *output) {
    int bid_x = blockIdx.x, bid_y = blockIdx.y;
    int tid_x = threadIdx.x, tid_y = threadIdx.y;
    // avoid bank conflict
    __shared__ float mat_s[TILE_SIZE][TILE_SIZE + 1];
    int x = bid_x * TILE_SIZE + tid_x;
    int y = bid_y * TILE_SIZE + tid_y;

    if(x < n && y < m) mat_s[tid_y][tid_x] = mat[y * n + x];

    __syncthreads();

    x  = bid_y * TILE_SIZE + tid_x;
    y  = bid_x * TILE_SIZE + tid_y;

    if(x < m && y < n) output[y * m + x] = mat_s[tid_x][tid_y];
}


void transepose() {
    int m = 3,n = 4;
    const int TILE_SIZE = 2;
    // float cpu_mat[] = {
    //     1,2,3,4,
    //     5,6,7,8,
    //     1,2,3,4,
    //     5,6,7,8
    // };
    float cpu_mat[m * n] = {
        1,2,3,4,
        5,6,7,8,
        1,2,3,4,
    };
    cudaSetDevice(0);
    float *gpu_mat, *gpu_output;
    int sizeBytes = sizeof(float) * (m*n);
    cudaMalloc((void**)&gpu_mat,sizeBytes);
    cudaMalloc((void**)&gpu_output,sizeBytes);
    cudaMemcpy(gpu_mat,cpu_mat,sizeBytes,cudaMemcpyHostToDevice);
    dim3 gridSize(CEIL_DIV(n,TILE_SIZE),CEIL_DIV(m,TILE_SIZE));
    dim3 blockSize(TILE_SIZE,TILE_SIZE);
    transpose_kernel<TILE_SIZE><<<gridSize,blockSize>>>(m,n,gpu_mat,gpu_output);
    cudaMemcpy(cpu_mat,gpu_output,sizeBytes,cudaMemcpyDeviceToHost);
    if(m != n) std::swap(m,n);
    for(int i = 0;i < m; i++) {
        for(int j = 0;j < n; j++) {
            std::cout<< cpu_mat[i * n + j] << " ";
        }
        std::cout<<std::endl;
    }
    cudaFree(gpu_mat);
    cudaFree(gpu_output);
}


int main() {
    transepose();
}


