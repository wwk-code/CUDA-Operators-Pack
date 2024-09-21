#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float *input, float *output, int N) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data to shared memory
    if (idx < N) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = -INFINITY;
    }
    __syncthreads();

    // Compute the max value
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    // Broadcast the max value
    float max_val = shared_data[0];
    __syncthreads();

    // Compute the exponentials and sum
    if (idx < N) {
        shared_data[tid] = expf(input[idx] - max_val);
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Broadcast the sum
    float sum = shared_data[0];
    __syncthreads();

    // Compute the softmax
    if (idx < N) {
        output[idx] = expf(input[idx] - max_val) / sum;
    }
}

void softmax(float *input, float *output, int N) {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_input, d_output, N);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}


int main() {
    const int N = 10;
    float input[N] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output[N];

    softmax(input, output, N);

    std::cout << "Softmax output: ";
    for (int i = 0; i < N; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

