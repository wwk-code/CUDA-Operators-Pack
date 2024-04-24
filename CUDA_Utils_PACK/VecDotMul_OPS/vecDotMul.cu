#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define NUM_SIZE 2048


// v1版本为使用 shaed_memory 的 warp_shuffle 优化版本kernel
__global__ void dotProductKernel_v1(float *a, float *b, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // 使用寄存器存储sum值
    float sum = 0.0f;

    if (index < n) {
        sum = a[index] * b[index];
    }

    // 对warp内的数据进行规约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 使用共享内存来汇总每个warp的结果
    __shared__ float warpSum[32]; // 假设每个block最多有1024个线程，最多32个warp
    int warpId = threadIdx.x / warpSize;

    // 每个 warp 的首个线程进行本warp的数据存储
    if (threadIdx.x % warpSize == 0) {
        warpSum[warpId] = sum;
    }

    __syncthreads();

    // 使用第一个warp来汇总所有warp的结果
    if (warpId == 0 && threadIdx.x < (blockDim.x / warpSize)) {  // 由第一个warp内的每个线程做最终的每个warp的数据归约
        // 用寄存器存放
        float blockSum = warpSum[threadIdx.x];
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(result, blockSum);
        }
    }
}



// v2版本的 dotProductKernel 减少了 shared_memory 的使用、以及线程同步的次数、线程束分化次数等等,同时对每一个线程做了工作负载均摊，
// 不过带来了 单个线程中的循环迭代以及最后的原子操作次数的增加，这里需要针对向量长度: 2048 以及 blcokSize 做详细的对比，用nvprof或Nvidia Nsight选出合适的版本
__global__ void dotProductKernel_v2(float *a, float *b, float *result, int n) {
    // int index = threadIdx.x + blockIdx.x * blockDim.x
    float sum = 0.0f;

    // 使用循环处理大于blockSize的数据段
    for(int index = threadIdx.x + blockIdx.x * blockDim.x; index < n; index += blockDim.x * gridDim.x) {
        sum += a[index] * b[index];
    }

    // while (index < n) {
    //     // 非连续访问，可能无法高效利用缓存
    //     sum += a[index] * b[index];
    //     index += blockDim.x * gridDim.x;  // 跳跃步长为网格x维度的大小
    // }

    // Warp内部规约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 每个warp的第一个线程做本warp的归约值存储操作
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(result, sum);
    }

}



__global__ void dotProductKernel_v3(float *a, float *b, float *result, int n) {
    __shared__ float sharedSum[32];  
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    float sum = 0.0f;

    for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < n; index += blockDim.x * gridDim.x) {
        sum += a[index] * b[index];
    }

    // Warp内部规约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 使用共享内存进一步归约
    if (laneId == 0) {
        sharedSum[warpId] = sum;
    }
    
    __syncthreads();

    // Block内的第一个warp完成最终归约
    if (warpId == 0) {
        float blockSum = (laneId < blockDim.x / warpSize) ? sharedSum[laneId] : 0.0f;
        // 使用 warp reduce操作来减少后续的原子操作的次数
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);
        }
        
        if (laneId == 0) {
            atomicAdd(result, blockSum);
        }
    }
}



// CPU版的向量点乘验证函数
float dotProductCPU(float *a, float *b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}



int main() {
    int n = NUM_SIZE;
    size_t bytes = n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(sizeof(float));
    *h_c = 0.0f;

    // 简单赋值
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f; 
        h_b[i] = 2.0f; 
    }

    // Allocate memory on device
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, sizeof(float));

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(float), cudaMemcpyHostToDevice);

    // 超参数
    int blockSizes[] = {16,32,64,128,256,512,1024};  // 建议的 blockSize测试
    // int blockSize = blockSizes[2];   
    // int blockSize = blockSizes[4];   // 选用 v1和v2版本的kernel时，  TB_size 为 256 效果较为不错
    int blockSize = blockSizes[6];   // 选用 v3 版本的kernel时，  TB_size 为 1024 效果较为不错
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 简易 Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // dotProductKernel_v1<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    // dotProductKernel_v2<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    dotProductKernel_v3<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    // 运行CPU验证函数
    float cpuResult = dotProductCPU(h_a, h_b, n);
    // printf("result: %f\n",*h_c);
    // printf("cpu_result: %f\n",cpuResult);
    // 验证kernel功能正确性
    assert(abs(*h_c - cpuResult) < 1e-5);
    printf("Dot product computed correctly.\n");


    int test_cnt = 10;
    float time = 0,total_time = 0;
    // kernel性能测试
    for(int i = 0; i < test_cnt; i++) {
        // 记录kernel启动前的时间
        cudaEventRecord(start);

        // Launch kernel
        // dotProductKernel_v1<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
        // dotProductKernel_v2<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
        dotProductKernel_v3<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

        // 记录kernel完成后的时间
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }
    total_time /= test_cnt;
    printf("kernel's time consumption: %f ms\n", total_time);

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}






