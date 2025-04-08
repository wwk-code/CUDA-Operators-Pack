#include <stdio.h>
#include <stdlib.h>

// 核函数：向量加法
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];  // 故意留一个越界bug（当tid == n时）
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(int);

    // 主机内存分配
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 设备内存分配
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 数据拷贝到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 启动核函数（故意用n+1以触发bug）
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证结果（检查前10个元素）
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}