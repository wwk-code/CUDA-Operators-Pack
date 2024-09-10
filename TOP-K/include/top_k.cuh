#ifndef TOP_K_CPU_H
#define TOP_K_CPU_H

#include "macro.cuh"
#include "utils.cuh"

#ifdef USE_GPU
    __global__ void top_k_gpu_kernel1(DATATYPE* input,DATATYPE* output, int n ,int k);
    __global__ void top_k_gpu_kernel2(DATATYPE* input,DATATYPE* output, int n ,int k,int *lock);
#endif

void top_k_cpu_serial(DATATYPE* input,DATATYPE* output, int n ,int k);

#endif