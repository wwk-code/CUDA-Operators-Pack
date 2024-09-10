#ifndef UTILS_H
#define UTILS_H

#include "macro.cuh"

#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_KERNEL_CALL(kernel_call) kernel_call

#define CUDA_CHECK(call) do { \
    cudaError_t cuda_ret = (call); \
    if(cuda_ret != cudaSuccess)  { \
        printf("CUDA Error at line %d in file %s \n",__LINE__,__FILE__); \
        printf("Error message: %s\n",cudaGetErrorString(cuda_ret)); \
        printf("In the function call %s \n",#call);  \
        exit(1); \
    }  \
} while(0)


void cpu_time_tic();
void cpu_time_toc();
float cpu_time();


void printArray(DATATYPE *arr,int n);
void generateArray(DATATYPE *arr,int n);

void *mallocCPUMem(int size);
void freeCPUMem(void *cpuData);


void *mallocGPUMem(int size);
void freeGPUMem(void *gpuData);
void cpu2gpu(void *gpuData, void *cpuData, int size);
void gpu2cpu(void *cpuData, void *gpuData, int size);
void gpu2gpu(void *gpuData1, void *gpuData2, int size);


int getDataSize();
void readData(DATATYPE *cpu_input_data);

#endif