#include <iostream>
#include <cuda_runtime.h>
#include <utility>

#define BLOCK_SIZE 512


__global__ void merge_kernel(float *Array, float *tempArr,int size, int subListSize) {
    int start = (threadIdx.x + blockIdx.x * blockDim.x) * 2 * subListSize;
    int mid = min(start + subListSize,size);
    int end = min(start + 2*subListSize,size);
    if(mid < end) {
        int i = start,j = mid, k = start;
        while(i < mid && j < end) {
            if(Array[i] <= Array[j]) tempArr[k++] = Array[i++];
            else tempArr[k++] = Array[j++];
        }
        while(i < mid) tempArr[k++] = Array[i++];
        while(j < end) tempArr[k++] = Array[j++];
    }
}


void mergeSort(float *h_array, int size)
{
    float *d_array, *d_temp_array;
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_array, bytes);
    cudaMalloc(&d_temp_array, bytes);
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    dim3 gridSize = (size + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
    dim3 blockSize = BLOCK_SIZE;

    for(int subListSize = 1; subListSize < size; subListSize *= 2) {
        merge_kernel<<<gridSize,blockSize>>>(d_array,d_temp_array,size,subListSize);
        cudaDeviceSynchronize();
        std::swap(d_array,d_temp_array);
    }

    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    cudaFree(d_temp_array);
}


int main()
{
    const int size = 1024;
    float h_array[size];

    // 初始化数组
    for (int i = 0; i < size; ++i)
    {
        h_array[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::cout << "Unsorted array: ";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_array[i] << " ";
    }
    std::cout << "..." << std::endl;

    mergeSort(h_array, size);

    std::cout << "Sorted array: ";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_array[i] << " ";
    }
    std::cout << "..." << std::endl;

    // 验证排序结果
    bool sorted = true;
    for (int i = 1; i < size; ++i)
    {
        if (h_array[i - 1] > h_array[i])
        {
            sorted = false;
            break;
        }
    }

    if (sorted)
    {
        std::cout << "Array is sorted correctly!" << std::endl;
    }
    else
    {
        std::cout << "Array is NOT sorted correctly!" << std::endl;
    }

    return 0;
}
