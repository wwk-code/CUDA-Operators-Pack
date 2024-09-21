#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

__global__ void merge_kernel(float *d_array, float *d_temp_array, int width, int sublist_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = 2 * idx * sublist_size;
    int mid = min(start + sublist_size, width);
    int end = min(start + 2 * sublist_size, width);

    if (mid < end)
    {
        int i = start, j = mid, k = start;
        while (i < mid && j < end)
        {
            if (d_array[i] <= d_array[j])
            {
                d_temp_array[k++] = d_array[i++];
            }
            else
            {
                d_temp_array[k++] = d_array[j++];
            }
        }
        while (i < mid)
        {
            d_temp_array[k++] = d_array[i++];
        }
        while (j < end)
        {
            d_temp_array[k++] = d_array[j++];
        }
    }
}

void mergeSort(float *h_array, int size)
{
    float *d_array, *d_temp_array;
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_array, bytes);
    cudaMalloc(&d_temp_array, bytes);
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((size + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE));

    for (int sublist_size = 1; sublist_size < size; sublist_size *= 2)
    {
        merge_kernel<<<blocks, threads>>>(d_array, d_temp_array, size, sublist_size);
        cudaDeviceSynchronize();
        std::swap(d_array, d_temp_array);
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
