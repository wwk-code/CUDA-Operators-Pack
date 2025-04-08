#include "top_k.cuh"
#include "utils.cuh"
#include "macro.cuh"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuda.h>

__device__ inline void replace_smaller(DATATYPE *array, int k, DATATYPE data)
{
    if (data < array[k - 1])
        return;
    for (int j = k - 2; j >= 0; j--)
    {
        if (data > array[j])
            array[j + 1] = array[j];
        else
        {
            array[j + 1] = data;
            return;
        }
    }
    array[0] = data;
}

// 现在先假设整个kernel只分配一个TB，threadIdx的维度也只有一维
__global__ void top_k_gpu_kernel1(DATATYPE *input, DATATYPE *output, int n, int k)
{
    // 动态 shared_memory
    extern __shared__ DATATYPE shared_buffer[];
    // 本线程待处理数据块的起始地址
    DATATYPE *mySegment = shared_buffer + threadIdx.x * k;
    // 全局线程 Id
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // 全局线程数目
    int threadNum = blockDim.x * gridDim.x;

    for (int index = 0, i = threadId; index < k; index++, i += threadNum)
    {
        mySegment[index] = NEG_INF;
        replace_smaller(mySegment, index + 1, input[i]);
    }
    for (int i = k * threadNum + threadId; i < n; i += threadNum)
    {
        replace_smaller(mySegment, k, input[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < k; i++)
            output[i] = mySegment[i];
        for (int i = k; i < k * threadNum; i++)
            replace_smaller(output, k, mySegment[i]);
    }
}

// 合并两个线程的 segment 区
__device__ void mergeTwoSegment(DATATYPE *seg1, DATATYPE *seg2, int k)
{
    for (int i = 0; i < k; i++)
        replace_smaller(seg1, k, seg2[i]);
}

/**
 *  策略:  相比于v1版本的kernel，每一个 TB 块 处理整个input的一个子段，然后分为3步:
 *  1. 每个TB块准备好大小为 k * blockThreadNum的连续shared_memory,这段shared_memory中的值每连续k个元素为一个单位
 *  2. 每个TB块会对上述提到的连续shared_memory以每k个为单位做归约，最终每个TB块的首k个元素为此TB块内的最大的k个元素
 *  3. 对每个TB块中的首k个元素再次进行归约，先归约到全局首个segment中去，此时这个segment中的k个数据已经是input中最大的k个元素了,再将这个segment复制到output中
 */
__global__ void top_k_gpu_kernel2(DATATYPE *input, DATATYPE *output, int n, int k, int *lock)
{
    extern __shared__ DATATYPE shared_buffer[];
    int blockThreadNum = blockDim.x, blockNum = gridDim.x;
    int tid = threadIdx.x, bid = blockIdx.x;
    int blockWorkLoads = n / blockNum; // 注意: 这里有BUG，我并未添加最后一个Block可能需要多处理一些数据，现在的逻辑是假设数据量 n 正好可以整除 blockNum

    DATATYPE *mySegment = shared_buffer + tid * k; // 本线程处理的数据块起始位置
    int i,index;
    for (index = 0,i = bid * blockWorkLoads + tid; index < k && i < (bid + 1) * blockWorkLoads; index++, i += blockThreadNum)
    {
        mySegment[index] = NEG_INF;
        replace_smaller(mySegment, index + 1, input[i]);
    }
    
    for (; i < (bid + 1) * blockWorkLoads; i += blockThreadNum)
    {
        replace_smaller(mySegment, k, input[i]);
        __syncthreads();
    }

    // 这里执行完时每个TB所负责的子段的前K个元素都是其中最大的K个

    // TB内归约
    for (int stride = (blockThreadNum * k) >> 1; stride >= k; stride >>= 1)
    {
        if ((tid*k) < stride)
            mergeTwoSegment(mySegment, mySegment + stride, k);
        __syncthreads();
    }
    // 每个TB块的首线程负责全局归约
    if (tid == 0)
    {
        // acquire lock
        while (atomicCAS(lock, 0, 1) != 0);
        mergeTwoSegment(output, mySegment, k);
        // release lock
        atomicExch(lock, 0);
    }
    
}

#endif