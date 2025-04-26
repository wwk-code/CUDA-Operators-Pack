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


// 合并两个线程的 segment 区
__device__ void mergeTwoSegment(DATATYPE *seg1, DATATYPE *seg2, int k)
{
    for (int i = 0; i < k; i++)
        replace_smaller(seg1, k, seg2[i]);
}


/**
 *  策略:  
 *  1. 每个TB块准备好大小为 k * blockThreadNum 的连续shared_memory,这段shared_memory中的值每连续k个元素为一个单位由一个线程负责
 *  2. 每个TB块会对上述提到的连续shared_memory以每k个为单位做归约，每个线程处理k个，最终每个TB块的首k个元素为此TB块内的最大的k个元素
 *  3. 对每个TB块中的首k个元素再次进行全局归约(用锁机制独占式地与全局输出数据output进行两个K块间的比较合并)
 */
__global__ void top_k_gpu_kernel2(DATATYPE *input, DATATYPE *output, int n, int k, int *lock)
{
    extern __shared__ DATATYPE shared_buffer[];
    int blockThreadNum = blockDim.x, blockNum = gridDim.x;
    int tid = threadIdx.x, bid = blockIdx.x;
    int blockWorkLoads = n / blockNum; // 注意: 这里有BUG，我并未添加最后一个Block可能需要多处理一些数据，现在的逻辑是假设数据量 n 正好可以整除 blockNum

    DATATYPE *mySegment = shared_buffer + tid * k; // 本线程处理的数据块起始位置
    int i,index;

    // mySegment中的有效数据数小于k时
    for (index = 0,i = bid * blockWorkLoads + tid; index < k && i < (bid + 1) * blockWorkLoads; index++, i += blockThreadNum)
    {
        mySegment[index] = NEG_INF;
        replace_smaller(mySegment, index + 1, input[i]);
    }
    
    // mySegment中的有效数据数等于k时
    for (; i < (bid + 1) * blockWorkLoads; i += blockThreadNum)
    {
        replace_smaller(mySegment, k, input[i]);
        __syncthreads();
    }

    // 这里执行完时此TB内的每个mySegment的前K个元素都是此线程所负责的 blockWorkLoad / blockThreadNum 中的最大的k个


    // TB内归约(脑海里要能想象出具体的一维TB块规约场景)
    for (int stride = (blockThreadNum * k) >> 1; stride >= k; stride >>= 1)
    {
        if ((tid*k) < stride)  // 以规约的"前半部分"作为最终承载块
            mergeTwoSegment(mySegment, mySegment + stride, k);
        __syncthreads();
    }

    // 到此时,每个TB块的首线程(维护的前k个元素)负责全局归约
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