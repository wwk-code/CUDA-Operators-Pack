#include "utils.cuh"
#include "top_k.cuh"

#define diff 1e-5
#define TEST_BENCH 10
#define KERNEL_VERSION 2

int main()
{
    int n = getDataSize(); // 数据个数
    int k = 20;
    int *lock;     // 全局计数变量，可用于实现 TB块 之间的同步

    CUDA_CHECK(cudaSetDevice(0));   
    DATATYPE *cpu_intput_data = (DATATYPE *)mallocCPUMem(n);
    DATATYPE *cpu_output_data = (DATATYPE *)mallocCPUMem(k);
    DATATYPE *cpu_preFillOutput = (DATATYPE *)mallocCPUMem(k);

    DATATYPE *gpu_intput_data = (DATATYPE *)mallocGPUMem(n);
    DATATYPE *gpu_output_data = (DATATYPE *)mallocGPUMem(k);
    DATATYPE *cpu_gpu_test_data = (DATATYPE *)mallocCPUMem(k);

    //分配锁变量
    CUDA_CHECK( cudaMalloc((void**)(&lock),sizeof(int)) );

    for(int i = 0; i < k; i++) cpu_preFillOutput[i] = NEG_INF;
    cpu2gpu(gpu_output_data,cpu_preFillOutput,k);

    readData(cpu_intput_data);
    // cudaDeviceSynchronize();
    cpu2gpu(gpu_intput_data, cpu_intput_data, n);

    cpu_time_tic();
    top_k_cpu_serial(cpu_intput_data, cpu_output_data, n, k);
    cpu_time_toc();

    float total_cpu_time = cpu_time();

#if KERNEL_VERSION == 1
    int blocks = 1;
    int threads = (GPU_THREADS < n / (4 * k) * 2) ? GPU_THREADS : (n / (4 * k) * 2);
    int shared_mem_usage = sizeof(DATATYPE) * k * threads;
    top_k_gpu_kernel1<<<blocks, threads, shared_mem_usage>>>(gpu_intput_data, gpu_output_data, n, k);

#elif KERNEL_VERSION == 2
    int blocks = GPU_BLOCKS;   // 增加了 TB 块的数目
    int threads = (GPU_THREADS < n / (4 * k) * 2) ? GPU_THREADS : (n / (4 * k) * 2);  
    int threads_sharedMemory_limit = ((49152 / sizeof(DATATYPE)) + k) / k - 1;    // 由于本卡的一个TB上的shared_memory限制是 4GB(49512 B)，所以要限制threads
    threads = (threads <= threads_sharedMemory_limit) ? threads : threads_sharedMemory_limit;
    int shared_mem_usage = sizeof(DATATYPE) * k * threads;
    top_k_gpu_kernel2<<<blocks, threads, shared_mem_usage>>>(gpu_intput_data, gpu_output_data, n, k, lock);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Error at line %d in file %s \n",__LINE__,__FILE__); 
        printf("Error message: %s\n",cudaGetErrorString(error)); 
        exit(1); 
    }

#endif

    gpu2cpu(cpu_gpu_test_data, gpu_output_data, k);

    for (int i = 0; i < k; i++)
    {
        assert(abs(cpu_gpu_test_data[i] - cpu_output_data[i]) < diff);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float time = 0;

    cudaEventRecord(start);
#if KERNEL_VERSION == 0
    top_k_gpu_kernel1<<<blocks, threads,shared_mem_usage>>>(gpu_intput_data, gpu_output_data, n, k);
#elif KERNEL_VERSION == 1
    top_k_gpu_kernel1<<<blocks, threads, shared_mem_usage>>>(gpu_intput_data, gpu_output_data, n, k);
#endif

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    printf("total cpu_time: %f ms\n", total_cpu_time);
    printf("total gpu_time: %f ms\n", time);

    // printArray(cpu_output_data,k);
    // printArray(cpu_gpu_test_data,k);
    puts("\nPASS!\n");

    freeCPUMem(cpu_intput_data);
    freeCPUMem(cpu_output_data);
    freeCPUMem(cpu_gpu_test_data);
    freeGPUMem(gpu_intput_data);
    freeGPUMem(gpu_output_data);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
