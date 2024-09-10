#include "utils.cuh"


// 数据文件
const char *file_path = "/root/Projects/cuda/my_top_k/input_datas/inputs_1280.txt";


void printArray(DATATYPE *arr,int n) {
    for (int i = 0; i < n; i++)  printf("%f ",arr[i]);
    puts("");
}


void generateArray(DATATYPE *arr,int n) {
    for(int i = 0; i < n; i++) {
        arr[i] = n - i;
    }
}


struct timeval cpu_start, cpu_end;

void cpu_time_tic()
{
    gettimeofday(&cpu_start, NULL);
}

void cpu_time_toc()
{
    gettimeofday(&cpu_end, NULL);
}

float cpu_time()
{
    float time_elapsed = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0;
    return time_elapsed;
}


// malloc gpu memory
void *mallocCPUMem(int size)
{
    if (size <= 0)
        return NULL;
    return (void*)malloc(size * sizeof(DATATYPE));
}

void freeCPUMem(void *cpuData)
{
    if (cpuData != NULL)
    {
        free(cpuData);
    }
}

// malloc gpu memory
void *mallocGPUMem(int size)
{
    if (size <= 0)
        return NULL;
    void *gpuData;
    CUDA_CHECK(cudaMalloc(&gpuData, sizeof(DATATYPE) * size));
    return gpuData;
}

void freeGPUMem(void *gpuData)
{
    if (gpuData != NULL)
    {
        CUDA_CHECK(cudaFree(gpuData));
    }
}

void cpu2gpu(void *gpuData, void *cpuData, int size)
{
    CUDA_CHECK( cudaMemcpy(gpuData,cpuData,sizeof(DATATYPE) * size,cudaMemcpyHostToDevice) );   
}


void gpu2cpu(void *cpuData, void *gpuData, int size)
{
    CUDA_CHECK( cudaMemcpy(cpuData,gpuData,sizeof(DATATYPE) * size,cudaMemcpyDeviceToHost) );   
}

void gpu2gpu(void *gpuData1, void *gpuData2, int size)
{
    CUDA_CHECK( cudaMemcpy(gpuData1,gpuData2,sizeof(DATATYPE) * size,cudaMemcpyDeviceToDevice) );   
}


int getDataSize() {
    FILE *file = fopen(file_path,"r");
    if(file == NULL) {
        printf("无法打开文件：%s\n", file_path);
        exit(1);
    }
    int data_count = 0;
    DATATYPE value;
    while(fscanf(file,"%f",&value) == 1) data_count++;
    fclose(file);
    return data_count;
}



void readData(DATATYPE *cpu_input_data) {
    FILE *file = fopen(file_path,"r");
    if(file == NULL) {
        printf("无法打开文件：%s\n", file_path);
        exit(1);
    }
    
    DATATYPE value;
    int i = 0;
    // 将文件指针重置到文件开头
    fseek(file, 0, SEEK_SET);

    while(fscanf(file,"%f",&value) == 1) {
        cpu_input_data[i++] = value;
    }
    //用完释放资源
    fclose(file);
}


