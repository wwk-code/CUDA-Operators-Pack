#include <iostream>
#include <cuda_runtime.h>

#define element_type float
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/*
    @brief: 串行卷积实现 CPU代码 NCHW (假设 batch_size == 1, 没有考虑多 batch_size)
    @param in inC inH inW: 输入矩阵(数组) channel height width
    @param out outC outH outW: 输出矩阵 channel height width
    @param kernel kernelH kernelW: 卷积核 height width
*/
void serial_convolution(element_type *in,element_type *out,element_type *kernel,int batch_size,
                        int inC,int inH,int inW,
                        int outC,int outH,int outW,
                        int kernelH, int kernelW )
{
    float val;
    int out_pos, in_pos, kernel_pos;
    for (int oc = 0; oc < outC; oc++) {
        for (int i = 0; i < outH; i++) {
            for (int j = 0; j < outW; j++) {
                val = 0;
                out_pos = oc * outH * outW + OFFSET(i,j,outW);
                for (int ic = 0; ic < inC; ic++) {
                    for (int ii = 0; ii < kernelH; ii++) {
                        for (int ij = 0; ij < kernelW; ij++) {
                            in_pos = ic * inH * inW + OFFSET(i + ii,j + ij, inW);  // 对于 i+ii 和 j+ij，需要对应上述卷积计算图去理解 
                            kernel_pos = oc * kernelH * kernelW + OFFSET(ii,ij,kernelW);
                            val += in[in_pos] * kernel[kernel_pos];
                        }
                    }
                }
                out[out_pos] = val;
            }
        }
    }



}





template <
    const int BLOCK_SIZE_ROW,
    const int BLOCK_SIZE_COL,
    const int THREAD_SIZE_ROW,
    const int THREAD_SIZEZ_COL,
    const int FILTER_SIZE>
__global__ void v2_convolution(element_type *org,
                               element_type *target,
                               element_type *filter,  // assume that filter kernel is a square matrix
                               int row, int col)  // row and col are the org mat's attributes
{
    // block id 与 thread id的读取与计算 分块是对target矩阵去分的
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y, thread_col = threadIdx.x, tid = thread_row * THREAD_SIZE_ROW + thread_col;
    // 目标矩阵尺寸
    int t_row = row - FILTER_SIZE + 1, t_col = col - FILTER_SIZE + 1;
    // 分块边界
    int row_boundary = t_row / BLOCK_SIZE_ROW - 1, col_boundary = t_col / BLOCK_SIZE_COL - 1;
    int row_edge = t_row % BLOCK_SIZE_ROW, col_edge = t_col % BLOCK_SIZE_COL;

    if (block_row == 0 && block_col == 0 && thread_row == 0 && thread_col == 0)
        printf("filter0:%.2f\n", filter[0]);
    // 转移存储 GMEM --> SMEM
    // __shared__ float s_filter[filter_size][filter_size];
    // __shared__ float s_org[BLOCK_SIZE_ROW + filter_size - 1][BLOCK_SIZE_COL + filter_size - 1];
    __shared__ float s_filter[FILTER_SIZE][FILTER_SIZE];
    __shared__ float s_org[BLOCK_SIZE_ROW + FILTER_SIZE - 1][BLOCK_SIZE_COL + FILTER_SIZE - 1];
    int begin_pos = block_row * BLOCK_SIZE_ROW * col + block_col * BLOCK_SIZE_COL * row; // 当前block的起始位置
    // 右下角元素负责filter_size^2的元素转移
    if (thread_row == BLOCK_SIZE_ROW - 1 && thread_col == BLOCK_SIZE_COL - 1)
    {
        for (int i = 0; i < FILTER_SIZE; i++)
        {
            for (int j = 0; j < FILTER_SIZE; j++)
            {
                s_org[thread_row + i][thread_col + j] =
                    org[begin_pos + OFFSET(thread_row + i, thread_col + j, col)];
            }
        }
    }
    else if (thread_row == BLOCK_SIZE_ROW - 1) // 下边界向外延伸
    {
        for (int i = 0; i < FILTER_SIZE; i++)
        {
            s_org[thread_row + i][thread_col] =
                org[begin_pos + OFFSET(thread_row + i, thread_col, col)];
        }
    }
    else if (thread_col == BLOCK_SIZE_COL - 1) // 右边界向外延伸
    {
        for (int i = 0; i < FILTER_SIZE; i++)
        {
            s_org[thread_row][thread_col + i] =
                org[begin_pos + OFFSET(thread_row, thread_col + i, col)];
        }
    }
    else // 边界内只需负责转移自身数据
    {
        s_org[thread_row][thread_col] =
            org[begin_pos + OFFSET(thread_row, thread_col, col)];
        // 0号线程同时转移filter
        if (thread_row == 0 && thread_col == 0)
        {
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    s_filter[i][j] = filter[OFFSET(i, j, FILTER_SIZE)];
                }
            }
        }
    }

    __syncthreads();

    // 计算部分
    if (block_row == row_boundary && block_col == col_boundary) // 最右下角的 负责处理edge部分
    {
        if (thread_row < row_edge && thread_col < col_edge)
        {
            int value = 0;
            // single_conv(FILTER_SIZE, s_org, target, s_filter, begin_pos, thread_row, thread_col, t_col);
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    value += s_org[thread_row + i][thread_col + j] * s_filter[thread_row + i][thread_col + j];
                }
            }
            target[begin_pos + OFFSET(thread_row, thread_col, t_col)] = value;
        }
    }
    else if (block_row == row_boundary) // 下边一条的edge
    {
        if (thread_row < row_edge)
        {
            int value = 0;
            // single_conv(FILTER_SIZE, s_org, target, s_filter, begin_pos, thread_row, thread_col, t_col);
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    value += s_org[thread_row + i][thread_col + j] * s_filter[thread_row + i][thread_col + j];
                }
            }
            target[begin_pos + OFFSET(thread_row, thread_col, t_col)] = value;
        }
    }
    else if (block_col == col_boundary) // 右边一条的edge
    {
        if (thread_col < col_edge)
        {
            int value = 0;
            // single_conv(FILTER_SIZE, s_org, target, s_filter, begin_pos, thread_row, thread_col, t_col);
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    value += s_org[thread_row + i][thread_col + j] * s_filter[thread_row + i][thread_col + j];
                }
            }
            target[begin_pos + OFFSET(thread_row, thread_col, t_col)] = value;
        }
    }
    else // 正常block
    {
        float value = 0;
        // single_conv(FILTER_SIZE, s_org, target, s_filter, begin_pos, thread_row, thread_col, t_col);
        for (int i = 0; i < FILTER_SIZE; i++)
        {
            for (int j = 0; j < FILTER_SIZE; j++)
            {
                // if (block_row == 13 && block_col == 11 && thread_row == 0 && thread_col == 0)
                //     printf("%d %d %.2f * %.2f\n", thread_row + i, thread_col + j, s_org[thread_row + i][thread_col + j], s_filter[i][j]);
                value += s_org[thread_row + i][thread_col + j] * s_filter[i][j];
            }
        }
        target[begin_pos + OFFSET(thread_row, thread_col, t_col)] = value;
        // if (block_row == 0 && block_col == 0 && thread_row == 0 && thread_col == 0)
        // printf("%d-%d.%d-%d : %.2f\n", block_row, block_col, thread_row, thread_col, value);
    }
}





int main() {
    // 输入参数
    const int batch_size = 1;
    const int inC = 1;
    const int inW = 5;
    const int inH = 5;
    const int outC = 1;
    const int kernelH = 3;
    const int kernelW = 3;
    const int outH = inH - kernelH + 1;
    const int outW = inW - kernelW + 1;

    // 输入数据
    element_type in[] = {
        1, 2, 3, 0, 1, 4, 5,
        0, 1, 2, 3, 1, 0, 1,
        1, 2, 0, 1, 2, 3, 4,
        2, 0, 1, 1, 0, 1, 5,
        1, 1, 2, 0, 1, 2, 3,
        3, 4, 5, 1, 0, 1, 0,
        1, 0, 2, 3, 4, 5, 6
    };

    // 卷积核
    element_type kernel[] = {
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1
    };

    // 输出数组
    element_type out[outH * outW];

    // 设备内存指针
    element_type *d_in, *d_out, *d_kernel;

    // 分配设备内存
    cudaMalloc((void**)&d_in, sizeof(in));
    cudaMalloc((void**)&d_out, sizeof(out));
    cudaMalloc((void**)&d_kernel, sizeof(kernel));

    // 复制数据到设备
    cudaMemcpy(d_in, in, sizeof(in), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);

    // 设置 CUDA kernel 的网格和块的维度
    dim3 blockSize(2, 2); // 根据实际需要设置
    dim3 gridSize((outW + blockSize.x - 1) / blockSize.x, 
                  (outH + blockSize.y - 1) / blockSize.y);

    // 调用 v2_convolution kernel
    v2_convolution<2, 2, 1, 1, 5><<<gridSize, blockSize>>>(d_in, d_out, d_kernel, inH, inW);

    // 将输出结果从设备复制回主机
    cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost);

    // 执行卷积
    // serial_convolution(in, out, kernel, batch_size, inC, inH, inW, outC, outH, outW, kernelH, kernelW);

    // 打印输出
    std::cout << "Output:\n";
    for (int i = 0; i < outH; ++i) {
        for (int j = 0; j < outW; ++j) {
            std::cout << out[OFFSET(i, j, outW)] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}