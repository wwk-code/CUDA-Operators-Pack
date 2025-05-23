/*
 convolution supported by CUDA implements
 matrix is orgnized in 1D array
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// #define DEBUG
#define element_type float
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

/*
    @brief: 串行卷积实现 CPU代码 NCHW
    @param in inC inH inW: 输入矩阵(数组) channel height width
    @param out outC outH outW: 输出矩阵 channel height width
    @param kernel kernelH kernelW: 卷积核 height width
*/
void serial_convolution(element_type *in, element_type *out, element_type *kernel, 
                        int batch_size,
                        int inC, int inH, int inW,
                        int outC, int outH, int outW,
                        int kernelH, int kernelW)
{
    float val;
    int out_pos, in_pos, kernel_pos;
    for (int oc = 0; oc < outC; oc++) // 每个输出通道
    {
        // 对一个位置的操作 用当前输入channel卷积去对相应的输出channel
        // 保证每个outChannel都是多inChannel累积的结果
        for (int i = 0; i < outH; i++)
        {
            for (int j = 0; j < outW; j++)
            {
                val = 0; // 避免累积和需要多次读取写入
                out_pos = oc * outH * outW + OFFSET(i, j, outW);
                for (int ic = 0; ic < inC; ic++) // 对每个输入通道
                {
                    for (int ii = 0; ii < kernelH; ii++)
                    {
                        for (int jj = 0; jj < kernelW; jj++)
                        {
                            in_pos = ic * inH * inW + OFFSET(i + ii, j + jj, inW);
                            kernel_pos = oc * kernelH * kernelW + OFFSET(ii, jj, kernelW);
                            val += in[in_pos] * kernel[kernel_pos];
                        }
                    }
                }
                out[out_pos] = val; // 与cudnn差个负号
            }
        }
    }
}

/*
    @brief: GPU Conv Kernel
    @param in inC inH inW: 输入矩阵(数组) channel height width
    @param out outC outH outW: 输出矩阵 channel height width
    @param kernel kernelH kernelW: 卷积核 height width
    @attention: template可以保证传入静态变量直接用于分配静态存储空间
*/
template <
    const int BLOCK_HEIGHT,
    const int BLOCK_WIDTH,
    const int KERNEL_HEIGHT,
    const int KERNEL_WIDTH,
    const int MALLOC_TEMP_SIZE,
    const int MALLOC_KERNEL_HEIGHT,
    const int MALLOC_KERNEL_WIDTH,
    const int MALLOC_BLOCK_HEIGHT,
    const int MALLOC_BLOCL_WIDTH>
__global__ void v1_convolution(element_type *in, element_type *out, element_type *kernel, int batch_size,
                               int inC, int inH, int inW,
                               int outC, int outH, int outW,
                               int kernelH, int kernelW)
{
    // block id 与 thread id的读取与计算 分块是对target矩阵去分的
    // 目前按一个线程负责一个in的计算
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y, thread_col = threadIdx.x;
    int threadH = BLOCK_HEIGHT, threadW = BLOCK_WIDTH; // 线程网格范围
    int thread_num_per_block = threadH * threadW, tid = thread_row * threadW + thread_col;
    // 分块边界 boundary是限制正常范围 edge是需要特殊处理的范围
    int row_boundary = outH / BLOCK_HEIGHT - 1,
        col_boundary = outW / BLOCK_WIDTH - 1;
    int row_edge = outH % BLOCK_HEIGHT, col_edge = outW % BLOCK_WIDTH;
    // 线程计算值暂存区大小 动态分配不是很方便 需要外部分配并带进在来   （注意这里有大学问：一个Block实际上分配的TB大小要大于其负责计算的子Matrix大小，否则其边缘元素是无法计算的）
    // 一般取单个计算元素和oc之积的2倍即可 因为block比较小
    const int temp_size = MALLOC_TEMP_SIZE;

    // if (tid==0)
    //     printf("(%d %d)\n", block_row, block_col);

    /// 转移存储 GMEM --> SMEM
    // __shared__ float s_in[BLOCK_HEIGHT + KERNEL_HEIGHT - 1][BLOCK_WIDTH + KERNEL_WIDTH - 1];
    __shared__ float s_kernel[MALLOC_KERNEL_HEIGHT][MALLOC_KERNEL_WIDTH]; // 开奇数内存会出错
    __shared__ float s_in[MALLOC_BLOCK_HEIGHT][MALLOC_BLOCL_WIDTH];       // 要满足修正的尺寸
    float load_reg[4];

    // 当前block的起始位置
    // int begin_pos = (block_row + thread_row) * BLOCK_HEIGHT + (block_col) * BLOCK_WIDTH + thread_col;
    int begin_pos = block_row * BLOCK_HEIGHT * inW + block_col * BLOCK_WIDTH;

    int single_trans_ele_num = 4;                               // 线程一次转移的数据数
    int cur_in_block_height = BLOCK_HEIGHT + KERNEL_HEIGHT - 1, // 读入in的block height
        cur_in_block_width = BLOCK_WIDTH + KERNEL_WIDTH - 1,    // 读入in的block width
        in_tile_thread_per_row,                                 // 以tile为单位转移数据，一行需要的thread数
        in_tile_row_start,                                      // tile的行起始位置
        in_tile_col,                                            // tile的列
        in_tile_row_stride;                                     // tile行跨度

    // 修正边缘block尺寸
    if (block_row == row_boundary)
    {
        cur_in_block_height = BLOCK_HEIGHT + row_edge + kernelH - 1;
    }
    if (block_col == col_boundary)
    {
        cur_in_block_width = BLOCK_WIDTH + col_edge + kernelW - 1;
    }

    in_tile_thread_per_row = cur_in_block_width / single_trans_ele_num;
    in_tile_row_start = tid / in_tile_thread_per_row;
    in_tile_col = tid % in_tile_thread_per_row * single_trans_ele_num;
    in_tile_row_stride = thread_num_per_block / in_tile_thread_per_row;

    // 下方都是读取第一个channel的数据
    // 按行读取 每行令线程以tile为单位读取 tile大小目前为single_trans_ele_num，余量特殊处理
    for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
         i += in_tile_row_stride)
    {
        FETCH_FLOAT4(load_reg[0]) =
            FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inW)]);
        s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
        s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
        s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
        s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
        if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
            cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
        {
            for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
            {
                s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, inW)];
            }
        }
    }

    // 读取第一个kernel的数据 (这里是让每个 thread_col == 0 的线程去负责读取这一个 thread row的kernel，因为kernel是比较小的，就没必要再去划分麻烦事)
    if (thread_row >= 0 && thread_row < KERNEL_HEIGHT && thread_col == 0)
    {
        for (int j = 0; j < KERNEL_WIDTH; j++)
        {
            s_kernel[thread_row][j] = kernel[OFFSET(thread_row, j, KERNEL_WIDTH)];
        }
    }

    __syncthreads();

    // 逐个channel计算 一个线程负责block中的一个元素计算
    // 修正out block的大小
    int cur_out_block_height = BLOCK_HEIGHT, // 输出block height
        cur_out_block_width = BLOCK_WIDTH,   // 输出block width
        single_calculate_num = 1,            // 线程一次负责计算的元素数目
        out_tile_thread_per_row,             // block按tile划分需要的线程数目
        out_tile_row_start,                  // tile的行起始位置
        out_tile_col,                        // tile的列起始位置
        out_tile_row_stride;                 // tile行跨度
    if (block_row == row_boundary)
    {
        cur_out_block_height = BLOCK_HEIGHT + row_edge;
    }
    if (block_col == col_boundary)
    {
        cur_out_block_width = BLOCK_WIDTH + col_edge;
    }

    out_tile_thread_per_row = cur_out_block_width / single_calculate_num;
    out_tile_row_start = tid / out_tile_thread_per_row;
    out_tile_col = tid % out_tile_thread_per_row * single_calculate_num;
    out_tile_row_stride = thread_num_per_block / out_tile_thread_per_row;

    float val[temp_size]; // 存储累积和 避免多次读取GMEM
    for (int i = 0; i < temp_size; i++)
        val[i] = 0;

    int out_pos, temp_pos;

    for (int oc = 0; oc < outC; oc++)
    {
        for (int ic = 0; ic < inC; ic++)
        {
            // i,j 是相当于当前block起始位置而言
            // 每个线程负责一个tile，元素数>线程数会进行轮替，会有少部分的重叠区域，代价不大（只要width不大）
            // 用ic的每个block去对oc的kernel进行计算
            for (int i = 0; i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height;
                 i += out_tile_row_stride)
            {
                for (int j = 0; j < single_calculate_num; j++)
                {
                    // 计算线程负责的元素 同一个oc的缓存顺序排列
                    // 不同oc偏移一个cur_out_block_height / out_tile_row_stride + 1的位置
                    temp_pos = i / out_tile_row_stride + j +
                               oc * (cur_out_block_height / out_tile_row_stride + 1);
                    for (int ii = 0; ii < KERNEL_HEIGHT; ii++)
                    {
                        for (int jj = 0; jj < KERNEL_WIDTH; jj++) // 更换的是SMEM中的内容，相对位置不变
                        {
                            val[temp_pos] += s_in[out_tile_row_start + i + ii][out_tile_col + j + jj] * s_kernel[ii][jj];
                        }
                    }
                }
            }
            //  
            if (ic + 1 < inC)
            {
                for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
                     i += in_tile_row_stride)
                {
                    FETCH_FLOAT4(load_reg[0]) =
                        FETCH_FLOAT4(in[begin_pos + (ic + 1) * inH * inW + OFFSET(in_tile_row_start + i, in_tile_col, inW)]);
                    s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
                    s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
                    s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
                    s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
                    if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
                        cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
                    {
                        for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
                        {
                            s_in[in_tile_row_start + i][j] = in[begin_pos + (ic + 1) * inH * inW + OFFSET(in_tile_row_start + i, j, inW)];
                        }
                    }
                }
                if (thread_row >= 0 && thread_row < KERNEL_HEIGHT && thread_col == 0)
                {
                    for (int j = 0; j < KERNEL_WIDTH; j++)
                    {
                        s_kernel[thread_row][j] = kernel[(oc * inC + ic + 1) * kernelH * kernelW + OFFSET(thread_row, j, KERNEL_WIDTH)];
                    }
                }
            }
            __syncthreads();
        }
        // 读取下一个kernel channel数据
        if (oc + 1 < outC)
        {
            if (thread_row >= 0 && thread_row < KERNEL_HEIGHT && thread_col == 0)
            {
                for (int j = 0; j < KERNEL_WIDTH; j++)
                {
                    s_kernel[thread_row][j] = kernel[(oc + 1) * inC * kernelH * kernelW + OFFSET(thread_row, j, KERNEL_WIDTH)];
                }
            }
        }
        __syncthreads();
        // 写回 利用线程id计算写回位置
        int i = 0, j = 0;
        while (i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height)
        {
            while (j < single_calculate_num)
            {
                out_pos = oc * outH * outW +
                          block_row * BLOCK_HEIGHT * outW + block_col * BLOCK_WIDTH +
                          OFFSET(out_tile_row_start + i, out_tile_col + j, outW);
                temp_pos = i / out_tile_row_stride + j +
                           oc * (cur_out_block_height / out_tile_row_stride + 1);
                out[out_pos] = val[temp_pos];
                j++;
            }
            i += out_tile_row_stride;
            j = 0;
        }
        // 读取下一个in channel数据
        for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
             i += in_tile_row_stride)
        {
            FETCH_FLOAT4(load_reg[0]) =
                FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inW)]);
            s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
            s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
            s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
            s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
            if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width && cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
            {       
                for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
                {
                    s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, inW)];
                }
            }
        }
    }
}
