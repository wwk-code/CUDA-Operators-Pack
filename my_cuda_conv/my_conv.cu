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



// 假设 Channel 数目均为1,且所有的ThreadBlock恰好能对original matrix进行加载并运算,避免了shape不匹配时复杂情况的 TB 额外处理
// 这里注意:  target matrix 的size一定 <= org matrix,因此如果所有 TB 能保证覆盖到 org matrix,也一定可以覆盖到 tgt matrix
template <
    const int BLOCK_SIZE_ROW,   // 一个ThreadBlock负责加载的 orgMatrix 的行数
    const int BLOCK_SIZE_COL,
    const int FILTER_SIZE,
    const int TGT_BLOCK_SIZE_ROW,   // 一个ThreadBlock负责运算的的 targetMatrix 的行数
    const int TGT_BLOCK_SIZE_COL,
    const int fillRowOffset,    // 对 org matrix 进行零填充的行数
    const int fillColOffset
    >    // 假设 Filter 是方针，这里是其边长
__global__ void v2_convolution(element_type *org,
                               element_type *target,
                               element_type *filter,  // assume that filter kernel is a square matrix
                               int row, int col)  // row and col are the org mat's attributes
{
    // block id 与 thread id的读取与计算 分块是对 target 矩阵去分的
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y, thread_col = threadIdx.x, tid = thread_row * BLOCK_SIZE_COL + thread_col;
    // 目标矩阵尺寸
    int t_row = row - FILTER_SIZE + 1, t_col = col - FILTER_SIZE + 1;
    // 分块边界
    int row_boundary = row / BLOCK_SIZE_ROW - 1, col_boundary = col / BLOCK_SIZE_COL - 1;
    // 还差多少行和多少列，需要处于边界的那个threadBlock去额外处理
    int row_edge = row % BLOCK_SIZE_ROW, col_edge = col % BLOCK_SIZE_COL;

    // 转移存储 GMEM --> SMEM
    __shared__ float s_filter[FILTER_SIZE][FILTER_SIZE];
    __shared__ float s_org[BLOCK_SIZE_ROW + FILTER_SIZE - 1][BLOCK_SIZE_COL + FILTER_SIZE - 1];
    __shared__ float s_tgt[TGT_BLOCK_SIZE_ROW][TGT_BLOCK_SIZE_COL];

    // 这个threadBlock块相对于 orgMatrix 的起始位置
    int orgBeginPos = OFFSET(block_row * BLOCK_SIZE_ROW,block_col * BLOCK_SIZE_COL,col);

    s_org[thread_row][thread_col] = org[orgBeginPos + OFFSET(thread_row,thread_col,BLOCK_SIZE_COL)];
    if(thread_row < FILTER_SIZE && thread_col < FILTER_SIZE) s_filter[thread_row][thread_col] = filter[thread_row][thread_col];

    __syncthreads();

    // 计算部分
    // 此线程在全局Org Matrix中的绝对行/列
    int globalOrgThreadRow = block_row * BLOCK_SIZE_ROW + thread_row, globalOrgThreadCol = block_col * BLOCK_SIZE_COL + thread_col;
    float val = 0;
    // 现在 (globalOrgThreadRow-fillRowOffset) 和 (globalOrgThreadCol-fillColOffset) 就是此线程负责的 tgt matrix 中的绝对行列位置
    if(globalOrgThreadRow >= fillRowOffset && globalOrgThreadRow < (row-fillRowOffset)  && globalOrgThreadCol > fillColOffset && globalOrgThreadCol < (col - fillColOffset)) {
        int globalTgtThreadRow = (globalOrgThreadRow-fillRowOffset), globalTgtThreadCol = (globalOrgThreadCol-fillColOffset);        
        for(int i = 0;i < FILTER_SIZE; i++) {
            for(int j = 0;j < FILTER_SIZE; j++) {
                // 这里有问题,在sharedMemory中使用了全局坐标
                val += s_filter[i][j] * s_org[globalOrgThreadRow-(FILTER_SIZE/2) + i][globalOrgThreadCol-(FILTER_SIZE/2) + j];
            }
        }
        s_tgt[globalTgtThreadRow][globalTgtThreadCol] = val;
    }   

    __syncthreads();

    // SMem -> GMem
    if(globalOrgThreadRow >= fillRowOffset && globalOrgThreadRow < (row-fillRowOffset)  && globalOrgThreadCol > fillColOffset && globalOrgThreadCol < (col - fillColOffset)) {
        int globalTgtThreadRow = (globalOrgThreadRow-fillRowOffset), globalTgtThreadCol = (globalOrgThreadCol-fillColOffset);        
        tgt[OFFSET(globalTgtThreadRow,globalTgtThreadCol,TGT_BLOCK_SIZE_COL)] = s_tgt[globalTgtThreadRow][globalTgtThreadCol];
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