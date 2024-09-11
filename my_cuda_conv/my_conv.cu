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
        1, 2, 3, 0, 1,
        0, 1, 2, 3, 1,
        1, 2, 0, 1, 2,
        2, 0, 1, 1, 0,
        1, 1, 2, 0, 1
    };

    // 卷积核
    element_type kernel[] = {
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
    };

    // 输出数组
    element_type out[outC * outH * outW];

    // 执行卷积
    serial_convolution(in, out, kernel, batch_size, inC, inH, inW, outC, outH, outW, kernelH, kernelW);

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



// int main () {

//     const int inC = 3;
//     const int inH = 768;
//     const int inW = 512;
//     const int kernelH = 6;
//     const int kernelW = 6;
//     const int outC = 3;
//     const int outH = inH - kernelH + 1;
//     const int outW = inH - kernelW + 1;

    


    


//     return 0;
// }


