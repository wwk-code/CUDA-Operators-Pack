#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FLOAT float
#define INT int
#define DTYPE FLOAT

#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))


#define A(i,j) A[i + j * lda]
#define B(i,j) B[i + j * ldb]
#define C(i,j) C[i + j * ldc]


#define CUDA_CALLER(call) do { \
    cudaError_t cuda_ret = (call); \
    if(cuda_ret != cudaSuccess) { \
        printf("CUDA Error at line %d in file %s \n",__LINE__,__FILE__); \
        printf("  Error message: %s\n", cudaGetErrorString(cuda_ret)); \
        printf(" In the function call %s \n",#call); \
        exit(1); \
    } \
} while(0)


void randomize_matrix(DTYPE *mat, int N) {
    srand(time(NULL));
    int i;
    for(i=0;i<N;i++) {
        DTYPE tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}


// double get_sec() {
//     struct timeval time;
//     gettimeofday(&time, NULL);
//     return (time.tv_sec + 1e-6 * time.tv_usec);  // 1e-6 = 10^(-6), tv_usec为微秒数，tv_sec为自1970年来过了多少秒
// }


bool verify_matrix(DTYPE *mat1, DTYPE *mat2, int n) {
    double diff = 0.0;
    int i ;
    for(i = 0; mat1 + i && mat2+i && i<n; i++) {
        diff = fabs((double)mat1[i] - (double)mat2[i]);
        if(diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}


// 自定义的矩阵拷贝函数
void copy_matrix(DTYPE *src, DTYPE *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}


void matrix_init(DTYPE *A,int n) {
    for (int i = 0; i < n; i++) {
        A[i] = 0.0f;
    }
}


