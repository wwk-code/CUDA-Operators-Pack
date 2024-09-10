#include "top_k.cuh"
#include "utils.cuh"


void replace_smaller(DATATYPE* array, int k, DATATYPE data)
{
    if(data < array[k-1]) return ;   
    for(int i = k-2; i >= 0; i--) {
        if(data > array[i]) array[i+1] = array[i];   
        else {
            array[i+1] = data;
            return ;
        }
    }
    array[0] = data;
}

// 当 n >> k 时，算法时间复杂度趋近于 O(n)
void top_k_cpu_serial(DATATYPE* input,DATATYPE* output, int n ,int k)
{
    output[0] = input[0];
    for(int i = 1; i < k; i++) {
        replace_smaller(output,i+1,input[i]);
    }

    for(int i = k; i < n; i++) {
        replace_smaller(output,k,input[i]);
    }
}



