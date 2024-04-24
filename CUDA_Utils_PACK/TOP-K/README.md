# Top-K GPU Kernel

## 一、项目说明

本项目为使用Nvidia GPU对 Top-K 算法进行加速(寻找出输入向量最大的K个元素)。采用的GPU为 Nvidia Geforce RTX 3060，每个TB的sharedMemory总量为 48 KB，所以需要注意kernel_v2在处理过大的K值时，可能会面临性能性能下降

## 二、Kernel Profiling Records

### 1. kernel_v1

当数据量为万级别时，GPU kernel的性能仅仅为CPU版本的kernel的 1/8 左右  ; 当数据量为百万级别以及百万级以下时，GPU kernel的性能仅仅为CPU版本的kernel的 1/2 以下 ; 但当数据量达到千万级时，两者性能已经相当，GPU kernel要略好一些，可以预见，当数据量再往上升时，GPU kernel的性能优势将会越来越明显

### 2. Kernel_v2

采用多ThreadBlock的方式提升kernel性能，实现了及其显著的性能提升效果，数据量达到千万级时kernel_v2的性能是cpu版本的数十万倍，通过Nvidia Compute测试，v2相较于v1的性能提升巨大，计算吞吐量和内存吞吐量相对提升500%和600%
