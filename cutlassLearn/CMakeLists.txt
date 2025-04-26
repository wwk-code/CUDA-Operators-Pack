cmake_minimum_required(VERSION 3.17...3.29)    

project(app LANGUAGES CXX CUDA)

# —— 1. 指定 C++ 与 CUDA 标准 ——#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加抑制警告
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20013>)

# —— 2. 导入 CUDA Toolkit ——#
find_package(CUDAToolkit REQUIRED)            # 导入 CUDA::cudart 等目标 :contentReference[oaicite:4]{index=4}

# —— 3. 添加可执行、包含目录等 ——#
add_executable(${PROJECT_NAME} src/main.cu)
target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # CUDA include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../cutlass/include
    ${CMAKE_SOURCE_DIR}/../cutlass/tools/util/include 
    ${CMAKE_SOURCE_DIR}/../cutlass/tools/library/include
)

# —— 4. 链接 CUDA 库 ——#
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    CUDA::cudart                             # 已由 find_package(CUDAToolkit) 导入 :contentReference[oaicite:5]{index=5}
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
)

# —— 5. 指定 GPU 架构 ——#
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_ARCHITECTURES 80                      # 对应 -gencode=compute_80,code=sm_80
)
