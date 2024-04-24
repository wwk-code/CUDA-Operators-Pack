# cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
# project(top_k LANGUAGES CXX CUDA)

# # CUDA Setup
# enable_language(CUDA)
# find_package(CUDAToolkit REQUIRED)

# # Include directories
# include_directories(include ${CUDAToolkit_INCLUDE_DIRS})

# # Source files
# file(GLOB_RECURSE SOURCE_FILES 
#     src/*.cpp
#     src/*.cu
#     src/**/*.cpp
#     src/**/*.cu
# )

# # Create executable
# add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# # Link CUDA libraries
# target_link_libraries(${PROJECT_NAME} CUDA::cudart)

# # Set properties for CUDA
# set_target_properties(${PROJECT_NAME} PROPERTIES
#     CUDA_ARCHITECTURES "75"
#     CUDA_SEPARABLE_COMPILATION ON
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
# )

# # Set C and CUDA flags for debugging
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -arch=sm_75")

# # Ensure we use the C compiler for linking
# set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(top_k LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

include_directories(include ${CUDAToolkit_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCE_FILES 
    src/*.cpp
    src/*.cu
    src/**/*.cpp
    src/**/*.cu
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

# Apply CUDA compile options only to CUDA files
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -Xcompiler=-O0 -arch=sm_75>)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)



