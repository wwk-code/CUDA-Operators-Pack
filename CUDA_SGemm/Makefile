################################################################# For Debug  #################################################################

# BINARY_NAME = app
# CUDA_PATH   = /usr/local/cuda
# CC          = $(CUDA_PATH)/bin/nvcc
# CFLAGS      = -O3 -g -G -Xcompiler -rdynamic -DUSE_MULTI_GPU
# LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand
# INCFLAGS    = -I$(CUDA_PATH)/include -Iinclude -I../cuda-samples/Common
# ARCHFLAGS   = -gencode arch=compute_80,code=sm_80   # Ampere通用


# SRC = $(wildcard src/*.cu)
# OBJ = $(SRC:src/%.cu=build/%.o)

# all: outputs/$(BINARY_NAME)

# outputs/$(BINARY_NAME): $(OBJ)
# 	@mkdir -p outputs
# 	$(CC) $(LDFLAGS) $^ -o $@

# build/%.o: src/%.cu
# 	@mkdir -p build
# 	$(CC) -c $(CFLAGS) $(INCFLAGS) $(ARCHFLAGS) $< -o $@

# clean:
# 	rm -rf build outputs





################################################################# For optimal performance  #################################################################

BINARY_NAME = app
CUDA_PATH   = /usr/local/cuda
CC          = $(CUDA_PATH)/bin/nvcc

# 编译选项
CFLAGS      = -O3 -Xcompiler -fno-strict-aliasing -DUSE_MULTI_GPU
PTXASFLAGS  = -Xptxas -O3,-v
FASTMATH    = -use_fast_math
ARCHFLAGS   = -gencode arch=compute_80,code=sm_80

INCFLAGS    = -I$(CUDA_PATH)/include -Iinclude -I../cuda-samples/Common
LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand

SRC = $(wildcard src/*.cu)
OBJ = $(SRC:src/%.cu=build/%.o)

all: outputs/$(BINARY_NAME)

outputs/$(BINARY_NAME): $(OBJ)
	@mkdir -p outputs
	$(CC) $(CFLAGS) $(PTXASFLAGS) $(FASTMATH) $(ARCHFLAGS) $(LDFLAGS) $^ -o $@

build/%.o: src/%.cu
	@mkdir -p build
	$(CC) -c $(CFLAGS) $(PTXASFLAGS) $(FASTMATH) $(ARCHFLAGS) $(INCFLAGS) $< -o $@

clean:
	rm -rf build outputs

