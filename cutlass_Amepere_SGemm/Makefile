BINARY_NAME = app
CUDA_PATH   = /usr/local/cuda
CC          = $(CUDA_PATH)/bin/nvcc
CFLAGS      = -O0 -g -G -std=c++17 -Xcompiler -rdynamic -DUSE_MULTI_GPU --expt-relaxed-constexpr
LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand
INCFLAGS    = -I$(CUDA_PATH)/include -Iinclude -I../cutlass/include -I../cutlass/tools/util/include -I../cutlass/tools/library/include
ARCHFLAGS   = -gencode arch=compute_80,code=sm_80   # Ampere通用


SRC = $(wildcard src/*.cu)
OBJ = $(SRC:src/%.cu=build/%.o)

all: outputs/$(BINARY_NAME)

outputs/$(BINARY_NAME): $(OBJ)
	@mkdir -p outputs
	$(CC) $(LDFLAGS) $^ -o $@

build/%.o: src/%.cu
	@mkdir -p build
	$(CC) -c $(CFLAGS) $(INCFLAGS) $(ARCHFLAGS) $< -o $@

clean:
	rm -rf build outputs