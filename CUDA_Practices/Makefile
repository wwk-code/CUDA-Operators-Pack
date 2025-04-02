BINARY_NAME = app
CUDA_PATH   = /usr/local/cuda-12.6
CC          = $(CUDA_PATH)/bin/nvcc
CFLAGS      = -O0 -g -G -Xcompiler -rdynamic -DUSE_MULTI_GPU
LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand
INCFLAGS    = -I$(CUDA_PATH)/include -Iinclude -I../cuda-samples/Common
ARCHFLAGS   = -gencode arch=compute_80,code=sm_80 \  # Ampere通用
              -gencode arch=compute_86,code=sm_86 \  # RTX 3090
              -gencode arch=compute_89,code=sm_89    # RTX 4060 Ti

SRC = $(wildcard src/*.cu)
OBJ = $(SRC:src/%.cu=build/%.o)

build: outputs/$(BINARY_NAME)

outputs/$(BINARY_NAME): $(OBJ)
	@mkdir -p outputs
	$(CC) $(LDFLAGS) $^ -o $@

build/%.o: src/%.cu
	@mkdir -p build
	$(CC) -c $(CFLAGS) $(INCFLAGS) $(ARCHFLAGS) $< -o $@

clean:
	rm -rf build outputs