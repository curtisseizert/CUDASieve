# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = compute_30
GPU_CODE = sm_30,sm_32,sm_35,sm_37,sm_50,sm_52,sm_53,sm_60,sm_61,sm_62
# Compilers to use
NVCC = $(CUDA_DIR)/bin/nvcc
CC = clang
# Flags for the host compiler
CCFLAGS = -c -O3 -std=c++11

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) -code=$(GPU_CODE) --ptxas-options=-dlcm=cg -lineinfo

INCLUDES = -I ./include/ -I ./src/ -I $(CUDA_DIR)/include/
CC_LIBS = -lm -lstdc++
NVCC_LIBS = -lcudart

CLI_SRC_DIR = src
SRC_DIR = src/CUDASieve
OBJ_DIR = obj

## Cannot use device.cu here because it is #include linked to global.cu!
## this is necessary because the nvcc linker sucks with device code!
NV_SRCS = src/CUDASieve/global.cu src/CUDASieve/launch.cu
_MAIN_OBJ = main.o
MAIN_OBJ = $(patsubst %,$(OBJ_DIR)/%,$(_MAIN_OBJ))
_OBJS = host.o cudasieve.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

MAIN = CUDASieve
CS_LIB = cudasieve.a

all: $(MAIN) $(CS_LIB)

$(MAIN): $(NV_SRCS) $(OBJS) $(MAIN_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@
	@echo  CUDASieve has been compiled

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<

$(OBJ_DIR)/%.o: $(CLI_SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<

$(CS_LIB): $(NV_SRCS) $(OBJS)
		$(NVCC) $(NVCC_FLAGS) -lib $(INCLUDES) $^ -o $@

clean:
	rm -f obj/*.o
