# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = sm_50
# Compilers to use
NVCC = $(CUDA_DIR)/bin/nvcc
CC = clang
# Flags for the host compiler
CCFLAGS = -c -O3 -std=c++11

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) --ptxas-options=-dlcm=cg
NVCC_PROFILE_FLAGS = -lineinfo

INCLUDES = -I ./include/ -I ./src/ -I $(CUDA_DIR)/targets/x86_64-linux/include/
CC_LIBS = -lm -lstdc++
NVCC_LIBS = -lcudart

CLI_SRC_DIR = src
SRC_DIR = src/CUDASieve
OBJ_DIR = obj

## Cannot use device.cu here because it is #include linked to global.cu!
## this is necessary because the nvcc linker sucks with device code!
NV_SRCS = src/CUDASieve/global.cu src/CUDASieve/launch.cu
#
_OBJS = main.o host.o cudasieve.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))
#
MAIN = CUDASieve

all: $(MAIN)

$(MAIN): $(NV_SRCS) $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@
	@echo  CUDASieve has been compiled

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<

$(OBJ_DIR)/%.o: $(CLI_SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f obj/*.o
