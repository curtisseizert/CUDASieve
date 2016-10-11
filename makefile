##
## The only thing that should need to be changed is CUDA_DIR, but fewer gpu
## architectures specified in GPU_CODE will make a smaller executable.
## CSTest is a small utility to extensively test the output of cudasieve.
## see src/cstest.cpp for more details.
##

# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = compute_30
GPU_CODE = sm_30,sm_32,sm_35,sm_37,sm_50,sm_52,sm_53,sm_60,sm_61,sm_62

# Compilers to use
NVCC = $(CUDA_DIR)/bin/nvcc
CC = clang
# Flags for the host compiler
CCFLAGS = -O3 -std=c++11 -c

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) -code=$(GPU_CODE) --ptxas-options=-dlcm=cs -lineinfo

INCLUDES = -I ./include/ -I ./src/ -I $(CUDA_DIR)/include/
LIB_DIR = -L ./
CC_LIBS = -lm -lstdc++
NVCC_LIBS = -lcudart $(CC_LIBS)

CLI_SRC_DIR = src
SRC_DIR = src/CUDASieve
NV_SRCS = src/CUDASieve/launch.cu src/CUDASieve/global.cu src/CUDASieve/cudasieve.cu
OBJ_DIR = obj

## Cannot use device.cu here because it is #include linked to global.cu!
## this is necessary because the nvcc linker sucks with device code!
#NV_SRCS = src/CUDASieve/global.cu src/CUDASieve/launch.cu src/CUDASieve/device.cu
_MAIN_OBJ = main.o
MAIN_OBJ = $(patsubst %,$(OBJ_DIR)/%,$(_MAIN_OBJ))
_OBJS = host.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

MAIN = cudasieve
CS_LIB = lib$(MAIN).a

all: $(MAIN)

# Tack on a main() function for the CLI
$(MAIN): $(MAIN_OBJ) $(CS_LIB)
	$(NVCC) $(NVCC_FLAGS) $(CC_LIBS) $(LIB_DIR) -l$(MAIN) $< -o $@
	@echo
	@echo  CUDASieve has been compiled.  cudasieve --help gives a list of options.

# Linking to make a library
$(CS_LIB): $(OBJS) $(NV_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -lib $(INCLUDES) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<

$(OBJ_DIR)/%.o: $(CLI_SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<

## The cudasieve testing utility depends on boost, openMP and primesieve.
test: src/cstest.cpp $(CS_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIB_DIR) -O3 -Xcompiler -fopenmp -l$(MAIN) $(NVCC_LIBS) -lprimesieve $< -o cstest
	@echo CSTest has been compiled.  To test the output of cudasieve over random ranges:
	@echo
	@echo cstest
	@echo
	@echo With an argument, cstest performs a more complete test of a range of 2^30 starting
	@echo at that number, e.g.:
	@echo
	@echo cstest 1667640106059223296
	@echo
	@echo Please let me know if you find an error: cseizert@gmail.com.

clean:
	rm -f obj/*.o *.a cudasieve
