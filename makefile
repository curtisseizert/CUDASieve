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
WIGNORE = -Wno-return-stack-address

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) -code=$(GPU_CODE) --ptxas-options=-dlcm=cs -lineinfo

INCLUDES = -I ./include/ -I ./src/ -I $(CUDA_DIR)/include/
LIB_DIR = -L ./
CC_LIBS = -lm -lstdc++
NVCC_LIBS = -lcudart $(CC_LIBS)

CLI_SRC_DIR = src
SRC_DIR = src/CUDASieve
NV_SRCS = src/CUDASieve/launch.cu src/CUDASieve/global.cu
OBJ_DIR = obj

## Cannot use device.cu here because it is #include linked to global.cu!
## this is necessary because the nvcc linker sucks with device code!
#NV_SRCS = src/CUDASieve/global.cu src/CUDASieve/launch.cu src/CUDASieve/device.cu
_MAIN_OBJ = main.o
MAIN_OBJ = $(patsubst %,$(OBJ_DIR)/%,$(_MAIN_OBJ))
_OBJS = host.o cudasieve.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

MAIN = cudasieve
CS_LIB = lib$(MAIN).a

all: $(MAIN) test

# Tack on a main() function for the CLI
$(MAIN): $(MAIN_OBJ) $(CS_LIB)
	@$(NVCC) $(NVCC_FLAGS) $(CC_LIBS) $(LIB_DIR) -l$(MAIN) $< -o $@
	@echo ""
	@echo  "    CUDASieve has been compiled"
	@echo  "    cudasieve --help gives a list of options."
	@echo ""

# Linking to make a library
$(CS_LIB): $(OBJS) $(NV_SRCS)
	@$(NVCC) $(NVCC_FLAGS) -lib $(INCLUDES) $^ -o $@
	@echo "    CUDA    " $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(CC) $(CCFLAGS) $(INCLUDES) $(WIGNORE) -o $@ $<
	@echo "    CXX     " $@

$(OBJ_DIR)/%.o: $(CLI_SRC_DIR)/%.cpp
	@$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $<
	@echo "    CXX     " $@
## The cudasieve testing utility depends on boost, openMP and primesieve.
test: src/cstest.cpp $(CS_LIB)
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIB_DIR) -O3 -Xcompiler -fopenmp -l$(MAIN) $(NVCC_LIBS) -lprimesieve $< -o cstest
	@echo "    CXX    " $@
	@echo "    To run tests: ./cstest"

clean:
	rm -f obj/*.o *.a cudasieve
