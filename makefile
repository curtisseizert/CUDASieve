CUDADIR = /opt/cuda/
GPUARCH = sm_61

NVCC = $(CUDADIR)/bin/nvcc
CC = g++
CCFLAGS = -c -O2
NVCCFLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPUARCH)
INCLUDES = -I ./src/ -I ./include/ -I $(CUDADIR)/targets/x86_64-linux/include/
CSRCS = src/CUDASieve/main.cpp src/CUDASieve/host.cpp src/CUDASieve/cudasieve.cpp
NVSRCS = src/CUDASieve/global.cu src/CUDASieve/launch.cu
OBJS = main.o host.o cudasieve.o
MAIN = CUDASieve

all:    $(MAIN)
	@echo  CUDASieve has been compiled

$(MAIN): $(NVSRCS) $(CSRCS)
	$(CC) $(CCFLAGS) $(INCLUDES) $(CSRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(NVSRCS) $(OBJS) -o $(MAIN)
	rm $(OBJS)
