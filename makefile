NVCC = /opt/cuda/bin/nvcc
CC = g++
CCFLAGS = -c -O2 -Wa,-mtune=corei7
NVCCFLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=sm_61
LIBS =
INCLUDES = -I /home/curtis/CUDASieve/src/ -I /home/curtis/CUDASieve/include/ -I /opt/cuda/targets/x86_64-linux/include/
CSRCS = src/CUDASieve/main.cpp src/CUDASieve/host.cpp src/CUDASieve/cudasieve.cpp
NVSRCS = src/CUDASieve/global.cu src/CUDASieve/launch.cu
OBJS = main.o host.o cudasieve.o
MAIN = CUDASieve

all:    $(MAIN)
	@echo  CUDASieve has been compiled

$(MAIN): $(NVSRCS) $(CSRCS)
	$(CC) $(CCFLAGS) $(INCLUDES) $(CSRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBS) $(NVSRCS) $(OBJS) -o $(MAIN)
	rm $(OBJS)
