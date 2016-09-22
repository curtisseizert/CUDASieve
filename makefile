CUDASieve: src/CUDASieveMain.cu src/CUDASieveGlobal.cu src/CUDASieveDevice.cu src/CUDASieveHost.cpp include/CUDASieveDevice.cuh include/CUDASieveGlobal.cuh include/CUDASieveHost.hpp
	/opt/cuda/bin/nvcc -ccbin /bin/g++-5 -std=c++11 --gpu-architecture=sm_61 -g -lineinfo -I /home/curtis/CUDASieve/src/ -I /home/curtis/CUDASieve/include/ -I /opt/cuda/targets/x86_64-linux/include/ -o CUDASieve src/CUDASieveMain.cu
utils: utils/readsieve.cpp utils/bitsievegen.cpp
	g++ -o readsieve utils/readsieve.cpp
	g++ -o bitsievegen utils/bitsievegen.cpp
install: bitsievegen CUDASieve
	cp bitsievegen /bin/
	cp CUDASieve /bin/
clean:
	rm include/*.gch
