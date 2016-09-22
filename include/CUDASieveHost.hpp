/*

CUDASieveHost.hpp

Host functions for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#pragma once

#define PL_SIEVE_WORDS 256
#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_LG 256

#include "CUDASieveDevice.cu"
#include "CUDASieveGlobal.cu"


#pragma once

class KernelData;
class CudaSieve;
class PrimeList;
class SmallSieve;
class BigSieve;

class PrimeList{

private:
  cudaEvent_t start, stop;
  uint32_t * h_primeListLength, * d_histogram, * d_histogram_lg, * d_primeListLength;
  uint32_t hist_size_lg, piHighGuess, PL_Max, maxPrime, blocks;
  uint16_t threads;
  uint32_t * d_primeList;

public:
  uint32_t getBlocks(){return blocks;}
  uint16_t getThreads(){return threads;}
  uint32_t * getPtr(){return d_primeList;}
  void sievePrimeList(CudaSieve * sieve);
  uint32_t getPrimeListLength();
  void allocate();

  PrimeList(uint32_t maxPrime);
  ~PrimeList();

  void displayTime();
  void cleanUp();
};

class SmallSieve{
private:
  cudaEvent_t start, stop;

public:

  SmallSieve(CudaSieve * sieve);
  ~SmallSieve(){};
  void launch(KernelData & kernelData, CudaSieve * sieve);
  void displaySieveTime(CudaSieve * sieve);
};

class BigSieve{

private:
  cudaEvent_t start, stop, start1, stop1, start2, stop2;
  cudaStream_t stream[2];
  uint32_t blocksSm, blocksLg, primeListLength, bigSieveKB, bigSieveBits, sieveKB, * d_bigSieve, * d_primeList, * ptr;
  uint64_t top, bottom, totIter, * d_nextMult;
  bool silent;

public:

  BigSieve(){}
  BigSieve(CudaSieve * sieve);
  ~BigSieve(){}

  void setParameters(CudaSieve * sieve); // this is only necessary if a CudaSieve was not specified on declaration;
  void allocate();
  void fillNextMult();

  void launchLoop(KernelData & kernelData);
  void launchLoop(KernelData & kernelData, CudaSieve * sieve);
  void displayCount(KernelData & kernelData);

  void cleanUp();
};

class KernelData{
  friend class BigSieve;
  friend class SmallSieve;
private:
  volatile uint64_t * h_count, * h_blocksComplete;
  volatile uint64_t * d_count, * d_blocksComplete;
public:
  uint64_t getCount(){return * h_count;}
  uint64_t getBlocks(){return * h_blocksComplete;}

  void displayProgress(CudaSieve * sieve);
  void displayProgress(uint64_t value, uint64_t totIter);

  KernelData();
  ~KernelData(){};
};

class CudaSieve {

  friend class KernelData;
  friend class PrimeList;
  friend class SmallSieve;
  friend class BigSieve;

private:
  bool flags[32];
  uint64_t bottom = 0, top = (1u << 30), kernelBottom, smKernelTop, totBlocks, count = 0;
  uint16_t gpuNum = 0;
  uint32_t sieveOutBits, bigSieveBits, bigSieveKB = 1024, sieveBits, sieveKB = 16, primeListLength, * d_primeList, * sieveOut, * ptr;
  clock_t start_time;


  void setFlags();
  void setKernelParam();

  void launchControl();
  void smallSieveCtl(KernelData & kernelData);
  void bigSieveCtl(KernelData & kernelData);

public:
  CudaSieve();
  ~CudaSieve(){};

  void setDefaults();
  void setSieveOutBits();
  void setSieveBitSize();

  void setTop(uint64_t top);
  void setBottom(uint64_t bottom);
  void setSieveKB(uint32_t sieveKB);
  void setBigSieveKB(uint32_t bigSieveKB);
  void setGpuNum(uint16_t gpuNum);

  void setFlagOn(uint8_t flagnum){this -> flags[flagnum] = 1;}
  void setFlagOff(uint8_t flagnum){this -> flags[flagnum] = 0;}

  bool isFlag(uint8_t flagnum){return this -> flags[flagnum];}

  void displayRange();
  void displaySieveAttributes();

  void makePrimeList();
  void makePrimeList(uint32_t maxPrime);

  void countPrimes();
  uint64_t countPrimes(uint64_t top);

  double elapsedTime();
};


namespace host {

    void help();
    uint64_t echo(char * argv);
    void parseOptions(int argc, char* argv[], CudaSieve * sieve);
    void makePrimeList(uint32_t *& d_primeList);
}
