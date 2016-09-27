/*

CUDASieveLaunch.cuh

Host functions for CUDASieve which interface with the device
Curtis Seizert - cseizert@gmail.com

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <stdint.h>

#ifndef _CUDASIEVE_LAUNCH
#define _CUDASIEVE_LAUNCH

#ifndef PL_SIEVE_WORDS
  #define PL_SIEVE_WORDS 256
#endif
#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK 256 // changing this causes a segfault
#endif
#ifndef THREADS_PER_BLOCK_LG
  #define THREADS_PER_BLOCK_LG 256
#endif

class KernelData;
class PrimeList;
class SmallSieve;
class BigSieve;
class PrimeOutList;
class CudaSieve;

class PrimeOutList{
  friend class BigSieve;
  friend class KernelData;

private:
  uint32_t * d_histogram, *d_histogram_lg;
  uint32_t hist_size_lg, blocks;
  uint16_t threads;
  uint64_t * h_primeOut, * d_primeOut;
  uint64_t numGuess;
  void allocate();
  void fetch(BigSieve * sieve);
  void cleanupMin();
  void cleanupAllDevice();

public:
  uint64_t * getPrimeOut();
  void printPrimes();

  PrimeOutList(CudaSieve * sieve);
  ~PrimeOutList();
};

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
  void sievePrimeList();
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
  void displaySieveTime();
};

class BigSieve{
  friend class PrimeOutList;

private:
  cudaEvent_t start, stop, start1, stop1, start2, stop2;
  cudaStream_t stream[2];
  uint16_t log2bigSieveSpan;
  uint32_t blocksSm, blocksLg, primeListLength, bigSieveKB, bigSieveBits, sieveKB, * d_bigSieve, * d_primeList, * ptr32;
  uint64_t top, bottom, totIter, *ptr64;
  bool silent;

  uint32_t * d_next;
  uint16_t * d_away;

public:

  BigSieve(){}
  BigSieve(CudaSieve * sieve);
  ~BigSieve(){}

  void setParameters(CudaSieve * sieve); // this is only necessary if a CudaSieve was not specified on declaration;
  void allocate();
  void fillNextMult();

  void launchLoop(KernelData & kernelData);
  void launchLoopCopy(KernelData & kernelData, CudaSieve * sieve);
  void launchLoopPrimes(KernelData & kernelData, CudaSieve * sieve);
  void displayCount(KernelData & kernelData);

  void cleanUp();
};

class KernelData{
  friend class BigSieve;
  friend class SmallSieve;
  friend class PrimeOutList;
private:
  static volatile uint64_t * h_count, * h_blocksComplete;
  static volatile uint64_t * d_count, * d_blocksComplete;
public:
  uint64_t getCount(){return * h_count;}
  uint64_t getBlocks(){return * h_blocksComplete;}

  void displayProgress(CudaSieve * sieve);
  void displayProgress(uint64_t value, uint64_t totIter);

  static void allocate();

  KernelData(){};
  ~KernelData(){};
};


#endif
