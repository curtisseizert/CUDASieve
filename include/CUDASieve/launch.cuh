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
#include "host.hpp"

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
class KernelTime;


class KernelTime{
  friend class BigSieve;
  friend class SmallSieve;
private:
  cudaEvent_t start_, stop_;
public:
  KernelTime();
  ~KernelTime();

  void displayTime();
  inline void start();
  inline void stop();
  float get_ms();
};

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
  void fetch(BigSieve & sieve);
  void cleanupAll();
  void cleanupAllDevice();

public:
  uint64_t * getPrimeOut();
  void printPrimes();

  PrimeOutList(CudaSieve & sieve);
  ~PrimeOutList();
};

class PrimeList{

private:
  KernelTime timer;
  uint32_t * h_primeListLength, * d_histogram, * d_histogram_lg, * d_primeListLength;
  uint32_t hist_size_lg, piHighGuess, PL_Max, maxPrime, blocks;
  uint16_t threads;
  uint32_t * d_primeList;

  uint32_t * getPtr(){return d_primeList;}
  void sievePrimeList();
  void allocate();

  PrimeList(uint32_t maxPrime);
public:

  ~PrimeList();
  static uint32_t * getSievingPrimes(uint32_t maxPrime, uint32_t & primeListLength, bool silent);

};

class SmallSieve{
private:
  KernelTime timer;
  float time_ms;
  SmallSieve(){};
  ~SmallSieve(){};
  void launch(CudaSieve & sieve);
  void displaySieveTime();
public:
  static void run(CudaSieve & sieve);
};

class BigSieve{
  friend class PrimeOutList;
  friend class CudaSieve;
  friend void host::displayAttributes(const BigSieve & bigsieve);

private:
  KernelTime timer;
  cudaStream_t stream[2];
  uint16_t log2bigSieveSpan;
  uint32_t blocksSm, blocksLg, primeListLength, bigSieveKB, bigSieveBits, sieveKB, * d_bigSieve, * d_primeList, * ptr32;
  uint64_t top, bottom, totIter, *ptr64;
  bool silent, noMemcpy;
  float time_ms;

  uint32_t * d_next;
  uint16_t * d_away;

  BigSieve(CudaSieve & sieve);
  ~BigSieve();

  void setParameters(CudaSieve & sieve);
  void allocate();
  void fillNextMult();

  void launchLoop();
  void launchLoopCopy(CudaSieve & sieve);
  void launchLoopPrimes(CudaSieve & sieve);

public:
  static void run(CudaSieve & sieve);

};

class KernelData{
  friend class BigSieve;
  friend class SmallSieve;
  friend class PrimeOutList;
  friend class CudaSieve;
private:
  static volatile uint64_t * h_count, * h_blocksComplete;
  static volatile uint64_t * d_count, * d_blocksComplete;
public:
  static uint64_t getCount(){return * h_count;}
  static uint64_t getBlocks(){return * h_blocksComplete;}

  static void displayProgress(CudaSieve & sieve);
  static void displayProgress(uint64_t value, uint64_t totIter);

  static void allocate();

  KernelData(){};
  ~KernelData(){};
};


#endif
