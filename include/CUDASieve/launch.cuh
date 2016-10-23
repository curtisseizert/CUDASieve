/*

launch.cuh

Host functions for CUDASieve which interface with the device
Curtis Seizert  <cseizert@gmail.com>

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/

#include <cuda.h>
#include <cuda_runtime.h>
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


class PrimeList;
class SmallSieve;
class BigSieve;
class PrimeOutList;
class CudaSieve;
class KernelTime;


class PrimeOutList{ // needs someone else's containers to put primes in.  Handles allocation.
  friend class BigSieve;
  friend class SmallSieve;
  friend class KernelData;

private:
  uint32_t * d_histogram = NULL, *d_histogram_lg = NULL;
  uint32_t hist_size_lg, blocks;
  uint16_t threads;
  void allocateDevice();
  void fetch(BigSieve & bigsieve, CudaSieve & sieve);
  void fetchPartial(BigSieve & sieve, uint64_t * d_primeOut);
  void fetch();

  void cleanupAll();
  void cleanupAllDevice();

public:
  PrimeOutList(CudaSieve & sieve);
  ~PrimeOutList();
};

class PrimeList{

private:
  KernelTime timer;
  uint32_t primeListLength, * d_histogram = NULL, * d_histogram_lg = NULL, * d_primeListLength = NULL;
  uint32_t hist_size_lg, piHighGuess, PL_Max, maxPrime, blocks, * d_primeList = NULL, * d_bigSieve = NULL, bigSieveKB = 1024;
  uint16_t threads,sieveKB = 16;

  KernelData kerneldata;

  uint32_t * getPtr(){return d_primeList;}
  void sievePrimeList();
  void iterSieve();
  void allocate();

  PrimeList(uint32_t maxPrime);
public:

  ~PrimeList();
  static uint32_t * getSievingPrimes(uint32_t maxPrime, uint32_t & primeListLength, bool silent = 1);

};

class SmallSieve{
  friend class CudaSieve;
  friend void host::displayAttributes(CudaSieve & sieve);
private:
  cudaStream_t stream[3];
  KernelTime timer;
  float time_ms;
  uint64_t totBlocks, kernelBottom, top;

  SmallSieve(){};
  ~SmallSieve(){};
  void count(CudaSieve & sieve);
  void copy(CudaSieve & sieve);
  void displaySieveTime();
  void createStreams();
public:
  static void run(CudaSieve & sieve);
};

class BigSieve{
  friend class PrimeOutList;
  friend class CudaSieve;
  friend class Debug;
  friend void host::displayAttributes(const BigSieve & bigsieve);

private:
  KernelTime timer;
  cudaStream_t stream[2];
  uint16_t log2bigSieveSpan;
  uint32_t blocksSm, blocksLg, primeListLength, bigSieveKB = 1024, bigSieveBits, sieveKB;
  uint32_t * d_bigSieve = NULL, * d_primeList = NULL, * ptr32 = NULL;
  uint64_t top, bottom, cutoff, totIter, *ptr64 = NULL;
  bool silent, partial, noPrint;
  float time_ms;

  uint32_t * d_next = NULL;
  uint16_t * d_away = NULL;

  BigSieve(CudaSieve & sieve);
  BigSieve() {}
  ~BigSieve();

  void setParameters(CudaSieve & sieve);
  void allocate();
  void fillNextMult();

  void launchLoop(CudaSieve & sieve);
  void launchLoopCopy(CudaSieve & sieve);
  void launchLoopPrimes(CudaSieve & sieve);
  void launchLoopPrimesSmall(CudaSieve & sieve);

  void countPartialTop(CudaSieve & sieve);

public:
  static void run(CudaSieve & sieve);

};




#endif
