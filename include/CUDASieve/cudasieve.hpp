/*

CUDASieveSieves.cuh

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/
#include <stdint.h>
#include <ctime>

#ifndef _CUDASIEVE
#define _CUDASIEVE

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

class CudaSieve {

  friend class KernelData;
  friend class PrimeList;
  friend class SmallSieve;
  friend class BigSieve;
  friend class PrimeOutList;

private:
  bool flags[32];
  uint64_t bottom = 0, top = (1u << 30), kernelBottom, smKernelTop, totBlocks, count = 0, * primeOut;
  uint16_t gpuNum = 0;
  uint32_t bigSieveBits, bigSieveKB = 1024, sieveBits, sieveKB = 16, primeListLength, * d_primeList;
  clock_t start_time;

  void setFlags();
  void setKernelParam();

  void allocateSieveOut();
  void allocateSieveOut(uint64_t size); // size in bytes

  void launchCtl();
  void smallSieveCtl();
  void bigSieveCtl();

public:
  uint32_t * sieveOut;

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

  uint64_t getBottom(){return bottom;}
  uint64_t getTop(){return top;}
  bool isFlag(uint8_t flagnum){return this -> flags[flagnum];}

  void displayRange();
  void displaySieveAttributes();

  void makePrimeList();
  void makePrimeList(uint32_t maxPrime);

  void countPrimes();
  uint64_t countPrimes(uint64_t top);
  uint64_t countPrimes(uint64_t bottom, uint64_t top);

  void printSieveOut();

  double elapsedTime();
};

#endif
