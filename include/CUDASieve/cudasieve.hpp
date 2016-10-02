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
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "host.hpp"


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

inline void safeFree(bool * array) {if(array != NULL){free(array); array = NULL;}}
inline void safeFree(uint32_t * array) {if(array != NULL){free(array); array = NULL;}}
inline void safeFree(uint64_t * array) {if(array != NULL){free(array); array = NULL;}}
inline void safeCudaFree(uint16_t * array) {if(array != NULL){cudaFree(array); array = NULL;}}
inline void safeCudaFree(uint32_t * array) {if(array != NULL){cudaFree(array); array = NULL;}}
inline void safeCudaFree(uint64_t * array) {if(array != NULL){cudaFree(array); array = NULL;}}
inline void safeCudaFreeHost(uint32_t * array) {if(array != NULL){cudaFreeHost(array); array = NULL;}}
inline void safeCudaFreeHost(uint64_t * array) {if(array != NULL){cudaFreeHost(array); array = NULL;}}

class CudaSieve {

  friend class KernelData;
  friend class PrimeList;
  friend class SmallSieve;
  friend class BigSieve;
  friend class PrimeOutList;
  friend class Debug;
  friend void host::displayAttributes(CudaSieve & sieve);
  friend void host::parseOptions(int argc, char* argv[], CudaSieve * sieve);

private:
  bool flags[32];
  uint64_t bottom = 0, top = (1u << 30), kernelBottom, smKernelTop, totBlocks, count = 0, * h_primeOut, * d_primeOut;
  uint16_t gpuNum = 0;
  uint32_t bigSieveBits, bigSieveKB = 1024, sieveBits, sieveKB = 16, primeListLength, * d_primeList;
  clock_t start_time;

  void setTop(uint64_t top);
  void setBottom(uint64_t bottom);
  void setSieveKB(uint32_t sieveKB);
  void setBigSieveKB(uint32_t bigSieveKB);
  void setGpuNum(uint16_t gpuNum);
  void setFlagOn(uint8_t flagnum){this -> flags[flagnum] = 1;}
  void setFlagOff(uint8_t flagnum){this -> flags[flagnum] = 0;}

  uint64_t getBottom(){return bottom;}
  uint64_t getTop(){return top;}

  void makePrimeList(uint32_t maxPrime);

  void setFlags();
  void setKernelParam();
  void checkRange();

  void allocateSieveOut();
  void allocateSieveOut(uint64_t size); // size in bytes

  void displayRange();
  void displaySieveAttributes();

  void run();
  void launchCtl();

  void reset();

public:
  uint32_t * sieveOut;

  CudaSieve();
  ~CudaSieve();

  bool isFlag(uint8_t flagnum){return this -> flags[flagnum];}

  void CLIPrimes(); // used by the CLI where options are set by host::parseOptions()
  uint64_t countPrimes(uint64_t top);
  uint64_t countPrimes(uint64_t bottom, uint64_t top);

  uint64_t * getHostPrimes(uint64_t bottom, uint64_t top, size_t & size);
  uint64_t * getDevicePrimes(uint64_t bottom, uint64_t top, size_t & size);

  double elapsedTime();
};

#endif
