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
#include <iostream>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDASieve/launch.cuh"
#include <vector>


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

namespace host{
void parseOptions(int argc, char* argv[], CudaSieve * sieve); // for CLI
void displayAttributes(const BigSieve & bigsieve);
void displayAttributes(CudaSieve & sieve);
}

template <typename T>
inline void safeFree(T * array) {if(array){free(array); array = NULL;}}

template <typename T>
inline void safeCudaFree(T * array) {if(array){cudaFree(array); array = NULL;}}

template <typename T>
inline void safeCudaFreeHost(T * array) {if(array != NULL){cudaFreeHost(array); array = NULL;}}

template <typename T>
inline T * safeCudaMalloc(T * d_a, size_t size)
{
  if(!d_a){
    if(cudaMalloc(&d_a, size) != cudaSuccess){
      std::cerr << "CUDA device memory allocation error: CUDA API error " << cudaMalloc(&d_a, size) << std::endl;
      std::cerr << "for attempted allocation of size " << size << " at " << &d_a << std::endl;
      exit(1);
    }
  }
  return (T *) d_a;
}

template <typename T>
inline T * safeCudaMallocHost(T * h_a, size_t size)
{
  if(!h_a){
    if(cudaMallocHost(&h_a, size) != cudaSuccess){
      std::cerr << "CUDA host memory allocation error: CUDA API error " << cudaMalloc(&h_a, size) << std::endl;
      exit(1);
    }
  }
  return (T *) h_a;
}

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
  bool flags[32] = {0};
  uint64_t bottom = 0, top = (1u << 30), count = 0;  // sieve parameters
  uint64_t * h_primeOut = NULL, * d_primeOut = NULL; // pointers to output arrays for prime generation

  uint16_t gpuNum = 0;            // used with cudaSetDevice
  uint32_t sieveBits, sieveKB = 16; // small sieve parameters, to be deleted from this class
  uint32_t primeListLength, * d_primeList = NULL, maxPrime_ = 0; // parameters and device pointer
                                                                 //for list of sieving primes
  clock_t start_time;
  uint64_t itop, irange, ibottom; // these are safeguard parameters for segment functions
  uint64_t sieveOutSize = 0;      // used with getBitSieve for debugging

  BigSieve bigsieve;
  SmallSieve smallsieve;

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
  void launchCtl(uint32_t maxPrime);

  void copyAndPrint();

  void reset();

public:
  uint32_t * sieveOut = NULL;             // used with getBitSieve for debugging - holds
                                          // a concatenation of all sieve segments

  void setSieveKB(uint32_t sieveKB);
  void setBigSieveKB(uint32_t bigSieveKB);
  void setGpuNum(uint16_t gpuNum);
  void setMaxPrime(uint32_t maxPrime);
  void setFlagOn(uint8_t flagnum){this -> flags[flagnum] = 1;}
  void setFlagOff(uint8_t flagnum){this -> flags[flagnum] = 0;}

  CudaSieve();
  CudaSieve(uint64_t bottom, uint64_t top, uint64_t range);
  ~CudaSieve();

  bool isFlag(uint8_t flagnum){return this -> flags[flagnum];}

  void CLIPrimes(); // used by the CLI where options are set by host::parseOptions()

  static uint64_t countPrimes(uint64_t top, uint16_t gpuNum = 0);
  static uint64_t countPrimes(uint64_t bottom, uint64_t top, uint16_t gpuNum = 0);

  static uint64_t * getHostPrimes(uint64_t bottom, uint64_t top, size_t & size, uint16_t gpuNum = 0);
  static std::vector<uint64_t> getHostPrimesVector(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum = 0);
  static uint64_t * getDevicePrimes(uint64_t bottom, uint64_t top, size_t & size, uint16_t gpuNum = 0);

  uint64_t countPrimesSegment(uint64_t bottom, uint64_t top, uint16_t gpuNum = 0);
  uint64_t * getHostPrimesSegment(uint64_t bottom, size_t & count, uint16_t gpuNum = 0);
  uint64_t * getDevicePrimesSegment(uint64_t bottom, size_t & count, uint16_t gpuNum = 0);

  uint32_t * getBitSieve();

  void printPrimes(uint64_t * h_primeOut);
  double elapsedTime();
};

#endif
