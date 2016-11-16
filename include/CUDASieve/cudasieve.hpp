/*

cudasieve.hpp

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
inline void safeFree(T * array) {if(array != NULL){free(array); array = NULL;}}

template <typename T>
inline void safeCudaFree(T * array) {if(array != NULL){cudaFree(array); array = NULL;}}

template <typename T>
inline void safeCudaFreeHost(T * array) {if(array != NULL){cudaFreeHost(array); array = NULL;}}

template <typename T>
inline T * safeCudaMalloc(T * d_a, size_t size)
{
  if(d_a == NULL){
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
  if(h_a == NULL){
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
  uint32_t * h_primeOut32 = NULL, * d_primeOut32 = NULL;

  uint16_t gpuNum = 0;            // used with cudaSetDevice
  uint32_t sieveBits, sieveKB = 16; // small sieve parameters, to be deleted from this class
  uint32_t primeListLength, * d_primeList = NULL, maxPrime_ = 0; // parameters and device pointer
                                                                 //for list of sieving primes
  uint64_t itop, irange, ibottom; // these are safeguard parameters for segment functions
  uint64_t sieveOutBytes = 0;     // used with getBitSieve for debugging

  uint32_t * sieveOut = NULL, * d_sieveOut = NULL;

  BigSieve bigsieve;
  SmallSieve smallsieve;
  KernelData kerneldata;

  void makePrimeList(uint32_t maxPrime);

  void setFlags();
  void setKernelParam();
  void checkRange();

  void displayRange();
  void displaySieveAttributes();

  void run();
  void launchCtl();
  void phiCtl(uint32_t a);

  void getPrimes32();

  void copyAndPrint();

  uint32_t * h_getSieveOut();
  uint32_t * d_getSieveOut();

  void reset();

public:
  CudaSieve();
  CudaSieve(uint16_t gpuNum);
  CudaSieve(uint64_t bottom, uint64_t top, uint64_t range = 0, bool devOnly = 1);
  ~CudaSieve();

  inline uint64_t getBottom(){return bottom;}
  inline uint64_t getTop(){return top;}

  inline void setSieveKB(uint32_t sieveKB){this -> sieveKB = sieveKB;}
  inline void setBigSieveKB(uint32_t bigSieveKB){this -> bigsieve.bigSieveKB = bigSieveKB;}
  inline void setMaxPrime(uint32_t maxPrime){this -> maxPrime_ = maxPrime;}
  inline void setFlagOn(uint8_t flagnum){this -> flags[flagnum] = 1;}
  inline void setFlagOff(uint8_t flagnum){this -> flags[flagnum] = 0;}
  inline void setGpu(uint16_t gpuNum){this -> gpuNum = gpuNum; cudaSetDevice(gpuNum);}

  inline bool isFlag(uint8_t flagnum){return this -> flags[flagnum];}

  static void listDevices();
  char * getCurrentDeviceName();

  void CLIPrimes(); // used by the CLI where options are set by host::parseOptions()

  static uint64_t countPrimes(uint64_t top, uint16_t gpuNum = 0);
  static uint64_t countPrimes(uint64_t bottom, uint64_t top, uint16_t gpuNum = 0);

  static uint64_t countPhi(uint64_t top, uint32_t a, uint16_t gpuNum = 0);
  static uint64_t countPhi(uint64_t bottom, uint64_t top, uint32_t a, uint16_t gpuNum = 0);

  static uint64_t * getHostPrimes(uint64_t bottom, uint64_t top, size_t & size, uint16_t gpuNum = 0);
  static std::vector<uint64_t> getHostPrimesVector(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum = 0);
  static uint64_t * getDevicePrimes(uint64_t bottom, uint64_t top, size_t & size, uint16_t gpuNum = 0);

  static uint32_t * getHostPrimes32(uint64_t bottom, uint64_t top, size_t & size, uint16_t gpuNum = 0);
  static uint32_t * getDevicePrimes32(uint64_t bottom, uint64_t top, size_t & size, uint16_t gpuNum = 0);


  uint64_t countPrimesSegment(uint64_t bottom, uint64_t top, uint16_t gpuNum = 0);
  uint64_t * getHostPrimesSegment(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum);
  uint64_t * getDevicePrimesSegment(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum);

  static uint32_t * genBitSieve(uint64_t bottom, uint64_t top, uint16_t gpuNum = 0);
  static uint32_t * genDeviceBitSieve(uint64_t bottom, uint64_t top, uint16_t gpuNum = 0);

  void printPrimes(uint64_t * h_primeOut);

  template <typename T>
  inline void allocateSieveOut(T bytes = 0)
  {
    if(bytes != 0) sieveOutBytes = bytes;
    if(sieveOut == NULL) sieveOut = (uint32_t *)malloc(sieveOutBytes);
  }

  template <typename T>
  inline void allocateDeviceSieveOut(T bytes = 0)
  {
    if(bytes != 0) sieveOutBytes = bytes;
    d_sieveOut = safeCudaMalloc(d_sieveOut, (size_t) bytes);
  }
};

#endif
