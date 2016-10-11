/*

CUDASieveHost.cpp

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

*/

#include "CUDASieve/launch.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/host.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cinttypes>

#include <thrust/copy.h>
#include <thrust/fill.h>

CudaSieve::CudaSieve()
{
  start_time = clock();
  KernelData::allocate();
}

CudaSieve::CudaSieve(uint64_t bottom, uint64_t top, uint64_t range) // used for getting necessary data up front
{                                                                       // for repetitive counts etc.
  start_time = clock();
  KernelData::allocate();

  flags[30] = 1;
  this->top = top;
  itop = top;
  this->bottom = bottom;
  ibottom = bottom;
  irange = range;

  setKernelParam();

  d_primeList = PrimeList::getSievingPrimes(maxPrime_, primeListLength, flags[30]);
  if(top > (1ul << 40)){
    bigsieve.setParameters(*this);
    bigsieve.allocate();
    bigsieve.fillNextMult();
  }
  if(range != 0){
    uint64_t numGuess = (uint64_t) ((bottom+range)/log((bottom+range))*(1+1.2762/log((bottom+range)))) -
      ((bottom/log(bottom))*(1+1.2762/log(bottom)));
    h_primeOut = safeCudaMallocHost(h_primeOut, numGuess*sizeof(uint64_t));
    d_primeOut =  safeCudaMalloc(d_primeOut, numGuess*sizeof(uint64_t));
  }
}

CudaSieve::~CudaSieve()
{
  safeCudaFreeHost(h_primeOut);
  safeCudaFree(d_primeOut);
  safeCudaFree(d_primeList);
  KernelData::deallocate();
}

void CudaSieve::reset()
{
  safeCudaFreeHost(h_primeOut);
  safeCudaFree(d_primeOut);
  *KernelData::h_count = 0;
  start_time = clock();
}

void CudaSieve::setSieveKB(uint32_t sieveKB){this -> sieveKB = sieveKB;}
void CudaSieve::setBigSieveKB(uint32_t bigSieveKB){this -> bigsieve.bigSieveKB = bigSieveKB;}
void CudaSieve::setGpuNum(uint16_t gpuNum){this -> gpuNum = gpuNum;}
void CudaSieve::setMaxPrime(uint32_t maxPrime){this -> maxPrime_ = maxPrime;}

void CudaSieve::allocateSieveOut(uint64_t size){sieveOut = new uint32_t[size/sizeof(uint32_t)];}

void CudaSieve::allocateSieveOut()
{
  sieveOutSize = (top-bottom)/2;
  sieveOut = new uint32_t[sieveOutSize/sizeof(uint32_t)];
}

inline void CudaSieve::setKernelParam()
{
  if(top > 1ull << 63 && !flags[18])  bigsieve.bigSieveKB = 1u << 12; // bigger sieve size is more efficient above 2^63 (2^12 kb vs 2^10 kb)
  if(top < 1u << 23)                  sieveKB = 2;                    // smaller sieve size is more efficient for very small numbers (< 2^23)
  if(maxPrime_ == 0)                  maxPrime_ = (uint32_t) sqrt(top); // maximum sieving prime is top^0.5

  bigsieve.bigSieveBits = bigsieve.bigSieveKB << 13;
  sieveBits = sieveKB << 13;
  uint64_t smTop = std::min((unsigned long long) top, 1ull << 40);
  smallsieve.kernelBottom = bottom - bottom% (2 *  sieveBits);
  smallsieve.totBlocks = (smTop - smallsieve.kernelBottom) / (2 *  sieveBits);
  smallsieve.top = smallsieve.kernelBottom + (smallsieve.totBlocks * sieveBits * 2);

  cudaSetDevice(gpuNum);
  checkRange();
  setFlags();
}

inline void CudaSieve::checkRange()
{
  if(bottom > top)
    {std::cerr << "CUDASieve Error: the bottom of the range must be smaller than the top." << std::endl; exit(1);}
  if(top < 128)
    {std::cerr << "CUDASieve Error: the top of the range must be above 128." << std::endl; exit(1);}
  if((unsigned long long)top > 18446744056529682432ull) // 2^64-2^35
    {std::cerr << "CUDASieve Error: top above supported range (max is 2^64-2^35)." << std::endl; exit(1);}
  if(flags[0] && bottom%64 != 0)
    {std::cerr << "CUDASieve Error: bottom of range must be a multiple of 64 when copying primes." << std::endl; exit(1);}
  if(flags[0] && (bottom-top)%(2 * sieveBits) != 0)
    {std::cerr << "CUDASieve Error: range must be a multiple of sieve size when copying primes." << std::endl; exit(1);}
}

void CudaSieve::setFlags()
{
  if(top > (1ull << 40)) flags[1] = 1;
  if(bottom >= (1ull << 40)) flags[2] = 1;
  if(smallsieve.kernelBottom != bottom) flags[3] = 1;
  if(std::min((unsigned long long) top, 1ull << 40) != smallsieve.top) flags[4] = 1;
  if((bottom %(sieveBits*2) !=0)) flags[5] = 1;
}

inline void CudaSieve::displayRange()
{
  std::cout << "\n" << "Counting primes from " << bottom << " to " << top << std::endl;
}

double CudaSieve::elapsedTime()
{
  return (clock() - start_time)/((double) CLOCKS_PER_SEC);
}

inline void CudaSieve::launchCtl()
{
  uint64_t numGuess = (uint64_t) (top/log(top))*(1+1.32/log(top)) -
  ((bottom/log(bottom))*(1+1.32/log(bottom)));

  if(flags[0]){
    d_primeOut =  safeCudaMalloc(d_primeOut, numGuess*sizeof(uint64_t));
    cudaMemset(d_primeOut, 0, numGuess*sizeof(uint64_t));
  }

  setKernelParam();
  d_primeList = PrimeList::getSievingPrimes(maxPrime_, primeListLength, flags[30]);

  if(!flags[30] && !flags[0])     host::displayAttributes(*this);
  if(!flags[2]  && !flags[0])     SmallSieve::run(*this);
  if(flags[1]   || flags[0])      BigSieve::run(*this);
  count = KernelData::getCount();
}

void CudaSieve::copyAndPrint()
{
  h_primeOut = safeCudaMallocHost(h_primeOut, count*sizeof(uint64_t));
  cudaMemcpy(h_primeOut, d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  for(uint64_t i = 0; i < count; i++) printf("%" PRIu64 "\n", h_primeOut[i]);
}

void CudaSieve::CLIPrimes()
{
  if(!flags[30])  displayRange();

  launchCtl();

  if(flags[0])                 copyAndPrint();
  if(flags[30] && !flags[0])   std::cout << count << std::endl;
}

uint64_t CudaSieve::countPrimesSegment(uint64_t bottom, uint64_t top, uint16_t gpuNum) // add range checks
{
  this->bottom = bottom;
  this->top = top;
  this->gpuNum = gpuNum;

  *KernelData::h_count = 0;
  flags[30] = 1;
  setKernelParam();

  if(!flags[2])   smallsieve.count(*this);
  if(flags[1])    bigsieve.launchLoop(bottom, top);

  cudaDeviceSynchronize();
  return *KernelData::h_count;
}

uint64_t * CudaSieve::getHostPrimesSegment(uint64_t bottom, size_t & count, uint16_t gpuNum)
{
  if(bottom < ibottom || bottom + irange > itop) {count = 0; return NULL;} // make sure the range provided is in bounds

  this->bottom = bottom;
  top = bottom + irange;
  this->gpuNum = gpuNum;

  flags[30] = 1;
  flags[0] = 1;
  flags[29] = 1;

  *KernelData::h_count = 0;
  setKernelParam();

  bigsieve.setParameters(*this);
  bigsieve.launchLoopPrimes(*this);

  cudaDeviceSynchronize();
  count = *KernelData::h_count;

  cudaMemcpy(h_primeOut, d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  return h_primeOut;
}

uint64_t * CudaSieve::getDevicePrimesSegment(uint64_t bottom, size_t & count, uint16_t gpuNum)
{
  if(bottom < ibottom || bottom + irange > itop) {count = 0; return NULL;} // make sure the range provided is in bounds

  this->bottom = bottom;
  top = bottom + irange;
  this->gpuNum = gpuNum;

  flags[30] = 1;
  flags[0] = 1;
  flags[29] = 1;

  *KernelData::h_count = 0;
  setKernelParam();

  bigsieve.setParameters(*this);
  bigsieve.launchLoopPrimes(*this);

  cudaDeviceSynchronize();
  count = *KernelData::h_count;

  return d_primeOut;
}

uint64_t CudaSieve::countPrimes(uint64_t top, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve;

  sieve->gpuNum = gpuNum;
  sieve->top = top;
  sieve->flags[30] = 1;
  sieve->launchCtl();

  uint64_t count = *KernelData::h_count;

  delete sieve;

  return count;
}

uint64_t CudaSieve::countPrimes(uint64_t bottom, uint64_t top, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve;

  sieve->gpuNum = gpuNum;
  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->launchCtl();

  uint64_t count = *KernelData::h_count;

  delete sieve;

  return count;
}

uint64_t * CudaSieve::getHostPrimes(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve;

  sieve->gpuNum = gpuNum;
  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;
  sieve->flags[20] = 1;

  sieve->launchCtl();
  count = *KernelData::h_count;

  sieve->h_primeOut = safeCudaMallocHost(sieve->h_primeOut, count*sizeof(uint64_t));
  cudaMemcpy(sieve->h_primeOut, sieve->d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  uint64_t * temp = sieve->h_primeOut;            // copy address to temp pointer
  sieve->h_primeOut = NULL;                       // prevent the array from being freed

  delete sieve;

  return temp;
}

std::vector<uint64_t> CudaSieve::getHostPrimesVector(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve;

  sieve->gpuNum = gpuNum;
  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;

  sieve->h_primeOut = (uint64_t*)0xffffffff; // this is dumb, but it prevents memory from being allocated, which we want

  sieve->setKernelParam();
  sieve->d_primeList = PrimeList::getSievingPrimes(sieve->maxPrime_, sieve->primeListLength, sieve->flags[30]);
  BigSieve::run(*sieve);

  count = *KernelData::h_count;

  std::vector<uint64_t> temp(count);

  cudaMemcpy(&temp[0], sieve->d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  sieve->h_primeOut = NULL;

  delete sieve;

  return temp;
}

uint64_t * CudaSieve::getDevicePrimes(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve;

  sieve->gpuNum = gpuNum;
  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;
  sieve->flags[20] = 1;


  sieve->launchCtl();
  count = *KernelData::h_count;
  uint64_t * temp = sieve->d_primeOut;
  sieve->d_primeOut = NULL;

  delete sieve;

  return temp;
}

void CudaSieve::getDevicePrimesVector(uint64_t bottom, uint64_t top, thrust::device_vector<uint64_t> & dv_primeOut, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve;

  sieve->gpuNum = gpuNum;
  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;
  sieve->flags[20] = 1;

  size_t numGuess = (uint64_t) (top/log(top))*(1+1.32/log(top)) -
  ((bottom/log(bottom))*(1+1.32/log(bottom)));

  dv_primeOut.reserve(numGuess);
  dv_primeOut.resize(numGuess);
  sieve->d_primeOut = thrust::raw_pointer_cast(&dv_primeOut[0]);

  sieve->launchCtl();
  dv_primeOut.resize(*KernelData::h_count);
  sieve->d_primeOut = NULL;

  delete sieve;
}

uint32_t * CudaSieve::getBitSieve()
{
  if(!flags[30])  displayRange();

  setKernelParam();

  d_primeList = PrimeList::getSievingPrimes(maxPrime_, primeListLength, flags[30]);

  bigsieve.setParameters(*this);
  bigsieve.allocate();

  bigsieve.fillNextMult();
  bigsieve.launchLoopCopy(*this);

  count = KernelData::getCount();

  return sieveOut;
}
