/*

cudasieve.cpp

CudaSieve class functions for CUDASieve
Curtis Seizert  <cseizert@gmail.com>

*/

#include "CUDASieve/launch.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/host.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cinttypes>

CudaSieve::CudaSieve()
{
  start_time = clock();
  kerneldata.allocate();
}

CudaSieve::CudaSieve(uint16_t gpuNum)
{
  this->setGpu(gpuNum);

  start_time = clock();
  kerneldata.allocate();
}

CudaSieve::CudaSieve(uint64_t bottom, uint64_t top, uint64_t range) // used for getting necessary data up front
{                                                                       // for repetitive counts etc.
  start_time = clock();
  kerneldata.allocate();

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
    h_primeOut =  safeCudaMallocHost(h_primeOut, numGuess*sizeof(uint64_t));
    d_primeOut =  safeCudaMalloc(d_primeOut, numGuess*sizeof(uint64_t));
  }
}

CudaSieve::~CudaSieve()
{
  safeCudaFreeHost(h_primeOut);
  safeCudaFree(d_primeOut);
  safeCudaFree(d_primeList);
}

void CudaSieve::reset()
{
  safeCudaFreeHost(h_primeOut);
  safeCudaFree(d_primeOut);
  *kerneldata.h_count = 0;
  start_time = clock();
}

void CudaSieve::setSieveKB(uint32_t sieveKB){this -> sieveKB = sieveKB;}
void CudaSieve::setBigSieveKB(uint32_t bigSieveKB){this -> bigsieve.bigSieveKB = bigSieveKB;}
void CudaSieve::setMaxPrime(uint32_t maxPrime){this -> maxPrime_ = maxPrime;}

void CudaSieve::setGpu(uint16_t gpuNum){this -> gpuNum = gpuNum; cudaSetDevice(gpuNum);}

void CudaSieve::allocateSieveOut(uint64_t size){sieveOut = new uint32_t[size/sizeof(uint32_t)];}

void CudaSieve::allocateSieveOut()
{
  sieveOutSize = (top-bottom)/2;
  sieveOut = new uint32_t[sieveOutSize/sizeof(uint32_t)];
}

void CudaSieve::listDevices()
{
  int count;
  cudaGetDeviceCount(&count);
  std::cout << "\n" << count << " CUDA enabled devices available:" << std::endl;

  for(int i = 0; i < count; i++){
    std::cout << "\t(" << i << ") : ";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << prop.name << std::endl;
  }
  std::cout << std::endl;
}

char * CudaSieve::getCurrentDeviceName()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpuNum);
  return prop.name;
}

inline void CudaSieve::setKernelParam()
{
  if(top > 1ull << 63 && !flags[18])  bigsieve.bigSieveKB = 1u << 12; // bigger sieve size is more efficient above 2^63 (2^12 kb vs 2^10 kb)
  if(top < 1ull << 40 && !flags[18] && (top - bottom) >= 1ull << 32) bigsieve.bigSieveKB = 1u << 14; // also optimization
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
  if(top - bottom < 10)
    {std::cerr << "CUDASieve Error: range must be greater than 10." << std::endl; exit(1);}
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
  std::cout << "\n" << "\tCounting primes from " << bottom << " to " << top << std::endl;
  std::cout << "\tUsing Device " << gpuNum << ": " << getCurrentDeviceName() << "\n" << std::endl;
}

double CudaSieve::elapsedTime()
{
  return (clock() - start_time)/((double) CLOCKS_PER_SEC);
}

inline void CudaSieve::launchCtl()
{
  if(flags[0]){
    uint64_t numGuess;
    if(bottom == 0) numGuess = (top/log(top))*(1+1.32/log(top));
    else if(top - bottom > 32768) numGuess = (top/log(top))*(1+1.12/log(top)) - (bottom/log(bottom))*(1+1.12/log(bottom)) + 256*log(top-bottom);
    else numGuess = ((1 + 2/log(top-bottom)) * (top - bottom)/log(bottom)) + 32;
    d_primeOut =  safeCudaMalloc(d_primeOut, numGuess*sizeof(uint64_t));
    cudaMemset(d_primeOut, 0, numGuess*sizeof(uint64_t));
  }

  setKernelParam();
  d_primeList = PrimeList::getSievingPrimes(maxPrime_, primeListLength, flags[30]);

  if(!flags[30] && !flags[0])     host::displayAttributes(*this);
  if(!flags[2]  && !flags[0])     SmallSieve::run(*this);
  if(flags[1]   || flags[0])      BigSieve::run(*this);
  count = kerneldata.getCount();
}

inline void CudaSieve::phiCtl(uint32_t a)
{
  setKernelParam();
  d_primeList = PrimeList::getSievingPrimes(maxPrime_, primeListLength, flags[30]);

  if(a >= 12)
    primeListLength = a - 12;
  else
    {primeListLength = 0; std::cout << "a must be >= 12" << std::endl;}

  if(!flags[30] && !flags[0])     host::displayAttributes(*this);
  if(!flags[2]  && !flags[0])     SmallSieve::run(*this);
  if(flags[1]   || flags[0])      BigSieve::run(*this);
  count = kerneldata.getCount();
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
  setGpu(gpuNum);

  *kerneldata.h_count = 0;
  flags[30] = 1;
  setKernelParam();

  if(!flags[2])   smallsieve.count(*this);
  if(flags[1])    bigsieve.launchLoop(*this);

  cudaDeviceSynchronize();
  return *kerneldata.h_count;
}

uint64_t * CudaSieve::getHostPrimesSegment(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  if(bottom < ibottom || top > itop || top-bottom > irange) {count = 0; return NULL;} // make sure the range provided is in bounds

  this->bottom = bottom;
  this->top = top;
  setGpu(gpuNum);

  flags[30] = 1;
  flags[0] = 1;
  flags[29] = 1;

  *kerneldata.h_count = 0;
  setKernelParam();

  bigsieve.setParameters(*this);
  bigsieve.launchLoopPrimes(*this);

  cudaDeviceSynchronize();
  count = *kerneldata.h_count;

  cudaMemcpy(h_primeOut, d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  return h_primeOut;
}

uint64_t * CudaSieve::getDevicePrimesSegment(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  if(bottom < ibottom || top > itop || top-bottom > irange) {count = 0; return NULL;} // make sure the range provided is in bounds

  this->bottom = bottom;
  this->top = top;
  setGpu(gpuNum);

  flags[30] = 1;
  flags[0] = 1;
  flags[29] = 1;

  *kerneldata.h_count = 0;
  setKernelParam();

  bigsieve.setParameters(*this);
  bigsieve.launchLoopPrimes(*this);

  cudaDeviceSynchronize();
  count = *kerneldata.h_count;

  return d_primeOut;
}

uint64_t CudaSieve::countPrimes(uint64_t top, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  sieve->top = top;
  sieve->flags[30] = 1;
  sieve->launchCtl();

  uint64_t count = *sieve->kerneldata.h_count;

  delete sieve;

  return count;
}

uint64_t CudaSieve::countPrimes(uint64_t bottom, uint64_t top, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  if(bottom == 1) bottom--;

  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->launchCtl();

  uint64_t count = *sieve->kerneldata.h_count;

  delete sieve;

  return count;
}

uint64_t CudaSieve::countPhi(uint64_t top, uint32_t a, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  sieve->top = top;
  sieve->flags[30] = 1;

  sieve->phiCtl(a);

  uint64_t count = 1 + *sieve->kerneldata.h_count - a;

  delete sieve;

  return count;
}

uint64_t CudaSieve::countPhi(uint64_t bottom, uint64_t top, uint32_t a, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  if(bottom == 1) bottom--;

  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->phiCtl(a);

  uint64_t count = 1 + *sieve->kerneldata.h_count - a;

  delete sieve;

  return count;
}

uint64_t * CudaSieve::getHostPrimes(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  if(bottom == 1) bottom--;

  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;
  sieve->flags[20] = 1;

  sieve->launchCtl();
  count = *sieve->kerneldata.h_count;

  sieve->h_primeOut = safeCudaMallocHost(sieve->h_primeOut, count*sizeof(uint64_t));
  cudaMemcpy(sieve->h_primeOut, sieve->d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  uint64_t * temp = sieve->h_primeOut;            // copy address to temp pointer
  sieve->h_primeOut = NULL;                       // prevent the array from being freed

  delete sieve;

  return temp;
}

std::vector<uint64_t> CudaSieve::getHostPrimesVector(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  if(bottom == 1) bottom--;

  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;

  sieve->h_primeOut = (uint64_t*)0xffffffff; // this is dumb, but it prevents memory from being allocated, which we want

  sieve->setKernelParam();
  sieve->d_primeList = PrimeList::getSievingPrimes(sieve->maxPrime_, sieve->primeListLength, sieve->flags[30]);
  BigSieve::run(*sieve);

  count = *sieve->kerneldata.h_count;

  std::vector<uint64_t> temp(count);

  cudaMemcpy(&temp[0], sieve->d_primeOut, count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  sieve->h_primeOut = NULL;

  delete sieve;

  return temp;
}

uint64_t * CudaSieve::getDevicePrimes(uint64_t bottom, uint64_t top, size_t & count, uint16_t gpuNum)
{
  CudaSieve * sieve = new CudaSieve(gpuNum);

  if(bottom == 1) bottom--;

  sieve->top = top;
  sieve->bottom = bottom;
  sieve->flags[30] = 1;
  sieve->flags[0] = 1;
  sieve->flags[29] = 1;
  sieve->flags[20] = 1;

  sieve->launchCtl();
  count = *sieve->kerneldata.h_count;
  uint64_t * temp = sieve->d_primeOut;
  sieve->d_primeOut = NULL;

  delete sieve;

  return temp;
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

  count = kerneldata.getCount();

  return sieveOut;
}
