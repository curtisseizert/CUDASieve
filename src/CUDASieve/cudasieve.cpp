/*

CUDASieveHost.cpp

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

*/

#include "CUDASieve/launch.cuh"
#include "CUDASieve/cudasieve.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>

void CudaSieve::reset()
{
  safeCudaFreeHost(h_primeOut);
  safeCudaFree(d_primeOut);
  safeCudaFree(d_primeList);
  *KernelData::h_count = 0;
  start_time = clock();
}

CudaSieve::CudaSieve()
{
  KernelData kerneldata;
  start_time = clock();
  cudaSetDeviceFlags(cudaDeviceMapHost);
  kerneldata.allocate();
}

void CudaSieve::setTop(uint64_t top){this -> top = top;}
void CudaSieve::setBottom(uint64_t bottom){this -> bottom = bottom;}
void CudaSieve::setSieveKB(uint32_t sieveKB){this -> sieveKB = sieveKB;}
void CudaSieve::setBigSieveKB(uint32_t bigSieveKB){this -> bigSieveKB = bigSieveKB;}
void CudaSieve::setGpuNum(uint16_t gpuNum){this -> gpuNum = gpuNum;}
void CudaSieve::allocateSieveOut(uint64_t size){sieveOut = new uint32_t[size/sizeof(uint32_t)];}

void CudaSieve::setKernelParam()
{
  if(top > 1ull << 63 && !flags[18]) bigSieveKB = 1u << 12;
  if(top < 1u << 23) sieveKB = 2;
  sieveBits = sieveKB << 13;
  bigSieveBits = bigSieveKB << 13;
  uint64_t smTop = std::min((unsigned long long) top, 1ull << 40);
  kernelBottom = bottom - bottom % (2 * sieveBits);
  totBlocks = (smTop - kernelBottom) / (2 *  sieveBits);
  smKernelTop = kernelBottom + (totBlocks * sieveBits * 2);
  cudaSetDevice(gpuNum);
  checkRange();
  setFlags();
}

void CudaSieve::checkRange()
{
  if(bottom > top)
    {std::cerr << "CUDASieve Error: the bottom of the range must be smaller than the top." << std::endl; exit(1);}
  if(top < 128)
    {std::cerr << "CUDASieve Error: the top of the range must be above 128." << std::endl; exit(1);}
  if((unsigned long long)top > 18446744056529682432ull) // 2^64-2^35
    {std::cerr << "CUDASieve Error: top above supported range (max is 2^64-2^35)." << std::endl; exit(1);}
  if((bottom < 1ull << 40) && (bottom %(sieveBits*2) !=0))
    {std::cerr << "CUDASieve Error: bottom must be a multiple of sieve size." << std::endl; exit(1);}
  if((bottom > 1ull << 40) && ((top-bottom)%(bigSieveBits*2) != 0))
    {std::cerr << "CUDASieve Error: above 2**40 range must be a multiple of sieve size." << std::endl; exit(1);}
}

void CudaSieve::setFlags()
{
  if(top > (1ull << 40)) flags[1] = 1;
  if(bottom >= (1ull << 40)) flags[2] = 1;
  if(kernelBottom != bottom) flags[3] = 1;
  if(std::min((unsigned long long) top, 1ull << 40) != smKernelTop) flags[4] = 1;
}

void CudaSieve::displayRange()
{
  std::cout << "\n" << "Counting primes from " << bottom << " to " << top << std::endl;
}

double CudaSieve::elapsedTime()
{
  return (clock() - start_time)/((double) CLOCKS_PER_SEC);
}

void CudaSieve::launchCtl()
{
  setKernelParam();
  d_primeList = PrimeList::getSievingPrimes(sqrt(top), primeListLength, flags[30]);

  if(!flags[30])  host::displayAttributes(*this);
  if(!flags[2])   SmallSieve::run(*this);
  if(flags[1])    BigSieve::run(*this);

  count = KernelData::getCount();
}

void CudaSieve::CLIPrimes()
{
  if(!flags[30])  displayRange();

  launchCtl();

  if(flags[30])   std::cout << count << std::endl;
}

uint64_t CudaSieve::countPrimes(uint64_t top)
{
  reset();
  this->top = top;
  flags[30] = 1;
  launchCtl();
  return count;
}

uint64_t CudaSieve::countPrimes(uint64_t bottom, uint64_t top)
{
  reset();
  this->bottom = bottom;
  this->top = top;
  flags[30] = 1;
  launchCtl();
  return count;
}

uint64_t * CudaSieve::getHostPrimes(uint64_t bottom, uint64_t top, size_t & size)
{
  reset();

  flags[0] = 1;
  flags[30] = 1;
  this->bottom = bottom;
  this->top = top;

  launchCtl();
  size = count;
  safeCudaFree(d_primeOut);
  safeCudaFree(d_primeList);
  return h_primeOut;
}

uint64_t * CudaSieve::getDevicePrimes(uint64_t bottom, uint64_t top, size_t & size)
{
  reset();

  flags[0] = 1;
  flags[30] = 1;
  flags[20] = 1;
  this->bottom = bottom;
  this->top = top;

  launchCtl();
  size = count;
  safeCudaFreeHost(h_primeOut);
  safeCudaFree(d_primeList);
  return d_primeOut;
}
