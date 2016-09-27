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

CudaSieve::CudaSieve()
{
  start_time = clock();
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

void CudaSieve::setTop(uint64_t top){this -> top = top;}
void CudaSieve::setBottom(uint64_t bottom){this -> bottom = bottom;}
void CudaSieve::setSieveKB(uint32_t sieveKB){this -> sieveKB = sieveKB;}
void CudaSieve::setBigSieveKB(uint32_t bigSieveKB){this -> bigSieveKB = bigSieveKB;}
void CudaSieve::setGpuNum(uint16_t gpuNum){this -> gpuNum = gpuNum;}
void CudaSieve::allocateSieveOut(uint64_t size){sieveOut = new uint32_t[size/sizeof(uint32_t)];}

void CudaSieve::setDefaults()
{
  bottom = 0;
  top = (1u << 30);
  sieveKB = 16;
  bigSieveKB = (1u << 10);
  gpuNum = 0;
}

void CudaSieve::setKernelParam()
{
  sieveBits = sieveKB << 13;
  bigSieveBits = bigSieveKB << 13;
  uint64_t smTop = std::min((unsigned long long) top, 1ull << 40);
  kernelBottom = bottom - bottom % (2 * sieveBits);
  totBlocks = (smTop - kernelBottom) / (2 *  sieveBits);
  smKernelTop = kernelBottom + (totBlocks * sieveBits * 2);
  cudaSetDevice(gpuNum);
  this->setFlags();
}

void CudaSieve::setFlags()
{
  if(top > (1ull << 40)) this -> flags[1] = 1;
  if(bottom >= (1ull << 40)) this -> flags[2] = 1;
  if(kernelBottom != bottom) this -> flags[3] = 1;
  if(std::min((unsigned long long) top, 1ull << 40) != smKernelTop) this -> flags[4] = 1;
}

void CudaSieve::displayRange()
  {if(!isFlag(30)) std::cout << "\n" << "Counting primes from " << bottom << " to " << top << std::endl;}

void CudaSieve::displaySieveAttributes()
{
  if(!this->isFlag(30)) std::cout << "\n" << primeListLength << " sieving primes in (37, " << (unsigned long) sqrt(top) << "]" << std::endl;

  if(!this->isFlag(2) && !this->isFlag(30)){
    std::cout << "Small Sieve parameters" << std::endl;
    std::cout << "Total Blocks    :  " << totBlocks << std::endl;
    std::cout << "Threads         :  " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Sieve Size      :  " << sieveKB << " kb" << std::endl;
  }
  if(!this->isFlag(30)) std::cout << "Initialization took " << elapsedTime() << " seconds.\n" << std::endl;
}

void CudaSieve::makePrimeList()
{
  uint32_t maxPrime = (unsigned long) sqrt(top);

  PrimeList primelist(maxPrime);

  primelist.allocate();
  primelist.sievePrimeList();
  primeListLength = primelist.getPrimeListLength();
  if(!flags[30]) primelist.displayTime();
  this -> d_primeList = primelist.getPtr();
  primelist.cleanUp();
}

void CudaSieve::makePrimeList(uint32_t maxPrime)
{
  PrimeList primelist(maxPrime);

  primelist.allocate();
  primelist.sievePrimeList();
  primeListLength = primelist.getPrimeListLength();
  if(!flags[30]) primelist.displayTime();
  this -> d_primeList = primelist.getPtr();
  primelist.cleanUp();
}

void CudaSieve::launchCtl()
{
  KernelData kernelData;
  if(!flags[2]){
    if(!flags[0]) smallSieveCtl();
    //else host::smallSieveCopy();
  }
  if(flags[1]) bigSieveCtl();
  count = kernelData.getCount();
}

void CudaSieve::bigSieveCtl()
{
  KernelData kernelData;
  BigSieve * bigsieve = new BigSieve;

  bigsieve ->setParameters(this);
  bigsieve ->allocate();
  bigsieve ->fillNextMult();
  if(flags[0] && !flags[8])   bigsieve ->launchLoopPrimes(kernelData, this);
      // if copy and not debug generate list of primes on the host
  if(!flags[0] && !flags[8])  bigsieve ->launchLoop(kernelData);
      // if not copy and not debug count primes
  if(flags[0] && flags[8])    bigsieve->launchLoopCopy(kernelData, this);
      // if copy and debug copy sieve back to host
  bigsieve ->displayCount(kernelData);
  bigsieve ->cleanUp();
}

void CudaSieve::smallSieveCtl()
{
  KernelData kernelData;
  SmallSieve smallsieve(this);
  smallsieve.launch(kernelData, this);
  cudaDeviceSynchronize();
  if(!flags[30]) smallsieve.displaySieveTime();
}

void CudaSieve::countPrimes()
{
  this->setKernelParam();
  this->displayRange();
  this->makePrimeList();
  this->displaySieveAttributes();
  this->launchCtl();
}

uint64_t CudaSieve::countPrimes(uint64_t top)
{
  this->top = top;
  flags[30] = 1;
  this->setKernelParam();
  this->makePrimeList();
  this->launchCtl();
  return count;
}

uint64_t CudaSieve::countPrimes(uint64_t bottom, uint64_t top)
{
  this->bottom = bottom;
  this->top = top;
  flags[30] = 1;
  this->setKernelParam();
  this->makePrimeList();
  this->launchCtl();
  return count;
}

double CudaSieve::elapsedTime()
{
  return (clock() - start_time)/((double) CLOCKS_PER_SEC);
}
