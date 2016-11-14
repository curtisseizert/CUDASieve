/*

host.cpp

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

*/
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/launch.cuh"

#include "CUDASieve/host.hpp"

#include <iostream>
#include <stdio.h>


// volatile uint64_t * KernelData::h_count;
// volatile uint64_t * KernelData::h_blocksComplete;
// volatile uint64_t * KernelData::d_count;
// volatile uint64_t * KernelData::d_blocksComplete;


void host::displayAttributes(CudaSieve & sieve)
{
  if(!sieve.flags[30]) std::cout << "\n" << sieve.primeListLength << " sieving primes in (37, " << sieve.maxPrime_ << "]" << std::endl;

  if(!sieve.flags[2] && !sieve.flags[30]){
    std::cout << "Small Sieve parameters" << std::endl;
    std::cout << "Full Blocks     :  " << sieve.smallsieve.totBlocks << std::endl;
    std::cout << "Threads         :  " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Sieve Size      :  " << sieve.sieveKB << " kb" << std::endl;
  }
}

void host::displayAttributes(const BigSieve & bigsieve)
{
  std::cout << "Big Sieve parameters" <<std::endl;
  std::cout << "Number of required iterations\t:  " << bigsieve.totIter << std::endl;
  std::cout << "Size of small sieve\t\t:  " << bigsieve.sieveKB << " kb" <<std::endl;
  std::cout << "Size of big sieve\t\t:  " << bigsieve.bigSieveKB << " kb" <<std::endl;
  std::cout << "Bucket arrays filled in\t\t:  " << bigsieve.time_ms << " ms\n" << std::endl;
}

void KernelData::allocate()
{
  cudaHostAlloc((void **)&h_count, sizeof(uint64_t), cudaHostAllocMapped);
  cudaHostAlloc((void **)&h_blocksComplete, sizeof(uint64_t), cudaHostAllocMapped);

  cudaHostGetDevicePointer((unsigned long **)&d_count, (unsigned long *)h_count, 0);
  cudaHostGetDevicePointer((unsigned long **)&d_blocksComplete, (unsigned long *)h_blocksComplete, 0);

  *h_count = 0;
  *h_blocksComplete = 0;
}

void KernelData::deallocate()
{
  safeCudaFreeHost((void *)h_count);
  safeCudaFreeHost((void *)h_blocksComplete);
}

void KernelData::displayProgress(uint64_t totBlocks)
{
  if(totBlocks != 0){
    uint64_t value = 0;
    uint64_t counter = 0;
    do{
      uint64_t value1 = * h_blocksComplete;
      counter = * h_count;
      if (value1 > value){
        std::cout << "\t" << (100*value/totBlocks) << "% complete\t\t" << counter << " primes counted.\r";
        std::cout.flush();
         value = value1;
       }
    }while (value < totBlocks);
    counter = * h_count;
  }
  cudaDeviceSynchronize();
  std::cout << "\t" << "100% complete\t\t" << * h_count << " primes counted.\r";
}

void KernelData::displayProgress(uint64_t value, uint64_t totIter)
{
  std::cout << "\t" << (100*value/totIter) << "% complete\t\t" << *h_count << " primes counted.\r";
  std::cout.flush();
}

KernelTime::KernelTime()
{
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}

KernelTime::~KernelTime()
{
  if(start_ != NULL) {cudaEventDestroy(start_); start_ = NULL;}
  if(stop_ != NULL) {cudaEventDestroy(stop_); stop_ = NULL;}
}

void KernelTime::start()
{
  cudaEventRecord(start_);
}

void KernelTime::stop()
{
  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);
}

float KernelTime::get_ms()
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_, stop_);
  return milliseconds;
}

void KernelTime::displayTime()
{
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start_, stop_);
  if(milliseconds >= 1000) std::cout << "kernel time: " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "kernel time: " << milliseconds << " ms.    " << std::endl;
}
