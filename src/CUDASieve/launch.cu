/*

CUDASieveLaunch.cu

Host functions for CUDASieve which interface with the device
Curtis Seizert - cseizert@gmail.com

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/

#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/device.cuh"
#include "CUDASieve/global.cuh"
#include "CUDASieve/launch.cuh"

#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>

volatile uint64_t * KernelData::h_count;
volatile uint64_t * KernelData::h_blocksComplete;
volatile uint64_t * KernelData::d_count;
volatile uint64_t * KernelData::d_blocksComplete;

/*
                      *************************
                      ****** PrimeOutList *****
                      *************************

PrimeOutList is the class that deals with getting lists of primes from pre-existing
sieve arrays.

*/

void PrimeOutList::printPrimes(){for(uint64_t i = 0; i < *KernelData::h_count; i++) printf("%llu\n", h_primeOut[i]);}
uint64_t * PrimeOutList::getPrimeOut(){return h_primeOut;}

PrimeOutList::PrimeOutList(CudaSieve * sieve)
{
  blocks = (sieve->bigSieveBits)/(32*PL_SIEVE_WORDS);
  threads = 512;

  hist_size_lg = blocks/512 + 1;
  numGuess = (uint64_t) (sieve->top/log(sieve->top))*(1+1.2762/log(sieve->top)) -
    ((sieve->bottom/log(sieve->bottom))*(1+1.2762/log(sieve->bottom)));
}

void PrimeOutList::allocate()
{
  if(cudaMallocHost(&h_primeOut, numGuess*sizeof(uint64_t)))
    {std::cerr << "PrimeOutList: CUDA host memory allocation error: h_primeOut" << std::endl; exit(1);}

  if(cudaMalloc(&d_primeOut, numGuess*sizeof(uint64_t)))
    {std::cerr << "PrimeOutList: CUDA device memory allocation error: d_primeOut" << std::endl; exit(1);}
  if(cudaMalloc(&d_histogram, blocks*sizeof(uint32_t)))
    {std::cerr << "PrimeOutList: CUDA device memory allocation error: d_histogram" << std::endl; exit(1);}
  if(cudaMalloc(&d_histogram_lg, hist_size_lg*sizeof(uint32_t)))
    {std::cerr << "PrimeOutList: CUDA device memory allocation error: d_histogram_lg" << std::endl; exit(1);}

  cudaMemset(d_primeOut, 0, numGuess*sizeof(uint64_t));
  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));
}

void PrimeOutList::fetch(BigSieve * sieve)
{
  uint64_t * d_ptr = d_primeOut + * KernelData::h_count;

  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

  device::makeHistogram_PLout<<<sieve->bigSieveKB, THREADS_PER_BLOCK>>> // this method of calculating blocks only works with 256 word shared sieves
    (sieve->d_bigSieve, d_histogram);
  device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
    (d_histogram, d_histogram_lg, blocks);
  device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
    (d_histogram_lg, KernelData::d_count, hist_size_lg);
  device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
    (d_histogram, d_histogram_lg, blocks);
  device::makePrimeList_PLout<<<sieve->bigSieveKB, THREADS_PER_BLOCK>>>
    (d_ptr, d_histogram, sieve->d_bigSieve, sieve->bottom, sieve->top);
}

void PrimeOutList::cleanupMin()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
}

void PrimeOutList::cleanupAllDevice()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
  cudaFree(d_primeOut);
}

PrimeOutList::~PrimeOutList()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
  cudaFree(d_primeOut);
  cudaFreeHost(h_primeOut);
}

PrimeList::PrimeList(uint32_t maxPrime)
{
  this -> maxPrime = maxPrime;

  blocks = 1+maxPrime/(64 * PL_SIEVE_WORDS);
  threads = min(512, blocks);

  hist_size_lg = blocks/512 + 1;
  piHighGuess = (int) (maxPrime/log(maxPrime))*(1+1.2762/log(maxPrime)); // this is an empirically derived formula to calculate a high bound for the prime counting function pi(x)
  if(maxPrime > 65536) PL_Max = sqrt(maxPrime);
  else PL_Max = maxPrime;
}

void PrimeList::allocate()
{
  h_primeListLength = (uint32_t *)malloc(sizeof(uint32_t));

  if(cudaMalloc(&d_primeList, piHighGuess*sizeof(uint32_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_primeList" << std::endl; exit(1);}
  if(cudaMalloc(&d_primeListLength, sizeof(uint32_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_primeListLength" << std::endl; exit(1);}
  if(cudaMalloc(&d_histogram, blocks*sizeof(uint32_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_histogram" << std::endl; exit(1);}
  if(cudaMalloc(&d_histogram_lg, hist_size_lg*sizeof(uint32_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_histogram_lg" << std::endl; exit(1);}

  cudaMemset(d_primeList, 0, piHighGuess*sizeof(uint32_t));
  cudaMemset(d_primeListLength, 0, sizeof(uint32_t));
  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

PrimeList::~PrimeList(){}

void PrimeList::sievePrimeList()
{
  cudaEventRecord(start);

  device::firstPrimeList<<<1, 256>>>(d_primeList, d_histogram, 32768, PL_Max);

  cudaMemcpy(h_primeListLength, &d_histogram[0], sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if(maxPrime > 65536){
    cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));

    device::makeHistogram<<<blocks, THREADS_PER_BLOCK>>>
      (d_primeList, d_histogram, 32*THREADS_PER_BLOCK, h_primeListLength[0]);
    device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
      (d_histogram, d_histogram_lg, blocks);
    device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
      (d_histogram_lg, d_primeListLength, hist_size_lg);
    device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
      (d_histogram, d_histogram_lg, blocks);
    device::makePrimeList<<<blocks, THREADS_PER_BLOCK>>>
      (d_primeList, d_histogram, 32*THREADS_PER_BLOCK, h_primeListLength[0], maxPrime);

    cudaMemcpy(h_primeListLength, &d_histogram[blocks-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
}

void PrimeList::displayTime()
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if(milliseconds >= 1000) std::cout << "List of sieving primes generated in " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "List of sieving primes generated in " << milliseconds << " ms.    " << std::endl;
}

uint32_t PrimeList::getPrimeListLength(){return h_primeListLength[0];}

void PrimeList::cleanUp()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
  cudaFree(d_primeListLength);
}

SmallSieve::SmallSieve(CudaSieve * sieve)
{
  if(!sieve->isFlag(30)) printf("\tCalling small sieve kernel\n");
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

void SmallSieve::launch(KernelData & kernelData, CudaSieve * sieve)
{
  cudaEventRecord(start);
  device::smallSieve<<<sieve->totBlocks, THREADS_PER_BLOCK, (sieve->sieveKB << 10)>>>(sieve->d_primeList, KernelData::d_count, sieve->kernelBottom, sieve->sieveBits, sieve->primeListLength, KernelData::d_blocksComplete);
  //if(flags[3]) smallSieveIncomplete<<<1, THREADS_PER_BLOCK, (sieve->sieveKB << 10)>>>(sieve->d_primeList, d_count, sieve->kernelBottom, sieve->sieveBits, sieve->primeListLength, sieve->bottom);
  if(sieve->isFlag(4)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK>>>(sieve->d_primeList, sieve->smKernelTop, sieve->sieveBits, sieve->primeListLength, sieve->top, KernelData::d_count, KernelData::d_blocksComplete);
  cudaEventRecord(stop);

  kernelData.displayProgress(sieve);

  cudaDeviceSynchronize();
  cudaEventSynchronize(stop);
}

void SmallSieve::displaySieveTime()
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if(milliseconds >= 1000) std::cout << "Kernel time: " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "Kernel time: " << milliseconds << " ms.    " << std::endl;
}

BigSieve::BigSieve(CudaSieve * sieve)
{
  // Inherit relevant sieve paramters
  this -> bigSieveKB = sieve -> bigSieveKB;
  this -> bigSieveBits = sieve -> bigSieveBits;
  this -> sieveKB = sieve -> sieveKB;
  this -> primeListLength = sieve -> primeListLength;
  this -> d_primeList = sieve -> d_primeList;
  this -> top = sieve -> top;
  this -> silent = sieve -> isFlag(30);

  // Calculate BigSieve specific parameters
  blocksSm = bigSieveKB/sieveKB;
  blocksLg = primeListLength/THREADS_PER_BLOCK_LG;
  log2bigSieveSpan = log2((double) bigSieveBits) + 1;
  this -> bottom = min((1ull << 40), (unsigned long long) sieve -> bottom);
  totIter = (this->top-this->bottom)/(2*this->bigSieveBits);
}

void BigSieve::setParameters(CudaSieve * sieve)
{
  // Inherit relevant sieve paramters
  this -> bigSieveKB = sieve -> bigSieveKB;
  this -> bigSieveBits = sieve -> bigSieveBits;
  this -> sieveKB = sieve -> sieveKB;
  this -> primeListLength = sieve -> primeListLength;
  this -> d_primeList = sieve -> d_primeList;
  this -> top = sieve -> top;
  this -> silent = sieve -> isFlag(30);

  // Calculate BigSieve specific parameters
  blocksSm = bigSieveKB/sieveKB;
  blocksLg = primeListLength/THREADS_PER_BLOCK_LG;
  log2bigSieveSpan = log2((double) bigSieveBits) + 1;
  this -> bottom = max((1ull << 40), (unsigned long long) sieve -> bottom);
  totIter = (this->top-this->bottom)/(2*this->bigSieveBits);
}

void BigSieve::allocate()
{
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if(cudaMalloc(&d_next, primeListLength*sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_next" << std::endl; exit(1);}
  if(cudaMalloc(&d_away, primeListLength*sizeof(uint16_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_away" << std::endl; exit(1);}
  if(cudaMalloc(&d_bigSieve, bigSieveKB*256*sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_bigSieve" << std::endl; exit(1);}

  cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
}

void BigSieve::fillNextMult()
{
  device::getNextMult30<<<blocksLg+1,THREADS_PER_BLOCK_LG>>>(d_primeList, d_next, d_away, primeListLength, bottom, bigSieveBits, log2bigSieveSpan);
  cudaDeviceSynchronize();
}

void BigSieve::launchLoop(KernelData & kernelData)
{
  cudaEventRecord(start);

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){
    cudaDeviceSynchronize();

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>(d_primeList, d_bigSieve, bottom,
      sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>(d_primeList, d_next, d_away, d_bigSieve,
     bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>(d_bigSieve, sieveKB, KernelData::d_count);
    kernelData.displayProgress(value, totIter);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  kernelData.displayProgress(totIter, totIter);
}

void BigSieve::launchLoopCopy(KernelData & kernelData, CudaSieve * sieve)
{
  cudaEventRecord(start);
  sieve->allocateSieveOut((top-bottom)/16);
  this -> ptr32 = sieve -> sieveOut;
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>(d_primeList, d_bigSieve, bottom,
      sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>(d_primeList, d_next, d_away, d_bigSieve,
     bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    cudaMemcpy(ptr32, d_bigSieve, bigSieveKB*1024, cudaMemcpyDeviceToHost);
    ptr32 +=  bigSieveKB*256;

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>(d_bigSieve, sieveKB, KernelData::d_count);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  kernelData.displayProgress(totIter, totIter);
}

void BigSieve::launchLoopPrimes(KernelData & kernelData, CudaSieve * sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);
  newlist.allocate();

  cudaEventRecord(start);

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>(d_primeList, d_bigSieve, bottom,
      sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>(d_primeList, d_next, d_away, d_bigSieve,
     bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    newlist.fetch(this);
    cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
    kernelData.displayProgress(value, totIter);

  }
  cudaDeviceSynchronize();
  cudaMemcpy(newlist.h_primeOut, newlist.d_primeOut, *KernelData::h_count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  kernelData.displayProgress(totIter, totIter);
  std::cout<<std::endl;
  newlist.printPrimes();
}

void BigSieve::displayCount(KernelData & kernelData)
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if(milliseconds >= 1000) std::cout << "Kernel time: " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "Kernel time: " << milliseconds << " ms.    " << std::endl;
}

void BigSieve::cleanUp()
{
  cudaFree(d_next);
  cudaFree(d_away);
  cudaFree(d_bigSieve);
}

void KernelData::allocate()
{
  cudaHostAlloc((void **)&KernelData::h_count, sizeof(uint64_t), cudaHostAllocMapped);
  cudaHostAlloc((void **)&KernelData::h_blocksComplete, sizeof(uint64_t), cudaHostAllocMapped);

  cudaHostGetDevicePointer((long **)&d_count, (long *)KernelData::h_count, 0);
  cudaHostGetDevicePointer((long **)&d_blocksComplete, (long *)KernelData::h_blocksComplete, 0);

  *KernelData::h_count = 0;
  *KernelData::h_blocksComplete = 0;
}

void KernelData::displayProgress(CudaSieve * sieve)
{
  if(!sieve->isFlag(30)){
    uint64_t value = 0;
    uint64_t counter = 0;
    do{
      uint64_t value1 = * KernelData::h_blocksComplete;
      counter = * KernelData::h_count;
      if (value1 > value){ // this is just to make it update less frequently
        std::cout << "\t" << (100*value/sieve->totBlocks) << "% complete\t\t" << counter << " primes counted.\r";
        std::cout.flush();
         value = value1;
       }
    }while (value < sieve->totBlocks+sieve->isFlag(4));
    counter = * KernelData::h_count;
  std::cout << "\t" << "100% complete\t\t" << counter << " primes counted.\r";
  }
}

void KernelData::displayProgress(uint64_t value, uint64_t totIter)
{
  std::cout << "\t" << (100*value/totIter) << "% complete\t\t" << *KernelData::h_count << " primes counted.\r";
  std::cout.flush();
}
