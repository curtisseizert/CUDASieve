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
#include "host.hpp"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/device.cuh"
#include "CUDASieve/global.cuh"
#include "CUDASieve/launch.cuh"

#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <cuda_profiler_api.h>

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

PrimeOutList::PrimeOutList(CudaSieve & sieve)
{
  blocks = (sieve.bigSieveBits)/(32*PL_SIEVE_WORDS);
  threads = 512;

  hist_size_lg = blocks/512 + 1;
  numGuess = (uint64_t) (sieve.top/log(sieve.top))*(1+1.2762/log(sieve.top)) -
    ((sieve.bottom/log(sieve.bottom))*(1+1.2762/log(sieve.bottom)));
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

void PrimeOutList::fetch(BigSieve & sieve)
{
  uint64_t * d_ptr = d_primeOut + * KernelData::h_count;

  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

  device::makeHistogram_PLout<<<sieve.bigSieveKB, THREADS_PER_BLOCK>>>
    (sieve.d_bigSieve, d_histogram);
  device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
    (d_histogram, d_histogram_lg, blocks);
  device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
    (d_histogram_lg, KernelData::d_count, hist_size_lg);
  device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
    (d_histogram, d_histogram_lg, blocks);
  device::makePrimeList_PLout<<<sieve.bigSieveKB, THREADS_PER_BLOCK>>>
    (d_ptr, d_histogram, sieve.d_bigSieve, sieve.bottom, sieve.top);
}

void PrimeOutList::cleanupAll()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
  cudaFree(d_primeOut);
  cudaFreeHost(h_primeOut);
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
}

uint32_t * PrimeList::getSievingPrimes(uint32_t maxPrime, uint32_t & primeListLength, bool silent=1)
{
  PrimeList primelist(maxPrime);

  primelist.allocate();
  primelist.sievePrimeList();
  primeListLength = primelist.h_primeListLength[0];
  if(!silent) std::cout << "List of sieving primes in " << primelist.timer.get_ms() << " ms." << std::endl;
  return primelist.d_primeList;
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
    {std::cerr << "PrimeList: CUDA memory allocation error: d_primeList - " << cudaMalloc(&d_primeList, piHighGuess*sizeof(uint32_t)) << std::endl; exit(1);}
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
}

void PrimeList::sievePrimeList()
{
  timer.start();

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
  timer.stop();
}

PrimeList::~PrimeList()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
  cudaFree(d_primeListLength);
}

void SmallSieve::run(CudaSieve & sieve)
{
  SmallSieve smallsieve;

  smallsieve.launch(sieve);

  if(!sieve.flags[30])
    smallsieve.timer.displayTime();
}

void SmallSieve::launch(CudaSieve & sieve)
{
  timer.start();
  device::smallSieve<<<sieve.totBlocks, THREADS_PER_BLOCK, (sieve.sieveKB << 10)>>>
    (sieve.d_primeList, KernelData::d_count, sieve.kernelBottom, sieve.sieveBits, sieve.primeListLength, KernelData::d_blocksComplete);
  if(sieve.isFlag(4)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK>>>
    (sieve.d_primeList, sieve.smKernelTop, sieve.sieveBits, sieve.primeListLength, sieve.top, KernelData::d_count, KernelData::d_blocksComplete);
  KernelData::displayProgress(sieve);
  cudaDeviceSynchronize();
  timer.stop();
}

void BigSieve::run(CudaSieve & sieve)
{
  BigSieve bigsieve(sieve);

  bigsieve.fillNextMult();

  if(!sieve.flags[30])                       host::displayAttributes(bigsieve);
  if(sieve.flags[0]   &&  !sieve.flags[8])   bigsieve.launchLoopPrimes(sieve);
  if(!sieve.flags[0]  &&  !sieve.flags[8])   bigsieve.launchLoop();
  if(sieve.flags[0]   &&  sieve.flags[8])    bigsieve.launchLoopCopy(sieve);
  if(!sieve.flags[30])                       bigsieve.timer.displayTime();
}

BigSieve::BigSieve(CudaSieve & sieve)
{
  setParameters(sieve);
  allocate();
}

void BigSieve::setParameters(CudaSieve & sieve)
{
  // Copy relevant sieve paramters
  this -> bigSieveKB = sieve.bigSieveKB;
  this -> bigSieveBits = sieve.bigSieveBits;
  this -> sieveKB = 32; // this is the optimal value for the big sieve
  this -> primeListLength = sieve.primeListLength;
  this -> d_primeList = sieve.d_primeList;
  this -> top = sieve.top;
  this -> silent = sieve.isFlag(30);
  this -> noMemcpy = sieve.isFlag(20);

  // Calculate BigSieve specific parameters
  blocksSm = bigSieveKB/sieveKB;
  blocksLg = primeListLength/THREADS_PER_BLOCK_LG;
  log2bigSieveSpan = log2((double) bigSieveBits) + 1;
  this -> bottom = max((1ull << 40), (unsigned long long) sieve.bottom);
  totIter = (this->top-this->bottom)/(2*this->bigSieveBits);
}

void BigSieve::allocate()
{
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  if(cudaMalloc(&d_next, primeListLength*sizeof(uint32_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_next" << std::endl; exit(1);}
  if(cudaMalloc(&d_away, primeListLength*sizeof(uint16_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_away" << std::endl; exit(1);}
  if(cudaMalloc(&d_bigSieve, bigSieveKB*256*sizeof(uint32_t)))
    {std::cerr << "PrimeList: CUDA memory allocation error: d_bigSieve" << std::endl; exit(1);}

  cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
}

void BigSieve::fillNextMult()
{
  timer.start();

  device::getNextMult30<<<blocksLg+1,THREADS_PER_BLOCK_LG>>>
    (d_primeList, d_next, d_away, primeListLength, bottom, bigSieveBits, log2bigSieveSpan);

  timer.stop();
  time_ms = timer.get_ms();
  cudaDeviceSynchronize();
}

void BigSieve::launchLoop()
{
  timer.start();

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){
    cudaDeviceSynchronize();

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>
      (d_bigSieve, sieveKB, KernelData::d_count);
    if(!silent) KernelData::displayProgress(value, totIter);
  }
  timer.stop();
  if(!silent) KernelData::displayProgress(totIter, totIter);
}

/*
this is only used for debugging at present.  It copies the bitsieve back to the host after
each iteration and increments the host pointer to the end of the data copied back.  This
gives a 'compressed' set of all the primes generated by the sieve.  An equivalent data set
can be generated from the output of a different prime number generator, and the sets can
be compared through various bitwise operations.  This is how the prime output of CUDASieve
is checked (against primesieve).
*/
void BigSieve::launchLoopCopy(CudaSieve & sieve)
{
  timer.start();
  sieve.allocateSieveOut((top-bottom)/16);
  this -> ptr32 = sieve.sieveOut;
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    cudaMemcpy(ptr32, d_bigSieve, bigSieveKB*1024, cudaMemcpyDeviceToHost);
    ptr32 +=  bigSieveKB*256;

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>
      (d_bigSieve, sieveKB, KernelData::d_count);
  }
  timer.stop();
  if(!silent) KernelData::displayProgress(totIter, totIter);
}

void BigSieve::launchLoopPrimes(CudaSieve & sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);
  newlist.allocate();

  timer.start();

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve,  bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    newlist.fetch(*this);
    cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
    if(!silent) KernelData::displayProgress(value, totIter);
  }

  // Post sieve stop timer, memcpy, print, pass pointers
  cudaDeviceSynchronize();
  if(!noMemcpy) cudaMemcpy(newlist.h_primeOut, newlist.d_primeOut, *KernelData::h_count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  timer.stop();
  if(!silent) {
    KernelData::displayProgress(totIter, totIter);
    std::cout<<std::endl;
    newlist.printPrimes();
  }
  sieve.h_primeOut = newlist.h_primeOut;
  sieve.d_primeOut = newlist.d_primeOut;
}

BigSieve::~BigSieve()
{
  safeCudaFree(d_next);
  safeCudaFree(d_away);
  safeCudaFree(d_bigSieve);
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

void KernelData::displayProgress(CudaSieve & sieve)
{
  if(!sieve.isFlag(30) && sieve.totBlocks != 0){
    uint64_t value = 0;
    uint64_t counter = 0;
    do{
      uint64_t value1 = * KernelData::h_blocksComplete;
      counter = * KernelData::h_count;
      if (value1 > value){
        std::cout << "\t" << (100*value/sieve.totBlocks) << "% complete\t\t" << counter << " primes counted.\r";
        std::cout.flush();
         value = value1;
       }
    }while (value < sieve.totBlocks+sieve.isFlag(4));
    counter = * KernelData::h_count;
  }
  cudaDeviceSynchronize();
  if(!sieve.isFlag(30)) std::cout << "\t" << "100% complete\t\t" << * KernelData::h_count << " primes counted.\r";
}

inline void KernelData::displayProgress(uint64_t value, uint64_t totIter)
{
  std::cout << "\t" << (100*value/totIter) << "% complete\t\t" << *KernelData::h_count << " primes counted.\r";
  std::cout.flush();
}

KernelTime::KernelTime()
{
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}

KernelTime::~KernelTime()
{
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

inline void KernelTime::start()
{
  cudaEventRecord(start_);
}

inline void KernelTime::stop()
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
