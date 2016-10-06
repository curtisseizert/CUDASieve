/*

CUDASieveLaunch.cu

Host functions for CUDASieve which interface with the device
Curtis Seizert  <cseizert@gmail.com>

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/
#include "CUDASieve/host.hpp"
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

//void PrimeOutList::printPrimes(uint64_t * h_primeOut){for(uint64_t i = 0; i < *KernelData::h_count; i++) printf("%llu\n", h_primeOut[i]);}

PrimeOutList::PrimeOutList(CudaSieve & sieve)
{
  blocks = (sieve.bigsieve.bigSieveBits)/(32*PL_SIEVE_WORDS);
  threads = 512;

  hist_size_lg = blocks/512 + 1;
  numGuess = (uint64_t) (sieve.top/log(sieve.top))*(1+1.32/log(sieve.top)) -
  ((sieve.bottom/log(sieve.bottom))*(1+1.32/log(sieve.bottom)));

  //if(sieve.maxPrime_ < sqrt(sieve.top)) numGuess *= (log(sieve.top)+1)/(log(sieve.maxPrime_*sieve.maxPrime_)-1);

  if(!sieve.flags[20])    allocateHost(sieve);
                          allocateDevice(sieve);
}

inline void PrimeOutList::allocateHost(CudaSieve & sieve)
{
  sieve.h_primeOut = safeCudaMallocHost(sieve.h_primeOut, numGuess*sizeof(uint64_t));
}

inline void PrimeOutList::allocateDevice(CudaSieve & sieve)
{
  sieve.d_primeOut =  safeCudaMalloc(sieve.d_primeOut, numGuess*sizeof(uint64_t));
  d_histogram =       safeCudaMalloc(d_histogram, blocks*sizeof(uint32_t));
  d_histogram_lg =    safeCudaMalloc(d_histogram_lg, hist_size_lg*sizeof(uint32_t));

  cudaMemset(sieve.d_primeOut, 0, numGuess*sizeof(uint64_t));
  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));
}

inline void PrimeOutList::fetch(BigSieve & bigsieve, uint64_t * d_primeOut)
{
  uint64_t * d_ptr = d_primeOut + * KernelData::h_count;

  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

  device::makeHistogram_PLout<<<bigsieve.bigSieveKB, THREADS_PER_BLOCK>>>
    (bigsieve.d_bigSieve, d_histogram);
  device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
    (d_histogram, d_histogram_lg, blocks);
  device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
    (d_histogram_lg, KernelData::d_count, hist_size_lg);
  device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
    (d_histogram, d_histogram_lg, blocks);
  device::makePrimeList_PLout<<<bigsieve.bigSieveKB, THREADS_PER_BLOCK>>>
    (d_ptr, d_histogram, bigsieve.d_bigSieve, bigsieve.bottom, bigsieve.top);
}

PrimeOutList::~PrimeOutList()
{
  safeCudaFree(d_histogram);
  safeCudaFree(d_histogram_lg);
}

/*
                        **************************
                        ******* PrimeList ********
                        **************************

PrimeList is the class that makes a list of sieving primes on the device.  This work is orchestrated
by the static function PrimeList::getSievingPrimes(...), which returns a device pointer.
*/

uint32_t * PrimeList::getSievingPrimes(uint32_t maxPrime, uint32_t & primeListLength, bool silent=1)
{
  PrimeList primelist(maxPrime);

  primelist.allocate();
  primelist.sievePrimeList();
  primeListLength = primelist.h_primeListLength[0];
  if(!silent) std::cout << "List of sieving primes in " << primelist.timer.get_ms() << " ms." << std::endl;
  uint32_t * temp = primelist.d_primeList;
  primelist.d_primeList = NULL;

  return temp;
}

PrimeList::PrimeList(uint32_t maxPrime)
{
  this -> maxPrime = maxPrime;

  blocks = 1+maxPrime/(64 * PL_SIEVE_WORDS);
  threads = min(512, blocks);

  hist_size_lg = blocks/512 + 1;
  piHighGuess = (int) (maxPrime/log(maxPrime))*(1+1.2762/log(maxPrime)); // this is an empirically derived formula to calculate a high bound for the prime counting function pi(x)

  PL_Max = std::min((uint32_t)65536, maxPrime);
}

void PrimeList::allocate()
{
  h_primeListLength = (uint32_t *)malloc(sizeof(uint32_t));

  d_primeList =       safeCudaMalloc(d_primeList, piHighGuess*sizeof(uint32_t));
  d_primeListLength = safeCudaMalloc(d_primeListLength, sizeof(uint32_t));
  d_histogram =       safeCudaMalloc(d_histogram, blocks*sizeof(uint32_t));
  d_histogram_lg =    safeCudaMalloc(d_histogram_lg, hist_size_lg*sizeof(uint32_t));

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
  safeFree(h_primeListLength);
}

void SmallSieve::run(CudaSieve & sieve)
{
  if(!sieve.flags[0])                   sieve.smallsieve.count(sieve);
  if(!sieve.flags[30])                  sieve.smallsieve.timer.displayTime();
}

void SmallSieve::createStreams() // this takes about 0.025 ms
{
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);
}

void SmallSieve::count(CudaSieve & sieve)
{
  createStreams();
  timer.start();
  device::smallSieve<<<totBlocks, THREADS_PER_BLOCK, (sieve.sieveKB << 10), stream[0]>>>
    (sieve.d_primeList, KernelData::d_count, kernelBottom, sieve.sieveBits, sieve.primeListLength, KernelData::d_blocksComplete);
  if(sieve.isFlag(4)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK, 0, stream[1]>>>
    (sieve.d_primeList, top, sieve.sieveBits, sieve.primeListLength, sieve.top, KernelData::d_count, KernelData::d_blocksComplete, 1);
  if(sieve.isFlag(5)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK, 0, stream[2]>>>
    (sieve.d_primeList, kernelBottom, sieve.sieveBits, sieve.primeListLength, sieve.bottom-1, KernelData::d_count, KernelData::d_blocksComplete, 0);
  KernelData::displayProgress(totBlocks+sieve.isFlag(4)+sieve.isFlag(5));
  cudaDeviceSynchronize();
  timer.stop();
}


void BigSieve::run(CudaSieve & sieve) // coordinates the functions of this class for the CLI
{
  sieve.bigsieve.setParameters(sieve);
  sieve.bigsieve.allocate();

  sieve.bigsieve.fillNextMult();

  if(!sieve.flags[30])                      host::displayAttributes(sieve.bigsieve);

  if(sieve.flags[0]   && !sieve.flags[2])   sieve.bigsieve.launchLoopPrimesSmall(sieve);
  if(sieve.flags[0]   &&  sieve.flags[2])   sieve.bigsieve.launchLoopPrimes(sieve);
  if(!sieve.flags[0])                       sieve.bigsieve.launchLoop();
  if(!sieve.flags[30])                      sieve.bigsieve.timer.displayTime();
}

BigSieve::BigSieve(CudaSieve & sieve)
{
  setParameters(sieve);
  allocate();
}

void BigSieve::setParameters(CudaSieve & sieve)
{
  // Copy relevant sieve paramters
  sieveKB = 32;                                        // this is the optimal value for the big sieve
  if(!sieve.flags[1]) this -> sieveKB = sieve.sieveKB; // this defaults to 16, which is faster < 2**40
  this -> primeListLength = sieve.primeListLength;
  this -> d_primeList = sieve.d_primeList;
  this -> top = sieve.top;
  silent = sieve.flags[30];

  // Calculate BigSieve specific parameters
  bigSieveBits = bigSieveKB << 13;
  blocksSm = bigSieveKB/sieveKB;
  blocksLg = primeListLength/THREADS_PER_BLOCK_LG;
  log2bigSieveSpan = log2((double) bigSieveBits) + 1;
  if(!sieve.flags[0])   this -> bottom = max((1ull << 40), (unsigned long long) sieve.bottom);
  else                  this -> bottom = sieve.bottom;
  totIter = (this->top-this->bottom)/(2*this->bigSieveBits);
}

void BigSieve::allocate()
{
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  d_next =      safeCudaMalloc(d_next, primeListLength*sizeof(uint32_t));
  d_away =      safeCudaMalloc(d_away, primeListLength*sizeof(uint16_t));
  d_bigSieve =  safeCudaMalloc(d_bigSieve, bigSieveKB*256*sizeof(uint32_t));

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

/*
for BigSieve, kernels are launched iteratively.  "bigSieveSm" is essentially the same as
the small sieve SMem kernel (32 kb sieving array in L1), except that it only sieves with the first 65536
primes and copies its output (in an atomicOr operation) to a sieve in global memory that is launched
concurrently.  That sieve is "bigSieveLg," and it sieves with the remaining primes using
Oliveira's bucket method (described in global.cu) with a much larger sieve array (1024 -
4096 kb stored in global memory).  At the end of the operation of these two kernels, the large
global mem sieve has all the composites crossed off, and is counted and zeroed with bigSieveCount.
*/

void BigSieve::launchLoop() // for CLI
{
  timer.start();

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){
    cudaDeviceSynchronize();

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
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

void BigSieve::launchLoop(uint64_t bottom, uint64_t top) // for library where display is not needed
{
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){
    cudaDeviceSynchronize();

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>
      (d_bigSieve, sieveKB, KernelData::d_count);
  }
}

/*
this is only used for debugging at present.  It copies the bitsieve back to the host after
each iteration and increments the host pointer to the end of the data copied back.  This
gives a 'compressed' set of all the primes generated by the sieve.  An equivalent data set
can be generated from the output of a different prime number generator, and the sets can
be compared through various bitwise operations.  This is how the prime output of CUDASieve
is checked against primesieve.
*/
void BigSieve::launchLoopCopy(CudaSieve & sieve)
{
  timer.start();
  sieve.allocateSieveOut((top-bottom)/16);
  this -> ptr32 = sieve.sieveOut;
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
    if(primeListLength > 65536) device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    cudaMemcpy(ptr32, d_bigSieve, bigSieveKB*1024, cudaMemcpyDeviceToHost); // copy global mem sieve to appropriate
                                                                            // elements of host bitsieve output
    ptr32 +=  bigSieveKB*256;                                               // increment pointer

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>
      (d_bigSieve, sieveKB, KernelData::d_count);                            // count and zero
  }
  timer.stop();
  if(!silent) KernelData::displayProgress(totIter, totIter);
}

void BigSieve::launchLoopPrimes(CudaSieve & sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);

  timer.start();

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve,  bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    newlist.fetch(*this, sieve.d_primeOut);
    cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
    if(!silent) KernelData::displayProgress(value, totIter);
  }
  cudaDeviceSynchronize();
  timer.stop();
  if(!silent) {KernelData::displayProgress(totIter, totIter); std::cout<<std::endl;}
}

void BigSieve::launchLoopPrimesSmall(CudaSieve & sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);

  timer.start();

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, primeListLength);

    cudaDeviceSynchronize();

    newlist.fetch(*this, sieve.d_primeOut);
    cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
    if(!silent) KernelData::displayProgress(value, totIter);
  }
  cudaDeviceSynchronize();
  timer.stop();
  if(!silent) {KernelData::displayProgress(totIter, totIter); std::cout<<std::endl;}
}

BigSieve::~BigSieve()
{
  safeCudaFree(d_next);
  safeCudaFree(d_away);
  safeCudaFree(d_bigSieve);
}
