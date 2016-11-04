/*

launch.cu

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
#include "CUDASieve/global.cuh"
#include "CUDASieve/launch.cuh"
#include "CUDASieve/primeoutlist.cuh"

#include <iostream>
#include <ctime>
#include <cmath>


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
    (sieve.d_primeList, sieve.kerneldata.d_count, kernelBottom, sieve.sieveBits, sieve.primeListLength, sieve.kerneldata.d_blocksComplete);
  if(sieve.isFlag(4)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK, 0, stream[1]>>>
    (sieve.d_primeList, top, sieve.sieveBits, sieve.primeListLength, sieve.top, sieve.kerneldata.d_count, sieve.kerneldata.d_blocksComplete, 1);
  if(sieve.isFlag(5)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK, 0, stream[2]>>>
    (sieve.d_primeList, kernelBottom, sieve.sieveBits, sieve.primeListLength, sieve.bottom-1, sieve.kerneldata.d_count, sieve.kerneldata.d_blocksComplete, 0);
  if(!sieve.isFlag(30)) sieve.kerneldata.displayProgress(totBlocks+sieve.isFlag(4)+sieve.isFlag(5));
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
  if(!sieve.flags[0])                       sieve.bigsieve.launchLoop(sieve);
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
  cutoff = bottom;
  bottom -= bottom%64;
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

void BigSieve::launchLoop(CudaSieve & sieve) // for CLI
{
  timer.start();
  if(totIter > 0){
    for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){
      cudaDeviceSynchronize();

      device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
        (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
      device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
        (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

      if(bottom < cutoff){
        cudaDeviceSynchronize();
        device::zeroBottomWord<<<1,1,0,stream[1]>>>(d_bigSieve, bottom, cutoff);
        }
      cudaDeviceSynchronize();
      device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t)), stream[0]>>>
        (d_bigSieve, sieveKB, sieve.kerneldata.d_count);

      if(!silent) sieve.kerneldata.displayProgress(value, totIter);
    }
  }
  if(bottom < top) countPartialTop(sieve);
  timer.stop();
  if(!silent) sieve.kerneldata.displayProgress(1, 1);
}

void BigSieve::countPartialTop(CudaSieve & sieve)
{
  cudaDeviceSynchronize();

  device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
    (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
  device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
    (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

  cudaDeviceSynchronize();

  if(bottom < cutoff){
    cudaDeviceSynchronize();
    device::zeroBottomWord<<<1,1,0,stream[1]>>>(d_bigSieve, bottom, cutoff);
    }
  device::bigSieveCountPartial<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>
    (d_bigSieve, sieveKB, bottom, top, sieve.kerneldata.d_count);

  cudaDeviceSynchronize();
}

void BigSieve::setupCopy(CudaSieve & sieve)
{
  uint64_t range = top - bottom;
  if(range < 1u << 24 && (range & ~(range -1)) != 0){
    range = 1u << (64 - clzll(range));
    bigSieveBits = range/2;
    bigSieveKB = bigSieveBits >> 13;
  }else if((range % (bigSieveBits << 1)) != 0){
    range += (bigSieveBits << 1) - (range % (bigSieveBits << 1));
  }
  top = bottom + range;
  blocksSm = bigSieveKB/sieveKB;

  sieve.allocateSieveOut((top-bottom)/16);
  sieve.allocateDeviceSieveOut((top-bottom)/16);
}

void BigSieve::launchLoopCopy(CudaSieve & sieve)
{
  ptr32 = sieve.sieveOut;
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, primeListLength);
    if(primeListLength > 65536) device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve, bigSieveBits, primeListLength, log2bigSieveSpan);

    cudaDeviceSynchronize();

    cudaMemcpy(ptr32, d_bigSieve, bigSieveKB*1024, cudaMemcpyDeviceToHost); // copy global mem sieve to appropriate
                                                                            // elements of host bitsieve output
    ptr32 +=  bigSieveKB*256;                                               // increment pointer

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>
      (d_bigSieve, sieveKB, sieve.kerneldata.d_count);                       // count and zero
  }
}

void BigSieve::launchLoopBitsieve(CudaSieve & sieve)
{
  d_bigSieve = sieve.d_sieveOut;
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10)>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, primeListLength);

    cudaDeviceSynchronize();

    d_bigSieve += bigSieveKB * 256;
  }
  d_bigSieve = NULL;
}

void BigSieve::launchLoopPrimes(CudaSieve & sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);

  timer.start();

  for(uint64_t value = 1; bottom < top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, 65536);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>
      (d_primeList, d_next, d_away, d_bigSieve,  bigSieveBits, primeListLength, log2bigSieveSpan);

    if(bottom < cutoff){
      cudaDeviceSynchronize();
      device::zeroBottomWord<<<1,1,0,stream[1]>>>(d_bigSieve, bottom, cutoff);
      }

    cudaDeviceSynchronize();

    newlist.fetch(*this, sieve);
    if(!silent && totIter != 0) sieve.kerneldata.displayProgress(value, max(1ul, (unsigned long)totIter));
  }
  cudaDeviceSynchronize();
  timer.stop();
  if(!silent) {sieve.kerneldata.displayProgress(1, 1); std::cout<<std::endl;}
}

void BigSieve::launchLoopPrimesSmall(CudaSieve & sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);

  timer.start();

  for(uint64_t value = 1; bottom < top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, primeListLength);

    if(bottom < sqrt(top) && bottom != 0){ // these conditionals add <<1% time to sieves
      cudaDeviceSynchronize();             // taking longer than 1 ms.
      device::zeroPrimeList<<<1,256,0,stream[1]>>>(d_bigSieve, bottom, d_primeList, primeListLength);
    }
    if(bottom < cutoff){
      cudaDeviceSynchronize();
      device::zeroBottomWord<<<1,1,0,stream[1]>>>(d_bigSieve, bottom, cutoff);
      }
    cudaDeviceSynchronize();

    newlist.fetch(*this, sieve);
    if(!silent) sieve.kerneldata.displayProgress(value, max(1ul, (unsigned long)totIter));
  }
  cudaDeviceSynchronize();
  timer.stop();
  if(!silent) {sieve.kerneldata.displayProgress(1, 1); std::cout<<std::endl;}
}

void BigSieve::launchLoopPrimesSmall32(CudaSieve & sieve) // makes the list of primes on the device and then copies them back to the host
{
  PrimeOutList newlist(sieve);

  timer.start();

  for(uint64_t value = 1; bottom < top; bottom += 2*bigSieveBits, value++){
;
    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>
      (d_primeList, d_bigSieve, bottom, sieveKB, primeListLength);

    if(bottom < sqrt(top) && bottom != 0){ // these conditionals add <<1% time to sieves
      cudaDeviceSynchronize();             // taking longer than 1 ms.
      device::zeroPrimeList<<<1,256,0,stream[1]>>>(d_bigSieve, bottom, d_primeList, primeListLength);
    }
    if(bottom < cutoff){
      cudaDeviceSynchronize();
      device::zeroBottomWord<<<1,1,0,stream[1]>>>(d_bigSieve, bottom, cutoff);
      }
    cudaDeviceSynchronize();

    newlist.fetch32(*this, sieve);
    if(!silent) sieve.kerneldata.displayProgress(value, max(1ul, (unsigned long)totIter));
  }
  cudaDeviceSynchronize();
  timer.stop();
  if(!silent) {sieve.kerneldata.displayProgress(1, 1); std::cout<<std::endl;}
}

BigSieve::~BigSieve()
{
  safeCudaFree(d_next);
  safeCudaFree(d_away);
  safeCudaFree(d_bigSieve);
}
