/*

CUDASieveGlobal.cu

Contains the __global__ functions in the device code for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

/*
                                  These functions are used for creating an ordered list of sieving primes on the GPU
*/

#include "CUDASieveGlobal.cuh"
#include <math_functions.h>
#include <iostream>

__global__ void device::firstPrimeList(uint32_t * d_primeList, uint32_t * d_histogram, uint32_t sieveBits, uint32_t maxPrime)
{
  uint64_t bidx = blockIdx.x;
  uint32_t sieveWords = sieveBits/32;
  __shared__ uint32_t s_sieve[1024];
  __shared__ uint32_t s_counts[1024];
  uint64_t bstart = bidx*sieveBits*2;

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  device::sieveFirst(s_sieve, sieveBits);
  __syncthreads();
  device::countPrimesHist(s_sieve, s_counts, sieveWords);
  __syncthreads();
  device::exclusiveScanBig(s_counts, sieveWords);
  device::movePrimesFirst(s_sieve, s_counts, sieveWords, d_primeList, d_histogram, bstart, maxPrime);
}

__global__ void device::inclusiveScan(uint32_t * d_array, uint16_t size)
{
  __shared__ uint32_t s_array[256];
  uint32_t tidx = threadIdx.x;

  s_array[tidx] = tidx;
  uint32_t sum;

  for(uint16_t offset = 1; offset <= size/2; offset *= 2){
    if(tidx >= offset){
      sum = s_array[tidx] + s_array[tidx - offset];
    }else{sum = s_array[tidx];}
    __syncthreads();
    s_array[tidx] = sum;
    __syncthreads();
  }

  d_array[tidx] = s_array[tidx];
}

__global__ void device::exclusiveScan(uint32_t * d_array, uint32_t size)
{
  extern __shared__ uint32_t s_array[];
  uint32_t tidx = threadIdx.x;

  s_array[tidx] = d_array[tidx];
  uint32_t sum;

  for(uint32_t offset = 1; offset <= size/2; offset *= 2){
    if(tidx >= offset){
      sum = s_array[tidx] + s_array[tidx - offset];
    }else{sum = s_array[tidx];}
    __syncthreads();
    s_array[tidx] = sum;
    __syncthreads();
  }
  if(tidx != 0) sum = s_array[tidx-1];
  else sum = 0;
  __syncthreads();
  s_array[tidx] = sum;
  d_array[tidx] = s_array[threadIdx.x];
}

__global__ void device::exclusiveScan(uint32_t * d_array, uint32_t * d_totals, uint32_t size)
{
  extern __shared__ uint32_t s_array[];
  uint32_t tidx = threadIdx.x;
  uint32_t block_offset = blockIdx.x * blockDim.x;

  if(tidx+block_offset < size) s_array[tidx] = d_array[tidx+block_offset];
  else  s_array[tidx] = 0;
  uint32_t sum = s_array[tidx];
  __syncthreads();

  for(uint32_t offset = 1; offset < blockDim.x; offset *= 2){
    if(tidx >= offset) sum = s_array[tidx] + s_array[tidx - offset];
    __syncthreads();
    s_array[tidx] = sum;
    __syncthreads();
  }
  if(tidx != 0) sum = s_array[tidx-1];
  else sum = 0;
  if(tidx+block_offset < size) d_array[tidx+block_offset] = sum;
  if(threadIdx.x == 0){ d_totals[blockIdx.x] = s_array[blockDim.x-1]; /*printf("%u\n", d_totals[blockIdx.x]);*/}
}

__global__ void device::exclusiveScanLazy(uint32_t * s_array, uint32_t size)
{
  uint32_t tidx = threadIdx.x;
  uint32_t sum;

  for(uint32_t offset = 1; offset <= size/2; offset *= 2){
    for(int32_t i = size- 1 - tidx; i >= 0; i -= threads){
      if(i >= offset){
        sum = s_array[i] + s_array[i - offset];
      }else{sum = s_array[i];}
      __syncthreads();
      s_array[i] = sum;
      __syncthreads();
    }
  }
  for(int32_t i = size - 1 - threadIdx.x; i >= 0; i -= threads){
    if (i > 0) sum = s_array[i-1];
    else sum = 0;
    __syncthreads();
    s_array[i] = sum;
  }
}

__global__ void device::increment(uint32_t * d_array, uint32_t * d_totals, uint32_t size)
{
  uint32_t tidx = threadIdx.x;
  uint32_t block_offset = blockIdx.x * blockDim.x;
  uint32_t arr_size = min(1024, (size-block_offset));
  uint32_t increment = d_totals[blockIdx.x];

  if(tidx < arr_size) d_array[tidx+block_offset] += increment;
}

__global__ void device::makeHistogram(uint32_t * d_primeList, uint32_t * d_histogram, uint32_t sieveBits, uint32_t primeListLength)
{
  uint64_t bidx = blockIdx.x;
  uint32_t sieveWords = sieveBits/32;
  __shared__ uint32_t s_sieve[256];
  uint64_t bstart = bidx*sieveBits*2;

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBasePL(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  else device::sieveMedPrimesPL(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::countPrimes(s_sieve, sieveWords);
  __syncthreads();
  device::moveCountHist(s_sieve, d_histogram);
}

__global__ void device::makePrimeList(uint32_t * d_primeList, uint32_t * d_histogram, uint32_t sieveBits, uint32_t primeListLength, uint32_t maxPrime)
{
  uint64_t bidx = blockIdx.x;
  uint32_t sieveWords = sieveBits/32;
  __shared__ uint32_t s_sieve[256];
  __shared__ uint16_t s_counts[256];
  uint64_t bstart = bidx*sieveBits*2;

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBasePL(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  else device::sieveMedPrimesPL(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::countPrimes(s_sieve, s_counts, sieveWords);
  __syncthreads();
  device::exclusiveScan(s_counts, sieveWords);
  device::movePrimes(s_sieve, s_counts, sieveWords, d_primeList, d_histogram, bstart, maxPrime);
}

/*
                              These kernels are for sieving small primes (< 2^40)
*/

__global__ void device::smallSieve(uint32_t * d_primeList, volatile uint64_t * d_count,
   uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, volatile uint64_t * d_blocksComplete)
{
  uint64_t bidx = blockIdx.x + blockIdx.y * gridDim.x;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  uint64_t bstart = bottom + bidx*sieveBits*2;
  float pstop = sqrtf(bstart + 2*sieveBits);
  unsigned int piHighGuess = (pstop/log(pstop))*(1+1.2762/log(pstop));
  primeListLength = min((unsigned int) primeListLength, piHighGuess);

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::countPrimes(s_sieve, sieveWords);
  __syncthreads();
  device::moveCount(s_sieve, d_count);
  if(threadIdx.x == 0)atomicAdd((unsigned long long *)d_blocksComplete,1ull);
}
/*
__global__ void device::smallSieveIncomplete(uint32_t * d_primeList, uint64_t * d_count,
   uint64_t kernelBottom, uint32_t sieveBits, uint32_t primeListLength, uint64_t bottom)
{
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  uint64_t bstart = kernelBottom - 2*sieveBits;

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::countPrimesRemBottom(s_sieve, sieveWords, bottom);
  __syncthreads();
  device::moveCountBottom(s_sieve, d_count, sieveWords);
  countPrimes(s_sieve, sieveWords);
  __syncthreads();
  moveCount(s_sieve, d_count, 0);
}
*/

__global__ void device::smallSieveIncompleteTop(uint32_t * d_primeList, uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, uint64_t top, volatile uint64_t * d_count, volatile uint64_t * d_blocksComplete)
{
  uint64_t bidx = blockIdx.x;
  uint32_t sieveWords = sieveBits/32;
  __shared__ uint32_t s_sieve[4096];
  __shared__ uint32_t s_counts[4096];
  uint64_t bstart = bottom+bidx*sieveBits*2;

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::countPrimesHist(s_sieve, s_counts, sieveWords);
  __syncthreads();
  device::exclusiveScanBig(s_counts, sieveWords);
  device::countTopPrimes(s_sieve, s_counts, sieveWords, bstart, top, d_count);
  if(threadIdx.x == 0)atomicAdd((unsigned long long *)d_blocksComplete,1ull);
}

__global__ void device::smallSieveCopy(uint32_t * d_primeList, uint64_t * d_count,
   uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, uint32_t * sieveOut)
{
  uint64_t bidx = blockIdx.x;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  uint64_t bstart = bottom + 2*bidx*sieveBits;

  device::sieveInit(s_sieve, sieveWords);
  __syncthreads();
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  //device::copyPrimes(s_sieve, sieveWords, bidx, sieveOut);
  __syncthreads();
  device::countPrimes(s_sieve, sieveWords);
  __syncthreads();
  device::moveCount(s_sieve, d_count);
}

/*
                        These kernels are used in the large sieve (>2^40)
*/

__global__ void device::getNextMult30(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t primeListLength, uint64_t bottom)
{
  uint32_t i = cutoff + threadIdx.x + blockIdx.x*blockDim.x;
  if(i < primeListLength){
    uint64_t n = 0;
    uint32_t p = d_primeList[i];
    uint64_t q = bottom/p;
    if(p > q) q = p;
    n |= (q / 30) << 3; // remember, this is used as a multiplier for a prime, so this will begin at the square of the prime.
    n |= lookup30[(q % 30)];
    while(p * ((30 * (n >> 3)) + wheel30[(n & 7)]) < bottom) n++;
    d_nextMult[i] = n;
  }
}

__global__ void device::getNextMult30_test(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t primeListLength, uint64_t bottom, uint32_t bigSieveBits)
{
  uint32_t i = cutoff + threadIdx.x + blockIdx.x*blockDim.x;
  if(i < primeListLength){
    uint32_t p = d_primeList[i];
    uint64_t q = bottom/p;
    if(p > q) q = p;
    uint64_t quot = q / 30; // remember, this is used as a multiplier for a prime, so this will begin at the square of the prime.
    uint8_t modIdx = lookup30[(q % 30)];
    uint64_t mult = p * (30*quot + wheel30[modIdx & 7]);
    while(mult < bottom){
      mult += p*wheel30Inc[modIdx & 7];
      modIdx = (modIdx + 1) & 7;
    }
    mult -= bottom;
    uint32_t offset = mult & ((bigSieveBits << 1) - 1); // this is where the next multiple will hit the appropriate sieve
    uint64_t away = mult >> 24; // this register begins as the number of sieves away this multiple is and need to not hard code that bit shift
    away |= modIdx << 24;
    away |= offset << 28;
    d_nextMult[i] = away;
  }
}

__global__ void device::bigSieveSm(uint32_t * d_primeList, uint32_t * bigSieve, // bigSieveBits will be determined by total available shared mem.
   uint64_t bottom, uint32_t primeListLength, uint32_t sieveKB)
{
  // starting each block at bottom + sieveBits*blockIdx.x covers the entire large sieve size due to how blocks are calculated on the host side
  // similarly, the large sieve array is covered when blocks are each spaced by sieveWords
  uint32_t sieveBits = sieveKB*8192;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];

  uint64_t bstart = bottom + 2*blockIdx.x*sieveBits;
  device::sieveInit(s_sieve, sieveWords);
  __syncthreads();
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  device::sieveMiddlePrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::makeBigSieve(bigSieve, s_sieve, sieveWords);
  __syncthreads();
}

__global__ void device::bigSieveLg(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t * bigSieve, // bigSieveBits will be determined by total available shared mem.
   uint64_t bstart, uint32_t bigSieveBits, uint32_t primeListLength, uint32_t sieveKB)
{
  uint64_t bidx = blockIdx.x + blockIdx.y * gridDim.x;
  uint32_t i = cutoff + threadIdx.x + bidx*blockDim.x;
  // __shared__ uint8_t s_wheel30[8];
  // if(threadIdx.x < 8) s_wheel30[threadIdx.x] = wheel30[threadIdx.x];

  uint64_t n;
  uint32_t p;
  if(i < primeListLength){
    p = d_primeList[i];
    n = d_nextMult[i];
    // uint64_t m = p * ((30 * (n >> 3)) + wheel30[n & 7u]) - bstart;
    uint64_t m = 30 * (n >> 3);
    m += wheel30[n & 7u];
    m*= p;
    m -= bstart;

    for(; m < (bigSieveBits << 1); n++){
      uint32_t idx = m >> 6;
      uint16_t sidx = (m & 63) >> 1;
      atomicOr(&bigSieve[idx], (1ul << sidx));
      m += p*wheel30Inc[n & 7];
    }
    d_nextMult[i] = n;
  }
}

__global__ void device::bigSieveLg_test(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t * bigSieve,
  uint64_t bstart, uint32_t bigSieveBits, uint32_t primeListLength, uint32_t sieveKB)
{
  uint64_t bidx = blockIdx.x + blockIdx.y * gridDim.x;
  uint32_t i = cutoff + threadIdx.x + bidx*blockDim.x;

  uint64_t n;
  uint32_t p;
  if(i < primeListLength){
    p = d_primeList[i];
    n = d_nextMult[i];
    if(n & 65535 != 0) n--;
    else{
      uint8_t modIdx = (n & 268435455) >> 24;
      uint64_t off = n >> 32;

    for(; off < (bigSieveBits << 1); modIdx = (modIdx + 1) & 7){
      uint32_t idx = off >> 6;
      uint16_t sidx = (off & 63) >> 1;
      atomicOr(&bigSieve[idx], (1ul << sidx));
      off += p*wheel30Inc[modIdx];
    }
    n = off >> 24;
    n |= modIdx << 24;
    n |= (off & ((bigSieveBits << 1)-1)) << 28;
    }
  }
}

__global__ void device::bigSieveCount(uint32_t * bigSieve, uint32_t sieveKB, volatile uint64_t * d_count)
{
  uint32_t sieveBits = sieveKB*8192;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  s_sieve[threadIdx.x] = 0;
  uint32_t count = 0;

  uint32_t blockStart = sieveWords*blockIdx.x;
  for(uint32_t i = threadIdx.x; i < sieveWords; i += threads){
    uint32_t x = bigSieve[i+blockStart];
    bigSieve[i+blockStart] ^= bigSieve[i+blockStart];
    for(uint8_t j = 0; j < 32; j++) count += 1 & ~(x >> j);
  }
  __syncthreads();
  s_sieve[threadIdx.x] = count;
  __syncthreads();

  device::moveCount(s_sieve, d_count);
}
