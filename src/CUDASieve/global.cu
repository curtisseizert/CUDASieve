/*

CUDASieveGlobal.cu

Contains the __global__ functions in the device code for CUDASieve 1.0
by Curtis Seizert
<cseizert@gmail.com>

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <iostream>

#include "CUDASieve/device.cuh"
#include "CUDASieve/global.cuh"
#include "CUDASieve/device.cu"
  // for some reason linking together two files with device code kills performance
  // so it is necessary to link them by including a source file like this

__constant__ uint8_t wheel30_g[8] = {1,7,11,13,17,19,23,29};
__constant__ uint8_t wheel30Inc_g[8] = {6,4,2,4,2,4,6,2};
__constant__ uint8_t lookup30_g[30] = {0,0,0,0,0,0,0,1,0,0,0,2,0,3,0,0,0,4,0,5,0,0,0,6,0,0,0,0,0,7};

__constant__ uint16_t threads_g = 256;
__constant__ uint32_t cutoff_g = 32768;

/*
        *******************************************************
        ******Kernels for making a list of sieving primes******
        *******************************************************
*/

/*
Kernel used for creating list of primes on the device with knowledge only of
3-37.  See device.cu - sieveFirst(...)
*/

__global__ void device::firstPrimeList(uint32_t * d_primeList, uint32_t * d_histogram,
   uint32_t sieveBits, uint32_t maxPrime)
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

/*
Some scan functions for incrementing the various histograms made in order
to index the primes on the device for creating a list.
*/

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
  if(threadIdx.x == 0) d_totals[blockIdx.x] = s_array[blockDim.x-1];
}

__global__ void device::exclusiveScan(uint32_t * d_array, volatile uint64_t * d_count, uint32_t size)
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
  if(threadIdx.x == 0){* d_count += s_array[blockDim.x-1];}
}

__global__ void device::exclusiveScanLazy(uint32_t * s_array, uint32_t size)
{
  uint32_t tidx = threadIdx.x;
  uint32_t sum;

  for(uint32_t offset = 1; offset <= size/2; offset *= 2){
    for(int32_t i = size- 1 - tidx; i >= 0; i -= threads_g){
      if(i >= offset){
        sum = s_array[i] + s_array[i - offset];
      }else{sum = s_array[i];}
      __syncthreads();
      s_array[i] = sum;
      __syncthreads();
    }
  }
  for(int32_t i = size - 1 - threadIdx.x; i >= 0; i -= threads_g){
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

/*
The same as a small sieve but moves the count to an array rather than to a single
data point.
*/

__global__ void device::makeHistogram(uint32_t * d_primeList, uint32_t * d_histogram,
   uint32_t sieveBits, uint32_t primeListLength)
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

/*
Uses the indexing data generated in the other kernels to place the primes
in an array rather than simply counting them.  Note that this kernel requires
that all the primes have been sieved and counted previously and then sieves them
again :-/
*/

__global__ void device::makePrimeList(uint32_t * d_primeList, uint32_t * d_histogram,
   uint32_t sieveBits, uint32_t primeListLength, uint32_t maxPrime)
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
          *****************************************************
          ******Kernels for sieving small primes (< 2^40)******
          *****************************************************
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

__global__ void device::smallSieveIncompleteTop(uint32_t * d_primeList, uint64_t bottom,
   uint32_t sieveBits, uint32_t primeListLength, uint64_t top, volatile uint64_t * d_count,
    volatile uint64_t * d_blocksComplete)
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
          *************************************************
          *****Kernels used in the large sieve (>2^40)*****
          *************************************************
*/

/*
This kernel does the initial "bucket" calculations.  Data is placed in a pair of arrays:
one uint16_t array is used to hold information on how many sieves away the next
time the prime will hit the sieve and the other uint32_t array holds both the information
on (1) what index in the wheel modulo 30 the next hit will be (bits 0-2) and where
in the appropriate sieve this hit will be (bits 3-31).  Since hits are far apart
on larger ranges, it pays to have most memory accesses confined to a single array
of a smaller data type (d_away) where the memory addresses for the needed data are
contiguous and a separate array with of a larger data type (d_next) for the less
frequent accesses when hits do occur.  Actually, most of the time, a thread will
simply decrement its element of d_away without ever accessing either d_next or the list
of prime numbers.  The reduction in time to sieve ranges above 2^58 with the use
of separate arrays in this manner is ~50%.
*/

__global__ void device::getNextMult30(uint32_t * d_primeList, uint32_t * d_next, uint16_t * d_away,
   uint32_t primeListLength, uint64_t bottom, uint32_t bigSieveBits, uint8_t log2bigSieveSpan)
{
  uint32_t i = cutoff_g + threadIdx.x + blockIdx.x*blockDim.x;
  if(i < primeListLength){
    uint64_t n = 0;
    uint32_t p = d_primeList[i];
    uint64_t q = bottom/p;
    //if(p > q) q = p; // this would begin at the start of the prime, but this
    // method does not work for ranges > 2^40.  It does not seem to provides
    // a significant speedup in any event.
    n |= (q / 30) << 3; // remember, this is used as a multiplier for a prime
    n |= lookup30_g[(q % 30)];
    while(p * ((30 * (n >> 3)) + wheel30_g[(n & 7)]) < bottom) n++; // this is clunky...change if this shows promise
    q = p * ((30 * (n >> 3)) + wheel30_g[(n & 7)]) - bottom;
    d_away[i] = q >> log2bigSieveSpan;
    d_next[i] = ((q & (2*bigSieveBits-1)) << 3) + (n & 7);
  }
}

/*
The sieve for smaller primes is essentially the same below and above 2^40.  The
differences are (1) there is a defined cutoff for sieving middle primes that is
much smaller than the length of the list of sieving primes.  Experimentally, this
cutoff does not matter much between 2^15 and 2^17.  (2) Rather than counting the
primes in the sieve, the elements are copied through an atomicOr operation to the
larger (global memory) sieve.
*/

__global__ void device::bigSieveSm(uint32_t * d_primeList, uint32_t * bigSieve,
   uint64_t bottom, uint32_t primeListLength, uint32_t sieveKB)
{
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

/*
This kernel utilizes global memory as well as the impelementation of the bucket
algorithm.  Most of the time, it just decrements d_away, but when more work is
required it (1) unpacks an element of d_away (2) translates the position of the hit
to word and bit of the sieve array with bits 3-31 (3) increments the next multiple
of the prime as necessary with the array of bucket increments and crosses off the
next position of the sieve if necessary, then places information in the necessary
arrays.  The sieves away is (calculated offset)/(the size of the sieve) - 1 [for
the current sieve].  The next hit is (calculated offset)%(size of the sieve)
and the position in the bucket is just incremented each hit.
*/

__global__ void device::bigSieveLg(uint32_t * d_primeList, uint32_t * d_next, uint16_t * d_away,
   uint32_t * bigSieve, uint32_t bigSieveBits, uint32_t primeListLength, uint8_t log2bigSieveSpan)
{
  uint64_t bidx = blockIdx.x + blockIdx.y * gridDim.x;
  uint32_t i = cutoff_g + threadIdx.x + bidx*blockDim.x;

  if(i < primeListLength){
    if(d_away[i] != 0) d_away[i]--;
    else{
      uint32_t p = d_primeList[i];
      uint32_t n = d_next[i];
      uint32_t off = n >> 3;
      n &= 7;

    for(; off < (bigSieveBits << 1); n = (n + 1) & 7){
      uint32_t idx = off >> 6;
      uint16_t sidx = (off & 63) >> 1;
      atomicOr(&bigSieve[idx], (1ul << sidx));
      off += p*wheel30Inc_g[n];
    }
    n |= (off & (2*bigSieveBits-1)) << 3;
    d_next[i] = n;
    d_away[i] = (off >> log2bigSieveSpan) - 1;
    }
  }
}

/*
This is a pretty simple kernel where data from the big sieve are counted and the
words are counted for number of bits remaining as 0s.  This also zeros the bigSieve
array.  Zeroing in this way is much faster than a separate cudaMemset operation.
*/

__global__ void device::bigSieveCount(uint32_t * bigSieve, uint32_t sieveKB, volatile uint64_t * d_count)
{
  uint32_t sieveBits = sieveKB*8192;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  s_sieve[threadIdx.x] = 0;
  uint32_t count = 0;

  uint32_t blockStart = sieveWords*blockIdx.x;
  for(uint32_t i = threadIdx.x; i < sieveWords; i += threads_g){
    uint32_t x = bigSieve[i+blockStart];
    bigSieve[i+blockStart] ^= bigSieve[i+blockStart];
    for(uint8_t j = 0; j < 32; j++) count += 1 & ~(x >> j);
  }
  __syncthreads();
  s_sieve[threadIdx.x] = count;
  __syncthreads();

  device::moveCount(s_sieve, d_count);
}

/*
        ***********************************************************
        ******Kernels for making lists of primes on the device*****
        ***********************************************************
*/


__global__ void device::makeHistogram_PLout(uint32_t * d_bigSieve, uint32_t * d_histogram)
{
  uint32_t sieveWords = 256;
  __shared__ uint32_t s_sieve[256];

  device::sieveInit(s_sieve, d_bigSieve, sieveWords);
  device::countPrimes(s_sieve, sieveWords);
  __syncthreads();
  device::moveCountHist(s_sieve, d_histogram);
}

__global__ void device::makePrimeList_PLout(uint64_t * d_primeOut, uint32_t * d_histogram,
   uint32_t * d_bigSieve, uint64_t bottom, uint64_t maxPrime)
{
  uint32_t sieveWords = 256;
  __shared__ uint32_t s_sieve[256];
  __shared__ uint16_t s_counts[256];
  uint64_t bstart = bottom+blockIdx.x*sieveWords*64;

  device::sieveInit(s_sieve, d_bigSieve, sieveWords);
  __syncthreads();
  device::countPrimes(s_sieve, s_counts, sieveWords);
  __syncthreads();
  device::exclusiveScan(s_counts, sieveWords);
  device::movePrimes(s_sieve, s_counts, sieveWords, d_primeOut, d_histogram, bstart, maxPrime);
}
