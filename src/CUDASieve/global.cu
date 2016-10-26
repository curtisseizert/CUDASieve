/*

global.cu

Contains the __global__ functions in the device code for CUDASieve
Curtis Seizert <cseizert@gmail.com>

The small sieve kernels start at about line 250, those relevant to the big sieve
start around 350.  The above statement may become inaccurate at any time.

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
__constant__ uint32_t cutoff_g = 65536;

/*
        *******************************************************
        ******Kernels for making a list of sieving primes******
        *******************************************************
*/

/*
Kernel used for creating list of primes on the device with knowledge only of
3-37.  See device.cu - sieveFirst(...) as well as SmallSieve(...) below.
*/

__global__ void device::firstPrimeList(uint32_t * d_primeList, volatile uint64_t * d_count,
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
  device::sieveFirstBottom(s_sieve, sieveBits);
  __syncthreads();
  device::countPrimesHist(s_sieve, s_counts, sieveWords);
  __syncthreads();
  device::exclusiveScanBig(s_counts, sieveWords);
  device::movePrimesFirst(s_sieve, s_counts, sieveWords, d_primeList, d_count, bstart, maxPrime);
}


__global__ void device::exclusiveScan(uint32_t * d_array, uint32_t * d_totals, uint32_t size)
{
  extern __shared__ uint32_t s_array[];
  uint32_t tidx = threadIdx.x;
  uint32_t block_offset = blockIdx.x * blockDim.x;

  if(tidx+block_offset < size) s_array[tidx] = d_array[tidx+block_offset];
  else  s_array[tidx] = 0;

  __syncthreads();

  device::inclusiveScan(s_array, (uint32_t) blockDim.x*2);
  uint32_t sum;

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

  __syncthreads();

  device::inclusiveScan(s_array, (uint32_t) blockDim.x*2);
  uint32_t sum;

  if(tidx != 0) sum = s_array[tidx-1];
  else sum = 0;
  if(tidx+block_offset < size) d_array[tidx+block_offset] = sum;
  if(threadIdx.x == 0){* d_count += s_array[blockDim.x-1];}
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
          *****************************************************
          ******Kernels for sieving small primes (< 2^40)******
          *****************************************************
*/

/*
device::smallSieve(...) is the prototype for all the shared memory based sieves that work
with sieving primes < 2^20 (which is arbitrary).  I think the process by which this operates
is very standard fare for sieves of Eratosthenes.  First, the sieve is an array of uint32_t
where each bit represents an odd number.  The bit sieve starts as all zeros, and composites
are crossed off through a bitwise or operation making their representative bits have
a value of one.  Primes remain as zeros.  Now that that's out of the way, (1) the shared memory is
zeroed; (2) the sieving primes 3-37 use a bitmask that resides in constant memory.  Keeping
separate arrays for each prime rather than a wheel array is much more efficient.  For one thing,
__constant__ memory serializes per address accessed unlike __shared__ memory, which serializes
per thread accessing the same address.  Thus it pays to have small arrays.  (3) Primes such that
37 < p <= x^(1/2) are used as sieves by calculating their offset (the position at which
they will first have a multiple in the sieve array and iteratively crossing off multiples
as appropriate.  (4) primes are counted as the number of _ones_ set in ~s_sieve[i].  The bit sieve
information is replaced by these counts, which exist per word in the sieve (5) The count is
summed in what has been called (by me) the worst sum reduction ever.
*/


__global__ void device::smallSieve(uint32_t * d_primeList, volatile uint64_t * d_count,
   uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, volatile uint64_t * d_blocksComplete)
{
  uint64_t bidx = blockIdx.x + blockIdx.y * gridDim.x;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  uint64_t bstart = bottom + bidx*sieveBits*2;

      /* this way of estimating the length of the list of sieving primes
         actually provides a huge speed-up (~30%) when covering large ranges
         e.g. 0 to 10^12 */
  float pstop = sqrtf(bstart + 2*sieveBits);
  unsigned int piHighGuess = (pstop/log(pstop))*(1+1.2762/log(pstop));
  primeListLength = min((unsigned int) primeListLength, piHighGuess);

  device::sieveInit(s_sieve, sieveWords);                 // (1)
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);  // (2)
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits, 0); // starts sieving at p^2 to not cross of primes
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits); // (3)
  __syncthreads();
  device::countPrimes(s_sieve, sieveWords);               // (4)
  __syncthreads();
  device::moveCount(s_sieve, d_count);                    // (5)
  if(threadIdx.x == 0)atomicAdd((unsigned long long *)d_blocksComplete,1ull);  // this is to show progress.  It adds negligible time.
}

__global__ void device::smallSieveIncompleteTop(uint32_t * d_primeList, uint64_t bottom,
   uint32_t sieveBits, uint32_t primeListLength, uint64_t top, volatile uint64_t * d_count,
    volatile uint64_t * d_blocksComplete, bool isTop = 1)
{
  uint64_t bidx = blockIdx.x;
  uint32_t sieveWords = sieveBits/32;
  __shared__ uint32_t s_sieve[4096];
  uint64_t bstart = bottom+bidx*sieveBits*2;

  device::sieveInit(s_sieve, sieveWords);
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits, 0);
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::countTopPrimes(s_sieve, sieveWords, bstart, top);
  device::moveCount(s_sieve, d_count, isTop);

  if(threadIdx.x == 0 && top)atomicAdd((unsigned long long *)d_blocksComplete,1ull);
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
of sieving prime numbers.  The reduction in time to sieve ranges above 2^58 with the use
of separate arrays in this manner is ~50%.
*/

__global__ void device::getNextMult30(uint32_t * d_primeList, uint32_t * d_next, uint16_t * d_away,
   uint32_t primeListLength, uint64_t bottom, uint32_t bigSieveBits, uint8_t log2bigSieveSpan)
{
  uint64_t i = cutoff_g + threadIdx.x + blockIdx.x*blockDim.x;
  if(i < primeListLength){
   uint64_t n = 0;
   uint32_t p = d_primeList[i];
   uint64_t q = bottom/p;
   n |= (q / 30) << 3;
   n |= lookup30_g[(q % 30)];
   while(p * ((30 * (n >> 3)) + wheel30_g[(n & 7)]) < bottom) n++;
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
   uint64_t bottom, uint32_t sieveKB, uint32_t primeListLength)
{
  uint32_t sieveBits = sieveKB*8192;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  uint64_t bstart = bottom + 2*blockIdx.x*sieveBits;

  float pstop = sqrtf(bstart + 2*sieveBits);
  unsigned int piHighGuess = (pstop/logf(pstop))*(1+1.2762/logf(pstop));
  primeListLength = min((unsigned int) primeListLength, piHighGuess);

  device::sieveInit(s_sieve, sieveWords);
  __syncthreads();
  device::sieveSmallPrimes(s_sieve, sieveWords, bstart);
  __syncthreads();
  if(bstart == 0) device::sieveMedPrimesBase(s_sieve, d_primeList, bstart, primeListLength, sieveBits, 0);
  else device::sieveMedPrimes(s_sieve, d_primeList, bstart, primeListLength, sieveBits);
  __syncthreads();
  device::makeBigSieve(bigSieve, s_sieve, sieveWords);
  __syncthreads();
}

/*
This kernel utilizes global memory as well as the impelementation of the bucket
algorithm.  Most of the time, it just decrements d_away, but when more work is
required it (1) unpacks an element of d_next (2) translates the position of the hit
to word and bit of the sieve array with bits 3-31 (3) increments the next multiple
of the prime as necessary with the array of wheel increments and (4) crosses off the
next position of the sieve if necessary, then places information in the necessary
arrays.  (5) The sieves away is (calculated offset)/(the size of the sieve) - 1 [for
the current sieve].  (6) The next hit is (calculated offset)%(size of the sieve)
and the position in the bucket is just incremented each hit.
*/

__global__ void device::bigSieveLg(uint32_t * d_primeList, uint32_t * d_next, uint16_t * d_away,
   uint32_t * bigSieve, uint32_t bigSieveBits, uint32_t primeListLength, uint8_t log2bigSieveSpan)
{
  uint64_t bidx = blockIdx.x + blockIdx.y * gridDim.x;
  uint64_t i = cutoff_g + threadIdx.x + bidx*blockDim.x;

  if(i < primeListLength){
    if(d_away[i] != 0) d_away[i]--;
    else{
      uint64_t p = d_primeList[i]; // If this is a 32 bit data type, as would
                                   // seem logical, there are overflows in intermediate values
                                   // above ((2^32)/6)^2 or ~2^58.83
      uint64_t n = d_next[i];
      uint64_t off = n >> 3;       // (1) and (2)
      n &= 7;

    for(; off < (bigSieveBits << 1); n = (n + 1) & 7){// (3)
      uint32_t idx = off >> 6;
      uint16_t sidx = (off & 63) >> 1;
      atomicOr(&bigSieve[idx], (1ul << sidx));        // (4)
      off += p*wheel30Inc_g[n];                       // (3)
    }
    n |= (off & (2*bigSieveBits-1)) << 3;             // (6) and gets packed in with the wheel index
    d_next[i] = n;
    d_away[i] = (off >> log2bigSieveSpan) - 1;        // (5)
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
    count += __popc(~x);
  }
  __syncthreads();
  s_sieve[threadIdx.x] = count;
  __syncthreads();

  device::moveCount(s_sieve, d_count);
}


__global__ void device::bigSieveCountPartial(uint32_t * bigSieve, uint32_t sieveKB, uint64_t bottom, uint64_t top, volatile uint64_t * d_count)
{
  uint32_t sieveBits = sieveKB*8192;
  uint32_t sieveWords = sieveBits/32;
  extern __shared__ uint32_t s_sieve[];
  s_sieve[threadIdx.x] = 0;
  uint32_t count = 0;

  uint32_t blockStart = sieveWords*blockIdx.x;
  bottom += 64*blockStart;
  for(uint32_t i = threadIdx.x; i < sieveWords; i += threads_g){
    uint32_t x = bigSieve[i+blockStart];
    for(uint8_t j = 0; j < 32; j++){
      bool r = 1u & ~(x >> j);
      if(r){
        uint64_t p = bottom + 64*i + 2*j + 1;
        if(p <= top) count++;
      }
    }
  }
  __syncthreads();
  s_sieve[threadIdx.x] = count;
  __syncthreads();

  device::moveCount(s_sieve, d_count);
}

// This zeros the appropriate part of the bottom word of the sieve when the bottom is not a multiple of 64

__global__ void device::zeroBottomWord(uint32_t * d_bigSieve, uint64_t bottom, uint64_t cutoff)
{
  uint16_t remBits = (cutoff - bottom)/2;
  uint32_t mask = (1u << remBits) -1;

  d_bigSieve[0] |= mask;
}

__global__ void device::zeroPrimeList(uint32_t * d_bigSieve, uint64_t bottom, uint32_t * d_primeList, uint32_t primeListLength)
{
  for(uint32_t i = threadIdx.x; i < primeListLength; i += blockDim.x)
  {
    uint32_t p = d_primeList[i];
    if(p >= bottom){
      uint32_t idx = (p - bottom)/64;
      uint16_t sidx = ((p - bottom)%64)/2;
      atomicAnd(&d_bigSieve[idx], ~(1u << sidx));
    }
  }
}

/*
      *************************************************************
      ****** Kernels for making lists of primes on the device *****
      *************************************************************
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

__global__ void device::makeHistogram_PLout(uint32_t * d_bigSieve, uint32_t * d_histogram, uint64_t bottom, uint64_t maxPrime)
{
  uint32_t sieveWords = 256;
  __shared__ uint32_t s_sieve[256];
  uint64_t bstart = bottom + blockIdx.x*sieveWords*64;

  device::sieveInit(s_sieve, d_bigSieve, sieveWords);
  device::countPrimesHist(s_sieve, sieveWords, bstart, maxPrime);
  __syncthreads();
  device::moveCountHist(s_sieve, d_histogram);
}

template <typename T>
__global__ void device::makePrimeList_PLout(T * d_primeOut, uint32_t * d_histogram,
   uint32_t * d_bigSieve, uint64_t bottom, T maxPrime)
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
  __syncthreads();
  device::movePrimes(s_sieve, s_counts, sieveWords, d_primeOut, d_histogram, bstart, maxPrime);
  d_bigSieve[256*blockIdx.x + threadIdx.x] = 0;
}

template __global__ void device::makePrimeList_PLout<uint64_t>(uint64_t * d_primeOut,
   uint32_t * d_histogram, uint32_t * d_bigSieve, uint64_t bottom, uint64_t maxPrime);
template __global__ void device::makePrimeList_PLout<uint32_t>(uint32_t * d_primeOut,
   uint32_t * d_histogram, uint32_t * d_bigSieve, uint64_t bottom, uint32_t maxPrime);
