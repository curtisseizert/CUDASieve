/*

CUDASieveDevice.cu

Contains the __device__ functions and __constant__s for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDASieve/global.cuh"
#include "CUDASieve/device.cuh"

/*
          ############################################
          ###### Bitmask arrays for small sieve ######
          ############################################

As far as I am aware, constant cache is the best place for these, although
the access pattern to them would not seem to be the best for this.  Actually,
it should be more suited to texture memory, but I don't know how to use that.
In any event, the profiler says this is not a big deal.

*/

__constant__ uint32_t p3[3] = {2454267026ul, 613566756ul, 1227133513ul};
__constant__ uint32_t p5[5] = {138547332ul, 1108378657ul, 277094664ul, 2216757314ul,
                              554189328ul};
__constant__ uint32_t p7[7] = {2164392968ul, 135274560ul, 1082196484ul, 67637280ul,
                              541098242ul, 33818640ul, 270549121ul};
__constant__ uint32_t p11[11] = {134283296ul, 268566592ul, 537133184ul, 1074266368ul,
                                2148532736ul, 2098176ul, 4196353ul, 8392706ul, 16785412ul,
                                33570824ul, 67141648ul};
__constant__ uint32_t p13[13] = {524352ul, 67117057ul, 1048704ul, 134234114ul, 2097408ul,
                                268468228ul, 4194816ul, 536936456ul, 8389632ul, 1073872912ul,
                                16779264ul, 2147745824ul, 33558528ul};
__constant__ uint32_t p17[17] = {33554688ul, 134218752ul, 536875008ul, 2147500032ul,
                                65536ul, 262146ul, 1048584ul, 4194336ul, 16777344ul,
                                67109376ul, 268437504ul, 1073750016ul, 32768ul, 131073ul,
                                524292ul, 2097168ul, 8388672ul};
__constant__ uint32_t p19[19] = {268435968ul, 32768ul, 2097156ul, 134217984ul, 16384ul,
                                1048578ul, 67108992ul, 8192ul, 524289ul, 33554496ul, 2147487744ul,
                                262144ul, 16777248ul, 1073743872ul, 131072ul, 8388624ul,
                                536871936ul, 65536ul, 4194312ul};
__constant__ uint32_t p23[23] = {2048ul, 33554436ul, 65536ul, 1073741952ul, 2097152ul,
                                4096ul, 67108872ul, 131072ul, 2147483904ul, 4194304ul,
                                8192ul, 134217744ul, 262144ul, 512ul, 8388609ul, 16384ul,
                                268435488ul, 524288ul, 1024ul, 16777218ul, 32768ul,
                                536870976ul, 1048576ul};
__constant__ uint32_t p27[27] = {8192ul, 256ul, 1073741832ul, 33554432ul, 1048576ul,
                                32768ul, 1024ul, 32ul, 134217729ul, 4194304ul, 131072ul,
                                4096ul, 128ul, 536870916ul, 16777216ul, 524288ul, 16384ul,
                                512ul, 2147483664ul, 67108864ul, 2097152ul, 65536ul,
                                2048ul, 64ul, 268435458ul, 8388608ul, 262144ul};
__constant__ uint32_t p29[29] = {16384ul, 2048ul, 256ul, 32ul, 2147483652ul, 268435456ul,
                                33554432ul, 4194304ul, 524288ul, 65536ul, 8192ul, 1024ul,
                                128ul, 16ul, 1073741826ul, 134217728ul, 16777216ul,
                                2097152ul, 262144ul, 32768ul, 4096ul, 512ul, 64ul,
                                8ul, 536870913ul, 67108864ul, 8388608ul, 1048576ul, 131072ul};
__constant__ uint32_t p31[31] = {32768ul, 16384ul, 8192ul, 4096ul, 2048ul, 1024ul,
                                512ul, 256ul, 128ul, 64ul, 32ul, 16ul, 8ul, 4ul, 2ul,
                                2147483649ul, 1073741824ul, 536870912ul, 268435456ul,
                                134217728ul, 67108864ul, 33554432ul, 16777216ul, 8388608ul,
                                4194304ul, 2097152ul, 1048576ul, 524288ul, 262144ul,
                                131072ul, 65536ul};
__constant__ uint32_t p37[37] = {262144ul, 8388608ul, 268435456ul, 0ul, 2ul, 64ul, 2048ul,
                                65536ul, 2097152ul, 67108864ul, 2147483648ul, 0ul, 16ul,
                                512ul, 16384ul, 524288ul, 16777216ul, 536870912ul, 0ul,
                                4ul, 128ul, 4096ul, 131072ul, 4194304ul, 134217728ul,
                                0ul, 1ul, 32ul, 1024ul, 32768ul, 1048576ul, 33554432ul,
                                1073741824ul, 0ul, 8ul, 256ul, 8192ul};
__constant__ uint8_t wheel30[8] = {1,7,11,13,17,19,23,29};
__constant__ uint8_t wheel30Inc[8] = {6,4,2,4,2,4,6,2};
__constant__ uint8_t lookup30[30] = {0,0,0,0,0,0,0,1,0,0,0,2,0,3,0,0,0,4,0,5,0,0,0,6,0,0,0,0,0,7};

__constant__ uint16_t threads = 256;
__constant__ uint32_t cutoff = 65536;

/*
            #############################################
            ###### Bitmask sieve for small primes #######
            #############################################

This is an idea used in Ben Buhrow's implementation and it provides a considerable
(~4x) speedup vs. sieving these primes individually.  For some reason, unrolling
this loop does not increase the speed, possibly due to divergence.  CUDASieve has
a janky little c++ (utils/bitsievegen.cpp) script for generating such bitmasks and
outputting them to the standard out, because doing this by hand would be an onerous
task.  This should allow anyone interested to try their own optimizations based
on chaning parameters of the sieve (size of words, etc.) without having to do this
part by hand.
*/

__device__ void device::sieveSmallPrimes(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart)
{
  #pragma unroll 1
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads){
    uint64_t j = i + bstart/64; // 64 is 32 bits per uint32_t*2(for only odds)
    s_sieve[i] |= p3[j%3];
    s_sieve[i] |= p5[j%5];
    s_sieve[i] |= p7[j%7];  // sieving with 37 in this way provides a consistent
    s_sieve[i] |= p11[j%11]; //  1-2% speedup over going up to only 31. Going
    s_sieve[i] |= p13[j%13]; // higher than 37 slows things down.  Using a premade
    s_sieve[i] |= p17[j%17]; //  wheel-type bitmask here is considerably slower
    s_sieve[i] |= p19[j%19]; // than sieving with each small prime individually.
    s_sieve[i] |= p23[j%23];
    s_sieve[i] |= p29[j%29];
    s_sieve[i] |= p31[j%31];
    s_sieve[i] |= p37[j%37];
  }
}
/*
          ######################################################
          ###### Specialized sieve for making primelist ########
          ######################################################

This sieve uses odds to cross off composites before a list of sieving primes is
created.  For all 16 bit primes, this only amounts to the odds between 41 and 255
starting at their squares.  While this would perhaps be more efficient using a
wheel, it is so fast anyway that who cares.  The entire process of generating the
first list takes only 0.1 ms even on the relatively weak GTX 750.
*/


__device__ void device::sieveFirst(uint32_t * s_sieve, uint32_t sieveBits)
{
  if(threadIdx.x == 0) atomicOr(&s_sieve[0], 1u);
  uint32_t p = 41 + 2*threadIdx.x;
  uint32_t off = p * p/2;
  for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
}

/*        #######################################################
          ###### Sieve functions for primes 37 < p < 2^20 #######
          #######################################################

These functions are meant to do the majority of the crossing-off in large sieves
to exploit the better latency characteristics of shared vs. global memory.  Their
calculation of the first number to mark off is based on modulo p, so it becomes
very inefficient for large prime numbers.  However, storing buckets for such primes
is not feasible with this implementation because of the large number of blocks
active at one time, so the modulo calculation is still apparently the best way
to deal with these relatively small primes in order to cross off their multiples
in shared memory.
*/

__device__ void device::sieveMedPrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart,
  uint32_t primeListLength, uint32_t sieveBits)
{
  #pragma unroll 1
  for(uint32_t pidx = threadIdx.x; pidx < primeListLength; pidx += threads){ // this accepts a list of sieving primes > 37
    uint32_t p = d_primeList[pidx];
    uint32_t off = p - bstart % p;
    if(off%2==0) off += p;
    off = off >> 1; // convert offset to align with half sieve
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
  }
}

__device__ void device::sieveMedPrimesPL(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart,
  uint32_t primeListLength, uint32_t sieveBits)
{
  for(uint32_t pidx = threadIdx.x; pidx < primeListLength; pidx += threads){ // this accepts a list of sieving primes > 37
    uint32_t p = d_primeList[pidx];
    if(p*p > bstart+2*sieveBits) break;
    uint32_t off = p - bstart % p;
    if(off%2==0) off += p;
    off = off >> 1; // convert offset to align with half sieve
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
  }
}

__device__ void device::sieveMiddlePrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart,
  uint32_t sieveBits)
{
  for(uint32_t pidx = threadIdx.x; pidx < cutoff; pidx += threads){ // this accepts a list of sieving primes > 37
      uint32_t p = d_primeList[pidx];
    uint32_t off = p - bstart % p;
    if(off%2==0) off += p;
    off = (off-1)/2; // convert offset to align with half sieve
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
  }
}

__device__ void device::sieveMedPrimesBase(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart,
  uint32_t primeListLength, uint32_t sieveBits)
{
  if(threadIdx.x == 0){
    //s_sieve[0] |= 1;
    s_sieve[0] ^= 314222ul; //this is a bitmask for primes 3-37
  }
  for(uint32_t pidx = threadIdx.x; pidx < primeListLength; pidx += threads){// this accepts a list of sieving primes > 37
    uint32_t p = d_primeList[pidx];
    uint32_t off = p*p/2; // convert offset to align with half sieve
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
  }
}

__device__ void device::sieveMedPrimesBasePL(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart,
  uint32_t primeListLength, uint32_t sieveBits)
{
  if(threadIdx.x == 0){
    s_sieve[0] |= 1;
    //s_sieve[0] ^= 52078ul; //this is a bitmask for primes 3-31 (use if you want to include 3-31 in the list of primes on the device)
    // P.S. Trying to run the sieve with such a list will slow it down by about 4x
  }
  for(uint32_t pidx = threadIdx.x; pidx < primeListLength; pidx += threads){// this accepts a list of sieving primes > 37
    uint32_t p = d_primeList[pidx];
    uint32_t off = p*p/2; // convert offset to align with half sieve
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
  }
}
/*            ##################################################
              ##### Functions to zero or load SMem sieves ######
              ##################################################

*/

__device__ void device::sieveInit(uint32_t * s_sieve, uint32_t sieveWords)
{
  //#pragma unroll
  for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x)
    s_sieve[i] ^= s_sieve[i];
}

__device__ void device::sieveInit(uint32_t * s_sieve, uint32_t * d_bigSieve, uint32_t sieveWords)
{
  uint32_t blockStart = sieveWords*blockIdx.x;

  for(uint32_t i = threadIdx.x; i < sieveWords; i += threads){
    s_sieve[i] = d_bigSieve[blockStart+i];
  }
}

/*                    ##################################
                      ######  Counting functions #######
                      ##################################

Note: some of these (as indicated) destroy the sieve data, and they are differentiated
from those that don't by overloading.

*/

__device__ void device::countPrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords) // retains the original sieve data, reduces to block size
{
  uint16_t count = 0;
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
  {
    uint32_t s = ~s_sieve[i];
    uint8_t c = 0;
    for(; s; c++) s &= s -1;
    count += c;
  }
  __syncthreads();
  s_counts[threadIdx.x] = count;
}

__device__ void device::countPrimesHist(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords) // retains the original sieve data, maintains primes per word
{
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
  {
    uint32_t s = ~s_sieve[i];
    uint8_t c = 0;
    for(; s; c++) s &= s -1;
    s_counts[i] = c;
  }
  __syncthreads();
}

__device__ void device::countPrimes(uint32_t * s_sieve, uint32_t sieveWords) // destroys original sieve data
{
  uint16_t count = 0;
  #pragma unroll
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
  {
    uint32_t s = ~s_sieve[i];
    s_sieve[i] ^= s_sieve[i];
    uint8_t c = 0;
    for(; s; c++) s &= s -1;
    count += c;
  }
  __syncthreads();
  s_sieve[threadIdx.x] = count;
}

__device__ void device::countTopPrimes(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint64_t bstart, uint64_t top, volatile uint64_t * d_count)
{
  uint32_t count = 0;
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads){
      uint16_t c = s_counts[i];
      uint32_t s = ~s_sieve[i];
      __syncthreads();
      if(threadIdx.x == 0) s_sieve[0] |= ~s_sieve[0];
      for(uint16_t j = 0; j < 32; j++){
        if(1 & (s >> j)){
          uint64_t p = bstart + 64*i + 2*j + 1;
          if(p > top && count == 0) {count = c; break;}
          c++;
        }
      }
    }
  __syncthreads();
  if(count != 0 && threadIdx.x > 1) atomicMin(&s_sieve[0], count);
  __syncthreads();
  if(threadIdx.x == 0 && ~s_sieve[0] != 0) atomicAdd((unsigned long long *)d_count, s_sieve[0]);
}

/*
      ##########################################################################
      ###### Functions for moving the count or sieve out of shared memory ######
      ##########################################################################

*/

__device__ void device::moveCount(uint32_t * s_sieve, volatile uint64_t * d_count)
{
  if(threadIdx.x == 0)
  {
    uint64_t count = 0;
    for(uint16_t i=0; i < threads; i++) count += s_sieve[i];
    atomicAdd((unsigned long long *)d_count, count);
  }
  __syncthreads();
}

__device__ void device::moveCountHist(uint32_t * s_sieve, uint32_t * d_histogram)
{
  if(threadIdx.x == 0)
  {
    uint64_t count = 0;
    for(uint16_t i=0; i < threads; i++)
      count += s_sieve[i];
    d_histogram[blockIdx.x] = count;
  }
  __syncthreads();
}

__device__ void device::makeBigSieve(uint32_t * bigSieve, uint32_t * s_sieve, uint32_t sieveWords)
{
  uint32_t blockStart = sieveWords*blockIdx.x;
  for(uint32_t i = threadIdx.x; i < sieveWords; i += threads)
    atomicOr(&bigSieve[i+blockStart], s_sieve[i]);
}

/*        ##################################################################
          ###### Functions for generating the list of sieving primes #######
          ##################################################################
*/

__device__ void device::exclusiveScan(uint16_t * s_array, uint32_t size)
{
  uint32_t tidx = threadIdx.x;
  uint32_t sum;

  for(uint32_t offset = 1; offset <= size/2; offset *= 2){
    if(tidx >= offset){
      sum = s_array[tidx] + s_array[tidx - offset];
    }else{sum = s_array[tidx];}
    __syncthreads();
    s_array[tidx] = sum;
    __syncthreads();
  }
  if(threadIdx.x != 0) sum = s_array[threadIdx.x-1];
  else sum = 0;
  __syncthreads();
  s_array[threadIdx.x] = sum;
}

/*
Exclusive scan function suitable for medium sized lists, as part of its operation
is serialized to avoid needing multiple stages whenever the number of items
to be incremented is greater than the number of threads
*/

__device__ void device::exclusiveScanBig(uint32_t * s_array, uint32_t size)
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
    __syncthreads();
  }
}

/*
IMPORTANT: movePrimes requires at least one prime to be greater than maxPrime to
correctly report the count.  However, this is only important in the case of
primelist generation with the 32 bit version.  This may change when the big sieve
gets support for better range granularity.
*/

__device__ void device::movePrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, uint32_t * d_histogram, uint64_t bstart, uint32_t maxPrime)
{
   // this is meant for when words per array == number of threads
  uint16_t i = threadIdx.x;
  uint16_t c = 0;
  uint32_t s = ~s_sieve[i];
  uint32_t idx = d_histogram[blockIdx.x] + s_counts[i];
  __syncthreads();
  if(threadIdx.x == 0) s_sieve[0] |= ~s_sieve[0];
  for(uint16_t j = 0; j < 32; j++){
    if(1 & (s >> j)){
      uint32_t p = bstart + 64*i + 2*j + 1;
      if(p > maxPrime) atomicMin(&s_sieve[0], idx+c);
      else d_primeList[idx+c] = p;
      c++;
    }
  }
  __syncthreads();
  if(threadIdx.x == 0 && ~s_sieve[0] != 0) d_histogram[blockIdx.x] = s_sieve[0];
}

__device__ void device::movePrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords, uint64_t * d_primeOut, uint32_t * d_histogram, uint64_t bstart, uint64_t maxPrime)
{
   // this is meant for when words per array == number of threads
  uint16_t i = threadIdx.x;
  uint16_t c = 0;
  uint32_t s = ~s_sieve[i];
  uint32_t idx = d_histogram[blockIdx.x] + s_counts[i];
  __syncthreads();
  if(threadIdx.x == 0) s_sieve[0] |= ~s_sieve[0];
  for(uint16_t j = 0; j < 32; j++){
    if(1 & (s >> j)){
      uint64_t p = bstart + 64*i + 2*j + 1;
      if(p > maxPrime) atomicMin(&s_sieve[0], idx+c);
      else d_primeOut[idx+c] = p;
      c++;
    }
  }
  __syncthreads();
  if(threadIdx.x == 0 && ~s_sieve[0] != 0) d_histogram[blockIdx.x] = s_sieve[0];
}

/*
This is the version of the movePrimes function that is used for generating the original
list of sieving primes to be used by the next round of list generating functions.  Unlike
above versions of the function, it supports array sizes greater than the number of
threads in the block.
*/

__device__ void device::movePrimesFirst(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, uint32_t * d_histogram, uint64_t bstart, uint32_t maxPrime)
{
   // this is for when words per array != number of threads
   uint16_t c;
    for(uint16_t i = threadIdx.x; i < sieveWords; i += threads){
    c = s_counts[i];
    uint32_t s = ~s_sieve[i];
    __syncthreads();
    if(i == 0) s_sieve[0] |= ~s_sieve[0];
    for(uint16_t j = 0; j < 32; j++){
      if(1 & (s >> j)){
        uint32_t p = bstart + 64*i + 2*j + 1;
        if(p > maxPrime) atomicMin(&s_sieve[0], c);
        else d_primeList[c] = p;
        c++;
      }
    }
  }
  if(threadIdx.x == 0 && ~s_sieve[0] != 0) d_histogram[0] = s_sieve[0];
  if((threadIdx.x == blockDim.x - 1) && ~s_sieve[0] == 0) d_histogram[0] = c;
}
