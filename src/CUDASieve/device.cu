/*

device.cu

Contains the __device__ functions and __constant__s for CUDASieve
by Curtis Seizert <cseizert@gmail.com>

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>

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

__constant__ uint32_t p3[3] =    {0x92492492, 0x24924924, 0x49249249};
__constant__ uint32_t p5[5] =    {0x08421084, 0x42108421, 0x10842108, 0x84210842,
                                  0x21084210};
__constant__ uint32_t p7[7] =    {0x81020408, 0x08102040, 0x40810204, 0x04081020,
                                  0x20408102, 0x02040810, 0x10204081};
__constant__ uint32_t p11[11] =  {0x08010020, 0x10020040, 0x20040080, 0x40080100,
                                  0x80100200, 0x00200400, 0x00400801, 0x00801002,
                                  0x01002004, 0x02004008, 0x04008010};
__constant__ uint32_t p13[13] =  {0x00080040, 0x04002001, 0x00100080, 0x08004002,
                                  0x00200100, 0x10008004, 0x00400200, 0x20010008,
                                  0x00800400, 0x40020010, 0x01000800, 0x80040020,
                                  0x02001000};
__constant__ uint32_t p17[17] =  {0x02000100, 0x08000400, 0x20001000, 0x80004000,
                                  0x00010000, 0x00040002, 0x00100008, 0x00400020,
                                  0x01000080, 0x04000200, 0x10000800, 0x40002000,
                                  0x00008000, 0x00020001, 0x00080004, 0x00200010,
                                  0x00800040};
__constant__ uint32_t p19[19] =  {0x10000200, 0x00008000, 0x00200004, 0x08000100,
                                  0x00004000, 0x00100002, 0x04000080, 0x00002000,
                                  0x00080001, 0x02000040, 0x80001000, 0x00040000,
                                  0x01000020, 0x40000800, 0x00020000, 0x00800010,
                                  0x20000400, 0x00010000, 0x00400008};
__constant__ uint32_t p23[23] =  {0x00000800, 0x02000004, 0x00010000, 0x40000080,
                                  0x00200000, 0x00001000, 0x04000008, 0x00020000,
                                  0x80000100, 0x00400000, 0x00002000, 0x08000010,
                                  0x00040000, 0x00000200, 0x00800001, 0x00004000,
                                  0x10000020, 0x00080000, 0x00000400, 0x01000002,
                                  0x00008000, 0x20000040, 0x00100000};
__constant__ uint32_t p29[29] =  {0x00004000, 0x00000800, 0x00000100, 0x00000020,
                                  0x80000004, 0x10000000, 0x02000000, 0x00400000,
                                  0x00080000, 0x00010000, 0x00002000, 0x00000400,
                                  0x00000080, 0x00000010, 0x40000002, 0x08000000,
                                  0x01000000, 0x00200000, 0x00040000, 0x00008000,
                                  0x00001000, 0x00000200, 0x00000040, 0x00000008,
                                  0x20000001, 0x04000000, 0x00800000, 0x00100000,
                                  0x00020000};
__constant__ uint32_t p31[31] =  {0x00008000, 0x00004000, 0x00002000, 0x00001000,
                                  0x00000800, 0x00000400, 0x00000200, 0x00000100,
                                  0x00000080, 0x00000040, 0x00000020, 0x00000010,
                                  0x00000008, 0x00000004, 0x00000002, 0x80000001,
                                  0x40000000, 0x20000000, 0x10000000, 0x08000000,
                                  0x04000000, 0x02000000, 0x01000000, 0x00800000,
                                  0x00400000, 0x00200000, 0x00100000, 0x00080000,
                                  0x00040000, 0x00020000, 0x00010000};
__constant__ uint32_t p37[37] =  {0x00040000, 0x00800000, 0x10000000, 0x00000000,
                                  0x00000002, 0x00000040, 0x00000800, 0x00010000,
                                  0x00200000, 0x04000000, 0x80000000, 0x00000000,
                                  0x00000010, 0x00000200, 0x00004000, 0x00080000,
                                  0x01000000, 0x20000000, 0x00000000, 0x00000004,
                                  0x00000080, 0x00001000, 0x00020000, 0x00400000,
                                  0x08000000, 0x00000000, 0x00000001, 0x00000020,
                                  0x00000400, 0x00008000, 0x00100000, 0x02000000,
                                  0x40000000, 0x00000000, 0x00000008, 0x00000100,
                                  0x00002000};

__constant__ uint8_t wheel30[8] = {1,7,11,13,17,19,23,29};
__constant__ uint8_t wheel30Inc[8] = {6,4,2,4,2,4,6,2};
__constant__ uint8_t lookup30[30] = {0,0,0,0,0,0,0,1,0,0,0,2,0,3,0,0,0,4,0,5,0,0,
                                     0,6,0,0,0,0,0,7};

__constant__ uint16_t threads = 256;

/*
            #############################################
            ###### Bitmask sieve for small primes #######
            #############################################

This is an idea used in Ben Buhrow's implementation and it provides a considerable
(~4x) speedup vs. sieving these primes individually.  For some reason, unrolling
this loop does not increase the speed, possibly due to divergence.  CUDASieve has
a janky little c++ (utils/bitsievegen.cpp) program for generating such bitmasks and
outputting them to the standard out, because doing this by hand would be an onerous
task.  This should allow anyone interested to try their own optimizations based
on chaning parameters of the sieve (size of words, etc.) without having to do this
part by hand.
*/


__device__ void device::sieveSmallPrimes(uint32_t * s_sieve, uint32_t sieveWords,
                                         uint64_t bstart)
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


__device__ void device::sieveFirstBottom(uint32_t * s_sieve, uint32_t sieveBits)
{
  if(threadIdx.x == 0 && blockIdx.x == 0) atomicOr(&s_sieve[0], 1u);
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

__device__ void device::sieveMedPrimes(uint32_t * s_sieve, uint32_t * d_primeList,
                                       uint64_t bstart, uint32_t primeListLength,
                                       uint32_t sieveBits)
{
  for(uint32_t pidx = threadIdx.x; pidx < primeListLength; pidx += threads){
    // this accepts a list of sieving primes > 37
    uint32_t p = d_primeList[pidx];
    uint32_t off = p - bstart % p;
    if(off%2==0) off += p;
    off = off >> 1; // convert offset to align with half sieve
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
    // this loop takes ~75% of the kernel time
  }
}

__device__ void device::sieveMedPrimesBase(uint32_t * s_sieve, uint32_t * d_primeList,
                                           uint64_t bstart, uint32_t primeListLength,
                                           uint32_t sieveBits, bool forPrimeList = 0)
{
  if(threadIdx.x == 0){
    if(forPrimeList) s_sieve[0] |= 1; // cross off one
    else s_sieve[0] ^= 0x0004cb6e; // un-cross off 3-37 if regular sieve
  }
  for(uint32_t pidx = threadIdx.x; pidx < primeListLength; pidx += threads){
    // this accepts a list of sieving primes > 37
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

__device__ void device::sieveInit(uint32_t * s_sieve, uint32_t * d_bigSieve,
                                  uint32_t sieveWords)
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

 // retains the original sieve data, reduces to block size
__device__ void device::countPrimes(uint32_t * s_sieve, uint16_t * s_counts,
                                    uint32_t sieveWords)
{
  uint16_t count = 0;
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
  {
    uint32_t s = ~s_sieve[i];
    count += __popc(s);
  }
  __syncthreads();
  s_counts[threadIdx.x] = count;
}

// retains the original sieve data, maintains primes per word
__device__ void device::countPrimesHist(uint32_t * s_sieve, uint32_t * s_counts,
                                        uint32_t sieveWords)
{
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
    s_counts[i] = __popc(~s_sieve[i]);

  __syncthreads();
}

// destroys the original sieve data, maintains primes per word
__device__ void device::countPrimesHist(uint32_t * s_sieve, uint32_t sieveWords,
                                        uint64_t bstart, uint64_t maxPrime)
{
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
  {
    uint32_t s = ~s_sieve[i];
    uint16_t count = 0;
    for(uint16_t j = 0; j < 32; j++){
      if(1 & (s >> j)){
        uint64_t p = bstart + 64*i + 2*j + 1;
        if(p <= maxPrime) count++; // only count primes less than top
      }
    }
    s_sieve[i] = count;
  }
  __syncthreads();
}

// destroys original sieve data
__device__ void device::countPrimes(uint32_t * s_sieve, uint32_t sieveWords)
{
  uint16_t count = 0;
  #pragma unroll
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads)
  {
    uint32_t s = ~s_sieve[i];
    s_sieve[i] ^= s_sieve[i];
    count += __popc(s);
  }
  __syncthreads();
  s_sieve[threadIdx.x] = count;
}

__device__ void device::countTopPrimes(uint32_t * s_sieve, uint32_t sieveWords,
   uint64_t bstart, uint64_t top)
{
  uint32_t count = 0;
  for(uint16_t i = threadIdx.x; i < sieveWords; i += threads){
    uint32_t s = ~s_sieve[i];
    s_sieve[i] ^= s_sieve[i]; // to make a number that can't be the result in
                              // order to see if it has been modified later
    for(uint16_t j = 0; j < 32; j++){
      if(1 & (s >> j)){
        uint64_t p = bstart + 64*i + 2*j + 1;
        if(p <= top) count++; // only count primes less than top
      }
    }
  }
  s_sieve[threadIdx.x] = count;
  __syncthreads();
}

/*
      ##########################################################################
      ###### Functions for moving the count or sieve out of shared memory ######
      ##########################################################################

*/

__device__ void device::moveCount(uint32_t * s_sieve, volatile uint64_t * d_count, bool isTop)
{
  if(threadIdx.x == 0)
  {
    uint64_t count = 0;
    for(uint16_t i=0; i < threads; i++) count += s_sieve[i];
    if(isTop) atomicAdd((unsigned long long *)d_count, count);
    else      atomicAdd((unsigned long long *)d_count, (int) -count);
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

__device__ void device::makeBigSieve(uint32_t * bigSieve, uint32_t * s_sieve,
                                     uint32_t sieveWords)
{
  uint32_t blockStart = sieveWords*blockIdx.x;
  for(uint32_t i = threadIdx.x; i < sieveWords; i += threads)
    atomicOr(&bigSieve[i+blockStart], s_sieve[i]);
}

/*        ##################################################################
          ###### Functions for generating the list of sieving primes #######
          ##################################################################
*/


__device__ void device::inclusiveScan(uint32_t * s_array, uint32_t size)
{
  uint32_t tidx = threadIdx.x;

  uint32_t sum;

  for(uint32_t offset = 1; offset <= size/2; offset *= 2){
    if(tidx >= offset){
      sum = s_array[threadIdx.x] + s_array[threadIdx.x - offset];
    }else{sum = s_array[threadIdx.x];}
    __syncthreads();
    s_array[threadIdx.x] = sum;
    __syncthreads();
  }
}

// 16 bit data type
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

template <typename T>
__device__ void device::movePrimes(uint32_t * s_sieve, uint16_t * s_counts,
                                   uint32_t sieveWords, T * d_primeOut,
                                   uint32_t * d_histogram, uint64_t bstart, T maxPrime)
{
  // this is meant for when words per array == number of threads
  uint16_t i = threadIdx.x;
  uint16_t c = 0;                 // used to hold the count
  uint32_t s = ~s_sieve[i];       // primes are now represented as 1s
  // offset for where each thread should put its first prime
  uint32_t idx = d_histogram[blockIdx.x] + s_counts[i];
  __syncthreads();
  // s_sieve[0] is made ~0 so we can tell if it has been changed
  if(threadIdx.x == 0) s_sieve[0] |= ~s_sieve[0];
  for(uint16_t j = 0; j < 32; j++){
    if(1 & (s >> j)){                       // if prime
      T p = bstart + 64*i + 2*j + 1;        // calculate value
      // if value is above threshold, submit and break
      if(p > maxPrime) {atomicMin(&s_sieve[0], idx+c); break;}
      else d_primeOut[idx+c] = p; // otherwise copy p to the output array
      c++;                        // incrememnt count
    }
  }

  __syncthreads();
  if(threadIdx.x == blockDim.x-1){
    if(~s_sieve[0] != 0) d_histogram[blockIdx.x] = s_sieve[0];
    else d_histogram[blockIdx.x] = idx + c;
  }
  // this covers up one since the sieve only holds odds
  if(threadIdx.x == 1 && bstart == 0) d_primeOut[0] = 2;
}

/*
This is the version of the movePrimes function that is used for generating the original
list of sieving primes to be used by the next round of list generating functions.  Unlike
above versions of the function, it supports array sizes greater than the number of
threads in the block.  I could probably get rid of one of the above.
*/

__device__ void device::movePrimesFirst(uint32_t * s_sieve, uint32_t * s_counts,
                                        uint32_t sieveWords, uint32_t * d_primeList,
                                        volatile uint64_t * d_count, uint64_t bstart,
                                        uint32_t maxPrime)
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
  __syncthreads();
  if(threadIdx.x == 0 && ~s_sieve[0] != 0) atomicAdd((unsigned long long *)d_count, s_sieve[0] );
  if((threadIdx.x == blockDim.x - 1) && ~s_sieve[0] == 0) atomicAdd((unsigned long long *)d_count, c);
}
