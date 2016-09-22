/*

CUDASieveDevice.cu

Contains the __device__ functions and __constant__s for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

#include "CUDASieveDevice.cuh"

  __device__ void device::sieveSmallPrimes(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart)
  {
    #pragma unroll 1
    for(uint16_t i = threadIdx.x; i < sieveWords; i += threads){
      uint64_t j = i + bstart/64; // 64 is 32 bits per uint32_t*2(for only odds)
      s_sieve[i] |= p3[j%3];
      s_sieve[i] |= p5[j%5];
      s_sieve[i] |= p7[j%7];
      s_sieve[i] |= p11[j%11]; // sieving with 37 in this way provides a consistent 1-2% speedup over going up to only 31.
      s_sieve[i] |= p13[j%13]; // going higher than 37 slows things down.  Using a premade wheel-type bitmask here is considerably slower
      s_sieve[i] |= p17[j%17]; // than sieving with each small prime individually.  Thanks to Ben Buhrow (bbuhrow@gmail.com) for this idea.
      s_sieve[i] |= p19[j%19];
      s_sieve[i] |= p23[j%23];
      s_sieve[i] |= p29[j%29];
      s_sieve[i] |= p31[j%31];
      s_sieve[i] |= p37[j%37];
    }
  }

  __device__ void device::sieveFirst(uint32_t * s_sieve, uint32_t sieveBits)
  {
    if(threadIdx.x == 0) atomicOr(&s_sieve[0], 1u);
    uint32_t p = 33 + 2*threadIdx.x;
    uint32_t off = p * p/2;
    for(; off < sieveBits; off += p) atomicOr(&s_sieve[off >> 5], (1u << (off & 31)));
  }


  __device__ void device::sieveMedPrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart,
    uint32_t primeListLength, uint32_t sieveBits)
  {
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
    uint32_t primeListLength, uint32_t sieveBits)
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

  __device__ void device::sieveInit(uint32_t * s_sieve, uint32_t sieveWords)
  {
    //#pragma unroll
    for(uint16_t i = threadIdx.x; i < sieveWords; i += blockDim.x)
      s_sieve[i] ^= s_sieve[i];
  }

  /*
                      Counting functions
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
                      Functions for moving the count or sieve out of shared memory
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

  /*
                      Functions for generating the list of sieving primes
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
