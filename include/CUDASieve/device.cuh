/*

device.cuh

Contains the __device__ functions and __constant__s for CUDASieve
Curtis Seizert  <cseizert@gmail.com>
*/

#include <stdint.h>

#ifndef _CUDASIEVE_DEVICE
#define _CUDASIEVE_DEVICE

namespace device
{
  __device__ void sieveSmallPrimes(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart);
  __device__ void sieveFirstBottom(uint32_t * s_sieve, uint32_t sieveBits);
  __device__ void sieveMiddlePrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t sieveBits);
  __device__ void sieveMedPrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesBase(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits, bool forPrimeList);
  __device__ void sieveInit(uint32_t * s_sieve, uint32_t sieveWords);
  __device__ void sieveInit(uint32_t * s_sieve, uint32_t * d_bigSieve, uint32_t sieveWords);
  __device__ void countPrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords); // retains the original sieve data
  __device__ void countPrimesHist(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords); // retains the original sieve data
  __device__ void countPrimesHist(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart, uint64_t maxPrime); // destroys the original sieve data
  __device__ void countPrimes(uint32_t * s_sieve, uint32_t sieveWords); // destroys original sieve data
  __device__ void countTopPrimes(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart, uint64_t top);
  __device__ void moveCount(uint32_t * s_sieve, volatile uint64_t * d_count, bool isTop = 1);
  __device__ void moveCountHist(uint32_t * s_sieve, uint32_t * d_histogram);
  __device__ void makeBigSieve(uint32_t * bigSieve, uint32_t * s_sieve, uint32_t sieveWords);
  __device__ void inclusiveScan(uint32_t * d_array, uint32_t size);
  __device__ void exclusiveScan(uint16_t * s_array, uint32_t size);
  __device__ void exclusiveScanBig(uint32_t * s_array, uint32_t size);
  __device__ void movePrimesFirst(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, volatile uint64_t * d_count, uint64_t bstart, uint32_t maxPrime);


  template <typename T>
  __device__ void movePrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords,
    T * d_primeOut, uint32_t * d_histogram, uint64_t bstart, T maxPrime);

} // namespace device

#endif
