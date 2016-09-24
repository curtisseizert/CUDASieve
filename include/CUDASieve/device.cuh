/*

CUDASieveDevice.cuh

Contains the __device__ functions and __constant__s for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

#include <stdint.h>

#ifndef _CUDASIEVE_DEVICE
#define _CUDASIEVE_DEVICE

// __constant__ uint32_t p3[3], p5[5], p7[7], p11[11], p13[13], p17[17], p19[19], p23[23], p29[29], p31[31], p37[37];
// __constant__ uint8_t wheel30[8], wheel30Inc[8], lookup30[30];
// __constant__ uint16_t threads;
// __constant__ uint32_t cutoff;

// __constant__ uint8_t wheel30[8];
// __constant__ uint8_t wheel30Inc[8];
// __constant__ uint8_t lookup30[30];
//
// __constant__ uint16_t threads;
// __constant__ uint32_t cutoff;

namespace device
{
  __device__ void sieveSmallPrimes(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart);
  __device__ void sieveFirst(uint32_t * s_sieve, uint32_t sieveBits);
  __device__ void sieveMiddlePrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesPL(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesBase(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesBasePL(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveInit(uint32_t * s_sieve, uint32_t sieveWords);
  __device__ void sieveInit(uint32_t * s_sieve, uint32_t * d_bigSieve, uint32_t sieveWords);
  __device__ void countPrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords); // retains the original sieve data
  __device__ void countPrimesHist(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords); // retains the original sieve data
  __device__ void countPrimes(uint32_t * s_sieve, uint32_t sieveWords); // destroys original sieve data
  __device__ void countPrimesRemBottom(uint32_t * s_sieve, uint32_t sieveWords, uint32_t bottom);
  __device__ void countTopPrimes(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint64_t bstart, uint64_t top, volatile uint64_t * d_count);
  __device__ void moveCount(uint32_t * s_sieve, volatile uint64_t * d_count);
  __device__ void moveCountHist(uint32_t * s_sieve, uint32_t * d_histogram);
  __device__ void makeBigSieve(uint32_t * bigSieve, uint32_t * s_sieve, uint32_t sieveWords);
  __device__ void exclusiveScan(uint16_t * s_array, uint32_t size);
  __device__ void exclusiveScanBig(uint32_t * s_array, uint32_t size);
  __device__ void movePrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, uint32_t * d_histogram, uint64_t bstart, uint32_t maxPrime);
  __device__ void movePrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords, uint64_t * d_primeOut, uint32_t * d_histogram, uint64_t bstart, uint64_t maxPrime);
  __device__ void movePrimesFirst(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, uint32_t * d_histogram, uint64_t bstart, uint32_t maxPrime);
}

#endif
