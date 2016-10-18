/*

global.cuh

Contains the __global__ functions in the device code for CUDASieve
Curtis Seizert  <cseizert@gmail.com>
*/

#include <stdint.h>
#include "CUDASieve/device.cuh"

#ifndef _CUDASIEVE_GLOBAL
#define _CUDASIEVE_GLOBAL

namespace device
{
  __global__ void firstPrimeList(uint32_t * d_primeList, volatile uint64_t * d_count, uint32_t sieveBits, uint32_t maxPrime);
  __global__ void exclusiveScan(uint32_t * d_array, uint32_t * d_totals, uint32_t size);
  __global__ void exclusiveScan(uint32_t * d_array, volatile uint64_t * d_count, uint32_t size);
  __global__ void increment(uint32_t * d_array, uint32_t * d_totals, uint32_t size);
  __global__ void smallSieve(uint32_t * d_primeList, volatile uint64_t * d_count, uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, volatile uint64_t * data);
  __global__ void smallSieveIncompleteTop(uint32_t * d_primeList, uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, uint64_t top, volatile uint64_t * d_count, volatile uint64_t * d_blocksComplete, bool isTop);
  __global__ void getNextMult30(uint32_t * d_primeList, uint32_t * d_next, uint16_t * d_away, uint32_t primeListLength, uint64_t bottom, uint32_t bigSieveBits, uint8_t log2bigSieveSpan);
  __global__ void bigSieveSm(uint32_t * d_primeList, uint32_t * bigSieve, uint64_t bottom, uint32_t sieveKB, uint32_t primeListLength);
  __global__ void bigSieveSm(uint32_t * d_primeList, uint32_t * bigSieve, uint64_t bottom, uint32_t sieveKB);
  __global__ void bigSieveLg(uint32_t * d_primeList, uint32_t * d_next, uint16_t * d_away, uint32_t * bigSieve, uint32_t bigSieveBits, uint32_t primeListLength, uint8_t log2bigSieveSpan);
  __global__ void bigSieveCount(uint32_t * bigSieve, uint32_t sieveKB, volatile uint64_t * d_count);
  __global__ void bigSieveCountPartial(uint32_t * bigSieve, uint32_t sieveKB, uint64_t bottom, uint64_t top, volatile uint64_t * d_count);
  __global__ void makeHistogram_PLout(uint32_t * d_bigSieve, uint32_t * d_histogram);
  __global__ void makeHistogram_PLout(uint32_t * d_bigSieve, uint32_t * d_histogram, uint64_t bottom, uint64_t maxPrime);
  __global__ void zeroBottomWord(uint32_t * d_bigSieve, uint64_t bottom, uint64_t cutoff);
  __global__ void zeroPrimeList(uint32_t * d_bigSieve, uint64_t bottom, uint32_t * d_primeList, uint32_t primeListLength);

  template <typename T>
  __global__ void makePrimeList_PLout(T * d_primeOut, uint32_t * d_histogram,
     uint32_t * d_bigSieve, uint64_t bottom, T maxPrime);

} // namespace device

#endif
