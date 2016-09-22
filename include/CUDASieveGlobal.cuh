/*

CUDASieveGlobal.cuh

Contains the __global__ functions in the device code for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

#pragma once

namespace device
{
  __global__ void firstPrimeList(uint32_t * d_primeList, uint32_t * d_histogram, uint32_t sieveBits, uint32_t maxPrime);
  __global__ void inclusiveScan(uint32_t * d_array, uint16_t size);
  __global__ void exclusiveScan(uint32_t * d_array, uint32_t size);
  __global__ void exclusiveScan(uint32_t * d_array, uint32_t * d_totals, uint32_t size);
  __global__ void exclusiveScanLazy(uint32_t * s_array, uint32_t size);
  __global__ void increment(uint32_t * d_array, uint32_t * d_totals, uint32_t size);
  __global__ void makeHistogram(uint32_t * d_primeList, uint32_t * d_histogram, uint32_t sieveBits, uint32_t primeListLength);
  __global__ void makePrimeList(uint32_t * d_primeList, uint32_t * d_histogram, uint32_t sieveBits, uint32_t primeListLength, uint32_t maxPrime);
  __global__ void smallSieve(uint32_t * d_primeList, volatile uint64_t * d_count, uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, volatile uint64_t * data);
  __global__ void smallSieveIncomplete(uint32_t * d_primeList, uint64_t * d_count, uint64_t kernelBottom, uint32_t sieveBits, uint32_t primeListLength, uint64_t bottom);
  __global__ void smallSieveIncompleteTop(uint32_t * d_primeList, uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, uint64_t top, volatile uint64_t * d_count, volatile uint64_t * d_blocksComplete);
  __global__ void smallSieveCopy(uint32_t * d_primeList, uint64_t * d_count, uint64_t bottom, uint32_t sieveBits, uint32_t primeListLength, uint32_t * sieveOut);
  __global__ void getNextMult30(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t primeListLength, uint64_t bottom);
  __global__ void getNextMult30_test(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t primeListLength, uint64_t bottom, uint32_t bigSieveBits);
  __global__ void bigSieveSm(uint32_t * d_primeList, uint32_t * bigSieve, uint64_t bottom, uint32_t primeListLength, uint32_t sieveKB);
  __global__ void bigSieveLg(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t * bigSieve, uint64_t bstart, uint32_t bigSieveSize, uint32_t primeListLength, uint32_t sieveKB);
  __global__ void bigSieveLg_test(uint32_t * d_primeList, uint64_t * d_nextMult, uint32_t * bigSieve, uint64_t bstart, uint32_t bigSieveBits, uint32_t primeListLength, uint32_t sieveKB);
  __global__ void bigSieveCount(uint32_t * bigSieve, uint32_t sieveKB, volatile uint64_t * d_count);
}
