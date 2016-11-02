/*

primelist.cuh

header for the primeList class, which generates a list of 32 bit sieving primes
on the device

(c) 2016 Curtis Seizert <cseizert@gmail.com>

 */

 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <stdint.h>
 #include "host.hpp"

class PrimeList;
class SmallSieve;
class BigSieve;
class PrimeOutList;
class CudaSieve;
class KernelTime;

class PrimeList{

private:
  KernelTime timer;
  uint32_t primeListLength, * d_histogram = NULL, * d_histogram_lg = NULL, * d_primeListLength = NULL;
  uint32_t hist_size_lg, piHighGuess, PL_Max, maxPrime, blocks, * d_primeList = NULL, * d_bigSieve = NULL, bigSieveKB = 1024;
  uint16_t threads,sieveKB = 16;

  KernelData kerneldata;

  uint32_t * getPtr(){return d_primeList;}
  void sievePrimeList();
  void iterSieve();
  void allocate();

  PrimeList(uint32_t maxPrime);
public:

  ~PrimeList();
  static uint32_t * getSievingPrimes(uint32_t maxPrime, uint32_t & primeListLength, bool silent = 1);

};
