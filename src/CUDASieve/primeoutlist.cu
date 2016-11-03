/*
primeoutlist.cu

Host function for the primeOutList class, which creates host and device arrays
of primes for output (primeList is internal)

(c) 2016
Curtis Seizert <cseizert@gmail.com>

 */


#include "CUDASieve/primeoutlist.cuh"

PrimeOutList::PrimeOutList(CudaSieve & sieve)
{
  blocks = (sieve.bigsieve.bigSieveBits)/(32*PL_sieveWords);
  threads = 512;

  hist_size_lg = blocks/512 + 1;

  allocateDevice();
}

PrimeOutList::~PrimeOutList()
{
  safeCudaFree(d_histogram);
  safeCudaFree(d_histogram_lg);
}

void PrimeOutList::allocateDevice()
{
  d_histogram =       safeCudaMalloc(d_histogram, blocks*sizeof(uint32_t));
  d_histogram_lg =    safeCudaMalloc(d_histogram_lg, hist_size_lg*sizeof(uint32_t));

  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));
}
