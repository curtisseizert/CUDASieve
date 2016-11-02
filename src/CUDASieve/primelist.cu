/*

primelist.cu

source for the primeList class, which generates a list of 32 bit sieving primes
on the device

(c) 2016 Curtis Seizert <cseizert@gmail.com>

*/

#include "CUDASieve/host.hpp"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/global.cuh"
#include "CUDASieve/launch.cuh"
#include "CUDASieve/primelist.cuh"

#include <stdint.h>
#include <iostream>
#include <cmath>

uint32_t * PrimeList::getSievingPrimes(uint32_t maxPrime, uint32_t & primeListLength, bool silent)
{
 PrimeList primelist(maxPrime);

 primelist.allocate();
 primelist.iterSieve();
 primeListLength = (uint32_t)* primelist.kerneldata.h_count;
 if(!silent) std::cout << "List of sieving primes in " << primelist.timer.get_ms() << " ms." << std::endl;
 uint32_t * temp = primelist.d_primeList;
 primelist.d_primeList = NULL;

 return temp;
}

PrimeList::PrimeList(uint32_t maxPrime)
{
 this -> maxPrime = maxPrime;
 if(maxPrime < pow(2,22)) bigSieveKB = 256;
 if(maxPrime > pow(2,30)) bigSieveKB = 16384;
 if(maxPrime > pow(2,31)) bigSieveKB = 32768;

 blocks = (bigSieveKB << 13)/(32*PL_SIEVE_WORDS);
 threads = 512;

 hist_size_lg = blocks/512 + 1;
 piHighGuess = (int) (maxPrime/log(maxPrime))*(1+1.2762/log(maxPrime)); // this is an empirically derived formula to calculate a high bound for the prime counting function pi(x)

 PL_Max = std::min((uint32_t)65536, maxPrime);
}

void PrimeList::allocate()
{
 kerneldata.allocate();

 d_bigSieve =        safeCudaMalloc(d_bigSieve, bigSieveKB*256*sizeof(uint32_t));
 d_primeList =       safeCudaMalloc(d_primeList, piHighGuess*sizeof(uint32_t));
 d_histogram =       safeCudaMalloc(d_histogram, blocks*sizeof(uint32_t));
 d_histogram_lg =    safeCudaMalloc(d_histogram_lg, hist_size_lg*sizeof(uint32_t));

 cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
 cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

 cudaMemset(d_primeList, 0, piHighGuess*sizeof(uint32_t));
}

void PrimeList::iterSieve() // makes the list of primes on the device and then copies them back to the host
{
 timer.start();

 cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));

 device::firstPrimeList<<<1, 256>>>(d_primeList, kerneldata.d_count, 32768, PL_Max);
 cudaDeviceSynchronize();
 primeListLength = (uint32_t)* kerneldata.h_count;
 if(maxPrime > PL_Max){

   for(uint64_t bottom = 65536; bottom < maxPrime; bottom += (bigSieveKB << 14)){

     cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
     cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

     device::bigSieveSm<<<bigSieveKB/sieveKB, THREADS_PER_BLOCK, (sieveKB << 10)>>>
       (d_primeList, d_bigSieve, bottom, sieveKB, primeListLength);

     uint32_t * d_ptr = d_primeList + * kerneldata.h_count;

     cudaDeviceSynchronize();

     device::makeHistogram_PLout<<<bigSieveKB, THREADS_PER_BLOCK>>>
       (d_bigSieve, d_histogram, (uint64_t)bottom, (uint64_t) maxPrime);
     device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
       (d_histogram, d_histogram_lg, blocks);
     device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
       (d_histogram_lg, kerneldata.d_count, hist_size_lg);
     device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
       (d_histogram, d_histogram_lg, blocks);
     device::makePrimeList_PLout<<<bigSieveKB, THREADS_PER_BLOCK>>>
       (d_ptr, d_histogram, d_bigSieve, bottom, maxPrime);
     cudaDeviceSynchronize();
   }
 }
 timer.stop();
}

PrimeList::~PrimeList()
{
 kerneldata.deallocate();
 cudaFree(d_bigSieve);
 cudaFree(d_histogram);
 cudaFree(d_histogram_lg);
}
