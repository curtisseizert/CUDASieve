/*
  cstest.cpp - a short program to automate extensive testing of cudasieve's output.
  It depends on primesieve, boost, and openMP and checks both count and correctness.
  Using a Miller-Rabin primality test (boost library) with 25 withnesses,
  the latter is checked only for the 65536 primes at both ends (for iterative tests),
  as these are the places where errors are most likely to occur, and this significantly
  speeds up testing.  If the executable is called with no arguments, a total of
  16k random intervals starting at a multiple of 64 and spanning 2^30 are checked.
  This takes about 20h on an i7 6700K with a GTX 1080 (but can be stopped by Ctrl + C
  at any time).  If the executable is called with a single argument, an interval of 2^30
  starting at this number is sieved and checked completely with a Miller-Rabin test.
  If you do find an error, please email me the bottom of the range so that I can look
  into it.

    Curtis Seizert <cseizert@gmail.com> 10/7/2016
*/

#include <iostream>
#include <stdint.h>
#include "CUDASieve/cudasieve.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <boost/multiprecision/miller_rabin.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <omp.h>
#include <primesieve.h>
#include <vector>

void dispFactors(uint64_t n, bool skipline = 0);
void mr_check(uint64_t * primes, int64_t numToCheck, size_t len, bool skipline = 0);

using namespace boost::multiprecision;

int main(int argc, char** argv)
{
  boost::random::ranlux48_base rng1;
  boost::random::lagged_fibonacci44497 rng2;
  boost::random::uniform_int_distribution<> dist(0,225726412); // 2^27.5 (to account for top of range below 2^64)
  boost::random::uniform_int_distribution<> dist_exp(1,30);
  boost::random::uniform_int_distribution<> dist_bool(0,1);

  size_t len;

  if(argc == 1){

    for(uint32_t i = 0; i < 16384; i++){

    uint64_t bottom;

      if(!dist_bool(rng2)) bottom = 64 * (uint64_t )dist(rng2) * (pow(2,(int)dist_exp(rng1)) - 1);
      else                bottom = 64 * (uint64_t )dist(rng1) * (pow(2,(int)dist_exp(rng2)) - 1);
      bottom -= bottom%64;
      uint64_t top = bottom + (1u << 30);

      std::cout << "\tTrial " << i << "  log2(bottom) = " << log2(bottom) << "     bottom =  " << bottom  << "          " << "\r";
      std::cout << std::flush;

      uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len);
      uint64_t numPsPrimes = primesieve_parallel_count_primes(bottom, top);

      if((uint64_t)len != numPsPrimes) std::cout << "\nLength mismatch: primesieve: " << numPsPrimes << "\t cudasieve: " << len << std::endl;

      mr_check(primes, 65536, len, 0);
      mr_check(primes, -65536, len, 0);

      cudaFreeHost(primes);
    }
  }

  if(argc == 2){
    uint64_t bottom = atol(argv[1]);
    uint64_t top = bottom + (1u << 30);

    std::cout << "\tlog2(bottom) = " << log2(bottom) << "     bottom =  " << bottom  << "          " << std::endl;
    uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len);
    uint64_t numPsPrimes = primesieve_parallel_count_primes(bottom, top);
    if((uint64_t)len != numPsPrimes) std::cout << "Length mismatch: primesieve: " << numPsPrimes << "\t cudasieve: " << len << std::endl;

    for(uint32_t i = 0; i < len; i++) std::cout << primes[i] << std::endl;;
    mr_check(primes, 0, len);

    cudaFreeHost(primes);
    }
  return 0;
}


inline void mr_check(uint64_t * primes, int64_t numToCheck, size_t len, bool skipline) // iterative miller rabin check with some safeguards
{
  if(primes[0] == 0) std::cout << "Invalid array: contains zeros" << std::endl;
  else{
    if(numToCheck == 0) numToCheck = len;
    if(numToCheck > 0){
      #pragma omp parallel for
      for(uint32_t i = 0; i < numToCheck; i++){
        if(!miller_rabin_test(primes[i], 25)) dispFactors(primes[i], skipline);}
    }else{
      #pragma omp parallel for
      for(uint32_t i = len + numToCheck - 1; i < len; i++){
        if(!miller_rabin_test(primes[i], 25)) dispFactors(primes[i], skipline);}
    }
  }
}


inline void dispFactors(uint64_t n, bool skipline) // simple trial division factorization when performance is not important
{
  uint8_t small[3] = {2,3,5};
  uint8_t wheel30[8] = {1, 7, 11, 13, 17, 19, 23, 29};
  uint32_t idx = 1, d = 7;
  uint64_t m = n, x = (uint64_t)std::sqrt((unsigned long long)n);
  std::vector<uint64_t> factors;

  for(uint8_t i = 0; i < 3;){
    if(n % small[i] == 0){
      factors.push_back(small[i]);
      n = n/small[i];
    }else{
      i++;
    }
  }

  while(n >= d && d <= x){
    if(n % d == 0){
      factors.push_back(d);
      n = n/d;
    }else{
      idx++;
      d = 30*(idx >> 3) + wheel30[idx&7];
    }
  }

  if(n != 1) factors.push_back(n);
  if(skipline) std::cout << std::endl;
  std::cout << m << " : ";
  for(uint16_t i = 0; i < factors.size(); i++) std::cout << factors[i] << " ";
  std::cout << std::endl;
}
