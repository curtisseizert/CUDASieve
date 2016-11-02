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
#include "CUDASieve/cstest.hpp"

using namespace boost::multiprecision;

int main()
{
  boost::random::ranlux48_base rng1;
  boost::random::lagged_fibonacci44497 rng2;
  boost::random::mt19937 rng3;
  boost::random::uniform_int_distribution<> dist(0,225726412); // 2^27.5 (to account for top of range below 2^64)
  boost::random::uniform_int_distribution<> dist_exp(1,36);
  boost::random::uniform_int_distribution<> dist_exp_range(2,25);
  boost::random::uniform_int_distribution<> dist_bool(0,1);
  boost::random::uniform_int_distribution<> dist_range_count(12,2000000000);
  boost::random::uniform_int_distribution<> dist_range_small(1024,65536);
  boost::random::uniform_int_distribution<> dist_range_test(16,128);

  size_t len;
  uint16_t testNum, gpuNum = 0;
  uint32_t numTrials = 1;
  uint32_t tests_with_error = 0;
  uint64_t bottom;
  uint64_t range;
  do{
  std::cout << "\nTests available:\n\t(1) Count - requires maximum 1.77 GB device memory" << std::endl;
  std::cout << "\t(2) Output - requires maximum  3.33 GB device memory" << std::endl;
  std::cout << "\t(3) Specific Range - Miller-Rabin test of all output primes in range" << std::endl;
  std::cout << "\t(4) Set CUDA-enabled device number (default 0)" << std::endl;
  std::cout << "::Selection? [1/2/3/4] ";
  std::cin >> testNum;
  if(testNum == 4){
    CudaSieve::listDevices();
    std::cout << "::Select Device ";
    std::cin >> gpuNum;
  }
}while(testNum == 4);

  if(testNum == 1 || testNum == 2){
    std::cout << "::Number of trials? [1 - 9999999] ";
    std::cin >> numTrials;
    std::cout << std::endl;
  }

  if(testNum == 3){
    std::cout << "::Bottom of range? ";
    std::cin >> bottom;
    std::cout << "::Range to check? ";
    std::cin >> range;
    std::cout << "::Number of trials to check for inconsistent results? [1 - 9999999] ";
    std::cin >> numTrials;
    std::cout << std::endl;
  }

  if(testNum == 1){

    std::cout << "\tErrors\tTrial\tlog2(bottom)\tbottom\t\t\trange" << std::endl;
    std::cout << "\t==========================================================================" << std::endl;

    for(uint32_t i = 0; i < numTrials; i++){

      if(dist_bool(rng3)) bottom = 1 * (uint64_t )dist(rng2) * (pow(2,(int)dist_exp(rng3)) - 1);
      else                bottom = 1 * (uint64_t )dist(rng1) * (pow(2,(int)dist_exp(rng2)) - 1);

      range = ((unsigned long)dist_range_test(rng3) * pow(2,(int)dist_exp_range(rng1)) - 1);
      uint64_t top = bottom + range;

      std::cout << "                                                                                                  \r";
      std::cout << "\t" << tests_with_error << "\t" << i+1 << "\t" << log2(bottom) << "\t\t" << bottom << "\r";
      std::cout << "\t\t\t\t\t\t\t\t" << range << "          \r";
      std::cout << std::flush;

      uint64_t primes = CudaSieve::countPrimes(bottom, top, gpuNum);
      uint64_t numPsPrimes = primesieve_parallel_count_primes(bottom, top);

      if(primes != numPsPrimes){
        std::cout << "\n\t\tLength mismatch: primesieve: " << numPsPrimes << "\t cudasieve: " << primes << std::endl;
        tests_with_error++;
      }
      if((i + 1) % 1024 == 0) cudaDeviceReset();
    }
  }

  if(testNum == 2){

    std::cout << "\tErrors\tTrial\tlog2(bottom)\tbottom\t\t\trange\t\tcount" << std::endl;
    std::cout << "\t=================================================================================" << std::endl;

    for(uint32_t i = 0; i < numTrials; i++){

      if(!dist_bool(rng1)) bottom = (uint64_t )dist(rng1) * (pow(2,(int)dist_exp(rng2)) - 1);
      else                bottom = (uint64_t )dist(rng2) * (pow(2,(int)dist_exp(rng3)) - 1);
      range = ((unsigned long)dist_range_test(rng1) * pow(2,(int)dist_exp_range(rng3)) - 1);
      uint64_t top = bottom + range;

      std::cout << "                                                                                                   \r";
      std::cout << "\t" << tests_with_error << "\t" << i+1 << "\t" << log2(bottom) << "\t\t" << bottom << "\r";
      std::cout << "\t\t\t\t\t\t\t\t" << range << "                            \r";
      std::cout << std::flush;

      uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len, gpuNum);

      std::cout << "\t\t\t\t\t\t\t\t\t\t" << len << "          \r";
      std::cout << std::flush;

      uint64_t numPsPrimes = primesieve_parallel_count_primes(bottom, top);



      if((uint64_t) len != numPsPrimes){
        std::cout << "\n\t\tLength mismatch: primesieve: " << numPsPrimes << "\t cudasieve: " << len << std::endl;
        tests_with_error++;
      }
      uint32_t fromEnds = std::min(65536u, (unsigned) len);
      if(len != 0){
        mr_check(primes, fromEnds, len, 0);
        mr_check(primes, fromEnds, len, 0);
      }

      cudaFreeHost(primes);
      if((i + 1) % 1024 == 0) cudaDeviceReset();
    }
  }


  if(testNum == 3){

    uint64_t top = bottom + range;

    std::cout << "\tlog2(bottom) = " << log2(bottom) << "     bottom =  " << bottom  << "          " << std::endl;
    uint64_t numPsPrimes = primesieve_parallel_count_primes(bottom, top);

    for(uint32_t i = 0; i < numTrials; i++){
      uint64_t primes = CudaSieve::countPrimes(bottom, top, gpuNum);
      std::cout << " Trial " << i+1 << "\r";
      std::cout << std::flush;
      if((uint64_t) primes != numPsPrimes){
        std::cout << "\n\t\tLength mismatch: primesieve: " << numPsPrimes << "\t cudasieve: " << primes << std::endl;
        tests_with_error++;
      }
      if((i + 1) % 1024 == 0) cudaDeviceReset();
    }

    std::cout << std::endl;
    uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len, gpuNum);

    mr_check(primes, 0, len);

    cudaFreeHost(primes);
  }

  if(testNum == 5){

    numGuess * guess1 = new numGuess(1);
    numGuess * guess2 = new numGuess(2);
    numGuess * guess3 = new numGuess(3);
    numGuess * guess4 = new numGuess(4);
    numGuess * guess5 = new numGuess(5);

    numTrials = 16384;
    std::cout << std::endl;
    std::cout << "\tTrial\tlog2(bottom)\tbottom\t\t\trange" << std::endl;
    std::cout << "\t===========================================================" << std::endl;

    for(uint32_t i = 0; i < numTrials; i++){

      if(dist_bool(rng1)) bottom = (uint64_t )dist(rng1) * (pow(2,(int)dist_exp(rng3)) - 1);
      else                bottom = (uint64_t )dist(rng3) * (pow(2,(int)dist_exp(rng2)) - 1);
      range = ((unsigned long)dist_range_test(rng2) * pow(2,(int)dist_exp_range(rng2)) - 1);
      // range = ((unsigned long)dist_range_small(rng2));
      uint64_t top = bottom + range;

      std::cout << "                                                                                                                       \r";
      std::cout << "\t" << i+1 << "\t" << log2(bottom) << "\t\t" << bottom << "\r";
      std::cout << "\t\t\t\t\t\t\t" << range << "                            \r";
      std::cout << std::flush;

      uint32_t g1 = ((1 + 2/log(top-bottom)) * (top - bottom)/log(bottom)) + 32;
      uint32_t g2 = (top/log(top))*(1+1.12/log(top)) - (bottom/log(bottom))*(1+1.12/log(bottom)) + 96*log(top-bottom);
      uint32_t g3 = (top/log(top))*(1+1.12/log(top)) - (bottom/log(bottom))*(1+1.12/log(bottom)) + 128*log(top-bottom);
      uint32_t g4 = (top/log(top))*(1+1.12/log(top)) - (bottom/log(bottom))*(1+1.12/log(bottom)) + 160*log(top-bottom);
      uint32_t g5 = (top/log(top))*(1+1.12/log(top)) - (bottom/log(bottom))*(1+1.12/log(bottom)) + 192*log(top-bottom);
      uint32_t g6 = (top/log(top))*(1+1.12/log(top)) - (bottom/log(bottom))*(1+1.12/log(bottom)) + 256*log(top-bottom);

      guess1->guess = g1;
      guess2->guess = g1;
      guess3->guess = g1;
      guess4->guess = g1;
      guess5->guess = g1;

      if(range > 32768){
        guess1->guess = g2;
        guess2->guess = g3;
        guess3->guess = g4;
        guess4->guess = g5;
        guess5->guess = g6;
      }

      uint64_t primes = CudaSieve::countPrimes(bottom, top, gpuNum);

      guess1->updateStats(primes, bottom, top);
      guess2->updateStats(primes, bottom, top);
      guess3->updateStats(primes, bottom, top);
      guess4->updateStats(primes, bottom, top);
      guess5->updateStats(primes, bottom, top);

      if((i + 1) % 1024 == 0) cudaDeviceReset();
    }

    std::cout << "\n" <<  std::endl;
    guess1->displayStats(numTrials);
    guess2->displayStats(numTrials);
    guess3->displayStats(numTrials);
    guess4->displayStats(numTrials);
    guess5->displayStats(numTrials);
  }


  std::cout << "\n" << std::endl;
  if(testNum == 1 || testNum == 2 || testNum ==3)
    std::cout << "Total " << tests_with_error << " errors over " << numTrials << " trials.\n" << std::endl;
  return 0;
}

void listDevices()
{
  int count;
  cudaGetDeviceCount(&count);
  std::cout << "\n" << count << " CUDA enabled devices available:" << std::endl;

  for(int i = 0; i < count; i++){
    std::cout << "\t(" << i << ") : ";
    cudaSetDevice(i);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << prop.name << std::endl;
  }
}

inline void mr_check(uint64_t * primes, int64_t numToCheck, size_t len, bool skipline) // iterative miller rabin check with some safeguards
{
  if(primes[0] == 0) std::cout << "\nInvalid array: contains zeros" << std::endl;
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
