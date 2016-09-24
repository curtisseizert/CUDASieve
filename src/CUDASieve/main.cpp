/*

CUDASieveMain.cu

Source for main for CUDASieve
by Curtis Seizert - cseizert@gmail.com

*/

#include <iostream>
#include <stdio.h>
#include <vector>

#include "CUDASieve/host.hpp"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/launch.cuh"

int main(int argc, char* argv[])
{
  // start the timer
  CudaSieve * sieve = new CudaSieve;
  KernelData::allocate();

  // parse the command line options passed to the executable and then set appropriate flags
  host::parseOptions(argc, argv, sieve);

  // start profiler
  // cudaProfilerStart();

  // this is for the -h and --help switches
  if(sieve->isFlag(31)) return 0;

  sieve->countPrimes();

  //uint32_t num = sieve->countPrimes(288230376151711744, 288230377225453568);
  //std::vector <uint64_t> primes;

  // if(sieve->isFlag(0)){
  // for(uint32_t i = 0; i < 16777216; i++){
  //   for(uint8_t j = 0; j < 32; j++){
  //   bool r = (sieve->sieveOut[i] >> j) & 1u;
  //   if(r) primes.push_back(64*i + 2* j +1 + sieve->getBottom());
  // }}
  // stop profiler
  // cudaProfilerStop();

  //for(uint32_t i = 0; i < num; i++) printf("%llu\t", primes[i]);
  // }
  //if(!sieve->isFlag(30))
  printf("\t%f seconds elapsed.\n", sieve->elapsedTime());
  cudaDeviceReset();

  return 0;
}
