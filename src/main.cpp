/*

CUDASieveMain.cu

Source for main for CUDASieve
by Curtis Seizert - cseizert@gmail.com

*/

#include <iostream>
#include <stdio.h>
#include <vector>

#include "host.hpp"
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

  printf("\t%f seconds elapsed.\n", sieve->elapsedTime());
  cudaDeviceReset();

  return 0;
}
