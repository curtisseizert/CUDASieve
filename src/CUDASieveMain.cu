/*

CUDASieveMain.cu

Source for main for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

//#include "CUDASieveMain.h"

#include "CUDASieveHost.cpp"
#include <iostream>
#include <stdio.h>

int main(int argc, char* argv[])
{
  // start the timer
  CudaSieve * sieve = new CudaSieve;

  // parse the command line options passed to the executable and then set appropriate flags
  // host::parseOptions(argc, argv, sieve);
  //
  // // this is for the -h and --help switches
  // if(sieve->isFlag(31)) return 0;
  //
  // sieve->countPrimes();

  for(uint32_t i = 10; i <= 10000; i++) std::cout << (10000000*i) << " " << sieve->countPrimes(10000000*i) << std::endl;

  // if(!sieve->isFlag(30)) printf("\t%f seconds elapsed.\n", sieve->elapsedTime());
  // cudaDeviceReset();

  return 0;
}
