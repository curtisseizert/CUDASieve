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
  host::parseOptions(argc, argv, sieve);

  // this is for the -h and --help switches
  if(sieve->isFlag(31)) return 0;

  sieve->countPrimes();

  if(!sieve->isFlag(30)) printf("\t%f seconds elapsed.\n", sieve->elapsedTime());
  cudaDeviceReset();

  return 0;
}
