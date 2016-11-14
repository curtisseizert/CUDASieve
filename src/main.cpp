/*

main.cpp

Source for main for CUDASieve
Curtis Seizert <cseizert@gmail.com>

*/

#include <iostream>
#include <stdio.h>
#include <vector>
#include <ctime>

#include "CUDASieve/host.hpp"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/launch.cuh"

namespace host{
  void parseOptions(int argc, char* argv[], CudaSieve * sieve);
  void help();
  uint64_t echo(char * argv);
}

int main(int argc, char* argv[])
{
  // start the timer
  clock_t start_time = clock();
  float elapsed_time;

  CudaSieve * sieve = new CudaSieve;

  // parse the command line options passed to the executable and then set appropriate flags
  host::parseOptions(argc, argv, sieve);

  // this is for the -h and --help switches
  if(sieve->isFlag(31)) return 0;

  sieve->CLIPrimes();

  elapsed_time = (clock() - start_time)/((double) CLOCKS_PER_SEC);
  if(!sieve->isFlag(30)) std::cout << "total time : " << elapsed_time << " seconds" << std::endl;

  cudaDeviceReset();
  return 0;
}

void host::parseOptions(int argc, char* argv[], CudaSieve * sieve)
{
  for(int i = 1; i < argc; i++){ // parsing cmd line arguments; flags 1-6 are set internally
    std::string arg = argv[i];
    if(arg == "-p" || arg == "--print"){
      sieve->setFlagOn(0);
      sieve->setFlagOn(7);
    }
    if(arg == "-h" || arg == "--help"){ // help
      help();
      sieve->setFlagOn(31);
    }
    if(arg == "-s" || arg == "--silent"){ // silent: displays only the smallest amount of information necessary
      sieve->setFlagOn(30);
    }
    if(arg == "-pg"){ // I don't remember what this does.  Probably should have commented this line before...
      sieve->setFlagOn(0);
      sieve->setFlagOn(28);
    }
    if(arg == "--profile") // enable profiling by nvprof
      sieve->setFlagOn(17);

    if(arg == "-l" || arg == "--list"){ // list devices then exit
      CudaSieve::listDevices();
      sieve->setFlagOn(31);
    }

    if(i + 1 <= argc){
      if(arg == "-t")           sieve->top = echo(argv[i+1]);
      if(arg == "-b")           sieve->bottom = echo(argv[i+1]);
      if(arg == "-bs"){         sieve->setBigSieveKB(echo(argv[i+1]));
                                sieve->setFlagOn(18);}
      if(arg == "-g")           sieve->setGpu(atoi(argv[i+1]));
      if(arg == "-sievekb")     sieve->setSieveKB(atoi(argv[i+1]));
      if(arg == "-partial")     sieve->setMaxPrime(atoi(argv[i+1]));

    }
  }
}

void host::help()
{
  printf("\t\n\tCUDASieve help:\n\n");
  printf("The default behavior of CUDASieve is to count primes in the range 0 - 2**30.  If that is\n");
  printf("not good enough, there are a few switches that provide additional options.  All inputs should\n");
  printf("be/be preceded by one of these switches.\n");
  printf("All numerical parameters accept the same mathmatical expressions as $echo does.\n");
  printf("Available switches are:\n\n");
  printf("\t-p --print\tprint outputted primes (will be ignored below 2**40)\n");
  printf("\t-b\t\tBottom of the sieving range\n");
  printf("\t-t\t\tTop of the sieving range\n");
  printf("\t-s --silent\tMinimal command line output\n");
  printf("\t-h --help\tDisplay this message.\n");
  //printf("\t-sievekb\tSet size in kb of the small Smem sieve (default 16).\n"); // this causes inaccuracies and bugs
  printf("\t-bs\t\tSet the block size in kb of the large number sieve.\n");
  printf("\t-g\t\tSet the (cuda) GPU number (default 0).\n");
  printf("\t-l --list\tList the available CUDA enabled devices.\n");
  printf("\n\t===Examples===\n");
  printf("cudasieve -b 2**50-2**30 -t 2**50+2**30 -s\n");
  printf("cudasieve -t 4685215875\n");
  printf("cudasieve -b 2**64-2**35-2**30 -t 2**64-2**35 -p\n\n");
}

uint64_t host::echo(char * argv) // for getting values bigger than the 32 bits that system() will return;
{
  uint64_t value;
  size_t len = 0;
  char * line = NULL;
  FILE * in;
  char cmd[256];

  sprintf(cmd, "echo $((%s))", argv);

  in = popen(cmd, "r");
  getline(&line, &len, in);
  value = atol(line);

  return value;
}
