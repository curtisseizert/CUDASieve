/*

CUDASieveHost.cpp

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

*/
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/launch.cuh"

#include "host.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <math.h>
#include <stdint.h>

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

void host::displayAttributes(CudaSieve & sieve)
{
  if(!sieve.flags[30]) std::cout << "\n" << sieve.primeListLength << " sieving primes in (37, " << sieve.maxPrime_ << "]" << std::endl;

  if(!sieve.flags[2] && !sieve.flags[30]){
    std::cout << "Small Sieve parameters" << std::endl;
    std::cout << "Full Blocks     :  " << sieve.totBlocks << std::endl;
    std::cout << "Threads         :  " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Sieve Size      :  " << sieve.sieveKB << " kb" << std::endl;
  }
  if(!sieve.flags[30]) std::cout << "Initialization took " << sieve.elapsedTime() << " seconds.\n" << std::endl;
}

void host::displayAttributes(const BigSieve & bigsieve)
{
  std::cout << "Big Sieve parameters" <<std::endl;
  std::cout << "Number of required iterations\t:  " << bigsieve.totIter << std::endl;
  std::cout << "Size of small sieve\t\t:  " << bigsieve.sieveKB << " kb" <<std::endl;
  std::cout << "Size of big sieve\t\t:  " << bigsieve.bigSieveKB << " kb" <<std::endl;
  std::cout << "Bucket arrays filled in\t\t:  " << bigsieve.time_ms << " ms\n" << std::endl;
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

    if(i + 1 <= argc){
      if(arg == "-t")           sieve->setTop(echo(argv[i+1]));
      if(arg == "-b")           sieve->setBottom(echo(argv[i+1]));
      if(arg == "-bs"){         sieve->setBigSieveKB(echo(argv[i+1]));
                                sieve->setFlagOn(18);}
      if(arg == "-g")           sieve->setGpuNum(atoi(argv[i+1]));
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
  printf("\n\tExamples\n");
  printf("CUDASieve -b 2**50-2**30 -t 2**50+2**30 -s\n");
  printf("CUDASieve -t 4685215875\n");
  printf("CUDASieve -b 2**64-2**35-2**30 -t 2**64-2**35 -p\n");
  printf("\t\nGood Luck!\n\n");
}
