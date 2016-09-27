/*

CUDASieveHost.cpp

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

*/
#include "CUDASieve/cudasieve.hpp"

#include "host.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>

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

void host::parseOptions(int argc, char* argv[], CudaSieve * sieve)
{
  for(int i = 1; i < argc; i++){ // parsing cmd line arguments; flags 1-6 are set internally
    std::string arg = argv[i];
    if(arg == "-p"){
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
      if(arg == "-t") sieve->setTop(echo(argv[i+1]));
      if(arg == "-b") sieve->setBottom(echo(argv[i+1]));
      if(arg == "-bs") sieve->setBigSieveKB(echo(argv[i+1]));
      if(arg == "-g") sieve->setGpuNum(atoi(argv[i+1]));
      if(arg == "-sievekb") sieve->setSieveKB(atoi(argv[i+1]));
    }
  }
}

void host::help()
{
  printf("\t\n\tCUDASieve 0.9 help:\n\n");
  printf("The default behavior of CUDASieve is to count primes in the range 0 - 2**30.  If the answer\n");
  printf("is other than 54 400 028, that is bad.  In any event, there are command line switches to modify\n");
  printf("this behavior, and all arguments must be preceded by the appropriate switch, or they will be\n");
  printf("ignored.  All numerical parameters accept the same mathmatical expressions as $echo does.\n");
  printf("Available switches are:\n\n");
  printf("\t-b\t\tBottom of the sieving range\n");
  printf("\t-t\t\tTop of the sieving range\n");
  printf("\t-s --silent\tMinimal command line output\n");
  printf("\t-h --help\tDisplay this message.\n\n");
  printf("\t-sievekb\tSet size in kb of the small Smem sieve (default 16).\n");
  printf("\t-bs\t\tSet the block size in kbof the large number sieve (Default 2**10).\n");
  printf("\t-g\t\tSet the (cuda) GPU number (default 0).\n");
  printf("\tGood Luck!\n\n");
}
