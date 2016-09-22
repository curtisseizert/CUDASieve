/*

CUDASieveHost.cpp

Host functions for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

#include "CUDASieveHost.hpp"

#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>


CudaSieve::CudaSieve()
{
  start_time = clock();
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

void CudaSieve::setTop(uint64_t top){this -> top = top;}
void CudaSieve::setBottom(uint64_t bottom){this -> bottom = bottom;}
void CudaSieve::setSieveKB(uint32_t sieveKB){this -> sieveKB = sieveKB;}
void CudaSieve::setBigSieveKB(uint32_t bigSieveKB){this -> bigSieveKB = bigSieveKB;}
void CudaSieve::setGpuNum(uint16_t gpuNum){this -> gpuNum = gpuNum;}

void CudaSieve::setDefaults()
{
  bottom = 0;
  top = (1u << 30);
  sieveKB = 16;
  bigSieveKB = (1u << 10);
  gpuNum = 0;
}

void CudaSieve::setKernelParam()
{
  sieveBits = sieveKB << 13;
  bigSieveBits = bigSieveKB << 13;
  uint64_t smTop = min((unsigned long long) top, 1ull << 40);
  kernelBottom = bottom - bottom % (2 * sieveBits);
  totBlocks = (smTop - kernelBottom) / (2 *  sieveBits);
  smKernelTop = kernelBottom + (totBlocks * sieveBits * 2);
  cudaSetDevice(gpuNum);
  this->setFlags();
}

void CudaSieve::setSieveOutBits(){sieveOutBits = (top - kernelBottom)/2;}

void CudaSieve::setFlags()
{
  if(top > (1ull << 40)) this -> flags[1] = 1;
  if(bottom >= (1ull << 40)) this -> flags[2] = 1;
  if(kernelBottom != bottom) this -> flags[3] = 1;
  if(min((unsigned long long) top, 1ull << 40) != smKernelTop) this -> flags[4] = 1;
}

void CudaSieve::displayRange(){if(!isFlag(30)) std::cout << "\n" << "Counting primes from " << bottom << " to " << top << std::endl;}

void CudaSieve::displaySieveAttributes()
{
  if(!this->isFlag(30)) std::cout << "\n" << primeListLength << " sieving primes in (37, " << (unsigned long) sqrt(top) << "]" << std::endl;

  if(!this->isFlag(2) && !this->isFlag(30)){
    std::cout << "Small Sieve parameters" << std::endl;
    std::cout << "Total Blocks    :  " << totBlocks << std::endl;
    std::cout << "Threads         :  " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Sieve Size      :  " << sieveKB << " kb" << std::endl;
  }
  if(!this->isFlag(30)) std::cout << "Initialization took " << elapsedTime() << " seconds.\n" << std::endl;
}

void CudaSieve::makePrimeList()
{
  uint32_t maxPrime = (unsigned long) sqrt(top);

  PrimeList primelist(maxPrime);

  primelist.allocate();
  primelist.sievePrimeList(this);
  primeListLength = primelist.getPrimeListLength();
  if(!flags[30]) primelist.displayTime();
  this -> d_primeList = primelist.getPtr();
  primelist.cleanUp();
}

void CudaSieve::makePrimeList(uint32_t maxPrime)
{
  PrimeList primelist(maxPrime);

  primelist.allocate();
  primelist.sievePrimeList(this);
  primeListLength = primelist.getPrimeListLength();
  if(!flags[30]) primelist.displayTime();
  this -> d_primeList = primelist.getPtr();
  primelist.cleanUp();
}

void CudaSieve::launchControl()
{
  KernelData kernelData;

  if(!flags[2]){
    if(!flags[0]) smallSieveCtl(kernelData);
    //else host::smallSieveCopy();
  }
  if(flags[1]) bigSieveCtl(kernelData);
  count = kernelData.getCount();
}

void CudaSieve::bigSieveCtl(KernelData & kernelData)
{
  BigSieve * bigsieve = new BigSieve;

  bigsieve ->setParameters(this);
  bigsieve ->allocate();
  bigsieve ->fillNextMult();
  if(!flags[0])bigsieve ->launchLoop(kernelData);
  else bigsieve ->launchLoop(kernelData, this);
  bigsieve ->displayCount(kernelData);
  bigsieve ->cleanUp();
}

void CudaSieve::smallSieveCtl(KernelData & kernelData)
{
  SmallSieve smallsieve(this);
  smallsieve.launch(kernelData, this);
  cudaDeviceSynchronize();
  if(!flags[30]) smallsieve.displaySieveTime(this);
}

void CudaSieve::countPrimes()
{
  this->setKernelParam();
  this->displayRange();
  this->makePrimeList();
  this->displaySieveAttributes();
  this->launchControl();
}

uint64_t CudaSieve::countPrimes(uint64_t top)
{
  this->top = top;
  flags[30] = 1;
  this->setKernelParam();
  this->makePrimeList();
  this->launchControl();
  return count;
}

// void CudaSieve::copySieve()
// {
//
// }


double CudaSieve::elapsedTime()
{
  return (clock() - start_time)/((double) CLOCKS_PER_SEC);
}

PrimeList::PrimeList(uint32_t maxPrime){

  this -> maxPrime = maxPrime;

  blocks = 1+maxPrime/(64 * PL_SIEVE_WORDS);
  threads = min(512, blocks);

  hist_size_lg = blocks/512 + 1;
  piHighGuess = (int) (maxPrime/log(maxPrime))*(1+1.2762/log(maxPrime)); // this is an empirically derived formula to calculate a high bound for the prime counting function pi(x)
  if(maxPrime > 65536) PL_Max = sqrt(maxPrime);
  else PL_Max = maxPrime;
}

void PrimeList::allocate(){

  h_primeListLength = (uint32_t *)malloc(sizeof(uint32_t));

  if(cudaMalloc(&d_primeList, piHighGuess*sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_primeList" << std::endl; exit(1);}
  if(cudaMalloc(&d_primeListLength, sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_primeListLength" << std::endl; exit(1);}
  if(cudaMalloc(&d_histogram, blocks*sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_histogram" << std::endl; exit(1);}
  if(cudaMalloc(&d_histogram_lg, hist_size_lg*sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_histogram_lg" << std::endl; exit(1);}

  cudaMemset(d_primeList, 0, piHighGuess*sizeof(uint32_t));
  cudaMemset(d_primeListLength, 0, sizeof(uint32_t));
  cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
  cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

PrimeList::~PrimeList(){}

void PrimeList::sievePrimeList(CudaSieve * sieve)
{
  cudaEventRecord(start);

  device::firstPrimeList<<<1, 256>>>(d_primeList, d_histogram, 32768, PL_Max);

  cudaMemcpy(h_primeListLength, &d_histogram[0], sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if(maxPrime > 65536){
    cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));

    device::makeHistogram<<<blocks, THREADS_PER_BLOCK>>>(d_primeList, d_histogram, 32*THREADS_PER_BLOCK, h_primeListLength[0]);
    device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>(d_histogram, d_histogram_lg, blocks);
    device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>(d_histogram_lg, d_primeListLength, hist_size_lg);
    device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>(d_histogram, d_histogram_lg, blocks);
    device::makePrimeList<<<blocks, THREADS_PER_BLOCK>>>(d_primeList, d_histogram, 32*THREADS_PER_BLOCK, h_primeListLength[0], maxPrime);

    cudaMemcpy(h_primeListLength, &d_histogram[blocks-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
}

void PrimeList::displayTime()
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if(milliseconds >= 1000) std::cout << "List of sieving primes generated in " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "List of sieving primes generated in " << milliseconds << " ms.    " << std::endl;
}

uint32_t PrimeList::getPrimeListLength()
{
  return h_primeListLength[0];
}

void PrimeList::cleanUp()
{
  cudaFree(d_histogram);
  cudaFree(d_histogram_lg);
  cudaFree(d_primeListLength);
}

SmallSieve::SmallSieve(CudaSieve * sieve)
{
  if(!sieve->isFlag(30)) printf("\tCalling small sieve kernel\n");
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

void SmallSieve::launch(KernelData & kernelData, CudaSieve * sieve)
{
  cudaEventRecord(start);
  device::smallSieve<<<sieve->totBlocks, THREADS_PER_BLOCK, (sieve->sieveKB << 10)>>>(sieve->d_primeList, kernelData.d_count, sieve->kernelBottom, sieve->sieveBits, sieve->primeListLength, kernelData.d_blocksComplete);
  //if(flags[3]) smallSieveIncomplete<<<1, THREADS_PER_BLOCK, (sieve->sieveKB << 10)>>>(sieve->d_primeList, d_count, sieve->kernelBottom, sieve->sieveBits, sieve->primeListLength, sieve->bottom);
  if(sieve->isFlag(4)) device::smallSieveIncompleteTop<<<1, THREADS_PER_BLOCK>>>(sieve->d_primeList, sieve->smKernelTop, sieve->sieveBits, sieve->primeListLength, sieve->top, kernelData.d_count, kernelData.d_blocksComplete);
  cudaEventRecord(stop);

  kernelData.displayProgress(sieve);

  cudaDeviceSynchronize();
  cudaEventSynchronize(stop);
}

void SmallSieve::displaySieveTime(CudaSieve * sieve)
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if(milliseconds >= 1000) std::cout << "Kernel time: " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "Kernel time: " << milliseconds << " ms.    " << std::endl;
}

BigSieve::BigSieve(CudaSieve * sieve)
{
  // Inherit relevant sieve paramters
  this -> bigSieveKB = sieve -> bigSieveKB;
  this -> bigSieveBits = sieve -> bigSieveBits;
  this -> sieveKB = sieve -> sieveKB;
  this -> primeListLength = sieve -> primeListLength;
  this -> d_primeList = sieve -> d_primeList;
  this -> top = sieve -> top;
  this -> silent = sieve -> isFlag(30);

  // Calculate BigSieve specific parameters
  blocksSm = bigSieveKB/sieveKB;
  blocksLg = primeListLength/THREADS_PER_BLOCK_LG;
  this -> bottom = min((1ull << 40), (unsigned long long) sieve -> bottom);
  totIter = (this->top-this->bottom)/(2*this->bigSieveBits);
}

void BigSieve::setParameters(CudaSieve * sieve)
{
  // Inherit relevant sieve paramters
  this -> bigSieveKB = sieve -> bigSieveKB;
  this -> bigSieveBits = sieve -> bigSieveBits;
  this -> sieveKB = sieve -> sieveKB;
  this -> primeListLength = sieve -> primeListLength;
  this -> d_primeList = sieve -> d_primeList;
  this -> top = sieve -> top;
  this -> silent = sieve -> isFlag(30);

  // Calculate BigSieve specific parameters
  blocksSm = bigSieveKB/sieveKB;
  blocksLg = primeListLength/THREADS_PER_BLOCK_LG;
  this -> bottom = max((1ull << 40), (unsigned long long) sieve -> bottom);
  totIter = (this->top-this->bottom)/(2*this->bigSieveBits);
}

void BigSieve::allocate()
{
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if(cudaMalloc(&d_nextMult, primeListLength*sizeof(uint64_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_nextMult" << std::endl; exit(1);}
  if(cudaMalloc(&d_bigSieve, bigSieveKB*256*sizeof(uint32_t))) {std::cerr << "PrimeList: CUDA memory allocation error: d_bigSieve" << std::endl; exit(1);}
  cudaMemset(d_bigSieve, 0, bigSieveKB*256*sizeof(uint32_t));
}

void BigSieve::fillNextMult()
{
  device::getNextMult30<<<blocksLg+1,THREADS_PER_BLOCK_LG>>>(d_primeList, d_nextMult, primeListLength, bottom);
  cudaDeviceSynchronize();
}

void BigSieve::launchLoop(KernelData & kernelData)
{
  cudaEventRecord(start);

  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>(d_primeList, d_bigSieve, bottom,
      primeListLength, sieveKB);
    device::bigSieveLg<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>(d_primeList, d_nextMult, d_bigSieve, bottom,
     bigSieveBits, primeListLength, sieveKB);

    cudaDeviceSynchronize();

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>(d_bigSieve, sieveKB, kernelData.d_count);
    kernelData.displayProgress(value, totIter);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  kernelData.displayProgress(totIter, totIter);
}

void BigSieve::launchLoop(KernelData & kernelData, CudaSieve * sieve)
{
  this -> ptr = sieve -> ptr;
  for(uint64_t value = 1; bottom + 2* bigSieveBits <= top; bottom += 2*bigSieveBits, value++){

    device::bigSieveSm<<<blocksSm, THREADS_PER_BLOCK, (sieveKB << 10), stream[0]>>>(d_primeList, d_bigSieve, bottom,
      primeListLength, sieveKB);
    device::bigSieveLg_test<<<blocksLg, THREADS_PER_BLOCK_LG, 0, stream[1]>>>(d_primeList, d_nextMult, d_bigSieve, bottom,
      bigSieveBits, primeListLength, sieveKB);

    cudaDeviceSynchronize();

    cudaMemcpy(ptr, d_bigSieve, bigSieveKB*1024, cudaMemcpyDeviceToHost);
    ptr = &sieve->sieveOut[value*bigSieveKB*256];

    device::bigSieveCount<<<blocksSm, THREADS_PER_BLOCK, (THREADS_PER_BLOCK*sizeof(uint32_t))>>>(d_bigSieve, sieveKB, kernelData.d_count);
  }
}

void BigSieve::displayCount(KernelData & kernelData)
{
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if(milliseconds >= 1000) std::cout << "Kernel time: " << milliseconds/1000 << " seconds." << std::endl;
  else std::cout << "Kernel time: " << milliseconds << " ms.    " << std::endl;
}

void BigSieve::cleanUp()
{
  cudaFree(d_nextMult);
  cudaFree(d_bigSieve);
}

KernelData::KernelData()
{
  cudaHostAlloc((void **)&h_count, sizeof(uint64_t), cudaHostAllocMapped);
  cudaHostAlloc((void **)&h_blocksComplete, sizeof(uint64_t), cudaHostAllocMapped);

  cudaHostGetDevicePointer((long **)&d_count, (long *)h_count, 0);
  cudaHostGetDevicePointer((long **)&d_blocksComplete, (long *)h_blocksComplete, 0);

  *h_count = 0;
  *h_blocksComplete = 0;
}

void KernelData::displayProgress(CudaSieve * sieve)
{
  if(!sieve->isFlag(30)){
    uint64_t value = 0;
    uint64_t counter = 0;
    do{
      uint64_t value1 = * h_blocksComplete;
      counter = * h_count;
      if (value1 > value){ // this is just to make it update less frequently
        std::cout << "\t" << (100*value/sieve->totBlocks) << "% complete\t\t" << counter << " primes counted.\r";
        std::cout.flush();
         value = value1;
       }
    }while (value < sieve->totBlocks+sieve->isFlag(4));
    counter = * h_count;
  std::cout << "\t" << "100% complete\t\t" << counter << " primes counted.\r";
  }
}

void KernelData::displayProgress(uint64_t value, uint64_t totIter)
{
  std::cout << "\t" << (100*value/totIter) << "% complete\t\t" << *h_count << " primes counted.\r";
  std::cout.flush();
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

    /*
    //Debugging flags start at 9

    if(arg == "-dc"){ // debug using primesieve generated prime number list to check against, print number of correct primes
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(10);
    }
    if(arg == "-dm"){ // debug using primesieve generated prime number list to check against, print number of missed primes
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(11);
    }
    if(arg == "-de"){ // debug using primesieve generated prime number list to check against, print number of extra primes
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(12);
    }
    if(arg == "-det"){ // debug using primesieve generated prime number list to check against, trial division factorize list of extra primes
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(12);
      sieve->setFlagOn(13);
    }
    if(arg == "--racecheck"){ // checks for race conditions by comparinng lists of extra primes from two separate runs (calls e.g. ./a.out -pass2 at the end of the first)
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(12);
      sieve->setFlagOn(14);
    }
    if(arg == "-pass2"){ // used automatically with -racecheck to run the program again
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(12);
      sieve->setFlagOn(15);
    }
    if(arg == "-ri"){ // iterative race check
      sieve->setFlagOn(0);
      sieve->setFlagOn(8);
      sieve->setFlagOn(12);
      sieve->setFlagOn(16);
      sieve->setFlagOn(30);
    }
    if(arg == "--profile") // enable profiling by nvprof
      sieve->setFlagOn(17);
    */
    if(i + 1 <= argc){
      if(arg == "-t") sieve->setTop(echo(argv[i+1]));
      if(arg == "-b") sieve->setBottom(echo(argv[i+1]));
      if(arg == "-bs") sieve->setBigSieveKB(echo(argv[i+1]));
      if(arg == "-g") sieve->setGpuNum(atoi(argv[i+1]));
      if(arg == "-sievekb") sieve->setSieveKB(atoi(argv[i+1]));
      // if(arg == "-pass2") passOne = atol(argv[i+1]);
      // if(arg == "-ri"){
      //   numIter = atoi(argv[i+1]);
      //   if(i + 2 < argc) passOne = atol(argv[i+2]);
      //   else passOne = 0;}
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
