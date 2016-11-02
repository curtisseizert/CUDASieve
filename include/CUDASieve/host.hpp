/*

host.hpp

Host functions for CUDASieve
Curtis Seizert  <cseizert@gmail.com>

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/

#include <stdint.h>
#include <cuda_runtime.h>

#ifndef _CUDASIEVE_HOST
#define _CUDASIEVE_HOST

class CudaSieve;
class BigSieve;
class SmallSieve;

namespace host {

    void displayAttributes(const BigSieve & bigsieve);
    void displayAttributes(CudaSieve & sieve);

}

class KernelData{
  friend class BigSieve;
  friend class SmallSieve;
  friend class PrimeOutList;
  friend class CudaSieve;
  friend class PrimeList;
private:
  volatile uint64_t * h_count = NULL, * h_blocksComplete = NULL;
  volatile uint64_t * d_count = NULL, * d_blocksComplete = NULL;
public:
  uint64_t getCount(){return * h_count;}
  uint64_t getBlocks(){return * h_blocksComplete;}

  void displayProgress(uint64_t totBlocks);
  void displayProgress(uint64_t value, uint64_t totIter);

  void allocate();
  void deallocate();

  KernelData(){};
  ~KernelData(){};
};

class KernelTime{
  friend class BigSieve;
  friend class SmallSieve;
private:
  cudaEvent_t start_ = NULL, stop_ = NULL;
public:
  KernelTime();
  ~KernelTime();

  void displayTime();
  void start();
  void stop();
  float get_ms();
};

#endif
