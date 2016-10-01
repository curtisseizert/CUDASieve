/*

CUDASieveHost.cuh

Host functions for CUDASieve
Curtis Seizert - cseizert@gmail.com

The naming convention for sieve sizes:
 sieveWords == number of 32-bit integers in the array
 sieveBits == number of total bits in the array (i.e. words * 32)
 sieveSpan == numbers covered by the sieve, since only odds are being sieved
              this means bits * 2

*/

#include <stdint.h>

#ifndef _CUDASIEVE_HOST
#define _CUDASIEVE_HOST

class CudaSieve;
class BigSieve;
class SmallSieve;

namespace host {

    void displayAttributes(const BigSieve & bigsieve);
    void displayAttributes(CudaSieve & sieve);

    void help();
    uint64_t echo(char * argv);
    void parseOptions(int argc, char* argv[], CudaSieve * sieve);
    void makePrimeList(uint32_t *& d_primeList);
}

#endif
