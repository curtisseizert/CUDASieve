#include <stdint.h>
#include <iostream>
#include <math.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "CUDASieve/cudasieve.hpp"

int main()
{
  uint64_t sum = 0, bottom = 0, intervalSize = pow(2,32);
  int64_t numRemaining = pow(10,9);
  size_t numInCurrentInterval;

  while(numRemaining > 0){
    uint64_t * d_primes = CudaSieve::getDevicePrimes(bottom, bottom + intervalSize, numInCurrentInterval);

    sum += thrust::reduce(thrust::device, d_primes, d_primes + std::min((int64_t)numInCurrentInterval, numRemaining));

    numRemaining -= numInCurrentInterval;
    bottom += intervalSize;
    cudaFree(d_primes);
  }

  std::cout << "Sum = " << sum << std::endl;

  return 0;
}
