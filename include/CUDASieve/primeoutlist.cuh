/*
primeoutlist.cuh

Host function for the primeOutList class, which creates host and device arrays
of primes for output (primeList is internal)

(c) 2016
Curtis Seizert <cseizert@gmail.com>

 */
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <stdint.h>

 #include "host.hpp"
 #include "CUDASieve/cudasieve.hpp"
 #include "CUDASieve/global.cuh"


 class SmallSieve;
 class BigSieve;

 class PrimeOutList{ // needs someone else's containers to put primes in.  Handles allocation.
   friend class BigSieve;
   friend class SmallSieve;
   friend class KernelData;

 private:
   uint32_t * d_histogram = NULL, *d_histogram_lg = NULL;
   uint32_t hist_size_lg, blocks;
   uint16_t threads;

   void allocateDevice();
   void fetch(BigSieve & bigsieve, CudaSieve & sieve)
   {
     uint64_t * d_ptr = sieve.d_primeOut + * sieve.kerneldata.h_count;

     cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
     cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

     device::makeHistogram_PLout<<<bigsieve.bigSieveKB, THREADS_PER_BLOCK>>>
       (bigsieve.d_bigSieve, d_histogram, bigsieve.bottom, bigsieve.top);
     device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
       (d_histogram, d_histogram_lg, blocks);
     device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
       (d_histogram_lg, sieve.kerneldata.d_count, hist_size_lg);
     device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
       (d_histogram, d_histogram_lg, blocks);
     device::makePrimeList_PLout<<<bigsieve.bigSieveKB, THREADS_PER_BLOCK>>>
       (d_ptr, d_histogram, bigsieve.d_bigSieve, bigsieve.bottom, bigsieve.top);
   }

   void fetch32(BigSieve & bigsieve, CudaSieve & sieve)
   {
     uint32_t * d_ptr = sieve.d_primeOut32 + * sieve.kerneldata.h_count;

     cudaMemset(d_histogram, 0, blocks*sizeof(uint32_t));
     cudaMemset(d_histogram_lg, 0, hist_size_lg*sizeof(uint32_t));

     device::makeHistogram_PLout<<<bigsieve.bigSieveKB, THREADS_PER_BLOCK>>>
       (bigsieve.d_bigSieve, d_histogram, bigsieve.bottom, bigsieve.top);
     device::exclusiveScan<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
       (d_histogram, d_histogram_lg, blocks);
     device::exclusiveScan<<<1,hist_size_lg,hist_size_lg*sizeof(uint32_t)>>>
       (d_histogram_lg, sieve.kerneldata.d_count, hist_size_lg);
     device::increment<<<hist_size_lg,threads,threads*sizeof(uint32_t)>>>
       (d_histogram, d_histogram_lg, blocks);
     device::makePrimeList_PLout<<<bigsieve.bigSieveKB, THREADS_PER_BLOCK>>>
       (d_ptr, d_histogram, bigsieve.d_bigSieve, bigsieve.bottom, (uint32_t)bigsieve.top);
   }

   void cleanupAll();
   void cleanupAllDevice();

 public:
   PrimeOutList(CudaSieve & sieve);
   ~PrimeOutList();
 };
