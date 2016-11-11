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
 // #include "CUDASieve/cudasieve.hpp"
 #include "CUDASieve/global.cuh"
