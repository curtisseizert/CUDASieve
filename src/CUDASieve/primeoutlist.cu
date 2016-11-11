/*
primeoutlist.cu

Host function for the primeOutList class, which creates host and device arrays
of primes for output (primeList is internal)

(c) 2016
Curtis Seizert <cseizert@gmail.com>

 */
#include "CUDASieve/global.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/launch.cuh"
#include <iostream>
