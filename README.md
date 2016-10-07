# CUDASieve
A GPU accelerated C++/CUDA C implementation of the segmented sieve of Eratosthenes


CUDASieve is a high performance segmented sieve of Eratosthenes for counting and generating prime numbers on NVidia GPUs.  This work contains some optimizations found in Ben Buhrow's <a href="https://sites.google.com/site/bbuhrow/home/cuda-sieve-of-eratosthenes">CUDA Sieve of Eratosthenes</a> as well as an attempt at implementing Tomás Oliveira e Silva's <a href="http://sweet.ua.pt/tos/software/prime_sieve.html">Bucket
algorithm</a> on the GPU.
While this code is in no way as elegant as that of Kim Walisch's <a href="http://primesieve.org">primesieve</a>, the use of GPU acceleration allows a
significant speedup.  On the author's hardware, device initialization takes a constant 0.10 seconds regardless of the
workload, but generation of small ranges (i.e. < 10<sup>10</sup>) is very fast thereafter.<br>

Benchmarks
----------
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>With GTX 1080:</b></p>
<table>
<tr><td><b>Range</td><td><b>Time to generate list<br> of sieving primes</td><td><b>Time to sieve<br> this range</td><td><b>Total running time</td><td><b>Count</td></tr>
<tr><td> 0 to 10<sup>7</sup> </td><td> 0.064 ms</td> <td> 0.10 ms</td><td> 0.088 s <td> 78 498</td></tr>
<tr><td> 0 to 10<sup>7</sup> </td><td> 0.055 ms</td> <td> 0.29 ms</td><td> 0.089 s <td> 664 579</td></tr>
<tr><td> 0 to 10<sup>8</sup></td><td>  0.068 ms </td><td> 0.95 ms </td><td> 0.108 s</td><td> 5 761 455</td></tr>  
<tr><td> 0 to 10<sup>9</sup></td><td> 0.064 ms  </td><td> 6.07 ms  </td><td> 0.092 s </td><td> 50 847 534</td></tr>  
<tr><td> 0 to 10<sup>10</sup></td><td> 0.125 ms</td><td> 63.5 ms</td><td> 0.158 s</td><td> 455 052 511</td></tr>  
<tr><td> 0 to 10<sup>12</sup></td><td> 0.127 ms</td><td> 12.4 s</td><td> 12.5 s</td><td> 37 607 912 018</td></tr>  
<tr><td> 0 to 2<sup>50</sup></td><td> 0.768 ms</td><td> *  </td><td> 28 653 s </td><td> 33 483 379 603 407</td></tr>  
<tr><td> 2<sup>40</sup> to 2<sup>40</sup> + 2<sup>30</sup></td><td> 0.128 ms</td><td> 31.7 ms</td><td> 0.120 s</td><td> 38 726 266</td></tr>  
<tr><td> 2<sup>50</sup> to 2<sup>50</sup> + 2<sup>30</sup></td><td> 0.771 ms</td><td> 33.5 ms</td><td> 0.127 s</td><td> 30 984 665</td></tr>  
<tr><td> 2<sup>60</sup> to 2<sup>60</sup> + 2<sup>30</sup></td><td> 19.4 ms</td><td> 126 ms</td><td> 0.243 s </td><td> 25 818 737</td></tr>
<tr><td> 2<sup>64</sup> - 2<sup>36</sup> to 2<sup>64</sup> - 2<sup>36</sup> + 2<sup>30</sup></td><td> 97 ms</td><td> 225 ms</td><td> 0.480 s </td><td> 24 201 154</td></tr></table>
<p>*Separate sieves for <2<sup>40</sup> and >=2<sup>40</sup></p>
<br>

<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Hardware Scaling for sieving 0 to 10<sup>9</sup>:</b></p>
<table>
<tr><td><b>GPU</td><td><b>Time to generate list<br> of sieving primes</td><td><b>Time to sieve<br> this range</td><td><b>Total running time</td></tr>
<tr><td>GTX 750</td><td>0.100 ms</td><td>67.5 ms</td><td>0.128 s</td></tr>
<tr><td>GTX 950</td><td>0.106 ms</td><td>36.4 ms</td><td>0.105 s</td></tr>
<tr><td>GTX 1070</td><td>0.075 ms</td><td>8.81 ms</td><td>0.140 s</td></tr>
<tr><td>GTX 1080</td><td>0.069 ms</td><td>6.03 ms</td><td>0.121 s</td></tr>
</table>

The output for each of these ranges has been verified against that of primesieve both in count and (for the ranges covering
less than a span of 2<sup>32</sup>) in the actual primes generated.  Additionally, this code contains a way of generating a
 list of sieving primes, in order, on the device that is much faster than the bottleneck of ordering them on the host.
  Generating the list of 189 961 800 primes from 38 to 4e9 takes just 89 ms.  This is about 8.3 GB of primes/second.  Primes
  are also prepared to be printed in the same way.  For example, the kernel time for preparing an array of all the (25 818 737) primes from 2<sup>60</sup> to 2<sup>60</sup>+2<sup>30</sup> and getting this array to the host is 157 ms with the GTX 1080.
  
  This implementation of Oliveira's bucket method requires a fixed 10 bytes of DRAM per prime, which equates to just over 2 GB
for sieving up to 2<sup>64</sup>.  The fact that
large primes are handled in global memory, rather than on-chip, means that increasing the number of blocks working on the
task of sieving these large primes does not increase the amount of memory used since the data set is not duplicated.<br><br>

Usability
---------

At this point, the code is barely more than a proof of principle, so I imagine that anyone who is interested in this can
modify the makefile to their needs (e.g. changing the CUDA_DIR variable and probably specifying fewer microarchitectures)  The include file names have not changed between CUDA 7.5 and 8.0, so this can be built without modifications to the source code (at least in linux) with CUDA 7.5 as well.  Windows support is currently being held up by my unwillingness to deal with the issue of Windows support.

Support for printing primes has just been added.

The provided binaries have been compiled for x86_64 linux with the compute capability 3.0 GPU virtual architecture and device code for each real architecture >= 3.0 (hence the size).  The executable 'CUDASieve' may need permissions changed to run.  If the CUDASieve/cudasieve.hpp header is #included, one can make use of several public member functions of the CudaSieve class for e.g. creating host or device arrays of primes by linking the cudasieve.a binary (with nvcc).  For example:

```C++
/* main.cpp */

#include <iostream>
#include <stdint.h>
#include <math.h>                   // pow()
#include <cuda_runtime.h>           // cudaFreeHost()
#include "CUDASieve/cudasieve.hpp"  // CudaSieve::getHostPrimes()

int main()
{
    uint64_t bottom = pow(2,63);
    uint64_t top = pow(2,63)+pow(2,30);
    size_t len;

    uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len);

    for(uint32_t i = 0; i < len; i++)
        std::cout << primes[i] << std::endl;

    cudaFreeHost(primes);            // must be freed with this call b/c page-locked memory is used.
    return 0;
}
```

placed in the CUDASieve directory compiles with the command 

```bash
nvcc -I include -lcudasieve -std=c++11 -arch=compute_30 main.cu
```

and prints out the primes in the range 2<sup>63</sup> to 2<sup>63</sup>+2<sup>30</sup>.

```C++
                /* CudaSieve static member functions */
/* Returns count from 0 to top */
uint64_t countPrimes(uint64_t top);

/* Returns count from bottom to top, caveats mentioned below apply */
uint64_t countPrimes(uint64_t bottom, uint64_t top);

/********* range must be a multiple of 2^24 *********/
/* Returns pointer to a page-locked host array of the primes in [bottom, top] of length size.
   Memory must be freed with cudaFreeHost() */
uint64_t * getHostPrimes(uint64_t bottom, uint64_t top, size_t & count);

/* Returns a std::vector of the primes in the interval [bottom, top] */
std::vector<uint64_t> getHostPrimesVector(uint64_t bottom, uin64_t top, size_t count);

/* Returns pointer to a device array of in [bottom, top] of length size */
uint64_t * getDevicePrimes(uint64_t bottom, uint64_t top, size_t & count);
```

For many iterations, it is preferable to avoid some of the overhead associated with memory allocation and creating the list of sieving primes repeatedly.  Fortunately, it is possible to do most of this work once by calling the CudaSieve constructor with two or three arguments (as shown below) and then calling the suitable non-static member function.

```C++
int main()
{
  size_t len;
  uint64_t bottom = pow(2,60);
  uint64_t top = bottom + pow(2,50);
  uint64_t range = pow(2,30);
  
  CudaSieve * sieve = new CudaSieve(bottom, top, range);

  for(; bottom <= top; bottom += range){
    uint64_t * primes = sieve->getHostPrimesSegment(bottom, len);
    /* something productive */
  }

  delete sieve;
  return 0;
}
```
The above code creates a CudaSieve object with an appropriate list of sieving primes for ranges up to ```top``` and with memory allocated for copying arrays of primes over range ```range``` as long as they are above ```bottom```.  If the range parameter is not included, memory will not be allocated for copying back primes, but a list of sieving primes will still be generated counting functions. The relevant functions are:
```C++
  CudaSieve(uint64_t bottom, uint64_t top, uint64_t range); 
  // third parameter is only necessary when copying primes

  uint64_t countPrimesSegment(uint64_t bottom, uint64_t top); 
  // counts primes in [bottom, top] as long as they are in
  // the range specified in the constructor call
                                                              
  uint64_t * getHostPrimesSegment(uint64_t bottom, size_t & count);   
  uint64_t * getDevicePrimesSegment(uint64_t bottom, size_t & count); 
  // returns a pointer to an array of primes in the
  // range [bottom, bottom+range] (the latter as specified
  // when calling the constructor).  Invalid bottom will
  // return NULL and count = 0;
```
<br>

Limitations
------------

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1) Above 2<sup>40</sup> or when printing primes, the range must be a multiple of 2<sup>24</sup>.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2) Above 2<sup>40</sup> or when printing primes, the bottom must be a multiple of 64.
<br>

State of the Project
-------------------
I am continuing verification tests on outputs of the sieve at various ranges as well as working out counting and printing ranges that include less than a full sieve segment.  Let me know if you have any requests for features, and I'll see what I can do.
