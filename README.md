# CUDASieve
A GPU accelerated C++/CUDA C implementation of the segmented sieve of Eratosthenes

Most of the testing has been done on a GTX 1080 gpu with CUDA 8.0 RC on the most recent version of <a href="https://www.archlinux.org"> Arch Linux</a> x86_64.  This work contains some optimizations found in Ben Buhrow's <a href="https://sites.google.com/site/bbuhrow/home/cuda-sieve-of-eratosthenes">CUDA Sieve of Eratosthenes</a> as well as an attempt at implementing Tomás Oliveira e Silva's <a href="http://sweet.ua.pt/tos/software/prime_sieve.html">Bucket
algorithm</a> on the GPU.
While this code is in no way as elegant as that of Kim Walisch's<a href="http://primesieve.org">primesieve</a>, the use of GPU acceleration allows a
significant speedup.  On the author's hardware, device initialization takes a constant 0.10 seconds regardless of the
workload, but generation of small ranges (i.e. < 10<sup>10</sup>) is very fast thereafter.  Here are some benchmarks for counts:

```sh
Range                 Time to generate            Time to sieve     Total running time  Count
                      list of sieving primes      this range        

0 to 10^7              0.069 ms                    0.37 ms           0.100 s             664 579
0 to 10^8              0.070 ms                    1.01 ms           0.106 s             5 761 455
0 to 10^9              0.068 ms                    6.51 ms           0.110 s             50 847 534
0 to 10^10             0.125 ms                    68.7 ms           0.168 s             455 052 511
0 to 10^12             0.132 ms                    13.0 s            13.1 s              37 607 912 018
0 to 2^50              0.745 ms                    *                 33 897 s            33 483 379 603 407
2^40 to 2^40 + 2^30    0.132 ms                    34.8 ms           0.137 s             38 726 266
2^50 to 2^50 + 2^30    0.739 ms                    36.2 ms           0.143 s             30 984 665
2^58 to 2^58 + 2^30    11.1 ms                     181 ms            0.303 s             26 707 352
```


The output for each of these ranges has been verified against that of primesieve both in count and in the actual primes
generated.  Additionally, this code contains a way of generating a list of (32 bit) primes, in order, on the device that is
much faster than the bottleneck of ordering them on the host.  Generating the list of 189 961 801 primes from 32 to 4e9
takes just 99 ms.  This is about 7.5 GB of primes/second.

At this point, the code is barely more than a proof of principle, so I imagine that anyone who is interested in this can
write their own makefile.  Also, the include file names for cuda are based on the cuda-8.0rc, so those may have to be
changed in order to build this with cuda 7.5.  Hopefully Nvidia will hurry up and actually release cuda 8.0.

Here are the known issues:
  (1) The bottom of the sieving range must be a multiple of 2<sup>17</sup>.  This will be fixed in the near future<br>
  (2) There are instances where the count is off by 1-4 on certain ranges where less than an entire sieving range is counted<br>
  (3) Only multiples of 2<sup>24</sup> are acceptable inputs for the top or bottom if above 2<sup>40</sup>.<br>
  (4) Somewhere above 2<sup>58</sup>, the sieve starts crossing off actual primes due to something other than a race condition
      I am looking into this.<br>

I am currently working on cleaning up the class structure of the host code as well as working out the kinks for sieving
the ranges above 2<sup>58</sup> and cleaning up the horrible mess that is my debugging code.
