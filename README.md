# CUDASieve
A GPU accelerated C++/CUDA C implementation of the segmented sieve of Eratosthenes

Most of the testing has been done on a GTX 1080 gpu with CUDA 8.0 RC on the most recent version of <a href="https://www.archlinux.org"> Arch Linux</a> x86_64.  This work contains some optimizations found in Ben Buhrow's <a href="https://sites.google.com/site/bbuhrow/home/cuda-sieve-of-eratosthenes">CUDA Sieve of Eratosthenes</a> as well as an attempt at implementing Tomás Oliveira e Silva's <a href="http://sweet.ua.pt/tos/software/prime_sieve.html">Bucket
algorithm</a> on the GPU.
While this code is in no way as elegant as that of Kim Walisch's<a href="http://primesieve.org">primesieve</a>, the use of GPU acceleration allows a
significant speedup.  On the author's hardware, device initialization takes a constant 0.10 seconds regardless of the
workload, but generation of small ranges (i.e. < 10<sup>10</sup>) is very fast thereafter.<br><br>

Benchmarks
----------

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<p><b>With GTX 1080:</b></p>
<table>
<tr><td><b>Range</td><td><b>Time to generate list<br> of sieving primes</td><td><b>Time to sieve<br> this range</td><td><b>Total running time</td><td><b>Count</td></tr>
<tr><td> 0 to 10<sup>7</sup> </td><td> 0.069 ms</td> <td> 0.37 ms</td><td> 0.100 s <td> 664 579</td></tr>
<tr><td> 0 to 10<sup>8</sup></td><td>  0.070 ms </td><td> 1.01 ms </td><td> 0.106 s</td><td> 5 761 455</td></tr>  
<tr><td> 0 to 10<sup>9</sup></td><td> 0.069 ms  </td><td> 6.03 ms  </td><td> 0.121 s </td><td> 50 847 534</td></tr>  
<tr><td> 0 to 10<sup>10</sup></td><td> 0.125 ms</td><td> 68.7 ms</td><td> 0.168 s</td><td> 455 052 511</td></tr>  
<tr><td> 0 to 10<sup>12</sup></td><td> 0.132 ms</td><td> 13.0 s</td><td> 13.1 s</td><td> 37 607 912 018</td></tr>  
<tr><td> 0 to 2<sup>50</sup></td><td> 0.745 ms</td><td> *  </td><td> 33 897 s </td><td> 33 483 379 603 407</td></tr>  
<tr><td> 2<sup>40</sup> to 2<sup>40</sup> + 2<sup>30</sup></td><td> 0.132 ms</td><td> 34.8 ms</td><td> 0.137 s</td><td> 38 726 266</td></tr>  
<tr><td> 2<sup>50</sup> to 2<sup>50</sup> + 2<sup>30</sup></td><td> 0.739 ms</td><td> 36.2 ms</td><td> 0.143 s</td><td> 30 984 665</td></tr>  
<tr><td> 2<sup>58</sup> to 2<sup>58</sup> + 2<sup>30</sup></td><td> 11.1 ms</td><td> 91.4 ms</td><td> 0.193 s </td><td> 26 707 352</td></tr></table>
<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Hardware Scaling for sieving 0 to 10<sup>9</sup>:<br><br></b>
<table>
<tr><td><b>GPU</td><td><b>Time to generate list<br> of sieving primes</td><td><b>Time to sieve<br> this range</td><td><b>Total running time</td></tr>
<tr><td>GTX 750</td><td>0.100 ms</td><td>67.5 ms</td><td>0.128 s</td></tr>
<tr><td>GTX 950</td><td>0.106 ms</td><td>36.4 ms</td><td>0.105 s</td></tr>
<tr><td>GTX 1070</td><td>0.075 ms</td><td>8.81 ms</td><td>0.140 s</td></tr>
<tr><td>GTX 1080</td><td>0.069 ms</td><td>6.03 ms</td><td>0.121 s</td></tr>
</table>

The output for each of these ranges has been verified against that of primesieve both in count and (for the ranges covering
less than a span of 2<sup>32</sup>) in the actual primes generated.  Additionally, this code contains a way of generating a
 list of (32 bit) primes, in order, on the device that is much faster than the bottleneck of ordering them on the host.
  Generating the list of 189 961 801 primes from 32 to 4e9 takes just 99 ms.  This is about 7.5 GB of primes/second.<br><br>
  
Usability
---------

At this point, the code is barely more than a proof of principle, so I imagine that anyone who is interested in this can
write their own makefile.  The include file names have not changed between CUDA 7.5 and 8.0 rc, so this can be built without
modifications to the source code (at least in linux) with CUDA 7.5 as well.  As far as I am aware, the only compatability
issue for older devices is the use of grids with x-dimensions larger than 65535 blocks.  However, this is only for devices
older than compute capability 3.0, and the source code here works without problems on compute capability >=5.0 devices
as I have verified.

This implementation of Oliveira's bucket method requires a fixed 10 bytes of DRAM per prime, which equates to just over 2 GB
for sieving up to 2<sup>64</sup>, which currently doesn't give the correct answer anyway (vide infra).  In any event, the fact that
large primes are handled in global memory, rather than on-chip, means that increasing the number of blocks working on the
task of sieving these large primes does not increase the amount of memory used since the data set is not duplicated.

The code here only demonstrates counting, but will very soon support creating lists of primes on the host.  This code already
exists, I just need to clean it up.
<br>

Known Issues
------------

  (1) The bottom of the sieving range must be a multiple of 2<sup>17</sup>.  This will be fixed in the near future<br>
  (2) There are instances where the count is off by 1-4 on certain ranges where less than an entire sieving range is counted<br>
  (3) Only multiples of 2<sup>24</sup> are acceptable inputs for the top or bottom if above 2<sup>40</sup>.<br>
  (4) Somewhere above 2<sup>58</sup>, the sieve starts crossing off actual primes due to something other than a race condition.  
      I am looking into this.<br>
<br>

State of the Project
-------------------
I am currently working on cleaning up the class structure of the host code as well as working out the kinks for sieving
the ranges above 2<sup>58</sup> and cleaning up the horrible mess that is my debugging code.
