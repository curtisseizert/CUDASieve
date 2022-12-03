#include <stdint.h>
#include <map>

#include "CUDASieve/cudasieve.hpp"

const uint64_t SIEVE_SIZE = 1000000;

const std::map<uint64_t, const int> resultsDictionary =
{
    {          10UL, 4         },               // Historical data for validating our results - the number of primes
    {         100UL, 25        },               // to be found under some limit, such as 168 primes under 1000
    {        1000UL, 168       },
    {       10000UL, 1229      },
    {      100000UL, 9592      },
    {     1000000UL, 78498     },
    {    10000000UL, 664579    },
    {   100000000UL, 5761455   },
    {  1000000000UL, 50847534  },
    { 10000000000UL, 455052511 },
};

void printResults(uint64_t sieveSize, size_t primeCount, double duration, int passes)
{
    auto expectedCount = resultsDictionary.find(sieveSize);
    auto validCount = expectedCount != resultsDictionary.end() && expectedCount->second == primeCount;

    fprintf(stderr, "Passes: %d, Time: %lf, Avg: %lf, Limit: %ld, Count: %d, Valid: %d\n", 
            passes,
            duration,
            duration / passes,
            sieveSize,
            primeCount,
            validCount);

    fprintf(stderr, "\n");
    printf("rbergen_cuda;%d;%f;1;algorithm=other,faithful=yes,bits=1\n", passes, duration);
}

int main()
{
    uint64_t passes = 0;
    auto tStart = steady_clock::now();
    size_t primeCount;

    while (true)
    {
        // Implementation is faithful because CudaSieve::getDevicePrimes creates and destroys a sieve class instance
        uint64_t *primes = CudaSieve::getDevicePrimes(0, SIEVE_SIZE, primeCount);
        passes++;
        if (duration_cast<seconds>(steady_clock::now() - tStart).count() >= 5)
        {
            printResults(SIEVE_SIZE, primeCount, duration_cast<microseconds>(steady_clock::now() - tStart).count() / 1000000.0, passes);
            break;
        }
    } 
}

