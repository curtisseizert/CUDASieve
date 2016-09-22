/*

CUDASieveDevice.cuh

Contains the __device__ functions and __constant__s for CUDASieve 1.0
by Curtis Seizert - cseizert@gmail.com

*/

  __constant__ uint32_t p3[3] = {2454267026ul, 613566756ul, 1227133513ul};
  __constant__ uint32_t p5[5] = {138547332ul, 1108378657ul, 277094664ul, 2216757314ul, 554189328ul};
  __constant__ uint32_t p7[7] = {2164392968ul, 135274560ul, 1082196484ul, 67637280ul, 541098242ul, 33818640ul, 270549121ul};
  __constant__ uint32_t p11[11] = {134283296ul, 268566592ul, 537133184ul, 1074266368ul, 2148532736ul, 2098176ul, 4196353ul, 8392706ul, 16785412ul, 33570824ul, 67141648ul};
  __constant__ uint32_t p13[13] = {524352ul, 67117057ul, 1048704ul, 134234114ul, 2097408ul, 268468228ul, 4194816ul, 536936456ul, 8389632ul, 1073872912ul, 16779264ul, 2147745824ul, 33558528ul};
  __constant__ uint32_t p17[17] = {33554688ul, 134218752ul, 536875008ul, 2147500032ul, 65536ul, 262146ul, 1048584ul, 4194336ul, 16777344ul, 67109376ul, 268437504ul, 1073750016ul, 32768ul, 131073ul, 524292ul, 2097168ul, 8388672ul};
  __constant__ uint32_t p19[19] = {268435968ul, 32768ul, 2097156ul, 134217984ul, 16384ul, 1048578ul, 67108992ul, 8192ul, 524289ul, 33554496ul, 2147487744ul, 262144ul, 16777248ul, 1073743872ul, 131072ul, 8388624ul, 536871936ul, 65536ul, 4194312ul};
  __constant__ uint32_t p23[23] = {2048ul, 33554436ul, 65536ul, 1073741952ul, 2097152ul, 4096ul, 67108872ul, 131072ul, 2147483904ul, 4194304ul, 8192ul, 134217744ul, 262144ul, 512ul, 8388609ul, 16384ul, 268435488ul, 524288ul, 1024ul, 16777218ul, 32768ul, 536870976ul, 1048576ul};
  __constant__ uint32_t p27[27] = {8192ul, 256ul, 1073741832ul, 33554432ul, 1048576ul, 32768ul, 1024ul, 32ul, 134217729ul, 4194304ul, 131072ul, 4096ul, 128ul, 536870916ul, 16777216ul, 524288ul, 16384ul, 512ul, 2147483664ul, 67108864ul, 2097152ul, 65536ul, 2048ul, 64ul, 268435458ul, 8388608ul, 262144ul};
  __constant__ uint32_t p29[29] = {16384ul, 2048ul, 256ul, 32ul, 2147483652ul, 268435456ul, 33554432ul, 4194304ul, 524288ul, 65536ul, 8192ul, 1024ul, 128ul, 16ul, 1073741826ul, 134217728ul, 16777216ul, 2097152ul, 262144ul, 32768ul, 4096ul, 512ul, 64ul, 8ul, 536870913ul, 67108864ul, 8388608ul, 1048576ul, 131072ul};
  __constant__ uint32_t p31[31] = {32768ul, 16384ul, 8192ul, 4096ul, 2048ul, 1024ul, 512ul, 256ul, 128ul, 64ul, 32ul, 16ul, 8ul, 4ul, 2ul, 2147483649ul, 1073741824ul, 536870912ul, 268435456ul, 134217728ul, 67108864ul, 33554432ul, 16777216ul, 8388608ul, 4194304ul, 2097152ul, 1048576ul, 524288ul, 262144ul, 131072ul, 65536ul};
  __constant__ uint32_t p37[37] = {262144ul, 8388608ul, 268435456ul, 0ul, 2ul, 64ul, 2048ul, 65536ul, 2097152ul, 67108864ul, 2147483648ul, 0ul, 16ul, 512ul, 16384ul, 524288ul, 16777216ul, 536870912ul, 0ul, 4ul, 128ul, 4096ul, 131072ul, 4194304ul, 134217728ul, 0ul, 1ul, 32ul, 1024ul, 32768ul, 1048576ul, 33554432ul, 1073741824ul, 0ul, 8ul, 256ul, 8192ul};
  __constant__ uint8_t wheel30[8] = {1,7,11,13,17,19,23,29};
  __constant__ uint8_t wheel30Inc[8] = {6,4,2,4,2,4,6,2};
  __constant__ uint8_t lookup30[30] = {0,0,0,0,0,0,0,1,0,0,0,2,0,3,0,0,0,4,0,5,0,0,0,6,0,0,0,0,0,7};

  __constant__ uint16_t threads = 256;
  __constant__ uint32_t cutoff = 32768;

namespace device
{
  __device__ void sieveSmallPrimes(uint32_t * s_sieve, uint32_t sieveWords, uint64_t bstart);
  __device__ void sieveFirst(uint32_t * s_sieve, uint32_t sieveBits);
  __device__ void sieveMiddlePrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimes(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesPL(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesBase(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveMedPrimesBasePL(uint32_t * s_sieve, uint32_t * d_primeList, uint64_t bstart, uint32_t primeListLength, uint32_t sieveBits);
  __device__ void sieveInit(uint32_t * s_sieve, uint32_t sieveWords);
  __device__ void countPrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords); // retains the original sieve data
  __device__ void countPrimesHist(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords); // retains the original sieve data
  __device__ void countPrimes(uint32_t * s_sieve, uint32_t sieveWords); // destroys original sieve data
  __device__ void countPrimesRemBottom(uint32_t * s_sieve, uint32_t sieveWords, uint32_t bottom);
  __device__ void countTopPrimes(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint64_t bstart, uint64_t top, volatile uint64_t * d_count);
  __device__ void moveCount(uint32_t * s_sieve, volatile uint64_t * d_count);
  __device__ void moveCountHist(uint32_t * s_sieve, uint32_t * d_histogram);
  __device__ void makeBigSieve(uint32_t * bigSieve, uint32_t * s_sieve, uint32_t sieveWords);
  __device__ void exclusiveScan(uint16_t * s_array, uint32_t size);
  __device__ void exclusiveScanBig(uint32_t * s_array, uint32_t size);
  __device__ void movePrimes(uint32_t * s_sieve, uint16_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, uint32_t * d_histogram, uint64_t bstart, uint32_t maxPrime);
  __device__ void movePrimesFirst(uint32_t * s_sieve, uint32_t * s_counts, uint32_t sieveWords, uint32_t * d_primeList, uint32_t * d_histogram, uint64_t bstart, uint32_t maxPrime);
}
