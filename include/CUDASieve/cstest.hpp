// Definitions for cstest.cpp

#include <stdint.h>
#include <stdlib.h>
#include <iostream>

#pragma once

void dispFactors(uint64_t n, bool skipline = 0);
void mr_check(uint64_t * primes, int64_t numToCheck, size_t len, bool skipline = 0);
void listDevices();

class numGuess{
private:
  uint16_t idx;
  uint32_t lessthan, maxOver = 0, maxUnder = 0, minOver = 100000;
  uint64_t maxFail = 0, sumOver;
  double maxOver_d, maxUnder_d, sumOver_d;

public:
  uint64_t guess;

  numGuess(uint16_t idx){this->idx = idx;}

  template <typename T, typename U>
  void updateStats(T primes, U bottom, U top)
  {
    if(guess < primes){
      lessthan++;
      if(top - bottom > maxFail) maxFail = top - bottom;
      if(primes - guess > maxUnder) {maxUnder = primes - guess; maxUnder_d = (double) (primes - guess)/primes;}
    }
    else {sumOver += guess - primes; sumOver_d += (double) (guess - primes)/ primes;}
    if(guess - primes > maxOver && guess > primes) {maxOver = guess - primes; maxOver_d = (double) (guess - primes)/primes;}
    if(guess - primes < minOver) {minOver = guess - primes;}
  }

  template <typename T>
  void displayStats(T numTrials)
  {

    std::cout << "=== numGuess" << idx << " ===" << std::endl;
    std::cout << "Number of times less than count : " << lessthan << std::endl;
    std::cout << "Maximum excess allocation       : " << maxOver << " (" << (double) 100*maxOver_d << "%)"<< std::endl;
    std::cout << "Minimum excess allocation       : " << minOver << std::endl;
    std::cout << "Maximum under-allocation        : " << maxUnder << " (" << (double) 100*maxUnder_d << "%)"<< std::endl;
    std::cout << "Average excess allocation       : " << sumOver/numTrials << " (" << (double) 100*sumOver_d/numTrials << "%)" << std::endl;
    std::cout << "Maximum range for failure       : " << maxFail << std::endl;
    std::cout << std::endl;
  }
};
