#ifndef _R_RAND_H_
#define _R_RAND_H_

#include <random>

#include "Matrix.h"

class Rand {
public:
    static double Random();
    static Vect RandomVect();
};

#endif