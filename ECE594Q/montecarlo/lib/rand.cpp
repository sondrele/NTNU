#include "rand.h"

double Rand::Random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

Vect Rand::RandomVect() {
    float x = 1 - 2 * (float) Rand::Random();
    float y = 1 - 2 * (float) Rand::Random();
    float z = 1 - 2 * (float) Rand::Random();
    Vect v(x, y, z);
    v.normalize();
    return v;
}
