#include "Utils.h"

float Utils::Sign(Vect v1, Vect v2, Vect v3) {
    return (v1.getX() - v3.getX()) * (v2.getY() - v3.getY()) -
        (v2.getX() - v3.getX()) * (v1.getY() - v3.getY());
}

bool Utils::PointInTriangle(Vect pt, Vect v1, Vect v2, Vect v3) {
    bool b1, b2, b3;

    b1 = Utils::Sign(pt, v1, v2) < 0.0f;
    b2 = Utils::Sign(pt, v2, v3) < 0.0f;
    b3 = Utils::Sign(pt, v3, v1) < 0.0f;

    return ((b1 == b2) && (b2 == b3));
}
