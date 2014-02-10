#ifndef _RAY_H_
#define _RAY_H_

#include "Matrix.h"

#define NV1      1.0f
#define NV2      1.5f

class Ray {
private:
    Vect origin;
    Vect direction;
    bool vacuum;

public:
    Ray();
    Ray(Vect, Vect);
    Vect getOrigin() { return origin;}
    void setOrigin(Vect o) { origin = o;}
    Vect getDirection() { return direction;}
    void setDirection(Vect d);
    void switchMedium();
    bool inVacuum();
};

#endif // _RAY_H_
