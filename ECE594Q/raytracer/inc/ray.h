#ifndef _RAY_H_
#define _RAY_H_

#include "Matrix.h"

class Ray {
private:
    Vect origin;
    Vect direction;

public:
    Ray();
    Ray(Vect, Vect);
    Vect getOrigin() { return origin;}
    void setOrigin(Vect o) { origin = o;}
    Vect getDirection() { return direction;}
    void setDirection(Vect d);
};

#endif // _RAY_H_
