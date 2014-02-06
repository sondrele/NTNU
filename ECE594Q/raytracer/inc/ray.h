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
    void setDirection(Vect d) { direction = d;}

};

class Intersection {
private:
    Ray ray;
    float pt;
    bool intersected;

public:
    Intersection();
    Intersection(Ray);
    Vect getOrigin() { return ray.getOrigin(); }
    Vect getDirection() { return ray.getDirection(); }
    void setIntersectionPoint(float);
    float getIntersectionPoint() {return pt; }
    bool hasIntersected() { return intersected; }
};

#endif // _RAY_H_
