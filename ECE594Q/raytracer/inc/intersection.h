#ifndef _INTERSECTION_H_
#define _INTERSECTION_H_

#include "ray.h"
#include "rayscene_shapes.h"

class Intersection {
private:
    Ray ray;
    Shape *shape;
    float pt;
    bool intersected;

public:
    Intersection();
    Intersection(Ray);
    Intersection(Ray, Shape *);
    ~Intersection();
    Vect getOrigin() { return ray.getOrigin(); }
    Vect getDirection() { return ray.getDirection(); }
    void setIntersectionPoint(float);
    float getIntersectionPoint() {return pt; }
    Vect calculateIntersectionPoint();
    bool hasIntersected() { return intersected; }

    Vect getSurfaceNormal();
    SColor getColor();
    float getTransparency();
    float getShininess();
    Shape * getShape() { return shape; }
};

#endif
