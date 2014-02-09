#ifndef _INTERSECTION_H_
#define _INTERSECTION_H_

#include "ray.h"
#include "rayscene_shapes.h"

class SColor;
class Shape;
class Material;

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
    bool hasIntersected() { return intersected; }
    Vect calculateIntersectionPoint();
    Vect calculateSurfaceNormal();

    void setShape(Shape *s) { shape = s; }
    Shape * getShape() { return shape; }
    Material *getMaterial();
};

#endif
