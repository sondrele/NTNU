#ifndef _INTERSECTION_H_
#define _INTERSECTION_H_

#include <cstdlib>
#include <sstream>

#include "ray.h"
#include "rayscene_shapes.h"

class SColor;
class Shape;
class Material;

class Intersection {
private:
    Ray ray;
    Shape *shape;
    // Point_2D coord;
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
    Ray calculateReflection();
    Ray calculateRefraction();
    // Point_2D calculateUVCoords();

    void setShape(Shape *s) { shape = s; }
    Shape * getShape() { return shape; }
    Material *getMaterial();

    std::string toString();
};

#endif
