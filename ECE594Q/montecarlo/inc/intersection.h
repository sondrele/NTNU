#ifndef _INTERSECTION_H_
#define _INTERSECTION_H_

#include <cstdlib>
#include <sstream>

#include "ray.h"
#include "shapes.h"

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
    bool hasRefracted() { return !ray.inVacuum(); }

    Vect calculateIntersectionPoint();
    Vect calculateSurfaceNormal();
    Ray calculateReflection();
    bool calculateRefraction(Ray &);
    // Point_2D calculateUVCoords();

    void setShape(Shape *s) { shape = s; }
    Shape * getShape() { return shape; }
    Material *getMaterial();
    SColor getColor();
    SColor diffColor();
    SColor specColor();
    SColor ambColor();

    std::string toString();
};

#endif
