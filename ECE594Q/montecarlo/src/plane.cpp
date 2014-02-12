#include "rayscene_shapes.h"

Intersection Plane::intersects(Ray r) {
    Vect origin = r.getOrigin();
    float t = (normal.dotProduct(point - origin)) / (origin.dotProduct(r.getDirection()));
    
    Intersection in(r);
    if (t >= 0)
        in.setIntersectionPoint(t);
    return in;
}
