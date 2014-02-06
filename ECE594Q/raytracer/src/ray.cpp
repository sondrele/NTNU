#include "ray.h"

Ray::Ray() {
    
}

Ray::Ray(Vect o, Vect d) {
    origin = o;
    direction = d;
}

Intersection::Intersection() {
    intersected = false;
}

Intersection::Intersection(Ray r) {
    intersected = false;
    ray = r;
}

void Intersection::setIntersectionPoint(float t) {
    pt = t;
    intersected = true;
}
