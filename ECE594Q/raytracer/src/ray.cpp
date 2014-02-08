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

Vect Intersection::calculateIntersectionPoint() {
    return (ray.getDirection()).linearMult(pt);
}

Vect Intersection::getSurfaceNormal() {
    throw "getSurfaceNormal not implemented";
}

Vect Intersection::getDiffuseReflectionCoeff() {
    throw "getDiffuseReflectionCoeff not implemented";
}

float Intersection::getScalarTransmissionCoeff() {
    return 0;
}

float Intersection::getScalarSpecularCoeff() {
    return 0;
}

float Intersection::getShininess() {
    return 0;
}
