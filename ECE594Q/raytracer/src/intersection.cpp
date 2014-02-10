#include "intersection.h"

Intersection::Intersection() {
    shape = NULL;
    intersected = false;
}

Intersection::Intersection(Ray r) {
    shape = NULL;
    intersected = false;
    ray = r;
}

Intersection::Intersection(Ray r, Shape *s) {
    shape = s;
    intersected = false;
    ray = r;
}

Intersection::~Intersection() {
    shape = NULL;
}

void Intersection::setIntersectionPoint(float t) {
    pt = t;
    intersected = true;
}

Vect Intersection::calculateIntersectionPoint() {
    return (ray.getDirection()).linearMult(pt) + ray.getOrigin();
}

Vect Intersection::calculateSurfaceNormal() {
    if (shape != NULL) {
        return shape->surfaceNormal(ray.getDirection(), 
            calculateIntersectionPoint());
    } else
        return Vect(0, 0, 0);
}

Ray Intersection::calculateReflection() {
    if (intersected) {
        Vect N = calculateSurfaceNormal();
        Vect d0 = getDirection().linearMult(-1);
        Ray r;
        r.setOrigin(calculateIntersectionPoint());
        r.setDirection(N.linearMult(d0.dotProduct(N) * 2) - d0);
        return r;
    } else {
        throw "Cannot calculate reflection when no intersection has occured";
    }
}

Material * Intersection::getMaterial() {
    if (shape != NULL && shape->getNumMaterials() > 0) {
        return shape->getMaterial();
    } else {
        return NULL;
    }
}
