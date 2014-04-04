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
        r.setOrigin(calculateIntersectionPoint() + N.linearMult(0.0001f));
        r.setDirection(N.linearMult(d0.dotProduct(N) * 2) - d0);
        return r;
    } else {
        throw "Cannot calculate reflection when no intersection has occured";
    }
}

bool Intersection::calculateRefraction(Ray &r) {
    if (intersected) {
        Vect I = getDirection();
        Vect N = calculateSurfaceNormal();
        float n = ray.inVacuum() ? NV1 / NV2 : NV2 / NV1;

        float cosI = N.dotProduct(I);
        if(cosI  > 0) {
           N = N.invert();
        }

        float c = I.dotProduct(N);
        float cosPhi2 = (1 - ((n * n) * (1 - (c * c))));
        if (cosPhi2 < 0) 
            return false;
        else {
            float cosPhi = sqrt(cosPhi2);
            Vect term1 = (I - N * c);
            term1.linearMult(n);
            Vect d0 = term1 - N.linearMult(cosPhi);
            r = ray;
            r.switchMedium();
            r.setOrigin(calculateIntersectionPoint() + d0.linearMult(0.01f));
            r.setDirection(d0);
            return true;
        }
    } else {
        throw "Cannot calculate refraction when no intersection has occured";
    }
}

SColor Intersection::getColor() {
    return shape->getColor(calculateIntersectionPoint());
}


SColor Intersection::diffColor() {
    return shape->getMaterial()->getDiffColor();
}

SColor Intersection::specColor() {
    return shape->getMaterial()->getSpecColor();
}

SColor Intersection::ambColor() {
    return shape->getMaterial()->getAmbColor();
}

Material * Intersection::getMaterial() {
    if (shape != NULL) {
        return shape->getMaterial();
    } else {
        return NULL;
    }
}

std::string Intersection::toString() {
    stringstream s;
    s << "Intersected: " << intersected << endl;
    s << "  Ray: O=" << ray.getOrigin() << ", D=" << ray.getDirection() << ", inVacuum: " << ray.inVacuum() << endl;
    s << "  Intersectionpoint: " << calculateIntersectionPoint() << endl;
    if (shape != NULL)
        s << "  Shape: " << shape->getType() << " - " << shape << endl;
    return s.str();
}
