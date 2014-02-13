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

Ray Intersection::calculateRefraction() {
    if (intersected) {
        // Vect N = calculateSurfaceNormal();
        // Vect I = getDirection().invert();
        // Vect d0 = getDirection();
        // float t1 = I.radians(N);
        // float t2 = asin(sin(t1) * NV1 / NV2);
        // Vect Q = N.linearMult(I.dotProduct(N));
        // Q = Q - I;
        // Vect M = Q.linearMult((float)(sin(t2) / sin(t1)));
        // Vect P = N.linearMult((float) -cos(t2));
        // Vect T = M + P;
        // T.normalize();
        // // cout << N.invert().radians(T) << endl;
        // Ray r;
        // r.switchMedium();
        // r.setOrigin(calculateIntersectionPoint() + d0.linearMult(0.001f));
        // r.setDirection(T);
        // return r;

        Vect N = calculateSurfaceNormal();
        Vect d0 = getDirection();
        Ray r;
        r.switchMedium();
        r.setOrigin(calculateIntersectionPoint() + d0.linearMult(0.001f));
        r.setDirection(d0);
        return r;
    } else {
        throw "Cannot calculate refraction when no intersection has occured";
    }
}

SColor Intersection::getColor() {
    if (shape->getType() == TRIANGLE) {
        return ((Triangle *) shape)->
            interpolatedColor(calculateIntersectionPoint());
    } else {
        return getMaterial()->getDiffColor();
    }
}

Material * Intersection::getMaterial() {
    if (shape != NULL) { // && shape->getNumMaterials() > 0) {
        return shape->getMaterial();
    } else {
        return NULL;
    }
}

// Point_2D Intersection::calculateUVCoords() {
//     return Point_2D();
// }

std::string Intersection::toString() {
    stringstream s;
    s << "Intersected: " << intersected << endl;
    s << "  Ray: O=" << ray.getOrigin() << ", D=" << ray.getDirection() << ", inVacuum: " << ray.inVacuum() << endl;
    s << "  Intersectionpoint: " << calculateIntersectionPoint() << endl;
    if (shape != NULL)
        s << "  Shape: " << shape->getType() << " - " << shape << endl;
    return s.str();
}
