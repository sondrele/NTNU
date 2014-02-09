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

SColor Intersection::getColor() {
    if (shape != NULL && shape->getNumMaterials() > 0) {
        return shape->getMaterial(0).getDiffColor();
    } else {
        return SColor(0, 0, 0);
    }
}

float Intersection::getTransparency() {
    if (shape != NULL && shape->getNumMaterials() > 0) {
        return shape->getMaterial(0).getTransparency();
    } else {
        // cout << "No transparency" << endl;
        return 0;
    }
}

float Intersection::getShininess() {
    if (shape != NULL && shape->getNumMaterials() > 0) {
        return shape->getMaterial(0).getShininess();
    } else {
        // cout << "No shininess" << endl;
        return 0;
    }
}
