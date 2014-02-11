#include "rayscene_shapes.h"

Sphere::Sphere() {
    radius = 0;
}

Vect Sphere::surfaceNormal(Vect o, Vect pt) {
    (void) o;
    Vect normal = pt - origin;
    normal.normalize();
    return normal;
}

Material * Sphere::getMaterial() {
    return materials[0];
}

Intersection Sphere::intersects(Ray ray) {
    Intersection is(ray, this);
    // Transforming ray to object space
    Vect transformedOrigin = ray.getOrigin() - getOrigin();

    //Compute A, B and C coefficients
    Vect dest = ray.getDirection();
    Vect orig = transformedOrigin;

    float a = dest.dotProduct(dest);
    float b = 2 * dest.dotProduct(orig);
    float c = orig.dotProduct(orig) - (radius * radius);

    //Find discriminant
    float disc = b * b - 4 * a * c;
    // if discriminant is negative there are no real roots, so return 
    // false as ray misses sphere
    if (disc < 0)
        return is;

    // compute q as described above
    float distSqrt = sqrtf(disc);
    float q;
    if (b < 0)
        q = (-b - distSqrt) / 2.0f;
    else
        q = (-b + distSqrt) / 2.0f;

    // compute t0 and t1
    float t0 = q / a;
    float t1 = c / q;

    // make sure t0 is smaller than t1
    if (t0 > t1) {
        // if t0 is bigger than t1 swap them around
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    // if t1 is less than zero, the object is in the ray's negative direction
    // and consequently the ray misses the sphere
    if (t1 < 0)
        return is;

    // if t0 is less than zero, the intersection point is at t1
    if (t0 < 0) {
        is.setIntersectionPoint(t1);
        return is;
    }
    // else the intersection point is at t0
    else {
        is.setIntersectionPoint(t0);
        return is;
    }
}
