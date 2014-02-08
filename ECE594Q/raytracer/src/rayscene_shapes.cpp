#include "rayscene.h"

SColor::SColor(Color c) {
    setX(c[0]);
    setY(c[1]);
    setZ(c[2]);
}

SColor::SColor(float r, float g, float b) {
    setX(r);
    setY(g);
    setZ(b);
}

Material::Material() {
    shininess = 1;
    transparency = 0;
}

Vertex::Vertex() {
    materialPos = 0;
}

Vertex::Vertex(uint p) {
    materialPos = p;
}

Vertex& Vertex::operator=(const Vect &other) {
    setX(other.getX());
    setY(other.getY());
    setZ(other.getZ());
    
    return *this;
}

Sphere::Sphere() {
    radius = 0;
}

ShapeType Sphere::getType() {
    return SPHERE;
}

ShapeType Triangle::getType() {
    return TRIANGLE;
}

ShapeType Mesh::getType() {
    return MESH;
}

bool Sphere::intersects(Ray ray, float &t) {
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
        return false;

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
        return false;

    // if t0 is less than zero, the intersection point is at t1
    if (t0 < 0) {
        t = t1;
        return true;
    }
    // else the intersection point is at t0
    else {
        t = t0;
        return true;
    }
}

void Mesh::addTriangle(Triangle t) {
    triangles.push_back(t);
}

bool Mesh::intersects(Ray ray, float &t) {
    for (uint i = 0; i < triangles.size(); i++) {
        Triangle t0 = triangles.at(i);
        if (t0.intersects(ray, t))
            return true;
    }
    return false;
}

bool Triangle::intersects(Ray ray, float &t) {
    Vect p = ray.getOrigin();
    Vect d = ray.getDirection();
    Vect v0 = getA();
    Vect v1 = getB();
    Vect v2 = getC();

    Vect e1, e2, h, s, q;
    float a0,f,u,v;

    e1 = v1 - v0;
    e2 = v2 - v0;

    h = d.crossProduct(e2);
    a0 = e1.dotProduct(h);

    if (a0 > -0.00001 && a0 < 0.00001)
        return false;

    f = 1/a0;
    s = p - v0;
    u = f * (s.dotProduct(h));

    if (u < 0.0 || u > 1.0)
        return false;

    q = s.crossProduct(e1);
    v = f * d.dotProduct(q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    // at this stage we can compute t to find out where
    // the intersection point is on the line
    t = f * e2.dotProduct(q);

    if (t > 0.00001) // ray intersection
        return true;

    else // this means that there is a line intersection
         // but not a ray intersection
         return false;
}
