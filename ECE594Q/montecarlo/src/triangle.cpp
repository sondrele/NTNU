#include "rayscene_shapes.h"

Vertex::Vertex() {
    mat = NULL;
    hasNormal = false;
}

Vertex::Vertex(float a, float b, float c) {
    mat = NULL;
    hasNormal = false;

    setX(a);
    setY(b);
    setZ(c);
}

void Vertex::setSurfaceNormal(Vect n) {
    normal = n;
    normal.normalize();
    hasNormal = true;
}

Vect Vertex::getSurfaceNormal() {
    if (hasNormal)
        return normal;
    else 
        throw "Vertex has no normal";
}

void Vertex::setTextureCoords(float x, float y) {
    coords.x = x;
    coords.y = y;
}

Point_2D Vertex::getTextureCoords() {
    return coords;
}

Vertex& Vertex::operator=(const Vect &other) {
    setX(other.getX());
    setY(other.getY());
    setZ(other.getZ());
    
    return *this;
}

Triangle::Triangle() {
    // mat = NULL;
}

Triangle::~Triangle() {
    // delete mat;
}

void Triangle::setMaterial(Material *m, char v) {
    if (v == 'a')
        a.setMaterial(m);
    else if (v == 'b')
        b.setMaterial(m);
    else if (v == 'c')
        c.setMaterial(m);
}

// Deprecated
Material * Triangle::getMaterial() {
    return a.getMaterial();//materials[0];
}

Material * Triangle::getMaterial(char v) {
    if (v == 'a') {
        return a.getMaterial();
    } else if(v == 'b') {
        return b.getMaterial();
    } else if (v == 'c') {
        return c.getMaterial();
    } else {
        throw "MaterialIndex out of bounds";
    }
}

Intersection Triangle::intersects(Ray ray) {
    Intersection is(ray, this);

    Vect p = ray.getOrigin();
    Vect d = ray.getDirection();
    Vect v0 = getA();
    Vect v1 = getB();
    Vect v2 = getC();

    Vect e1, e2, h, s, q;
    float a0, f, u, v;

    e1 = v1 - v0;
    e2 = v2 - v0;

    h = d.crossProduct(e2);
    a0 = e1.dotProduct(h);

    if (a0 > -0.00001 && a0 < 0.00001)
        return is;

    f = 1 / a0;
    s = p - v0;
    u = f * (s.dotProduct(h));

    if (u < 0.0 || u > 1.0)
        return is;

    q = s.crossProduct(e1);
    v = f * d.dotProduct(q);

    if (v < 0.0 || u + v > 1.0)
        return is;

    // at this stage we can compute t to find out where
    // the intersection point is on the line
    float t = f * e2.dotProduct(q);

    if (t > 0.00001) { // ray intersection
        is.setIntersectionPoint(t);
        return is;
    }

    else // this means that there is a line intersection
         // but not a ray intersection
         return is;
}

Vect Triangle::normal() {
    Vect v = getB() - getA();
    Vect w = getC() - getA();
    Vect N = v.crossProduct(w);
    N.normalize();
    return N;
}

Vect Triangle::surfaceNormal(Vect dir, Vect pt) {
    (void) pt;

    Vect v = getB() - getA();
    Vect w = getC() - getA();
    Vect N = v.crossProduct(w);
    N.normalize();

    if (N.dotProduct(dir) > 0)
        N = N.linearMult(-1);
    
    return N;
}

float Triangle::getArea() {
    return Triangle::getArea((Vect) a, (Vect) b, (Vect) c);
}

float Triangle::getArea(Vect A, Vect B, Vect C) {
    Vect AB = B - A;
    Vect AC = C - A;
    return AB.crossProduct(AC).length() * 0.5f;
}

Vect Triangle::interPolatedNormal(Vect pt) {
    float A = getArea();
    float A0 = getArea((Vect) a, (Vect) b, pt) / A;
    float A1 = getArea((Vect) c, (Vect) a, pt) / A;
    float A2 = getArea((Vect) b, (Vect) c, pt) / A;
    if (A0 > 1 || A1 > 1 || A2 > 1) 
        throw "Point is outside triangle";

    Vect interpolated = a.getSurfaceNormal().linearMult(A2)
        + b.getSurfaceNormal().linearMult(A1)
        + c.getSurfaceNormal().linearMult(A0);
    interpolated.normalize();
    return interpolated;
}
