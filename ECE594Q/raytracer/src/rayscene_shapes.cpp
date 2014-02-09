#include "rayscene.h"

SColor::SColor(Color c) {
    R(c[0]);
    G(c[1]);
    B(c[2]);
}

SColor::SColor(Vect v) {
    R(v.getX());
    G(v.getY());
    B(v.getZ());
}

SColor::SColor(float r, float g, float b) {
    R(r);
    G(g);
    B(b);
}

SColor& SColor::operator=(const Vect &other) {
    R(other.getX());
    G(other.getY());
    B(other.getZ());
    
    return *this;
}

void SColor::R(float r) {
    if (r > 1.0f)
        r = 1;
    setX(r);
}

void SColor::G(float g) {
    if (g > 1.0f)
        g = 1;
    setY(g);
}

void SColor::B(float b) {
    if (b > 1.0f)
        b = 1;
    setZ(b);
}

SColor SColor::mult(SColor other) {
    SColor c;
    c.R(R() * other.R());
    c.G(G() * other.G());
    c.B(B() * other.B());
    return c;
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

Shape::Shape() {

}

Shape::~Shape() {

}

Sphere::Sphere() {
    radius = 0;
}

Vect Sphere::surfaceNormal(Vect o, Vect pt) {
    (void) o;
    Vect normal = pt - origin;
    normal.normalize();
    return normal;
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

Mesh::~Mesh() {
    for (uint i = 0; i < triangles.size(); i++) {
        delete triangles[i];
    }
}

void Mesh::addTriangle(Triangle t) {
    Triangle *tri = new Triangle();
    *tri = t;
    triangles.push_back(tri);
}

Intersection Mesh::intersects(Ray ray) {
    Intersection is;
    for (uint i = 0; i < triangles.size(); i++) {
        Triangle *t0 = triangles.at(i);
        is = t0->intersects(ray);
        if (is.hasIntersected())
            return is;
    }
    return is;
}

Vect Mesh::surfaceNormal(Vect o, Vect pt) {
    (void) o;
    throw "Mesh has no surface normal";
    return pt;
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
