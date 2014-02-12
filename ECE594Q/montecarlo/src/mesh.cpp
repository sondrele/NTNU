#include "rayscene_shapes.h"

Mesh::Mesh() {
    textureCoords = false;
    hasPerSurfaceNormal = false;
    objectMaterial = false;
}

Mesh::~Mesh() {
    for (uint i = 0; i < triangles.size(); i++) {
        delete triangles[i];
    }
}

Material * Mesh::getMaterial() {
    return materials[0];
}

void Mesh::addTriangle(Triangle *t) {
    t->setPerVertexNormal(hasPerSurfaceNormal);
    t->setPerVertexMaterial(objectMaterial);
    triangles.push_back(t);
}

Intersection Mesh::intersects(Ray ray) {
    Intersection ins;
    for (uint i = 0; i < triangles.size(); i++) {
        Triangle *t0 = triangles.at(i);
        Intersection j = t0->intersects(ray);
        if (!ins.hasIntersected()) {
            ins = j;
        } else if (ins.hasIntersected() &&
            j.getIntersectionPoint() < ins.getIntersectionPoint())
        {
            ins = j;
        }
    }
    return ins;
}

Vect Mesh::surfaceNormal(Vect o, Vect pt) {
    (void) o;
    throw "Mesh has no surface normal";
    return pt;
}
