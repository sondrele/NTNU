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

    BBox b = t->getBBox();
    Vect l0 = bbox.getLowerLeft();
    Vect l1 = b.getLowerLeft();
    l1.setX(min(l1.getX(), l0.getX()));
    l1.setY(min(l1.getY(), l0.getY()));
    l1.setZ(min(l1.getZ(), l0.getZ()));
    bbox.setLowerLeft(l1);

    Vect r0 = bbox.getUpperRight();
    Vect r1 = b.getUpperRight();
    r1.setX(max(r1.getX(), r0.getX()));
    r1.setY(max(r1.getY(), r0.getY()));
    r1.setZ(max(r1.getZ(), r0.getZ()));
    bbox.setUpperRight(r1);
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

SColor Mesh::getColor(Vect pt) {
    (void) pt;
    throw "Mesh has no color";
}

BBox Mesh::getBBox() {
    return bbox;
}