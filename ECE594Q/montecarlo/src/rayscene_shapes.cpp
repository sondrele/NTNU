#include "rayscene.h"

/*********************************
* BBox
*********************************/
BBox::BBox() {

}

bool operator < (const BBox &a, const BBox &b) {
    return a.getCentroid() < b.getCentroid();
}

const BBox operator +(const BBox &a, const BBox &b) {
    Vect pmin;
    pmin.setX(min(a.pmin.getX(), b.pmin.getX()));
    pmin.setY(min(a.pmin.getY(), b.pmin.getY()));
    pmin.setZ(min(a.pmin.getZ(), b.pmin.getZ()));

    Vect pmax;
    pmax.setX(max(a.pmax.getX(), b.pmax.getX()));
    pmax.setY(max(a.pmax.getY(), b.pmax.getY()));
    pmax.setZ(max(a.pmax.getZ(), b.pmax.getZ()));

    BBox box;
    box.setMin(pmin); box.setMax(pmax);

    return box;
}

bool BBox::intersects(Ray r) {
    Vect orig = r.getOrigin();
    Vect dir = r.getDirection();

    float tmin = (pmin.getX() - orig.getX()) / dir.getX();
    float tmax = (pmax.getX() - orig.getX()) / dir.getX();
    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (pmin.getY() - orig.getY()) / dir.getY();
    float tymax = (pmax.getY() - orig.getY()) / dir.getY();
    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (pmin.getZ() - orig.getZ()) / dir.getZ();
    float tzmax = (pmax.getZ() - orig.getZ()) / dir.getZ();
    if (tzmin > tzmax) swap(tzmin, tzmax);
    
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    return true;
}

Vect BBox::getCentroid() const {
    return pmin.linearMult(0.5f) + pmax.linearMult(0.5f);
}

/*********************************
* Shape
*********************************/
Shape::Shape() {
    texture = NULL;
    cShader = NULL;
    iShader = new IShader();
}

Shape::~Shape() {
    for (uint i = 0; i < materials.size(); i++) {
        delete materials[i];
    }

    if (texture != NULL) {
        delete texture;
    }

    if (cShader != NULL) {
        delete cShader;
    }

    if (iShader != NULL) {
        delete iShader;
    }
}

void Shape::addMaterial(Material *m) {
    materials.push_back(m);
}

Material * Shape::getMaterial() {
    return materials[0];
}

void Shape::setTexture(Texture *t) {
    texture = t;
}

Texture * Shape::getTexture() {
    return texture;
}

void Shape::setCShader(CShader *s) {
    if (cShader != NULL) {
        delete  cShader;
    }
    cShader = s;
    cShader->setTexture(texture);
    cShader->setMaterial(materials[0]);
}

CShader * Shape::getCShader() {
    return cShader;
}

void Shape::setIShader(IShader *s) {
    if (iShader != NULL) {
        delete iShader;
    }
    iShader = s;
}

IShader * Shape::getIShader() {
    return iShader;
}

bool Shape::CompareX(Shape *a, Shape *b) {
    return (a->getBBox().getMin()).getX() < (b->getBBox().getMin()).getX();
}

bool Shape::CompareY(Shape *a, Shape *b) {
    return (a->getBBox().getMin()).getY() < (b->getBBox().getMin()).getY();
}

bool Shape::CompareZ(Shape *a, Shape *b) {
    return (a->getBBox().getMin()).getZ() < (b->getBBox().getMin()).getZ();
}

bool operator < (Shape &a, Shape &b) {
    return a.getBBox() < b.getBBox();
}