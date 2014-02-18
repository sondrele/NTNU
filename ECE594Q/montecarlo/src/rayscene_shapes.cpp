#include "rayscene.h"

/*********************************
* BBox
*********************************/
BBox::BBox() {

}

bool operator < (const BBox &a, const BBox &b) {
    return a.getLowerLeft() < b.getLowerLeft();
}

// bool BBox::intersects(Ray r) {
//     Vect invDir = r.getDirection().invert();
//     Vect origin = r.getOrigin();
//     double tx1 = (lowerLeft.getX() - origin.getX()) * invDir.getX();
//     double tx2 = (upperRight.getX() - origin.getX()) * invDir.getX();

//     double txmin = min(tx1, tx2);
//     double txmax = max(tx1, tx2);

//     double ty1 = (lowerLeft.getY() - origin.getY()) * invDir.getY();
//     double ty2 = (upperRight.getY() - origin.getY()) * invDir.getY();

//     double tymin = min(txmin, min(ty1, ty2));
//     double tymax = max(txmax, max(ty1, ty2));

//     double tz1 = (lowerLeft.getZ() - origin.getZ()) * invDir.getZ();
//     double tz2 = (upperRight.getZ() - origin.getZ()) * invDir.getZ();

//     double tzmin = min(tymin, min(tz1, tz2));
//     double tzmax = max(tymax, max(tz1, tz2));

//     return tzmax >= tzmin && tzmax >= 0;
// }

bool BBox::intersects(Ray r) {
    Vect orig = r.getOrigin();
    Vect dir = r.getDirection();

    float tmin = (lowerLeft.getX() - orig.getX()) / dir.getX();
    float tmax = (upperRight.getX() - orig.getX()) / dir.getX();
    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (lowerLeft.getY() - orig.getY()) / dir.getY();
    float tymax = (upperRight.getY() - orig.getY()) / dir.getY();
    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (lowerLeft.getZ() - orig.getZ()) / dir.getZ();
    float tzmax = (upperRight.getZ() - orig.getZ()) / dir.getZ();
    if (tzmin > tzmax) swap(tzmin, tzmax);
    
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    
    // if (tzmin > tmin)
    //     tmin = tzmin;
    // if (tzmax < tmax)
    //     tmax = tzmax;
    // if ((tmin > r.tmax) || (tmax < r.tmin)) return false;
    // if (r.tmin < tmin) r.tmin = tmin;
    // if (r.tmax > tmax) r.tmax = tmax;
    return true;
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
    return (a->getBBox().getLowerLeft()).getX() < (b->getBBox().getLowerLeft()).getX();
}

bool Shape::CompareY(Shape *a, Shape *b) {
    return (a->getBBox().getLowerLeft()).getY() < (b->getBBox().getLowerLeft()).getY();
}

bool Shape::CompareZ(Shape *a, Shape *b) {
    return (a->getBBox().getLowerLeft()).getZ() < (b->getBBox().getLowerLeft()).getZ();
}

bool operator < (Shape &a, Shape &b) {
    return a.getBBox() < b.getBBox();
}