#include "rayscene.h"

/*********************************
* BBox
*********************************/
BBox::BBox() {

}

bool operator < (const BBox &a, const BBox &b) {
    return a.getLowerLeft() < b.getLowerLeft();
}

bool BBox::intersects(Ray r) {
    Vect invDir = r.getDirection().invert();
    Vect origin = r.getOrigin();
    double tx1 = (lowerLeft.getX() - origin.getX()) * invDir.getX();
    double tx2 = (upperRight.getX() - origin.getX()) * invDir.getX();

    double tmin = min(tx1, tx2);
    double tmax = max(tx1, tx2);

    double ty1 = (lowerLeft.getY() - origin.getY()) * invDir.getY();
    double ty2 = (upperRight.getY() - origin.getY()) * invDir.getY();

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    return tmax >= tmin && tmax >= 0;
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

bool operator < (Shape &a, Shape &b) {
    return a.getBBox() < b.getBBox();
}
