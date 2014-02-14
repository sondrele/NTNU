#include "rayscene.h"

/*********************************
* SColor
*********************************/
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
    if (r < 0)
        r = 0;
    setX(r);
}

void SColor::G(float g) {
    if (g > 1.0f)
        g = 1;
    if (g < 0)
        g = 0;
    setY(g);
}

void SColor::B(float b) {
    if (b > 1.0f)
        b = 1;
    if (b < 0)
        b = 0;
    setZ(b);
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
