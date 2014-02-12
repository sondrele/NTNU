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
* Material
*********************************/
Material::Material() {
    shininess = 1;
    transparency = 0;
}

bool Material::hasTexture() {
    return false; // TODO: Fiske
}

bool Material::isReflective() {
    return specColor.length() > 0;
}

bool Material::isRefractive() {
    return transparency > 0;
}

/*********************************
* Shape
*********************************/
Shape::Shape() {

}

Shape::~Shape() {
    for (uint i = 0; i < materials.size(); i++) {
        delete materials[i];
    }
}

Material * Shape::getMaterial() {
    return materials[0];
}
