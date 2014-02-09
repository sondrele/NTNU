#include "Matrix.h"

Vect::Vect() : Matrix(3, 1) {
    setX(0);
    setY(0);
    setZ(0);
}

Vect::Vect(float x, float y, float z) : Matrix(3, 1) {
    setX(x);
    setY(y);
    setZ(z);
}

Vect::Vect(const Matrix m) : Matrix(3, 1) {
    setX(m.getCell(0, 0));
    setY(m.getCell(1, 0));
    setZ(m.getCell(2, 0));
}

Vect::Vect(const Vect &vect) : Matrix(3, 1) {
    setX(vect.getX());
    setY(vect.getY());
    setZ(vect.getZ());
}

bool Vect::equal(Vect other) {
    return (getX() == other.getX()) && (getY() == other.getY())
        && (getZ() == other.getZ());
}

void Vect::normalize() {
    float absVal = sqrt((getX() * getX()) + (getY() * getY()) + (getZ() * getZ()));
    if (absVal != 0) {
        setX(getX() / absVal);
        setY(getY() / absVal);
        setZ(getZ() / absVal);
    }
}

Vect Vect::crossProduct(Vect other) {
    Vect v;
    v.setX(getY() * other.getZ() - getZ() * other.getY());
    v.setY(getZ() * other.getX() - getX() * other.getZ());
    v.setZ(getX() * other.getY() - getY() * other.getX());
    return v;
}

float Vect::dotProduct(Vect other) {
    float f = getX() * other.getX() + getY() * other.getY() + getZ() * other.getZ();
    return f;
}

Vect Vect::linearMult(float f) {
    Vect v;
    v.setX(getX() * f);
    v.setY(getY() * f);
    v.setZ(getZ() * f);

    return v;
}

Vect Vect::linearMult(Vect other) {
    Vect v;
    v.setX(getX() * other.getX());
    v.setY(getY() * other.getY());
    v.setZ(getZ() * other.getZ());

    return v;
}

float Vect::euclideanDistance(Vect other) {
    float a = getX() - other.getX();
    float b = getY() - other.getY();
    float c = getZ() - other.getZ();
    return sqrt(a * a + b * b + c * c);
}
