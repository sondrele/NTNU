#include "Matrix.h"

Vect::Vect() {
    x = 0;
    y = 0;
    z = 0;
}

Vect::Vect(float xval, float yval, float zval) {
    x = xval;
    y = yval;
    z = zval;
}

Vect::Vect(const Vect &vect) {
    x = vect.x;
    y = vect.y;
    z = vect.z;
}

bool Vect::equal(Vect other) {
    return (x == other.x) && (y == other.y)
        && (z == other.z);
}

float Vect::length() {
    return sqrt((x * x) + (y * y) + (z * z));
}

void Vect::normalize() {
    float absVal = length();
    if (absVal != 0) {
        x = x / absVal;
        y = y / absVal;
        z = z / absVal;
    }
}

Vect Vect::crossProduct(Vect other) {
    Vect v;
    v.x = y * other.z - z * other.y;
    v.y = z * other.x - x * other.z;
    v.z = x * other.y - y * other.x;
    return v;
}

float Vect::dotProduct(Vect other) {
    float f = x * other.x + y * other.y + z * other.z;
    return f;
}

Vect Vect::linearMult(float f) const{
    Vect v;
    v.x = x * f;
    v.y = y * f;
    v.z = z * f;

    return v;
}

Vect Vect::linearMult(Vect other) {
    Vect v;
    v.x = x * other.x;
    v.y = y * other.y;
    v.z = z * other.z;

    return v;
}

Vect Vect::invert() {
    return linearMult(-1);
}

float Vect::euclideanDistance(Vect other) {
    float a = x - other.x;
    float b = y - other.y;
    float c = z - other.z;
    return sqrt(a * a + b * b + c * c);
}

float Vect::radians(Vect other) {
    Vect v = *this;
    v.normalize();
    other.normalize();
    return acos(v.dotProduct(other));
}

bool operator < (const Vect &a, const Vect &b) {
    if (a.getX() > b.getX())
        return false;
    if (a.getX() < b.getX())
        return true;
    else if (a.getY() < b.getY())
        return true;
    else if (a.getY() == b.getY() && a.getZ() < b.getZ())
        return true;
    return false;
}


Vect &Vect::operator = (const Vect &other) {
    x = other.x;
    y = other.y;
    z = other.z;

    return *this;
}

const Vect operator +(const Vect &a, const Vect &b) {
    Vect v(a.x + b.x, a.y + b.y, a.z + b.z);
    return v;
}

const Vect operator +(const Vect &a, const float f) {
    Vect v;
    v.x = a.x + f;
    v.y = a.y + f;
    v.z = a.z + f;
    return v;
}

const Vect operator +(const float f, const Vect &a) {
    Vect v;
    v.x = f + a.x;
    v.y = f + a.y;
    v.z = f + a.z;
    return v;
}


const Vect operator -(const Vect &a, const Vect &b) {
    Vect v(a.x - b.x, a.y - b.y, a.z - b.z);
    return v;
}

const Vect operator -(const Vect &a, const float f) {
    Vect v;
    v.x = a.x - f;
    v.y = a.y - f;
    v.z = a.z - f;
    return v;
}

const Vect operator -(const float f, const Vect &a) {
    Vect v;
    v.x = f - a.x;
    v.y = f - a.y;
    v.z = f - a.z;
    return v;
}

const Vect operator *(const Vect &a, const Vect &b) {
    Vect v;
    v.x = a.x * b.x;
    v.y = a.y * b.y;
    v.z = a.z * b.z;
    return v;
}

const Vect operator *(const Vect &a, const float f) {
    Vect v;
    v.x = a.x * f;
    v.y = a.y * f;
    v.z = a.z * f;
    return v;
}

const Vect operator *(const float f, const Vect &a) {
    return a * f;
}

string Vect::toString() const {
    stringstream s;
    s << '(' << x << ", " << y << ", " << z << ")";
    return s.str();
}

ostream& operator <<(ostream& stream, const Vect &m) {
    stream << m.toString();
    return stream;
}
