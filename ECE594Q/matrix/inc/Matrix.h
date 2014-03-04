#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cmath>

using namespace std;

class Matrix {
private:
    int ROWS;
    int COLS;
    float *mat;

public:
    Matrix(const Matrix &);
    Matrix(int, int);
    Matrix(int, int, float*);
    ~Matrix();
    
    int getRows();
    int getCols();
    void setRow(int, float*);
    float* getRow(int);
    void setCol(int, float*);
    float* getCol(int);
    void setCell(int, int, float);
    float getCell(int, int) const;
    
    float getX() const { return mat[0]; }
    float getY() const { return mat[1]; }
    float getZ() const { return mat[2]; }
    void setX(float val) { mat[0] = val; }
    void setY(float val) { mat[1] = val; }
    void setZ(float val) { mat[2] = val; }

    string toString() const;
    friend ostream& operator <<(ostream&, const Matrix&);

    Matrix &operator = (const Matrix &);
    const friend Matrix operator +(const Matrix &, const Matrix &);
    const friend Matrix operator -(const Matrix &, const Matrix &);
    const friend Matrix operator *(const Matrix &, const Matrix &);
};

class Vect {
protected:
    float x;
    float y;
    float z;

public:
    Vect();
    Vect(float, float, float);
    Vect(const Vect &);

    float getX() const { return x; }
    float getY() const { return y; }
    float getZ() const { return z; }
    void setX(float val) { x = val; }
    void setY(float val) { y = val; }
    void setZ(float val) { z = val; }

    bool equal(Vect other);

    float length();
    void normalize();
    Vect crossProduct(Vect v);
    float dotProduct(Vect v);
    Vect linearMult(float) const;
    Vect linearMult(Vect);
    Vect invert();
    float euclideanDistance(Vect);
    float radians(Vect);

    string toString() const;
    friend ostream& operator <<(ostream&, const Vect&);

    Vect &operator = (const Vect &);
    const friend Vect operator +(const Vect &, const Vect &);
    const friend Vect operator +(const Vect &, const float);
    const friend Vect operator +(const float, const Vect &);
    const friend Vect operator -(const Vect &, const Vect &);
    const friend Vect operator -(const Vect &, const float);
    const friend Vect operator -(const float, const Vect &);
    const friend Vect operator *(const Vect &, const Vect &);
    const friend Vect operator *(const Vect &, const float);
    const friend Vect operator *(const float, const Vect &);
};

bool operator < (const Vect&, const Vect&);

class Vect_h : public Matrix {
public:
    Vect_h();
    Vect_h(float, float, float);
    Vect_h(const Matrix);
    Vect_h(const Vect_h &);

    void setW(float val) { setCell(3, 0, val); }
    float getW() const { return getCell(3, 0); }

    void homogenize();

    static void Scale(Vect_h &, const float, const float, const float);
    static void Rotate(Vect_h &, const char, const double);
    static void Translate(Vect_h &, const float, const float, const float);
    static void Project(Vect_h &, float, float, float, float);
};

#endif // _MATRIX_H_
