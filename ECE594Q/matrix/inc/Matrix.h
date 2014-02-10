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
    
    float getX() const { return getCell(0, 0); }
    float getY() const { return getCell(1, 0); }
    float getZ() const { return getCell(2, 0); }
    void setX(float val) { setCell(0, 0, val); }
    void setY(float val) { setCell(1, 0, val); }
    void setZ(float val) { setCell(2, 0, val); }

    string toString() const;
    friend ostream& operator <<(ostream&, const Matrix&);

    Matrix &operator = (const Matrix &);
    const friend Matrix operator +(const Matrix &, const Matrix &);
    const friend Matrix operator -(const Matrix &, const Matrix &);
    const friend Matrix operator *(const Matrix &, const Matrix &);
};

class Vect : public Matrix {
public:
    Vect();
    Vect(float, float, float);
    Vect(const Matrix);
    Vect(const Vect &);

    bool equal(Vect other);

    void normalize();
    Vect crossProduct(Vect v);
    float dotProduct(Vect v);
    Vect linearMult(float);
    Vect linearMult(Vect);
    Vect invert();
    float euclideanDistance(Vect);
};

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
