#include "Matrix.h"

using namespace std;

Matrix::Matrix(const Matrix &m) {
    ROWS = m.ROWS;
    COLS = m.COLS;

    mat = new float[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) {
        mat[i] = m.mat[i];
    }
}

Matrix::Matrix(int rows, int cols) {
    ROWS = rows;
    COLS = cols;

    mat = new float[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) {
        mat[i] = 0;
    }
}

Matrix::Matrix(int rows, int cols, float *vals) {
    ROWS = rows;
    COLS = cols;

    mat = new float[ROWS * COLS];

    for (int i = 0; i < ROWS * COLS; i++) {
        mat[i] = vals[i];
    }
}

Matrix::~Matrix() {
    delete [] mat;
}

int Matrix::getRows() {
    return ROWS;
}

int Matrix::getCols() {
    return COLS;
}

void Matrix::setRow(int row, float* vals) {
    for (int i = 0; i < COLS; i++) {
        mat[row * COLS + i] = vals[i];
    }
}

float* Matrix::getRow(int row) {
    float *vals = new float[COLS];

    for (int i = 0; i < COLS; i++) {
        vals[i] = mat[row * COLS + i];
    }

    return vals;
}

void Matrix::setCol(int col, float* vals) {
    for (int i = 0; i < ROWS; i++) {
        mat[COLS * i + col] = vals[i];
    }
}

float* Matrix::getCol(int col) {
    float *vals = new float[ROWS];

    for (int i = 0; i < ROWS; i++) {
        vals[i] = mat[COLS * i + col];
    }

    return vals;
}

void Matrix::setCell(int row, int col, float val) {
    if (row >= 0 and row < ROWS and col >= 0 and col < COLS)
        mat[row * COLS + col] = val;
    else
        throw "SetCellException: Out of bounds\n";
}


float Matrix::getCell(int row, int col) const {
    if (row >= 0 and row < ROWS and col >= 0 and col < COLS)
        return mat[row * COLS + col];

    throw "GetCellException: Out of bounds\n";
}

Matrix &Matrix::operator = (const Matrix &other) {
    ROWS = other.ROWS;
    COLS = other.COLS;

    float *newMat = new float[ROWS * COLS];
    delete [] mat;
    mat = newMat;

    for (int i = 0; i < ROWS * COLS; i++) {
        mat[i] = other.mat[i];
    }

    return *this;
}

const Matrix operator +(const Matrix &a, const Matrix &b) {
    if (a.ROWS == b.ROWS and a.COLS == b.COLS) {
        Matrix m(a.ROWS, a.COLS);

        for (int i = 0; i < a.ROWS * a.COLS; i++) {
            m.mat[i] = a.mat[i] + b.mat[i];
        }
        return m;
    }
    throw "AddException: Wrong dimensions\n";
}

const Matrix operator -(const Matrix &a, const Matrix &b) {
    if (a.ROWS == b.ROWS and a.COLS == b.COLS) {
        Matrix m(a.ROWS, a.COLS);

        for (int i = 0; i < a.ROWS * a.COLS; i++) {
            m.mat[i] = a.mat[i] - b.mat[i];
        }
        return m;
    }
    throw "AddException: Wrong dimensions\n";
}

const Matrix operator *(const Matrix &a, const Matrix &b) {
    if (a.COLS == b.ROWS) {
        Matrix m(a.ROWS, b.COLS);

        for (int i = 0; i < a.ROWS; i++) {
            for (int j = 0; j < b.COLS; j++) {
                float mult = 0;
                for (int k = 0; k < a.COLS; k++) {
                    mult += a.getCell(i, k) * b.getCell(k, j);
                }
                m.setCell(i, j, mult);
            }
        }
        return m;
    }
    throw "MultException: Wrong dimensions\n";
}

string Matrix::toString() const {
    stringstream s;
    // s << "Matrix[" << ROWS << "][" << COLS << "]\n";
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            s << '|' << getCell(i, j) << "|";
        }
        // s << endl;
    }
    return s.str();
}

ostream& operator <<(ostream& stream, const Matrix &m) {
    stream << m.toString();
    return stream;
}
