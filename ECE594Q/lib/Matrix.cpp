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

Matrix &Matrix::operator = (Matrix &other) {
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

const Matrix operator +(const Matrix& a, const Matrix& b) {
    if (a.ROWS == b.ROWS and a.COLS == b.COLS) {
        Matrix m(a.ROWS, a.COLS);

        for (int i = 0; i < a.ROWS * a.COLS; i++) {
            m.mat[i] = a.mat[i] + b.mat[i];
        }
        return m;
    }
    throw "AddException: Wrong dimensions\n";
}

const Matrix operator *(const Matrix& a, const Matrix& b) {
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
    s << "Matrix[" << ROWS << "][" << COLS << "]\n";
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            s << '|' << getCell(i, j) << "|";
        }
        s << endl;
    }
    return s.str();
}

ostream& operator <<(ostream& stream, const Matrix &m) {
    stream << m.toString();
    return stream;
}

Vect::Vect(float x, float y, float z) : Matrix(4, 1) {
    setX(x);
    setY(y);
    setZ(z);
    setW(1);
}

Vect::Vect(const Matrix mat) : Matrix(4, 1) {
    setX(mat.getCell(0, 0));
    setY(mat.getCell(1, 0));
    setZ(mat.getCell(2, 0));
    setW(mat.getCell(3, 0));
}

Vect::Vect(const Vect &vect) : Matrix(4, 1) {
    setX(vect.getX());
    setY(vect.getY());
    setZ(vect.getZ());
    setW(vect.getW());
}

void Vect::homogenize() {
    setX(getX() / getW());
    setY(getY() / getW());
    setZ(getZ() / getW());
    setW(getW() / getW());
}

void Vect::Scale(Vect &vect, const float dx, const float dy, const float dz) {
    if (vect.getRows() == 4 and vect.getCols() == 1) {
        Matrix scaleMat(4, 4);
        scaleMat.setCell(0, 0, dx);
        scaleMat.setCell(1, 1, dy);
        scaleMat.setCell(2, 2, dz);
        scaleMat.setCell(3, 3, 1);

        Matrix scaled = scaleMat * vect;
        vect.setX(scaled.getCell(0, 0));
        vect.setY(scaled.getCell(1, 0));
        vect.setZ(scaled.getCell(2, 0));
        vect.setW(scaled.getCell(3, 0));
    } else
        throw "VectScaleException: Wrong dimensions";
}

void Vect::Rotate(Vect &vect, const char axis, const double theta) {
    Matrix rotMat(4, 4);
    float c = cos(M_PI);
    float s = sin(M_PI);
    float sn = -s;

    if (axis == 'x') {
        rotMat.setCell(0, 0, 1);
        rotMat.setCell(1, 1, c);
        rotMat.setCell(1, 2, sn);
        rotMat.setCell(2, 1, s);
        rotMat.setCell(2, 2, c);
    } else if (axis == 'y') {
        rotMat.setCell(0, 0, c);
        rotMat.setCell(0, 2, s);
        rotMat.setCell(1, 1, 1);
        rotMat.setCell(2, 0, sn);
        rotMat.setCell(2, 2, c);
    } else if (axis == 'z') {
        rotMat.setCell(0, 0, c);
        rotMat.setCell(0, 1, sn);
        rotMat.setCell(1, 0, s);
        rotMat.setCell(1, 1, c);
        rotMat.setCell(2, 2, 1);
    }
    rotMat.setCell(3, 3, 1);
    Matrix rotated = rotMat * vect;
    vect.setX(rotated.getCell(0, 0));
    vect.setY(rotated.getCell(1, 0));
    vect.setZ(rotated.getCell(2, 0));
    vect.setW(rotated.getCell(3, 0));
}

void Vect::Translate(Vect &vect, const float dx, const float dy, const float dz) {
    Matrix transMat(4, 4);
    transMat.setCell(0, 0, 1);
    transMat.setCell(0, 3, dx);
    transMat.setCell(1, 1, 1);
    transMat.setCell(1, 3, dy);
    transMat.setCell(2, 2, 1);
    transMat.setCell(2, 3, dz);
    transMat.setCell(3, 3, 1);

    Matrix translated = transMat * vect;
    vect.setX(translated.getCell(0, 0));
    vect.setY(translated.getCell(1, 0));
    vect.setZ(translated.getCell(2, 0));
    vect.setW(translated.getCell(3, 0));
}

void Vect::Project(Vect &vect, float hither, float yon, float fov, float aspectRatio) {
    float a = yon / (yon - hither);
    float b = -yon * hither / (yon - hither);
    Matrix projectMat(4, 4);
    projectMat.setCell(0, 0, 1 / tan(fov / 2.0));
    projectMat.setCell(1, 1, 1 / tan(fov / (2.0 * aspectRatio)));
    projectMat.setCell(2, 2, a);
    projectMat.setCell(2, 3, b);
    projectMat.setCell(3, 2, 1);

    Matrix projected = projectMat * vect;
    vect.setX(projected.getCell(0, 0));
    vect.setY(projected.getCell(1, 0));
    vect.setZ(projected.getCell(2, 0));
    vect.setW(projected.getCell(3, 0));
}
