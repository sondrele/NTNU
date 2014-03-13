#include "Matrix.h"

Vect_h::Vect_h() : Matrix(4, 1) {
    setX(0);
    setY(0);
    setZ(0);
    setW(1);
}

Vect_h::Vect_h(float x, float y, float z) : Matrix(4, 1) {
    setX(x);
    setY(y);
    setZ(z);
    setW(1);
}

Vect_h::Vect_h(const Matrix m) : Matrix(4, 1) {
    setX(m.getCell(0, 0));
    setY(m.getCell(1, 0));
    setZ(m.getCell(2, 0));
    setW(m.getCell(3, 0));
}

Vect_h::Vect_h(const Vect_h &vect) : Matrix(4, 1) {
    setX(vect.getX());
    setY(vect.getY());
    setZ(vect.getZ());
    setW(vect.getW());
}

void Vect_h::homogenize() {
    setX(getX() / getW());
    setY(getY() / getW());
    setZ(getZ() / getW());
    setW(getW() / getW());
}

void Vect_h::Scale(Vect_h &vect, const float dx, const float dy, const float dz) {
    if (vect.getRows() == 4 && vect.getCols() == 1) {
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

void Vect_h::Rotate(Vect_h &vect, const char axis, const double theta) {
    Matrix rotMat(4, 4);
    float c = (float) cos(theta);
    float s = (float) sin(theta);
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

void Vect_h::Translate(Vect_h &vect, const float dx, const float dy, const float dz) {
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

void Vect_h::Project(Vect_h &vect, float hither, float yon, float fov, float aspectRatio) {
    float a = yon / (yon - hither);
    float b = -yon * hither / (yon - hither);
    Matrix projectMat(4, 4);
    projectMat.setCell(0, 0, 1 / (float) (tan(fov / 2.0)));
    projectMat.setCell(1, 1, 1 / (float) (tan(fov / (2.0 * aspectRatio))));
    projectMat.setCell(2, 2, a);
    projectMat.setCell(2, 3, b);
    projectMat.setCell(3, 2, 1);

    Matrix projected = projectMat * vect;
    vect.setX(projected.getCell(0, 0));
    vect.setY(projected.getCell(1, 0));
    vect.setZ(projected.getCell(2, 0));
    vect.setW(projected.getCell(3, 0));
}
