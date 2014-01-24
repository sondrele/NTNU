#include "Matrix.h"
#include "ctest.h"
#include <assert.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;

void matrix_is_2x3() {
    Matrix a(2, 3);
    int rows = a.getRows();
    int cols = a.getCols();

    ASSERT_EQUAL_INT(rows, 2);
    ASSERT_EQUAL_INT(cols, 3);
}

void can_set_and_get_row() {
    Matrix a(2, 2);
    float *r1 = new float[2];
    r1[0] = 0;
    r1[1] = 1;
    float *r2 = new float[2];
    r2[0] = 2;
    r2[1] = 3;

    a.setRow(0, r1);
    a.setRow(1, r2);

    float *q1 = a.getRow(0);
    float *q2 = a.getRow(1);

    ASSERT_EQUAL_FLOAT(q1[0], 0.0, 0.001);
    ASSERT_EQUAL_FLOAT(q1[1], 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(q2[0], 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(q2[1], 3.0, 0.001);
    delete [] r1;
    delete [] r2;
    delete [] q1;
    delete [] q2;
}

void can_set_and_get_col() {
    Matrix a(2, 3);
    float *q1 = new float[3];
    q1[0] = 0;
    q1[1] = 1;
    q1[2] = 2;
    float *q2 = new float[3];
    q2[0] = 2;
    q2[1] = 3;
    q2[2] = 4;

    a.setCol(0, q1);
    a.setCol(1, q2);

    float *r0 = a.getCol(0); // 0 1 2
    float *r1 = a.getCol(1); // 2 3 4
    float *c1 = a.getRow(0); // 0 2
    float *c2 = a.getRow(1); // 1 3
    float *c3 = a.getRow(2); // 2 4

    ASSERT_EQUAL_FLOAT(r0[0], 0.0, 0.001);
    ASSERT_EQUAL_FLOAT(r0[1], 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(r0[2], 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(r1[0], 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(r1[1], 3.0, 0.001);
    ASSERT_EQUAL_FLOAT(r1[2], 4.0, 0.001);

    ASSERT_EQUAL_FLOAT(c1[0], 0.0, 0.001);
    ASSERT_EQUAL_FLOAT(c1[1], 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(c2[0], 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(c2[1], 3.0, 0.001);
    ASSERT_EQUAL_FLOAT(c3[0], 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(c3[1], 4.0, 0.001);
    delete [] q1;
    delete [] q2;
    delete [] r0;
    delete [] r1;
    delete [] c1;
    delete [] c2;
    delete [] c3;
}

void shoud_get_cell() {
    Matrix a(2, 4);
    float *r1 = new float[4];
    r1[0] = 0;r1[1] = 1;r1[2] = 2;r1[3] = 3;
    float *r2 = new float[4];
    r2[0] = 5;r2[1] = 6;r2[2] = 7;r2[3] = 8;

    a.setRow(0, r1);
    a.setRow(1, r2);

    ASSERT_EQUAL_FLOAT(a.getCell(0, 0), 0.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a.getCell(0, 1), 1.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a.getCell(1, 0), 5.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a.getCell(1, 1), 6.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a.getCell(1, 2), 7.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a.getCell(1, 3), 8.0f, 0.001);
    delete [] r1;
    delete [] r2;
}

void shoud_set_cell() {
    Matrix *a = new Matrix(2, 2);

    a->setCell(0, 0, 0);
    a->setCell(0, 1, 1);
    a->setCell(1, 0, 2);
    a->setCell(1, 1, 3);

    ASSERT_EQUAL_FLOAT(a->getCell(0, 0), 0.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a->getCell(0, 1), 1.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a->getCell(1, 0), 2.0f, 0.001);
    ASSERT_EQUAL_FLOAT(a->getCell(1, 1), 3.0f, 0.001);
    delete a;
}

void can_assign() {
    Matrix a(2, 2);
    a.setCell(0, 0, 0);
    a.setCell(0, 1, 1);
    a.setCell(1, 0, 2);
    a.setCell(1, 1, 3);

    Matrix c = a;

    ASSERT_EQUAL_FLOAT(c.getCell(0, 0), 0.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(0, 1), 1.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(1, 0), 2.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(1, 1), 3.0f, 0.001);
}

void should_add_linearly() {
    float as[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    Matrix *a = new Matrix(2, 2, as);
    Matrix *b = new Matrix(2, 2, as);

    Matrix c = a[0] + b[0];

    ASSERT_EQUAL_FLOAT(c.getCell(0, 0), 0.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(0, 1), 2.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(1, 0), 4.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(1, 1), 6.0f, 0.001);
    delete a;
    delete b;
}

void should_multiply_correctly() {
    float as[6] = {
        0, 1, 2, 
        3, 4, 5
    };
    Matrix *a = new Matrix(2, 3, as);
    float bs[6] = {
        -1, -2, 
        3, 4, 
        5, 6
    };
    Matrix *b = new Matrix(3, 2, bs);

    ASSERT_EQUAL_INT(a->getRows(), b->getCols());
    ASSERT_EQUAL_INT(a->getCols(), b->getRows());

    Matrix c = a[0] * b[0];

    ASSERT_EQUAL_INT(c.getRows(), 2);
    ASSERT_EQUAL_INT(c.getCols(), 2);

    ASSERT_EQUAL_FLOAT(c.getCell(0, 0), 13.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(0, 1), 16.0f, 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(1, 0), (float)(-3+12+25), 0.001);
    ASSERT_EQUAL_FLOAT(c.getCell(1, 1), (float)(-6+16+30), 0.001);
    delete a;
    delete b;
}

void can_scale_4x1() {
    Vect v(1, 2, -1);
    Vect::Scale(v, 1, -2, 3);

    ASSERT_EQUAL_FLOAT(v.getCell(0, 0), 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(v.getCell(1, 0), -4.0, 0.001);
    ASSERT_EQUAL_FLOAT(v.getCell(2, 0), -3.0, 0.001);
    ASSERT_EQUAL_FLOAT(v.getCell(3, 0), 1.0, 0.001);
}

void can_rotate_4x1() {
    Vect vect(1, 2, 3);

    Vect::Rotate(vect, 'x', M_PI);
    ASSERT_EQUAL_FLOAT(vect.getCell(0, 0), 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(vect.getCell(1, 0), -2.0, 0.001);
    ASSERT_EQUAL_FLOAT(vect.getCell(2, 0), -3.0, 0.001);

    Vect v2(1, 2, 3);
    Vect::Rotate(v2, 'y', M_PI * 1.5);
    ASSERT_EQUAL_FLOAT(v2.getCell(0, 0), -1.0, 0.001);
    ASSERT_EQUAL_FLOAT(v2.getCell(1, 0), 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(v2.getCell(2, 0), -3.0, 0.001);

    Vect v3(1, 2, 3);
    Vect::Rotate(v3, 'z', M_PI / 2.0);
    ASSERT_EQUAL_FLOAT(v3.getCell(0, 0), -1.0, 0.001);
    ASSERT_EQUAL_FLOAT(v3.getCell(1, 0), -2.0, 0.001);
    ASSERT_EQUAL_FLOAT(v3.getCell(2, 0), 3.0, 0.001);
}

void can_translate_4x1() {
    Vect vect(1, 2, 3);

    Vect::Translate(vect, -2, 2, 3);

    ASSERT_EQUAL_FLOAT(vect.getCell(0, 0), -1, 0.001);
    ASSERT_EQUAL_FLOAT(vect.getCell(1, 0), 4, 0.001);
    ASSERT_EQUAL_FLOAT(vect.getCell(2, 0), 6, 0.001);
    ASSERT_EQUAL_FLOAT(vect.getCell(3, 0), 1, 0.001);
}

void Matrix_TestSuite() {
    TEST_CASE(matrix_is_2x3);
    TEST_CASE(can_set_and_get_row);
    TEST_CASE(shoud_get_cell);
    TEST_CASE(shoud_set_cell);
    TEST_CASE(can_assign);
    TEST_CASE(should_add_linearly);
    TEST_CASE(should_multiply_correctly);
}

void vector_initializes() {
    Vect v;
    ASSERT_EQUAL_INT(v.getRows(), 4);
    ASSERT_EQUAL_INT(v.getCols(), 1);
    Vect *v1 = new Vect(3, 3, 3);
    ASSERT_EQUAL_FLOAT(v1->getX(), 3, 0.0001);
    delete v1;
}

void can_set_coords() {
    Vect v;
    v.setX(1.0);
    v.setY(3.0);
    v.setZ(2.0);

    ASSERT_EQUAL_FLOAT(v.getX(), 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(v.getY(), 3.0, 0.001);
    ASSERT_EQUAL_FLOAT(v.getZ(), 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(v.getW(), 1.0, 0.001);

    ASSERT_EQUAL_FLOAT(v.getX(), v.getCell(0, 0), 0.001);
    ASSERT_EQUAL_FLOAT(v.getY(), v.getCell(1, 0), 0.001);
    ASSERT_EQUAL_FLOAT(v.getZ(), v.getCell(2, 0), 0.001);
    ASSERT_EQUAL_FLOAT(v.getW(), v.getCell(3, 0), 0.001);
}

void can_homogenize() {
    Vect v(10, 10, 10);
    v.homogenize();
    ASSERT_EQUAL_FLOAT(v.getX(), 10, 0.00001);
    ASSERT_EQUAL_FLOAT(v.getY(), 10, 0.00001);
    ASSERT_EQUAL_FLOAT(v.getZ(), 10, 0.00001);
    ASSERT_EQUAL_FLOAT(v.getW(), 1, 0.00001);

    v.setW(10);
    v.homogenize();
    ASSERT_EQUAL_FLOAT(v.getX(), 1, 0.00001);
    ASSERT_EQUAL_FLOAT(v.getY(), 1, 0.00001);
    ASSERT_EQUAL_FLOAT(v.getZ(), 1, 0.00001);
    ASSERT_EQUAL_FLOAT(v.getW(), 1, 0.00001);
}

void can_project() {
    float hither = 1;
    float yon = 1000;
    float fov = M_PI / 3.0;
    float aspectRatio = 1;

    Vect v(-10, -10, 100);
    Vect::Project(v, hither, yon, fov, aspectRatio);
    ASSERT_EQUAL_FLOAT(v.getX(), -17.3, 0.1);
    ASSERT_EQUAL_FLOAT(v.getY(), -17.3, 0.1);
    v.homogenize();
    ASSERT_EQUAL_FLOAT(v.getX(), -0.17, 0.01);
    ASSERT_EQUAL_FLOAT(v.getY(), -0.17, 0.01);
}

void Vector_testsuite() {
    TEST_CASE(vector_initializes);
    TEST_CASE(can_set_coords);
    TEST_CASE(can_homogenize);
    TEST_CASE(can_scale_4x1);
    TEST_CASE(can_rotate_4x1);
    TEST_CASE(can_translate_4x1);
    TEST_CASE(can_project);
}

int main(int argc, char const *argv[]) {
    try {
        RUN_TEST_SUITE(Matrix_TestSuite);
        RUN_TEST_SUITE(Vector_testsuite);
    } catch (const char *str) {
        printf("%s\n", str);
        return 0;
    }
}
