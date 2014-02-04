#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "Matrix.h"
#include <cmath>

TEST_GROUP(Matrix) {
    void setup() {
    }
    void teardown() {
    }
};


TEST(Matrix, matrix_is_2x3) {
    Matrix a(2, 3);
    int rows = a.getRows();
    int cols = a.getCols();

    CHECK_EQUAL(2, rows);
    CHECK_EQUAL(3, cols);
}

TEST(Matrix, shoud_get_cell) {
    Matrix a(2, 4);
    float *r1 = new float[4];
    r1[0] = 0;r1[1] = 1;r1[2] = 2;r1[3] = 3;
    float *r2 = new float[4];
    r2[0] = 5;r2[1] = 6;r2[2] = 7;r2[3] = 8;

    a.setRow(0, r1);
    a.setRow(1, r2);

    DOUBLES_EQUAL(0.0f, a.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(1.0f, a.getCell(0, 1), 0.001);
    DOUBLES_EQUAL(5.0f, a.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(6.0f, a.getCell(1, 1), 0.001);
    DOUBLES_EQUAL(7.0f, a.getCell(1, 2), 0.001);
    DOUBLES_EQUAL(8.0f, a.getCell(1, 3), 0.001);
    delete [] r1;
    delete [] r2;
}

TEST(Matrix, shoud_set_cell) {
    Matrix *a = new Matrix(2, 2);

    a->setCell(0, 0, 0);
    a->setCell(0, 1, 1);
    a->setCell(1, 0, 2);
    a->setCell(1, 1, 3);

    DOUBLES_EQUAL(0.0f, a->getCell(0, 0), 0.001);
    DOUBLES_EQUAL(1.0f, a->getCell(0, 1), 0.001);
    DOUBLES_EQUAL(2.0f, a->getCell(1, 0), 0.001);
    DOUBLES_EQUAL(3.0f, a->getCell(1, 1), 0.001);
    delete a;
}

TEST(Matrix, can_assign) {
    Matrix a(2, 2);
    a.setCell(0, 0, 0);
    a.setCell(0, 1, 1);
    a.setCell(1, 0, 2);
    a.setCell(1, 1, 3);

    Matrix c = a;

    DOUBLES_EQUAL(0.0f, c.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(1.0f, c.getCell(0, 1), 0.001);
    DOUBLES_EQUAL(2.0f, c.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(3.0f, c.getCell(1, 1), 0.001);
}

TEST(Matrix, should_add_linearly) {
    float as[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    Matrix *a = new Matrix(2, 2, as);
    Matrix *b = new Matrix(2, 2, as);

    Matrix c = a[0] + b[0];

    DOUBLES_EQUAL(0.0f,c.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(2.0f,c.getCell(0, 1), 0.001);
    DOUBLES_EQUAL(4.0f,c.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(6.0f,c.getCell(1, 1), 0.001);
    delete a;
    delete b;
}

TEST(Matrix, should_multiply_correctly) {
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

    Matrix c = a[0] * b[0];

    CHECK_EQUAL(2, c.getRows());
    CHECK_EQUAL(2, c.getCols());

    DOUBLES_EQUAL(13.0f, c.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(16.0f, c.getCell(0, 1), 0.001);
    DOUBLES_EQUAL((float)(-3+12+25), c.getCell(1, 0), 0.001);
    DOUBLES_EQUAL((float)(-6+16+30), c.getCell(1, 1), 0.001);
    delete a;
    delete b;
}

TEST(Matrix, can_scale_4x1) {
    Vect v(1, 2, -1);
    Vect::Scale(v, 1, -2, 3);

    DOUBLES_EQUAL(1.0, v.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(-4.0, v.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(-3.0, v.getCell(2, 0), 0.001);
    DOUBLES_EQUAL(1.0, v.getCell(3, 0), 0.001);
}

TEST(Matrix, can_rotate_4x1) {
    Vect vect(1, 2, 3);

    Vect::Rotate(vect, 'x', M_PI);
    DOUBLES_EQUAL(1.0, vect.getX(), 0.001);
    DOUBLES_EQUAL(-2.0, vect.getY(), 0.001);
    DOUBLES_EQUAL(-3.0, vect.getZ(), 0.001);

    Vect v2(1, 2, 3);
    Vect::Rotate(v2, 'y', M_PI * 1.5);
    DOUBLES_EQUAL(-3.0, v2.getX(), 0.001);
    DOUBLES_EQUAL(2.0, v2.getY(), 0.001);
    DOUBLES_EQUAL(1.0, v2.getZ(), 0.001);

    Vect v3(1, 2, 3);
    Vect::Rotate(v3, 'z', M_PI / 2.0);
    DOUBLES_EQUAL(-2.0, v3.getX(), 0.001);
    DOUBLES_EQUAL(1.0, v3.getY(), 0.001);
    DOUBLES_EQUAL(3.0, v3.getZ(), 0.001);
}

TEST(Matrix, can_translate_4x1) {
    Vect vect(1, 2, 3);

    Vect::Translate(vect, -2, 2, 3);

    DOUBLES_EQUAL(-1,vect.getCell(0, 0),  0.001);
    DOUBLES_EQUAL(4, vect.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(6, vect.getCell(2, 0), 0.001);
    DOUBLES_EQUAL(1, vect.getCell(3, 0), 0.001);
}

TEST(Matrix, vector_initializes) {
    Vect v;
    CHECK_EQUAL(4, v.getRows());
    CHECK_EQUAL(1, v.getCols());
    Vect *v1 = new Vect(3, 3, 3);
    DOUBLES_EQUAL(3, v1->getX(), 0.0001);
    delete v1;
}

TEST(Matrix, can_set_coords) {
    Vect v;
    v.setX(1.0);
    v.setY(3.0);
    v.setZ(2.0);

    DOUBLES_EQUAL(1.0, v.getX(), 0.001);
    DOUBLES_EQUAL(3.0, v.getY(), 0.001);
    DOUBLES_EQUAL(2.0, v.getZ(), 0.001);
    DOUBLES_EQUAL(1.0, v.getW(), 0.001);

    DOUBLES_EQUAL(v.getX(), v.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(v.getY(), v.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(v.getZ(), v.getCell(2, 0), 0.001);
    DOUBLES_EQUAL(v.getW(), v.getCell(3, 0), 0.001);
}

TEST(Matrix, can_homogenize) {
    Vect v(10, 10, 10);
    v.homogenize();
    DOUBLES_EQUAL(10, v.getX(), 0.00001);
    DOUBLES_EQUAL(10, v.getY(), 0.00001);
    DOUBLES_EQUAL(10, v.getZ(), 0.00001);
    DOUBLES_EQUAL(1, v.getW(), 0.00001);

    v.setW(10);
    v.homogenize();
    DOUBLES_EQUAL(1, v.getX(), 0.00001);
    DOUBLES_EQUAL(1, v.getY(), 0.00001);
    DOUBLES_EQUAL(1, v.getZ(), 0.00001);
    DOUBLES_EQUAL(1, v.getW(), 0.00001);
}

TEST(Matrix, can_project) {
    float hither = 1;
    float yon = 1000;
    float fov = (float) M_PI / 3.0;
    float aspectRatio = 1;

    Vect v(-10, -10, 100);
    Vect::Project(v, hither, yon, fov, aspectRatio);
    DOUBLES_EQUAL(-17.3, v.getX(), 0.1);
    DOUBLES_EQUAL(-17.3, v.getY(), 0.1);
    v.homogenize();
    DOUBLES_EQUAL(-0.17, v.getX(), 0.01);
    DOUBLES_EQUAL(-0.17, v.getY(), 0.01);
}

TEST(Matrix, should_init) {
    Vect v;
    CHECK_EQUAL(v.getX(), 0);
    CHECK_EQUAL(v.getW(), 1);
}

TEST(Matrix, vector_can_normalize) {
    Vect v(3, 4, 5);
    v.normalize();
    DOUBLES_EQUAL(v.getX(), 0.424264, 0.0001);
    DOUBLES_EQUAL(v.getY(), 0.565685, 0.0001);
    DOUBLES_EQUAL(v.getZ(), 0.707106, 0.0001);
}

TEST(Matrix, cross_product) {
    Vect v1(1, 2, 3);
    Vect v2(3, 4, 5);

    Vect v3 = v1.crossProduct(v2);
    CHECK_EQUAL(-2, v3.getX());
    CHECK_EQUAL(4, v3.getY());
    CHECK_EQUAL(-2, v3.getZ());
}

TEST(Matrix, dot_product) {
    Vect v1(1, 2, 3);
    Vect v2(3, 4, 5);

    float dotP = v1.dotProduct(v2);
    CHECK_EQUAL(26, dotP);
}

TEST(Matrix, linear_mult) {
    Vect v1(-1, 2, 3);

    Vect v2 = v1.linearMult(2);
    CHECK_EQUAL(-2, v2.getX());
    CHECK_EQUAL(4, v2.getY());
    CHECK_EQUAL(6, v2.getZ());
}
