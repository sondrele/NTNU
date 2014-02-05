#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "Matrix.h"
#include <cmath>

TEST_GROUP(MatrixTest) {
    void setup() {
    }
    void teardown() {
    }
};


TEST(MatrixTest, matrix_is_2x3) {
    Matrix a(2, 3);
    int rows = a.getRows();
    int cols = a.getCols();

    CHECK_EQUAL(2, rows);
    CHECK_EQUAL(3, cols);
}

TEST(MatrixTest, shoud_get_cell) {
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

TEST(MatrixTest, shoud_set_cell) {
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

TEST(MatrixTest, can_assign) {
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

TEST(MatrixTest, should_add_linearly) {
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

TEST(MatrixTest, should_multiply_correctly) {
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
