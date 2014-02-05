#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "Matrix.h"
#include <cmath>

TEST_GROUP(HomogenousVectTest) {
    void setup() {
    }
    void teardown() {
    }
};

TEST(HomogenousVectTest, should_init) {
    Vect_h v;
    CHECK_EQUAL(v.getX(), 0);
    CHECK_EQUAL(v.getW(), 1);
}

TEST(HomogenousVectTest, can_scale_4x1) {
    Vect_h v(1, 2, -1);
    Vect_h::Scale(v, 1, -2, 3);

    DOUBLES_EQUAL(1.0, v.getCell(0, 0), 0.001);
    DOUBLES_EQUAL(-4.0, v.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(-3.0, v.getCell(2, 0), 0.001);
    DOUBLES_EQUAL(1.0, v.getCell(3, 0), 0.001);
}

TEST(HomogenousVectTest, can_set_coords) {
    Vect_h v;
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
TEST(HomogenousVectTest, can_rotate_4x1) {
    Vect_h vect(1, 2, 3);

    Vect_h::Rotate(vect, 'x', M_PI);
    DOUBLES_EQUAL(1.0, vect.getX(), 0.001);
    DOUBLES_EQUAL(-2.0, vect.getY(), 0.001);
    DOUBLES_EQUAL(-3.0, vect.getZ(), 0.001);

    Vect_h v2(1, 2, 3);
    Vect_h::Rotate(v2, 'y', M_PI * 1.5);
    DOUBLES_EQUAL(-3.0, v2.getX(), 0.001);
    DOUBLES_EQUAL(2.0, v2.getY(), 0.001);
    DOUBLES_EQUAL(1.0, v2.getZ(), 0.001);

    Vect_h v3(1, 2, 3);
    Vect_h::Rotate(v3, 'z', M_PI / 2.0);
    DOUBLES_EQUAL(-2.0, v3.getX(), 0.001);
    DOUBLES_EQUAL(1.0, v3.getY(), 0.001);
    DOUBLES_EQUAL(3.0, v3.getZ(), 0.001);
}

TEST(HomogenousVectTest, can_translate_4x1) {
    Vect_h vect(1, 2, 3);

    Vect_h::Translate(vect, -2, 2, 3);

    DOUBLES_EQUAL(-1,vect.getCell(0, 0),  0.001);
    DOUBLES_EQUAL(4, vect.getCell(1, 0), 0.001);
    DOUBLES_EQUAL(6, vect.getCell(2, 0), 0.001);
    DOUBLES_EQUAL(1, vect.getCell(3, 0), 0.001);
}

TEST(HomogenousVectTest, vector_initializes) {
    Vect_h v;
    CHECK_EQUAL(4, v.getRows());
    CHECK_EQUAL(1, v.getCols());
    Vect *v1 = new Vect(3, 3, 3);
    DOUBLES_EQUAL(3, v1->getX(), 0.0001);
    delete v1;
}

TEST(HomogenousVectTest, can_project) {
    float hither = 1;
    float yon = 1000;
    float fov = (float) M_PI / 3.0;
    float aspectRatio = 1;

    Vect_h v(-10, -10, 100);
    Vect_h::Project(v, hither, yon, fov, aspectRatio);
    DOUBLES_EQUAL(-17.3, v.getX(), 0.1);
    DOUBLES_EQUAL(-17.3, v.getY(), 0.1);
    v.homogenize();
    DOUBLES_EQUAL(-0.17, v.getX(), 0.01);
    DOUBLES_EQUAL(-0.17, v.getY(), 0.01);
}

TEST(HomogenousVectTest, can_homogenize) {
    Vect_h v(10, 10, 10);
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
