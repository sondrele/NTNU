#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "Matrix.h"
#include <cmath>

TEST_GROUP(VectTest) {
    void setup() {
    }
    void teardown() {
    }
};

TEST(VectTest, vector_initializes) {
    Vect v;
    CHECK_EQUAL(3, v.getRows());
    CHECK_EQUAL(1, v.getCols());
    Vect *v1 = new Vect(3, 3, 3);
    DOUBLES_EQUAL(3, v1->getX(), 0.0001);
    delete v1;
}

TEST(VectTest, vector_can_normalize) {
    Vect v(3, 4, 5);
    v.normalize();
    DOUBLES_EQUAL(v.getX(), 0.424264, 0.0001);
    DOUBLES_EQUAL(v.getY(), 0.565685, 0.0001);
    DOUBLES_EQUAL(v.getZ(), 0.707106, 0.0001);
}

TEST(VectTest, cross_product) {
    Vect v1(1, 2, 3);
    Vect v2(3, 4, 5);

    Vect v3 = v1.crossProduct(v2);
    CHECK_EQUAL(-2, v3.getX());
    CHECK_EQUAL(4, v3.getY());
    CHECK_EQUAL(-2, v3.getZ());
}

TEST(VectTest, dot_product) {
    Vect v1(1, 2, 3);
    Vect v2(3, 4, 5);

    float dotP = v1.dotProduct(v2);
    CHECK_EQUAL(26, dotP);
}

TEST(VectTest, linear_mult) {
    Vect v1(-1, 2, 3);

    Vect v2 = v1.linearMult(2);
    CHECK_EQUAL(-2, v2.getX());
    CHECK_EQUAL(4, v2.getY());
    CHECK_EQUAL(6, v2.getZ());
}
