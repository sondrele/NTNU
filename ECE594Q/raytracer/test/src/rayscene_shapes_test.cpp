#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene_shapes.h"

TEST_GROUP(RaySceneShapesTest) {
    void setup() {

    }
    void teardown() {

    }
};

TEST(RaySceneShapesTest, sphere_intesects) {
    Sphere s;
    s.setOrigin(Vect(0, 0, 1));
    s.setRadius(1);

    float t = -10;

    Ray r0(Vect(0, 0, -1), Vect(-1, 0, 1));
    CHECK_FALSE(s.intersects(r0, t));
    DOUBLES_EQUAL(-10, t, 0.0000001);

    Ray r1(Vect(0, 0, -1), Vect(-0.25f, 0, 1));
    CHECK(s.intersects(r1, t));

    Ray r2(Vect(0, 0, -1), Vect(0, 1, 1));
    CHECK_FALSE(s.intersects(r2, t));

    Ray r3(Vect(0, 0, -1), Vect(0, 0.25, 1));
    CHECK(s.intersects(r3, t));

    Ray r4(Vect(0, 0, -1), Vect(0, 0, 1));
    CHECK(s.intersects(r4, t));
    DOUBLES_EQUAL(1, t, 0.000001);
}

TEST(RaySceneShapesTest, triangle_intersects) {
    Triangle t0;
    t0.setA(Vect(0, 0, 1));
    t0.setB(Vect(-1, -1, 2));
    t0.setC(Vect(1, -1, 2));

    float t = -10;

    Ray r0(Vect(0, 0, 0), Vect(0, -0.25, 1));
    CHECK(t0.intersects(r0, t));

    Ray ra(Vect(0, 0, 0), Vect(0, -0.5, 1));
    CHECK(t0.intersects(ra, t));
    DOUBLES_EQUAL(2, t, 0.00001);

    Ray r1(Vect(0, 0, 0), Vect(0, -1, 2));
    CHECK(t0.intersects(r1, t));
    DOUBLES_EQUAL(1, t, 0.00001);

    Ray r2(Vect(0, 0, 0), Vect(0, 0.1f, 2));
    CHECK_FALSE(t0.intersects(r2, t));

    Ray r3(Vect(0, 0, 0), Vect(0.25, -0.1f, 2));
    CHECK_FALSE(t0.intersects(r3, t));
}

TEST(RaySceneShapesTest, mesh_intesects) {
    Triangle t0;
    t0.setA(Vect(0, 0, 1));
    t0.setB(Vect(-1, -1, 2));
    t0.setC(Vect(1, -1, 2));

    Triangle t1;
    t1.setA(Vect(0, 0, 1));
    t1.setB(Vect(-1, 1, 2));
    t1.setC(Vect(1, 1, 2));

    Mesh m;
    m.addTriangle(t0);
    m.addTriangle(t1);

    float t = -10;

    Ray r0(Vect(0, 0, 0), Vect(0, -0.25, 1));
    CHECK(m.intersects(r0, t));

    Ray r1(Vect(0, 0, 0), Vect(1, 1, 2));
    CHECK(m.intersects(r1, t));

    Ray r2(Vect(0, 0, 0), Vect(1, 1, -2));
    CHECK_FALSE(m.intersects(r2, t));
}

