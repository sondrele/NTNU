#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene_shapes.h"

TEST_GROUP(RaySceneShapesTest) {
    void setup() {

    }
    void teardown() {

    }
};

TEST(RaySceneShapesTest, triangle_intersects) {
    Triangle t0;
    t0.setA(Vertex(0, 0, 1));
    t0.setB(Vertex(-1, -1, 2));
    t0.setC(Vertex(1, -1, 2));

    Intersection is;
    Ray r0(Vect(0, 0, 0), Vect(0, -0.25, 1));
    is = t0.intersects(r0);
    CHECK(is.hasIntersected());

    Ray r1(Vect(0, 0, 0), Vect(0, -0.5, 2));
    is = t0.intersects(r1);
    CHECK(is.hasIntersected());
    DOUBLES_EQUAL(1.37437, is.getIntersectionPoint(), 0.00001);

    Ray r2(Vect(0, 0, 0), Vect(0, 0.1f, 2));
    is = t0.intersects(r2);
    CHECK_FALSE(is.hasIntersected());

    Ray r3(Vect(0, 0, 0), Vect(0.25, -0.1f, 2));
    is = t0.intersects(r3);
    CHECK_FALSE(is.hasIntersected());
}

TEST(RaySceneShapesTest, mesh_intesects) {
    Material *m0 = new Material();
    Triangle *t0 = new Triangle();
    t0->addMaterial(m0);
    t0->setA(Vertex(0, 0, 1));
    t0->setB(Vertex(-1, -1, 2));
    t0->setC(Vertex(1, -1, 2));

    Material *m1 = new Material();
    Triangle *t1 = new Triangle();
    t1->addMaterial(m1);
    t1->setA(Vertex(0, 0, 1));
    t1->setB(Vertex(-1, 1, 2));
    t1->setC(Vertex(1, 1, 2));

    Mesh m;
    m.addTriangle(t0);
    m.addTriangle(t1);

    Ray r0(Vect(0, 0, 0), Vect(0, -0.25, 1));
    Intersection is = m.intersects(r0);
    CHECK(is.hasIntersected());

    Ray r1(Vect(0, 0, 0), Vect(1, 1, 2));
    is = m.intersects(r1);
    CHECK(is.hasIntersected());

    Ray r2(Vect(0, 0, 0), Vect(1, 1, -2));
    is = m.intersects(r2);
    CHECK_FALSE(is.hasIntersected());

}

