#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene_shapes.h"

Triangle *t;

TEST_GROUP(RaySceneShapesTest) {
    void setup() {
        t = new Triangle();
        Vertex v0(-1, 0, 0);
        Vertex v1(1, 0, 0);
        Vertex v2(0, 0, -1);
        v0.setSurfaceNormal(Vect(0, 1, 0));
        v1.setSurfaceNormal(Vect(0, 1, 0));
        v2.setSurfaceNormal(Vect(0, 0, -1));
        t->setA(v0); t->setB(v1); t->setC(v2);
    }
    void teardown() {
        delete t;
    }
};

TEST(RaySceneShapesTest, ray_can_intersect_with_triangle) {
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

TEST(RaySceneShapesTest, ray_can_intersect_with_mesh) {
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

TEST(RaySceneShapesTest, triangle_has_per_vertex_normals) {
    Vect n0 = t->getA().getSurfaceNormal();
    Vect n1 = t->getB().getSurfaceNormal();
    Vect n2 = t->getC().getSurfaceNormal();
    CHECK_EQUAL(1, n0.getY());
    CHECK_EQUAL(1, n1.getY());
    CHECK_EQUAL(-1, n2.getZ());
}

TEST(RaySceneShapesTest, can_get_interpolation_of_triangle_surface) {
    Ray r(Vect(0, 1, -1), Vect(0, -1, 0));
    Intersection in = t->intersects(r);
    CHECK(in.hasIntersected());

    Vect n = t->interPolatedNormal(Vect(0, 0, -1));
    CHECK_EQUAL(0, n.getX());
    CHECK_EQUAL(0, n.getY());
    CHECK_EQUAL(-1, n.getZ());

    n = t->interPolatedNormal(Vect(-1, 0, 0));
    CHECK_EQUAL(0, n.getX());
    CHECK_EQUAL(1, n.getY());
    CHECK_EQUAL(0, n.getZ());

    n = t->interPolatedNormal(Vect(0, 0, -0.5f));
    CHECK_EQUAL(0, n.getX());
    DOUBLES_EQUAL(0.707101f, n.getY(), 0.00001);
    DOUBLES_EQUAL(-0.707101f, n.getZ(), 0.00001);
}

TEST(RaySceneShapesTest, triangle_has_area) {
    Triangle t0;
    t0.setA(Vertex(0, 0, -4));
    t0.setB(Vertex(2, 2, -4));
    t0.setC(Vertex(2, 0, -4));
    float area = t0.getArea();
    DOUBLES_EQUAL(2, area, 0.00001);
}

TEST(RaySceneShapesTest, can_get_long_and_lat_from_sphere) {
    Sphere s;
    s.setOrigin(Vect(0, -1, -1));
    s.setRadius(1);

    // Point_2D pt = s.getLongAndLat(Vect(0, 0, -1));
    // CHECK_EQUAL(0.25, pt.x);
    // CHECK_EQUAL(1, pt.y);
    // TODO: fiks
    // pt = s.getLongAndLat(Vect(0, -0.5f, 0));
    // DOUBLES_EQUAL(0, pt.x, 0.00001);
    // DOUBLES_EQUAL(0.5f, pt.y, 0.00001);
}
