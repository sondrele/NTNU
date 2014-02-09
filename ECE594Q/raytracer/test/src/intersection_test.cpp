#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "intersection.h"

#define SIN_PI_4  0.7071067812f

Shape *s0;
Shape *m0;
Ray *r0;
Ray *r1;

TEST_GROUP(IntersectionTest) {
    void setup() {
        Sphere *sp = new Sphere();
        sp->setOrigin(Vect(0, 0, -5));
        sp->setRadius(1);
        s0 = sp;

        r0 = new Ray(Vect(0, 0, 0), Vect(0, 0, -1));
        r1 = new Ray(Vect(0, SIN_PI_4, 0), Vect(0, 0, -1));

        Mesh *m = new Mesh();
        Triangle *t = new Triangle();
        t->setA(Vertex(2, 0, -3));
        t->setB(Vertex(-2, 0, -3));
        t->setC(Vertex(0, 2, -1));
        m->addTriangle(t);
        m0 = m;
    }
    void teardown() {
        delete s0;
        delete r0;
        delete r1;
        delete m0;
    }
};

TEST(IntersectionTest, ray_creates_intersection) {
    Intersection is = s0->intersects(*r0);
    POINTERS_EQUAL(s0, is.getShape());
    
    DOUBLES_EQUAL(4, is.getIntersectionPoint(), 0.00001);
    Vect v = is.calculateIntersectionPoint();
    DOUBLES_EQUAL(0, v.getX(), 0.00001);
    DOUBLES_EQUAL(0, v.getY(), 0.00001);
    DOUBLES_EQUAL(-4, v.getZ(), 0.00001);

}

TEST(IntersectionTest, has_surface_normal_of_sphere) {
    Intersection is = s0->intersects(*r0);
    POINTERS_EQUAL(s0, is.getShape());
    
    Vect n = is.calculateSurfaceNormal();
    DOUBLES_EQUAL(0, n.getX(), 0.00001);
    DOUBLES_EQUAL(0, n.getY(), 0.00001);
    DOUBLES_EQUAL(1, n.getZ(), 0.00001);
}

TEST(IntersectionTest, ray_creates_intersection_2) {
    Intersection is = s0->intersects(*r1);
    DOUBLES_EQUAL(4.292893, is.getIntersectionPoint(), 0.00001);

    Vect v = is.calculateIntersectionPoint();
    DOUBLES_EQUAL(0, v.getX(), 0.00001);
    DOUBLES_EQUAL(SIN_PI_4, v.getY(), 0.00001);
    DOUBLES_EQUAL(-4.292893, v.getZ(), 0.00001);

    POINTERS_EQUAL(s0, is.getShape());
}

TEST(IntersectionTest, surface_normal_test_2) {
    Intersection is = s0->intersects(*r1);

    Vect n = is.calculateSurfaceNormal();
    DOUBLES_EQUAL(0, n.getX(), 0.00001);
    DOUBLES_EQUAL(0.707106, n.getY(), 0.00001);
    DOUBLES_EQUAL(0.707106, n.getZ(), 0.00001);
}

TEST(IntersectionTest, can_intersect_with_mesh) {
    Intersection is = m0->intersects(*r1);
    DOUBLES_EQUAL(2.292893, is.getIntersectionPoint(), 0.00001);
    
    Vect v = is.calculateIntersectionPoint();
    DOUBLES_EQUAL(0, v.getX(), 0.00001);
    DOUBLES_EQUAL(SIN_PI_4, v.getY(), 0.00001);
    DOUBLES_EQUAL(-2.292893, v.getZ(), 0.00001);
}

TEST(IntersectionTest, can_get_surface_normal_of_mesh) {
    Intersection is = m0->intersects(*r1);

    Shape *s = is.getShape();
    CHECK_EQUAL(TRIANGLE, s->getType());

    Vect n = is.calculateSurfaceNormal();
    DOUBLES_EQUAL(0, n.getX(), 0.00001);
    DOUBLES_EQUAL(-0.707106, n.getY(), 0.00001);
    DOUBLES_EQUAL(0.707106, n.getZ(), 0.00001);
}
