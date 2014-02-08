#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene.h"
#include "rayscene_factory.h"

Shape *sphere0;
Shape *sphere1;

TEST_GROUP(RaySceneTest) {
    void setup() {
        Sphere *s = new Sphere();
        s->setRadius(1);
        s->setOrigin(Vect(0, 0, -2));
        sphere0 = s;

        s = new Sphere();
        s->setRadius(1);
        s->setOrigin(Vect(0, -3, 0));
        sphere1 = s;
    }

    void teardown() {
        // delete sphere0;
        // delete sphere1;
    }
};

TEST(RaySceneTest, can_add_shapes) {
    RayScene s;
    s.addShape(sphere0);
    s.addShape(sphere1);

    Shape *s1 = s.getShape(0);
    CHECK(sphere0 == s1);

    Sphere *sp = (Sphere *) s1;
    CHECK_EQUAL(1, sp->getRadius());
}

TEST(RaySceneTest, ray_intersects_with_shape) {
    RayScene s;
    s.addShape(sphere0);
    s.addShape(sphere1);
    Ray r(Vect(0, 0, 0), Vect(0, 0, -1));

    Intersection is = s.calculateRayIntersection(r);
    CHECK(is.hasIntersected());
    DOUBLES_EQUAL(1.0, is.getIntersectionPoint(), 0.000001);
    POINTERS_EQUAL(sphere0, is.getShape());

    Ray r0(Vect(0, 0, 0), Vect(0, -1, 0));
    is = s.calculateRayIntersection(r0);
    CHECK(is.hasIntersected());
    DOUBLES_EQUAL(2.0, is.getIntersectionPoint(), 0.000001);
}

