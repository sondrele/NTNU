#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "raytracer.h"


TEST_GROUP(RayTracer) {
    void setup() {

        // Sphere *s = new Sphere();
        // s->setRadius(1);
        // s->setOrigin(Vect(0, 0, -2));
        // sphere0 = s;
    }
    void teardown() {
        // delete sphere0;
    }
};

TEST(RayTracer, should_init) {
    RayTracer rt(20, 10);
    CHECK_EQUAL(20, rt.getWidth());
    CHECK_EQUAL(10, rt.getHeight());
}

TEST(RayTracer, has_camera) {
    RayTracer rt(10, 10);
    Camera camera = rt.getCamera();
    CHECK_EQUAL(0, camera.getX());
    CHECK_EQUAL(0, camera.getY());
    CHECK_EQUAL(0, camera.getZ());
}

TEST(RayTracer, can_add_camera) {
    Camera cam;
    cam.setPos(Vect(10, 10, 10));
    cam.setViewDir(Vect(-1, -1, -1));
    cam.setOrthoUp(Vect(1, 1, 0));
    cam.setVerticalFOV((float)M_PI / 2.0f);

    RayTracer rt(50, 50);
    rt.setCamera(cam);
    Camera c0 = rt.getCamera();
    CHECK_EQUAL(10, c0.getX());
    DOUBLES_EQUAL(-1/sqrt(3), c0.getViewDir().getZ(), 0.00001);
    DOUBLES_EQUAL(1/sqrt(2), c0.getOrthoUp().getX(), 0.00001);
    DOUBLES_EQUAL(1/sqrt(2), c0.getOrthoUp().getY(), 0.00001);
    DOUBLES_EQUAL((float)M_PI / 2.0f, cam.getVerticalFOV(), 0.000001);
}

TEST(RayTracer, has_normalized_vectors) {
    RayTracer rt(10, 10);
    rt.setViewDirection(Vect(3, 4, 5));
    Vect viewDir = rt.getViewDirection();
    DOUBLES_EQUAL(0.424264, viewDir.getX(), 0.0001);
    DOUBLES_EQUAL(0.565685, viewDir.getY(), 0.0001);
    DOUBLES_EQUAL(0.707106, viewDir.getZ(), 0.0001);
}

TEST(RayTracer, can_calculate_image_plane) {
    RayTracer rt(20, 10);

    Vect viewDir = rt.getViewDirection();
    CHECK_EQUAL(0, viewDir.getX());
    CHECK_EQUAL(0, viewDir.getY());
    CHECK_EQUAL(-1, viewDir.getZ());

    Vect camPos = rt.getCameraPos();
    CHECK_EQUAL(0, camPos.getX());
    CHECK_EQUAL(0, camPos.getY());
    CHECK_EQUAL(0, camPos.getZ());
    
    Vect pRight = rt.getParallelRight();
    CHECK_EQUAL(1, pRight.getX());
    CHECK_EQUAL(0, pRight.getY());
    CHECK_EQUAL(0, pRight.getZ());

    Vect pUp = rt.getParallelUp();
    CHECK_EQUAL(0, pUp.getX());
    CHECK_EQUAL(1, pUp.getY());
    CHECK_EQUAL(0, pUp.getZ());

    Vect iCtr = rt.getImageCenter();
    CHECK_EQUAL(0, iCtr.getX());
    CHECK_EQUAL(0, iCtr.getY());
    CHECK_EQUAL(-100, iCtr.getZ());
}

TEST(RayTracer, has_vertical_vector) {
    RayTracer r(20, 10);
    r.setViewDirection(Vect(0, 0, -1));
    r.setOrthogonalUp(Vect(0, 1, 0));

    Vect v = r.vertical();
    CHECK_EQUAL(0, v.getX());
    DOUBLES_EQUAL(100, v.getY(), 0.000001);
    CHECK_EQUAL(0, v.getZ());
}

TEST(RayTracer, has_horizontal_fov) {
    RayTracer rt(30, 20);
    DOUBLES_EQUAL(2.35619449, rt.getHorizontalFOV(), 0.000001);
}

TEST(RayTracer, has_horizontal_vector) {
    RayTracer r(30, 20);
    r.setViewDirection(Vect(0, 0, -1));
    r.setOrthogonalUp(Vect(0, 1, 0));

    Vect h = r.horizontal();
    DOUBLES_EQUAL(241.421, h.getX(), 0.001);
    CHECK_EQUAL(0, h.getY());
    CHECK_EQUAL(0, h.getZ());
}

TEST(RayTracer, x_and_y_both_works) {
    RayTracer r(2, 2, Vect(0, 0, -1), Vect(0, 1, 0));

    Vect v = r.vertical();
    Vect h = r.horizontal();

    CHECK_EQUAL(0, 0);
}

TEST(RayTracer, can_compute_point) {
    RayTracer r(30, 20, Vect(0, 0, -1), Vect(0, 1, 0));

    Point_2D pt = r.computePoint(10, 5);
    DOUBLES_EQUAL(0.35, pt.x, 0.00001);
    DOUBLES_EQUAL(0.275, pt.y, 0.00001);
}

TEST(RayTracer, can_compute_multiple_points_on_viewplane) {
    RayTracer r(2, 2, Vect(0, 0, -1), Vect(0, 1, 0));

    Point_2D p0 = r.computePoint(0, 0);
    DOUBLES_EQUAL(0.25, p0.x, 0.00001);
    DOUBLES_EQUAL(0.25, p0.y, 0.00001);
    Point_2D p1 = r.computePoint(1, 0);
    DOUBLES_EQUAL(0.75, p1.x, 0.00001);
    DOUBLES_EQUAL(0.25, p1.y, 0.00001);
    Point_2D p2 = r.computePoint(0, 1);
    DOUBLES_EQUAL(0.25, p2.x, 0.00001);
    DOUBLES_EQUAL(0.75, p2.y, 0.00001);
    Point_2D p3 = r.computePoint(1, 1);
    DOUBLES_EQUAL(0.75, p3.x, 0.00001);
    DOUBLES_EQUAL(0.75, p3.y, 0.00001);
}

TEST(RayTracer, can_compute_direction) {
    RayTracer r(2, 2, Vect(0, 0, -1), Vect(0, 1, 0));

    Vect p0 = r.computeDirection(0, 0);
    CHECK_EQUAL(-50, p0.getX());
    CHECK_EQUAL(-50, p0.getY());
    Vect p1 = r.computeDirection(1, 0);
    CHECK_EQUAL(50, p1.getX());
    CHECK_EQUAL(-50, p1.getY());
    Vect p2 = r.computeDirection(0, 1);
    CHECK_EQUAL(-50, p2.getX());
    CHECK_EQUAL(50, p2.getY());
    Vect p3 = r.computeDirection(1, 1);
    CHECK_EQUAL(50, p3.getX());
    CHECK_EQUAL(50, p3.getY());
}

TEST(RayTracer, ray_inits) {
    Ray r(Vect(0, 0, -1), Vect(-1, 0, 0));
    Vect o = r.getOrigin();
    Vect d = r.getDirection();
    CHECK_EQUAL(-1, o.getZ());
    CHECK_EQUAL(-1, d.getX());
}

TEST(RayTracer, can_compute_ray) {
    RayTracer r(2, 2, Vect(0, 0, -1), Vect(0, 1, 0));

    Ray r0 = r.computeRay(0, 0);
    Vect ctr = r0.getOrigin();
    CHECK_EQUAL(0, ctr.getX());
    CHECK_EQUAL(0, ctr.getY());
    CHECK_EQUAL(0, ctr.getZ());

    Vect dir = r0.getDirection();
    DOUBLES_EQUAL(-0.408248, dir.getX(), 0.00001);
    DOUBLES_EQUAL(-0.408248, dir.getY(), 0.00001);
    DOUBLES_EQUAL(-0.816497, dir.getZ(), 0.00001);
}
