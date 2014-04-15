#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "raytracer.h"

TEST_GROUP(RayTracer) {
    void setup() {
    }
    void teardown() {
    }
};

TEST(RayTracer, should_init) {
    DirectIllumTracer rt(20, 10, 2);
    CHECK_EQUAL(20, rt.getWidth());
    CHECK_EQUAL(10, rt.getHeight());
}

TEST(RayTracer, has_camera) {
    DirectIllumTracer rt(10, 10, 2);
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

    DirectIllumTracer rt(50, 50, 2);
    rt.setCamera(cam);
    Camera c0 = rt.getCamera();
    CHECK_EQUAL(10, c0.getX());
    DOUBLES_EQUAL(-1/sqrt(3), c0.getViewDir().getZ(), 0.00001);
    DOUBLES_EQUAL(1/sqrt(2), c0.getOrthoUp().getX(), 0.00001);
    DOUBLES_EQUAL(1/sqrt(2), c0.getOrthoUp().getY(), 0.00001);
    DOUBLES_EQUAL((float)M_PI / 2.0f, cam.getVerticalFOV(), 0.000001);
}

TEST(RayTracer, has_normalized_vectors) {
    DirectIllumTracer rt(10, 10, 2);
    rt.setViewDirection(Vect(3, 4, 5));
    Vect viewDir = rt.getViewDirection();
    DOUBLES_EQUAL(0.424264, viewDir.getX(), 0.0001);
    DOUBLES_EQUAL(0.565685, viewDir.getY(), 0.0001);
    DOUBLES_EQUAL(0.707106, viewDir.getZ(), 0.0001);
}

TEST(RayTracer, can_calculate_image_plane) {
    DirectIllumTracer rt(20, 10, 2);

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
    CHECK_EQUAL(-10000, iCtr.getZ());
}

TEST(RayTracer, has_vertical_vector) {
    DirectIllumTracer r(20, 10, 2);
    r.setViewDirection(Vect(0, 0, -1));
    r.setOrthogonalUp(Vect(0, 1, 0));

    Vect v = r.vertical();
    CHECK_EQUAL(0, v.getX());
    DOUBLES_EQUAL(10000, v.getY(), 0.000001);
    CHECK_EQUAL(0, v.getZ());
}

TEST(RayTracer, has_horizontal_fov) {
    DirectIllumTracer rt(30, 20, 2);
    DOUBLES_EQUAL(2.35619449, rt.getHorizontalFOV(), 0.000001);
}

TEST(RayTracer, can_compute_point) {
    DirectIllumTracer r(30, 20, 2);
    r.setViewDirection(Vect(0, 0, -1));
    r.setOrthogonalUp(Vect(0, 1, 0));

    Point_2D pt = r.computePoint(10, 5);
    DOUBLES_EQUAL(0.35, pt.x, 0.00001);
    DOUBLES_EQUAL(0.275, pt.y, 0.00001);
}

TEST(RayTracer, can_compute_multiple_points_on_viewplane) {
    DirectIllumTracer r(2, 2, 2);
    r.setViewDirection(Vect(0, 0, -1));
    r.setOrthogonalUp(Vect(0, 1, 0));

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

TEST(RayTracer, ray_inits) {
    Ray r(Vect(0, 0, -1), Vect(-1, 0, 0));
    Vect o = r.getOrigin();
    Vect d = r.getDirection();
    CHECK_EQUAL(-1, o.getZ());
    CHECK_EQUAL(-1, d.getX());
}

TEST(RayTracer, can_compute_ray) {
    DirectIllumTracer r(2, 2, 2);
    r.setViewDirection(Vect(0, 0, -1));
    r.setOrthogonalUp(Vect(0, 1, 0));
    Ray r0 = r.computeRay(0, 0);
    Vect ctr = r0.getOrigin();
    CHECK_EQUAL(0, ctr.getX());
    CHECK_EQUAL(0, ctr.getY());
    CHECK_EQUAL(0, ctr.getZ());

    Vect dir = r0.getDirection();
    DOUBLES_EQUAL(-0.57735, dir.getX(), 0.00001);
    DOUBLES_EQUAL(-0.57735, dir.getY(), 0.00001);
    DOUBLES_EQUAL(-0.57735, dir.getZ(), 0.00001);
}
