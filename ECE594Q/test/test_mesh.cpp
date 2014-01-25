#include "Mesh.h"
#include "ctest.h"
#include <cstdlib>
#include <iostream>

void point_intersects_triangle() {
    Vect v1(-1, -1, -1);
    Vect v2(1, -1, -1);
    Vect v3(0, 1, -1);

    ASSERT_TRUE(Utils::PointInTriangle(Vect(-1, -1, 0), v1, v2, v3));
    ASSERT_TRUE(Utils::PointInTriangle(Vect(1, -1, 3), v1, v2, v3));
    ASSERT_TRUE(Utils::PointInTriangle(Vect(0, 1, 3), v1, v2, v3));
    ASSERT_TRUE(Utils::PointInTriangle(Vect(0, 0, 3), v1, v2, v3));
    ASSERT_FALSE(Utils::PointInTriangle(Vect(-2, 0, 3), v1, v2, v3));
}

void meshpoint_inits_correctly() {
    MeshPoint p(1.0, 2.0, 3.0);

    ASSERT_EQUAL_FLOAT(p.getCell(0, 0), 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(p.getCell(1, 0), 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(p.getCell(2, 0), 3.0, 0.001);
    ASSERT_EQUAL_FLOAT(p.getCell(3, 0), 1.0, 0.001);

    MeshPoint p1;
    ASSERT_EQUAL_FLOAT(p1.getX(), 0, 0.001);
    ASSERT_EQUAL_FLOAT(p1.getY(), 0, 0.001);
    ASSERT_EQUAL_FLOAT(p1.getZ(), 0, 0.001);
    ASSERT_EQUAL_FLOAT(p1.getW(), 1, 0.001);
}

void meshpoint_can_rotate() {
    MeshPoint p(10, 10, 0);
    Vect::Rotate(p, 'x', U_PI / 2.0);

    ASSERT_EQUAL_FLOAT(p.getX(), 10, 0.0001);
    ASSERT_EQUAL_FLOAT(p.getY(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(p.getZ(), 10, 0.0001);
}

void mesh_can_be_rotated() {
    // ASSERT_EQUAL_FLOAT(v3.getX(), -2.0, 0.001);
    // ASSERT_EQUAL_FLOAT(v3.getY(), 1.0, 0.001);
    // ASSERT_EQUAL_FLOAT(v3.getZ(), 3.0, 0.001);

    Mesh m(2, 2);
    m.addPoint(MeshPoint(0, 0, 0));
    m.addPoint(MeshPoint(1, 2, 3));
    m.rotate('z', 90.0);

    MeshPoint m0 = m.getPoint(0);
    ASSERT_EQUAL_FLOAT(m0.getX(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(m0.getY(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(m0.getZ(), 0, 0.0001);

    MeshPoint m1 = m.getPoint(1);
    ASSERT_EQUAL_FLOAT(m1.getX(), -2, 0.0001);
    ASSERT_EQUAL_FLOAT(m1.getY(), 1, 0.0001);
    ASSERT_EQUAL_FLOAT(m1.getZ(), 3, 0.0001);
}

void mesh_can_be_translate() {
    Mesh m(2, 2);
    m.addPoint(MeshPoint(0, 0, 0));
    m.addPoint(MeshPoint(1, 2, 3));

    m.translate(-2, 2, 3);

    MeshPoint m0 = m.getPoint(0);
    ASSERT_EQUAL_FLOAT(m0.getX(), -2, 0.001);
    ASSERT_EQUAL_FLOAT(m0.getY(), 2, 0.001);
    ASSERT_EQUAL_FLOAT(m0.getZ(), 3, 0.001);
    ASSERT_EQUAL_FLOAT(m0.getW(), 1, 0.001);

    MeshPoint m1 = m.getPoint(1);
    ASSERT_EQUAL_FLOAT(m1.getX(), -1, 0.001);
    ASSERT_EQUAL_FLOAT(m1.getY(), 4, 0.001);
    ASSERT_EQUAL_FLOAT(m1.getZ(), 6, 0.001);
    ASSERT_EQUAL_FLOAT(m1.getW(), 1, 0.001);
}

void MicroPolygon_has_boundingbox() {
    MeshPoint a(-1, -2, 3);
    MeshPoint b(2, 2, 3);
    MeshPoint c(3, 2, 3);
    MeshPoint d(-2, 2, 3);
    MicroPolygon mp;
    mp.a = a;
    mp.b = b;
    mp.c = c;
    mp.d = d;

    BoundingBox box = mp.getBoundingBox();
    ASSERT_EQUAL_INT(box.X_start, -2);
    ASSERT_EQUAL_INT(box.Y_start, -2);
    ASSERT_EQUAL_INT(box.X_stop, 3);
    ASSERT_EQUAL_INT(box.Y_stop, 2);
}

void point_intersects_with_micropolygon() {
    MeshPoint a(-2, -2, 3);
    MeshPoint b(2, -2, 3);
    MeshPoint c(-2, 2, 3);
    MeshPoint d(2, 2, 3);
    MicroPolygon mp;
    mp.a = a;
    mp.b = b;
    mp.c = c;
    mp.d = d;

    Vect point(0, 0, 0);
    bool intersects = mp.intersects(point);
    ASSERT_TRUE(intersects);
    ASSERT_TRUE(mp.intersects(Vect(-2, -2, 3)));
    ASSERT_TRUE(mp.intersects(Vect(2, -2, 3)));
    ASSERT_TRUE(mp.intersects(Vect(-2, 2, 3)));
    ASSERT_TRUE(mp.intersects(Vect(2, 2, 3)));
    ASSERT_TRUE(mp.intersects(Vect(-0.1, -0.1, 3)));
    ASSERT_TRUE(mp.intersects(Vect(0.5, 0.5, 3)));
    ASSERT_FALSE(mp.intersects(Vect(-2, -3, 7)));
    ASSERT_FALSE(mp.intersects(Vect(2, -3, 7)));
    ASSERT_FALSE(mp.intersects(Vect(3, -2, 7)));
    ASSERT_FALSE(mp.intersects(Vect(3, -2, 7)));
    ASSERT_FALSE(mp.intersects(Vect(3, 0, 7)));
    ASSERT_FALSE(mp.intersects(Vect(-3, 0, 7)));
    ASSERT_FALSE(mp.intersects(Vect(0, -3, 7)));
}

void micropolygon_has_depth() {
    MeshPoint a(-2, -2, 3);
    MeshPoint b(2, -2, 4);
    MeshPoint c(-2, 2, 5);
    MeshPoint d(2, 2, 6);
    MicroPolygon mp;
    mp.a = a;
    mp.b = b;
    mp.c = c;
    mp.d = d;

    float depth = mp.getDepth();
    ASSERT_EQUAL_FLOAT(depth, 4.5, 0.000001);
}

void mesh_inits_and_deletes() {
    Mesh m(2, 2);
    m.addPoint(MeshPoint(3, 3, 3));
    ASSERT_EQUAL_INT(m.getWidth(), 2);
    ASSERT_EQUAL_INT(m.getHeight(), 2);
}

void sphere_has_right_size() {
    RiSphere s(3.0, 2);
    ASSERT_EQUAL_INT(s.getWidth(), 2);
    ASSERT_EQUAL_FLOAT(s.getRadius(), 3.0, 0.0001);
    ASSERT_EQUAL_INT(s.getSize(), 2 * 2);

    RiSphere s1(3.0, 16);
    ASSERT_EQUAL_INT(s1.getWidth(), 16);
    ASSERT_EQUAL_FLOAT(s1.getRadius(), 3.0, 0.0001);
    ASSERT_EQUAL_INT(s1.getSize(), 16 * 16);
}

void sphere_has_points_with_right_length() {
    RiSphere s(100, 32);

    MeshPoint p = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p.getX(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(p.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p.getZ(), 0, 0.0001);
}

void sphere_has_micropolygons() {
    RiSphere s(10, 2);
    std::vector<MicroPolygon> v = s.getMicroPolygons();
    ASSERT_EQUAL_INT(v.size(), 2);
    MicroPolygon p = v[0];
    MicroPolygon q = v[1];
    ASSERT_TRUE(p.a.equal(q.b));
    ASSERT_TRUE(p.b.equal(q.a));
    ASSERT_TRUE(p.c.equal(q.d));
    ASSERT_TRUE(p.d.equal(q.c));
    
    RiSphere s2(10, 4);
    std::vector<MicroPolygon> v2 = s2.getMicroPolygons();
    ASSERT_EQUAL_INT(v2.size(), 12);
    MicroPolygon p0 = v2[0];
    MicroPolygon p3 = v2[3];
    ASSERT_TRUE(p0.a.equal(p3.b));
    ASSERT_TRUE(p0.c.equal(p3.d));
    MicroPolygon p1 = v2[1];
    ASSERT_TRUE(p0.b.equal(p1.a));
    ASSERT_TRUE(p0.d.equal(p1.c));
    MicroPolygon p4 = v2[4];
    ASSERT_TRUE(p0.c.equal(p4.a));
    ASSERT_TRUE(p0.d.equal(p4.b));
    MicroPolygon p8 = v2[8];
    ASSERT_TRUE(p4.c.equal(p8.a));
    ASSERT_TRUE(p4.d.equal(p8.b));
}

void sphere_can_rotate() {
    RiSphere s(100, 32);

    MeshPoint p0 = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p0.getX(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(p0.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p0.getZ(), 0, 0.0001);

    s.rotate('z', 90);
    MeshPoint p1 = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p1.getX(), -100, 0.0001);
    ASSERT_EQUAL_FLOAT(p1.getY(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(p1.getZ(), 0, 0.0001);
}

void sphere_can_translate() {
    RiSphere s(100, 32);

    MeshPoint p0 = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p0.getX(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(p0.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p0.getZ(), 0, 0.0001);

    s.translate(100, 0, 0);
    MeshPoint p1 = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p1.getX(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p1.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p1.getZ(), 0, 0.0001);
}

void micropolygon_and_sphere_is_correct() {
    RiSphere s(100, 32);

    MeshPoint p0 = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p0.getX(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(p0.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p0.getZ(), 0, 0.0001);
    std::vector<MicroPolygon> mps = s.getMicroPolygons();
    MeshPoint q0 = mps[0].a;
    ASSERT_EQUAL_FLOAT(q0.getX(), 0, 0.0001);
    ASSERT_EQUAL_FLOAT(q0.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(q0.getZ(), 0, 0.0001);

    s.translate(100, 0, 0);
    MeshPoint p1 = s.getPoint(0);
    ASSERT_EQUAL_FLOAT(p1.getX(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p1.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(p1.getZ(), 0, 0.0001);
    
    mps = s.getMicroPolygons();
    MeshPoint q1 = mps[0].a;
    ASSERT_EQUAL_FLOAT(q1.getX(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(q1.getY(), 100, 0.0001);
    ASSERT_EQUAL_FLOAT(q1.getZ(), 0, 0.0001);
}

void mesh_test_suite() {
    TEST_CASE(meshpoint_inits_correctly);
    TEST_CASE(MicroPolygon_has_boundingbox);
    TEST_CASE(point_intersects_with_micropolygon);
    TEST_CASE(micropolygon_has_depth);
    TEST_CASE(mesh_inits_and_deletes);
    TEST_CASE(meshpoint_can_rotate);

    TEST_CASE(mesh_can_be_rotated);
    TEST_CASE(mesh_can_be_translate);

    TEST_CASE(sphere_has_right_size);
    TEST_CASE(sphere_can_rotate);
    TEST_CASE(sphere_can_translate);
    TEST_CASE(sphere_has_points_with_right_length);
    TEST_CASE(micropolygon_and_sphere_is_correct);
    TEST_CASE(sphere_has_micropolygons);
}

void utils_test_suite() {
    TEST_CASE(point_intersects_triangle);
}

int main() {
    try {
        RUN_TEST_SUITE(utils_test_suite);
        RUN_TEST_SUITE(mesh_test_suite);
    } catch (const char *str) {
        cout << str << endl;
    }
    return 0;
}
