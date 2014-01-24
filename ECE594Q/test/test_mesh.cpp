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

    ASSERT_EQUAL_FLOAT(p.point.getCell(0, 0), 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(p.point.getCell(1, 0), 2.0, 0.001);
    ASSERT_EQUAL_FLOAT(p.point.getCell(2, 0), 3.0, 0.001);
    ASSERT_EQUAL_FLOAT(p.point.getCell(3, 0), 1.0, 0.001);

    MeshPoint p1;
    ASSERT_EQUAL_FLOAT(p1.getX(), 0, 0.001);
    ASSERT_EQUAL_FLOAT(p1.getY(), 0, 0.001);
    ASSERT_EQUAL_FLOAT(p1.getZ(), 0, 0.001);
    ASSERT_EQUAL_FLOAT(p1.getW(), 1, 0.001);
}


void MicroPolygon_has_boundingbox() {
    MeshPoint a(-1, -2, 3);
    MeshPoint b(2, 2, 3);
    MeshPoint c(3, 2, 3);
    MeshPoint d(-2, 2, 3);
    MicroPolygon mp;
    mp.a = &a;
    mp.b = &b;
    mp.c = &c;
    mp.d = &d;

    float *f = mp.getBoundingBox();
    ASSERT_EQUAL_FLOAT(f[0], -2, 0.00001);
    ASSERT_EQUAL_FLOAT(f[1], -2, 0.00001);
    ASSERT_EQUAL_FLOAT(f[2], 3, 0.00001);
    ASSERT_EQUAL_FLOAT(f[3], 2, 0.00001);
    delete [] f;
}

void point_intersects_with_micropolygon() {
    MeshPoint a(-2, -2, 3);
    MeshPoint b(2, -2, 3);
    MeshPoint c(-2, 2, 3);
    MeshPoint d(2, 2, 3);
    MicroPolygon mp;
    mp.a = &a;
    mp.b = &b;
    mp.c = &c;
    mp.d = &d;

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
    ASSERT_EQUAL_FLOAT(p.getZ(), 50, 0.0001);
}

void sphere_has_micropolygons() {
    RiSphere s(10, 2);
    std::vector<MicroPolygon> v = s.getMicroPolygons();
    ASSERT_EQUAL_INT(v.size(), 2);
    ASSERT_SAME(v[0].a, v[1].b);
    ASSERT_SAME(v[0].b, v[1].a);
    ASSERT_SAME(v[0].c, v[1].d);
    ASSERT_SAME(v[0].d, v[1].c);
    
    RiSphere s2(10, 4);
    std::vector<MicroPolygon> v2 = s2.getMicroPolygons();
    ASSERT_EQUAL_INT(v2.size(), 12);
    MicroPolygon p0 = v2[0];
    MicroPolygon p3 = v2[3];
    ASSERT_SAME(p0.a, p3.b);
    ASSERT_SAME(p0.c, p3.d);
    MicroPolygon p1 = v2[1];
    ASSERT_SAME(p0.b, p1.a);
    ASSERT_SAME(p0.d, p1.c);
    MicroPolygon p4 = v2[4];
    ASSERT_SAME(p0.c, p4.a);
    ASSERT_SAME(p0.d, p4.b);
    MicroPolygon p8 = v2[8];
    ASSERT_SAME(p4.c, p8.a);
    ASSERT_SAME(p4.d, p8.b);
}


void mesh_test_suite() {
    TEST_CASE(meshpoint_inits_correctly);
    TEST_CASE(MicroPolygon_has_boundingbox);
    TEST_CASE(point_intersects_with_micropolygon);
    TEST_CASE(mesh_inits_and_deletes);
    TEST_CASE(sphere_has_right_size);
    TEST_CASE(sphere_has_points_with_right_length);
    TEST_CASE(sphere_has_micropolygons);
}

void utils_test_suite() {
    TEST_CASE(point_intersects_triangle);
}

int main() {
    RUN_TEST_SUITE(utils_test_suite);
    RUN_TEST_SUITE(mesh_test_suite);
    return 0;
}
