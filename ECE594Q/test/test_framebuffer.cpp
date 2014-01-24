#include "ctest.h"
#include "FrameBuffer.h"
#include <iostream>

// void framebuffer_inits() {
//     FrameBuffer fb(500, 500, 2, 2);
//     float *samples = fb.getSamples();
//     ASSERT_EQUAL_FLOAT(samples[0], 0.333, 0.001);
//     ASSERT_EQUAL_FLOAT(samples[1], 0.333, 0.001);

//     ASSERT_EQUAL_FLOAT(samples[2], 0.333, 0.001);
//     ASSERT_EQUAL_FLOAT(samples[3], 0.666, 0.001);

//     ASSERT_EQUAL_FLOAT(samples[4], 0.666, 0.001);
//     ASSERT_EQUAL_FLOAT(samples[5], 0.333, 0.001);

//     ASSERT_EQUAL_FLOAT(samples[6], 0.666, 0.001);
//     ASSERT_EQUAL_FLOAT(samples[7], 0.666, 0.001);
// }

void can_add_points() {
    FrameBuffer fb(100, 100);
    Matrix x(4, 1);
    Matrix y(4, 1);
    x.setCell(0, 0, 1);
    fb.addPoint(x);
    fb.addPoint(y);
    
    Matrix p = fb.getPoint(0);
    ASSERT_EQUAL_INT(p.getRows(), x.getRows());
    ASSERT_EQUAL_INT(p.getCols(), x.getCols());
    ASSERT_EQUAL_FLOAT(p.getCell(0, 0), x.getCell(0, 0), 0.000001);

    Matrix q = fb.getPoint(1);
    ASSERT_EQUAL_INT(y.getCols(), q.getCols());
}

void can_project_and_scale_points() {
    FrameBuffer fb(500, 500);
    Vect v = Vect(0, 0, 100);
    fb.projectAndScalePoint(v);
    ASSERT_EQUAL_FLOAT(v.getX(), 250, 0.1);
    ASSERT_EQUAL_FLOAT(v.getY(), 250, 0.1);
}

void can_project_mesh_point() {
    MeshPoint mp(0, 0, 100);
    FrameBuffer fb(500, 500);
    fb.projectMeshPoint(mp);

    ASSERT_EQUAL_FLOAT(mp.point.getX(), 250, 0.1);
    ASSERT_EQUAL_FLOAT(mp.point.getY(), 250, 0.1);
}

void can_project_micropolygon() {
    MeshPoint mp1(-10, -10, 100);
    MeshPoint mp2(10, -10, 100);
    MeshPoint mp3(-10, 10, 100);
    MeshPoint mp4(10, 10, 100);
    MicroPolygon poly;
    poly.a = mp1;
    poly.b = mp2;
    poly.c = mp3;
    poly.d = mp4;

    FrameBuffer fb(500, 500);
    fb.projectMicroPolygon(poly);

    ASSERT_EQUAL_FLOAT(poly.a.getX(), 206.699, 0.1);
    ASSERT_EQUAL_FLOAT(poly.a.getY(), 206.699, 0.1);

    ASSERT_EQUAL_FLOAT(poly.b.getX(), 293.301, 0.1);
    ASSERT_EQUAL_FLOAT(poly.b.getY(), 206.699, 0.1);

    ASSERT_EQUAL_FLOAT(poly.c.getX(), 206.699, 0.1);
    ASSERT_EQUAL_FLOAT(poly.c.getY(), 293.301, 0.1);

    ASSERT_EQUAL_FLOAT(poly.d.getX(), 293.301, 0.1);
    ASSERT_EQUAL_FLOAT(poly.d.getY(), 293.301, 0.1);
}

void bounding_box_for_microPolygon_has_right_coords() {
    MeshPoint mp1(-10, -10, 100);
    MeshPoint mp2(10, -10, 100);
    MeshPoint mp3(-10, 10, 100);
    MeshPoint mp4(10, 10, 100);
    MicroPolygon poly;
    poly.a = mp1;
    poly.b = mp2;
    poly.c = mp3;
    poly.d = mp4;

    FrameBuffer fb(500, 500);
    fb.projectMicroPolygon(poly);

    float *f = poly.getBoundingBox();
    ASSERT_EQUAL_FLOAT(f[0], 206.699, 0.1);
    ASSERT_EQUAL_FLOAT(f[1], 206.699, 0.1);
    ASSERT_EQUAL_FLOAT(f[2], 293.301, 0.1);
    ASSERT_EQUAL_FLOAT(f[3], 293.301, 0.1);
    delete [] f;
}

void plot_image() {
    FrameBuffer fb(500, 500);

    RiSphere s(20, 128);
    fb.addMesh(s);

    fb.plotPoints("fb_test_plot.jpg");
    ASSERT_EQUAL_INT(4, 4);
}

void draw_microPolygons() {
    // FrameBuffer fb(500, 500);
    FrameBuffer fb(500, 500, 4, 4);
    RiSphere s(10, 64);
    fb.addMesh(s);

    fb.drawMicroPolygons("fb_test_poly.jpg");
    ASSERT_EQUAL_INT(4, 4);
}

void FrameBufferTestSuite() {
    // TEST_CASE(framebuffer_inits);
    TEST_CASE(can_add_points);
    TEST_CASE(can_project_and_scale_points);
    TEST_CASE(can_project_mesh_point);
    TEST_CASE(can_project_micropolygon);
    TEST_CASE(bounding_box_for_microPolygon_has_right_coords);
    TEST_CASE(plot_image);
    TEST_CASE(draw_microPolygons);
}

int main() {
    try {
        return RUN_TEST_SUITE(FrameBufferTestSuite);
    } catch(const char *x) {
        cout << x << endl;
    }
}