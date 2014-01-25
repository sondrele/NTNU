#include "ctest.h"
#include "FrameBuffer.h"
#include <iostream>

void framebuffer_inits() {
    FrameBuffer fb(500, 500, 2, 2);

    ASSERT_EQUAL_INT(fb.getSize(), 500 * 500);
    FramePixel px = fb.getPixel(0, 0);
    ASSERT_EQUAL_INT(px.getX(), 0);
    ASSERT_EQUAL_INT(px.getM(), 2);
    ASSERT_EQUAL_INT(px.getY(), 0);
    ASSERT_EQUAL_INT(px.getN(), 2);

    FramePixel px1 = fb.getPixel(249, 100);
    ASSERT_EQUAL_INT(px1.getX(), 249);
    ASSERT_EQUAL_INT(px1.getM(), 2);
    ASSERT_EQUAL_INT(px1.getY(), 100);
    ASSERT_EQUAL_INT(px1.getN(), 2);
}

void can_project_and_scale_points() {
    FrameBuffer fb(500, 500);
    Vect v = Vect(0, 0, 100);
    fb.projectAndScalePoint(v);
    ASSERT_EQUAL_FLOAT(v.getX(), 250, 0.1);
    ASSERT_EQUAL_FLOAT(v.getY(), 250, 0.1);
}

void can_project_mesh_point() {
    FrameBuffer fb(500, 500);
    MeshPoint mp(0, 0, 100);
    fb.projectAndScalePoint(mp);

    ASSERT_EQUAL_FLOAT(mp.getX(), 250, 0.1);
    ASSERT_EQUAL_FLOAT(mp.getY(), 250, 0.1);

    MeshPoint mp2(-10, -10, 100);
    fb.projectAndScalePoint(mp2);

    ASSERT_EQUAL_FLOAT(mp2.getX(), 206.699, 0.1);
    ASSERT_EQUAL_FLOAT(mp2.getY(), 206.699, 0.1);
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

    BoundingBox box = poly.getBoundingBox();
    ASSERT_EQUAL_INT(box.X_start, 206);
    ASSERT_EQUAL_INT(box.Y_start, 206);
    ASSERT_EQUAL_INT(box.X_stop, 294);
    ASSERT_EQUAL_INT(box.Y_stop, 294);
}

void plot_translated_point() {
    FrameBuffer fb(500, 500);
    MeshPoint mp(0, 0, 50);
    fb.addPoint(mp);
    Vect::Translate(mp, -10, 0, 0);
    fb.addPoint(mp);

    fb.plotPoints("./imgs/fb_test_plot.jpg");

    MeshPoint p0 = fb.getPoint(0);
    MeshPoint p1 = fb.getPoint(1);
    ASSERT_EQUAL_FLOAT(p0.getX(), 0, 0.00001);
    ASSERT_EQUAL_FLOAT(p1.getX(), -10, 0.00001);
}

void plot_image() {
    FrameBuffer fb(500, 500);

    RiSphere s(20, 128);
    fb.addMesh(s);

    fb.plotPoints("fb_test_plot.jpg");
    ASSERT_EQUAL_INT(4, 4);
}

void pixels_has_sample() {
    FrameBuffer fb(1, 1, 1, 1);
    PX_Sample s = {1, 0.5, {255, 255, 255}};
    fb.setPixel(0, 0, 0, 0, s);
    FramePixel fp = fb.getPixel(0, 0);
    PX_Color pc = fp.getColor();
    ASSERT_EQUAL_INT(pc.R, 127);
    ASSERT_EQUAL_INT(pc.G, 127);
    ASSERT_EQUAL_INT(pc.B, 127);
}


void pixels_has_samples() {
    FrameBuffer fb(1, 1, 2, 2);
    fb.setPixel(0, 0, 0, 0, {1, 1, {255, 255, 255}});
    fb.setPixel(0, 0, 0, 1, {1, 1, {0, 0, 0}});
    fb.setPixel(0, 0, 1, 0, {1, 1, {255, 255, 255}});
    fb.setPixel(0, 0, 1, 1, {1, 1, {1, 2, 3}});
    FramePixel px = fb.getPixel(0, 0);
    PX_Color pc = px.getColor();
    ASSERT_EQUAL_INT(pc.R, 127);
    ASSERT_EQUAL_INT(pc.G, 128);
    ASSERT_EQUAL_INT(pc.B, 128);
    fb.exportImage("./imgs/_test.jpg");
}

void pixel_has_zbuffer() {
    FrameBuffer fb(1, 1, 2, 2);
    fb.setPixel(0, 0, 0, 0, {10, 1, {16, 32, 64}});
    FramePixel px = fb.getPixel(0, 0);
    PX_Color pc = px.getColor();
    ASSERT_EQUAL_INT(pc.R, 4);
    ASSERT_EQUAL_INT(pc.G, 8);
    ASSERT_EQUAL_INT(pc.B, 16);

    fb.setPixel(0, 0, 0, 0, {100, 1, {255, 255, 255}});
    px = fb.getPixel(0, 0);
    pc = px.getColor();
    ASSERT_EQUAL_INT(pc.R, 4);
    ASSERT_EQUAL_INT(pc.G, 8);
    ASSERT_EQUAL_INT(pc.B, 16);

    fb.setPixel(0, 0, 0, 0, {1, 1, {0, 0, 0}});
    px = fb.getPixel(0, 0);
    pc = px.getColor();
    ASSERT_EQUAL_INT(pc.R, 0);
    ASSERT_EQUAL_INT(pc.G, 0);
    ASSERT_EQUAL_INT(pc.B, 0);
    fb.exportImage("./imgs/_test.jpg");
}

void draw_simple_sphere() {
    FrameBuffer fb(500, 500, 2, 2);

    RiSphere s(10, 4);
    s.translate(0, 0, 50);
    fb.addMesh(s);

    fb.drawShapes("./imgs/fb_test_simple.jpg");
    ASSERT_EQUAL_INT(0, 0);
}

void draw_simple_spheres() {
    FrameBuffer fb(500, 500, 2, 2);

    RiSphere s(10, 16);
    s.translate(-10, -10, 50);
    fb.addMesh(s);

    RiSphere s1(10, 16);
    s1.rotate('x', 90);
    s1.translate(10, -10, 50);
    fb.addMesh(s1);

    RiSphere s2(10, 16);
    s2.rotate('x', 180);
    s2.translate(-10, 10, 50);
    fb.addMesh(s2);

    RiSphere s3(10, 16);
    s3.rotate('x', 270);
    s3.translate(10, 10, 50);
    fb.addMesh(s3);

    fb.drawShapes("./imgs/fb_test_simple.jpg");
    ASSERT_EQUAL_INT(0, 0);
}

void draw_shapes_behind() {
    // FrameBuffer fb(500, 500);
    FrameBuffer fb(500, 500, 2, 2);
    RiSphere s(10, 32);
    s.rotate('z', 180);
    s.rotate('x', 90);
    s.translate(0, 0, 70);
    fb.addMesh(s);
    
    RiSphere s2(5, 32);
    s2.translate(10, -10, 50);
    fb.addMesh(s2);

    RiSphere s3(5, 32);
    s3.rotate('x', -90);
    s3.translate(-10, -10, 50);
    fb.addMesh(s3);

    fb.drawShapes("./imgs/fb_test_sphere.jpg");
    ASSERT_EQUAL_INT(4, 4);
}

void FrameBufferTestSuite() {
    TEST_CASE(framebuffer_inits);
    TEST_CASE(can_project_and_scale_points);
    TEST_CASE(can_project_mesh_point);
    TEST_CASE(can_project_micropolygon);
    TEST_CASE(bounding_box_for_microPolygon_has_right_coords);
    TEST_CASE(pixels_has_sample);
    TEST_CASE(pixels_has_samples);
    TEST_CASE(pixel_has_zbuffer);
}

void drawings() {
    TEST_CASE(plot_translated_point);
    TEST_CASE(draw_simple_sphere);
    TEST_CASE(draw_simple_spheres);
    TEST_CASE(draw_shapes_behind);
}

int main() {
    try {
        RUN_TEST_SUITE(FrameBufferTestSuite);
        RUN_TEST_SUITE(drawings);
        return 0;
    } catch(const char *x) {
        cout << x << endl;
    }
}