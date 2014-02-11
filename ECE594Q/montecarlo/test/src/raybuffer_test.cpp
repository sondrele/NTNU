#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "raybuffer.h"

TEST_GROUP(RayPixel) {
    void setup() {

    }
    void teardown() {

    }
};

TEST(RayPixel, pixel_has_color) {
    RayPixel p(10, 5);

    CHECK_EQUAL(10, p.getX());
    CHECK_EQUAL(5, p.getY());
    p.setColor(0, 255, 127);
    PX_Color c = p.getColor();
    CHECK_EQUAL(0, c.R);
    CHECK_EQUAL(255, c.G);
    CHECK_EQUAL(127, c.B);

    PX_Color c1 = {0, 0, 0};
    p.setColor(c1);
    c = p.getColor();
    CHECK_EQUAL(0, c.R);
    CHECK_EQUAL(0, c.G);
    CHECK_EQUAL(0, c.B);
}

TEST_GROUP(RayBuffer) {
    void setup() {

    }
    void teardown() {

    }
};

TEST(RayBuffer, can_init) {
    RayBuffer rb(50, 40);

    CHECK_EQUAL(50, rb.getWidth());
    CHECK_EQUAL(40, rb.getHeight());
    CHECK_EQUAL(2000, rb.size());
}

TEST(RayBuffer, standard_pixel_color_is_black) {
    RayBuffer rb(50, 40);
    RayPixel p0 = rb.getPixel(0, 0);
    PX_Color c = p0.getColor();

    CHECK_EQUAL(0, c.R);
    CHECK_EQUAL(0, c.G);
    CHECK_EQUAL(0, c.B);
}

TEST(RayBuffer, can_set_pixel) {
    RayBuffer rb(30, 20);
    PX_Color c0 = {10, 20, 30};
    rb.setPixel(0, 0, c0);
    RayPixel px = rb.getPixel(0, 0);
    CHECK_EQUAL(0, px.getX());
    CHECK_EQUAL(0, px.getY());
    PX_Color c = px.getColor();
    CHECK_EQUAL(10, c.R);
    CHECK_EQUAL(20, c.G);
    CHECK_EQUAL(30, c.B);
}
