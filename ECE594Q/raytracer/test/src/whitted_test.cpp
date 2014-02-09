#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "whitted.h"

TEST_GROUP(WhittedTest) {
    void setup() {
    }
    void teardown() {
    }
};

TEST(WhittedTest, specular_color) {
    float q = 0.2f * 128;
    SColor ks = SColor(0, 0, 0);
    Vect N(0, 1, 1);
    N.normalize();
    Vect Dj(0, 0, 1);
    Vect V(0, 0, 1);
    SColor spec = Whitted::SpecularLightning(q, ks, N, Dj, V);

    DOUBLES_EQUAL(0, spec.R(), 0.0001);
    DOUBLES_EQUAL(0, spec.G(), 0.0001);
    DOUBLES_EQUAL(0, spec.B(), 0.0001);
}
