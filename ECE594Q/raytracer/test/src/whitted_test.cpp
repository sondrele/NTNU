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

TEST(WhittedTest, outgoing_angle) {
    double inGoing = 45;
    double inGoingRads = inGoing * M_PI / 180.0;
    
    double outgoingRads = Whitted::GetOutgoingRads((float)inGoingRads, 1, 1.5f);

    double outgoing = 180 * outgoingRads / M_PI;
    DOUBLES_EQUAL(28.1255, outgoing, 0.0001);
}
